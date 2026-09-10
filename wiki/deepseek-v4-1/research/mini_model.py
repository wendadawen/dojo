#!/usr/bin/env python3
"""官方 model.py 的注入加载器: 架构代码一行不改, 只替换 CUDA-only 依赖.

做法 (llm-arch-evidence-audit 技能 §4):
  1. mini_kernel (纯 PyTorch 语义复现) 注册为 sys.modules["kernel"]
  2. engram/image_processor/vision 官方文件原样 exec, 前置 future import (py3.9 注解)
  3. model.py 官方文件原样 exec
加载后 model 模块的全局符号 (Transformer/ModelArgs/shared_attn) 可直接使用.
"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
OFFICIAL = HERE / "official" / "inference"


def _exec_official(name: str, path: Path):
    src = path.read_text()
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile("from __future__ import annotations\n" + src, str(path), "exec"), mod.__dict__)
    return mod


def load_model_module():
    sys.path.insert(0, str(HERE))
    import mini_kernel

    sys.modules["kernel"] = mini_kernel
    _exec_official("engram", OFFICIAL / "engram.py")
    _exec_official("image_processor", OFFICIAL / "image_processor.py")
    _exec_official("vision", OFFICIAL / "vision.py")
    return _exec_official("model_v41", OFFICIAL / "model.py")


def mini_args(M):
    """等比缩小配置: 保留全部结构特征 (CSA2 三模式/候选池/mHC/Engram/DSpark/Vision).

    层型布局对齐真实模型的结构特征:
      layer 0-1: 纯 SWA (ratio 0); engram 在 layer 1 (与真实模型一致)
      layer 2:   ratio 2, Full  (kv+index source)
      layer 3:   ratio 2, Reuse
      layer 4:   ratio 1, Full  (decoder 第一层, 候选池 source)
      layer 5:   ratio 1, Reindex (用候选池)
      layer 6:   ratio 1, Reuse
      mtp 0-1:   DSpark (ratio 0, 纯 SWA)
    """
    return M.ModelArgs(
        max_batch_size=1,
        max_seq_len=64,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=129280,  # 真实词表: engram 压缩映射与 noise_token_id 依赖它
        dim=64,
        moe_inter_dim=48,
        n_layers=7,
        n_mtp_layers=2,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        norm_topk_prob=True,
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=32,
        head_dim=32,       # fp8 块 32 | fp4 块 16 整除
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=16,
        window_size=4,
        compress_ratios=(0, 0, 2, 2, 1, 1, 1, 0, 0),
        kv_source_layers=(2, 4),
        index_source_layers=(2, 4, 5),
        compress_rope_theta=160000.0,
        original_seq_len=32,  # 开 YaRN, 覆盖 rope_scaling 机制
        rope_theta=10000.0,
        rope_factor=4.0,
        beta_fast=8,
        beta_slow=2,
        index_n_heads=4,
        index_head_dim=64,   # fp4 块 32 整除
        index_topk=4,
        candidate_source_layer=4,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        engram_layer_ids=(1,),
        engram_num_embeddings=(25874,),  # 24 个素数(从 999 起找: 1009..1163)之和, 与官方布局算法一致
        engram_max_ngram_size=4,
        engram_vocab_size=1000,
        engram_n_heads=8,
        engram_head_dim=32,  # fp8 块 32 整除
        engram_pad_id=2,
        engram_compressed_vocab_size=99092,  # 真实 tokenizer 实测值
        vision_n_layers=2,
        vision_dim=32,
        vision_n_heads=4,
        vision_inter_dim=48,
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=32,
        image_token_id=129264,
        dspark_block_size=3,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=(5, 6),
        dspark_markov_rank=16,
        dspark_n_routed_experts=4,
        dspark_n_activated_experts=2,
    )


def init_params(model, seed=0):
    """官方推理代码从 ckpt 加载权重, 没有 _init_weights; 这里统一初始化.

    规则: >=2 维 normal(0, 0.02); 1 维按语义 (norm 类已自带 ones; attn_sink 置 0;
    hc_scale 置 1; hc_base 置 0; 量化 scale 置 1). 同一初始化服务全部 probe.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        with torch.no_grad():
            if name.endswith(".scale"):
                p.fill_(1.0)  # 量化 scale 必须为正, 先于维度判断
            elif p.dim() >= 2:
                w = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.02
                p.copy_(w.to(p.dtype))
            else:
                if name.endswith("attn_sink"):
                    p.zero_()
                elif name.endswith("hc_attn_scale") or name.endswith("hc_ffn_scale"):
                    p.fill_(1.0)
                elif "hc_attn_base" in name or "hc_ffn_base" in name:
                    p.zero_()
                elif name.endswith(".scale"):
                    p.fill_(1.0)
                elif "norm" in name or "q_weight" in name or "k_weight" in name:
                    pass  # RMSNorm/q_weight/k_weight 构造时已是 ones
                else:
                    p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * 0.02)
    return model
