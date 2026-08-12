"""实测七：端到端真实跑通 DeepSeek-V4 前向（缩小配置，CPU）。

做法：直接 import 官方 inference/model.py，只把它依赖的 CUDA/tilelang 算子
替换为 kernel_ref.py 的纯 PyTorch 等价实现，架构代码一行不改。
这样跑出来的数据流路径就是官方实现的路径。

替换清单（仅这些，且都是 kernel.py 里的 GPU 算子）：
  act_quant, fp4_act_quant, sparse_attn, hc_split_sinkhorn, fp8_gemm, fp4_gemm
  rotate_activation（依赖 fast_hadamard_transform CUDA 包，用等价 Hadamard 变换替代）
"""
import sys, types, math, json
import torch
import torch.nn.functional as F

sys.path.insert(0, "/tmp/dsv4/exp")
import kernel_ref

# 构造假的 kernel 模块，令 model.py 的 `from kernel import ...` 拿到纯 PyTorch 实现
fake = types.ModuleType("kernel")
fake.act_quant = kernel_ref.act_quant
fake.fp4_act_quant = kernel_ref.fp4_act_quant
fake.sparse_attn = kernel_ref.sparse_attn
fake.hc_split_sinkhorn = kernel_ref.hc_split_sinkhorn
fake.fp8_gemm = kernel_ref.fp8_gemm
fake.fp4_gemm = kernel_ref.fp4_gemm
sys.modules["kernel"] = fake

sys.path.insert(0, "/tmp/dsv4/inference")
import model as M

# rotate_activation 依赖 fast_hadamard_transform（CUDA-only），用等价快速 Hadamard 替代
def hadamard(x):
    n = x.size(-1)
    assert n & (n - 1) == 0, "Hadamard 需要 2 的幂维度"
    y = x.clone().float()
    h = 1
    while h < n:
        y = y.unflatten(-1, (-1, 2, h))
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y = torch.stack([a + b, a - b], dim=-2).flatten(-3)
        h *= 2
    return (y * n ** -0.5).to(x.dtype)

M.rotate_activation = hadamard

torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)

# 缩小配置：保留官方所有结构性特征（层交替、hash 层、overlap 压缩、indexer、mHC）
cfg = json.load(open("/tmp/dsv4/config.json"))
N_LAYERS = 6
args = M.ModelArgs(
    max_batch_size=1, max_seq_len=512,
    dtype="bf16", scale_fmt=None, expert_dtype=None, scale_dtype="fp32",
    vocab_size=512, dim=256, moe_inter_dim=128,
    n_layers=N_LAYERS, n_hash_layers=2, n_mtp_layers=1,
    n_heads=8, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
    score_func=cfg["scoring_func"], route_scale=cfg["routed_scaling_factor"],
    swiglu_limit=cfg["swiglu_limit"],
    q_lora_rank=128, head_dim=128, rope_head_dim=64,
    o_groups=4, o_lora_rank=64,
    window_size=cfg["sliding_window"],
    compress_ratios=(128, 128, 4, 128, 4, 128, 0),   # 与官方同构：HCA/CSA 交替
    compress_rope_theta=cfg["compress_rope_theta"],
    original_seq_len=cfg["rope_scaling"]["original_max_position_embeddings"],
    rope_theta=cfg["rope_theta"], rope_factor=cfg["rope_scaling"]["factor"],
    beta_fast=cfg["rope_scaling"]["beta_fast"], beta_slow=cfg["rope_scaling"]["beta_slow"],
    index_n_heads=8, index_head_dim=64, index_topk=16,
    hc_mult=cfg["hc_mult"], hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"], hc_eps=cfg["hc_eps"],
)

print("=== 配置（缩小版，结构与官方同构）===")
print(f"n_layers={args.n_layers}  compress_ratios={args.compress_ratios[:N_LAYERS]}")
print(f"hc_mult={args.hc_mult}  n_hash_layers={args.n_hash_layers}  window_size={args.window_size}")
print(f"score_func={args.score_func}  index_topk={args.index_topk}")
print()

model = M.Transformer(args)
# 随机初始化（checkpoint 里是 empty 张量，需填值才能跑）
for n, p in model.named_parameters():
    if p.dtype.is_floating_point:
        with torch.no_grad():
            p.normal_(0, 0.02) if p.dim() > 1 else p.fill_(0.02)
for n, b in model.named_buffers():
    pass
# hash 层的 tid2eid 需为合法专家索引
for layer in model.layers:
    if getattr(layer.ffn.gate, "hash", False):
        with torch.no_grad():
            layer.ffn.gate.tid2eid.copy_(
                torch.randint(0, args.n_routed_experts, layer.ffn.gate.tid2eid.shape, dtype=torch.int32)
            )
# RMSNorm 权重设为 1，避免 0.02 缩放导致数值过小
for n, m in model.named_modules():
    if isinstance(m, M.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)

print("=== 逐层类型（读实际构造出的模块）===")
for i, layer in enumerate(model.layers):
    a = layer.attn
    kind = "CSA(ratio=4,overlap,带Indexer)" if a.compress_ratio == 4 else (
           f"HCA(ratio={a.compress_ratio},无Indexer)" if a.compress_ratio else "纯滑窗")
    has_idx = a.indexer is not None if a.compress_ratio else False
    g = layer.ffn.gate
    route = "Hash查表" if g.hash else "分数topk(带bias)"
    print(f"  层{i}: {kind:<32} indexer={has_idx!s:<5} kv_cache={tuple(a.kv_cache.shape)}  路由={route}")
print()

print("=== prefill：真实前向 ===")
SEQ = 256
ids = torch.randint(0, args.vocab_size, (1, SEQ))
logits = model(ids, 0)
print(f"输入 input_ids {tuple(ids.shape)} -> logits {tuple(logits.shape)}")
print(f"logits dtype={logits.dtype}  是否含 NaN/Inf: {bool(torch.isnan(logits).any() or torch.isinf(logits).any())}")
print(f"logits 范围 [{logits.min().item():.4f}, {logits.max().item():.4f}]")
print()

print("=== decode：逐 token 增量前向 ===")
for pos in [SEQ, SEQ + 1, SEQ + 2, SEQ + 3]:
    nxt = torch.randint(0, args.vocab_size, (1, 1))
    lg = model(nxt, pos)
    print(f"  start_pos={pos}: 输入 {tuple(nxt.shape)} -> logits {tuple(lg.shape)}  NaN={bool(torch.isnan(lg).any())}")
print()

print("=== MTP 模块 ===")
h = torch.randn(1, SEQ, args.hc_mult, args.dim)
mtp_logits = model.mtp[0](h, 0, ids)
print(f"MTP 输入 hidden {tuple(h.shape)}（含 hc_mult={args.hc_mult} 份残差流）-> logits {tuple(mtp_logits.shape)}")
print()
print("端到端跑通：官方 model.py 架构代码未修改，仅替换 CUDA 算子为 PyTorch 等价实现。")
