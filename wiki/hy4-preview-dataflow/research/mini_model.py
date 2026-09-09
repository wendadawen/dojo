# 验证目标：把官方 modeling_hy_v4.py 原样 exec（架构代码零改动，只补框架侧桩），
# 用缩小配置跑通 prefill 与逐 token decode，forward hook 抓真实张量形状作为数据流图依据。
# 官方文件：src/modeling_hy_v4.py（transformers commit cbc1651a，快照于 research/src/）。
# 桩的语义依据：transformers 装饰器（use_kernel_forward_from_hub 等）官方语义即
# 「装了 kernel hub 就换实现，否则用文件内 PyTorch 参考实现」，恒等替换后走的正是参考分支。
import contextlib
import math
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
SRC = HERE / "src" / "modeling_hy_v4.py"

# ---------------- 桩：transformers 框架侧 ----------------


def _identity(*a, **k):
    if len(a) == 1 and callable(a[0]) and not k:
        return a[0]

    def deco(f):
        return f

    return deco


use_kernel_forward_from_hub = _identity
use_experts_implementation = _identity
deprecate_kwarg = _identity
dynamic_rope_update = _identity
merge_with_config_defaults = _identity
capture_outputs = _identity
auto_docstring = _identity
can_return_tuple = _identity


class init:
    @staticmethod
    def normal_(t, mean=0.0, std=1.0):
        return torch.nn.init.normal_(t, mean, std)

    @staticmethod
    def zeros_(t):
        return torch.nn.init.zeros_(t)

    @staticmethod
    def ones_(t):
        return torch.nn.init.ones_(t)

    @staticmethod
    def constant_(t, v):
        return torch.nn.init.constant_(t, v)

    @staticmethod
    def copy_(dst, src):
        with torch.no_grad():
            dst.copy_(src)
        return dst


ACT2FN = {"silu": F.silu}


@contextlib.contextmanager
def maybe_autocast(device_type=None, enabled=True):
    yield


class _AllAttention:
    @staticmethod
    def get_interface(name, default):
        return default


ALL_ATTENTION_FUNCTIONS = _AllAttention()
ROPE_INIT_FUNCTIONS = {}
GradientCheckpointingLayer = nn.Module
TransformersKwargs = object
Unpack = lambda x: x


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: torch.Tensor = None
    past_key_values: object = None
    hidden_states: object = None
    attentions: object = None


@dataclass
class CausalLMOutputWithPast:
    loss: object = None
    logits: torch.Tensor = None
    past_key_values: object = None
    hidden_states: object = None
    attentions: object = None


class GenerationMixin:
    pass


class DynamicCache:
    """桩：按 transformers cache_utils 的调用点实现最小语义。
    update 返回拼接后的完整 K/V（[B,H,S,D]，seq 在 dim=2）；
    update_indexer 返回拼接后的 indexer k（[B,S,D]，seq 在 dim=1）。"""

    def __init__(self, config=None):
        self.key_cache = {}
        self.value_cache = {}
        self.indexer_cache = {}

    def update(self, key_states, value_states, layer_idx):
        if layer_idx in self.key_cache:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        else:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_indexer(self, k, layer_idx):
        if layer_idx in self.indexer_cache:
            self.indexer_cache[layer_idx] = torch.cat([self.indexer_cache[layer_idx], k], dim=1)
        else:
            self.indexer_cache[layer_idx] = k
        return self.indexer_cache[layer_idx]

    def get_seq_length(self):
        if not self.key_cache:
            return 0
        return self.key_cache[next(iter(self.key_cache))].shape[2]


def create_causal_mask(config, inputs_embeds, attention_mask, past_key_values,
                       position_ids, allow_is_causal_skip=False, **kw):
    """桩：返回与 transformers eager 口径一致的加性因果掩码 [B,1,S,T]。
    query i（绝对位置 past+i）可见 key j <= past+i；不可见处为 dtype 的 finfo.min。"""
    B, S, _ = inputs_embeds.shape
    past = past_key_values.get_seq_length() if past_key_values is not None else 0
    T = S + past
    dtype = inputs_embeds.dtype
    keep = torch.ones(S, T, dtype=torch.bool).tril(diagonal=past)
    mask = torch.where(keep, torch.zeros((), dtype=dtype), torch.full((), torch.finfo(dtype).min, dtype=dtype))
    return mask[None, None]


class HYV4Config:
    """桩：字段与 __post_init__ 逻辑照抄官方 configuration_hy_v4.py。"""

    model_type = "hy_v4"
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(self, **kw):
        self.vocab_size = 120832
        self.hidden_size = 2816
        self.intermediate_size = 6912
        self.moe_intermediate_size = 768
        self.num_hidden_layers = 34
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.head_dim = 256
        self.hidden_act = "silu"
        self.max_position_embeddings = 262144
        self.initializer_range = 0.006
        self.rms_norm_eps = 1e-5
        self.use_cache = True
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.n_routed_experts = 256
        self.n_shared_experts = 1
        self.num_experts_per_tok = 8
        self.routed_scaling_factor = 2.827
        self.norm_topk_prob = True
        self.n_group = 1
        self.topk_group = 1
        self.q_lora_rank = 1536
        self.kv_lora_rank = 512
        self.qk_nope_head_dim = 192
        self.qk_rope_head_dim = 64
        self.v_head_dim = 256
        self.mlp_layer_types = None
        self.layer_types = None
        self.index_topk = 2048
        self.index_head_dim = 128
        self.index_n_heads = 16
        self.indexer_types = None
        self.hc_mult = 4
        self.hc_magnitude = 2.0
        self.hc_eps = 1e-6
        self.learnable_sink_init = 0.0
        self.swiglu_limit = 10.0
        self.rope_parameters = None
        self.pad_token_id = None
        self.bos_token_id = None
        self.eos_token_id = None
        self._attn_implementation = "eager"
        for k, v in kw.items():
            setattr(self, k, v)
        self.__post_init__()

    def __post_init__(self):
        # MLA expands the latent to one key/value per query head, so keys are never grouped.
        self.num_key_value_heads = self.num_attention_heads
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so `head_dim` points at it.
        self.head_dim = self.qk_rope_head_dim
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(1, self.num_hidden_layers) + [
                "sparse" * 1
            ] * max(self.num_hidden_layers - 1, 0)
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
        if self.indexer_types is None:
            self.indexer_types = [
                "full" if layer_idx == 0 or (layer_idx - 1) % 4 == 0 else "shared"
                for layer_idx in range(self.num_hidden_layers)
            ]

    def __getattr__(self, k):
        # attribute_map 语义照抄 transformers PreTrainedConfig
        am = {"num_local_experts": "n_routed_experts"}
        if k in am:
            return getattr(self, am[k])
        raise AttributeError(k)

    def get(self, k, default=None):
        return getattr(self, k, default)


class HYV4PreTrainedModel(nn.Module):
    """桩基类：__init__ 存 config，post_init 应用 _init_weights（逻辑照抄官方
    778-803 行 + transformers 基类对 Linear/Embedding/LayerNorm 的默认分支）。"""

    base_model_prefix = "model"
    _no_split_modules = ["HYV4DecoderLayer"]
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias", "fn", "scale", "base", "hc_fn", "hc_scale",
        "hc_base", "weights_proj", "k_norm", "sinks",
    ]

    def __init__(self, config):
        super().__init__()
        self.config = config

    def post_init(self):
        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, HYV4TopkRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, HYV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, HYV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.constant_(module.scale, 0.01)
            base_value = -math.log(max(module.hc_mult - 1, 1))
            base = torch.zeros_like(module.base)
            base[: module.hc_mult] = base_value
            init.copy_(module.base, base)
        elif isinstance(module, HYV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.constant_(module.hc_scale, 0.01)
            base_value = -math.log(max(module.hc_mult - 1, 1))
            init.constant_(module.hc_base, base_value)
        elif isinstance(module, HYV4Attention):
            init.constant_(module.sinks, self.config.learnable_sink_init)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, (HYV4RMSNorm, nn.LayerNorm)):
            if getattr(module, "weight", None) is not None:
                init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)


# ---------------- exec 官方源码（架构零改动） ----------------

G = dict(globals())
G.update(dict(
    HYV4Config=HYV4Config, HYV4PreTrainedModel=HYV4PreTrainedModel,
    Cache=object, DynamicCache=DynamicCache, GenerationMixin=GenerationMixin,
))

lines = SRC.read_text().split("\n")
# 段 1：46 行（HYV4RMSNorm 装饰器）到 743 行（DecoderLayer.forward 结束）
FIRST, LAST1 = 46, 743
body1 = "\n".join(lines[FIRST - 1 : LAST1])
assert "class HYV4RMSNorm" in body1 and "class HYV4DecoderLayer" in body1
assert "class HYV4PreTrainedModel" not in body1
exec(compile("from __future__ import annotations\n" + body1, "<official-modeling-1>", "exec"), G)

# 段 2：806 行（HYV4Model）到 977 行（HYV4ForCausalLM.forward 结束）
FIRST2, LAST2 = 806, 977
body2 = "\n".join(lines[FIRST2 - 1 : LAST2])
assert "class HYV4Model" in body2 and "class HYV4ForCausalLM" in body2
assert "class HYV4PreTrainedModel" not in body2
exec(compile("from __future__ import annotations\n" + body2, "<official-modeling-2>", "exec"), G)

for name in ["HYV4RMSNorm", "HYV4Indexer", "HYV4Attention", "HYV4MLP", "HYV4TopkRouter",
             "HYV4Experts", "HYV4MoE", "HYV4HyperConnection", "HYV4HyperHead",
             "HYV4DecoderLayer", "HYV4Model", "HYV4ForCausalLM", "eager_attention_forward",
             "apply_rotary_pos_emb", "yarn_apply_mscale", "HYV4RotaryEmbedding"]:
    globals()[name] = G[name]


# ---------------- 缩小配置（保留全部结构特征） ----------------
# 保留：层 0 dense + 其余 sparse；indexer_types 默认规则（10 层 → full 在 0,1,5,9）；
# hc_mult=4；top-8 of 16 experts；swiglu clamp；sink；gated MLA 全链路。
def small_config(**over):
    kw = dict(
        vocab_size=1000, hidden_size=512, num_hidden_layers=10,
        num_attention_heads=8, q_lora_rank=128, kv_lora_rank=96,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=48,
        intermediate_size=140, moe_intermediate_size=48,
        n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=8,
        index_n_heads=4, index_head_dim=64, index_topk=16,
        initializer_range=0.006, rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"},
        max_position_embeddings=4096,
        routed_scaling_factor=2.827, swiglu_limit=10.0,
    )
    kw.update(over)
    return HYV4Config(**kw)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = small_config()
    print(f"indexer_types={cfg.indexer_types}")
    print(f"mlp_layer_types={cfg.mlp_layer_types[:4]}... dense_layers={[i for i,t in enumerate(cfg.mlp_layer_types) if t=='dense']}")
    model = HYV4ForCausalLM(cfg)
    model.eval()

    # 参数计数
    n_params = sum(p.numel() for p in model.parameters())
    print(f"small model params: {n_params:,}")

    # forward hook 抓形状
    shapes = {}

    def mk_hook(name):
        def hook(m, inp, out):
            i0 = inp[0]
            if isinstance(out, tuple):
                shapes[name] = (tuple(i0.shape), tuple(out[0].shape))
            else:
                shapes[name] = (tuple(i0.shape), tuple(out.shape))
        return hook

    for name, mod in model.named_modules():
        if not name:
            continue
        last = name.split(".")[-1]
        if last in {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
                    "gate_proj", "attn_hc", "ffn_hc", "mlp", "gate", "experts",
                    "shared_experts", "indexer", "wq_b", "wk", "weights_proj", "hc_head", "norm"}:
            mod.register_forward_hook(mk_hook(name))

    # prefill
    ids = torch.randint(0, 1000, (1, 64))
    with torch.no_grad():
        out = model(ids, use_cache=True)
    print(f"\nprefill logits: {tuple(out.logits.shape)}, cache len={out.past_key_values.get_seq_length()}")
    print(f"hidden nan/inf: {torch.isnan(out.logits).any().item()}/{torch.isinf(out.logits).any().item()}")

    # ---- prefill/decode 一致性：整段前向 vs 增量前向，逐位对比 ----
    N = 48
    seq = torch.randint(0, 1000, (1, N))
    with torch.no_grad():
        full = model(seq, use_cache=False).logits          # 整段
        inc = model(seq[:, :-1], use_cache=True)            # prefill N-1
        last = model(seq[:, -1:], past_key_values=inc.past_key_values).logits  # decode 第 N 个
    diff = (full[:, -1, :] - last[:, -1, :]).abs()
    rel = diff / (full[:, -1, :].abs() + 1e-9)
    print(f"\nprefill/decode consistency: max_abs={diff.max().item():.3e} "
          f"mean_abs={diff.mean().item():.3e} max_rel={rel.max().item():.3e}")

    # 全部 N 个位置逐位一致性（贪心逐 token 推进）
    with torch.no_grad():
        o = model(seq[:, :1], use_cache=True)
        logits_seq = [o.logits]
        for t in range(1, N):
            o = model(seq[:, t : t + 1], past_key_values=o.past_key_values)
            logits_seq.append(o.logits)
    inc_all = torch.cat(logits_seq, dim=1)
    diff_all = (full - inc_all).abs()
    print(f"full-vs-incremental all positions: max_abs={diff_all.max().item():.3e}")

    # 逐 token decode
    logits_stream = [out.logits[:, -1]]
    cur = ids[:, -1:]
    for step in range(5):
        with torch.no_grad():
            out2 = model(cur, past_key_values=out.past_key_values, use_cache=True)
        logits_stream.append(out2.logits[:, -1])
        cur = out2.logits[:, -1].argmax(-1, keepdim=True)
        out = out2
    print(f"decode 5 steps ok, cache len={out.past_key_values.get_seq_length()}")

    # 关键形状
    for k in ["hc_head", "model.hc_head", "model.layers.0.attn_hc", "model.layers.0.self_attn.q_a_proj",
              "model.layers.0.self_attn.q_b_proj", "model.layers.0.self_attn.kv_a_proj_with_mqa",
              "model.layers.0.self_attn.kv_b_proj", "model.layers.0.self_attn.indexer",
              "model.layers.0.self_attn.indexer.wq_b", "model.layers.0.mlp",
              "model.layers.1.mlp", "model.layers.1.mlp.experts", "model.layers.1.mlp.shared_experts",
              "model.layers.2.self_attn", "model.norm"]:
        if k in shapes:
            print(f"  {k}: in={shapes[k][0]} out={shapes[k][1]}")

    # 主干单层 hidden 流形状（DecoderLayer 的输入输出是 [B,S,4,H]）
    h0 = model.model.layers[0]
    print(f"decoder layer0 module: {type(h0).__name__}")

    # 缓存口径：每层 K/V 形状
    pkv = out.past_key_values
    k0 = pkv.key_cache[0]
    print(f"\ncache K layer0: {tuple(k0.shape)}  (B,H,S,D)")
    ix0 = pkv.indexer_cache[0]
    print(f"indexer cache layer0: {tuple(ix0.shape)}  (B,S,D)")
    n_full = sum(1 for i in pkv.indexer_cache)
    print(f"indexer cached layers: {n_full}")
