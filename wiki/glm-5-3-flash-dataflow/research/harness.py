"""
GLM-5.3-Flash 官方架构代码运行框架。

做法：从官方 modeling_glm5_next.py 中原样抽取第 60–1330 行（RMSNorm ... DecoderLayer），
架构代码一行不改；只把本机 transformers 4.57 缺失的框架装饰器/工具替换为等价空实现：

  - use_experts_implementation / use_kernel_forward_from_hub /
    use_kernel_func_from_hub_with_fallback / use_kernelized_func /
    force_accelerate_hooks / auto_docstring  → 恒等装饰器（官方语义即"若装了 kernel hub 就换实现，
    否则用文件内的 PyTorch 参考实现"，本机走 fallback 分支，正是我们要验证的公式）
  - GradientCheckpointingLayer → nn.Module
  - ALL_ATTENTION_FUNCTIONS.get_interface → 返回文件内的 eager_attention_forward
  - Cache / init / ACT2FN → 最小等价实现

因此本文件跑的是官方前向逻辑本身，不是我复述的逻辑。
"""
from __future__ import annotations

import json, math, re, types, sys
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

SRC = "/tmp/glm53f/modeling_glm5_next.py"
CFG_PATH = "/tmp/glm53f/config.json"


# ---------------- 框架桩（不含任何架构语义） ----------------
def _identity_decorator(*a, **k):
    if len(a) == 1 and not k and (callable(a[0]) or isinstance(a[0], type)):
        return a[0]
    return lambda f: f


use_experts_implementation = _identity_decorator
use_kernel_forward_from_hub = _identity_decorator
use_kernel_func_from_hub_with_fallback = _identity_decorator
use_kernelized_func = _identity_decorator
force_accelerate_hooks = _identity_decorator
auto_docstring = _identity_decorator
GradientCheckpointingLayer = nn.Module
Unpack = dict
TransformersKwargs = dict
FlashAttentionKwargs = dict

ACT2FN = {"silu": F.silu, "sigmoid": torch.sigmoid, "gelu": F.gelu}


class _AttnRegistry:
    """官方通过 ALL_ATTENTION_FUNCTIONS 选后端；本机固定回落到文件内的 eager 实现。"""

    def get_interface(self, name, default):
        return default


ALL_ATTENTION_FUNCTIONS = _AttnRegistry()


class Cache:
    pass


class _Init:
    def __getattr__(self, name):
        return lambda *a, **k: None


init = _Init()


def torch_compilable_check(cond, msg):
    assert bool(cond), msg


# 官方文件里这些名字只出现在类型注解位置（`config: Glm5NextTextConfig`），
# 运行时不参与任何计算；用占位类满足名字解析。
class Glm5NextConfig:
    pass


class Glm5NextTextConfig:
    pass


class Glm5NextVisionConfig:
    pass


# ---------------- 最小 Cache：语义对齐官方 DynamicCache 的用法 ----------------
class LayerState:
    def __init__(self):
        self.conv_states = [None]
        self.recurrent_states = [None]
        self.keys = None
        self.values = None
        self.indexer = None

    def get_seq_length(self):
        return 0 if self.keys is None else self.keys.shape[-2]


class MiniCache(Cache):
    """按官方调用点实现：update / update_conv_state / update_recurrent_state /
    update_indexer / has_previous_state / layers[i].{keys,get_seq_length()}"""

    def __init__(self, num_layers):
        self.layers = [LayerState() for _ in range(num_layers)]

    def has_previous_state(self, layer_idx):
        return self.layers[layer_idx].recurrent_states[0] is not None

    def update(self, key_states, value_states, layer_idx):
        st = self.layers[layer_idx]
        if st.keys is None:
            st.keys, st.values = key_states, value_states
        else:
            st.keys = torch.cat([st.keys, key_states], dim=-2)
            st.values = torch.cat([st.values, value_states], dim=-2)
        return st.keys, st.values

    def update_conv_state(self, mixed_qkv, layer_idx, conv_kernel_size):
        st = self.layers[layer_idx]
        # 官方语义：prefill 时把 conv 状态拼在序列前，并把尾部 kernel-1 帧存下来
        if st.conv_states[0] is None:
            st.conv_states[0] = torch.zeros(
                mixed_qkv.shape[0], mixed_qkv.shape[1], conv_kernel_size - 1,
                dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        out = torch.cat([st.conv_states[0], mixed_qkv], dim=-1)
        st.conv_states[0] = out[:, :, -(conv_kernel_size - 1):].clone()
        return out

    def update_recurrent_state(self, state, layer_idx):
        self.layers[layer_idx].recurrent_states[0] = state

    def update_indexer(self, packed_states, layer_idx):
        st = self.layers[layer_idx]
        if st.indexer is None:
            st.indexer = packed_states
        else:
            st.indexer = torch.cat([st.indexer, packed_states], dim=1)
        return st.indexer

    def get_seq_length(self):
        for st in self.layers:
            if st.keys is not None:
                return st.keys.shape[-2]
        return 0


# ---------------- 原样抽取官方架构代码 ----------------
def load_official_code(first_line=63, last_line=1330):
    """抽取官方 modeling 文件的 [first_line, last_line] 行（1-based，含端点）。
    区间起止经人工核对：62 行是最后一条 import，1332 行起是 PreTrainedModel（含框架继承）。"""
    lines = open(SRC).read().split("\n")
    body = "\n".join(lines[first_line - 1: last_line])
    # 断言抽取区间的边界符号，防止上游文件行号漂移后静默抽错
    assert "class Glm5NextTextRMSNorm(nn.Module):" in body
    assert "class Glm5NextTextDecoderLayer(" in body
    assert "return hidden_states, topk_indices" in body
    assert "class Glm5NextPreTrainedModel" not in body
    return body


_OFFICIAL = load_official_code()
_g = globals()
# 本机 python 3.9 会在运行时求值 `X | None` 形式的注解（PEP 604 需 3.10+）。
# 官方文件本身带该语法，这里用 PEP 563 把全部注解延迟为字符串，
# 只影响注解求值时机，不改变任何执行语义。
exec(compile("from __future__ import annotations\n" + _OFFICIAL,
             "<official modeling_glm5_next.py L63-L1330>", "exec"), _g)

# 抽取后必须存在的官方符号
for _sym in ["Glm5NextTextRMSNorm", "Glm5NextTextMoE", "Glm5NextTextHyperConnection",
             "Glm5NextTextForgetGate", "Glm5NextTextLinearAttention", "Glm5NextTextIndexer",
             "Glm5NextTextAttention", "Glm5NextTextDecoderLayer", "Glm5NextTextHyperHead",
             "chunk_kimi_delta_attention", "recurrent_kimi_delta_attention", "l2norm"]:
    assert _sym in _g, f"官方符号缺失: {_sym}"


# ---------------- 配置：复刻官方 Glm5NextTextConfig.__post_init__ ----------------
class TextConfig:
    """字段值全部来自 zai-org/GLM-5.3-Flash 的 config.json['text_config']；
    派生逻辑（linear_attn_config 展开、head_dim=qk_rope_head_dim、qk_head_dim）
    复刻 configuration_glm5_next.py 的 __post_init__ 与 attribute_map。"""

    def __init__(self, raw: dict, overrides: dict | None = None):
        for k, v in raw.items():
            setattr(self, k, v)
        # attribute_map = {"num_local_experts": "n_routed_experts"}
        self.num_local_experts = raw["n_routed_experts"]
        d = raw["linear_attn_config"]
        self.linear_head_dim = d["head_dim"]
        self.linear_num_heads = d["num_heads"]
        self.linear_conv_kernel_dim = d["short_conv_kernel_size"]
        self.linear_lower_bound = d["gate_lower_bound"]
        self.head_dim = raw["qk_rope_head_dim"]
        self.qk_head_dim = raw["qk_rope_head_dim"] + raw["qk_nope_head_dim"]
        self._attn_implementation = "eager"
        self.training = False
        if overrides:
            for k, v in overrides.items():
                setattr(self, k, v)


def real_config(overrides: dict | None = None) -> TextConfig:
    raw = json.load(open(CFG_PATH))["text_config"]
    return TextConfig(raw, overrides)


def shrink_config(num_layers=8, **extra) -> TextConfig:
    """缩小配置：只缩层数与专家数，保留全部结构特征
    （KDA/DSA 4 层一循环、前 3 层 dense、indexer full/shared、hc_mult=4）。"""
    raw = json.load(open(CFG_PATH))["text_config"]
    ov = {
        "num_hidden_layers": num_layers,
        "layer_types": raw["layer_types"][:num_layers],
        "indexer_types": raw["indexer_types"][:num_layers],
        "mlp_layer_types": raw["mlp_layer_types"][:num_layers],
        "n_routed_experts": 16,
        "num_local_experts": 16,
        "num_experts_per_tok": 8,
    }
    ov.update(extra)
    return TextConfig(raw, ov)


def banner(title):
    print("=" * 78)
    print(title)
    print("=" * 78)
