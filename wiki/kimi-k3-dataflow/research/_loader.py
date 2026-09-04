# 共享 loader：把官方 modeling_kimi_k3.py 加载为可实测模块。
# 外部依赖（transformers / flash-attn / 同包其他文件）以最小 stub 替换，
# 被测组件（视觉塔、merger、融合函数）全部保持官方源码原文。
import sys
import types
import torch
import torch.nn as nn


def _mk_module(name, attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def load_official(path):
    # --- 伪造 transformers 依赖 ---
    _mk_module("transformers", {"activations": _mk_module("transformers.activations", {})})

    class _GELUTanh(nn.Module):
        def forward(self, x):
            return nn.functional.gelu(x, approximate="tanh")

    _mk_module("transformers.activations", {"PytorchGELUTanh": _GELUTanh})

    class PretrainedConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    _mk_module("transformers.configuration_utils", {"PretrainedConfig": PretrainedConfig})

    class PreTrainedModel(nn.Module):
        def __init__(self, config=None, *a, **k):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

    _mk_module("transformers.modeling_utils", {"PreTrainedModel": PreTrainedModel})
    _mk_module("transformers.models.llava", {})
    _mk_module("transformers.models.llava.modeling_llava",
               {"LlavaCausalLMOutputWithPast": type("LlavaCausalLMOutputWithPast", (), {})})
    _mk_module("transformers.utils", {"is_flash_attn_2_available": lambda: False})

    with open(path, encoding="utf-8") as f:
        src = f.read()
    # stub 掉同包相对导入（configuration_kimi_k3 / modeling_kimi_linear）
    src = src.replace("from .configuration_kimi_k3 import KimiK3Config",
                      "class KimiK3Config:  # stub\n    pass")
    src = src.replace("from .modeling_kimi_linear import KimiLinearForCausalLM",
                      "class KimiLinearForCausalLM:  # stub\n    pass")
    ns = {"__name__": "official_modeling_kimi_k3"}
    exec(compile(src, path, "exec"), ns)
    return types.SimpleNamespace(**ns)
