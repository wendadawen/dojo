#!/usr/bin/env python3
"""把 dojo:tag 归并到受控词表。

原状态是自由文本：60 种取值里 42 种只用一次，同一概念多种写法
（「MoE 负载均衡」/「MoE 专家并行负载均衡」/「load-balancing」），
中英混杂，5 个页面把多个值塞进一个字段（最长 51 字）。

词表只收「讲什么技术」这一个维度——与 dojo:topics（粗分类）互补。
技术栈名（vLLM、llama.cpp、SGLang）是实现细节，文档形态（速查、论文、
概念解释）由 dojo:type 承担，都不进词表。

    python3 .dojo/scripts/normalize_tags.py            # 只报告映射
    python3 .dojo/scripts/normalize_tags.py --fix      # 写回页面
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

# slug -> 受控取值。按「这篇讲什么技术」判定，不按它属于哪个模型系列。
TAG_OF = {
    # 模型总览
    "deepseek-v4-1": "模型架构",
    "hy4-preview-lite": "模型架构",
    "kimi-k3": "模型架构",
    # 前向数据流（文档形态明确的一类，读者会按它检索）
    "deepseek-v4-1-dataflow": "数据流",
    "deepseek-v4-dataflow": "数据流",
    "glm-5-3-flash-dataflow": "数据流",
    "hy4-preview-dataflow": "数据流",
    "kimi-k3-dataflow": "数据流",
    "qwen3-5-dataflow": "数据流",
    "qwen3-8-flash-next-dataflow": "数据流",
    # 专家混合：架构、路由、负载均衡、专家并行、MoE 内核
    "aux-loss-free-routing": "MoE",
    "deepseek-moe": "MoE",
    "eplb": "MoE",
    "expertplex": "MoE",
    "fused-moe": "MoE",
    "latent-moe": "MoE",
    "megamoe": "MoE",
    "moe-serving": "MoE",
    "quantile-balancing": "MoE",
    "stable-latent-moe": "MoE",
    "ultraep": "MoE",
    # 注意力
    "block-attnres": "注意力",
    "causal-mask": "注意力",
    "delta-rule": "注意力",
    "dsa": "注意力",
    "gated-deltanet": "注意力",
    "kda": "注意力",
    "linear-attention": "注意力",
    "mla": "注意力",
    "mqa-gqa": "注意力",
    "sliding-window-attention": "注意力",
    "standard-attention": "注意力",
    # 位置编码
    "mrope": "位置编码",
    "nope": "位置编码",
    "positional-encoding": "位置编码",
    "rope": "位置编码",
    # KV cache
    "attention-sink": "KV cache",
    "cross-layer-kv-sharing": "KV cache",
    "kv-cache": "KV cache",
    "kv-cache-layout": "KV cache",
    "paged-attention": "KV cache",
    "prefix-caching": "KV cache",
    # 推理系统
    "beyond-buzz-disaggregation": "推理系统",
    "chunked-prefill": "推理系统",
    "dualpath": "推理系统",
    "hetero-pd": "推理系统",
    "hisparse": "推理系统",
    "increase-kv": "推理系统",
    "mooncake": "推理系统",
    "ppd-disaggregation": "推理系统",
    "strata": "推理系统",
    "vllm-cudagraph": "推理系统",
    "vllm-framework-map": "推理系统",
    "vllm-mm-image-two-stage": "推理系统",
    "vllm-mm-unified-embeds-cudagraph": "推理系统",
    "vllm-v1-two-process-arch": "推理系统",
    # 推理加速（草稿/投机解码一类）
    "dflash": "推理加速",
    "dflash2": "推理加速",
    "eagle-speculative": "推理加速",
    "speculative-decoding": "推理加速",
    # 量化
    "fp8-block-quant": "量化",
    "mixed-precision-quant": "量化",
    "mxfp4-qat": "量化",
    "quantization-basics": "量化",
    "sherry-ternary-quant": "量化",
    # 并行与通信
    "deepep": "并行与通信",
    "gpu-communication": "并行与通信",
    "model-parallelism": "并行与通信",
    "moonep": "并行与通信",
    "pcp-dcp": "并行与通信",
    "pp-load-balancing": "并行与通信",
    # 训练
    "block-diffusion": "训练",
    "flash-kda": "训练",
    "knowledge-distillation": "训练",
    "mopd": "训练",
    "opd": "训练",
    "pretraining": "训练",
    "sft": "训练",
    # 优化器
    "muon-optimizer": "优化器",
    "newton-schulz": "优化器",
    "per-head-muon": "优化器",
    # 网络结构（激活、归一化、残差、卷积）
    "depthwise-conv": "网络结构",
    "glu": "网络结构",
    "hyper-connections": "网络结构",
    "ihc": "网络结构",
    "residual-connection": "网络结构",
    "rmsnorm": "网络结构",
    "situ-glu": "网络结构",
    "swiglu": "网络结构",
    # 视觉与多模态
    "clip": "视觉与多模态",
    "moonvit-v2": "视觉与多模态",
    "siglip": "视觉与多模态",
    "vit": "视觉与多模态",
    # 数学与数值
    "cross-entropy": "数学与数值",
    "gpu-execution-model": "数学与数值",
    "low-rank-projection": "数学与数值",
    "ngram": "数学与数值",
    "svd": "数学与数值",
}

VOCABULARY = sorted(set(TAG_OF.values()))
# 只锚定 name 与 content 两个属性——部分页面的 meta 上还挂着别的属性
TAG_RE = re.compile(r'(name="dojo:tag" content=")([^"]*)(")')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    pages = {Path(p).parent.name: Path(p) for p in sorted(glob.glob("wiki/*/index.html"))}
    missing = sorted(set(pages) - set(TAG_OF))
    extra = sorted(set(TAG_OF) - set(pages))
    if missing:
        print("error: 词表漏了这些页面: " + ", ".join(missing))
        return 2
    if extra:
        print("error: 词表里有不存在的页面: " + ", ".join(extra))
        return 2

    print(f"受控词表 {len(VOCABULARY)} 个取值：")
    counts = {}
    for tag in TAG_OF.values():
        counts[tag] = counts.get(tag, 0) + 1
    for tag in VOCABULARY:
        print(f"  {tag:12} {counts[tag]:>3} 页")

    changed = 0
    for slug, path in pages.items():
        old = TAG_RE.search(path.read_text(encoding="utf-8")).group(2)
        new = TAG_OF[slug]
        if old != new:
            changed += 1
            if args.fix:
                text = path.read_text(encoding="utf-8")
                text = TAG_RE.sub(lambda m: m.group(1) + new + m.group(3), text, count=1)
                path.write_text(text, encoding="utf-8")
    print(f"\n{'已改写' if args.fix else '需改写'} {changed} / {len(pages)} 页")
    return 0


if __name__ == "__main__":
    sys.exit(main())
