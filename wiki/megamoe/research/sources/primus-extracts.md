# AMD ROCm Primus 的同名 MegaMoE（同名排除项摘录）

来源：AMD ROCm Primus 官方文档，"MegaMoE" 技术指南页
URL：https://rocm.docs.amd.com/projects/primus/en/main/04-technical-guides/mega-moe.html
访问日期：2026-09-03

## 原文摘录

> MegaMoE is a FlyDSL-based fused MoE layer that replaces Megatron's native MoELayer. It fuses the expert-parallel all-to-all communication into the grouped GEMMs via two fused kernels: dispatch grouped GEMM (dispatch_grouped_gemm): fuses token dispatch (all-to-all) into the L1 grouped GEMM. grouped GEMM combine (grouped_gemm_combine): fuses the L2 grouped GEMM into combine (all-to-all) + weighted reduce.

> Runtime target is EP-only (TP=1) + bf16.

## 用途

仅用于本页"来源与范围说明"中的同名项目排除声明：该 MegaMoE 是 AMD 训练栈（Megatron/Primus）里的融合 MoE 层，与本文主体（DeepSeek DeepGEMM 的 Mega MoE）是不同项目，本文不展开。
