# PR #304 摘录：Introducing Mega MoE

来源：https://github.com/deepseek-ai/DeepGEMM/pull/304
标题：[Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes
作者：LyricZhao（Chenggang Zhao），2026-04-17 合并进 main（merge commit 7f2a703ed51ac1f7af07f5e1453b2d3267d37d50），改动量 12135 additions / 3253 deletions。

## New features（原文）

- Mega MoE, fusing & overlapping dispatch/linear 1/SwiGLU/linear 2/combine into a single mega-kernel, overlapping NVLink communication and tensor core computation
  - Performance number: #316
  - Only FP8 x FP4 MoE is supported
  - Requires PyTorch >= 2.9
- FP4 Indexer (MQA logits) with larger MTP support
- FP8 x FP4 GEMM
- PDL
- Refactors on GEMM heuristics
- Faster JIT compilation
- GEMM optimizations (dynamic swap A/B, much faster MoE GEMM)
- DeepEPv2 MoE GEMM layout

## Bug fixes

- JIT may crash on distributed FS
- Some kernel hangs and IMA

## Contributors（原文）

- Mega MoE: @LyricZhao @zheanxu @bucket-xv @RayWang96 @interestingLSY @kurisu6912 @xay5421 @yukuai26
- FP4 Indexer: @zheanxu @xay5421 @interestingLSY @kurisu6912
- GEMM, PDL, JIT and bug fixes: @zheanxu @bucket-xv @xay5421 @yukuai26 @LyricZhao

## Additional notes（原文）

Mega MoE is still under development and optimizations, stay tuned and optimization ideas are welcome!
Disclaimer: this release is only related to DeepGEMM's development, has nothing to do with internal model release.
