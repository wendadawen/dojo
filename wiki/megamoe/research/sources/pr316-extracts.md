# PR #316 摘录：Mega MoE benchmarks

来源：https://github.com/deepseek-ai/DeepGEMM/pull/316
标题：Add various optimizations and Mega MoE benchmarks
作者：zheanxu，2026-04-24 合并进 main（分支 mega-update），改动量 1272 additions / 368 deletions。

## PR 描述（原文）

We benchmarked Mega MoE on DeepSeek-V4-Flash and DeepSeek-V4-Pro under 8-way expert parallelism (EP8), testing at various batch sizes (i.e., the number of tokens per rank) to cover different serving scenarios. All values are averaged across 8 ranks.

### DeepSeek-V4-Flash

DeepSeek-V4-Flash has 256 experts with top-k=6 (each token is routed to 6 experts), a hidden dimension of 4096, and an intermediate hidden dimension of 2048.

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | Interconnect (GB/s) | Speedup (vs legacy) |
|---|---|---|---|---|---|
| 1 | 56.5 | 5 | 1311 | 1 | 1.96x |
| 512 | 146.5 | 1056 | 3192 | 266 | 1.73x |
| 8192 | 1283.1 | 1928 | 998 | 499 | 1.56x |
| 32768 | 4855.5 | 2038 | 794 | 529 | 1.62x |

### DeepSeek-V4-Pro

DeepSeek-V4-Pro has 384 experts with top-k=6, a hidden dimension of 7168, and an intermediate hidden dimension of 3072.

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | Interconnect (GB/s) | Speedup (vs legacy) |
|---|---|---|---|---|---|
| 1 | 108.1 | 7 | 1758 | 1 | 1.61x |
| 512 | 369.6 | 1098 | 4619 | 182 | 1.54x |
| 8192 | 2818.5 | 2304 | 1094 | 393 | 1.50x |
| 32768 | 10655.2 | 2438 | 692 | 417 | 1.54x |

## 关键评论（原文，GitHub issue comments API）

zheanxu（2026-04-27，协作者，回答 batch size 定义）：

> @kiankyars Yes, the batch size listed is the number of tokens per rank as stated. So for the row showing 512 tokens per rank under EP8, the total across the node would be 512 × 8 = 4,096. Your understanding is correct.

yiakwy-xpu-ml-framework-team（2026-08-12，第三方评论者，注意非官方声明）：

> The original PR is for blackwell platform with NVFP4 support, hence FP8 x FP4 mlp. The codes are guarded by `#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)`, and extensively uses TMEM features.
>
> However efforts for Hopper is still on the way:
> - sm90: https://github.com/deepseek-ai/DeepGEMM/pull/323#issuecomment-4498738653
> - sm90 (sglang): https://github.com/sgl-project/DeepGEMM/pull/36
>
> For the moments, without TMEM, the performance is not significant, hence MegaMoE can be replaced with Fp4 EP V2 + FP8 DeepGeem + PDL in hopper platform.

## 基线说明（来自 tests/test_mega_moe.py，源码可核）

基线（"legacy"）是 DeepEP dispatch/combine + DeepGEMM grouped GEMM + TileLang SwiGLU 的非重叠流水线；正确性验证：无共享专家时 fused 与 baseline 逐位一致（torch.equal），有共享专家时差异 < 1e-8。
