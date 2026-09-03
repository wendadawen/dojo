# X-Stage 论文中关于 MegaMoE 的段落摘录

来源：arXiv:2607.23264v1，"X-Stage: An Overlooked Pipeline Stage for Communication-Computation Overlap in DiT Inference"，Jianwen Xian 等，2026-07-25 提交。
HTML 版：https://ar5iv.labs.arxiv.org/html/2607.23264
第三方学术论文（非 DeepSeek 官方），实验使用 DeepGEMM at commit 7f2a703（即 PR #304 合入 Mega MoE 的 commit）。

## §2.2 MegaMoE Execution（原文）

> MegaMoE is a representative fine-grained, communication–computation fused MoE kernel. In an expert-parallel MoE layer, a router selects one or more experts for each token. Because experts reside on different GPUs, Dispatch first sends each token to its destination rank. Each expert then evaluates a two-layer feed-forward network: Linear-1 projects the hidden dimension to an intermediate dimension, an activation such as SwiGLU is applied, and Linear-2 projects back to the hidden dimension. Combine returns expert outputs to the source rank and aggregates the top-k results using the router weights.

> A conventional implementation launches separate kernels for Dispatch, grouped GEMM, activation, the second grouped GEMM, and Combine. MegaMoE fuses these stages into a persistent mega-kernel and uses warp specialization for Dispatch, data movement, Tensor Core computation, and epilogue work. In the public implementation, the Linear-2 epilogue reads accumulator results and writes them to a symmetric Combine buffer with remote stores. Although separate roles execute the Tensor Core mainloop and the epilogue, finite on-chip accumulator and staging resources couple them as producer and consumer.

> MegaMoE groups local experts into expert waves to organize locality and execution. The original schedule generally executes a wave's Linear-1 work, then its activation and Linear-2 work, before advancing to the next wave. The Linear-1 work of different waves has no direct neural-network dependency. Once Dispatch data and a destination buffer are ready, a later wave's Linear-1 can, in principle, begin before the previous wave's remote Combine stores complete.

## §1 引言（原文，节选）

> We encountered this limitation while analyzing MegaMoE, DeepGEMM's persistent mixture-of-experts (MoE) kernel. MegaMoE fuses token Dispatch, Linear-1, activation, Linear-2, and Combine into a persistent kernel organized as expert waves. Its task-level timeline represents Combine as one communication phase. Under a conservative completion-coupled interpretation—in which the remote-store issuer is assumed unable to resume subsequent local work until the issued stores become remotely visible—the entire Combine interval lies on the local critical path. Using the published stage times, this interpretation predicts at most approximately 1.5x speedup over the serial stage sum, below the 1.56x reported by the implementation. This mismatch suggests that the issuer can resume useful execution after issuing the stores while the requests continue to progress toward remote-visible completion.

> In MegaMoE, an expert wave executes many Linear-2 epilogues consecutively, concentrating Combine stores into long bursts. We interleave ready Linear-1 work from later waves with Linear-2 work, redistributing computation between bursts without changing the dependencies or communication volume. Across 84 configurations, the resulting interleaved scheduler achieves a 1.18x geometric-mean speedup, a 1.17x median speedup, and a 1.62x maximum speedup over the Expert-Wave baseline.

## §2.4 图 2 说明（原文，节选）

> A Linear-2 epilogue first reads accumulators, performs data conversion and address calculation, and writes the results to the symmetric Combine buffer. The remote-store issue portion ends when the sender accepts those write instructions; it does not imply that the corresponding data is already visible remotely.

> Once the remote stores are accepted, the issuing role may continue with later computation while the requests progress toward the destination. This decoupling explains why a completion-coupled estimate can understate MegaMoE speedup.

## 使用注意

- "expert waves" 是该论文对 MegaMoE 调度结构的描述术语；DeepGEMM 官方源码（scheduler/mega_moe.cuh）中没有 "expert wave" 一词，官方调度单元是 TaskInfo（BlockPhase 分 Linear1/Linear2/SharedLinear1/SharedLinear2 四相），并有 L1 warmup waves + L1/L2 交替调度。引用 wave 术语时需注明是论文描述。
- 该论文的 interleaved 调度改进（1.18x/1.62x）是论文自己的修改版，不是 DeepGEMM main 的实现。
