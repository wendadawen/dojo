# 资料

- terapipe-2102.07988v2.pdf / src-2102.07988v2/｜https://arxiv.org/abs/2102.07988v2｜v2（ICML 2021）｜2026-10-10｜TeraPipe：把一条序列沿 token 维度切成片段做流水线，提出序列内流水与最优切分（本页 token 维度切分的原始论文）
- mooncake-2407.00079.pdf / src-2407.00079/｜https://arxiv.org/abs/2407.00079｜v3｜2026-10-10｜Mooncake：KVCache 为中心的分离式推理架构，§5.1 在 prefill 池实现 CPP（chunked pipeline parallelism）
- gllm-2504.14775v2.pdf / src-2504.14775v2/｜https://arxiv.org/abs/2504.14775v2｜v2｜2026-10-10｜gLLM：把 GPU 空闲分成 inter-stage 与 inter-batch 两类依赖/失衡，用 token throttling 配平 micro-batch 之间的计算量
- blog/sglang-pp-blog.txt｜https://lmsys.org/blog/2026-01-15-chunked-pipeline/｜2026-01-15｜2026-10-10｜SGLang 官方博客《Pipeline Parallelism in SGLang》：CPP、异步 P2P、动态 chunk 与 smooth factor，附 H20 上的 PP4×TP8 基准
