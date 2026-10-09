# 资料冲突

- 输出缓存的形状｜repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L355-L381 的 docstring 说输出 C 的形状是 (M, topk, N)，M 是补齐后的总 token 数；L1742 里 intermediate_cache1 却是 `cache13[: M * top_k_num * N].view(M, top_k_num, N)`，M 是 `hidden_states.size(0)`，没有补齐位｜页面采用代码：输出是 (M, k, 2I)，补齐位只在内核里被掩码挡住，不占输出行
- 门控激活的写法｜repo/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py L866-L867 的 docstring 写输出维是 N//2，因为 `gate × activation(up)`；repo/vllm/vllm/model_executor/layers/activation.py L119 是 `silu(x[:d]) * x[d:]`，即 activation(门) 乘值｜页面采用 activation.py：SiLU(门) × 值
- 门控分数乘在哪里｜blog/moe_kernel_features.md L17 说 topk==1 的模型要把分数用在输入激活上；fused_moe.py L1798 与 L1837 在 `fused_experts_impl` 里两次调用传相反的开关，分数乘在第一次 GEMM 输出上；模块化路径 prepare_finalize/no_dp_ep.py L73-L79 在 prepare 里直接乘输入，experts/triton_moe.py L426 与 L547 的开关也不同｜页面只写 `fused_experts_impl` 这条路径，并注明函数名
