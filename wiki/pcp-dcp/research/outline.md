# PCP 与 DCP

- 描述：PCP 与 DCP 是推理时上下文并行的两个阶段形态：预填充上下文并行（PCP）把一条长请求的预填充序列计算切到多张卡上摊开，解码上下文并行（DCP）把解码时不断增长的 KV cache 沿 token 位置切到多张卡上分片存储，两者都用来服务长上下文请求。
- 摘要：同一条长请求，预填充一次算完全部输入、压力在算力和等待首 token 的时间，解码每步只出一个 token 却要读全部历史 KV、压力在显存容量。PCP 把预填充的新 token 切成多段并行算，DCP 把 KV cache 沿 token 位置分片、每步用 LSE 合并各卡的部分注意力结果。
- 主题：推理系统, 并行与通信, 内存与缓存
- 标签：并行与通信

## 长页

- 开头｜上下文并行是什么：把一条长序列沿 token 维切到多张卡；同一条请求的预填充与解码分成两种形态｜440｜blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-dcp-blog.md 1. Introduction
- ### 预填充算得多，解码存得多｜两阶段压力不同：预填充一次算完全部输入、控首 token 时间；解码每步只出一个 token 却读全部历史 KV，KV 装多少决定并发｜460｜blog/vllm-context-parallel-deployment.md Context Parallel Deployment
- ### DCP：把解码的 KV cache 沿序列维分掉｜只有标题，内容在两个小节｜0｜blog/vllm-dcp-blog.md 4. What is DCP?
- #### 沿头维切到头就开始重复｜张量并行先沿注意力头切 KV，头数由架构定死，tp 超过后每卡重复；GQA 与 MLA 的 KV 头都少，都撞这天花板｜440｜blog/vllm-dcp-blog.md 1. Introduction；blog/vllm-dcp-blog.md 3. Challenges of Serving Long Contexts；blog/vllm-context-parallel-deployment.md Decode Context Parallel
- #### 每张卡只存一段 token 的 KV｜DCP 改沿 token 位置维切、每卡持唯一一份切片；200K 请求四卡各 50K 示意、交错落位、注意力后这批卡合起来摊 FFN 权重｜540｜blog/vllm-dcp-blog.md 4. What is DCP?；blog/vllm-dcp-blog.md 7. Conclusion；blog/vllm-context-parallel-deployment.md Decode Context Parallel；repo/vllm/vllm/config/parallel.py L380-L392
- #### 一步解码的通信与合并｜每步聚齐查询、在本地切片上算部分注意力、再合并各卡结果；折叠合并公式与一组小数字手算例｜560｜blog/vllm-dcp-blog.md 4.1 Decode Context Parallelism Process；blog/vllm-dcp-blog.md 5.3 MLA Backend；blog/vllm-dcp-blog.md 5.4 GQA Backend；repo/vllm/vllm/v1/worker/cp_utils.py L21-L57
- #### dcp 的取值与约束｜dcp 越大、头维切剩的重复越少但通信越多，范围 [1, t/H]；GQA 上界是重复因子且要整除，MLA 可切满 tp；无 PCP 时要求 tp 整除 dcp｜520｜blog/vllm-context-parallel-deployment.md Decode Context Parallel；blog/vllm-dcp-blog.md 5.3 MLA Backend；blog/vllm-dcp-blog.md 5.4 GQA Backend；repo/vllm/vllm/config/parallel.py L587-L600；repo/vllm/vllm/config/model.py L1558-L1587
- ### PCP：把预填充的序列计算摊到多张卡｜预填充 T 个新 token 切成 N 段各算 Q/K/V；装得下全量 KV 则各算各段，装不下用 ring attention 逐块收发；折叠因果负载与配对｜560｜blog/vllm-context-parallel-deployment.md Prefill Context Parallel；blog/vllm-rfc-26133-context-parallelism.md Prefill；blog/vllm-rfc-26133-context-parallelism.md Partition Sequence Dimension
- ### 两个机制为什么要分开｜PCP 扩展启动的卡数，DCP 不扩卡数、落在已有的卡上；一条请求预填充用 PCP、解码用 DCP，各管一段｜320｜repo/vllm/vllm/config/parallel.py L143-L146；repo/vllm/vllm/config/parallel.py L380-L384；repo/vllm/vllm/config/parallel.py L940-L944；blog/vllm-dcp-blog.md 6. Future Work
- ### 适用条件与边界｜注意力后端要能返回 LSE 才能用 DCP；PCP 两种策略都在开发；与 chunked prefill 的一两句区别；两条常见误解｜420｜repo/vllm/vllm/v1/worker/cp_utils.py L21-L57；blog/vllm-context-parallel-deployment.md Prefill Context Parallel；repo/vllm/docs/configuration/optimization.md Chunked Prefill；blog/vllm-dcp-blog.md 6. Future Work；repo/vllm/vllm/config/parallel.py L380-L384

## 短页

- ### 把一条长请求切到多张卡上｜是什么：上下文并行把一条长序列沿 token 维切到多张卡；PCP 切预填充要算的 token，DCP 切解码存的 KV cache｜320｜blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-dcp-blog.md 1. Introduction
- ### 预填充费算力，解码费显存｜为什么需要：预填充一次算完全部输入、慢在首 token 时间，要把它摊薄；解码每步读全部历史 KV，KV 决定并发｜320｜blog/vllm-context-parallel-deployment.md Context Parallel Deployment
- ### 预填充切计算，解码切缓存｜怎么做：PCP 把 T 个新 token 切成 N 段各算，装得下聚齐 KV、否则用 ring attention；DCP 每卡存一段 KV，用 LSE 合并｜330｜blog/vllm-context-parallel-deployment.md Prefill Context Parallel；blog/vllm-context-parallel-deployment.md Decode Context Parallel；blog/vllm-dcp-blog.md 4.1 Decode Context Parallelism Process
