# DCP

- 描述：DCP（解码上下文并行，decode context parallelism）把解码时不断增长的 KV cache 沿 token 位置维切到多张卡上分片存放，每张卡只保留一段，用更小的缓存占用换取更大的批和更高的吞吐。
- 摘要：解码每步只出一个 token 却要读全部历史 KV，缓存能装多少决定并发；DCP 把 KV cache 沿 token 位置分给各卡、每卡只存一段，一步解码先聚齐查询、各卡算部分注意力，再按 LSE 合并。
- 主题：推理系统, 并行与通信, 内存与缓存
- 标签：并行与通信

## 长页

- 开头｜上下文并行是什么、预填充与解码两阶段、DCP 是什么（切解码反复读的 KV cache）、为什么需要、大概怎么工作｜360｜blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-dcp-blog.md 1. Introduction
- ### 预填充算得多，解码存得多｜解码每步只出一个 token 却要读全部历史 KV、缓存装多少决定并发，这是 DCP 的压力；预填充一次算完全部输入、慢在首 token 时间，一句对照｜420｜blog/vllm-context-parallel-deployment.md Context Parallel Deployment
- ### DCP：把解码的 KV cache 沿 token 位置维分掉｜只有标题，内容在四个小节｜0｜blog/vllm-dcp-blog.md 4. What is DCP?
- #### 沿头维切到头就开始重复｜张量并行先沿注意力头切 KV，头数由模型结构定死，tp 超过后每卡只剩重复副本；GQA 与 MLA 的 KV 头都少，都先撞这条天花板｜440｜blog/vllm-dcp-blog.md 1. Introduction；blog/vllm-dcp-blog.md 3. Challenges of Serving Long Contexts；blog/vllm-context-parallel-deployment.md Decode Context Parallel
- #### 每张卡只存一段 token 的 KV｜DCP 改沿 token 位置维切、每卡持唯一一份切片；交错落位让新 token 自然分到各卡；这批卡再重组起来摊前馈权重；200K 请求四卡各 50K 示意｜540｜blog/vllm-dcp-blog.md 4. What is DCP?；blog/vllm-dcp-blog.md 7. Conclusion；blog/vllm-context-parallel-deployment.md Decode Context Parallel；repo/vllm/vllm/config/parallel.py L414-L424
- #### 一步解码的通信与合并｜一步解码聚齐查询、各卡在本地切片上算部分注意力、再合并各卡结果；折叠合并公式与一组小数字手算例｜560｜blog/vllm-dcp-blog.md 4.1 Decode Context Parallelism Process；blog/vllm-dcp-blog.md 5.3 MLA Backend；blog/vllm-dcp-blog.md 5.4 GQA Backend
- #### dcp 的取值与约束｜dcp 越大重复越少、通信越多，无 PCP 时范围 [1, t/H]、要求 tp 整除 dcp；t 为并行度、H 为 KV 头数｜520｜blog/vllm-context-parallel-deployment.md Decode Context Parallel；blog/vllm-dcp-blog.md 5.3 MLA Backend；blog/vllm-dcp-blog.md 5.4 GQA Backend；repo/vllm/vllm/config/parallel.py L587-L600；repo/vllm/vllm/config/model.py L1558-L1587
- ### 适用条件与边界｜注意力后端要能返回 LSE 才能用 DCP；与相近的 PCP 一两句区别并链接；两条常见误解｜420｜repo/vllm/vllm/v1/worker/cp_utils.py L21-L57；blog/vllm-context-parallel-deployment.md Prefill Context Parallel；blog/vllm-context-parallel-deployment.md Decode Context Parallel

## 短页

- ### 每张卡只留一段 KV cache｜是什么：先一句话说明解码是逐 token 生成、每步读全部历史 KV 的阶段；DCP 把这 KV cache 沿 token 位置维切到多张卡，每卡存一段｜320｜blog/vllm-context-parallel-deployment.md Decode Context Parallel；blog/vllm-dcp-blog.md 4. What is DCP?
- ### 头维切到头就只剩重复｜为什么需要：张量并行先沿注意力头切 KV，头数由结构定死；GQA 与 MLA 的 KV 头都少，tp 超过头数后每卡只剩重复副本，重复副本占走显存、压低了并发｜320｜blog/vllm-dcp-blog.md 1. Introduction；blog/vllm-dcp-blog.md 3. Challenges of Serving Long Contexts
- ### 聚查询、算部分注意力、按 LSE 合并｜怎么做：每卡存一段 KV，一步解码先聚齐查询，各卡在本地切片上算部分注意力和它的 LSE，再用 LSE 加权合并成与单卡一致的结果｜340｜blog/vllm-dcp-blog.md 4.1 Decode Context Parallelism Process；blog/vllm-context-parallel-deployment.md Decode Context Parallel
