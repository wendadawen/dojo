# PCP

- 描述：PCP（预填充上下文并行，prefill context parallelism）把一条长请求的预填充序列计算切成几段、摊到多张卡上并行算，用来缩短预填充时间、降低用户等待首 token 的时间。
- 摘要：预填充要一次算完全部输入 token、慢在首 token 时间；PCP 把这 $T$ 个新 token 切成 $N$ 段，每张卡算一段的查询、键、值，装得下全量键值就各算各段，装不下用 ring attention 逐块收发。
- 主题：推理系统, 并行与通信
- 标签：并行与通信

## 长页

- 开头｜上下文并行是什么、预填充与解码两阶段、PCP 是什么（切预填充要算的 token）、为什么需要、大概怎么工作｜360｜blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-rfc-26133-context-parallelism.md Motivation
- ### 预填充算得多，解码存得多｜预填充一次算完全部输入、慢在首 token 时间，这是 PCP 的压力；解码每步只出一个 token 却读全部历史 KV、吃显存，一句对照｜420｜blog/vllm-context-parallel-deployment.md Context Parallel Deployment
- ### PCP：把预填充的序列计算摊到多张卡｜预填充 $T$ 个 token 切成 $N$ 段各算一段查询键值；装得下全量键值就聚齐各卡各算，装不下用 ring attention 收发并择交换侧｜560｜blog/vllm-context-parallel-deployment.md Prefill Context Parallel；blog/vllm-rfc-26133-context-parallelism.md Prefill
- #### 因果负载与首尾配对｜因果掩码下位置越后计算越重，切成 $2N$ 块、第 $i$ 卡拿第 $i$ 块与第 $2N-1-i$ 块配平；小数字手算折叠｜320｜blog/vllm-rfc-26133-context-parallelism.md Partition Sequence Dimension
- ### 适用条件与边界｜PCP 是独立的并行方式，增加要启动的卡数、但不增加 KV cache 的分片数；两种策略都在开发中；与相近的 DCP 一两句区别并链接｜360｜blog/vllm-context-parallel-deployment.md Prefill Context Parallel；repo/vllm/vllm/config/parallel.py L143-L146；blog/vllm-context-parallel-deployment.md Context Parallel Deployment

## 短页

- ### 每张卡只算一段 token｜是什么：先一句话说明预填充是模型一次读入整段输入、算出结果的阶段；PCP 把这段序列计算沿 token 维切分、摊到多张卡上并行做｜320｜blog/vllm-context-parallel-deployment.md Prefill Context Parallel
- ### 首 token 的等待太长｜为什么需要：预填充要一次算完全部输入 token、慢在首 token 时间，把这段计算摊到多张卡上才能缩短它｜320｜blog/vllm-context-parallel-deployment.md Context Parallel Deployment
- ### 切成 N 段各算一段｜怎么做：把 $T$ 个 token 切成 $N$ 段各算一段的查询、键、值；装得下全量键值就聚齐各卡、各算本段，装不下用 ring attention 收发｜340｜blog/vllm-context-parallel-deployment.md Prefill Context Parallel
