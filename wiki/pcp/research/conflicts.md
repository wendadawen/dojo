# 冲突记录

- 位置：「PCP：把预填充的序列计算摊到多张卡」与「因果负载与首尾配对」里预填充序列的切分块数。两处说法：官方文档 Prefill Context Parallel 写「we can split the request into `N` chunks, and each GPU computes one chunk」（每张卡一段，共 `N` 段）；官方 RFC「Partition Sequence Dimension」写「the sequence is divided into 2 × cp_world_size chunks. Each CP rank i is assigned both the i-th chunk and the (2 × cp_world_size - i - 1)-th chunk」（共 `2N` 块、每张卡拿首尾两块）。页面采用：主机制按文档写「切成 $N$ 段、每卡一段」，负载配平按 RFC 写「切成 $2N$ 块、每卡拿首尾两块」，两处各自照资料写，不在页面上讨论两处口径。
- 位置：PCP 是否只在预填充阶段工作。两处说法：官方文档把 PCP 写成预填充专属（「we need to implement context parallel separately for them」，PCP 段通篇只讲 prefill）；官方仓库 vllm/v1/attention/ops/pcp.py 写「PCP-only execution shards decode requests across ranks」（PCP 也参与解码请求分片）。页面采用：按文档写 PCP 是预填充侧机制、解码侧的显存压力归 DCP，不展开 PCP 参与解码分片这一实现细节。
