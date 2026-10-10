# HiSparse

- 描述：HiSparse 是稀疏注意力解码时的分层 KV 存储：完整 KV 放在主机内存，GPU 上每个请求只留一块固定大小的热缓冲，每步按 top-k 把缺的条目换入。
- 摘要：稀疏注意力每步只读 top-k 个 KV，但完整 KV 仍占满显存；HiSparse 把全量 KV 放主机内存，GPU 只留热缓冲，用换入 kernel 按需取回，解码并发随之提高。
- 主题：推理系统, 内存与缓存
- 标签：KV cache

## 长页

- 开头｜先交代 KV cache、HBM、预填充与解码、top-k 稀疏注意力（DSA 即此类）；HiSparse 是什么；为什么需要；大概怎么工作｜550｜blog/lmsys-sglang-hisparse.md L8-L21；blog/sglang-hisparse-guide.md L1-L18
- ### 容量瓶颈：稀疏注意力为什么仍受显存限制｜每步只读 top-k 条，却要留着全部历史；单请求显存随长度增长，决定解码并发上限；小数字例子折叠｜350｜blog/lmsys-sglang-hisparse.md L8-L15；blog/sglang-hisparse-guide.md L11-L17
- ### 两层存储：主机放全量，GPU 放热缓冲｜先讲固定内存、页；完整 KV 在主机，GPU 上每个请求最多占一块固定大小的热缓冲，末尾另留一页给最新 token｜400｜blog/lmsys-sglang-hisparse.md L19-L23；blog/sglang-hisparse-guide.md L7；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L504-L550；repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L322-L330
- ### 解码一步：从 top-k 到注意力｜一步解码是逐层前向，每层选 top-k 并做注意力；五个动作及先后；换入由 GPU kernel（并行函数）完成；备份是异步拷贝，供之后取回｜400｜blog/sglang-hisparse-guide.md L19-L32；blog/sglang-hisparse-guide.md L121-L126；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L1104-L1114；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L724-L760
- ### 换入：命中、LRU 与未命中｜本章回答缺的条目怎么换入，只有标题，内容在两个小节｜0｜repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L322-L330
- #### 短序列快路径与命中检测｜序列不超过缓冲时全在缓冲里直接查位置；否则把 top-k 放进哈希表，扫缓冲看是否命中；top-k 含最新 token 时直接绑定到末尾页｜400｜repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L384-L445；repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L484-L545
- #### LRU 重排与未命中拷贝｜没被选中的位置按最久未用排前，未命中条目依次占用并从主机拷入，命中的排到最近使用端；缓冲大小影响未命中数；手算例子折叠｜400｜repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L597-L721；blog/lmsys-sglang-hisparse.md L24-L31
- #### 共享索引时的提前换入｜侧流是什么；多层复用同一层的 top-k 时，起点层记下未命中计划，后面的层在侧流上提前回放拷贝｜200｜blog/sglang-hisparse-guide.md L121-L126；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L1088-L1104
- ### 与 PD 分离结合：KV 直达主机｜先用一两句说预填充、解码分在不同实例，并讲清 RDMA；预填充实例把 KV 直接写进解码实例的主机池，绕过解码侧 GPU，避免显存尖峰｜350｜blog/sglang-hisparse-guide.md L33-L47；blog/sglang-hisparse-guide.md L17；blog/lmsys-sglang-hisparse.md L19-L23
- ### 参数怎么选｜三个参数各管什么；host_to_device_ratio 按主机可用内存配，文档示例约 1TB 配 5、约 2TB 配 10｜300｜blog/sglang-hisparse-guide.md L78-L119；blog/sglang-hisparse-guide.md L196-L206
- ### 效果与代价｜并发上升后吞吐近线性增长；数字标明模型与部署，条件和数字折叠｜300｜blog/lmsys-sglang-hisparse.md L8-L17；blog/lmsys-sglang-hisparse.md L33-L40
- ### 适用条件与边界｜目前支持 DSA 架构模型、DeepSeek V4、MiniMax M3；低并发有额外 IO 开销，作者预期靠重叠缓解；误解：不改变原生 top-k 选择｜400｜blog/sglang-hisparse-guide.md L1-L10；blog/lmsys-sglang-hisparse.md L8-L17；blog/lmsys-sglang-hisparse.md L74-L79

## 短页

- ### 把全量 KV 放到主机内存｜HiSparse 是什么：解码每步只挑 top-k 个 KV；全量放主机，GPU 留热缓冲（每请求一块固定大小的小缓存）；讲 KV cache｜250｜blog/lmsys-sglang-hisparse.md L8-L23；blog/sglang-hisparse-guide.md L1-L10
- ### 稀疏省了计算，没省容量｜为什么需要：每步只读 top-k 条 KV，却要把全部历史留在显存；同时解码的请求越多显存越不够，并发被卡住｜250｜blog/lmsys-sglang-hisparse.md L8-L15；blog/sglang-hisparse-guide.md L11-L17
- ### GPU 只留缓冲，缺的按需换入｜怎么做：每步把 top-k 里不在缓冲的条目从主机换入；换入后缓冲里最久没用的位置被占用；一件事，不写备份｜300｜blog/sglang-hisparse-guide.md L19-L32；blog/lmsys-sglang-hisparse.md L19-L31
