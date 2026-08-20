# paged-attention 术语表

| 术语/符号 | 首次出现 | 定义或含义 |
|---|---|---|
| PagedAttention（分页注意力） | 页面开头 | KV cache 按固定 token 数分页、页可非连续存储的管理机制（vLLM 提出） |
| 页 / 块（page / block） | 页面开头 | KV cache 分页管理的固定单元；本文统一用"页"，vLLM 论文用 block，二者同义 |
| 页大小（page size） | 页面开头 | 每页包含的 token 数 |
| 连续预留（pre-allocation） | 第 1 章 | 旧方案：按请求最大可能长度一次性分配连续显存 |
| 预留槽（reserved slots） | 第 1 章 | 为未来将生成的 token 保留但尚未使用的空间 |
| 内部碎片（internal fragmentation） | 第 1 章 | 分配单元内部未被使用的部分（实际长度 < 分配长度） |
| 外部碎片（external fragmentation） | 第 1 章 | 分配器切出的、因尺寸不匹配而无法使用的空闲空洞 |
| 块表（block table） | 第 2 章 | 请求的逻辑块号到物理块号的映射表 |
| 按需分配（on-demand allocation） | 第 2 章 | token 写满当前页才申请新页的分配方式 |
| 逻辑 / 物理位置 | 第 2 章 | 序列中的顺序位置 / 显存中的实际地址 |
| 块级共享（block-level sharing） | 第 2 章 | 以页为粒度让多个序列共用同一份 KV cache（细节归 prefix-caching 页） |
| PCIe | 第 3 章 | CPU-GPU 互联总线（带宽语境；详见 gpu-communication 页） |
| 页间不连续 | 第 3 章 | 一个序列的各页在显存中物理上互不相邻 |
| 每页字节数 | 第 3 章 | 页大小 × 每 token KV cache 字节数（kv-cache 页 F1/F2 换算） |
