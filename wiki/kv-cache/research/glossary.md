# kv-cache 术语表

| 术语/符号 | 首次出现 | 定义或含义 |
|---|---|---|
| KV cache（键值缓存） | 页面开头 | 推理时保存的各层各 token 注意力 key/value 张量 |
| prefill（预填充） | 第 2 章 | 一次性并行处理输入 token、写入其 KV cache 并产出首个输出 token 的阶段 |
| decode（解码） | 第 2 章 | 逐 token 自回归生成、每步追加 1 token 缓存并读取全部缓存的阶段 |
| 自回归（autoregressive） | 第 1 章 | 每个新 token 依赖已生成的全部前文 |
| $q_t$、$k_i$、$v_i$ | 第 1 章 | 第 $t$ 个 token 的查询向量 / 第 $i$ 个 token 的 key、value 向量（沿用 standard-attention 页记号） |
| $L_{\text{layers}}$ | 第 3 章 | Transformer 层数 |
| $H_q$、$H_{\text{kv}}$ | 第 3 章 | 查询头数 / KV 头（组）数 |
| $d_{\text{head}}$ | 第 3 章 | 每注意力头维度 |
| $b$ | 第 3 章 | 每元素字节数（bf16/FP16=2，FP8=1） |
| MHA（multi-head attention） | 第 3 章 | 多头注意力：$H_{\text{kv}}=H_q$ |
| GQA（grouped-query attention） | 第 3 章 | 分组查询注意力：多组查询头共享一组 KV 头，$H_{\text{kv}}<H_q$ |
| MQA（multi-query attention） | 第 4 章折叠 | 极端 GQA：全部查询头共享 1 组 KV 头 |
| 上下文窗口（context window） | 第 4 章 | 模型可接受的最大输入 token 数（长度上限，非容量） |
| HBM / 显存 | 第 4 章 | GPU 高带宽内存；权重与 KV cache 共用 |
| 激活值（activation） | 第 4 章 | 前向计算的临时中间张量 |
| 逐出（eviction）/ 分层（tiering） | 第 4 章过渡句 | 显存不足时丢缓存重算 / 搬到更慢的存储层（仅过渡提及，细节在后两页） |
