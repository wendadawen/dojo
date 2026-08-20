# kv-cache 核心论断与证据

编号规则：C 论断 / F 公式 / N 数字。来源优先级：原始论文 > 权威文档。全部条目置信：已确认（除非另注）。

## C 论断

- C1：Transformer 推理中，KV cache 指与注意力机制关联的 key/value 张量，表示前文 token 的上下文，用于按序生成新 token。来源：vLLM 论文（arXiv:2309.06180）§1（"these states consist of the key and value tensors associated with the attention mechanism, commonly referred to as KV cache, which represent the context from earlier tokens"）。
- C2：LLM 推理分 prefill 与 decode 两阶段：prefill 处理新输入 token 与上下文 token 并产生 KV cache；decode 自回归逐 token 生成，持续复用并扩展 KV cache。来源：Strata 论文（arXiv:2508.18572v2）§2.1（"LLM inference operates in two phases: prefill and decode. During prefill, the model typically processes both (i) new tokens from the user query and (ii) context tokens... In the subsequent decode phase, the model generates tokens autoregressively, continually reusing and extending the KV cache"）。
- C3：K/V 只依赖已处理 token 与模型参数、与当前查询 token 无关，因此可对每 token 计算一次并跨步复用——这是缓存的可行性基础。来源：标准注意力公式结构（standard-attention 页）+ 推断标记：该"与查询无关"表述为对注意力公式的直接观察（K/V 由各 token 自身的输入经同一权重矩阵映射得到）。
- C4：13B 参数模型在 A100 40GB 上，约 65% 显存为模型权重、近 30% 为请求动态状态（KV cache），其余少量为激活。来源：vLLM 论文 §1 Figure 1 及正文。
- C5：SGLang 指出既有系统在生成请求完成后即丢弃 KV cache，阻止跨请求复用；其 RadixAttention 保留缓存（本页仅在 Q4 过渡处用一句，细节归 prefix-caching 页）。来源：SGLang 论文（arXiv:2312.07104）§3.2。

## F 公式

- F1：每 token KV cache 字节数 $=2\cdot L_{\text{layers}}\cdot H_{\text{kv}}\cdot d_{\text{head}}\cdot b$。其中 2 为 K 与 V 两份；$L_{\text{layers}}$ 层数、$H_{\text{kv}}$ 每 KV 头组数（GQA 下的 KV 头数）、$d_{\text{head}}$ 每头维度、$b$ 每元素字节数。来源：vLLM 论文 §3 OPT-13B 算例的推广形式（原文算式 "2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)" 为 MHA 情形 $H_{\text{kv}}\cdot d_{\text{head}}=d_{\text{hidden}}$；GQA 形式为对多头结构的直接推广，标注推断性质：由注意力结构直接得出，与社区通用口径一致）。置信：已确认（两算例均可复算）。
- F2（F1 的直接推论，中间式）：单请求 KV cache 总量 $=$ token 数 × 每 token 字节数；MHA 模型每 token $=2\cdot L\cdot d_{\text{hidden}}\cdot b$。

## N 数字

- N1：OPT-13B 单 token KV cache 800 KB（2 × 5120 × 40 × 2）；单请求最长（2048 token）1.6 GB。来源：vLLM 论文 §3。
- N2：Llama-3.1-8B-Instruct：32 层、8 KV 头、$d_{\text{head}}=128$、上下文窗口 128K、权重 bf16。来源：Meta Llama 3.1 模型卡/HuggingFace 配置（写页面时核实）。由此按 F1 得 128 KB/token（构造算例：20,000 token → 2.5 GB）。
- N3：13B 模型 A100 40GB 显存分布：约 65% 权重、近 30% KV cache。来源：vLLM 论文 §1。
- N4：Strata 论文口径：40 GB HBM 对 Llama-8B 约只存 0.3M token（与 N2×F1 复算 40GB/128KB≈0.31M 一致，交叉验证用，正文标为 Strata 论文数字）。来源：Strata 论文 §1。
- N5：128K token 单请求 KV cache（Llama-3.1-8B、128 KB/token）= 16 GB。来源：F1 × N2 推算（构造算例，标注）。

## 冲突与不确定项

- 无实质冲突。F1 的 GQA 推广形式已标注推断性质并给出 OPT（MHA）与 Llama（GQA）两个可复算算例。
