# 滑动窗口注意力 核心论断与证据

来源代号：
- **M** = Mistral 7B 论文（Jiang et al., *Mistral 7B*, arXiv:2310.06825）§2 "Architectural details"
- **S** = StreamingLLM 论文（Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, arXiv:2309.17453）§3
- **CFG** = DeepSeek-V4.1-Flash 官方推理配置 `wiki/deepseek-v4-1/research/official/inference/config.json`
- **SRC** = `wiki/deepseek-v4-1/research/official/inference/model.py` 行号
- **MEAS** = 本页 `research/verify_swa_accounts.py` 与 `verify_swa_accounts.out`；跨页引用 `wiki/deepseek-v4-1/research/ckpt/verify_sparse_attn_window.out`

## 核心论断（C）

### C1：窗口定义与逐层前移
- 论断：位置 $i$ 在第 $k$ 层的隐状态只关注上一层位置在 $[i-W, i]$ 的隐状态；信息每层最多前移 $W$，$k$ 层后最多前移 $k\times W$。
- 来源定位：M §2（"Sliding Window Attention" 段与 Figure 1）
- 引文依据：「the hidden state in position i of the layer k, h_i, attends to all hidden states from the previous layer with positions between i−W and i. Recursively, h_i can access tokens from the input layer at a distance of up to W×k tokens」；「At each attention layer, information can move forward by W tokens. Hence, after k attention layers, information can move forward by up to k×W tokens.」
- 适用条件：窗口按层独立、固定大小；不含额外全局连接。
- 置信状态：**已确认**

### C2：环形缓冲（rolling buffer cache）
- 论断：缓存固定为 $W$ 条，位置 $i$ 的键值写入槽位 $i \bmod W$；位置超过 $W$ 后旧值被覆盖，缓存大小停止增长。
- 来源定位：M §2（"Rolling Buffer Cache" 段与 Figure 2）
- 引文依据：「The cache has a fixed size of W, and the keys and values for the timestep i are stored in position i mod W of the cache. As a result, when the position i is larger than W, past values in the cache are overwritten, and the size of the cache stops increasing.」
- 置信状态：**已确认**（并有 MEAS 规则复算）

### C3：窗口单独使用会在超长流式场景失效
- 论断：只保留最近窗口的注意力在序列长度超过缓存规模后会崩（困惑度暴增），原因是初始 token 承担了注意力汇聚点的角色，移除它们会移走 softmax 分母的一大块。
- 来源定位：S §3.1、§3.2、Table 1/Table 2
- 引文依据：「Window attention (0+y) has a drastic increase in perplexity.」（Table 2 语境）；「removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function in attention computation. This alteration leads to a significant shift in the distribution of attention scores away from what would be expected in normal inference settings.」
- 适用条件：指"只用窗口、丢弃初始 token"的做法；保留 sink 后窗口注意力可稳定工作。
- 置信状态：**已确认**（论文结论；本页只陈述结论并指向注意力汇聚点专页）

### C4：DeepSeek-V4.1-Flash 的窗口实例
- 论断：V4.1-Flash 的窗口 $W=128$；每层都有窗口注意力（前两层只有窗口）；窗口 KV 保持 FP8；窗口 KV 与压缩全局 KV 拼进同一次稀疏注意力调用。
- 来源定位：CFG `window_size=128`；SRC L700-720（`_window_kv`，含 `act_quant(kv, fp8_block_size, ...)` 与环形缓冲写入）、L774-778（`torch.cat([kv, compress_kv], dim=1)`）、L707 精度注释。
- 引文依据：SRC L701-702 注释「This layer's sliding-window K and the window positions every query may attend to. The K stays fp8, quantized over the whole post-RoPE vector」；R 行 698-700「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」
- 置信状态：**已确认**

### C5：位置编码用原序列位置，不用缓存内位置
- 论断：V4.1 给窗口 KV 施加 RoPE 时用原序列位置（prefill 取 `freqs_cis[start_pos : start_pos+seqlen]`，decode 取 `freqs_cis[start_pos]`）；StreamingLLM 为了让初始 sink token 与滚动窗口共处一个缓存，改用"缓存内位置"。
- 来源定位：SRC L767（`freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]`）；S §3.2
- 引文依据：S §3.2「StreamingLLM focuses on positions within the cache rather than those in the original text. This distinction is crucial for StreamingLLM's performance.」
- 适用条件：V4.1 的窗口只含最近 $W$ 个位置，因此窗口内相对距离不超过 $W$；sink 场景需要跨很远的距离时，位置口径的选择会改变结果。
- 置信状态：**已确认**（源码 + 论文）

## 核心公式（F）

### F1：感受野
$$\text{span}(k) = k \times W$$
- 来源定位：M §2（k×W 表述）；MEAS 用 V4.1 的 $W=128$、$k=40$ 复算得 5120。
- 置信状态：**已确认**

### F2：环形缓冲槽位
$$\text{slot}(i) = i \bmod W$$
- 来源定位：M §2 引文；SRC L718 `self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`。
- 置信状态：**已确认**

### F3：每层注意力打分次数
$$n_{\text{score}}(N) = N \times W \quad(\text{窗口}) \qquad \text{vs} \qquad N^2 \ (\text{全注意力})$$
- 来源定位：M §2 Figure 1 说明「The number of operations in vanilla attention is quadratic in the sequence length」；MEAS 用 $N=4096/131072/1048576$ 复算。
- 置信状态：**已确认**

## 外部数字与实验条件（N）

| 编号 | 数值 | 来源 | 条件 |
|---|---|---|---|
| N1 | Mistral 7B：$W=4096$、32 层、上下文 8192；理论上限注意力跨度约 131K | M Table 1、§2 | 4096 × 32 = 131072 |
| N2 | Mistral 7B：32K 序列下滚动缓存把缓存显存减少 8× | M §2 | 32768 / 4096 = 8 |
| N3 | StreamingLLM：Llama-2-13B 在 PG19 上，纯窗口（0+1024）PPL 5158.07，加 4 个初始 token（4+1020）后 5.40 | S Table 1 | 65K token 的首本书；x+y 表示 x 个初始 token + y 个近期 token |
| N4 | StreamingLLM：只用可学习 sink token 训练的模型在 1+1023 配置下 PPL 18.01，而 vanilla 模型需多个初始 token | S Table 3 | 160M 参数模型、PG19 第一个样本 |
| N5 | V4.1-Flash：$W=128$、窗口 KV 每层 64 KiB、40 层共 2.50 MiB；相对 1M 全缓存为 1/8192 | CFG + MEAS | FP8（1 B/元素）、512 维、单序列 |
| N6 | V4.1-Flash：40 层感受野 5120 个 token，占 1M 的 0.49% | MEAS | $W=128$ |
| N7 | V4.1-Flash：decode 环形槽位映射回的位置集合与 prefill 因果窗口在 300 步上完全一致 | MEAS + 跨页 `verify_sparse_attn_window.out` | 单序列、$W=128$；官方函数在 W=8 时同样成立 |

## 无法独立核对、需降级处理的条目

- Mistral 的"2x 速度提升"与 StreamingLLM 的"22.2× 加速"属具体 kernel 与实现条件下的测量，本页不引用。
- StreamingLLM 的 PPL 数值按论文原值登记（N3、N4），本页不重跑实验。

## 构造示例与辅助解释

- **构造示例**：用 $W=3$、4 层说明感受野（手算 $4\times3=12$）；用 V4.1 的 $W=128$ 计算 64 KiB 与 2.50 MiB。
- **辅助解释**：把层堆叠比作"接力传话"，每层只能把消息往前传 $W$ 步；边界：它不产生新的信息，只是让远处信息逐层靠近，因此长距离信息的保真度随层数衰减——这一点本页只作定性提示，不量化。

## 简化条件

- 字节账只算窗口 KV，不含压缩全局 KV、索引器缓存与压缩器残态。
- 复杂度按"每个 query 的打分次数"计数，不含 softmax、值加权与投影的开销。
- 索引规则复算用 $W=128$、序列长 300 与 $W=8$、长 20 两种规模，未覆盖多 batch 与 chunked prefill 的分块边界。
