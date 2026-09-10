# 跨层 KV 复用 核心论断与证据

来源代号：
- **Y** = YOCO 论文（Sun et al., *You Only Cache Once: Decoder-Decoder Architectures for Language Models*, arXiv:2405.05254v2）§2、Eq.(2)、Table 1–3
- **R** = DeepSeek-V4.1-Flash 技术报告（`wiki/deepseek-v4-1/research/official/tech_report.txt` 行号）§2.2、§2.3.1
- **SRC** = `wiki/deepseek-v4-1/research/official/inference/model.py` 行号
- **CFG** = `wiki/deepseek-v4-1/research/official/inference/config.json`
- **MEAS** = `wiki/deepseek-v4-1/research/verify_csa2_modes.py` 与 `ckpt/verify_csa2_modes.out`

## 核心论断（C）

### C1：复用对象——上段各层共用下段末层投影出的全局 KV
- 论断：把 $L$ 层分成自解码器（前 $L/2$）与交叉解码器（后 $L/2$）；自解码器逐层计算，其末层输出 $X^{L/2}$ 被投影成唯一的全局 KV，交叉解码器的各层通过交叉注意力复用这一份，不再从本层隐状态生成全局 KV。
- 来源定位：Y §2：「YOCO is stacked with L layers, where the first L/2 layers are self-decoder while the rest modules are cross-decoder.」；「The KV caches K̂, V̂ are reused by all the L/2 cross-decoder modules」
- 引文依据：Y Abstract：「a cross-decoder stacked upon a self-decoder. The self-decoder efficiently encodes global key-value (KV) caches that are reused by the cross-decoder via cross-attention.」
- 置信状态：**已确认**

### C2：缓存复杂度
- 论断：全局 KV 的份数与层数无关，KV cache 复杂度由 $\mathcal{O}(LND)$ 降为 $\mathcal{O}((N+L)D)$；相较标准 Transformer 约省 $L$ 倍缓存。
- 来源定位：Y Table 2、§2.3
- 引文依据：Y §2.3：「because global KV caches are reused and efficient self-attention needs constant caches, the number of caches is O(N+CL) ... about O(N) caches are required, i.e., you only cache once.」；「In comparison, Transformer decoders have to store N × L keys and values during inference. So YOCO roughly saves L times GPU memory for caches compared to Transformer decoders.」
- 适用条件：$CL \ll N$（长序列）；局部窗口 KV 仍需按层保存。
- 置信状态：**已确认**

### C3：prefill 可提前退出
- 论断：因为上段复用下段的输出，prefill 阶段可以在进入上段之前提前退出，计算复杂度由 $\mathcal{O}(LN^2D)$ 降为 $\mathcal{O}(LND)$，论文称至少省一半层计算。
- 来源定位：Y §2.3 与 Table 1/3
- 引文依据：Y §2.3：「we can exit early before entering the cross-decoder during the prefill stage.」；「First, only half the layers are needed for forward computation, i.e., at least half prefilling latency reduction.」
- 置信状态：**已确认**（论文结论；DeepSeek-V4.1-Flash 的报告表述为"nearly halves prefill computation"）

### C4：DeepSeek-V4.1-Flash 的两级复用与三模式
- 论断：CSA2 把"主 KV 与索引器 K 的共享"与"Top-K 索引的复用"解耦：Full 层自算主 KV 与索引器 K 并产出新索引；Reindex 层复用主 KV 与索引器 K、自算索引；Reuse 层两者都复用。与 CED 结合后，解码器的全局 KV 由解码器中被指派 Full 模式的层（层 20）从编码器末层隐状态计算。
- 来源定位：R §2.3.1（行 490–545）；SRC L496-503、L722-763
- 引文依据：R 行 500-505「Full Mode. The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices.」；R 行 536-540「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer」
- 置信状态：**已确认**（报告 + 源码 + MEAS）

### C5：运行期的共享分组
- 论断：每个压缩层实际读取的主 KV cache 都来自其组内的 source 层；实测分组为层 2–7、8–13、14–19、20–39，source 层分别是 2、8、14、20；source 层读到的是自己刚发布的 cache。
- 来源定位：CFG `kv_source_layers=[2,8,14,20]`；MEAS `ckpt/verify_csa2_modes.out`
- 引文依据：MEAS 第 5 节输出「每个压缩层读到的 cache 都来自其组内 source 层: True」「源层读到自己刚发布的 cache: True」
- 置信状态：**已确认**（本机实测，运行期 cache 对象归属）

### C6：分组的前提是同组压缩比一致
- 论断：同组层共用一份主 KV 的前提是压缩比相同（否则条目覆盖的 token 数不同、无法共用）；不同压缩比的组各有一份，跨组不能复用。
- 来源定位：CFG `compress_ratios`（层 2–19 为 2、层 20–39 为 1）；SRC L439（`compress_ratio = args.compress_ratios[layer_id]`）
- 引文依据：MEAS 输出中各组 ratio 唯一（层 2–7 / 8–13 / 14–19 为 2，层 20–39 为 1）
- 置信状态：**已确认**

### C7："once"只指全局 KV
- 论断：YOCO 的"只缓存一次"严格指全局 KV；自解码器仍需保存一定量的局部缓存，只是它被限制在常数大小（如窗口），在长序列下可忽略。
- 来源定位：Y 脚注 1
- 引文依据：「The word 'once' refers to global KV cache. Strictly, self-decoder also needs to store a certain number of caches. As the self-decoder utilizes an efficient attention module, the cache size is bounded to a constant, which can be ignored compared to global caches when the sequence length is large.」
- 置信状态：**已确认**

## 核心公式（F）

### F1：全局 KV 的投影
$$\hat K = \mathrm{LN}(X^{L/2})\,W_K,\qquad \hat V = \mathrm{LN}(X^{L/2})\,W_V$$
- 符号：$X^{L/2}$ 为自解码器末层输出；$W_K, W_V \in \mathbb{R}^{d\times d}$ 为可学习权重；$\hat K, \hat V$ 为被所有交叉解码器层复用的全局 KV。
- 来源定位：Y Eq.(2) 与「where W_K, W_V ∈ ℝ^{d×d} are learnable weights.」
- 置信状态：**已确认**

### F2：缓存复杂度对照
$$\text{Transformer: }\mathcal{O}(LND)\qquad\text{vs}\qquad\text{YOCO: }\mathcal{O}((N+L)D)$$
- 符号：$N$ 序列长度、$L$ 层数、$D$ 维度（$C$ 为窗口等常数）。
- 来源定位：Y Table 2 与 §2.3。
- 置信状态：**已确认**

### F3：prefill 计算复杂度对照
$$\text{Transformer: }\mathcal{O}(LN^2D)\qquad\text{vs}\qquad\text{YOCO: }\mathcal{O}(LND)$$
- 来源定位：Y Table 3 与 §2.3。
- 置信状态：**已确认**

## 外部数字与实验条件（N）

| 编号 | 数值 | 来源 | 条件 |
|---|---|---|---|
| N1 | YOCO 宣称缓存显存可减少约 80×（65B 模型） | Y §4.4 | 论文的实验设置；本页只登记宣称，不推导 |
| N2 | DeepSeek-V4.1-Flash 的共享分组：2–7 / 8–13 / 14–19 / 20–39，source 为 2 / 8 / 14 / 20 | MEAS + CFG | 真实层分配（等比缩小维度），运行期 cache 归属实测 |
| N3 | 分组内 ratio：层 2–19 为 2、层 20–39 为 1 | CFG `compress_ratios` | 与 HF 配置一致 |
| N4 | 报告称 CED + CSA2 让 prefill 计算接近减半 | R 行 322-326「This nearly halves prefill computation」 | 报告宣称；参考实现不含该优化（见 [C3] 与主页面边界说明） |

## 无法独立核对、需降级处理的条目

- N1 的 80× 与 YOCO 的加速数字：属论文实验条件下的测量，本页只登记。
- N4 的"接近减半"：属服务实现的优化，参考实现是单栈逐层执行；本页按报告宣称登记并注明边界。

## 构造示例与辅助解释

- **构造示例**：用 $L=40$、$N=1\,048\,576$ 说明"每层一份"与"每组一份"的份数差别（40 份 vs 4 份）。
- **辅助解释**：把跨层复用比作"整栋楼共用一套供水"而不是每层各打一口井；边界：它只覆盖全局 KV，每层仍需自己的"局部水管"（窗口 KV）。

## 简化条件

- 份数账按"每个 source 层各一份主 KV 与一份索引器 K"计算，不含局部窗口 KV、压缩器残态与运行时缓冲。
- 运行期实测用等比缩小维度（dim 64）的模型，验证的是共享关系（谁读谁的 cache），不是真实体量的性能。
