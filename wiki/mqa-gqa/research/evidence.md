# MQA 与 GQA 核心论断与证据

编号约定：C 论断 / F 公式 / N 数字。仅覆盖核心内容。

## C 论断

### C1 自回归推理的瓶颈是内存带宽而非算力

- 论断内容：Transformer 自回归解码时，每生成一个 token 都要把之前所有 token 的 K/V 从 HBM 加载一遍；这部分数据量随头数 h 与序列长度 n 线性增长，且 GPU 算力增长快于内存带宽增长，使解码受内存带宽限制而非算力限制。
- 来源定位：Shazeer 2019, arXiv:1911.02150, §1 第 2 段"the speed of incremental Transformer inference on modern computing hardware is limited by the memory bandwidth necessary to reload the large 'keys' and 'values' tensors"；§2.4.1 给出性能分析。
- 适用条件：自回归解码（逐 token 生成）、batch 较小或序列较长时；训练时全序列并行、KV 一次性算出，不受此瓶颈。
- 置信状态：已确认。

### C2 内存访问与算术的比值为 Θ(n/d + 1/b)

- 论断内容：Shazeer 2019 §2.4.1 的性能分析给出，incremental MHA 每步解码的"内存访问量 / 算术运算量"比值随 $\Theta(n/d + 1/b)$ 增长，其中 n 为当前序列长度、d 为模型维度、b 为 batch size。当 n 接近 d 或 b 小时比值接近 1，GPU 在等内存。
- 来源定位：Shazeer 2019, §2.4.1 Performance Analysis；§3.1 Performance Analysis for Incremental Multi-Query Attention 直接给出"the offensive n/d by a factor of h"的结论。
- 适用条件：现代 GPU/TPU 的 roofline 模型下；n 与 d 同量级或 b 小时成立。
- 置信状态：已确认（论文一手分析，原文有 "We have reduced the offensive n/d by a factor of h"）。

### C3 MQA 把所有 query 头的 K/V 共享为一组

- 论断内容：MQA 与 MHA 的唯一区别是 K/V 投影去掉头维度——$P^Q,P^O$ 保持 $\mathbb{R}^{h\times d\times d_k}$ 多头，$P^K,P^V$ 变为 $\mathbb{R}^{d\times d_k}$、$\mathbb{R}^{d\times d_v}$ 全模型一组；所有 query 头读同一组 K 和 V。
- 来源定位：Shazeer 2019, §3，原文 "Multi-query attention is identical except that the different heads share a single set of keys and values."；§2.2 给出 MHA 的 einsum 形状作为对照。
- 适用条件：无特殊条件，是 MQA 的定义。
- 置信状态：已确认。

### C4 MQA 把每 token 每 layer 的 KV cache 减为 2 d_k（减少 h 倍）

- 论断内容：MHA 每 token 每 layer cache $2 h d_k$（h 个 K 头 + h 个 V 头，各 $d_k$ 维）；MQA 共享一组 K/V 后 cache $2 d_k$；减少倍数为 h。
- 来源定位：由 C3 直接推出；Shazeer 2019 §3.1 末段 "We have reduced the offensive n/d by a factor of h"。
- 适用条件：head_dim $d_k$ 不变、只减头数。
- 置信状态：已确认。

### C5 MQA 有质量代价与训练不稳定性

- 论断内容：MQA 质量略低于 MHA（"minor quality degradation"，Shazeer 2019 abstract）；大模型上 MQA 训练不稳定，GQA 则稳定。
- 来源定位：Shazeer 2019 abstract "incur only minor quality degradation from the baseline"；Ainslie 2023 §1 "MQA can lead to quality degradation"；Ainslie 2023 Appendix A "multi-query attention can lead to training instability during fine-tuning...Uptrained grouped-query attention models, however, appear to be stable."
- 适用条件：质量代价在小模型上可忽略，大模型与长输入任务上更明显。
- 置信状态：已确认。

### C6 GQA 把 query 头分 G 组、每组共享一组 K/V，是 MHA 与 MQA 的插值

- 论断内容：GQA 把 h 个 query 头分成 G 组，每组共享一组 K/V；$G=h$ 等价于 MHA，$G=1$ 等价于 MQA，$1<G<h$ 为中间插值；KV cache 为 $2 G d_k$ 每 token 每 layer。
- 来源定位：Ainslie 2023, arXiv:2305.13245, §2，原文 "Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head."；Figure 2 给出 MHA/GQA/MQA 的对照。
- 适用条件：无特殊条件，是 GQA 的定义。
- 置信状态：已确认。

### C7 uptraining 用均值池化 + 5% 继续预训练

- 论断内容：从已有 MHA 检查点转 GQA/MQA 分两步——（a）把 h 个 K/V 投影矩阵按组均值池化为 G 个（均值池化优于选第一个或随机初始化）；（b）用原预训练计算量的 α=5% 继续预训练。
- 来源定位：Ainslie 2023 §2.1 "The projection matrices for key and value heads are mean pooled...we find works better than selecting a single key and value head or randomly initializing new key and value heads from scratch."；§3.1 "The converted checkpoint is then pre-trained for a small proportion α of its original training steps"；α=0.05 见实验配置。Figure 4 给出三种转换方法的消融，Figure 5 给出 α 的影响。
- 适用条件：已有 MHA 检查点；α=5% 是论文选定值，10% 后收益递减。
- 置信状态：已确认。

### C8 GQA-8 在 T5-XXL 上质量接近 MHA、速度接近 MQA，G=8 是选定折中点

- 论断内容：Ainslie 2023 在 T5-XXL 上实验——MHA-XXL 推理 1.51s/样本、平均质量 47.2；MQA-XXL 0.24s、46.6；GQA-8-XXL 0.28s、47.1。GQA-8 质量接近 MHA、速度接近 MQA。G=8 是 Figure 6 消融后选定的折中点。
- 来源定位：Ainslie 2023 Table 1（实验数据）、Figure 6（G 的消融）；§4 "We selected 8 groups as a favorable middle ground."
- 适用条件：T5-XXL 架构与实验配置；G=8 不是普适最优，但论文认定为该规模的折中点。
- 置信状态：已确认。

### C9 MQA/GQA 与 MLA 是两条不同的 KV 压缩路线

- 论断内容：MQA/GQA 通过共享 K/V 头减 cache，被缓存的每个 K/V 仍是完整的 $d_k$ 维 head 向量、只是份数减少；MLA 不共享头，而是把所有头的 K/V 联合压成 $d_c$ 维潜向量 $c_t^{KV}$，推理时再用上投影重建各头 K/V。机制不同，cache 公式不同：MQA = $2 d_k l$、GQA = $2 G d_k l$、MHA = $2 h d_k l$、MLA = $(d_c + d_h^R) l$。
- 来源定位：Shazeer 2019 §3（MQA 共享机制）；Ainslie 2023 §2（GQA 分组机制）；DeepSeek-V2 §2.1.2 Eq.(9)–(11)（MLA 压缩公式）、§2.1.4 Table 1（四种机制 cache 对照）。
- 适用条件：无特殊条件。
- 置信状态：已确认。

## F 公式

### F1 MHA 每 token 每 layer KV cache

- 公式：$\text{KV cache}_{\text{MHA}} = 2 h d_k$（每 token 每 layer 的元素数）；$n$ 个 token、$l$ 层为 $2 h d_k n l$。
- 来源：由 Vaswani 2017 §3.2.2 多头定义直接推出（每头一份 K 和一份 V，各 $d_k$ 维）；前置页 standard-attention 已建立。
- 置信状态：已确认。

### F2 MQA 每 token 每 layer KV cache

- 公式：$\text{KV cache}_{\text{MQA}} = 2 d_k$（每 token 每 layer）；减少 h 倍。
- 来源：由 C3（共享一组 K/V）直接推出；Shazeer 2019 §3.1 "reduced...by a factor of h"。
- 置信状态：已确认。

### F3 GQA 每 token 每 layer KV cache

- 公式：$\text{KV cache}_{\text{GQA}} = 2 G d_k$（每 token 每 layer，G 为组数）。
- 来源：由 C6（G 组各一组 K/V）直接推出；Ainslie 2023 §2。
- 置信状态：已确认。

### F4 MLA 每 token 每 layer KV cache（对照用，不展开机制）

- 公式：$\text{KV cache}_{\text{MLA}} = (d_c + d_h^R)$（每 token 每 layer，$d_c$ 为压缩潜维度、$d_h^R$ 为解耦 RoPE 维度）。
- 来源：DeepSeek-V2 §2.1.4 Table 1；本页只引公式对照，不展开 $c_t^{KV}=W^{DKV}h_t$ 等机制（属 MLA 页）。
- 置信状态：已确认。

### F5 Shazeer 2019 性能分析比值

- 公式：incremental MHA 每步解码的内存访问/算术比值 $\sim \Theta(n/d + 1/b)$；MQA 把 n/d 项的系数减为 1/h。
- 来源：Shazeer 2019 §2.4.1、§3.1。
- 置信状态：已确认。

## N 数字

### N1 Shazeer 2019 实验定性结论

- 数字：Shazeer 2019 abstract 报告 MQA "much faster to decode" 且 "incur only minor quality degradation from the baseline"；实验在 WMT 翻译与语言建模任务上。
- 来源：Shazeer 2019 abstract、§4 Experiments and Results。
- 实验条件：WMT 翻译任务与语言建模；具体 BLEU 与速度倍数未在本文核到一手精确数值，不引用二手数字。
- 置信状态：定性结论已确认；具体倍数因 PDF 一手数据未取到精确值，标注为"未核实"不在正文引用。

### N2 Ainslie 2023 Table 1 T5-XXL 实验数据

- 数字：
  - MHA-XXL：推理 1.51s/样本（每 TPUv4 chip），平均质量 47.2
  - MQA-XXL：0.24s，46.6
  - GQA-8-XXL：0.28s，47.1
  - MHA-Large：0.37s，46.0（作为小模型对照）
- 来源：Ainslie 2023 Table 1；评估数据集为 CNN/DailyMail、arXiv、PubMed、MediaSum、MultiNews、WMT EnDe、TriviaQA 的平均。
- 实验条件：T5-XXL 架构（MHA 检查点来自公开 T5.1.1）；MQA/GQA 通过 uptraining（α=5%）得到；MQA/GQA 仅用于 decoder self-attention 与 cross-attention，encoder self-attention 仍 MHA。
- 置信状态：已确认。

### N3 GQA-8 折中点选择

- 数字：Ainslie 2023 §4 选定 G=8 为折中点；Figure 6 显示从 G=1（MQA）到 G=8 推理开销适度增加，继续增大 G 收益递减。
- 来源：Ainslie 2023 §4 "We selected 8 groups as a favorable middle ground."；Figure 6。
- 实验条件：T5-XXL 规模。
- 置信状态：已确认。

### N4 uptraining 计算开销

- 数字：α=5% 的继续预训练对 T5-XXL 约消耗 600 TPUv3 chip-days。
- 来源：Ainslie 2023 §3.1。
- 实验条件：T5-XXL 规模；不同模型规模开销不同。
- 置信状态：已确认。

### N5 DeepSeek-V2 MLA 配置（对照用）

- 数字：DeepSeek-V2 配置 $n_h=128$、$d_h=128$、$d_c=512$、$d_h^R=64$；MLA 每 token 每 layer cache = $(512+64)=576$，对应 MHA = $2\times 128\times 128=32768$，比值约 1/57。
- 来源：DeepSeek-V2 §2.1.4 Table 1；MLA 页 scope.md §2.3 已记录。
- 实验条件：DeepSeek-V2 配置。
- 置信状态：已确认；本页只用于 Q5 对照，不展开 MLA 机制。
