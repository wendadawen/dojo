# Multi-head Latent Attention（MLA）独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源逐条核查）
- 页面版本：index.html `f7449e581e05cfa5251290128692c10ac61ab312` / overview.html `f6bcbd3660a9e63a7917c6621c26ccf8d482c844`
- 时间：2026-08-09
- 审查依据：guides/concept/check.md
- 来源：DeepSeek-V2 论文（arXiv:2405.04434 v5，§2.1 Eq.1–19 + Table 1 + 摘要）；Kimi K3 技术报告 §2.1.2（第 352–376 行）；Kimi-K3 官方 config.json（huggingface.co/moonshotai/Kimi-K3/raw/main/config.json）

## 段 A 盲读小结

按页面顺序阅读 index.html，主线理解顺畅：开篇动机 → S1 KV 联合压缩 → S2 矩阵吸收 → S3 解耦 RoPE → S4 cache 对比 → S5 K3 改动。学习目标 5 题在正文中均有对应章节回答。前置概念页（GQA/MQA、低秩投影、RoPE、线性注意力）均已标注"待生成"占位。教学例子的折叠块均为补充，收起后不影响主线。

盲读中卡点集中在：S2 手算例子中 $W^{UK}$ 突然变成 $\mathbb{R}^{4 \times 3}$（S1 定义是 $\mathbb{R}^{d_h n_h \times d_c} = \mathbb{R}^{8 \times 3}$），$\mathbf{q}$ 也从 8 维变 4 维，未说明切换视角；S4 表格出现未定义符号 $n_g$。其余术语首现处均有解释，公式推导可跟进。

## 段 B 对照来源结论

核心论断与来源逐条对照：

- C1（KV 联合压缩）：Eq.(9)(10)(11) 完全一致，矩阵维度一致 ✓
- C2（全局注意力保留）：Eq.(18) 与 Eq.(7) 对照一致 ✓
- C3（RoPE 与吸收不兼容）：§2.1.3 第一段原文一致 ✓
- C4（矩阵吸收）：§2.1.2 末段"$W^{UK}$ can be absorbed into $W^Q$, and $W^{UV}$ can be absorbed into $W^O$"一致 ✓
- C5（K3 NoPE）：K3 报告 §2.1.2 第二段原文一致；config `mla_use_nope=true` ✓
- C6（K3 output gate）：K3 报告 Eq.(7) 一致；config `mla_use_output_gate=true` ✓
- C7（cache 公式与对照）：Table 1 四公式一致 ✓
- C8（93.3% baseline）：摘要"Compared with DeepSeek 67B"确认 ✓
- F1–F6 公式：均与论文/K3 报告一致 ✓
- N1（DeepSeek-V2 配置 $d=5120, n_h=128, d_h=128, l=60, d_c=512, d_c'=1536, d_h^R=64$）：§2.1.2 末段 + §3.1.2 一致 ✓
- N4（K3 config `q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, num_attention_heads=96, hidden_size=7168, num_hidden_layers=93`）：全部核对一致 ✓
- N5（K3 MLA 层 24 / KDA 层 69）：config `full_attn_layers` 24 项、`kda_layers` 69 项 ✓

## 问题

- [重要·技术] index.html L653（开篇第一段）："约 2 MB/token（BF16），128K 上下文下整个 KV cache 约 256 GB" 数字少算一半。按页面自己给的 $2 n_h d_h = 2 \times 128 \times 128 = 32768$ 数/层/token × 60 层 = 1,966,080 数/token；BF16 为 2 bytes/数，应为 3,932,160 bytes ≈ 3.75 MB/token；× 128K（131072）≈ 480 GB。页面将"1.97M 元素"误当"1.97 MB"，漏了 BF16 的 2 字节因子，导致 2 MB/token 和 256 GB 都少了一半。：将"约 2 MB/token（BF16）"改为"约 3.75 MB/token（BF16）"，将"约 256 GB"改为"约 480 GB"；或保留元素数表述"每 token 约 197 万个数"并明确 128K 下约 491 G 个数。修后须与 S4 的 $1{,}966{,}080$ 元素/token 一致。 ｜ 修复：已将 L653 改为"共 $1{,}966{,}080$ 个数/token，BF16 下约 3.75 MB/token，128K 上下文下整个 KV cache 约 480 GB"，显式给出元素数与字节数双重表述，与 S4 的 $1{,}966{,}080$ 元素/token 一致。 ｜ 复验：validate.py 通过（exit 0，"validation ok"）；S4 表格 L878 仍为 $1{,}966{,}080$ 元素/token，开篇数字与之一致。
- [重要·技术] index.html L773（S2 手算吸收）与 L834（S3 手算解耦）：例子中 $W^{UK} \in \mathbb{R}^{4 \times 3}$、$\mathbf{q} = (1,1,0,0) \in \mathbb{R}^4$，但 S1 L712 明确定义 $W^{UK} \in \mathbb{R}^{d_h n_h \times d_c} = \mathbb{R}^{8 \times 3}$、$\mathbf{q}_t \in \mathbb{R}^{d_h n_h} = \mathbb{R}^8$（贯穿例子 $d_h=4, n_h=2$）。S3 L836 的 $\mathbf{q}_{t,i}^C = (1,1,0,0)$ 同样是单头 4 维，与 S1 的 $\mathbf{q}_t^C \in \mathbb{R}^{d_h n_h}=\mathbb{R}^8$ 冲突。页面未说明 S2/S3 切换到了"单头视角"（即取第 $i$ 个头，$W^{UK}_i \in \mathbb{R}^{d_h \times d_c}$、$\mathbf{q}_{t,i} \in \mathbb{R}^{d_h}$），读者会卡在"为什么维度变了"。：在 S2 例子的引言句加一句说明，例如"为便于手算，下面取第 $i$ 个头视角：$\mathbf{q}_{t,i} \in \mathbb{R}^{d_h}=\mathbb{R}^4$、$W^{UK}_i \in \mathbb{R}^{d_h \times d_c}=\mathbb{R}^{4 \times 3}$（即 $W^{UK}$ 第 $i$ 个头对应的子块），吸收关系 $\mathbf{q}_{t,i}^\top W^{UK}_i \mathbf{c}_t^{KV} = (W^{UK\,\top}_i \mathbf{q}_{t,i})^\top \mathbf{c}_t^{KV}$ 与全头视角等价"；S3 同步标注 $\mathbf{q}_{t,i}^C$ 是单头 content query。 ｜ 修复：S2 引言句已加单头视角说明（$\mathbf{q}_{t,i} \in \mathbb{R}^{d_h}=\mathbb{R}^4$、$W^{UK}_i \in \mathbb{R}^{d_h \times d_c}=\mathbb{R}^{4 \times 3}$ 是 $W^{UK}$ 第 $i$ 个头子块，吸收关系与全头视角等价），并把后续 $\mathbf{q}, W^{UK}, \mathbf{k}_t^C, \mathbf{q}'$ 同步改为带下标 $i$ 的 $\mathbf{q}_{t,i}, W^{UK}_i, \mathbf{k}_{t,i}^C, \mathbf{q}_{t,i}'$；S3 引言句改为"沿用 S2 的单头视角"并把 $\mathbf{q}^C, \mathbf{k}^C$ 标注为单头 content query $\mathbf{q}_{t,i}^C, \mathbf{k}_{t,i}^C$。 ｜ 复验：validate.py 通过（exit 0，"validation ok"）；S1 的 $W^{UK} \in \mathbb{R}^{8 \times 3}$ 定义未动，S2/S3 的单头视角说明与 S1 全头定义已通过下标 $i$ 显式衔接。
- [重要·技术] index.html L937（S5 output gate 符号说明）：" $\tilde{\mathbf{o}}_t$ 是未 gate 的 MLA 输出（即 S1 公式里的 $\mathbf{u}_t$，只是换记号强调"ungated"）"与 gate 公式 L933 $\mathbf{y}_t = W^O [\mathrm{Sigmoid}(W^g \mathbf{x}_t) \odot \tilde{\mathbf{o}}_t]$ 矛盾。S1 L758 的 $\mathbf{u}_t = W^O \sum_j \mathrm{softmax} \mathbf{v}_j^C$ 是 $W^O$ 之后的输出；而 gate 公式中 $W^O$ 在外层，作用在 $[\mathrm{Sigmoid} \odot \tilde{\mathbf{o}}_t]$ 上，说明 $\tilde{\mathbf{o}}_t$ 是 $W^O$ 之前的拼接向量 $[\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}]$，不是 $W^O$ 之后的 $\mathbf{u}_t$。若 $\tilde{\mathbf{o}}_t = \mathbf{u}_t$，gate 公式会出现两次 $W^O$。K3 报告 §2.1.2 原文"Let õt denote the ungated MLA output at position t"配合 Eq.(7) 的 $W^O$ 外层位置也支持此判断。：将"即 S1 公式里的 $\mathbf{u}_t$"改为"即 S1 公式里 $W^O$ 输入端的拼接向量 $[\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}]$（$W^O$ 之前）"，避免与 $\mathbf{u}_t$（$W^O$ 之后）混淆。 ｜ 修复：L937 已改为"$\tilde{\mathbf{o}}_t$ 是未 gate 的 MLA 输出（即 S1 公式里 $W^O$ 输入端的拼接向量 $[\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}]$，$W^O$ 之前；区别于 $W^O$ 之后的 $\mathbf{u}_t$，仅强调"ungated"）"，消除了与 gate 公式中 $W^O$ 外层位置的矛盾。 ｜ 复验：validate.py 通过（exit 0，"validation ok"）；gate 公式 L933 中 $W^O$ 在外层、$\tilde{\mathbf{o}}_t$ 在内层，与 S1 的 $\mathbf{u}_t = W^O \sum_j \mathrm{softmax} \mathbf{v}_j^C$（$W^O$ 之后）区分清晰，不再出现"两次 $W^O$"的矛盾。
- [轻微·盲读] index.html L868（S4 表格 GQA 行）：$n_g$ 首次出现但未定义。论文 Table 1 注释说"$n_g$ denotes the number of groups in GQA"。小白不知道 $n_g$ 含义。：在表格下方或 GQA 行首次出现处加"$n_g$ 为 GQA 的 KV 头分组数"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] overview.html L68："GQA = $2 n_g d_h l$"同样未定义 $n_g$。：与上条同步补充 $n_g$ 定义。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html L956（S5 gate 例子解读）："gate 在通道 0 放大（gate $> 0.5$，原值 1 → 0.73，相对其他通道被强调）"用词不准确。1 → 0.73 是数值缩小（gate < 1），"放大"通常指乘以 > 1 的因子，会让读者误以为 gate > 1。准确表述是"相对保留更多"或"相对其他通道被强调"。：将"放大"改为"相对保留"或"强调"；保留"相对其他通道被强调"的解释。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 3
- 处置：进入修复

重要问题均为数字/符号/维度一致性问题，修法明确且改动局部，不涉及研究范围或教学大纲调整。修复后须重跑 validate.py 并复验。

核心机制（KV 联合压缩、矩阵吸收、解耦 RoPE、K3 NoPE 与 output gate）与来源逐条对照均一致；DeepSeek-V2 与 K3 的 config 数值（$d_c=512, d_c'=1536, d_h^R=64, n_h=128/96, l=60/93$，K3 MLA 24 层 / KDA 69 层）全部核对通过；93.3% baseline（DeepSeek 67B）澄清正确；K3 NoPE 下 cache 不确定性的 callout 标注得当。
