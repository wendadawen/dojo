# 恒等超连接（iHC）审查记录（第 1 轮）

- 页面版本：index.html 工作树哈希 `0278e7516cd722ae8b89165a50b21a906135e074`
- 审查时间：2026-09-03 14:06
- 审查者：独立审查者（未参与写作与前序轮次，未读取 research/ 下任何规划产物）
- 已完整阅读章节：按顺序通读 index.html 全文——引言（meta 引言、核心问题、常见误解）、1. iHC 拿掉了混合矩阵（1.1、1.2、本章问题）、2. 一个子块的完整路径（2.1、2.2、本章问题）、3. 读写门从哪来、从哪里出发（3.1、3.2、3.3、本章问题）、4. 把混合矩阵钉死在 $I$ 的理由（4.1、4.2、4.3、本章问题）、5. Hy4 里的 iHC：配置、权重与代价（5.1、5.2、5.3、本章问题）、来源与范围说明（六个小节）；以及 overview.html 全文。所有折叠块（含两段代码与预期输出）均已展开阅读。

## 核对方法

外部来源均由审查者本机自行抓取核对：Hy4 README（GitHub main 与 HF 模型卡两份）、config.json、vLLM PR #54160 的 .diff（含 hc.py / model.py / hpc_ihc.py 全文）、model.safetensors.index.json 与 4 个 shard 头（HTTP range 拉取）、知乎文章（zhuanlan.zhihu.com/p/2010852389670908320）、HC 论文 arXiv:2409.19606v3（HTML 全文）、mHC 论文 arXiv:2512.24880v2（HTML 全文）、transformers main 的 modeling_glm5_next.py 与 configuration_glm5_next.py。

## 来源核对记录（引文依据）

### 论断与来源（C）

- **C1 ✓**：README Model Introduction："The residual pathway uses [iHC (identity Hyper-Connections)](https://zhuanlan.zhihu.com/p/2010852389670908320) to expand inter-layer information flow."；规格表行 "Residual Streams | 4"。HF 模型卡同句存在（line 65）。注意：README 正文无日期字段；「2026-08-28 发布」与仓库元数据一致（GitHub created 2026-08-27 / pushed 2026-08-28；HF created 2026-08-27 / lastModified 2026-08-28），日期本身可接受，但依据是仓库元数据而非 README 文内日期。
- **C2 ✓**：hc.py 模块 docstring："Each decoder sub-block reduces the channels to one hidden state (``HYV4HCPreLayer``), runs the sub-block, then scatters the result back over the channels (``HYV4HCPostLayer``). The final ``HYV4HCHeadLayer`` merges the channels before the model's output norm."——pre 收缩、post 散射、head 合并与页面表述一致。
- **C3 ✓**：hpc_ihc.py HpcIHCPost docstring："Unlike mHC there is no comb matrix, so each output channel only needs its own residual channel and the whole thing is one fused multiply-add per element."——与页面引文逐字一致。
- **C4 ✓**：hc.py 首行 "iHC (independent Hyper-Connections) layers for HY V4 (NVIDIA)."；hpc_ihc.py 首行 "HPC fused iHC (independent Hyper-Connections) kernels for HY V4."（"independent" 恰为两处）；README/HF 模型卡写 "identity Hyper-Connections"。
- **C5 ✓**：model.py HYV4DecoderLayer.__init__ 中 `self.hc_attn_layer = HYV4HCLayer(config, layer_idx, prefix=f"{prefix}.hc_attn_layer")` 与 `self.hc_mlp_layer = HYV4HCLayer(...)`——每层两个独立实例、独立参数前缀；dense 层（mlp_layer_types[0]=="dense"）同样构造 hc_mlp_layer。checkpoint 侧：index.json 中 hc_pre.hc_fn 共 156 个，layers 0–77 的 hc_attn_layer/hc_mlp_layer 各 78 个、完整无缺。
- **C6 ✓**：model.py `_forward_ihc`：hc_attn_layer.pre → input_layernorm → self_attn → hc_attn_layer.post；hc_mlp_layer.pre → post_attention_layernorm → mlp → hc_mlp_layer.post；注释 "# Under iHC the residual is carried inside hidden_states."——与页面引文一致，前向顺序一致。
- **C7 ✓**：hc.py `_prepare_input_to_3d`："return hidden_states.unsqueeze(1).repeat(1, hc, 1)"（2D 输入广播复制）；model.py HYV4Model.forward：`if self.enable_ihc: hidden_states = self.hc_head(hidden_states)` 之后 `return self.norm(hidden_states)`——head 在 final norm 之前。
- **C8 ✓**：hc.py reset_parameters：`nn.init.constant_(self.hc_scale, 0.01)`；`self.hc_base[: self.hc_mult].fill_(-torch.log(torch.tensor(self.hc_mult - 1.0, ...)))`；`self.hc_base[self.hc_mult : 2 * self.hc_mult].fill_(0.0)`；head 侧 `nn.init.constant_(self.hc_head_scale, 0.01)`、`self.hc_head_base.fill_(-log(hc_mult - 1))`——与页面（$s=0.01$、读偏置 $-\ln(n-1)$、写偏置 0、head 偏置全 $-\ln(n-1)$）一致。
- **C9 ✓**：hc.py forward 内 `x.flatten(1).float()`、`params_dtype=torch.float32`；hpc_ihc.py 模块 docstring："hc_fn weights stay float32 (the checkpoint keeps them out of fp8 quantization via modules_to_not_convert)"。
- **C10 ✓**：config.json：`"enable_ihc": true`、`"hc_mult": 4`、`"hc_magnitude": 2.0`、`"hc_eps": 1e-06`、`"hidden_size": 6144`、`"num_hidden_layers": 78`、`"rms_norm_eps": 1e-05`、`"mlp_layer_types": ["dense", "sparse"×77]`——页面配置表 8 行全部一致。
- **C11 ✓**：index.json 实测：hc 相关张量共 471 个 = hc_pre.hc_fn ×156 + hc_pre.hc_scale ×156 + hc_pre.hc_base ×156 + hc_head_fn/scale/base 各 1；抽样 shard 头全部 F32。
- **C12 ✓**：本机复算：每站点 8×24576+2+8=196,618，head 4×24576+1+4=98,309，156×196,618+98,309=30,770,717；shard 头形状支持（见 N3）。
- **C13 ✓**：index.json：model.mtp_layers.* 恰 27 个张量（eh_proj、hnorm、enorm、input/post_attention/final_layernorm、self_attn.*、mlp.*），其中无任何 hc 张量。
- **C14 ✓**：知乎文章（2026-02-28 编辑）："单层 H_res（depth=1）：接近单位阵（对角线 ~0.96，非对角线 ~0.01）"、"累积乘积（depth≥10）：坍缩为全 0.25 矩阵（均匀混合矩阵）"；实验对象为 Qwen3 系列。Hy4 README 的「iHC」链接确实指向此文。
- **C15 ✓**：知乎文章："当双随机矩阵满足一致正性条件（即所有元素有正的下界 δ > 0）时，其 Dobrushin 遍历系数 τ(P) ≤ 1 - dδ < 1，连乘的遍历系数以几何速率衰减 τ(A_n) ≤ (1 - dδ)^n → 0，迫使所有行趋于一致，最终收敛到均匀矩阵 (1/d)·11^T"；"如果矩阵序列是纯置换矩阵或可约的，则不满足一致正性条件，不会坍缩——但 Sinkhorn 输出的矩阵通常是严格正的"。页面 Dobrushin 补充中的次乘性、$1-n\delta$ 界、极限均匀、置换反例均有对应原文；数学复核（TV ≤ 1−nδ 的证明）成立。
- **C16 ✓**：知乎文章："目前结论是Identity HC > mHC > mHC lite > mHC orthogonal(例如cayley正交)。实验在qwen3 1.7B和8B dense上完成，150B tokens"；另见 "qwen1.7B from scratch, 150B tokens"——从头训练条件成立。
- **C17 ✓**：知乎文章："20 步不保证收敛：我们实测行和的标准差为 0.12"；"mHC-lite也说，约 27.9% 的 Sinkhorn 输入的 relative range 1/ν ≥ 10^13，此时 20 步迭代后列和偏差可达 100%"。
- **C18 ✓**：知乎文章："流 0 永远在位置 0，流 1 永远在位置 1——流的语义在深度方向上完全一致"、"累积乘积 I^L = I，不会坍缩也不会混乱"。
- **C19 ✓**：hpc_ihc.py："The eager path issues 20 / 5 / 15 kernels for pre / post / head; each HPC op is a single kernel."；"Requires VLLM_ENABLE_HPC_OPS=1"、"Only sm100 / sm103 (compute capability 100, 103)"、"Only hc_mult == 4 and hidden_size in {4096, 6144}"；`_SUPPORTED_HIDDEN_SIZES = frozenset({4096, 6144})`、`_SUPPORTED_HC_MULTS = frozenset({4})`、`_SUPPORTED_CAPABILITIES = frozenset({100, 103})`。
- **C20 ✓**：hpc_ihc.py NOTE："The reference implementation also offers a cross-layer post+pre fusion (``HpcIHCPostPre``) that folds one segment's post into the next segment's pre. It requires reworking the decoder-layer forward scheduling and is not ported here. TODO: add it once the decoder dataflow is restructured for it."
- **C21 ✓（部分）**：transformers main modeling_glm5_next.py（L219–302 区域实测确认）：`pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps`；`post = 2 * torch.sigmoid(post_w * post_scale + post_b)`（无 ε）；comb 为 `torch.softmax(...)` 后交替行列归一化 `hc_sinkhorn_iters` 步；decoder layer `self.attn_hc = Glm5NextTextHyperConnection(config)`、`self.ffn_hc = ...`；head 类 docstring："Final GLM-5.3-Flash HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."（forward 为 `hidden_streams.mean(dim=2)`）。configuration_glm5_next.py 默认值：`hc_mult: int = 4`、`hc_sinkhorn_iters: int = 20`、`hidden_size: int = 4096`、`num_hidden_layers: int = 45` → fn 形状 (2+4)×4=24 行 × 4×4096=16384 列，与「fn [24, 16384]」一致；参数量本机复算 45×2×(24×16384+24+3)=35,391,870 吻合。唯「321.32B」总参数无法从该定位核出（见问题 3）。mHC 论文 Eq.(8)（H^post = 2σ(·)、Sinkhorn t_max=20）与 GLM 实现互证一致。

### 公式与来源（F）

- **F1 ✓**：hc.py HYV4HCPreLayer.forward 逐行比对：`x_flat = x.flatten(1).float()`；`rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.layernorm_epsilon)`；`mixes = self.hc_fn(x_flat)[0] * rsqrt`；`pre = sigmoid(pre_raw * hc_scale[0] + hc_base[:hc]) + hc_eps`；`post = magnitude * sigmoid(post_raw * hc_scale[1] + hc_base[hc:2hc]) + hc_eps`；`y = sum(pre.unsqueeze(-1) * x, dim=1)`——页面 §3.1 四式与其一一对应（含 $W\in\mathbb{R}^{2n\times nd}$、RMS 因子乘在投影后、ε 加法位置、幅度只乘写门）。
- **F2 ✓**：HYV4HCPostLayer docstring 原文："y[n, i, d] = post[n, i] * x[n, d] + residual[n, i, d]"——与页面 $\hat{x}_i = H_{post,i}\cdot z + x_i$ 一致；forward 实现（post.unsqueeze(-1)*x.unsqueeze(-2)+residual）一致。
- **F3 ✓**：HYV4HCHeadLayer.forward：投影输出 hc 个 logits、`sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps`、无写门输出、加权和合并——「与 pre 同构、纯合并」成立。
- **F4 ✓**：reset_parameters（Pre 与 Head 两处），取值同 C8。
- **F5 ✓**：本页推导，数学正确：$\sigma(-\ln 3)=1/(1+e^{\ln 3})=1/4$、$2\sigma(0)=1$；并由代码块 1 实际输出验证（读门 0.250001、写门 1.000001）。
- **F6 ✓**：HC 论文 arXiv:2409.19606v3 Eq.(2) 原文（LaTeX）：`\mathbf{\hat{H}}=\mathbf{B}^{\intercal}\mathcal{T}(\mathbf{H}^{\intercal}\mathbf{A_{m}})^{\intercal}+\mathbf{A_{r}}^{\intercal}\mathbf{H}`——与页面 §1.1 公式逐符号一致；$\mathbf{A_r}=\mathbf{I}$ 时 $\mathbf{A_r}^{\intercal}\mathbf{H}=\mathbf{H}$ 特化正确。页面「实现无 $\mathbf{A_r}$ 项、不声称逐字复现」的限定与事实相符（HC 论文用 $\mathbf{A_m}/\mathbf{B}/\mathbf{A_r}$ 记号，mHC 论文用 $H^{pre}/H^{post}/H^{res}$）。

### 外部数字与实验条件（N）

- **N1 ✓**：config.json（同 C10，8 个字段逐一核对）。
- **N2 ✓**：复算 156×196,618+98,309=30,770,717；30,770,717/770×10⁹=0.003996%≈0.004%；「量级不到主干的万分之一」成立。
- **N3 ✓**：shard 头实测：`model.layers.0.hc_attn_layer.hc_pre.hc_fn [8, 24576] F32`、`hc_pre.hc_scale [2] F32`、`hc_pre.hc_base [8] F32`、`model.hc_head.hc_head_fn [4, 24576] F32`、`hc_head_scale [1] F32`、`hc_head_base [4] F32`。
- **N4 ✓**：知乎文章（同 C14）。
- **N5 ✓**：知乎文章（同 C17 行和 std 0.12）。
- **N6 ✓**：hpc_ihc.py docstring "20 / 5 / 15 kernels"。
- **N7 ✓**：README 规格表：Total Parameters 770B、Activated 49B、Layers 78、Context Length 1M、Attention Heads 64、FFN Intermediate Size 18432、MoE Intermediate Size 2048、Routed Experts 256、Shared Experts 1、Activated Routed Experts per Token 8、Vocabulary Size 120832——全部一致。

### 其他外部事实核对

- 「vLLM 官方实现 PR #54160（commit b2f685834a645，腾讯工程师提交）」✓：GitHub API 确认 commit `b2f685834a6456197e7033966fdef52a23f1abcd` 为 PR #54160 的 squash 合并提交（"[Hy4] support Hy4-preview model (#54160)"，2026-08-29，47 files，sign-off 含 chengvjiang@tencent.com 与 russellfeng@tencent.com）；PR 作者 thisjiang（Cheng Jiang）。
- 「2026-08-28 发布」：GitHub 仓库 pushed_at 2026-08-28、HF lastModified 2026-08-28——日期成立，但 README 正文无日期字段（见 C1 记录，不单列问题）。
- §5.3 表中「读门 $\sigma+\varepsilon$ / 写门 $2\sigma$ / Sinkhorn 20 步 / head 无权重均值 / 每层两站点 / fn [24,16384]」：均经 transformers main 源码与配置默认值独立复核（见 C21）。

## 机械检查记录

- **引用双向差集**：正文 `<sup>` 上标集合 = {C1–C21, F1–F6, N1–N7}，来源章节条目集合同——双向无缺口（脚本 set-diff 两向均为空）。
- **validate.py**：`python3 .dojo/scripts/validate.py wiki/ihc/index.html` → "validation ok"，exit 0。
- **数学字符**：剔除代码块与 `$...$` 后，全文仅 1 处 Unicode 数学运算符「×」（C12 条目，见问题 2）；「→」仅用于流程方向描述（pre → input_layernorm → …），与 content-examples A5 中 dg-arrow 的用法同类，不计为数学记号。h1/h2/h3/summary/表格内数学符号均用 `$...$`。
- **代码块 1（迷你 iHC 前向）**：以指定解释器运行，exit 0，stdout 与页面 `<pre><code class="language-text">` 预期输出逐字符一致（仅末尾换行符差异，为 HTML 内嵌产物）。第一幕（门 0.250001/1.000001、流间最大差 0.0e+00、head 输出 [2.602972, -1.739864]、与标准 Pre-Norm 残差差 1.21e-05）与第二幕（极差 [0.2, 0.1]→[1.4684, 0.2144]）全部复算吻合，页面正文与「展开」手算数值与代码输出一致。
- **代码块 2（双随机坍缩）**：同上，exit 0，输出逐字符一致（Sinkhorn 后行/列和回到 1、最小元素 0.1019、层 6 起 max|P−1/n| 在 4 位小数下为 0）。§4.2 补充中「δ≈0.10、界 0.6、实测约 0.2 每层」与输出相符。
- **图①②③**：图①（§1.2）dg-stack、图②（§2.1）dg-flow 均为 HTML 结构；图③（§5.1）为内联 SVG（viewBox 0 0 680 470，dg-box/dg-line/dg-arrowhead 类，主题变量取色）。SVG 内 9 处 `<foreignObject>`，8 处含 `$...$` 公式、1 处纯文字（"final RMSNorm，再进 lm_head"）；`<text>` 仅含纯文字（embedding、复制 4 份、一个站点等），无 `R_1 q` 类 ASCII 数学近似。无等宽字符框线图。无头 Chrome 渲染（--dump-dom）：全文 317 个 KaTeX span，foreignObject 内无残留未渲染 `$`，标签坐标检查无压线重叠。
- **callout 颜色**：3 黄 + 1 紫，0 蓝（蓝 ≤1 合规；红/绿/灰未用作 callout；红仅 misconceptions、绿仅 learning-goals）。
- **details summary 前缀**：全部为 解答：/补充：/展开：/代码：，无其他前缀。
- **来源说明 h3**：恰为六个固定小节（论断与来源（C）/公式与来源（F）/外部数字与实验条件（N）/构造示例/辅助解释与类比边界/简化条件及其限制），无多余小节。
- **head meta**：description 为纯文本；dojo:summary 含 `$...$`（KaTeX 允许）；dojo:type=concept；dojo:topics=模型结构（AGENTS.md 固定词表内）；dojo:tag=残差结构。
- **交叉链接**：hyper-connections、residual-connection、rmsnorm、dsa、speculative-decoding、deepseek-moe 六页 index.html 均存在；overview.html 与 index.html 互链；overview.html 内嵌链接（../../index.html、../hyper-connections/index.html、index.html）均有效。
- **问题块**：页面级核心问题 5 题、各章本章问题 2/2/3/3/3 题，每题 li 内均紧跟 `<details><summary>解答：…</summary>`，答案自含结论、推理与成立条件；核心问题答案均指明完整论证所在章节。
- **占位符残留**：无【…】、@content、TODO、待生成等残留。
- **可读性**：术语首次出现均有解释或链接（残差连接/超连接与 mHC/RMSNorm/DSA/DeepSeek MoE/投机解码均在首次依赖处链接）；公式均附符号表与用途说明；构造示例标注输入、变化项与结果含义并明确「教学构造，非 checkpoint 数值」；折叠块收起后正文结论完整；章节间有衔接（intro 路线图 + 各章问题承接）。

## 问题

- [重要·技术] §1.1 第 3 段（「无约束的 $\mathbf{A_r}$ 连乘会让信号爆炸或消失（复合映射增益峰值实测约 3000，理想值 1）」）：具体外部数字「约 3000」「理想值 1」无 C/N 上标，「来源与范围说明」的 N 小节无对应条目；「复合映射」一词全文首次且唯一出现、未定义，也未给出度量名称与实验条件（该数字的度量是 Amax Gain Magnitude——最大绝对行和/列和，实验为 27B 模型）｜引文依据：mHC 论文 arXiv:2512.24880v2 §3.1 "the Amax Gain Magnitude yields extreme values with peaks of 3000, a stark divergence from 1"（Fig. 3(b)，27B 模型；§5.4 "the maximum gain magnitude of nearly 3000 in HC"）——数字本身属实，但页面未定位｜修复要求：在「外部数字与实验条件（N）」新增条目，定位 mHC 论文 §3.1 / Fig. 3(b)，注明度量（Amax Gain Magnitude，最大绝对行和/列和）与实验规模（27B 模型），并在该括号处加对应 `<sup>[Nx]</sup>`；或删去括号内具体数字、改为定性表述（如「增益峰值与理想值相差三个数量级」，同样需挂新 N 条目）。复验以能按新条目定位到 3000/1 数值为准｜修复：§1.1 句子改写为「多层 $\mathbf{A_r}$ 乘积的最大绝对行和/列和（Amax Gain Magnitude）峰值约 3000、理想值 1（27B 模型）」并加 `<sup>[N8]</sup>`；「外部数字与实验条件」新增 N8 条目，定位 mHC 论文 §3.1 Fig. 3(b) 与 §5.4，注明度量与实验规模。同步把 §5.3 表中 GLM 总参 321.32B 改为 321B 并补 [C21] 来源定位到 zai-org/GLM-5.3-Flash HF 模型卡 Model size 栏；C21 来源说明中的 321.32B 一并改为 321B｜复验：`grep -c "321.32B" wiki/ihc/index.html` = 0；正文 `<sup>` 集合与来源说明条目集合双向差集均空（36 = 36）；KaTeX 节点 319（比修复前 317 多 2：N8 与 $\times$）；`python3 .dojo/scripts/validate.py` → ok。
- [轻微·格式] 来源与范围说明 C12 条目：「checkpoint 张量形状加算术（156 × 196,618 + 98,309）」中的「×」是 Unicode 数学运算符且未包在 `$...$` 中，违反 style-guide §11（数学运算符一律 LaTeX）｜引文依据：不适用｜修复要求：改写为 `$156\times 196{,}618+98{,}309$`，使该行通过「公式定界符之外无数学字符」检查｜修复：C12 行改为「checkpoint 张量形状加算术（$156\times 196{,}618+98{,}309$）」｜复验：`grep -c "156 × 196,618" wiki/ihc/index.html` = 0；validate.py → ok。
- [轻微·技术] §5.3 对照表「残差通路参数」行与 C21 条目：「占 321.32B 的 0.011%」中的 321.32B（GLM-5.3-Flash 总参数）在 C21 给出的定位（transformers main modeling_glm5_next.py L219-302）中不存在，本页无独立可核验来源（35,391,870 已经审查者用 configuration_glm5_next.py 默认值 hidden_size=4096、num_hidden_layers=45 独立复算核实；0.011% 为 35,391,870/321.32B 的算术结果，自洽）｜引文依据：transformers configuration_glm5_next.py 仅有 `hidden_size: int = 4096`、`num_hidden_layers: int = 45` 等字段，无总参数数字；审查者允许输入中无可引出 321.32B 的来源｜修复要求：为 321.32B 补充可定位来源（如 GLM-5.3-Flash 官方模型卡/README 的总参数行，写入 C21 定位），或在 C21 条目中明确标注「321.32B 总参数与 0.011% 占比为超连接页转引数字，本页未独立复核」｜修复：§5.3 表 35,391,870 行改为「占 321B 的 0.011%」并加 `<sup>[C21]</sup>`；C21 来源说明的 321.32B 同步改为 321B；N8 条目末尾追加 GLM-5.3-Flash 总参数 ≈ 321B 的来源定位（zai-org/GLM-5.3-Flash HF 模型卡 Model size 栏，WebFetch 实测该栏「321B params」，Introduction 行写 320B）｜复验：321.32B 在页面中 0 处；引用编号双向闭环。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复

三条问题均为局部修改（补一个 N 条目与上标、改一处 LaTeX、补一处来源定位/标注），不涉及大纲与范围调整。全部 34 条 C/F/N 引用经外部来源逐条核对，未发现定位不到或内容不符的论断；两个代码块实际运行且输出与页面逐字符一致；结构、格式与交互检查全部通过。修复并复验后可进入下一轮独立审查。
