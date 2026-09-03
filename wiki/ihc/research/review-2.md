# iHC（恒等超连接）审查记录（第 2 轮）

- 页面版本：index.html `25dac9c9faf197de6075f1c42e86f14d241199be`（overview.html `d7ee0bf21697b6c33732c78b2ac1ac544500277b`）
- 审查时间：2026-09-03
- 审查者：独立审查者（第 2 轮，未参与写作、前序审查与规划）
- 已完整阅读章节：引言（核心问题与三个子问题及解答）、1 恒等超连接是什么（1.1 从 HC 到 iHC、1.2 为什么恒等是合理默认）、2 不混合为什么不掉精度（2.1 单层门控的表达力、2.2 双随机矩阵的坍缩问题）、3 一个最小 iHC 模型（3.1 公式与符号、3.2 第一幕：初始化即标准残差、3.3 第二幕：分化门的作用）、4 拆开一个 iHC 站点（4.1 三块结构、4.2 初始化、4.3 为什么这样设计）、5 Hy4 里的 iHC（5.1 配置、5.2 checkpoint 实测、5.3 与 GLM-5.3-Flash 的 mHC 对比及本章问题解答）、来源与范围说明全部五个小节（论断与来源、公式与来源、外部数字与实验条件、构造示例、辅助解释与类比边界、简化条件及其限制）；overview.html 全文。

## 前轮修复复核

本轮独立打开页面与外部来源逐项复核，结果如下：

1. §1.1 Amax Gain Magnitude：**已到位**。index.html:829 现为「多层 $\mathbf{A_r}$ 乘积的最大绝对行和/列和（Amax Gain Magnitude）峰值约 3000、理想值 1（27B 模型）<sup>[N8]</sup>」，内联定义与标注均在。核对 mHC 论文 arXiv:2512.24880v2 §3.1（Numerical Instability）：「We refer to these metrics as the Amax Gain Magnitude of the composite mapping. As shown in Fig. 3 (b), the Amax Gain Magnitude yields extreme values with peaks of 3000, a stark divergence from 1」；§5.4（Stability Analysis）：「maximum gain magnitude of nearly 3000 in HC」；实验规模 27B。N8 的来源描述（index.html:1614）与论文位置一一对应。
2. §5.3 表格 321B：**数值已改、来源未补全**。index.html:1536 现为「35,391,870（占 321B 的 0.011%）」；zai-org/GLM-5.3-Flash Hugging Face 模型卡确认「Model size: 321B params」；复算 transformers 默认配置 45 层 × 2 站点 × 393,243（24×16384+24+3）= 35,391,870，与页面一致。但 C21 的来源描述（index.html:1592）仍只写「转引自超连接页 research/evidence.md（transformers main modeling_glm5_next.py L219-302）」，321B 与层数默认值在该标注位置定位不到——见问题 1。
3. C12 乘号：**已到位**。index.html:1583 现为「（$156\times 196{,}618+98{,}309$）」，乘号在数学定界符内，KaTeX 可渲染。

## 机械检查记录

- `.dojo/scripts/validate.py`：index.html 与 overview.html 均返回 validation ok。
- 引用双向性：正文 `<sup>` 引用集合与来源节定义集合（C1–C21、F1–F6、N1–N8，共 35 条）做集合差，两侧均为空，无孤儿引用、无未引用来源。
- 代码执行：提取正文 2 个 `<pre><code class="language-python">` 块，用 `/Users/wendadawen/.workbuddy/binaries/python/envs/default/bin/python` 执行，均退出码 0，stdout 与各自随后的 `language-text` 预期块逐行一致（46 行与 22 行）。
- 前置概念链接：residual-connection、hyper-connections、rmsnorm、dsa、deepseek-moe、speculative-decoding 六个页面均存在，锚点有效。
- overview.html 与 index.html 双向链接一致，内容与 index.html 结论无冲突。

## 来源核对记录（引文依据摘要）

按 check.md §2.2 对全部 35 条来源逐条打开实际来源核对，关键片段如下（页面标注位置均定位到并确认支持页面表述）：

- C1/C10/N1/N7（Hy4 README + config.json）：README「The residual pathway uses iHC (identity Hyper-Connections) to expand inter-layer information flow」「Residual Streams: 4」「770B」「49B activated」「1M context」；config.json `enable_ihc`、`hc_mult: 4`、`hc_magnitude: 2.0`、`hc_eps: 1e-06`、`hidden_size: 6144`、`num_hidden_layers: 78`、`rms_norm_eps: 1e-05`、`mlp_layer_types`（第 1 层 dense）均确认。
- C2–C9、F1–F4（vLLM PR #54160 diff，commit b2f685834a645，与 GitHub `gh api` 取得的 merge_commit_sha `b2f685834a6456197e7033966fdef52a23f1abcd` 一致）：hc.py 模块 docstring 三块职责、`y[n,i,d] = post[n,i] * x[n,d] + residual[n,i,d]`、reset_parameters（scale 0.01、base 前 n 个 $-\log(n-1)$ 后 n 个 0）、forward 内 `.float()`；hpc_ihc.py「Unlike mHC there is no comb matrix...one fused multiply-add per element」、kernel 数 20/5/15、`_SUPPORTED_*` 常量（sm100/sm103、hc_mult=4、hidden ∈ {4096, 6144}）、HpcIHCPostPre NOTE；model.py `hc_attn_layer`/`hc_mlp_layer` 两站点、`_forward_ihc`「Under iHC the residual is carried inside hidden_states」均确认。提交签署人为 chengvjiang@tencent.com，支持「腾讯工程师提交」表述。
- C3/C4/C19/C20/N6（hpc_ihc.py）：见上，命名分歧（README identity vs docstring independent 两处）确认。
- C11–C13/N2/N3（index.json + 4 个 shard 头实测）：156 站点 × 3 张量 + 3 head 张量 = 471 个 hc 张量，全部 F32；hc_fn [8,24576]、hc_scale [2]、hc_base [8]、hc_head_fn [4,24576]、hc_head_scale [1]、hc_head_base [4]；196,618/98,309/30,770,717 复算一致；model.mtp_layers.* 27 张量中无 hc 张量。
- C14–C18/N4/N5（知乎文章《你的deepseek mHC可能不需要"m"》）：对角约 0.96/非对角约 0.01、深度 ≥10 累积乘积全 0.25、Dobrushin 遍历系数 τ ≤ 1−nδ、纯置换或可约序列不坍缩、Identity HC > mHC > mHC lite > mHC orthogonal（Qwen3 1.7B/8B dense、150B tokens 从头训练）、Sinkhorn 20 步行和 std 0.12、27.9% 输入 relative range ≥10¹³ 时列和偏差 100%、流语义一致 I^L=I 均在文中定位。
- F5（GLM 末端无权重均值推导）：与 modeling_glm5_next.py 实现一致的推导，标注为构造。
- F6（HC 论文 Eq (2)）：arXiv:2409.19606v3 公式 (2) $\hat{\mathbf{H}}=\mathbf{B}^{\intercal}\mathcal{T}((\mathbf{H}^{\intercal}\mathbf{A_{m}})^{\intercal})+\mathbf{A_{r}}^{\intercal}\mathbf{H}$ 逐字一致。
- C21/N8（GLM-5.3-Flash）：modeling_glm5_next.py `hc_mult=4`、`attn_hc`/`ffn_hc`、fn [24,16384]、pre=σ+ε、post=2σ、comb=softmax+Sinkhorn 20 步、head 无权重均值均确认；configuration_glm5_next.py 默认 `num_hidden_layers=45`、`hc_sinkhorn_iters=20`；zai-org 模型卡「Model size: 321B params」。见问题 1。

## 问题

- [重要·技术] §5.3 对比表（index.html:1536）与 C21 来源描述（index.html:1592）：表格行「35,391,870（占 321B 的 0.011%）」中，总参数 321B 无法在 C21 标注的来源位置定位。C21 写「转引自超连接页 research/evidence.md（transformers main modeling_glm5_next.py L219-302）」，该代码段是 mHC 模块实现，含 hc_mult、fn 形状与门形式，但不含 321B 总参数；层数 45（复算 35,391,870 所需）来自 configuration_glm5_next.py 默认值，也不在标注范围内。321B 实际由 zai-org 模型卡支撑（N8 已含），但 §5.3 正文只引 C21 未引 N8。核对依据见「前轮修复复核」第 2 条。｜引文依据：zai-org/GLM-5.3-Flash HF 模型卡「Model size: 321B params」；configuration_glm5_next.py 默认 `num_hidden_layers: 45`、`hc_mult: 4`；复算 90×(24×16384+24+3)=35,391,870；modeling_glm5_next.py L219-302 内无 321B。｜修复要求：在 C21 来源描述中补入 zai-org/GLM-5.3-Flash Hugging Face 模型卡（321B 来源）与 configuration_glm5_next.py（层数等默认值来源），或将 §5.3 表格行/导语处 321B 与 0.011% 一并加注 <sup>[N8]</sup>。｜修复：N8 拆为 N8（mHC Amax 3000）+ N9（GLM-5.3-Flash 321B 总参数 + 45 层等默认值来自 configuration_glm5_next.py）；C21 来源描述补全为三段（modeling_glm5_next.py L219-302 + configuration_glm5_next.py 默认值 + zai-org HF 模型卡 Model size 栏），并在来源说明首句明示「事实与汇总经超连接页核对一致」；§5.3 表 cell 加 <sup>[C21, N9]</sup>（35,391,870 与 321B 分别由 C21 与 N9 支撑）。｜复验：cited-only/listed-only 均空（37 = 37），§5.3 表 cell 与 C21 描述互锁，validate.py ok。
- [轻微·格式] 全页（index.html:797、952、956、958、983、1031、1037、1042、1227、1234、1241、1532–1533、1535、1541、1549、1563、1592 对比 index.html:998、1006–1007、1479）：同一个量（hc_eps=1e-6，加在门上的小量）存在两种写法——正文、对比表与解答块用 $\varepsilon$，§3.1 公式与符号表、§5.1 配置表用 $\epsilon_{hc}$；且 index.html:1549 中 RMS 的 $\epsilon=10^{-5}$ 又用裸 $\epsilon$。违反 style-guide「同一变量全页写法一致」。｜引文依据：不适用。｜修复要求：统一为一种写法（建议全部用 $\epsilon_{hc}$ 指门上的小量、$\epsilon_{rms}$ 指 RMS 下限），替换上列全部 $\varepsilon$ 出现处。｜修复：正则将所有裸 \varepsilon 替换为 \epsilon_{hc}（46 处替换，覆盖 §3.1/3.2/3.3/2.2/5.1/概述等所有正文与解答块）；\epsilon_{hc} 出现 46 次、\epsilon_{rms} 出现 3 次、裸 \varepsilon 出现 0 次。｜复验：`grep -c "\\varepsilon" wiki/ihc/index.html` = 0；validate.py ok。
- [轻微·技术] C1 来源描述（index.html:1572）：「官方 README（GitHub main，2026-08-28）」中的日期在 README 内定位不到——README 无任何日期字段。核对 GitHub 仓库元数据：created 2026-08-27T09:27:34Z、pushed 2026-08-28T12:48:34Z；HF 模型卡 createdAt 2026-08-27、lastModified 2026-08-28。日期合理但取自仓库元数据而非标注的来源文档。｜引文依据：GitHub API `created_at: 2026-08-27T09:27:34Z` / `pushed_at: 2026-08-28T12:48:34Z`；HF API `createdAt: 2026-08-27` / `lastModified: 2026-08-28`；README 全文无日期。｜修复要求：把日期改为可定位表述（如「GitHub main，最后推送 2026-08-28」），或在 C1 中注明日期取自仓库/模型卡元数据。｜修复：C1 描述改为「Tencent-Hunyuan/Hy4-preview 官方 README（GitHub main，仓库 pushed_at 2026-08-28、HF 模型卡 lastModified 2026-08-28，README 正文无日期字段）Model Introduction 与 Model Specifications」｜复验：grep -c "2026-08-28" 出现位置现在均带元数据来源说明。
- [轻微·格式] N8 来源描述（index.html:1614）：单条内捆绑两条不相关事实（mHC 论文的 Amax Gain Magnitude 峰值 3000 与 zai-org 模型卡的 GLM-5.3-Flash 总参数 321B），来源、主题均不同，降低可维护性。｜引文依据：不适用。｜修复要求：拆为两条（如 N8 保留 mHC 论文增益，新增 N9 记 zai-org 模型卡 321B），或在本条内明确分段说明两个子来源。｜修复：N8 拆为两条独立条目——N8 仅保留 mHC 论文增益（Amax Gain Magnitude 峰值约 3000），N9 记 GLM-5.3-Flash 总参数约 321B（zai-org HF 模型卡 Model size 栏）+ 层数等默认值（configuration_glm5_next.py 默认 num_hidden_layers=45、hc_mult=4、hc_sinkhorn_iters=20）。｜复验：cited-only/listed-only 均空（N8 与 N9 分别被 §1.1 与 §5.3 表引用）。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复

三项前轮修复中两项确认到位（§1.1 Amax 定义与 N8 映射、C12 乘号），§5.3 表格数值正确但 C21 来源定位不完整需补全。机械检查全部通过，全页 35 条来源逐条核对未发现其他定位失败或内容不符。
