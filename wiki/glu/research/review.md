# GLU 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 强推理模型对照来源）
- 页面版本：wiki/glu/index.html（git hash: 388714c30f94b3244defed4ed24a23f1d584646a）、wiki/glu/overview.html
- 时间：2026-08-09

## 段 A 盲读（小白视角，按页面顺序）

主线可跟：钩子 callout → 动机（两层缺口）→ 公式与手算 → 梯度通路 → 家族派生 → 经验边界。逐节记录如下。

**callout / context-box**：开篇用"线性投影""sigmoid""逐元素相乘"作为钩子术语，均在 §1–§2 首次定义处内联解释（sigmoid $\sigma(z)=1/(1+e^{-z})$、$\otimes$ 逐元素乘积、$xW+b$ 线性变换），钩子→解释间距不超过两节，不构成卡点。context-box 中"SiTU-GLU 是家族下游成员"——SiTU-GLU 首次出现，页面未给任何定义、链接或占位提示，小白无法从此页知道它是什么；不阻塞 GLU 主线理解，但属于无法解释的首现术语。

**§1 why-gate**：两层缺口（缺按维度调节、缺不杀梯度的非线性）讲清。GTU 引入后写"第 3 章会展开它的梯度结构"，§3 确实展开了——指向正确。sigmoid 导数 $\sigma'(z)=\sigma(z)(1-\sigma(z))$、tanh 导数内联给出。结构图（ASCII）两分支清晰。无卡点。

**§2 formula-and-example**：公式符号逐项说明（$\mathbf{X}$、$*$ 卷积→矩阵乘退化、$\mathbf{W},\mathbf{V}$、$\mathbf{b},\mathbf{c}$、值分支/门分支、$\otimes$）。$*$ 退化标注为教学简化。手算取 $W=I$，页面已说明"取 $W=I$ 让值分支近似直通，便于把注意力集中到门的作用上"——教学目的明确。三个边界检查（门全 1/全 0/中间）清晰。极端对照在折叠块内，收起不影响主线。无卡点。

**§3 gradient-path**：简化形式 $Y=\mathbf{X}\otimes\sigma(\mathbf{X})$ 标注为教学简化。$\nabla\mathbf{X}$ 在主文公式前已定义"上游梯度——损失对 GLU 输出的梯度，由下游回传过来"——折叠块收起时可读。梯度两项职责分述、GTU 对照表、推导步骤折叠块——均不依赖折叠块即可理解主线。边界说明"GLU 不是永不消失"到位。无卡点。

**§4 family**：Bilinear（去 σ）、ReGLU/GEGLU/SwiGLU（换激活）派生规则清晰。记号差异表（Dauphin σ 在 V 分支 vs Shazeer σ 在 W 分支）明确标注 $W\leftrightarrow V$ 等价。FFN 三矩阵与 $2/3$ 缩放推导完整，参数量等式 $3d\cdot d_{ff}'=2d\cdot d_{ff}$ 逐步给出。折叠块内 $d=768$ 验算。§4 末尾提到"SiTU-GLU、SwiGLU 的概念页会沿用 Shazeer 记法"——SiTU-GLU 第二次出现，仍无解释或链接。

**§5 evidence-and-boundary**：Table 1 八行 log-perplexity 标注条件（T5 base、$d_{model}=768$、12 头、$d_{ff}=3072\to2048$）。"divine benevolence"原句引用并标注"明确标注无理论解释"。五条边界、四条误解——逐条正确。无卡点。

**§6 sources-and-teaching-notes**：C1–C9 核心论断、F1–F8 公式、N1–N2 数字、教学示例、类比边界、教学简化逐项登记。来源可定位。

**overview.html**：快速阅读版，五节（是什么/为什么/核心直觉/关键结论与边界/来源）。与 index.html 一致，无矛盾。overview 中"SiTU-GLU、SwiGLU 都是 GLU 家族下游成员"——SiTU-GLU 第三次出现，同样无解释。

**逐题核对学习目标（折叠块全收起）**：
- Q1（一句话说清 GLU + 解决什么问题）：§1 + §2 主文 ✓
- Q2（公式解释非线性 + 梯度线性通路，与 GTU 对比）：§3 主文 ✓
- Q3（手算 GLU 输出 + 翻译放行/压低）：§2 主文 ✓
- Q4（家族派生 + Dauphin/Shazeer 记号差异）：§4 主文 ✓
- Q5（FFN 缩 $2/3$ + GLU 不保证什么）：§4 FFN 推导 + §5 边界 ✓

五项学习目标均由正文章节完整回答，不依赖折叠块。

## 段 B 对照来源

**1. 定义与机制**：GLU 定义 $(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$ = Dauphin §2 Eq.(1)（搜索结果一致："GLU(x,W,V,b,c)=(xW+b)⊗σ(xV+c)"）。Bilinear 去 σ、归因 Mnih & Hinton 2007 = Dauphin §5.3（搜索结果一致）。GTU $=\tanh(\mathbf{X})\otimes\sigma(\mathbf{X})$ = Dauphin §3（搜索结果一致："gated tanh unit (GTU) of van den Oord et al. (2016)"——页面只引 Dauphin §3 Eq.(2) 的梯度分析，未声称 Dauphin 发明 GTU，无误）。ReGLU/GEGLU/SwiGLU = Shazeer §2 Eq.(5)（结构一致）。FFN_GLU 三矩阵 = Shazeer §2 Eq.(6)（一致）。记号差异 Dauphin σ 在 V 分支 / Shazeer σ 在 W 分支 = 两篇 Eq.(1) vs Eq.(4) 直接比对（一致，$\otimes$ 可交换故等价）。未发现扩大来源结论。

**2. 公式与推导**：
- GLU 梯度 $\nabla[\mathbf{X}\otimes\sigma(\mathbf{X})]=\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\mathbf{X}\otimes\sigma'(\mathbf{X})\nabla\mathbf{X}$：用乘积求导复算 $Y_i=X_i\sigma(X_i)\Rightarrow\partial Y_i/\partial X_i=\sigma(X_i)+X_i\sigma'(X_i)$，乘以上游梯度得回传两项，与 Dauphin Eq.(3) 一致。✓
- GTU 梯度两项分别带 $\tanh'$ 和 $\sigma'$：复算 $Y_i=\tanh(X_i)\sigma(X_i)\Rightarrow\partial Y_i/\partial X_i=\tanh'(X_i)\sigma(X_i)+\sigma'(X_i)\tanh(X_i)$，与 Dauphin Eq.(2) 一致。✓
- 手算 $\sigma(1.0)\approx0.7311$、$\sigma(-0.5)\approx0.3775$、GLU$=[0.7311,0.1888]$：Python 复算一致。✓
- 极端对照 $\sigma(10)\approx0.99995$、$\sigma(-5)\approx0.00669$：Python 复算一致。✓
- 参数量 $2\times768\times3072=4{,}718{,}592=3\times768\times2048$：Python 复算一致。✓
- $d_{ff}'=\tfrac23 d_{ff}$ 推导：$3d\cdot d_{ff}'=2d\cdot d_{ff}\Rightarrow d_{ff}'=\tfrac23 d_{ff}$，正确。✓
- 符号首次出现处均有定义（$\nabla\mathbf{X}$ 在主文已定义）。

**3. 可运行代码**：本页无可运行代码块（仅 ASCII 结构图），无代码需执行。

**4. 事实与推断**：Table 1 八行数字与搜索结果中"all GLU variants outperform ReLU/GELU baselines""GEGLU and SwiGLU produce the best perplexities"一致（页面 GEGLU 1.633 最优、SwiGLU 1.636 次优；所有变体 < 所有基线）。"divine benevolence"原句 = Shazeer §4（搜索结果一致）。**发现一处事实越界**：§4 FFN 末段"这是现代用 SwiGLU 的大模型（如 LLaMA）内部维度取 $\tfrac{8}{3}d$ 的来源"——此论断不在 Dauphin 2017 或 Shazeer 2020 中（两篇均未提及 LLaMA），属对下游模型设计实践的推断；且 LLaMA 7B 实际 $d_{ff}/d=11008/4096\approx2.6875\neq\tfrac83\approx2.6667$（经 round 到 256 倍数），页面说"取 $\tfrac{8}{3}d$"是近似值当精确值表述。"来源与教学说明"未将此条登记为教学示例或推断。

**5. 前置知识引用**：sigmoid、$\otimes$、$xW+b$、链式法则按基础记号内联定义（与 content-examples 对 sigmoid 的处理一致）。GELU/ReLU/Swish 作为激活名出现、Swish$_\beta(z)=z\cdot\sigma(\beta z)$ 内联给出——不展开已标注简化。SiTU-GLU 被提及 3 次但无链接或占位提示。页面提到"它们各自的完整讲解属于独立概念页""SiTU-GLU、SwiGLU 的概念页"——这些概念页引用均为纯文本、无 `<a href>` 链接或占位标记。（本次审查禁止读取其他页面，无法确认这些概念页是否存在。）

**6. 教学简化**：$*$ 退化为矩阵乘、梯度分析用共享输入简化形式、Swish/GELU/ReLU 不展开、基础记号内联——四项均在"教学简化及其限制"逐条写明可推出/不可推出边界。类比（"门""乘性跳连""门控选择 vs 输入幅值副作用"）均标失效边界。未发现简化导致核心结论失真。

**7. 页面功能**：KaTeX 定界符配置（`$$` display / `$` inline）与 auto-render 匹配；3 处 details 折叠块结构正确；h2/h3 均有 id（scroll-margin-top 避开固定导航）；TOC 由脚本生成。index.html ↔ overview.html 互相链接、均链回 ../../index.html。机械项（validate.py）未在本次审查中执行（任务范围仅段 A + 段 B）。

## 问题

- [轻微·盲读] context-box（"家族角色"行）+ §4 notation-diff 末段 + overview.html "关键结论与边界"：SiTU-GLU 共 3 处出现，页面未给出任何定义、链接或占位提示，小白读者无法从此页得知它是什么。修法：首次出现处加一句括注（如"SiTU-GLU（另一种 GLU 变体，见其概念页）"）或添加概念页链接/占位标记。｜ 修复： ｜ 复验：
- [轻微·来源] §4 ffn 小节末段"这是现代用 SwiGLU 的大模型（如 LLaMA）内部维度取 $\tfrac{8}{3}d$ 的来源"：此论断不在 Dauphin 2017 或 Shazeer 2020 中，属对 LLaMA 设计实践的推断；且 LLaMA 7B 实际 $d_{ff}/d\approx2.6875\neq\tfrac83\approx2.6667$（round 到 256 倍数），"取 $\tfrac{8}{3}d$"是近似值当精确值表述。修法：改为"约为 $\tfrac{8}{3}d$（实际按 256 倍数取整）"并在"来源与教学说明"中将其登记为推断而非来源结论，或直接删去具体型号只留 $d_{ff}'=\tfrac83 d$ 的数学推导。｜ 修复： ｜ 复验：
- [轻微·前置知识] §4 derivatives 末段"它们各自的完整讲解属于独立概念页" + §4 notation-diff 末段"SiTU-GLU、SwiGLU 的概念页会沿用 Shazeer 记法"：概念页引用为纯文本，无 `<a href>` 链接或占位标记。修法：若概念页已存在则加链接，若不存在则加占位提示（如"〔待建概念页〕"）。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：进入修复（3 条轻微问题均可逐条修复，不改变范围或大纲）
- 独立性说明：本次审查由未参与初稿生成的独立上下文执行，满足 check.md §1 的独立审查硬性要求。段 A 盲读 + 段 B 对照来源均已完成。未执行 validate.py 及 §4 末段自动发布流程（任务范围仅审查+产物）。
