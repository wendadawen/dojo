# SiTU-GLU 独立审查

- 审查者：独立上下文（AI 模拟，强推理模型 GLM-5.2，未参与页面生成）
- 页面版本：index.html 工作树哈希 `fa33e5f1c7d6745524054c36dd6abd4b507c1824`（仓库 HEAD `d4f9e4e`）
- 时间：2026-08-09 14:30 CST

## 审查范围与方法

- 审查对象：`wiki/situ-glu/index.html`、`wiki/situ-glu/overview.html`
- 来源：`/tmp/kimi-k3-research/k3-report.txt`（§2.3.2 lines 487-545；§B 附录 lines 2757-2781；Fig.4 lines 438-458；§2.3 开头 lines 425-432；§4 对比表 line 756）；HuggingFace `config.json`（`activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`、`hidden_act="situ"`）
- 方法：段 A 盲读（小白视角，不参考来源）+ 段 B 对照来源逐条核对
- 核对重点：softcap(x,β)=β·tanh(x/β) 公式、输出上界 β1·β2=100、与 SwiGLU/GLU 对比、§B 附录推导
- 未执行：`validate.py`（本次为审查未改文档）；页面无可运行代码块需重跑；KaTeX 渲染与折叠交互为静态检查（未在浏览器实际打开），机械项以 `validate.py` 为准

## 段 A 盲读记录（小白视角）

按页面顺序阅读，主线理解卡点：

- 主线清晰：callout 抛出"两无界因子乘积爆炸"问题 → §why-bound 解释 SwiGLU 无界性 → §formula-and-example 给公式与手算 → §local-and-bound 证近原点等价与上界 → §softcap-vs-clip 对比梯度 → §usage-and-boundary 定位与边界。
- softcap、tanh、σ、⊙、O(·)、Swish、GLU、SwiGLU 等术语首现处均有内联定义。
- 折叠块（x=2/50 手算、Taylor 推导、上界拆解、梯度渐近）均为补充，正文主线不依赖折叠块即可成立。
- §why-bound 对比表（GLU/SwiGLU/SiTU-GLU）出现卡点：GLU 行"门支线性因子"列写"x（无界）"，但同段正文（line 709）写"GLU 的 sigmoid 门虽然有界"——表格暗示 GLU 门支有无界线性因子，正文说 GLU 门有界，二者冲突，小白无法判断 GLU 门到底有界还是无界（详见问题 1）。
- W_↓/W_↑（line 707）仅一句话提及，已标注"属 Stable LatentMoE 概念页"，可接受。

学习目标核对（Q1-Q5）：Q1 由 §why-bound 定位段回答；Q2 由 §local-and-bound 回答；Q3 由 §formula-and-example 手算表回答；Q4 由 §softcap-vs-clip 回答；Q5 由 §usage-and-boundary 回答。五个学习目标均由正文章节完整回答，不依赖折叠块。

## 段 B 对照来源核对

### 核对通过项

- softcap(x,β)=β·tanh(x/β)：index.html:713 与来源 k3-report.txt:498 一致。
- SiTU-GLU 定义 Eq.(12) `[β1·tanh(Wg x/β1)⊙σ(Wg x)]⊙[β2·tanh(Wu x/β2)]`：index.html:744 与来源 k3-report.txt:500-502 一致。
- β1=4（gate）、β2=25（up）：与来源 k3-report.txt:541 一致；config.json `activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`、`hidden_act="situ"` 佐证。
- 输出上界 |SiTU-GLU|≤β1·β2=100：index.html:835 与来源 §B Eq.(19) k3-report.txt:2778 一致；逐项证明（|tanh|<1、0<σ<1）复算正确。
- 局部展开一阶等价 SwiGLU：与来源 §B Eq.(18) k3-report.txt:2768-2772 一致；折叠块 Taylor 推导（tanh u = u - u³/3 + O(u⁵)）复算正确，`-z³/(3β²)` 项正确。
- 极限 β1,β2→∞ 逐点收敛到 SwiGLU：index.html:859 与来源 k3-report.txt:2773-2774 一致。
- softcap 套门支线性因子、保留 σ、套值支："to the up branch ... preventing either branch from dominating the product" 引文与来源 k3-report.txt:498, 2763-2766 verbatim 一致。
- softcap 导数 1-tanh²(x/β)、指数衰减非零：复算正确（d/dx[β·tanh(x/β)]=tanh'(x/β)=1-tanh²(x/β)）。
- §B 末段引文 "Unlike hard clamping...preserves nonzero gradients away from saturation boundaries...better training behavior"：index.html:896 与来源 k3-report.txt:2780-2781 verbatim 一致。
- "better training behavior" 标注为经验陈述非对照实验：index.html:911 判断正确，来源确无 SiTU-GLU vs SwiGLU ablation。
- β1=4,β2=25 标注为工程设定非普适最优：index.html:861 判断正确，来源 k3-report.txt:541 仅给值无依据。
- Stable LatentMoE 三件套（RMSNorm §2.3.1 / SiTU-GLU §2.3.2 / QB §2.3.3）：index.html:931-940 与来源 k3-report.txt:461-463 一致。
- K2→K3 激活函数 SwiGLU→SiTU-GLU：index.html:926 与来源 §4 对比表 k3-report.txt:756 一致。
- 手算数值全部复算通过：x=0（g=0,u=0,y=0）、x=10（g≈3.9463,u≈9.4987,y≈37.485,SwiGLU≈99.995）、x=100（g≈4,u≈24.983,y≈99.933,SwiGLU=10000）、x=2（g≈1.6281,u≈1.9957,y≈3.2493）、x=50（g≈4,u≈24.101,y≈96.403）、z=0.5 验证（4·tanh(0.125)≈0.49741，理论 -z³/(3β²)≈-0.00260）、梯度数值（β=4: 4e⁻⁵⁰≈7.7e-22；β=25: 4e⁻⁸≈1.3e-3）。
- 互链与前置链接：index.html↔overview.html 互相链接正确；GLU 概念页 `wiki/glu/index.html` 与 `evidence.md` 均存在。

## 问题

- [重要·技术] index.html:725（§why-bound 对比表 GLU 行）：表格将 GLU 的"门支线性因子"列为"$x$（无界）"，但来源 Fig.4（k3-report.txt:441）显示 GLU 的 Gate branch = σ(x)（无线性因子），且来源 §2.3.2（k3-report.txt:493）明确"The sigmoid gate of the original GLU avoids unbounded gate growth"——即 GLU 门支有界。此条目同时与页面自身正文 index.html:709"GLU 的 sigmoid 门虽然有界"自相矛盾，造成小白读者无法判断 GLU 门是否有界。：将对比表 GLU 行的"门支线性因子"改为"—"或"1（常数）"，"乘积上界"列保留"无界（值支无界）"，使表格与来源 Fig.4（GLU gate=σ(x)）及正文 line 709 一致。 ｜ 修复：已将 index.html:725 GLU 行的"门支线性因子"由"$x$（无界）"改为"—"（GLU 门支无线性因子，门支即 σ(x) 有界），与来源 Fig.4 及正文 line 709"GLU 的 sigmoid 门虽然有界"一致；"门支激活""值支""乘积上界"列不变。表格下方 line 732"GLU 和 SwiGLU 的乘积都至少有一个无界因子"仍成立（GLU 值支无界），无需改动。 ｜ 复验：

- [轻微·技术] index.html:707、overview.html:52（§why-bound / overview"为什么需要它"）：页面写"四连矩阵相乘"，来源 k3-report.txt:430 原文为"a chain of nearly four consecutive matrix multiplications"（近四连），页面漏掉"nearly/近"。：改为"近四连矩阵相乘"或"将近四次连续矩阵相乘"。 ｜ 修复： ｜ 复验：

- [轻微·技术] index.html:1010（§sources 教学示例汇总）：教学注记写"导数 ≈ 1.4×10⁻²¹（深饱和）"，但其引用的折叠块 index.html:905 写"导数 ≈ 4e⁻⁵⁰ ≈ 7.7×10⁻²²"。复算 4e⁻⁵⁰=4×1.9287e-22=7.71e-22，折叠块数值 7.7×10⁻²² 正确，教学注记 1.4×10⁻²¹ 与之不符（差约 1.8 倍）。：将 index.html:1010 的"1.4×10⁻²¹"改为"7.7×10⁻²²"，与折叠块 index.html:905 一致。 ｜ 修复： ｜ 复验：

- [轻微·技术] index.html:654、707；overview.html:52：callout 与正文将"pre-activation 可以大到 100、1000，乘积到 10000、1000000"以事实口吻陈述，但来源 k3-report.txt:431-432 仅说"exploding internal activations"未给具体量级。这些数字为页面教学构造。：在 callout/正文首次出现处加"例如"或括注"（教学示意量级）"，与 index.html:1007 已标注的"数字为教学构造"口径统一。 ｜ 修复： ｜ 复验：

- [轻微·技术] index.html:707（§why-bound）：页面写"低精度算术（FP8/INT8）下直接溢出"，来源 k3-report.txt:492 仅说"low-precision arithmetic"未指定 FP8/INT8；且 K3 config.json 量化配置为 mxfp4（4-bit），非 FP8/INT8。：将"（FP8/INT8）"删除改为"低精度算术"，或改为"（如 FP8/INT8 等；K3 实际用 mxfp4）"并标注为一般性举例。 ｜ 修复： ｜ 复验：

- [轻微·技术] index.html:811、overview.html:62（§local-and-bound / overview"核心直觉"）：主文 Eq.(18) 写 β·tanh(z/β)=z+O((z/β)³)，来源 §B Eq.(18)（k3-report.txt:2769-2772 文本渲染）为 O(z³/β²)。O((z/β)³)=O(z³/β³) 与 O(z³/β²) 差一个 β 因子；页面折叠块 index.html:828 已说明"O((z/β)³) 量纲上对应 β·(z/β)³=z³/β²"作量纲弥合，但主文公式形式与来源不一致。：将主文 Eq.(18) 的 O((z/β)³) 改为来源原文 O(z³/β²)，或在主文公式旁加注"（量纲说明见折叠块）"使主文与来源形式一致。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：进入修复。1 个重要问题（GLU 对比表门支误述，与来源 Fig.4 及 §2.3.2 直接冲突且与正文 line 709 自相矛盾）需修复后方可发布；5 个轻微问题建议一并修复。
- 核心结论状态：SiTU-GLU 的定义（Eq.12）、softcap 公式 β·tanh(x/β)、输出上界 β1·β2=100、局部展开一阶等价、β→∞ 收敛、softcap vs clip 梯度对比、config.json β 值——均与来源一致，核心结论未受影响。重要问题集中在 GLU 对比变体的表述，非 SiTU-GLU 自身机制。
- 未独立验证项：`validate.py` 未运行（本次为审查未改文档）；页面无可运行代码块需重跑；KaTeX 公式渲染与折叠交互为静态检查（未在浏览器实际打开），机械项以 `validate.py` 为准。
