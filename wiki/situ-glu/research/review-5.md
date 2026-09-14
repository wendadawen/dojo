<!-- review-meta
round: 5
page: wiki/situ-glu/index.html
reviewed_content_sha256: 149149fe7a865d73
-->
# SiTU-GLU 审查记录（第 5 轮）

- 页面版本：149149fe7a865d73
- 审查时间：2026-09-13 20:25
- 审查者：独立子代理
- 已完整阅读章节：核心问题、最容易误解、1. 为什么需要给 SwiGLU 加上界——激活爆炸从哪里来、2. SiTU-GLU 的公式与手算——定义、符号与三个边界点、3. 近原点像 SwiGLU、远点饱和——两个性质怎么同时成立、4. 为什么是 softcap 而不是 clip——饱和区里梯度差别、5. 在 K3 中的使用位置与不解决——两处使用与四条边界、来源与范围说明

来源核对摘要（本轮全部回源核对）：K3 报告取 arXiv:2607.24653（v2 HTML 全文）——§2.3.2 Eq.(12) 定义与 F1 逐字符一致；softcap 定义式亦即 F2；"For Kimi K3, we set the soft-cap hyperparameters to β1=4 for the gate branch and β2=25 for the up branch"（C4/N1）；"both multiplicative factors in SwiGLU are unbounded, so coincident large coordinates can produce activation outliers and increase overflow risk in low-precision arithmetic"（C3/C9）；"The sigmoid gate of the original GLU avoids unbounded gate growth, but it does not retain the approximately linear positive regime of Swish"（C3）；附录 §B Eq.(18) `β tanh(z/β)=z+O(z³/β²)`（C5/F3）与紧随其后的 "It also recovers SwiGLU pointwise as β1,β2→∞"（C6/F5）；Eq.(19) `‖SiTU-GLU(x)‖∞ ≤ β1β2=100`（C7/F4）；§B 末段 "Unlike hard clamping of gate pre-activations, the smooth cap preserves nonzero gradients away from saturation boundaries, which we find to give better training behavior"（C8）；§2.3 "a chain of nearly four consecutive matrix multiplications" + "2.8-trillion-parameter scale"（C9）；Table 1 行 "Activation Function: SwiGLU → SiTU-GLU" 与 "Hybrid KDA–MLA"（C10）。SwiGLU 公式取 Shazeer 2020（arXiv:2002.05202）§2 Eq.(5) `SwiGLU(x,W,V,b,c,β)=Swish_β(xW+b)⊗(xV+c)`，F6 定位正确。逐点复算三处手算表与两个折叠块（x=0/2/10/50/100 的 g、u、y 与对照 SwiGLU，以及 z=0.5、β=4 的展开残差、x=100/β=4、25 的导数）全部与页面数值吻合。报告无 SiTU-GLU vs SwiGLU 消融（grep 全篇 "ablation/SwiGLU baseline/w/o" 仅命中视觉塔与数据采样），页面反复声明的"报告未给对照实验"成立。

## 问题

- [轻微·格式] dojo:summary（第 7 行）与正文：同一函数 softcap 在 dojo:summary 中写作 `$\operatorname{softcap}$`，正文（第 141、357、505 行）与 §4 表格写作 `$\mathrm{softcap}$`，同页两种写法｜引文依据：不适用｜修复要求：全页统一为一种写法（建议 `\operatorname{softcap}` 或统一 `\mathrm{softcap}`），并同步 §F2、§4 表格｜修复：｜复验：
- [轻微·技术] §3 折叠块「补充：βtanh(z/β)=z+O(z³/β²) 的推导」末句（第 113 行）："吻合到四位有效数字"所对比的是页面自列的两个偏差 −0.00259 与 −0.00260，二者仅在 1–2 位有效数字上一致，不是四位｜引文依据：页面自算偏差 −0.00259、理论 −0.00260；二者相差约 1.5×10⁻⁵，恰为被截去的五阶项 z⁵/1920 ≈ 0.03125/1920 ≈ 1.6×10⁻⁵｜修复要求：改成与所列数字相符的表述（例如"两者相差约 1.6×10⁻⁵，来自被截去的五阶项"），或改比对象（`4tanh(0.125)≈0.49741` 与二项截断值 `0.5−0.00260≈0.49740` 吻合到四位有效数字）；§「构造示例」中同项"吻合"（第 232 行）一并核准｜修复：｜复验：
- [轻微·可读性] §4 边界澄清段（第 159 行）："'away from saturation boundaries'不是'饱和后梯度恒非零'，而是远离饱和边界处非零"一句自相否定——后半句"远离饱和边界处非零"与前半句所否定的"饱和后梯度恒非零"所指几乎相同，读者无法从中读出区分；且紧跟的"进入深饱和后梯度指数衰减……但始终严格大于 0"又回到"饱和后非零"，整段读起来自相矛盾｜引文依据："the smooth cap preserves nonzero gradients away from saturation boundaries, which we find to give better training behavior."｜修复要求：删去"不是 A 而是 B"的辨析写法，直接说明来源含义——softcap 导数对任意有限输入严格大于 0，深饱和时按约 4e^{−2x/β} 指数衰减但不为 0，因此区别于 hard clamping 的严格 0｜修复：｜复验：
- [轻微·表述] 核心问题第 1 条解答（第 13 行）："它解决 K3 Stable LatentMoE 路由分支中……引发的激活爆炸"把来源的"抑制（suppress）"写成"解决"，措辞强于来源，且与本页其余各处一律使用的"抑制"（第 25、103、145、174、179、196、198、199 行等）不一致｜引文依据："Sigmoid Tanh Unit GLU (SiTU-GLU) to suppress activation explosion"｜修复要求：本处统一改为"抑制激活爆炸"｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：修复（无阻断/重要问题；上列 4 条轻微问题修复并复验后，页面可发布）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
