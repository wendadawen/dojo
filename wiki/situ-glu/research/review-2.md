# SiTU-GLU 独立审查（第二次）

- 审查者：独立上下文（AI 模拟小白读者 + 段B对照来源）
- 页面版本：index.html 0bde9d672c78b422515cd7d1135a00785e8d7532（commit ac5b744）
- 时间：2026-08-09 17:15 +0800
- 来源：/tmp/kimi-k3-research/k3-report.txt §2.3.2、§B（Eq.12/18/19）；HuggingFace moonshotai/Kimi-K3 config.json；Shazeer 2020 / Dauphin 2017 经 GLU evidence 登记

## 问题

- [重要·技术] index.html 开篇 callout 与 overview.html"为什么需要它"以 K3 事实口吻陈述"单个坐标的 pre-activation 可以大到 100、1000，乘积就到 10000、1000000，直接撑爆低精度训练"：源 §2.3.2（行 492-493）与 §B 仅泛述"coincident large coordinates can produce activation outliers and increase overflow risk in low-precision arithmetic"，未给出 100/1000/10000/1000000 这些具体数值。这些数值是教学构造（其中 x=100→SwiGLU=10000 在 §"公式与手算"教学示例小节才标注为构造），但开篇 callout 处未标注，读者会误以为 K3 实测观察到 pre-activation=1000、乘积=1000000。：在开篇 callout 处将该句的"100、1000 / 10000、1000000"标注为"教学示例数值（说明量级关系，非 K3 实测）"，或改为引用源的可定位依据（源 §2.3.2 行 492-493 + §B Eq.19 上界 100）。overview.html 同步处理。 ｜ 修复：已在 index.html 开篇 callout（L654）与 §"为什么需要它"（L707）两处"100、1000 / 10000、1000000"后补"（教学示例数值，说明量级关系，非 K3 实测）"，其中 callout 处另补源 §2.3.2 行 492-493 与 §B 的原始泛述引文以标明可定位依据；overview.html L52 同步补"（教学示例数值，说明量级关系，非 K3 实测）"。 ｜ 复验：
- [重要·技术] §"softcap vs clip"折叠块"$\tanh$ 的指数渐近形式"给出 x=100,β=4 时 softcap 导数 ≈ 4e⁻⁵⁰ ≈ 7.7×10⁻²²（正确，已用 Python 复算 4e⁻⁵⁰=7.715e-22 证实）；但"来源与教学说明·教学示例"小结对同一量写为"导数 ≈ 1.4×10⁻²¹（深饱和）"。两处数值不一致，7.7×10⁻²² 正确，1.4×10⁻²¹ 有误（β=25 浅饱和项 1.3×10⁻³ 两处一致，正确）。：将"来源与教学说明·教学示例"小结中的"1.4×10⁻²¹"改为"7.7×10⁻²²"（与 §softcap-vs-clip 折叠块及 4e⁻⁵⁰ 一致）。 ｜ 修复：已将 index.html L1010"来源与教学说明·教学示例"小结的"导数 $\approx 1.4\times 10^{-21}$（深饱和）"改为"导数 $\approx 7.7\times 10^{-22}$（深饱和）"，与 §softcap-vs-clip 折叠块 L905 的"4e⁻⁵⁰≈7.7×10⁻²²"一致。 ｜ 复验：
- [轻微·盲读] §"近原点像 SwiGLU"折叠块"补充：上界证明的逐项拆解"写"四项相乘：$\beta_1\cdot 1\cdot 1\cdot\beta_2\cdot 1=\beta_1\beta_2=100$"：称"四项"但实际列出 5 个乘因子（β1、1、1、β2、1）。：将"四项相乘"改为"五项相乘"或"两个常数 β1、β2 与三个有界因子相乘"，使计数与公式一致。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 1
- 段A盲读核对学习目标：5 条学习目标（Q1 一句话定位+解决什么、Q2 公式同时实现近原点近似与上界100、Q3 手算 x=0/10/100、Q4 softcap vs clip 梯度差别、Q5 K3 中使用位置与不解决的问题）均由正文（§为什么需要上界、§公式与手算、§近原点像SwiGLU/远点饱和、§softcap vs clip、§使用位置与边界）完整回答，无悬空。
- 段B对照来源结论：SiTU-GLU 公式 Eq.12（源行 499-502）逐项一致；β1=4/β2=25（行 541 + config activation_situ_beta=4.0/activation_situ_linear_beta=25.0）；softcap 局部展开 Eq.18 z+O((z/β)³)（§B 行 2768-2772）；β→∞ 逐点收敛 SwiGLU（§B 行 2773-2774）；输出上界 Eq.19 ‖SiTU-GLU‖∞≤β1β2=100（§B 行 2776-2779）；hard clamping 对比英文原句"Unlike hard clamping of gate pre-activations, the smooth cap preserves nonzero gradients away from saturation boundaries, which we find to give better training behavior."（§B 行 2780-2781）与页面引用逐字一致；"preventing either branch from dominating the product"（§B 行 2766）逐字一致；K2→K3 激活函数 SwiGLU→SiTU-GLU（Table 1 行 756）核实通过；GLU/SwiGLU/SiTU-GLU 三者门支/值支/上界对照表与源 Fig.4（行 438-458）一致。手算 x=0/10/100 三个点（g/u/y 及 SwiGLU 对照 0/99.995/10000）复算正确；z=0.5,β=4 局部展开验证 4·tanh(0.125)≈0.49741 与 -z³/(3β²)≈-0.00260 吻合。
- 处置：进入修复（2 重要 + 1 轻微）。两处重要均为数值/事实表述问题（一处教学构造未标注、一处内部数值不一致），不改变"SwiGLU 两因子无界→SiTU-GLU 两支 softcap 限界→乘积上界100"这一核心结论（该结论源充分支持），但需修复后发布。
