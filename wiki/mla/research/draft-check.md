# Multi-head Latent Attention（MLA）初稿检查

## 输入版本

- scope.md：已编写，含概念歧义处理（已裁定）、5 个学习目标（Q1–Q5）、内容分级、前置知识映射（mqa-gqa/low-rank-projection/rope 未生成标占位）、4 个常见误解（M1–M4）、适用边界。
- evidence.md：已编写，含 8 条核心论断（C1–C8）、6 条核心公式（F1–F6）、6 条外部数字（N1–N6），全部标注来源定位与置信状态。
- outline.md：已编写，含页面开头（钩子+学习承诺+贯穿例子配置）、5 个正文章节（S1–S5，每章对应一个学习目标）、讲解顺序检查、贯穿例子 5 次推进、讲解材料职责表、正文与折叠块分工。
- glossary.md：已编写，含缩写与机制名 8 条、符号 30+ 条、易混淆记号说明 3 条。

## 大纲落实

逐项检查：

- **章节**：S1（KV 联合压缩）、S2（矩阵吸收）、S3（解耦 RoPE）、S4（KV cache 对照）、S5（K3 Gated MLA）+ 文末"来源与教学说明"——全部落实，无增删。
- **学习目标**：5 个学习目标在"读完你能回答"组件中列出，每章末尾完成检查对应一个目标。
- **前置知识**：mqa-gqa、low-rank-projection、rope 三处前置概念在正文首次用到时给出占位提示（`<em>...概念页（待生成）</em>`），无 `<a href>` 链接（避免 validate.py broken reference）。linear-attention 概念页已存在，使用正常链接。
- **贯穿例子**：d_h=4, n_h=2, d_c=3, d_h^R=2, h_t=(1,0,1,0) 贯穿 S1–S5。S1 手算 c_t^KV、S2 手算 q' 与等价性验证、S3 手算拼接与 cache 更新、S4 用真实数值（DeepSeek-V2、K3）、S5 手算 output gate。5 次推进全部落实。
- **误解和边界**：M1（MLA 不是 GQA 极端）在 S4 处理；M2（缓存的是潜向量不是压缩 K/V）在 S1 处理；M3（仍全局注意力）在 S1 处理；M4（RoPE 不能直接作用在 c_t 上）在 S3 处理。适用边界在 scope.md 记录、正文 S1/S4 体现。
- **过渡**：每章末尾有过渡句总结本章结论、指出下一章要解决的问题——全部落实。

## 学习目标闭环

逐题核对：

- **Q1**（MLA 用低秩联合压缩如何减少 KV cache，同时保留全局注意力）：S1 完整回答。给出 c_t^KV=W^DKV h_t、k_t^C=W^UK c_t^KV、v_t^C=W^UV c_t^KV 三公式（F2），手算 h_t→c_t^KV 与 cache 从 16 降到 3，并明确"注意力公式仍是 softmax 全局求和"。
- **Q2**（为什么 MLA 要把 RoPE 解耦）：S3 完整回答。给出 RoPE 作用在 k_t^C 上会破坏 W^UK 吸收的反例代数（折叠块），给出解耦方案 q^R/k^R/拼接/√(d_h+d_h^R) 公式（F4），手算拼接后 6 维向量与注意力分数。
- **Q3**（KV cache 每 token 占多少，相比 MHA/GQA/MQA 减少多少）：S4 完整回答。给出四种机制 cache 公式对照表，代入 DeepSeek-V2 数值（MHA=1,966,080 / MQA=15,360 / MLA=34,560 / 比值 1/57）与 K3 数值（比值 1/43），并澄清 93.3% 的 baseline 是 DeepSeek 67B 而非同配置 MHA。
- **Q4**（推理时如何避免显式重建 K/V）：S2 完整回答。给出 q^T k^C = (W^UK^T q)^T c_t^KV 的代数变形，W^UK 吸进 W^Q、W^UV 吸进 W^O 的结论，手算 q'=(1,1,0) 与 q'^T c_t^KV=1 验证等价性，ASCII 图示对比 MHA 与 MLA 推理路径。
- **Q5**（K3 的两处改动）：S5 完整回答。改动一 NoPE（动机：分工+免去 RoPE 频率调整），改动二 output gate（公式 F6、满秩 W^g 的意义），手算 Sigmoid(1,0,-1,0)⊙(1,2,3,4)=(0.73,1.0,0.81,2.0)。

全部 5 个学习目标由正文章节完整回答，不依赖折叠块。

## 代码运行

无可运行代码。本文为概念讲解页，核心机制用手算例子验证，不需要可运行代码。

## 机械检查

```
$ python3 .dojo/scripts/validate.py wiki/mla/index.html
validation ok: wiki/mla/index.html
退出码 0

$ python3 .dojo/scripts/validate.py wiki/mla/overview.html
validation ok: wiki/mla/overview.html
退出码 0
```

两个页面均通过 validate.py，无占位符残留、无组件标记残留、无重复 ID、无断链。

## 手算例子复算

用 Python 独立复算全部手算数字：

```
S1 cache ratio: 3/16 = 0.1875 ≈ 19% ✓（页面写 3/16 ≈ 19%）
S2 q^T k^C = 1, q'^T c = 1 ✓（两种算法结果相同）
S3 score = 1/sqrt(6) = 0.4082 ✓（页面写 ≈ 0.408）
S4 DV: MHA=1966080, MQA=15360, MLA=34560, ratio=56.9x ✓（页面写 ≈ 1/57）
S4 K3: MHA/layer=24576, MLA/layer=576, ratio=42.7x ✓（页面写 ≈ 1/43）
S5: Sig(1)=0.7311, Sig(0)=0.5, Sig(-1)=0.2689 ✓
S5 gate⊙o = [0.7311, 1.0, 0.8068, 2.0] ✓（页面写 0.8067，差异来自使用四舍五入的中间值 0.2689×3=0.8067，内部一致）
```

注：S5 第 3 通道页面写 0.8067，Python 用未四舍红入的 Sigmoid(-1)=0.26894... 算出 0.8068。差异 0.0001，因页面用四舍五入后的 0.2689 作为中间值（0.2689×3=0.8067 正确），便于读者手算复验。内部一致，不影响教学正确性。

## 公式渲染与交互

- KaTeX 定界符：行内 $...$、块级 $$...$$，与模板 auto-render 配置一致。
- 关键公式已逐个检查：矩阵符号 $W^{DKV}$、$\mathbf{c}_t^{KV}$、$\sqrt{d_h + d_h^R}$、$\mathrm{Sigmoid}$、$\odot$、$\top$（转置）等 LaTeX 语法正确。
- 折叠块：4 个 `<details>` 块（S1 形状说明、S2 W^UV 推导、S3 反例代数 + k^R 共享说明、S5 W^g 满秩说明），全部有具体 summary 标题，收起时正文仍完整。
- 表格：S4 四机制对照表，使用 `table-scroll` 组件。
- ASCII 图示：S2 推理路径对比图，使用 `diagram` 组件。
- callout：3 处（S1 blue 关键转变、S4 yellow 93.3% baseline 澄清、S4 purple K3 NoPE 不确定性说明）。
- context-box：贯穿例子维度配置。
- learning-goals：5 个学习目标。
- blockquote.meta：主要依据摘要。

## 写作偏差

无。所有章节、学习目标、前置知识、贯穿例子、误解和边界均按 outline.md 落实，未引入大纲外内容，未改变前置知识映射，未把正文必要内容移入折叠块。

## 不确定性标注

- **N6（K3 NoPE 对 cache 的影响）**：K3 报告 §2.1.2 第二段说 MLA 层用 NoPE，但 config.json 仍保留 qk_rope_head_dim=64。报告未明确 NoPE 下解耦 RoPE 分支是否完全移除。本页按 config 数值计算 cache=(d_c+d_h^R)=576，并在 S4 purple callout 中明确标注此不确定性。若分支移除则实际为 d_c=512。此标注已落入 evidence.md N6 和正文 S4。

## 未完成项（按任务指令）

- **不更新 content.json**：按要求，本次只执行 plan + write，不更新 content.json。
- **不做 check**：按要求，独立审查（check 阶段）由编排者安排，不在本次任务范围。
- **前置概念页未生成**：mqa-gqa、low-rank-projection、rope 三处按任务指令标占位，不递归生成。
