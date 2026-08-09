# 知识蒸馏独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源核查）
- 页面版本：index.html 537cee5a7453dc0bad6553b493f22681169953d1 / overview.html b0973c9d4fb0a062c13de81d95c1c8fa676091a5
- 时间：2026-08-09
- 来源：Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531, 2015（NIPS 2014 Deep Learning Workshop）。下文简称 HVD15。

## 段 A 盲读（小白视角，按页面顺序）

通读 index.html + overview.html，记录主线理解卡点；结束时逐题核对"读完你能回答"的 5 个学习目标是否由正文章节完整回答。

### 盲读卡点

1. 引子"70B 参数的教师模型，推理一次要数百毫秒、显存上百 GB"——70B 是什么单位、参数是什么概念，页面未解释。但属引子背景，读者可推断为"很大的模型"，不构成主线卡点。
2. §why-soft-targets 教师软分布表格中"类别 1"概率写为 0.000。softmax 不会输出严格 0，此处是四舍五入但未说明，读者可能困惑"教师为何对 1 完全零概率"。
3. §why-soft-targets "HVD15 §3 的'无数字 3'实验"——§3 是什么未在前文说明，读者需自行推断为论文第 3 节。
4. §kd-loss 训练流程图（`<pre class="diagram">`）中用 `v` 表示教师 logits、`z` 表示学生 logits。`v` 在前文 §temperature-softmax 的符号定义中未出现，定义藏在 §why-t-squared 折叠块内。读者初次看图不知道 `v` 是什么。
5. §kd-loss 总损失公式 $\mathcal{L} = \alpha \cdot T^2 \cdot \mathrm{KL}(p_T^{\text{teacher}} \| p_T^{\text{student}}) + (1-\alpha)\cdot \mathrm{CE}(y, p_1^{\text{student}})$ 中 $p_T^{\text{teacher}}$ 含义未在公式下方符号列表明说（"教师模型在温度 T 下的输出分布"），需读者从上下文推断。

### 学习目标闭环核对

页面"读完你能回答"5 项目标，逐题核对：

1. "为什么不能只用硬标签训练，必须借教师模型的软概率分布？软标签额外携带什么信息？" → §why-soft-targets 完整回答（硬标签 one-hot 丢类间关系；软分布携带类间相似度）。✓
2. "温度 $T$ 在 softmax 中具体起什么作用，过大和过小各会发生什么？" → §temperature-softmax 完整回答（三个极限行为 + callout yellow 说明过大引入噪声、过小退化为硬标签）。✓
3. "手算 logits $(2,1,0)$ 在 $T=1$ 与 $T=5$ 下的概率分布分别是什么？" → §hand-compute-softmax 完整手算（T=1 ≈ (0.665, 0.245, 0.090)；T=5 ≈ (0.402, 0.329, 0.269)）。✓
4. "KD 的总损失怎么写、各项的职责是什么、为什么要乘 $T^2$？" → §kd-loss + §why-t-squared 完整回答（公式 + 软项传暗知识、硬项保正确答案、$T^2$ 来自梯度压低补偿的推导）。✓
5. "知识蒸馏解决什么问题、不解决什么问题？它与 K3 MOPD 是什么关系？" → §boundaries 完整回答（解决部署压缩 ensemble；不解决超过教师；MOPD 是 on-policy 变体，表格对比 4 维差异）。✓

5 项学习目标全部由正文章节完整回答，闭环。

## 段 B 对照来源（HVD15）

### 核查项 1：定义与机制

| 页面论断 | 来源对照 | 结论 |
|---|---|---|
| C1 硬标签丢弃类间关系，暗知识 | HVD15 摘要与 §1、§2；"dark knowledge" 出自 §1 | ✓ |
| C2 温度 softmax 公式 $q_i = \exp(z_i/T)/\sum_j \exp(z_j/T)$ | HVD15 §2 公式直接给出 | ✓ |
| C3 温度极限行为（T→0 one-hot、T=1 标准 softmax、T→∞ 均匀） | HVD15 §2 描述 + 公式直接推出 | ✓ |
| C4 温度与师生容量关系（大教师配大 T；小学生配小 T） | HVD15 §2、§3：每层 ≥300 单元时 T>8 都相近；减到 30 单元时 T∈[2.5,4] 最佳 | ✓ |
| C5 KD 总损失（KL 形式） | HVD15 §2 原文用 cross-entropy 形式；教师固定时 CE = KL + H(p^t)，H(p^t) 常数。教学说明已标注等价性 | ✓（教学简化已说明） |
| C6 $T^2$ 缩放 | HVD15 §2 末段原文"Since the magnitudes of the gradients produced by the soft targets scale as $1/T^2$, they must be multiplied by $T^2$" | ✓ |
| C10 学生上限受教师限制 | HVD15 §3–§4 实验结论直接推出 | ✓ |
| C11 与 MOPD 关系 | HVD15 §2 off-policy 设定；MOPD 见 K3 报告（超出 HVD15 范围，页面已标注为独立概念页链接） | ✓ |

### 核查项 2：公式与推导

| 公式 | 复算 | 结论 |
|---|---|---|
| F1 温度 softmax $q_i = \exp(z_i/T)/\sum_j \exp(z_j/T)$ | 与 HVD15 §2 一致 | ✓ |
| F2 KD 总损失 $\mathcal{L} = \alpha T^2 \mathrm{KL}(p_T^t \| p_T^s) + (1-\alpha)\mathrm{CE}(y, p_1^s)$ | HVD15 §2 原为 CE 形式；页面采用现代 KL 形式，教学说明已标注等价 | ✓ |
| F3 KL 散度 $\mathrm{KL}(p\|q) = \sum_i p_i \ln(p_i/q_i)$ | 信息论标准定义 | ✓ |
| F4 高温零均值极限梯度 $\partial \mathcal{L}_{\text{soft}}/\partial z_i \approx \frac{1}{NT^2}(z_i - v_i)$ | HVD15 §2 末段推导；折叠块内推导链：标准 softmax 梯度 $(q_i - p^t_i)$ → 链式法则 $1/T$ → 高温极限 $1/(NT^2)$。推导自洽 | ✓ |
| 手算 T=1：logits (2,1,0)，$\exp(2)=7.389, \exp(1)=2.718, \exp(0)=1$，sum=11.107，q=(0.665, 0.245, 0.090) | 复算 7.389/11.107=0.6652，2.718/11.107=0.2447，1/11.107=0.0900 | ✓ |
| 手算 T=5：$\exp(0.4)=1.492, \exp(0.2)=1.221, \exp(0)=1$，sum=3.713，q=(0.402, 0.329, 0.269) | 复算 1.492/3.713=0.4018，1.221/3.713=0.3289，1/3.713=0.2693 | ✓ |
| logit matching 等价：高温零均值下软损失最小化等价于最小化 $\frac{1}{2N}\|z-v\|^2$ | HVD15 §2 末段；推导链自洽，页面已标注"仅在 $T\to\infty$ 且 $\sum_i z_i=0$ 时严格成立" | ✓ |

### 核查项 3：可运行代码

执行 §hand-compute-softmax 折叠块中的 Python 脚本（`/tmp/kd_verify.py`）：

```
=== T=1 vs T=5 main example ===
T=1.0: probs=[0.665241, 0.244728, 0.090031]
T=5.0: probs=[0.40176, 0.328933, 0.269307]

=== Limit behaviors ===
T=0.5: probs=[0.866813, 0.11731, 0.015876]
T=2.0: probs=[0.50648, 0.307196, 0.186324]
T=10.0: probs=[0.367165, 0.332225, 0.30061]
T=100.0: probs=[0.336672, 0.333322, 0.330006]
1/3 = 0.333333  (uniform limit)
```

与页面"预期输出"逐行一致。✓

### 核查项 4：事实与推断

| 页面数字 | 来源对照 | 结论 |
|---|---|---|
| N1 MNIST 错误数：教师 67、硬标签学生 146、蒸馏学生 74（T=20） | HVD15 §3 表格完全一致 | ✓ |
| N2 "无数字 3" 实验：996/1010 测试集 3 正确分类（经偏差校正） | HVD15 §3 原文：未校正蒸馏模型 206 错误（133 是 3，即 3 中正确 877/86.7%）；3 类 bias 增加 3.5 后 109 错误（14 是 3，即 3 中正确 996/98.6%）。**996 是 bias 校正后的数字** | 见问题 1 |
| N3 语音识别：基线 58.9%/10.9%；ensemble 61.1%/10.7%；蒸馏 60.8%/10.7%；8 层 2560 单元、约 85M 参数、约 2000 小时英语语音 | HVD15 §4 Table 1 与正文完全一致 | ✓ |
| N4 温度取值：MNIST 主实验 T=20；小容量学生 T≈2.5–4 | HVD15 §3 原文"每层 30 单元时，2.5 至 4 范围内的温度明显优于更高或更低的温度" | ✓ |
| N5 $\alpha$ 接近 1，论文未给统一最优值 | HVD15 §2、§3 报告硬项权重远小于软项 | ✓ |
| "7 的图片"软分布示例数字 | 教学示例，页面已标注"教学示例，数字为说明用，不代表具体教师模型的真实输出" | ✓（已标记） |

### 核查项 5：前置知识引用

- index.html ↔ overview.html 互相链接：index.html 有 `<a href="overview.html">快速阅读</a>`；overview.html 有 `<a href="index.html">深度教学 →</a>`。✓
- MOPD 概念页链接：index.html 用 `../../wiki/mopd/index.html`，overview.html 用 `../mopd/index.html`。两者解析到同一目标 `/Users/wendadawen/code/dojo/wiki/mopd/index.html`，文件存在。✓（index.html 路径多绕一层但功能正确）
- 返回首页链接：`../../index.html`，目标存在。✓

### 核查项 6：教学简化

- 用 KL 形式而非 cross-entropy 形式：HVD15 原用 CE 形式，教师固定时 CE = KL + H(p^t)，等价。教学说明已标注。✓
- softmax 与交叉熵只给最小定义：已说明"本文不展开 softmax 自身的推导"。✓
- 只覆盖 logit-based 输出层蒸馏：FitNets/自蒸馏/在线蒸馏不进入范围，已说明。✓
- $T^2$ 推导使用简化条件：已标注"仅在 $T\to\infty$ 且 $\sum_i z_i=0$ 时严格成立，实际有限 $T$ 下是近似"。✓
- "温度软化"类比：已标注"只帮助理解 T 的几何作用，不覆盖 T→0 数值溢出、T 与师生容量匹配等工程问题"。✓
- "暗知识"类比：已标注"只说明软分布携带的信息类型，不保证教师在所有任务上的非目标类概率都有意义"。✓

### 核查项 7：页面功能

- `python3 .dojo/scripts/validate.py wiki/knowledge-distillation/index.html` → `validation ok`，退出码 0。✓
- `python3 .dojo/scripts/validate.py wiki/knowledge-distillation/overview.html` → `validation ok`，退出码 0。✓
- KaTeX 渲染：delimiters 配置 `$$...$$`（display）与 `$...$`（inline），与正文公式用法一致。✓
- 折叠交互：`<details>` 标签使用正确，summary 可点击。✓
- 目录锚点：`scroll-margin-top: 90px` 避开固定导航。✓
- §hand-compute-softmax 标题含 `$T=1$`/`$T=5$` KaTeX 行内公式，目录 JS 用 textContent 提取会显示原始 `$...$` 标记。见问题 7。

## 问题

- [重要·技术] §why-soft-targets "无数字 3" 段落（index.html 第 696 行）：正文写"结果是测试集中 1010 个 3 里有 996 个仍被正确分类"，但 996 是 HVD15 §3 中偏差校正（3 类 bias 增加 3.5）后的结果；原始未校正蒸馏模型错误 206 个、其中 133 个是 3，即 3 中正确仅 877（86.7%）。正文未标注"经偏差校正"，读者会误以为 KD 直接做到 996/1010（98.6%）的 3 识别率，与来源事实的呈现方式不一致。修法：在正文中明确"经偏差校正（3 类 bias 增加 3.5）后，1010 个 3 中 996 个被正确分类"，或先给原始数据"未校正时 206 错误中 133 个是 3；将 3 类 bias 增加 3.5 后 109 错误中仅 14 个是 3，即 996/1010 正确"。 ｜ 修复：已在 line 696 正文改为"经偏差校正（将 3 类的 logit bias 增加 3.5）后，测试集中 1010 个 3 里有 996 个仍被正确分类；未校正时错误 206 个、其中 133 个是 3（即 3 中正确仅 877/86.7%）"；同步将 line 994 来源 C8 加上"（经偏差校正）"与 N2 一致。 ｜ 复验：validate.py 通过
- [轻微·盲读] §why-soft-targets 教师软分布表格（index.html 第 685 行）：类别 1 的概率写为 0.000，softmax 不会输出严格 0（这是四舍五入），读者可能困惑"教师为何对 1 完全零概率"。修法：将 0.000 改为 0.0001 量级的非零数字（如 0.0003），或在表格下注明"四舍五入到 3 位小数，0.000 表示小于 0.0005"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] §why-soft-targets（index.html 第 696 行）："HVD15 §3 的'无数字 3'实验"中 §3 是什么未在前文说明，读者需自行推断为论文第 3 节。修法：首次出现 §N 时注明"HVD15 论文第 N 节"，或改为"HVD15 第 3 节"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] §kd-loss 训练流程图（index.html 第 858–880 行 `<pre class="diagram">`）：图中用 `v` 表示教师 logits、`z` 表示学生 logits，但 `v` 在前文 §temperature-softmax 的符号定义中未出现，定义藏在 §why-t-squared 折叠块内。读者初次看图不知道 `v` 是什么。修法：在图中 `教师 v` 旁标注"教师 logits"，或在 §kd-loss 正文首次出现 v 时定义"v 表示教师 logits"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] §kd-loss 总损失公式（index.html 第 846 行）：公式下方符号列表未列出 $p_T^{\text{teacher}}$ 的含义（"教师模型在温度 T 下的输出分布，由教师 logits 经温度 softmax 得到"），需读者从上下文推断。修法：在公式下方符号列表中补一条"$p_T^{\text{teacher}}$：教师模型在温度 T 下的输出分布"。 ｜ 修复： ｜ 复验：
- [轻微·技术] §sources-and-teaching-notes C8 与 N2（index.html 第 994、1012 行）：C8 "996/1010 测试集 3 被正确分类"未标注偏差校正；N2 "996/1010 测试集 3 正确分类（经偏差校正）"标注了。同一数字两处标注不一致。修法：C8 也加上"（经偏差校正）"，与 N2 保持一致。 ｜ 修复： ｜ 复验：
- [轻微·页面功能] §hand-compute-softmax 标题（index.html 第 751 行）：`<h2 id="hand-compute-softmax">手算温度 softmax——3 类 logits 在 $T=1$ 与 $T=5$ 下的分布</h2>` 含 KaTeX 行内公式。目录生成 JS（第 1091 行）用 `h.textContent` 提取标题文本，会显示原始 `$T=1$` 而非渲染后的 `T=1`。修法：标题去掉 `$` 符号改为纯文本"在 T=1 与 T=5 下的分布"（KaTeX 不渲染但可读），或目录 JS 改为读取渲染后文本。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 6
- 学习目标闭环：5 项学习目标全部由正文章节完整回答。
- 代码验证：T=1/T=5/T=0.5/T=2/T=10/T=100 输出与页面预期逐行一致。
- validate.py：index.html、overview.html 均退出码 0。
- 链接：mopd 概念页存在；index.html 与 overview.html 互相链接；返回首页链接有效。
- 处置：进入修复。无阻断；1 个重要问题（996 数字未标注偏差校正）可在不改变研究范围与教学大纲前提下修复；6 个轻微问题可顺手修复。修复完成后重跑 validate.py 并交回原审查者复验。
