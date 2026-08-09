# GLU 教学大纲

## 1. 页面开头

- 钩子问题：把一个向量送进神经网络的一层，你能让"每个维度自己决定放多少出去"吗？现有读者只需知道"一层网络通常是把输入做线性变换再套个激活"。能不能不靠固定激活、而靠"另一个算出来的门"来逐维度调节？
- 一句话解释：GLU 就是"算两份线性投影，一份过 sigmoid 当门，再逐元素乘回去"。
- 要解决的具体问题：深层网络里 sigmoid/tanh 门控会把梯度也压没（GTU 现象）；单一线性+激活又缺一种"数据相关的乘性调制"。GLU 想同时拿到"门控非线性"和"梯度线性通路"。
- 学习承诺：列出 Q1–Q5（见 scope.md §2.2，用 learning-goals 组件）。
- 首个具体场景：用一个小向量 $x=[1.0,\,0.5]$ 与 $2\times2$ 权重算 GLU 输出，看门怎么把第一维放行、第二维压低（贯穿例子，首次出现）。
- 与第一章过渡：先把"为什么需要门"讲清楚，再给公式。

## 2. 章节设计

### S1：为什么需要"门"——从一层线性+激活的局限说起

- 主要教学问题：一层网络"线性变换+固定激活"缺了什么？门控想补什么？
- 对应范围：Q1（动机部分）。K1/K2（核心内容：动机、门控直觉）。
- 正文要点：
  - 一个常规层：$y=\mathrm{act}(xW+b)$，激活是固定的、对所有维度同一规则。
  - 缺什么：没有"按输入内容、逐维度决定放多少"的能力；用 sigmoid/tanh 做门又会把梯度也压下去（点出 GTU 现象，留到 S3 展开）。
  - 门控直觉：再算一份线性投影、过 sigmoid 压到 $(0,1)$ 当"门"，逐元素乘到"值"上——开门≈放行，关门≈屏蔽。
  - 一句话定位 GLU（Q1 的"做什么"部分）。
- 讲解材料及职责：
  - 图示（ASCII）：一层"线性+激活" vs "两份投影 + 门乘" 的结构对照（A5）。
  - 一行 sigmoid 定义（基础记号，内联）。
- 前置知识安排：sigmoid、$\otimes$、$xW+b$ 均为基础记号，首次使用处内联一行定义（见 scope §2.4）。
- 完成检查：
  - 说出常规层与门控层在"激活来源"上的差别；
  - 解释为什么门用 sigmoid 而不是用原始线性值。
- 过渡：知道了"想要门"，下一章给出 GLU 的精确公式并手算。

### S2：GLU 的公式与手算

- 主要教学问题：GLU 到底怎么算？给一个小输入能算出输出吗？
- 对应范围：Q1（定义部分）、Q3。K1/K3。
- 正文要点：
  - 给出 Dauphin 原始公式 F1：$h(\mathbf{X})=(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$，逐符号解释：值分支 $\mathbf{X}*\mathbf{W}+\mathbf{b}$、门分支 $\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$、$\otimes$ 逐元素乘、$\sigma$ sigmoid。
  - 说明 $*$ 在 Dauphin 是卷积，在 Transformer/FFN 语境退化为矩阵乘 $xW$；本文后续按向量-矩阵乘 $xW$ 书写以简化（教学简化，说明）。
  - 维度约定：$x\in\mathbb{R}^{1\times d_{in}}$，$W,V\in\mathbb{R}^{d_{in}\times d_{out}}$，$b,c\in\mathbb{R}^{d_{out}}$，输出 $\in\mathbb{R}^{1\times d_{out}}$。
  - 边界检查：门全为 1 ⇒ 输出=值分支（线性直通）；门全为 0 ⇒ 输出=0（屏蔽）；门在 $(0,1)$ ⇒ 逐维度缩放。
  - 贯穿数字例子（Q3）：$x=[1.0,0.5]$，$W=I$，$V=\mathrm{diag}(1,-1)$，$b=c=0$。算值分支 $xW+b=[1,0.5]$、门分支 $\sigma(xV+c)=\sigma([1,-0.5])=[0.7311,0.3775]$、输出 $=[0.7311,0.1888]$。翻译：第一维门 0.73 放行 73%，第二维门 0.38 压到 38%×0.5。
  - 把"教学示例"标记清楚（数字为构造）。
- 讲解材料及职责：
  - 公式 F1 + 符号说明（A2）。
  - 数字例子：正文中给出代入与中间值（A4，折叠块承载更长手算/对照）。
  - 折叠块：极端门控制对照（门→0、门→1 的退化）与另一组权重对照。
- 前置知识：sigmoid 数值（$\sigma(1),\sigma(-0.5)$）内联计算。
- 完成检查：
  - 给定 $x,W,V,b,c$，写出 GLU 运算的三个步骤；
  - 解释门为全 1 / 全 0 时输出退化为什么。
- 过渡：会算了，但 GLU 被提出是为了"不杀梯度"——下一章用梯度路径说明为什么。

### S3：为什么 GLU 给梯度留了一条线性通路

- 主要教学问题：为什么说 GLU 缓解梯度消失？它和 GTU 的差别在哪一项梯度里？
- 对应范围：Q2。K2（梯度通路分析）。
- 正文要点：
  - 先简化到 Dauphin 的分析形式 $Y=\mathbf{X}\otimes\sigma(\mathbf{X})$（两分支共享 $\mathbf{X}$），说明这是为隔离梯度结构做的简化（教学简化）。
  - 给 GLU 梯度 F4：$\nabla[\mathbf{X}\otimes\sigma(\mathbf{X})]=\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\mathbf{X}\otimes\sigma'(\mathbf{X})\nabla\mathbf{X}$。
  - 指出第一项 $\nabla\mathbf{X}\otimes\sigma(\mathbf{X})$：不含 $\sigma'$，被"门值本身"缩放；对开门单元 $\sigma(\mathbf{X})\approx1$ 时近似 $\nabla\mathbf{X}$——"乘性跳连"。
  - 对照 GTU 梯度 F3：$\nabla[\tanh(\mathbf{X})\otimes\sigma(\mathbf{X})]=\tanh'(\mathbf{X})\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\sigma'(\mathbf{X})\nabla\mathbf{X}\otimes\tanh(\mathbf{X})$，两项分别带 $\tanh'$ 和 $\sigma'$，随 $|\mathbf{X}|$ 增大趋零。
  - 结论：GLU 把"梯度是否被压没"从"激活导数是否趋零"改成"门值是否开门"，门可控、可学。
  - 边界：门饱和到 0 时该通路也趋零——不是"永不消失"，是"门控决定何时通"。
- 讲解材料及职责：
  - 公式 F3/F4 + 逐项标注哪项含导数、哪项含门值（A2/A3）。
  - 折叠块：推导 F4 的链式法则步骤（乘积求导 + sigmoid 导数）。
  - 对照表：GLU vs GTU 的梯度通路逐项（A14 对照表格）。
- 前置知识：链式法则、$\sigma'=\sigma(1-\sigma)$ 内联一行。
- 完成检查：
  - 指出 GLU 梯度哪一项不含 $\sigma'$；
  - 解释为什么 GTU 的 $\tanh'$ 会让深层梯度消失而 GLU 不一定。
- 过渡：理解了 GLU 本身，下一章看它怎么变成一整个家族。

### S4：GLU 家族——换激活、去门控、塞进 FFN

- 主要教学问题：Bilinear、SwiGLU、GeGLU、ReGLU 怎么从 GLU 派生？记号在两篇论文间有何差异？塞进 Transformer FFN 时为什么缩 $2/3$？
- 对应范围：Q4、Q5（FFN 2/3 部分）。K4。
- 正文要点：
  - 派生规则（C6）：门分支的 sigmoid 可替换——ReGLU 用 ReLU、GEGLU 用 GELU、SwiGLU 用 Swish；去激活得 Bilinear（C5）。
  - 给出 F6 三变体与 F5 Bilinear 公式。
  - 记号差异（C9）：Dauphin $\sigma$ 在 $V$ 分支；Shazeer $\sigma$/激活在 $W$ 分支。强调读 SwiGLU 时用 Shazeer 记法。
  - FFN 用法（F7）：$\mathrm{FFN}_{GLU}(x,W,V,W_2)=(\sigma(xW)\otimes xV)W_2$，三矩阵、无偏置。
  - 参数量等式（F8）：$3\cdot d\cdot d_{ff}'=2\cdot d\cdot d_{ff}\Rightarrow d_{ff}'=\tfrac23 d_{ff}$；实例 $3072\to2048$（N2）。
  - 用 $d=768,d_{ff}=3072$ 手算参数量对照（教学示例，验证 2/3）。
- 讲解材料及职责：
  - 对照表（A14）：Dauphin vs Shazeer 记法；GLU/Bilinear/ReGLU/GEGLU/SwiGLU 的激活与公式。
  - 折叠块：参数量等式逐步推导（$d_{ff}'=\tfrac23 d_{ff}$）。
  - 公式 F5/F6/F7 + 符号说明（A2）。
- 前置知识：Swish $x\cdot\sigma(x)$、GELU、ReLU 作为激活名出现，不展开（家族成员各自的完整机制不属本文）。
- 完成检查：
  - 说出把 GLU 变成 SwiGLU 改了什么；
  - 算出三矩阵 FFN 要保持参数量不变需把 $d_{ff}$ 改成多少。
- 过渡：家族讲完，最后一章说清经验结论的边界与不保证什么。

### S5：经验结论与边界——GLU 不保证什么

- 主要教学问题：Shazeer 的实验说明了什么？它没说明什么？GLU 适合/不适合什么？
- 对应范围：Q5（不保证部分）。B1（Shazeer Table 1 数字）、B2（divine benevolence）。
- 正文要点：
  - 经验结论（C8）：T5 语言建模上所有 GLU 变体优于 ReLU/GELU 基线，GEGLU/SwiGLU 最优（N1 数字）。
  - 边界：实验限定 T5 base、segment-filling、参数与计算量匹配；不是普适保证。
  - 原文态度（C8）：Shazeer §4 "divine benevolence"——未给理论解释。
  - GLU 适合：需要数据相关乘性调制、深层前向、门控可学的场景。
  - GLU 不适合/不解决：不替代归一化、不做跨位置信息聚合（那是注意力）、不保证全局最优；门全关时近似零输出。
  - 误解处理：把 scope §2.6 的 4 条误解中尚未在正文处理的收尾说明（GLU≠sigmoid 激活、不必然优于 ReLU、门不必 sigmoid、记号差异）。
- 讲解材料及职责：
  - 对照表（A14）：Shazeer Table 1 的 log-perplexity（N1）。
  - 事实引用：divine benevolence 原句 + 来源定位。
- 前置知识：无新前置。
- 完成检查：
  - 说出 Shazeer 实验的一个限定条件；
  - 指出 GLU 不解决的一类问题。
- 过渡：文末来源与教学说明。

## 3. 讲解顺序

S1（动机/为什么需要门）→ S2（公式+手算，是什么）→ S3（梯度通路，为什么有效）→ S4（家族派生+FFN 用法）→ S5（边界/不保证）。先为什么再是什么；sigmoid/$\otimes$/线性投影/链式法则在首次用到处内联一行定义；GTU 在 S3 用到时才引入；家族变体在 S4 才引入。

## 4. 贯穿例子

主线例子：$x=[1.0,\,0.5]$，$W=I$（值分支近似直通），$V=\mathrm{diag}(1,-1)$，$b=c=0$。
- S2 首次出现：算出 GLU 输出 $[0.7311,0.1888]$，解释门对两维的不同调制。
- S3 复用：用同一 $x$ 与 $V$ 说明门值 $\sigma([1,-0.5])=[0.7311,0.3775]$ 如何作为梯度通路的缩放系数。
- S4 复用：把同一 $x$ 套进 Shazeer FFN 形式（加一个 $W_2$），看三矩阵结构。
- 极端对照（折叠块）：把 $V$ 放大成 $\mathrm{diag}(10,-10)$，门→1/0 的退化，验证"开门直通、关门屏蔽"。
所有数字便于手算，sigmoid 取四位小数。

## 5. 正文与折叠块分工

正文必有：F1 公式与符号、门控直觉、F4 梯度通路结论与对照 GTU、F5/F6/F7 家族公式、F8 参数量结论、N1 经验数字、divine benevolence 引用、Q1–Q5 完成答案。
折叠块：F4 的链式法则推导；极端门对照的手算；参数量等式推导；Shazeer Table 1 完整表（若过长）。折叠块全收起时，正文仍答全 Q1–Q5。

## 6. 范围与证据约束

全部内容来自 scope.md 已纳入范围与 evidence.md 已确认论断。无未消歧项。教学数字均标记。无证据不足论断进入生产。
