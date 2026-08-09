# SwiGLU 教学大纲

## 1. 页面开头

- 钩子问题：现代 LLM（LLaMA、PaLM、Mistral、Qwen、DeepSeek）的 FFN 几乎都用同一种激活——SwiGLU。它和 GLU 只差一个激活函数，为什么这个替换让它在 LLM 里成了事实标准？它"换门"换出的到底是什么能力？
- 一句话解释：SwiGLU 就是"把 GLU 门分支的 sigmoid 换成 Swish，让门分支能输出负值和无界正值，再与另一份线性投影逐元素相乘"。
- 要解决的具体问题：GLU 的 sigmoid 门恒正有界 $(0,1)$，只能"压低或放行"值分支，不能翻号也不能放大；Shazeer 2020 把它换成 Swish，让门分支可以输出负值（翻号）与无界正值（放大），并在 T5 上经验性观察到优于 ReLU/GELU 基线。
- 学习承诺：列出 Q1–Q5（见 scope.md §2.2，用 learning-goals 组件）。
- 首个具体场景：用 $x=[1.0,\,0.5]$ 与 $2\times2$ 权重算 SwiGLU 输出，看 Swish 门在第二维取负值如何让输出翻号（贯穿例子，首次出现）。
- 与第一章过渡：先把"为什么换门"讲清楚，再给公式与手算。

## 2. 章节设计

### S1：从 GLU 到 SwiGLU——为什么要换门

- 主要教学问题：GLU 的 sigmoid 门有什么形状限制？换成 Swish 后门分支多了什么能力？
- 对应范围：Q1（动机与定位）、Q4（Swish vs sigmoid 机制差别的引子）。
- 正文要点：
  - 回顾 GLU（链接 [GLU 概念页](../../wiki/glu/index.html)）：门分支 $\sigma(xV+c)\in(0,1)$，恒正、有界。门只能把值分支"压低"（乘 $(0,1)$）或"放行"（乘 $\approx1$），不能翻号也不能放大。
  - Shazeer 2020 §2 的动作：把门分支的 sigmoid 换成其它激活（ReLU/GELU/Swish），得 ReGLU/GEGLU/SwiGLU；SwiGLU 用 Swish。
  - Swish 的最小定义（F1）：$\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$。一行内联，不展开 Swish 论文（Ramachandran 2017）。
  - Swish 与 sigmoid 的形状差别（C4）：
    - sigmoid：恒正、有界 $(0,1)$、单调递增。
    - Swish：可负（$z<0$ 时 $\mathrm{Swish}(z)<0$）、正侧无界（$z\to+\infty$ 时 $\mathrm{Swish}(z)\sim z$）、非单调（先降后升）。
  - 一句话定位 SwiGLU（Q1 的"做什么"部分）：SwiGLU 是 GLU 家族中把门分支的 sigmoid 换成 Swish 的变体，由 Shazeer 2020 §2 Eq.(5) 定义。
- 讲解材料及职责：
  - 对照表（A14）：sigmoid 门 vs Swish 门的形状（恒正/可负、有界/无界、单调/非单调）。
  - 一行 Swish 定义 + 一行 sigmoid 回顾（基础记号，内联）。
  - ASCII 图示（A5）：GLU 与 SwiGLU 的结构对照（只差门分支的激活）。
- 前置知识安排：GLU 已有概念页，首次依赖处给链接；sigmoid、$\otimes$、$xW+b$ 沿用 GLU 页的基础记号定义。
- 完成检查：
  - 说出 GLU 与 SwiGLU 在"门分支激活"上的差别；
  - 说出 sigmoid 恒正有界 vs Swish 可负无界 这两条形状差别。
- 过渡：知道了"换门换出什么能力"，下一章给 SwiGLU 的精确公式并手算验证。

### S2：SwiGLU 的公式、Swish 定义与手算

- 主要教学问题：SwiGLU 怎么算？给一个小输入能算出输出吗？Swish 的边界值是什么？
- 对应范围：Q1（定义部分）、Q2、Q4（手算对照）。
- 正文要点：
  - 给出 SwiGLU 公式 F2：$\mathrm{SwiGLU}(x,W,V,b,c,\beta)=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$，逐符号解释：$x$ 输入、$W,V$ 两套权重、$b,c$ 两套偏置、$\beta$ Swish 超参（实验固定 $1$）、$\otimes$ 逐元素乘。
  - 明确 Shazeer 记法：激活在 $W$ 分支、$V$ 为值分支；与 [GLU 概念页](../../wiki/glu/index.html) 的 Dauphin 主记法（$\sigma$ 在 $V$ 分支）相差 $W\leftrightarrow V$ 标签，因 $\otimes$ 可交换而等价。读 SwiGLU 部署代码时通常用 Shazeer 记法。
  - $\beta$ 说明：Shazeer §1 与 §3.1 实验固定 $\beta=1$，现代 LLM 部署也固定 $\beta=1$；$\beta$ 是可调超参不是可学习参数。
  - Swish 边界值（C2）：$\mathrm{Swish}_1(0)=0\cdot\sigma(0)=0$；$\mathrm{Swish}_1(1)=1\cdot\sigma(1)\approx1\times0.7311=0.7311$；$\mathrm{Swish}_1(-1)=-1\cdot\sigma(-1)\approx-1\times0.2689=-0.2689$。正侧 $z\to+\infty$ 时 $\mathrm{Swish}_1(z)\sim z$；负侧 $z\to-\infty$ 时 $\mathrm{Swish}_1(z)\to0$。
  - 维度约定：$x\in\mathbb{R}^{1\times d_{in}}$，$W,V\in\mathbb{R}^{d_{in}\times d_{out}}$，$b,c\in\mathbb{R}^{d_{out}}$，输出 $\in\mathbb{R}^{1\times d_{out}}$。
  - 贯穿数字例子（Q2、Q4）：$x=[1.0,0.5]$，$W=\mathrm{diag}(1,-1)$，$V=I$，$b=c=0$，$\beta=1$。
    - 门分支内部线性：$xW+b=[1.0,-0.5]$。
    - Swish 门：$\mathrm{Swish}_1([1.0,-0.5])=[1.0\times\sigma(1.0),\,-0.5\times\sigma(-0.5)]\approx[0.7311,\,-0.1888]$。
    - 值分支：$xV+c=[1.0,0.5]$。
    - SwiGLU 输出：$[0.7311,\,-0.1888]\otimes[1.0,\,0.5]\approx[0.7311,\,-0.0944]$。
  - 与 GLU 同输入对照（Q4 核心）：同 $x,W,V,b,c$ 下 GLU 门 $\sigma(xW+b)=\sigma([1.0,-0.5])\approx[0.7311,\,0.3775]$，GLU 输出 $=[0.7311,\,0.1888]$。第二维 SwiGLU 输出 $-0.0944$（翻号）而 GLU 输出 $+0.1888$——Swish 门取负值让门控不仅能压低还能翻号。
  - 把"教学示例"标记清楚（数字为构造）。
- 讲解材料及职责：
  - 公式 F1+F2 + 符号说明（A2）。
  - 数字例子：正文中给出代入与中间值（A4，折叠块承载 Swish 边界值推导与同输入 GLU 对照的完整计算）。
  - 折叠块：Swish 边界值 $\mathrm{Swish}(0)=0$、$\mathrm{Swish}(1)\approx0.731$、$\mathrm{Swish}(-1)\approx-0.269$ 的逐步代入；同输入下 GLU 与 SwiGLU 的逐维对照。
- 前置知识：sigmoid 数值（$\sigma(1),\sigma(-0.5),\sigma(-1)$）内联计算。
- 完成检查：
  - 写出 SwiGLU 运算的三个步骤（门分支线性、Swish、值分支、相乘）；
  - 算出 $\mathrm{Swish}(0)$、$\mathrm{Swish}(1)$、$\mathrm{Swish}(-1)$ 的近似值；
  - 说出 SwiGLU 在第二维输出为什么是负值而 GLU 是正值。
- 过渡：会算了，下一章看它怎么塞进 Transformer FFN，以及为什么 LLaMA 内部维度取 $\tfrac83 d$。

### S3：把 SwiGLU 塞进 Transformer FFN：三矩阵与 $2/3$ 缩放

- 主要教学问题：SwiGLU 在 FFN 里长什么样？为什么要缩 $2/3$？LLaMA 的 $\tfrac83 d$ 从哪里来？
- 对应范围：Q3。C5/C6/C7/C11。
- 正文要点：
  - 标准 Transformer FFN（Shazeer §1，无偏置版本）：$\mathrm{FFN}_{\mathrm{ReLU}}(x,W_1,W_2)=\max(xW_1,0)W_2$，两个权重矩阵。
  - SwiGLU FFN（F3）：$\mathrm{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2)=(\mathrm{Swish}_1(xW)\otimes xV)W_2$，三个权重矩阵、无偏置。
  - 多一个矩阵，参数量就多。为保持参数量与计算量与双矩阵基线相等，Shazeer 把 $d_{ff}$ 缩为 $2/3$（C6 原文引用）。
  - 参数量等式（F4）：双矩阵 $2\cdot d\cdot d_{ff}$；三矩阵 $3\cdot d\cdot d_{ff}'$；令等 ⇒ $d_{ff}'=\tfrac23 d_{ff}$。
  - Shazeer §3.1 实例（C7、N2）：基线 $d_{ff}=3072$，SwiGLU 变体 $d_{ff}=2048$。
  - LLaMA 风格 $\tfrac83 d$（C11、F5）：标准 Transformer FFN 常取 $d_{ff}=4d$；SwiGLU FFN 取 $d_{ff}'=\tfrac23\times 4d=\tfrac83 d$。这是 LLaMA、Mistral、Qwen 等模型 FFN 内部维度的来源。
  - 部署事实（C10）：LLaMA（Touvron et al. 2023）采用 SwiGLU，引用 PaLM（Chowdhery et al. 2022）；后续 Mistral、Qwen、DeepSeek 等跟随。
- 讲解材料及职责：
  - 公式 F3 + 符号说明（A2）。
  - 参数量等式 F4 + LLaMA 风格 F5 推导（A3 简短推导）。
  - 对照表（A14）：双矩阵 ReLU FFN vs 三矩阵 SwiGLU FFN 的参数量与 $d_{ff}$。
  - 折叠块：参数量等式逐步推导 + $d=4096$（LLaMA-7B 风格）手算验证。
- 前置知识：$xW+b$、矩阵乘基础记号；FFN 一行定义（首次使用处）。
- 完成检查：
  - 写出 $\mathrm{FFN}_{\mathrm{SwiGLU}}$ 公式与三矩阵结构；
  - 算出三矩阵 FFN 要保持参数量不变需把 $d_{ff}$ 改成 $\tfrac23 d_{ff}$；
  - 推出 LLaMA 风格 $d_{ff}=\tfrac83 d$。
- 过渡：会部署了，最后一章看经验结论与边界——Shazeer 实验说了什么、没说什么，以及为什么社区选了 SwiGLU。

### S4：经验结论与边界——SwiGLU 为什么成为 LLM 标配

- 主要教学问题：Shazeer 实验里 SwiGLU 是最优吗？为什么不是最优却被社区广泛采用？SwiGLU 不解决什么？
- 对应范围：Q5。C8/C9/C10/C12。
- 正文要点：
  - 经验结论（C8、N1）：Shazeer Table 1 中 SwiGLU heldout log-perplexity $1.636$，低于 ReLU 基线 $1.677$；但 GEGLU $1.633$ 略优于 SwiGLU——SwiGLU 不是实验最优，是与 GEGLU 并列的"最优两变体之一"。
  - 实验限定（C7、C8）：T5 base、segment-filling、参数与计算量匹配；不是普适保证。
  - Shazeer 态度（C9）：§4 "divine benevolence" 原句引用——未给理论解释。
  - 社区采用路径（C10）：PaLM（2022）采用 SwiGLU → LLaMA（2023）跟随 → Mistral、Qwen、DeepSeek 等通过 HuggingFace/llama.cpp/vLLM 工具链锁定。SwiGLU 成为事实标准是路径依赖与轻微经验优势的组合，不构成"在所有任务上都最优"的保证。
  - GLU 家族变体派生对照（Q4 收尾）：GLU/ReGLU/GEGLU/SwiGLU/Bilinear 统一形式 $\mathrm{激活}(xW+b)\otimes(xV+c)$，差别只在门分支激活——sigmoid、ReLU、GELU、Swish、恒等。
  - SwiGLU 不解决什么：
    - 不保证在所有任务/模态上最优（实验限定 T5）；
    - 不替代归一化（与 RMSNorm/LayerNorm 正交）；
    - 不做跨位置信息聚合（那是注意力）；
    - Swish 门正侧无界，深层堆叠或路由分支可能放大激活——这是 SiTU-GLU 加 softcap 想解决的问题（C12，[SiTU-GLU 概念页](../../wiki/situ-glu/index.html)）。
  - 误解处理：scope §2.6 的 5 条误解中尚未在正文处理的收尾说明（SwiGLU≠GLU、Shazeer 未证最优、门无界、$\beta$ 固定 1、$\tfrac83 d$ 来自 Shazeer）。
- 讲解材料及职责：
  - 对照表（A14）：Shazeer Table 1 八行 log-perplexity（N1）。
  - 对照表（A14）：GLU 家族五变体的门分支激活与门值形状。
  - 事实引用：divine benevolence 原句 + 来源定位；LLaMA/PaLM 采用 SwiGLU 的事实 + 来源。
- 前置知识：无新前置。
- 完成检查：
  - 说出 Shazeer Table 1 中 SwiGLU 与 GEGLU 的 log-perplexity，并指出哪个更低；
  - 说出 Shazeer §4 "divine benevolence" 标注的是什么；
  - 说出 SwiGLU 不解决的一类问题（举一个非注意力、非归一化的例子也可）；
  - 说出 SiTU-GLU 是对 SwiGLU 的什么改进。
- 过渡：文末来源与教学说明。

## 3. 讲解顺序

S1（动机/为什么换门）→ S2（公式+Swish 边界值+手算，是什么）→ S3（FFN 部署+$2/3$ 缩放+$\tfrac83 d$）→ S4（经验结论+边界+下游关系）。先为什么再是什么；GLU 在 S1 首次依赖处给链接；sigmoid、$\otimes$、$xW+b$、Swish、FFN 在首次用到处内联一行定义；家族变体派生在 S1 简提、S4 收尾对照。

## 4. 贯穿例子

主线例子：$x=[1.0,\,0.5]$，$W=\mathrm{diag}(1,-1)$（Shazeer 记法，$W$ 是门分支权重），$V=I$（值分支权重近似直通），$b=c=0$，$\beta=1$。
- S2 首次出现：算出 SwiGLU 输出 $\approx[0.7311,\,-0.0944]$，并对照同输入 GLU 输出 $\approx[0.7311,\,0.1888]$——第二维 SwiGLU 翻号、GLU 不翻号。
- S3 复用：把同一 $x$ 套进 $\mathrm{FFN}_{\mathrm{SwiGLU}}$ 形式（加一个 $W_2$），看三矩阵结构。
- 折叠块：Swish 边界值 $\mathrm{Swish}(0)=0$、$\mathrm{Swish}(1)\approx0.731$、$\mathrm{Swish}(-1)\approx-0.269$ 的逐步代入；同输入下 GLU 与 SwiGLU 逐维对照。
所有数字便于手算，sigmoid 取四位小数。

## 5. 正文与折叠块分工

正文必有：F1 Swish 定义、F2 SwiGLU 公式与符号、Swish 边界值（$\mathrm{Swish}(0)=0$、$\mathrm{Swish}(1)\approx0.731$）、F3 FFN 公式、F4 参数量结论、F5 LLaMA $\tfrac83 d$ 推导、N1 Table 1 数字、divine benevolence 引用、C10 LLM 采用事实、C12 SiTU-GLU 下游关系、Q1–Q5 完成答案。
折叠块：Swish 边界值逐步代入；同输入 GLU vs SwiGLU 逐维对照；参数量等式逐步推导 + LLaMA-7B 风格 $d=4096$ 验算。折叠块全收起时，正文仍答全 Q1–Q5。

## 6. 范围与证据约束

全部内容来自 scope.md 已纳入范围与 evidence.md 已确认论断。无未消歧项。教学数字均标记。无证据不足论断进入生产。
