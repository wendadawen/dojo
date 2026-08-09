# SiTU-GLU 教学大纲

## 1. 页面开头

- 钩子问题：神经网络一层 FFN 的输出是两个数的乘积——如果两个数都没有上界，乘积会怎么涨？K3 在 2.8 万亿参数 + Stable LatentMoE 四连矩阵相乘结构下恰好遇到这件事：单个坐标的 pre-activation 可以大到 100、1000，乘积就到 10000、1000000，直接撑爆低精度训练。给输出加个 clip 就够了吗？clip 的梯度在饱和后是 0——模型进了饱和区就死。SiTU-GLU 要回答的是：能不能让两个乘性因子**自己**饱和到固定上界，同时保留 SwiGLU 在原点附近的近线性响应？
- 一句话解释：SiTU-GLU 把 SwiGLU 的门分支线性因子和值分支同时套上 softcap $\beta\tanh(x/\beta)$，让两个因子都平滑饱和，乘积上界锁在 $\beta_1\beta_2=100$。
- 要解决的具体问题：Stable LatentMoE 路由分支的四连矩阵相乘把 SwiGLU 的无界乘性因子放大到激活爆炸；clip 会让饱和区梯度归零。
- 学习承诺：列出 Q1–Q5（见 scope.md §2.2，用 learning-goals 组件）。
- 首个具体场景：标量输入 $x$（设两支 pre-act 均为 $x$ 以隔离函数行为），看 $x=0/10/100$ 时 SiTU-GLU 与 SwiGLU 的输出差异（贯穿例子，首次出现）。
- 与第一章过渡：先把"SwiGLU 为什么会爆炸、GLU 为什么不够"讲清楚，再给 SiTU-GLU 公式。

## 2. 章节设计

### S1：为什么需要给 SwiGLU 加上界——激活爆炸从哪里来

- 主要教学问题：SwiGLU 的两个乘性因子为什么会让 K3 出现激活爆炸？GLU 的 sigmoid 门为什么不够？
- 对应范围：Q1（动机部分）。K1/K2（核心内容：动机、softcap 函数）。
- 正文要点：
  - 回顾 SwiGLU 公式 F6：$\mathrm{SwiGLU}(x)=(W_g x\cdot\sigma(W_g x))\odot(W_u x)$，引用 [GLU 概念页](../../wiki/glu/index.html) 不重复家族派生。
  - 两个乘性因子无界（C3）：门支线性因子 $W_g x$ 在 $W_g x\to\infty$ 时近似 $W_g x$（被 $\sigma$ 压到 1 后只剩线性因子），值支 $W_u x$ 也无界；乘积在两支同时大时近似 $W_g x\cdot W_u x$，量级 $O(x^2)$。
  - K3 的放大场景（C9）：Stable LatentMoE 路由分支 $W_\downarrow\to$ expert FFN $\to W_\uparrow$ 四连矩阵相乘，加上 2.8T 参数规模，单个 pre-activation 可以大到 100、1000，乘积到 10000、1000000。
  - GLU 为什么不够（C3 第二句）：GLU 的 sigmoid 门有界但值 $W_u x$ 仍无界，且 GLU 没保留 Swish 近原点的正侧线性响应（Swish 在 $x>0$ 时近似线性 $x$、在 $x<0$ 时压到 0，GLU 的 $\sigma$ 门在 $x>0$ 时趋 1 但在 $x<0$ 时趋 0 但缺乏 Swish 的近原点线性过渡）。
  - 一句话定位 SiTU-GLU（Q1 的"做什么"部分）：把 SwiGLU 的两个无界线性因子各自换成 softcap $\beta\tanh(x/\beta)$，让它们平滑饱和到 $\pm\beta$。
  - softcap 函数 F2 一行定义：$\mathrm{softcap}(x,\beta)=\beta\tanh(x/\beta)$；$\tanh$ 把任意实数压到 $(-1,1)$，乘 $\beta$ 后压到 $(-\beta,\beta)$；K3 取 $\beta_1=4$（门支）、$\beta_2=25$（值支）。
- 讲解材料及职责：
  - 对照表（A14）：GLU、SwiGLU、SiTU-GLU 三者的"门支线性因子""门支激活""值支"形状对照，直接给视觉对比（不抄 Fig.4 图片，用文字表格表达）。
  - softcap 一行定义（基础记号，内联）。
- 前置知识安排：SwiGLU 公式引用 [GLU 概念页](../../wiki/glu/index.html)；$\tanh,\sigma,\odot,W x$ 内联一行定义。
- 完成检查：
  - 指出 SwiGLU 的两个无界乘性因子分别是哪两个；
  - 解释 GLU 的 sigmoid 门有界但 K3 仍不采用 GLU 的两个原因。
- 过渡：知道"为什么需要给两支都加上界"，下一章给出 SiTU-GLU 的精确公式并手算。

### S2：SiTU-GLU 的公式——softcap 套到门支与值支

- 主要教学问题：SiTU-GLU 到底怎么算？给一个标量输入能算出输出吗？
- 对应范围：Q1（定义部分）、Q3。K1/K3。
- 正文要点：
  - 给出 F1 公式：
    $$\mathrm{SiTU\text{-}GLU}(x)=\big[\beta_1\tanh(W_g x/\beta_1)\odot\sigma(W_g x)\big]\odot\big[\beta_2\tanh(W_u x/\beta_2)\big].$$
  - 逐项说明每个符号：
    - $x$：输入向量。
    - $W_g, W_u$：门分支与值分支的权重矩阵。
    - $\beta_1=4, \beta_2=25$：K3 的 soft-cap 超参（C4）。
    - 门支 $\beta_1\tanh(W_g x/\beta_1)\odot\sigma(W_g x)$：把 Swish 门 $W_g x\cdot\sigma(W_g x)$ 的线性因子 $W_g x$ 换成 softcap $\beta_1\tanh(W_g x/\beta_1)$，保留 $\sigma(W_g x)$ 因子不变（C2）。
    - 值支 $\beta_2\tanh(W_u x/\beta_2)$：把线性值 $W_u x$ 换成 softcap。
    - $\odot$：逐元素乘积。
  - 维度约定：所有量同形（逐元素运算），不像矩阵乘那样改变形状。
  - 边界检查三个：
    - $W_g x=0$：门支 $\beta_1\tanh(0)\cdot\sigma(0)=0\cdot 0.5=0$，值支 $\beta_2\tanh(0)=0$，输出 0——原点直通零。
    - $W_g x\to\infty$：门支 $\beta_1\cdot 1\cdot 1=\beta_1=4$，值支 $\beta_2\cdot 1=\beta_2=25$，输出 $\to\beta_1\beta_2=100$——正侧饱和到上界。
    - $W_g x\to -\infty$：门支 $\beta_1\cdot(-1)\cdot 0=0$（被 $\sigma$ 杀到 0），值支 $\beta_2\cdot(-1)=-\beta_2$，输出 $\to 0$——负侧被 sigmoid 压到 0。
  - 贯穿数字例子（Q3）：标量 $x$（设两支 pre-act 均为 $x$）：
    - $x=0$：$g=0,u=0,y=0$（与 SwiGLU 同点同值）。
    - $x=10$：$g=4\tanh(2.5)\sigma(10)\approx 3.9463$，$u=25\tanh(0.4)\approx 9.4987$，$y\approx 37.485$。对比 SwiGLU 同点 $10\cdot\sigma(10)\cdot 10\approx 99.995$——SiTU-GLU 在这里已经被 softcap 压住，远低于 SwiGLU。
    - $x=100$：$g=4\tanh(25)\sigma(100)\approx 4$，$u=25\tanh(4)\approx 24.983$，$y\approx 99.933$——接近上界 100；对比 SwiGLU 同点 $100\cdot\sigma(100)\cdot 100=10000$——无界增长 100 倍。
  - 把"教学示例"标记清楚（数字为构造，目的是隔离函数行为）。
- 讲解材料及职责：
  - 公式 F1 + 符号说明（A2）。
  - 数字例子：正文给出代入与中间值（A4），对照表（SwiGLU vs SiTU-GLU 同输入输出）。
  - 折叠块：$x=2$、$x=50$ 中间点的完整手算，验证从原点到饱和的过渡是平滑的（不是台阶）。
- 前置知识：$\tanh$ 数值（$\tanh(2.5), \tanh(0.4), \tanh(4)$）内联计算或折叠块给出。
- 完成检查：
  - 给定 $x$ 与 $\beta_1,\beta_2$，写出 SiTU-GLU 的运算三步（门支、值支、相乘）；
  - 解释 $x=0$、$x=10$、$x=100$ 三个点的输出差别与上界 100 的关系。
- 过渡：会算了，但为什么"近原点像 SwiGLU、远点饱和到 100"两件事能同时成立——下一章用局部展开和上界证明回答。

### S3：近原点像 SwiGLU、远点饱和——两个性质怎么同时成立

- 主要教学问题：SiTU-GLU 怎么做到"近原点一阶等于 SwiGLU"和"输出有界 100"两件事同时成立？极限情况下它和 SwiGLU 是什么关系？
- 对应范围：Q2。K2（局部展开 + 上界证明）。
- 正文要点：
  - 局部展开 F3（C5）：对 $|z|\ll\beta$，$\beta\tanh(z/\beta)=z+O((z/\beta)^3)$。代入 F1：
    - 门支 $\beta_1\tanh(W_g x/\beta_1)\cdot\sigma(W_g x)\approx (W_g x+O((W_g x/\beta_1)^3))\cdot\sigma(W_g x)$，当 $|W_g x|\ll 4$ 时退化为 Swish 门 $W_g x\cdot\sigma(W_g x)$。
    - 值支 $\beta_2\tanh(W_u x/\beta_2)\approx W_u x+O((W_u x/\beta_2)^3)$，当 $|W_u x|\ll 25$ 时退化为线性值 $W_u x$。
    - 乘积在两支都近原点时一阶等价于 SwiGLU。
  - 上界证明 F4（C7）：$|\tanh|\le 1$、$0<\sigma<1$ ⇒ 每个坐标 $|\mathrm{SiTU\text{-}GLU}|\le\beta_1\cdot 1\cdot 1\cdot\beta_2\cdot 1=\beta_1\beta_2=100$。
    - 强调："两支都套"是上界成立的关键——若只套门支，值支仍无界，乘积仍无界（C2 / 误解 4）。
  - 极限行为 F5（C6）：$\beta_1,\beta_2\to\infty$ 时 $\beta\tanh(z/\beta)\to z$（因 $\tanh(u)\to u$ 当 $u\to 0$），SiTU-GLU 逐点收敛到 SwiGLU。这是 SiTU-GLU 是 SwiGLU 的"有界化版本"的形式保证。
  - 把"$\beta_1=4,\beta_2=25$ 的选择"标为 K3 工程设定（C4 / 误解 3），不是普适最优，无对照实验。
- 讲解材料及职责：
  - 公式 F3/F4/F5 + 代入推导（A2/A3）。
  - 折叠块：$\beta\tanh(z/\beta)$ 的局部展开推导（用 $\tanh$ 的泰勒展开 $\tanh u = u - u^3/3 + O(u^5)$ 代入 $u=z/\beta$，乘 $\beta$）。
  - 折叠块：上界证明的逐项拆解（含 $|\tanh|\le1$ 与 $0<\sigma<1$ 的标准界）。
- 前置知识：$\tanh$ 的泰勒展开作为基础记号内联说明；$\sigma(z)\in(0,1)$ 与 $|\tanh(z)|\le 1$ 的标准界内联。
- 完成检查：
  - 写出 $\beta\tanh(z/\beta)$ 在 $|z|\ll\beta$ 时的局部展开式；
  - 解释为什么 $\beta_1,\beta_2\to\infty$ 时 SiTU-GLU 收敛到 SwiGLU；
  - 指出若只套门支、值支不套，上界 100 是否成立。
- 过渡：知道"近原点像 SwiGLU、远点饱和"两件事同时成立，但 K3 为什么选 softcap 而不是直接 clip——下一章对比 softcap 与 hard clamping 的梯度。

### S4：为什么是 softcap 而不是 clip——饱和区里梯度差别

- 主要教学问题：softcap 与 hard clamping 在饱和区的梯度有什么差别？为什么这个差别让 K3 选 softcap？
- 对应范围：Q4。K4（softcap vs hard clamping 对比）。
- 正文要点：
  - hard clamping 定义 F8：$\mathrm{clip}(x,c)=\min(\max(x,-c),c)$，把 $x$ 钳到 $[-c,c]$。
  - softcap 定义 F2：$\beta\tanh(x/\beta)$，把 $x$ 平滑压到 $(-\beta,\beta)$，上界严格不到 $\pm\beta$。
  - 输出对比：$|x|\gg c$ 时 clip 输出 $\pm c$，softcap 输出 $\to\pm\beta$（设 $c=\beta$）；远看类似。
  - 梯度对比 F7/F9：
    - clip 的导数：$|x|<c$ 时为 1，$|x|>c$ 时为 0（边界处不可导）。**饱和后梯度严格为 0**。
    - softcap 的导数：$\frac{d}{dx}[\beta\tanh(x/\beta)]=1-\tanh^2(x/\beta)$。$|x|\ll\beta$ 时近似 1，$|x|\gg\beta$ 时指数衰减 $\sim 4e^{-2|x|/\beta}\to 0$，但**严格非零**（在有限 $x$ 处）。
  - K3 §B 末段原文（C8）："the smooth cap preserves nonzero gradients away from saturation boundaries, which we find to give better training behavior"——饱和区里梯度仍能传，模型不至于在饱和后完全失能。
  - 边界澄清（误解 5）："away from saturation boundaries" 不是"饱和后梯度恒非零"，而是"远离饱和边界处非零"；进入深饱和后梯度指数衰减但仍非零，不是严格 0。
- 讲解材料及职责：
  - 对照表（A14）：clip 与 softcap 的"输出""梯度""$|x|\to\infty$ 时梯度行为"三项对比。
  - 折叠块：$\tanh$ 的指数渐近形式推导（$\tanh z = 1 - 2/(e^{2z}+1)$，$z\to\infty$ 时 $1 - 2e^{-2z}$，导数 $4e^{-2z}$），代入 $z=x/\beta$。
- 前置知识：clip 与 $\tanh'$ 内联一行定义。
- 完成检查：
  - 写出 clip 与 softcap 在 $|x|>c$（或 $|x|>\beta$）时的导数；
  - 解释为什么"饱和后梯度仍非零"对训练有利；
  - 指出 K3 §B 末段"better training behavior"是经验陈述不是对照实验。
- 过渡：理解了 softcap 的梯度优势，最后一章说清 SiTU-GLU 在 K3 中的使用位置与不解决的问题。

### S5：在 K3 中的使用位置与不解决的问题

- 主要教学问题：SiTU-GLU 在 K3 哪里用、不解决什么？
- 对应范围：Q5。K5（使用位置与边界）。
- 正文要点：
  - 使用位置（C9/C10）：
    - Stable LatentMoE 路由分支 FFN：抑制 §2.3 开头描述的"四连矩阵相乘 + 2.8T 规模"激活爆炸。
    - Dense FFN：K2→K3 对比表（§4）显示激活函数从 SwiGLU 整体换成 SiTU-GLU。
  - SiTU-GLU 在 Stable LatentMoE 三件套中的角色：RMSNorm（前置于 up-projection）+ SiTU-GLU（抑制激活爆炸）+ Quantile Balancing（负载均衡）。SiTU-GLU 只负责第二项。
  - 不解决的问题（误解 2 / §2.3.2 末段）：
    - 不解决 Quantile Balancing 的负载均衡（§2.3.3 独立处理）。
    - 不解决 MLA 注意力内部的稳定性。
    - 不保证训练损失一定下降——§2.3.2 末段只承诺"preserves the local response of SwiGLU while controlling both factors"，是结构上的有界性保证，不是性能保证。
    - 不解决 low-precision 算术的全部问题，只是降低溢出风险。
  - 误解处理（收尾）：把 scope §2.6 的 5 条误解中尚未在正文处理的收尾说明：
    - "SiTU-GLU = clip 输出"（S4 已处理）；
    - "SiTU-GLU 一定更稳"（本节）；
    - "$\beta_1,\beta_2$ 普适最优"（S3 已处理）；
    - "只套门支就够"（S3 已处理）；
    - "softcap 等价于 hard clamping"（S4 已处理）。
- 讲解材料及职责：
  - 对照表（A14）：Stable LatentMoE 三件套各自负责什么，SiTU-GLU 只占一行。
  - K2→K3 对比表激活函数一行（N2）。
- 前置知识：无新前置。
- 完成检查：
  - 说出 SiTU-GLU 在 K3 的两个使用位置；
  - 指出 SiTU-GLU 在 Stable LatentMoE 三件套中只负责哪一项；
  - 指出 SiTU-GLU 不解决的一类问题（举一个非 QB、非 MLA 的例子也可）。
- 过渡：文末来源与教学说明。

## 3. 讲解顺序

S1（动机/为什么需要给 SwiGLU 加上界）→ S2（公式 + 手算，是什么）→ S3（局部展开 + 上界证明 + 极限行为，为什么同时成立）→ S4（softcap vs hard clamping 梯度对比，为什么是 softcap）→ S5（使用位置与边界）。先为什么再是什么；$\tanh,\sigma,\odot,W x$、hard clamping、$O(\cdot)$ 在首次用到处内联一行定义；softcap 函数在 S1 引入；局部展开在 S3 用到时展开；clip 在 S4 用到时引入。

## 4. 贯穿例子

主线例子：标量 $x$（设两支 pre-act 均为 $x$ 以隔离函数行为，明确标为教学简化），$\beta_1=4,\beta_2=25$。
- S2 首次出现：算 $x=0,10,100$ 的 SiTU-GLU 输出，并与 SwiGLU 同输入对比。
- S3 复用：用 $x=0.5$（近原点）验证 $\beta\tanh(x/\beta)\approx x$ 一阶等价。
- S4 复用：用 $x=100$（饱和区）对比 clip 与 softcap 的梯度（clip 梯度 0、softcap 梯度 $\sim 4e^{-50}$ 非零但极小）。
所有数字便于手算或用 `math.tanh/math.exp` 复算，保留四位有效数字。

## 5. 正文与折叠块分工

正文必有：F1 公式与符号、激活爆炸动机、F3 局部展开结论、F4 上界结论、F5 极限行为、F6 SwiGLU 公式引用、F7 $\tanh'$、F8 clip 定义、F9 clip 导数、N1 $\beta_1\beta_2=100$、N2 K2→K3 对比、C8 K3 §B 末段原句、Q1–Q5 完成答案。

折叠块：
- S2 中间点 $x=2,50$ 的完整手算（验证过渡平滑）。
- S3 $\beta\tanh(z/\beta)$ 的泰勒展开推导。
- S3 上界证明的逐项拆解。
- S4 $\tanh$ 指数渐近形式推导。

折叠块全收起时，正文仍答全 Q1–Q5。

## 6. 范围与证据约束

全部内容来自 scope.md 已纳入范围与 evidence.md 已确认论断。无未消歧项。教学数字均标记。无证据不足论断进入生产。
