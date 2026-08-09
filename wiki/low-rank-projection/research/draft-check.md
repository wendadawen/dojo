# 低秩分解初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成（2026-08-09）

## 大纲落实

- 章节结构：S1 大矩阵为什么贵 → S2 矩阵的秩 → S3 SVD 与最优低秩近似 → S4 LoRA → S5 MLA → S6 适用边界 → 来源与教学说明（与 outline.md 一致）
- 学习目标：Q1-Q5 在页面开头 learning-goals 组件列出，与 scope.md 一致
- 前置知识：矩阵乘法（无概念页，正文一句话说明"$Wx$ 表示用矩阵 $W$ 对向量 $x$ 做线性变换"，占位）；标准注意力 K/V 与 KV cache（链接 standard-attention）；MLA 完整机制（链接 mla）；KDA 门控（链接 kda）
- 贯穿例子：维度算术主线（$mn \to r(m+n) \to$ SVD 误差 $\to d^2 \to 2dr \to 2n_h d_h \to d_c$）+ 局部 3×3 对角矩阵手算
- 误解和边界：S6 专章处理（无损误解、r 越大越好误解、任何矩阵都能压缩误解）；scope.md 三条误解均落实
- 过渡：每章末尾有过渡句指向下一章的逻辑缺口

## 学习目标闭环

- Q1（为什么大矩阵可用两小矩阵近似、何时误差小）：S2 正文章节完整回答（秩定义、$W=AB \Rightarrow \mathrm{rank} \le r$、参数对比、满秩无法无损）；S3 补充误差条件。正文完整，不依赖折叠块。
- Q2（SVD 截断如何最优、误差由什么决定）：S3 正文章节完整回答（SVD 形式、截断 SVD、Eckart-Young 定理结论、两个误差公式）。手算例子在折叠块但正文已有公式和结论。
- Q3（LoRA 如何减参数、省哪些）：S4 正文章节完整回答（$\Delta W = BA$、$h = W_0 x + BAx$、$d^2 \to 2dr$、初始化、不是近似 $W_0$）。
- Q4（MLA 如何减 KV cache、缓存什么）：S5 正文章节完整回答（$c_t^{KV} = W^{DKV} h_t$、只缓存 $c_t^{KV}$、K/V 由上投影重建、$2n_h d_h \to d_c$、有损性）。
- Q5（适用边界）：S6 正文章节完整回答（奇异值平坦失效、LoRA 假设、MLA 有损、K3 full-rank 教训）。

## 代码运行

- 无可运行代码。本页不含可运行代码块（机制用手算例子和公式说明即可，不需要代码验证）。
- 手算例子已用 Python 独立验证（见下方机械检查），数值与页面一致。

## 机械检查

命令与结果：

```
$ python3 .dojo/scripts/validate.py wiki/low-rank-projection/index.html
validation ok: wiki/low-rank-projection/index.html

$ python3 .dojo/scripts/validate.py wiki/low-rank-projection/overview.html
validation ok: wiki/low-rank-projection/overview.html
```

退出码均为 0。无占位符、无组件标记、无重复 id、无断链。

手算数值独立验证（Python）：

```
W = diag(3, 1, 0.5), sigma = [3, 1, 0.5]
Rank-1: F-error = sqrt(1^2 + 0.5^2) = sqrt(1.25) = 1.118034, spectral = 1
Rank-2: F-error = sqrt(0.5^2) = 0.5, spectral = 0.5

LoRA d=12288, r=2: full=150994944, lora=49152, ratio=0.0326%
LoRA d=4096, r=8: full=16777216, lora=65536, ratio=0.3906% (完成检查题)

MLA: MHA cache = 2*128*128 = 32768, MLA latent = d_c = 512, ratio = 1/64
Table 7: Small MoE 110.6K→15.6K (减 85.9%), Large MoE 860.2K→34.6K (减 96.0%)

完成检查 diag(5,2,0.1) rank-1: F-error = sqrt(4.01) = 2.002498, spectral = 2
```

页面中所有数字与上述验证一致。

## 公式渲染与交互

- KaTeX：页面使用 $...$（行内）和 $$...$$（行间）定界符，外壳脚本 auto-render 自动渲染。公式符号与 glossary.md 一致。
- 折叠块：3 个 details（Eckart-Young 直觉补充、3×3 手算例子、均为补充非核心）；全部收起时正文仍完整回答 Q1-Q5。
- 目录：外壳脚本自动从 h2/h3 生成侧边目录。
- 链接：standard-attention、mla、kda 三个概念页链接均已验证存在。

## 写作偏差

无。未改变大纲结构、未增删学习目标、未更换贯穿例子、未把正文必要内容移入折叠块。所有论断/公式/数字附 C/F/N 编号，与 evidence.md 一致。
