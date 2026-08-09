# GLU 术语表

全文术语与符号首次出现位置及含义。全文含义保持一致。

| 术语/符号 | 首现位置 | 定义/含义 |
|---|---|---|
| GLU（Gated Linear Unit，门控线性单元） | S1 | 一种用 sigmoid 门逐元素缩放线性投影的神经网络单元。 |
| $\sigma$（sigmoid） | S1 首次提及 | $\sigma(z)=1/(1+e^{-z})$，把任意实数压到 $(0,1)$。基础记号，内联定义。 |
| $\otimes$（逐元素/Hadamard 乘积） | S1 | 两个同形向量逐位相乘，结果同形。基础记号。 |
| $xW+b$（线性/仿射变换） | S1 | 输入向量乘权重矩阵加偏置。基础记号。 |
| 值分支 | S2 | GLU 中不经 sigmoid 的那份线性投影 $\mathbf{X}*\mathbf{W}+\mathbf{b}$（Dauphin 记法）。 |
| 门分支 / 门 | S2 | GLU 中过 sigmoid 的那份 $\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$，逐维度控制放行量。 |
| $*$（Dauphin 中的卷积） | S2 公式 F1 | Dauphin 原文指卷积；在 Transformer/FFN 语境退化为矩阵乘，本文统一记 $xW$。 |
| GTU（Gated Tanh Unit） | S3 | $\tanh(\mathbf{X})\otimes\sigma(\mathbf{X})$，梯度含 $\tanh'$ 缩放项，作为 GLU 对比项。 |
| $\tanh'$ | S3 | $\tanh$ 的导数，随 $\lvert x\rvert$ 增大趋零。 |
| $\sigma'$ | S3 | sigmoid 导数 $\sigma'(z)=\sigma(z)(1-\sigma(z))$，门饱和时趋零。 |
| 乘性跳连（multiplicative skip connection） | S3 | Dauphin 对 GLU 梯度项 $\nabla\mathbf{X}\otimes\sigma(\mathbf{X})$ 的描述——被门值（非门导数）缩放的近似直通路径。 |
| Bilinear 层 | S4 | GLU 去 sigmoid 的退化形式 $(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes(\mathbf{X}*\mathbf{V}+\mathbf{c})$（Dauphin §5.3，归因 Mnih & Hinton 2007）。 |
| ReGLU / GEGLU / SwiGLU | S4 | 把 GLU 的 sigmoid 换成 ReLU / GELU / Swish 的变体（Shazeer 2020 §2）。 |
| Swish / SiLU | S4 | $\mathrm{Swish}_\beta(x)=x\cdot\sigma(\beta x)$，作为激活名出现，不展开。 |
| GELU | S4 | 另一种激活函数，作为名出现，不展开。 |
| ReLU | S4 | $\max(0,x)$，作为名出现。 |
| Dauphin 记法 | S4 | $\sigma$ 在 $V$ 分支、$W$ 为值分支（F1）。 |
| Shazeer 记法 | S4 | $\sigma$/激活在 $W$ 分支、$V$ 为值分支（F6/F7）。与 Dauphin 差 $W\leftrightarrow V$。 |
| $d_{ff}$（FFN 隐藏维） | S4 | Transformer FFN 中间层维度。 |
| $d_{model}$ | S4 | Transformer 模型维度（仅 Shazeer 实验条件中提及）。 |
| divine benevolence | S5 | Shazeer §4 结语原句，表示对 GLU 变体有效未给理论解释。 |
| log-perplexity（对数困惑度） | S5 | 语言模型评价指标，越低越好（Shazeer Table 1）。 |

无术语漂移。$W,V$ 在 Dauphin 与 Shazeer 间的角色差异在 S4 显式说明。
