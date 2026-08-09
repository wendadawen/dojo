# SwiGLU 术语表

全文术语与符号首次出现位置及含义。全文含义保持一致。

| 术语/符号 | 首现位置 | 定义/含义 |
|---|---|---|
| SwiGLU（Swish-Gated Linear Unit，Swish 门控线性单元） | S1 | GLU 家族中把门分支的 sigmoid 换成 Swish 的变体，Shazeer 2020 §2 Eq.(5) 定义。 |
| GLU（Gated Linear Unit，门控线性单元） | S1 | SwiGLU 的父概念，门分支用 sigmoid。完整讲解见 [GLU 概念页](../../wiki/glu/index.html)。 |
| $\sigma$（sigmoid） | S1 回顾 | $\sigma(z)=1/(1+e^{-z})$，把任意实数压到 $(0,1)$。基础记号，内联定义。 |
| Swish / SiLU | S1 | $\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$。SiLU 是 $\mathrm{Swish}_1$ 的别称。本文以 Swish 为主记法。 |
| $\beta$（Swish 超参） | S1 | Swish 中 sigmoid 的缩放系数。Shazeer §1 与 §3.1 实验固定 $\beta=1$；现代 LLM 部署也固定 $\beta=1$。 |
| $\otimes$（逐元素/Hadamard 乘积） | S1 | 两个同形向量逐位相乘，结果同形。基础记号。 |
| $xW+b$（线性/仿射变换） | S1 | 输入向量乘权重矩阵加偏置。基础记号。 |
| 值分支 | S2 | SwiGLU 中不经激活的那份线性投影 $xV+c$（Shazeer 记法）。 |
| 门分支 / Swish 门 | S2 | SwiGLU 中过 Swish 的那份 $\mathrm{Swish}_\beta(xW+b)$，可输出负值与无界正值。 |
| Shazeer 记法 | S2 | $\sigma$/激活在 $W$ 分支、$V$ 为值分支（F2）。本文主记法。 |
| Dauphin 记法 | S2 | $\sigma$ 在 $V$ 分支、$W$ 为值分支。[GLU 概念页] 主记法。与 Shazeer 差 $W\leftrightarrow V$。 |
| ReGLU / GEGLU | S4 | 把 GLU 的 sigmoid 换成 ReLU / GELU 的变体（Shazeer 2020 §2）。 |
| Bilinear 层 | S4 | GLU 去激活的退化形式 $(xW+b)\otimes(xV+c)$（Dauphin §5.3，归因 Mnih & Hinton 2007）。 |
| ReLU | S4 | $\max(0,x)$，作为名出现，不展开。 |
| GELU | S4 | 另一种激活函数，作为名出现，不展开。 |
| FFN（Feed-Forward Network，前馈网络） | S3 | Transformer 每层后的两层 MLP。基础语境，内联一行定义。 |
| $\mathrm{FFN}_{\mathrm{SwiGLU}}$ | S3 | $(\mathrm{Swish}_1(xW)\otimes xV)W_2$，三矩阵、无偏置（Shazeer §2 Eq.(6)）。 |
| $d_{model}$ | S3 | Transformer 模型维度。 |
| $d_{ff}$（FFN 隐藏维） | S3 | Transformer FFN 中间层维度。基线常取 $4d$；SwiGLU 变体取 $\tfrac23 d_{ff}=\tfrac83 d$。 |
| $\tfrac83 d$ | S3 | LLaMA 风格 SwiGLU FFN 隐藏维。由 $\tfrac23\times 4d$ 推出。 |
| segment-filling | S4 | T5/Shazeer 实验的预训练任务名（Table 1 caption 原词）。 |
| log-perplexity（对数困惑度） | S4 | 语言模型评价指标，越低越好（Shazeer Table 1）。 |
| divine benevolence | S4 | Shazeer §4 结语原句，表示对 GLU 变体有效未给理论解释。 |
| SiTU-GLU | S4 | 对 SwiGLU 加 softcap 有界化的下游改进，K3 §2.3.2 提出。完整讲解见 [SiTU-GLU 概念页](../../wiki/situ-glu/index.html)。 |
| softcap | S4 | $\beta\tanh(\cdot/\beta)$ 形式的平滑有界化算子，SiTU-GLU 用它替代 SwiGLU 的线性因子。 |
| PaLM / LLaMA / Mistral / Qwen / DeepSeek | S4 | 采用 SwiGLU 作为 FFN 激活的主流 LLM。 |

无术语漂移。$W,V$ 在 Shazeer 与 Dauphin 间的角色差异在 S2 显式说明。$\beta$ 在实验与部署中固定为 $1$，在公式中以 $\beta$ 形式给出但取值在 S2 末段锁定。
