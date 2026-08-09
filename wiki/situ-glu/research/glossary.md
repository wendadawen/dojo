# SiTU-GLU 术语表

全文术语与符号首次出现位置及含义。全文含义保持一致。

| 术语/符号 | 首现位置 | 定义/含义 |
|---|---|---|
| SiTU-GLU（Sigmoid Tanh Unit GLU） | 页面标题/S1 | K3 §2.3.2 提出的 GLU 变体，把 SwiGLU 的门支线性因子与值支同时套 softcap，使两乘性因子平滑饱和、乘积有界。 |
| GLU（Gated Linear Unit） | S1 首次提及 | 门控线性单元家族基础。本文不展开，引用 [GLU 概念页](../../wiki/glu/index.html)。 |
| SwiGLU | S1 | Shazeer 2020 的 GLU 变体，公式 $\mathrm{SwiGLU}(x)=(W_g x\cdot\sigma(W_g x))\odot(W_u x)$。SiTU-GLU 的修改起点。引用 [GLU 概念页](../../wiki/glu/index.html)。 |
| $\sigma$（sigmoid） | S1 | $\sigma(z)=1/(1+e^{-z})$，把任意实数压到 $(0,1)$。基础记号，内联定义。 |
| $\tanh$ | S1 | 双曲正切，把任意实数压到 $(-1,1)$，导数 $1-\tanh^2$。基础记号。 |
| $\odot$（逐元素乘积） | S1 | 两个同形张量逐位相乘，结果同形。基础记号。与 GLU 概念页一致。 |
| $W_g, W_u$ | S2 公式 F1 | SiTU-GLU 的门分支、值分支权重矩阵。沿用 K3 §2.3.2 Eq.(12) 记法。 |
| $\beta_1$ | S1/S2 | 门支的 soft-cap 超参。K3 设定 $\beta_1=4$。 |
| $\beta_2$ | S1/S2 | 值支的 soft-cap 超参。K3 设定 $\beta_2=25$。 |
| softcap（平滑限幅） | S1 | $\mathrm{softcap}(x,\beta)=\beta\tanh(x/\beta)$。K3 §2.3.2 Eq.(12) 前定义。把 $x$ 平滑压到 $(-\beta,\beta)$。 |
| 门支 / 门分支 | S2 | SiTU-GLU 公式第一个方括号部分 $\beta_1\tanh(W_g x/\beta_1)\odot\sigma(W_g x)$。 |
| 值支 / 值分支 | S2 | SiTU-GLU 公式第二个方括号部分 $\beta_2\tanh(W_u x/\beta_2)$。 |
| 线性因子 | S1/S2 | Swish 门 $x\cdot\sigma(x)$ 中的 $x$ 部分（被 softcap 替换的就是这部分），以及值支的 $x$。 |
| softcap 局部展开 | S3 | $\beta\tanh(z/\beta)=z+O((z/\beta)^3)$，$|z|\ll\beta$ 时一阶等于 $z$。K3 §B Eq.(18)。 |
| 上界 / 输出上界 | S3 | $|\mathrm{SiTU\text{-}GLU}(x)|\le\beta_1\beta_2=100$。K3 §B Eq.(19)。 |
| 极限行为 | S3 | $\beta_1,\beta_2\to\infty$ 时 SiTU-GLU 逐点收敛到 SwiGLU。K3 §B Eq.(18) 后。 |
| 一阶等价 | S3 | 近原点 SiTU-GLU 与 SwiGLU 在局部展开到一阶相同。 |
| hard clamping（硬裁剪） | S4 | $\mathrm{clip}(x,c)=\min(\max(x,-c),c)$，把 $x$ 钳到 $[-c,c]$。基础记号。 |
| clip 导数 | S4 | $\frac{d}{dx}\mathrm{clip}(x,c)=1$ 当 $|x|<c$，$=0$ 当 $|x|>c$（边界不可导）。 |
| $\tanh'$（$\tanh$ 的导数） | S4 | $1-\tanh^2(z)$，$|z|\to\infty$ 时指数衰减趋于 0，但不严格为 0。 |
| $\tanh$ 指数渐近 | S4 折叠块 | $\tanh z = 1 - 2/(e^{2z}+1)$，$z\to\infty$ 时 $1 - 2e^{-2z}$；导数 $4e^{-2z}$。代入 $z=x/\beta$ 用于 softcap 饱和区分析。 |
| Stable LatentMoE | S5 | K3 §2.3 的路由分支 MoE 结构。SiTU-GLU 在其路由 expert FFN 中使用。 |
| Dense FFN | S5 | K3 稠密 FFN（非 MoE）。SiTU-GLU 也在此使用，K2→K3 对比表行。 |
| Quantile Balancing（QB） | S5 | K3 §2.3.3 的负载均衡算法。与 SiTU-GLU 并列的 Stable LatentMoE 三件套之一，SiTU-GLU 不解决它。 |
| Stable LatentMoE 三件套 | S5 | RMSNorm（前置 up-projection）+ SiTU-GLU（抑制激活爆炸）+ QB（负载均衡）。 |
| 渐近记号 $O(\cdot)$ | S3 | 标准大 O 记号，表示同阶量级。基础记号。 |
| low-precision arithmetic | S1 | 低精度算术（FP8/INT8）。激活爆炸在低精度下加剧溢出风险。本文只用语境，不展开。 |
| 激活爆炸 / activation explosion | S1 | K3 §2.3 开头描述的 routed branch 内部激活数值过大现象，由四连矩阵相乘与 2.8T 规模放大。 |
| better training behavior | S4/S5 | K3 §B 末段对 softcap 相对 hard clamping 的经验陈述。本文标为经验陈述、非对照实验。 |

无术语漂移。$W_g,W_u$ 在本文与 K3 §2.3.2 Eq.(12) 一致；$\beta_1,\beta_2$ 在全文恒为 4 与 25；$\tanh$ 与 $\tanh'$ 在 S3/S4 含义一致。
