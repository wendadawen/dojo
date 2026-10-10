# SiTU-GLU

- 描述：SiTU-GLU 是 SwiGLU 的一种变体，把门分支的线性因子和值分支各自换成一个平滑饱和的 softcap，让两支都落到固定上界内，从而给二者的乘积一个固定上界。
- 摘要：$\mathrm{SiTU\text{-}GLU}(x)=\big[\beta_1\tanh(W_g x/\beta_1)\odot\sigma(W_g x)\big]\odot\big[\beta_2\tanh(W_u x/\beta_2)\big]$，把 SwiGLU 里两个无界的线性因子各自换成 $\beta\tanh(x/\beta)$，输出逐坐标落在 $[-\beta_1\beta_2,\beta_1\beta_2]$ 内。
- 主题：模型结构
- 标签：模型架构

## 长页

- 开头｜SiTU-GLU 是什么、为什么需要、大致怎么做，只作概览，不讲机制，链接 [SwiGLU](wiki/swiglu)、[GLU](wiki/glu)｜220｜paper/k3-src/2-model-architecture.tex L253-L271；paper/k3-src/appendix.tex L7-L13
- ### 为什么 SwiGLU 需要上界｜SwiGLU 两支无界、大坐标可能出激活外点（个别坐标异常大）与低精度溢出；GLU 门有界但丢 Swish 正侧线性区；K3 路由分支在 2.8 万亿规模下放大｜280｜paper/k3-src/2-model-architecture.tex L218-L221；paper/k3-src/2-model-architecture.tex L253-L258
- ### SiTU-GLU 的公式｜softcap 定义；公式与各符号；门支只换线性因子、留 sigmoid，值支同样换，逐元素相乘｜300｜paper/k3-src/2-model-architecture.tex L262-L271；paper/k3-src/appendix.tex L10-L13
- ### 手算一遍｜两支线性输出取同一标量，代入 β1=4、β2=25 算三点的输出并与 SwiGLU 对比，过程折叠｜100｜自行构造
- ### 近原点像 SwiGLU、正侧饱和到上界｜局部展开（推导折叠）使近原点一阶等于 SwiGLU；输出上界 β1β2=100；β→∞ 收敛到 SwiGLU｜320｜paper/k3-src/appendix.tex L15-L31；paper/k3-src/2-model-architecture.tex L248-L251
- ### 适用条件与边界｜β1=4、β2=25 是 K3 的设定值；局部展开要 |z|≪β；上界是乘积的逐坐标上界；与硬截断不同，平滑饱和在远离饱和边界处仍保留非零梯度｜280｜paper/k3-src/2-model-architecture.tex L270-L271；paper/k3-src/appendix.tex L24-L33

## 短页

- ### 两个因子都加上界的前馈激活｜是什么：SiTU-GLU 给 SwiGLU 的两个乘性因子各加一个上界；一句交代 GLU、SwiGLU、Swish、前馈层｜280｜paper/k3-src/2-model-architecture.tex L253-L254；paper/k3-src/2-model-architecture.tex L262-L271
- ### SwiGLU 的两个无界因子｜为什么：SwiGLU 两支无界，大坐标可能出激活外点（个别坐标值异常大）与低精度溢出风险；GLU 门有界但丢 Swish 正侧近似线性区｜280｜paper/k3-src/2-model-architecture.tex L257-L258；paper/k3-src/appendix.tex L7-L8
- ### 用 softcap 分别饱和两个因子｜怎么做：写出 softcap 与公式，说明两支各套一次、输出落在 ±β1β2 内，近原点仍近似线性｜300｜paper/k3-src/2-model-architecture.tex L262-L271；paper/k3-src/appendix.tex L15-L31
