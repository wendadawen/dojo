# evidence.md：深度可分离卷积

来源缩写：
- [MN] Howard et al., MobileNets, arXiv:1704.04861v1 (2017)
- [GLM] transformers main, modeling_glm5_next.py Glm5NextTextLinearAttention L603-680
- [实测] research/concept_probes.out 探针 C

## C 论断

- C1 标准卷积一步完成滤波与组合；拆开是「分滤波层与组合层」：[MN] §3.1 原文「A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers」。已确认
- C2 depthwise 逐通道单滤波器（$D_K\times D_K\times1$）；pointwise 为 $1\times1$ 跨通道线性组合：[MN] §3.1。已确认
- C3 $3\times3$ 深度可分离比标准卷积省 8~9 倍计算、精度小降：[MN] §3.1 原句。已确认
- C4 MobileNet 第一层是普通卷积，其余为深度可分离（含 BN+ReLU）：[MN] §3.2/图 1。已确认
- C5 GLM KDA 的 conv1d：in=out=conv_dim=24576、kernel=4、groups=24576、bias=False、因果（右 padding 后取尾部）；q/k/v 各自独立卷积（checkpoint 拆为 q/k/v_conv1d）：[GLM] L603-615 与 GLM 数据流页张量头交叉验证。已确认
- C6 KDA 无 pointwise 配对，跨通道由 q/k/v 投影承担：[GLM] L642-649（投影在前、卷积在后于拼接的 qkv 上）；该论断的「跨通道混合由投影承担」部分为结构解读，页面标注为解读

## F 公式

- F1 标准卷积定义 $\mathbf{G}_{k,l,n}=\sum_{i,j,m}\mathbf{K}_{i,j,m,n}\mathbf{F}_{k+i-1,l+j-1,m}$：[MN] Eq.(1)。已确认
- F2 标准卷积成本 $D_K D_K M N D_F D_F$：[MN] Eq.(2)。已确认
- F3 depthwise 定义与成本：[MN] Eq.(3)(4)。已确认
- F4 深度可分离总成本：[MN] Eq.(5)。已确认
- F5 缩减比 $1/N+1/D_K^2$：[MN] §3.1 推导（Eq.5/Eq.2 逐项相除）。已确认；构造数字例（$D_K=3,M=N=64,D_F=16$）实算比值 0.126736=公式值、7.9 倍：[实测] C3。已确认
- F6 因果 1D depthwise 前向：手算 -3.5 与 torch 输出一致；groups=2 两通道互不干扰：[实测] C1/C2。已确认

## N 数字

- N1 8~9 倍（3×3）：[MN] §3.1。已确认
- N2 构造例 9,437,184 vs 1,196,032 次乘加（depthwise 147,456 + pointwise 1,048,576）：[实测] C3。构造示例
- N3 GLM：conv_dim=qkv_dim×3=8192×3=24576、kernel=4、checkpoint 拆三个 [8192,1,4]：[GLM] 与张量头。已确认

## 冲突与缺口

- C6 后半句（投影承担跨通道混合）为解读非源码直陈：页面标注
- [MN] 的「small reduction in accuracy」无具体数字（正文层面）：引用时保持定性
