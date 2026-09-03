# evidence.md：iHC

来源缩写：
- [README] Tencent-Hunyuan/Hy4-preview 官方 README（GitHub main，2026-08-28 发布；HF 同步），Model Introduction 一节与 Model Specifications 表
- [CFG] tencent/Hy4-preview config.json（HF 主检查点）
- [vLLM] vLLM PR #54160 "[Hy4] support Hy4-preview model"（commit b2f685834a645，2026-08-29 合并，腾讯工程师署名）：vllm/models/hy_v4/nvidia/hc.py、vllm/models/hy_v4/nvidia/model.py、vllm/model_executor/layers/hpc/hpc_ihc.py
- [CKPT] tencent/Hy4-preview model.safetensors.index.json 与 shard 头部（本机 2026-09-03 拉取实测）
- [知乎] 《你的deepseek mHC可能不需要"m"》，zhuanlan.zhihu.com/p/2010852389670908320（2026-02-28；Hy4 README「iHC」链接指向此文）
- [HC] Zhu et al., Hyper-Connections, arXiv:2409.19606v3（公式经 hyper-connections 页核对）
- [mHC] Xie et al., mHC: Manifold-Constrained Hyper-Connections, arXiv:2512.24880v2（结论经 hyper-connections 页核对）
- [实测] 本页 research/ 下的探针脚本输出（写作阶段生成）

## C 论断

- C1 Hy4 官方对残差通路的表述："The residual pathway uses iHC (identity Hyper-Connections) to expand inter-layer information flow"，规格表 Residual Streams: 4。[README]。已确认
- C2 iHC 用 hc_mult 条并行残差通道替换单一残差流；每个子块把通道收缩成一份隐状态（pre）、运行子块、再把结果散射回通道（post）；末端 head 在模型输出 norm 前合并通道。[vLLM] hc.py 模块 docstring。已确认
- C3 iHC 相对 mHC 没有混合矩阵："Unlike mHC there is no comb matrix, so each output channel only needs its own residual channel and the whole thing is one fused multiply-add per element"。[vLLM] hpc_ihc.py HpcIHCPost docstring。已确认
- C4 命名分歧：官方 README 写 identity Hyper-Connections；vLLM 实现 docstring 写 independent Hyper-Connections（hc.py 与 hpc_ihc.py 两处）。同一机制。[README] 对照 [vLLM]。已确认
- C5 每个解码层有两个 iHC 边界（hc_attn_layer、hc_mlp_layer），参数各自独立；第 1 层（dense FFN）同样有。[vLLM] model.py HYV4DecoderLayer.__init__；[CKPT] 78 层 × 2 站点张量。已确认
- C6 iHC 前向顺序：pre 合并 → input_layernorm → self_attn → post 写回 → pre 合并 → post_attention_layernorm → mlp → post 写回；"Under iHC the residual is carried inside hidden_states"。[vLLM] model.py _forward_ihc。已确认
- C7 模型入口 [n, d] 经 prepare_input 广播复制成 [n, hc, d]（unsqueeze + repeat）；主干末端 hc_head 合并后才过 self.norm。[vLLM] hc.py _prepare_input_to_3d、model.py HYV4Model（hidden_states = self.hc_head(...) 后 return self.norm(...)）。已确认
- C8 初始化：hc_scale 置 0.01；hc_base 前 n 个分量 = -log(n-1)，后 n 个 = 0；head 同构（base 全 -log(n-1)）。n=4 时 -log(3)。[vLLM] hc.py reset_parameters（Pre 与 Head 两处）。已确认
- C9 门控计算在 float32 中进行（forward 内 .float()）；hc_* 权重为 float32，FP8 检查点中经 modules_to_not_convert 排除在量化外。[vLLM] hc.py forward、hpc_ihc.py 模块 docstring（"hc_fn weights stay float32 (the checkpoint keeps them out of fp8 quantization via modules_to_not_convert)"）。已确认
- C10 Hy4 配置：enable_ihc=true、hc_mult=4、hc_magnitude=2.0、hc_eps=1e-6、hidden_size=6144、num_hidden_layers=78、rms_norm_eps=1e-5、mlp_layer_types=1 dense + 77 sparse。[CFG]。已确认
- C11 checkpoint 张量结构：78 层 × {hc_attn_layer, hc_mlp_layer} × hc_pre.{hc_fn[8,24576], hc_scale[2], hc_base[8]}，全部 F32；全局 model.hc_head.{hc_head_fn[4,24576], hc_head_scale[1], hc_head_base[4]}，F32；共 471 个张量。[CKPT]。已确认
- C12 iHC 总参数量 30,770,717 = 156 × (8×24576 + 2 + 8) + (4×24576 + 1 + 4)；占 770B 主干的约 0.004%。[CKPT] 逐项计算。已确认
- C13 MTP 草稿层不含 iHC：checkpoint 的 27 个 model.mtp_layers.* 张量中无任何 hc 张量（结构为 eh_proj/hnorm/enorm + 单流 layer norm）。[CKPT]。已确认
- C14 mHC 学出的单层混合矩阵接近单位阵：对角约 0.96、非对角约 0.01；深度≥10 的累积乘积坍缩为全 0.25 的均匀混合矩阵。[知乎]（其实测）。已确认（标注社区实验）
- C15 坍缩机理（该文论证）：双随机矩阵满足一致正性条件（元素有正下界 δ）时，Dobrushin 遍历系数 τ ≤ 1-dδ，连乘几何衰减，行趋于一致，收敛到 (1/d)·11^T；纯置换或可约序列不满足条件、不坍缩，但 Sinkhorn 输出通常严格正。[知乎]。已确认（标注：该文给出的论证，本页用构造实验复现现象）
- C16 实验排序：Identity HC > mHC > mHC lite > mHC orthogonal；条件 Qwen3 1.7B 与 8B dense、150B tokens、from scratch。[知乎]。已确认（标注社区实验条件）
- C17 Sinkhorn 近似误差：该文实测 20 步迭代后行和标准差 0.12；并引 mHC-lite 的 27.9% 输入 relative range ≥ 10^13 时列和偏差可达 100%。[知乎]。已确认（标注其实验条件）
- C18 流语义一致性：identity 相当于所有层用同一个恒等置换，流 i 永远在位置 i，读写门无需追踪流重排；I^L = I 不坍缩不混乱。[知乎]。已确认（标注该文论证）
- C19 HPC 融合（扩展内容）：eager 路径 pre/post/head 分别发出 20/5/15 个 kernel，HPC 各融合为 1 个；约束 VLLM_ENABLE_HPC_OPS=1、sm100/sm103、hc_mult=4、hidden ∈ {4096, 6144}。[vLLM] hpc_ihc.py 模块 docstring 与 _SUPPORTED_* 常量。已确认
- C20 跨层 post+pre 融合（HpcIHCPostPre）在参考实现中存在但 vLLM 未移植（TODO）。[vLLM] hpc_ihc.py NOTE。已确认（扩展）
- C21 GLM-5.3-Flash mHC 实现细节（第 5 章对照表用）：hc_mult=4、每层两站点（attn_hc/ffn_hc）、fn=[24,16384]（读 $n$ + 写 $n$ + 混合 $n^2$ 共 24 个 logits）、pre=$\sigma$+$\varepsilon$、post=$2\sigma$（无 $\varepsilon$）、comb=softmax+$\varepsilon$ 后 Sinkhorn 20 步、末端 hc_head 无权重均值、残差通路总参数 35,391,870（占 321.32B 的 0.011%）。来源：wiki/hyper-connections/research/evidence.md C11/C13/F8（transformers main, modeling_glm5_next.py L219-302），该页已核对。已确认（转引自超连接页）

## F 公式

- F1 pre 块前向（[vLLM] hc.py HYV4HCPreLayer.forward，逐 token）：
  $$x^{flat}=\mathrm{flatten}(x)\in\mathbb{R}^{nd},\quad r=\frac{1}{\sqrt{\mathrm{mean}\left((x^{flat})^2\right)+\epsilon_{rms}}}$$
  $$\mathrm{mixes}=(x^{flat}W^{T})\cdot r\in\mathbb{R}^{2n}$$
  $$H_{pre}=\sigma(\mathrm{mixes}_{[:n]}\cdot s_{0}+b_{[:n]})+\epsilon_{hc}$$
  $$H_{post}=m\cdot\sigma(\mathrm{mixes}_{[n:]}\cdot s_{1}+b_{[n:]})+\epsilon_{hc}$$
  $$y=\sum_{i=1}^{n}H_{pre,i}\cdot x_{i}$$
  其中 $W\in\mathbb{R}^{2n\times nd}$（hc_fn），$s\in\mathbb{R}^{2}$（hc_scale），$b\in\mathbb{R}^{2n}$（hc_base），$m$ 为幅度（hc_magnitude）。已确认
- F2 post 块前向（[vLLM] hc.py HYV4HCPostLayer.forward）：
  $$\hat{x}_{i}=H_{post,i}\cdot z+x_{i},\quad i=1,\dots,n$$
  $z$ 为子层输出（各流加同一份，仅系数不同）。已确认
- F3 head 块前向（[vLLM] hc.py HYV4HCHeadLayer.forward）：与 F1 同构但投影输出 $n$ 个 logits、无 $H_{post}$：
  $$\mathrm{mixes}_{h}=(x^{flat}W_{h}^{T})\cdot r,\quad H_{head}=\sigma(\mathrm{mixes}_{h}\cdot s_{h}+b_{h})+\epsilon_{hc},\quad y=\sum_{i}H_{head,i}\cdot x_{i}$$
  已确认
- F4 初始化（[vLLM] hc.py reset_parameters）：$s=0.01$（全分量）；$b_{[:n]}=-\ln(n-1)$；$b_{[n:]}=0$；head 的 $b_{h}$ 全取 $-\ln(n-1)$。已确认
- F5 初始行为推导：$n=4$ 时 $H_{pre}\approx\sigma(-\ln 3)=\tfrac{1}{1+3}=0.25$（等权平均），$H_{post}\approx 2\sigma(0)=1$（满幅写回）。由 F1+F4 代入。已确认（推导）
- F6 iHC 单层传播（对齐 [HC] Eq.(2) 记号，$\mathbf{A_r}=\mathbf{I}$）：$\hat{\mathbf{H}}=\mathbf{B}^{\intercal}\mathcal{T}((\mathbf{H}^{\intercal}\mathbf{A_{m}})^{\intercal})+\mathbf{H}$。已确认（形式对应，见 C3/F1-F2 的实现映射）

## N 数字

- N1 4 流、hidden 6144、78 层、幅度 2.0、hc_eps 1e-6、rms_norm_eps 1e-5。[CFG]。已确认
- N2 每站点参数 196,618（8×24576+2+8）；156 站点 + head 98,309 → 总 30,770,717。[CKPT] 实测形状 + 算术。已确认
- N3 hc_fn 形状 [8, 24576]（=2×4 × 4×6144）、hc_head_fn [4, 24576]，均 F32。[CKPT] shard 头。已确认
- N4 mHC 学出的混合矩阵对角 ≈0.96 / 非对角 ≈0.01；累积乘积 ≈ 全 0.25。[知乎]（Qwen3 系列实验观测）。引用时标注实验条件
- N5 Sinkhorn 20 步后行和标准差 0.12。[知乎]（其复现条件）。引用时标注
- N6 eager pre/post/head 的 kernel 数 20 / 5 / 15。[vLLM] hpc_ihc.py docstring。已确认（扩展）
- N7 Hy4 主干 770B 总参数、49B 激活、78 层、1M 上下文、64 注意力头、FFN 中间维 18432、MoE 中间维 2048、256 路由专家 top-8 + 1 共享、词表 120,832。[README] 规格表 + [CFG]。已确认（背景数字，只用于第 5 章定位）

## 冲突与缺口

- iHC 没有正式论文：机制定义以 [vLLM] 官方实现为最高依据（与 [CKPT] 张量结构交叉验证一致）；「为什么 identity」以 [知乎]（Hy4 README 官方引用）为依据，全部标注社区实验性质，不写成官方结论
- [知乎] 作者自述文中部分表格文字由 AI 生成、实验数据为实测：页面对该文只引用数值结论与论证思路，不引用其 AI 生成的叙述文字
- identity vs independent 命名：并列写出（C4），不裁决
- 「为什么是 4 条流」无公开依据：如实标注缺口，不推测
- HC 论文 Eq.(2) 原式与 vLLM 实现的记号对应（$\mathbf{A_m}$↔读门、$\mathbf{B}$↔写门、$\mathbf{A_r}$↔无）为形式对应：页面写作时给出对应关系并注明「实现记号与论文记号的映射」，不声称实现逐字复现论文公式
