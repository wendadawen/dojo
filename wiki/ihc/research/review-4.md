<!-- review-meta
round: 4
page: wiki/ihc/index.html
reviewed_content_sha256: 8fd1acbe1c27241b
-->
# 恒等超连接（iHC）审查记录（第 4 轮）

- 页面版本：c8f931e85188bf8d0f9577129cc3392df46b641f（`git hash-object wiki/ihc/index.html`）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取 research/ 下任何规划、修复或前序审查记录）
- 已完整阅读章节：核心问题 / 常见误解 / 1. iHC 拿掉了混合矩阵（1.1 从三映射到一刀、1.2 三条路线怎么传旧状态、本章问题）/ 2. 一个子块的完整路径（2.1 从入口复制到末端合并、2.2 第一幕：初始化状态、本章问题）/ 3. 读写门从哪来、从哪里出发（3.1 门控计算、3.2 初始化、3.3 第二幕：训练后的门、本章问题）/ 4. 把混合矩阵钉死在 $I$ 的理由（4.1 学出的混合接近单位阵、4.2 双随机矩阵连乘的坍缩、4.3 identity 还买到什么、本章问题）/ 5. Hy4 里的 iHC：配置、权重与代价（5.1 配置与架构位置、5.2 checkpoint 里的张量与参数量、5.3 与 GLM-5.3-Flash 的 mHC 实现对照、本章问题）/ 来源与范围说明（C/F/N 三张清单、构造示例、简化条件）。两段代码块与两段「预期输出」已逐字执行比对。

## 问题

- [轻微·表述] 引言第 4 段（承接「……把『读多少、写多少』留给训练。」之后的一段）：「本页先在家族里定位 iHC 改了哪一刀，再走完一个子块的完整数据流，拆开门控计算与初始化，论证为什么把混合矩阵钉死在 $I$，最后落到 Hy4 的配置、权重与代价。」｜引文依据：check.md 2.2 第 12 条把「本页将…」明列为须排除的元话语，并单列「以『本页』为主语的自我指代」；该句正是「本页先…再…最后…」的结构通报，与页面级「核心问题」逐条重复。旁证：全库 98 个概念页中只有 2 页出现「本页先」（wiki/ihc、wiki/mixed-precision-quant），「本页将」0 页，说明这不是页面模板套语。｜修复要求：删去这段结构通报，或改写为不通报「本页将做什么」的直述句。｜修复：｜复验：

## 来源核对（无问题项，逐条留档）

- 官方 README（Tencent-Hunyuan/Hy4-preview，`gh api .../readme`，pushed_at 2026-08-28T12:48:34Z）原文：「The residual pathway uses [iHC (identity Hyper-Connections)](https://zhuanlan.zhihu.com/p/2010852389670908320) to expand inter-layer information flow.」；规格表「| Residual Streams | 4 |」——支持 C1、[C7]/2.1 的 4 流与 README 指向知乎文章。
- config.json（tencent/Hy4-preview，`lastModified` 2026-08-28T14:58:32Z）：`enable_ihc=true`、`hc_mult=4`、`hc_magnitude=2.0`、`hc_eps=1e-06`、`hidden_size=6144`、`num_hidden_layers=78`、`rms_norm_eps=1e-05`、`mlp_layer_types=['dense', 77×'sparse']`、`n_routed_experts=256`、`n_shared_experts=1`、`num_experts_per_tok=8`、`moe_intermediate_size=2048`、`intermediate_size=18432`、`vocab_size=120832`、`num_attention_heads=64`、`max_position_embeddings=1048576`——支持 C10、N1、N7 与 5.1 配置表全部取值。
- vLLM PR #54160 / commit b2f685834a6456197e7033966fdef52a23f1abcd（`hg api repos/vllm-project/vllm/commits/b2f685834a645`："[Hy4] support Hy4-preview model (#54160)"，作者 Cheng Jiang，2026-08-29）：
  - `hc.py` 模块 docstring：「Each decoder sub-block reduces the channels to one hidden state (``HYV4HCPreLayer``), runs the sub-block, then scatters the result back over the channels (``HYV4HCPostLayer``). The final ``HYV4HCHeadLayer`` merges the channels before the model's output norm.」——支持 C2、F1/F2/F3。
  - `hc.py` HYV4HCPreLayer.forward：`x_flat = x.flatten(1).float()`；`rsqrt = torch.rsqrt(x_flat.square().mean(-1,keepdim=True) + self.layernorm_epsilon)`；`pre = torch.sigmoid(pre_raw*self.hc_scale[0].float()+self.hc_base[:hc].float())+hc_eps`；`post = self.magnitude*torch.sigmoid(post_raw*self.hc_scale[1].float()+self.hc_base[hc:2*hc].float())+hc_eps`；`y = torch.sum(pre.unsqueeze(-1)*x.reshape(shape),dim=1)`——与 3.1 三个公式及变量表逐项一致（含读门、写门各自的 scale/base 切分）。
  - `hc.py` reset_parameters：`nn.init.constant_(self.hc_scale, 0.01)`；`self.hc_base[:hc_mult].fill_(-log(hc_mult-1))`；`self.hc_base[hc_mult:2*hc_mult].fill_(0.0)`；head 的 `hc_head_base.fill_(-log(hc_mult-1))`、`hc_head_scale=0.01`——支持 C8/F4 与 3.2 初始化。
  - `hc.py` HYV4HCPostLayer docstring：「y[n, i, d] = post[n, i] * x[n, d] + residual[n, i, d]」——支持 F2 与 2.1 的 $\hat x_i=H_{post,i}z+x_i$。
  - `hc.py` `_prepare_input_to_3d`：`hidden_states.unsqueeze(1).repeat(1, hc, 1)`——支持 C7 入口广播复制。
  - `model.py` `_forward_ihc`：`hc_attn_layer.pre → input_layernorm → self_attn → hc_attn_layer.post`，`hc_mlp_layer.pre → post_attention_layernorm → mlp → hc_mlp_layer.post`，注释「Under iHC the residual is carried inside hidden_states.」；HYV4DecoderLayer.__init__ 中 `hc_attn_layer`/`hc_mlp_layer` 各自独立构造（dense 层同样构造 `hc_mlp_layer`）；HYV4Model 中 `hidden_states = self.hc_head(hidden_states)` 紧接 `return self.norm(hidden_states)`——支持 C5、C6、C7 与 2.1/5.1。
  - `hpc_ihc.py` 模块 docstring：「The eager path issues 20 / 5 / 15 kernels for pre / post / head; each HPC op is a single kernel.」「VLLM_ENABLE_HPC_OPS=1」「Only sm100 / sm103」「Only hc_mult == 4 and hidden_size in {4096, 6144}」；`_SUPPORTED_HIDDEN_SIZES={4096,6144}`、`_SUPPORTED_HC_MULTS={4}`、`_SUPPORTED_CAPABILITIES={100,103}`；NOTE 段「The reference implementation also offers a cross-layer post+pre fusion (``HpcIHCPostPre``) ... is not ported here.」——支持 C19、C20 与 5.2 折叠块（20/5/15、启用条件、PostPre 未移植）。
  - `hpc_ihc.py` HpcIHCPost docstring：「Unlike mHC there is no comb matrix, so each output channel only needs its own residual channel and the whole thing is one fused multiply-add per element」——支持 C3。
  - 命名分歧：`hc.py`、`hpc_ihc.py` 模块 docstring 均写「iHC (independent Hyper-Connections)」，README 写 identity——支持 C4 与 1.1 命名注记。
- checkpoint（model.safetensors.index.json 2006 个张量 + shard 头 HTTP Range 实测）：hc 张量共 471 个 = 156 站点 ×(hc_fn/hc_scale/hc_base) + hc_head 三个；`model.layers.N.hc_{attn,mlp}_layer.hc_pre.hc_fn` = F32 [8, 24576]、`hc_scale` = F32 [2]、`hc_base` = F32 [8]、`model.hc_head.hc_head_fn` = F32 [4, 24576]；`model.mtp_layers.*` 27 个张量中无 hc 张量（结构含 eh_proj、enorm、hnorm、input/post_attention/final_layernorm、self_attn、mlp）——支持 C11、C13、N3 与 5.2 两张表、MTP 段。
- 参数量复算（独立旁证）：156×196618+98309 = 30,770,717；HF API `safetensors.parameters` 中 F32 合计 30,795,421，减去与 iHC 无关的 `learnable_sink_param`[64]×78 + `mlp.gate.e_score_correction_bias`[256]×77 = 4992+19712 = 24,704，恰为 30,770,717——支持 C12、N2 与 5.2「30,770,717 / 约 0.004%」。迷你站点 8×8+2+8=74、迷你 head 4×8+1+4=37、Hy4 站点 8×24576+2+8=196,618、head 4×24576+1+4=98,309 均可复算。
- mHC 论文（arXiv:2512.24880v2）：摘要「we propose Manifold-Constrained Hyper-Connections (mHC), a general framework that projects the residual connection space of HC onto a specific manifold」；§3.1/Fig.3「the Amax Gain Magnitude yields extreme values with peaks of 3000, a stark divergence from 1」（27B 模型，度量=最大绝对行和/列和）；§5.4「compared to the maximum gain magnitude of nearly 3000 in HC, mHC significantly reduces it by three orders of magnitude」——支持 N8 与 1.1「峰值约 3000、理想值 1（27B 模型）」。
- HC 论文（arXiv:2409.19606v3）Eq.(2)：`𝐇^ = ℋ𝒞(𝒯,𝐇) = 𝐁⊺𝒯(𝐇⊺𝐀𝐦)⊺ + 𝐀𝐫⊺𝐇`，三矩阵为 𝐀𝐦（读）、𝐁（写）、𝐀𝐫（残差/流间）——与 1.1 首个公式逐字一致，支持 F6（`A_r=I` 时退化为第二个公式）。
- 知乎文章《你的deepseek mHC可能不需要"m"》（zhuanlan.zhihu.com/p/2010852389670908320，Hy4 README 的 iHC 链接，作者谢天／MSRA）——知乎原站 403，经两处转载副本核对原文：单层 H_res「接近单位阵（对角线元素 ~0.96，非对角线 ~0.01）」「累积乘积 Π H_res：坍缩为全 0.25 的均匀混合矩阵」（另有「累积乘积在约 10 层后便坍缩为均匀矩阵」）；「当双随机矩阵满足一致正性条件时，其 Dobrushin 遍历系数 γ<1，多层连乘后所有行向量会以指数速率趋于一致，最终收敛到均匀矩阵」；边界「如果矩阵序列是纯置换矩阵或可约的，则不满足一致正性条件，不会坍缩」；「实测行和的标准差为 0.12」；「Identity HC > mHC > mHC lite > mHC orthogonal（例如 cayley 正交）」，「实验在 qwen3 1.7B 和 8B dense」上方标「Qwen 1.7B 从头预训练（150B Tokens）」；「流的语义一致性：流0永远在位置0，流1永远在位置1」——支持 C14、C15、C16、C18 及 4.1/4.2/4.3。
- mHC-lite 论文（arXiv:2601.05732）：`ν := min/max` 之比，「approximately 27.9% of SK inputs satisfy 1/ν ≥ 10^{13}」；「the column sum of a single residual matrix in mHC may deviate from 1 by up to 100%」——支持 C17/N5「relative range 达 $10^{13}$（占其实测输入的 27.9%）时列和偏差可达 100%」。
- GLM-5.3-Flash（zai-org/GLM-5.3-Flash）：模型卡 Introduction「With 320B total parameters and just 18B active parameters」；HF safetensors `total` = 321,323,031,390（≈321B，与页面「约 321B」一致，并解释了 320B/321B 的口径差）；config.json `text_config`：`hc_mult=4`、`hc_sinkhorn_iters=20`、`hc_eps=1e-06`、`hidden_size=4096`、`num_hidden_layers=45`；transformers `modeling_glm5_next.py` Glm5NextTextHyperConnection：`mix=(2+hc_mult)*hc_mult`→fn [24, 16384]、`pre=σ(pre_w*scale+base)+hc_eps`、`post=2*σ(post_w*scale+base)`（无 eps）、`comb=softmax+eps` 后 Sinkhorn 迭代、docstring「The decoder layer instantiates two of these (one for the attention site, one for the mlp site)」；Glm5NextTextHyperHead「this is an unweighted mean」（`hidden_streams.mean(dim=2)`）——支持 C21/N9 与 5.3 对照表全部行；GLM 残差通路参数复算 90 站点 ×(24×16384+24+3)=90×393,243=35,391,870 ✓。
- 复算类：两段 Python 代码（迷你 iHC 前向、Sinkhorn 坍缩实验）逐字执行，输出与页面「预期输出」完全一致（含「流间最大差 0.0e+00」「两者之差的最大分量 1.21e-05」「[1.4684 0.2144]」「max|P-1/n| 0.7500→0.1481→0.0319→0.0048→…」）；3.3「约 7.3 倍」= 1.4684/0.20 ✓；4.2 补充「δ≈0.10，界 1-4×0.10=0.6，实测约 0.2 每层」与代码输出一致（衰减比 0.2154/0.1505/0.208）✓。
- 机械项：全部本地链接（hyper-connections、residual-connection、rmsnorm、dsa、deepseek-moe、speculative-decoding、hy4-preview-dataflow）真实存在；无 research/ 引用、无「（待生成）」占位；`overview.html` 与 `index.html` 双向链接；`python3 .dojo/scripts/validate.py wiki/ihc/index.html` 返回 `validation ok`；SVG 图内公式均置于 `<foreignObject>`、`<text>` 内无 ASCII 数学写法；核心问题与各章「本章问题」均有解答折叠块且答案指向正确章节。

（审查者备注：知乎原站及其 API 均返回 403，上述知乎文章内容取自 blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/158591690 与 yunpan.plus/t/15997-1-1 两处转载副本，两者互相一致且与 mHC-lite 原文吻合。页面 4.3 补充末句称「该文作者自述文中部分表格文字由 AI 生成」，在上述可访问副本中未能定位该自述——两副本均会剥除公式/表格/脚注，无法据此判定其在知乎原文中不存在，故未列为问题，供有原站访问权限者复核。）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（无阻断与重要问题；遗留 1 项轻微表述项）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
