<!-- review-meta
round: 9
page: wiki/kimi-k3-dataflow/index.html
reviewed_content_sha256: e7d43c24b9f0edc0
-->
# Kimi K3 前向数据流审查记录（第 9 轮）

- 页面版本：wiki/kimi-k3-dataflow/index.html（sha256 30d978818f1f602b…；工作树与 HEAD 7dec795 一致，无未提交改动）
- 审查时间：2026-09-14 17:03 CST
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查；本轮未读取本页 research/ 下任何文件）
- 已完整阅读章节：1 关键规格 / 2 交互式数据流（含 8 个 noscript 表格与全部视图的 JS 节点、边、tooltip）/ 3 要点 / 4 视觉编码器与多模态融合（4.1–4.5）/ 5 长上下文开销的来源 / 6 核对方式 / 来源与范围说明

## 本轮实际取到的来源（用于回源核对）

- 官方 config.json：huggingface.co/moonshotai/Kimi-K3/raw/main/config.json（本轮下载，7006 字节）
- 官方源码：modeling_kimi_k3.py、modeling_kimi_linear.py、configuration_kimi_k3.py、preprocessor_config.json（本轮下载）
- fla（flash-linear-attention）：fla/ops/kda/gate.py、fla/ops/kda/chunk.py（本轮从 GitHub raw 取下）
- 技术报告：arXiv:2607.24653 PDF（v1 与 v2 均下载，pdfinfo 报 Pages: 47；表 1 与 §2.1/§2.2/§2.3.2/§2.3.3/§2.4/§3.4/§4.1.4 正文）
- 本机复算：用站点自带 libs/katex.min.js 对页面全部公式逐条 KaTeX 渲染；对第 5 节开销表、视觉塔/merger 参数量逐项复算

## 已回源核对并确认一致的条目（关键项）

1. **表 1 数字**：原文 “Total Parameters 1.04T / 2.78T ↑167%”“Activated Parameters 32.6B / 104.2B ↑220%”“#Layers 61 / 93”“Routed Experts 384 / 896”“Experts Active per Token 8 / 16”“Shared Experts 1 / 2”“Attention Heads 64 / 96”“Latent MoE Dimension – / 3584 (0.5×)”“MoE Hidden Dimension per Expert 2,048 / 3,072”“Attention-Layer Composition 61 MLA / 69 KDA + 24 MLA”“Number of MTP Layers 1 layer / 1 layer”“Total Parameters of ViT – / 401M”“#ViT Layers – / 27 layers”“#Attention Heads of ViT – / 12”。页面 2.78T、104.2B、896、top-16、2 共享、3584、3072、69+24、401M、27 层、12 头全部吻合。
2. **层分布**：config `full_attn_layers=[4,8,…,88,92,93]`、`kda_layers=[1,2,3,5,…,91]`；configuration_kimi_k3.py:152 `is_kda_layer` 用 `(layer_idx + 1) in kda_layers`（1-based）→ 0-based MLA 在 3,7,…,87,91,92，KDA 69 层、层 0 为 KDA。与页面“块首 0,4,…,84 / MLA 在 3,7,…,87 / 88–90 连续 KDA / 91,92 连续 MLA”完全一致。
3. **KDA safe_gate**：fla gate.py docstring `g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`，实现 `g = lower_bound * F.sigmoid(g)`，非安全版 `g = -A_log.exp().unsqueeze(-1) * softplus(...)`，并写明 “Recommended value: -5 (i.e., per-step decay exp(-5) ≈ 0.0067)”。页面公式、值域 (-5,0)、exp(-5)≈0.0067、“不是 softplus 形式”逐字吻合。
4. **KDA 其余公式来源**：modeling_kimi_linear.py chunk_kda 调用 `use_qk_l2norm_in_kernel=True`（→ q̂/k̂）、`use_beta_sigmoid_in_kernel=True`（→ β_w 核内 sigmoid）；fla chunk.py `if scale is None: scale = K ** -0.5`（→ 1/√128）；`assert initial_state.dtype == torch.float32`（→ 第 5 节 6.0 MiB/层 的 fp32 口径）。
5. **参数量复算**：视觉塔每层 = 2×RMSNorm(1024) + MLP2(1024×4096+4096×1024) + wqkv(1024×4608) + wo(1536×1024) = 14,682,112；27 层 = 396,417,024；+patch 602,112 + 位置表 64×64×1024=4,194,304 + final RMSNorm 1,024 = **401,214,464**；merger = 4096² + 4096×7168 + 7168 = **46,144,512**；合计 447,358,976 ≈ 447.4M ≈ 2.78T 的 0.0161%。源码对照：`wqkv = Linear(hidden_dim, qkv_hidden_size*3, bias=attn_bias=False)`、`hidden_size_per_attention_head = qkv_hidden_size // num_heads`(=1536//12=128)、`norm_type=="rmsnorm"→nn.RMSNorm`、`MoonViT3dEncoder.final_layernorm`。
6. **2D RoPE 槽位**：modeling_kimi_k3.py:386-403 `x_pos = flat_pos % max_width; y_pos = flat_pos // max_width; freqs_cis = cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], -1)`，docstring 却写 “height axis: ret[h, w, 2*i] = cis(h·…)”。页面“偶数槽编码宽（x_pos=flat%W）、奇数槽编码高、docstring 与实现相反”正确；`Rope2DPosEmbRepeated(qkv_hidden_size//num_heads, 512, 512)`、`theta_base=10000` 支持“作用在头维 128、网格上限 512×512”。
7. **位置嵌入**：`assert t <= self.num_frames`、`register_buffer('time_weight', …, persistent=False)`、`t==1 → pos_emb_2d；else pos_emb_2d.repeat(t,1,1) + time_weight[0:t]`、weight 形状 [64,64,1024]。页面 t≤4、非持久 buffer 仅 4 帧、t>1 重复 t 份再加时间 sincos、4,194,304 参数全部吻合。
8. **merger**：`tpool_patch_merger` 的 `permute(...).mean(dim=0)`（时间全平均）与 `view(new_h*new_w, kh*kw, -1)`（4 份特征不相加）；`PatchMergerMLPV2 = Linear(4096,4096,bias=False)→GELU→Linear(4096,7168,bias=False)→RMSNorm(7168)`，V1 `PatchMergerMLP` 有 `pre_norm=LayerNorm(1024, eps=projector_ln_eps)`。页面“V1 pre_norm 在 V2 移除”正确；config `linear_bias/attn_bias=false` 支持“全部无 bias”。
9. **融合**：`_merge_input_ids_with_image_features` 的 occupation 表（`torch.ones_like` 后按占位符写 `feature_lengths`）、`new_token_positions = cumsum(occ,-1)-1`、文本写新位/其余位填视觉特征、labels 初值 `ignore_index`、`position_ids = cumsum(mask)-1`、`raise ValueError` 于 `image_to_overwrite.sum() != image_features.numel()`；forward 守卫 `if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1`，decode 走 `elif (past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1)`。页面三步描述、6→10 位示例（10=6−1+5、文本落 0–2/8–9、视觉落 3–7）、三段边界行为、decode 行为全部吻合。
10. **量化与 MTP 分歧**：config ignore 六条（self_attn / shared_experts / mlp.(gate|up|gate_up|down)_proj / lm_head / vision_tower / mm_projector）与页面列的豁免逐条一致；KimiMoEGate 的 `self.weight = nn.Parameter(torch.empty((num_experts, gating_dim)))` 为裸参数，支持“路由器不属 Linear 量化目标”；报告 §4.1.4 原文 “all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision” 支持页面所述两来源分歧。`num_nextn_predict_layers=0` 且两个 modeling 文件无 NextN/MTP 类；报告表 1 “Number of MTP Layers 1 layer / 1 layer” 与 §4.1.4 “…fine-tune the pre-trained MTP layer into an EAGLE-3-style draft model” 支持“配置未体现、报告称有 1 层”。
11. **第 5 节开销表全部复算通过**：60 KiB/token/层 = 96×(192+128)×2B；4K/32K/128K/256K/1M → 5.625/45/180/360/1440 GiB；潜压缩 576×2B=1.125 KiB → 4K 0.105 GiB、1M 27 GiB；93/24×1440 = 5580 GiB = 5.449 TiB、24/93 = 25.81%；KDA 96×128×128×4B = 6 MiB ＋ 3×12288×4×2B = 288 KiB，×69 = 0.423 GiB；448×448→256 token→256×60KiB×24 = 360 MiB。
12. **站内链接与对照页**：moonvit-v2、nope、rope、kda、mla、kv-cache、qwen3-8-flash-next-dataflow 七个目标文件均存在。qwen 页确有 `masked_scatter` 原地替换、视觉塔 LayerNorm、视觉 token 数 = prod(grid_thw)/merge²（∝ t），页面三处对照成立。
13. **机械项**：`.dojo/scripts/validate.py` 通过；noscript 覆盖全部 8 个视图，且节点/维度/公式/边与 JS 视图逐条同值；页面无 `<pre>` 框图、无 `alt` 内 `$...$`、交互视图无脚本可读。KaTeX 逐条渲染 92 条 `f:` 公式 + 正文/表格/图注行内公式，throwOnError:true 下 0 失败（唯一报错项是 HTML 实体 `$t&gt;1$`，浏览器解码后为 `t>1`，正常）。本节与其它节未出现同数字两处不一致、算式与结论不符、层型比例与图注读数不符的情况。

## 问题

- [轻微·技术] “Block AttnRes” 视图（index.html 第 627、632 行；noscript 同视图第 149–150、154 行）：符号 `p` 在同一视图内表示两个不同量——残差流与注意力权重。｜引文依据：本页“机制概览”节点 `$p = x_L + a + m$`、“当前残差流”节点 `$p = \text{current residual stream}$`、“堆叠”节点 `$v = [\,v_1,...,v_8,\; p\,]$`；而同视图“softmax 加权”节点写 `$p_j=\frac{e^{s_j}}{\sum_i e^{s_i}}$`、`$h = \sum_{j=1}^{9} p_j\, v_j$`（与“整体总览”的 `h = \sum_j p_j v_j,\; p=\mathrm{softmax}(s)` 一致）。官方源码对这两个量用了两个名字：modeling_kimi_linear.py:1080-1087 `v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)` 与 `probs = scores.softmax(-1).unsqueeze(1); hidden_states = torch.matmul(probs, v_float)`（`prefix_sum` 与 `probs`）。因此 `v_9` 恰是 `p`，同时 `p_j` 又是加权系数，读者可把残差流当成概率读。｜修复要求：把两处之一改名并同步全部出现位置——建议把 softmax 权重改为本视图未占用的符号（如 `\pi_j` 或 `a_j`），需改动 noscript “RMSNorm+打分/softmax 加权”两行公式、JS `note/cur/stack/soft` 四个节点的 `f` 字段，以及“整体总览”视图 `attnres` 节点公式与正文第 154、587 行附近的同符号写法；改后确认 AttnRes 相关公式中不再有同名不同义的符号。｜修复：｜复验：
- [轻微·表述] 第 6 行 `page-meta` 与 `<head>` 的 `meta description`（index.html 第 6、86 行）：把页面的核对来源写成“官方 config.json、官方实现源码与本机实测”/“全部数字按官方 config 与源码核对并经本机实测”，与同页来源表所述不一致。｜引文依据：本页来源表写“总参数 2.78T / 激活 104.2B｜技术报告 Table 1”，而这两项及 MoonViT-V2 401M、MTP 层数并不出现在 config.json 或两个 modeling 源文件里；报告表 1 原文即 “Total Parameters 1.04T / 2.78T”“Activated Parameters 32.6B / 104.2B”“Total Parameters of ViT – / 401M”。｜修复要求：把第 6 行与第 86 行的来源表述改为与来源表一致（例如“按官方 config.json、官方实现源码与技术报告 Table 1 核对，并经本机实测”），或删去“全部”这一限定词；改后两处来源表述与实际引用来源一一对应。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：修复（仅两处轻微项；画面无阻断、无重要问题）
