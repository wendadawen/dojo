<!-- review-meta
round: 4
page: wiki/kimi-k3-dataflow/index.html
reviewed_content_sha256: f76d4674c742c126
-->
# Kimi K3 前向数据流审查记录（第 4 轮）

- 页面版本：e3c737c818620af4d93ec6e075efeb05f51480fb89e1c430922abe59c4a9f027（工作树 SHA-256）／ b2b1e3af7111f3121db6b64e4ed1906a049c1ba6（SHA-1）
- 审查时间：2026-09-13 20:19
- 审查者：独立子代理（第 4 轮；未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：head 元信息与页面导语 / 1. 关键规格 / 2. 交互式数据流（含 8 个视图的 noscript 节点·维度·公式表与边表）/ 3. 要点 / 4. 视觉编码器与多模态融合 / 5. 长上下文开销的来源 / 6. 核对方式 / 来源与范围说明
- 核对来源：官方 `config.json`（huggingface.co/moonshotai/Kimi-K3）、官方源码 `modeling_kimi_linear.py` / `modeling_kimi_k3.py` / `configuration_kimi_k3.py`、技术报告 arXiv:2607.24653（v2，已下载 PDF 计页 = 47）、flash-linear-attention 的 `chunk_kda` / `fused_recurrent_kda` / `ShortConvolution`。
- 已核对为一致（不逐条列出）：config 全部结构数值（93 层、hidden 7168、96 头、896 专家 top-16、共享专家 2、routed_expert_hidden_size 3584、moe_intermediate_size 3072、first_k_dense_replace 1、q_lora 1536 / kv_lora 512 / nope 128 / rope 64 / v 128、attn_res_block_size 12、gate_lower_bound −5、use_full_rank_gate、situ β=4/25、vocab 163840、tie_word_embeddings false、media_placeholder_token_id 163605、num_nextn_predict_layers 0）；`full_attn_layers`=[4,8,…,88,92,93] 与 `kda_layers`=[1,2,3,5,…,89,90,91]；Table 1 的 2.78T / 104.2B / 401M / 27 层 ViT / patch 14 / 12 头；MoE 门控、SiTU-GLU、MLA、KDA（safe_gate `lower_bound*sigmoid(exp(A_log)*(g+dt_bias))`、输出 1/√128、L2 归一）、AttnRes（block_residual 8 快照 + 当前 = 9 候选、打分 `(k⊙w_norm⊙w_proj).sum(-1)`、加权作用于原始 v）、视觉塔全部结构（Conv2d 无 bias、64×64 可学习表 + 4 帧 sincos、2D RoPE 偶数槽=宽/奇数槽=高、时间全池化 + 2×2 合并、PatchMergerMLPV2 4096→4096→7168、V1 pre_norm 已移除）、占位符扩展融合（occupation 表 / cumsum−1 / T′=T−#PH+Σn_i）、参数 401,214,464 与 46,144,512 的分项复算、MLA KV 60 KiB/token/层 与 1M=1440 GiB / 潜压缩 27 GiB 的复算。

## 问题

- [阻断·技术] §5「长上下文开销的来源」表「KDA 状态 × 69 层」列 + 表后「参考实现口径」段 + §6「关键实测结论」｜KDA 侧状态按 bf16 计：递归状态「3.0 MiB（96×128×128）」、短卷积状态「216 KiB（q/k/v 三个 12288×3 滑窗）」、69 层合计 0.216 GiB。与官方 fla 实现不符：(a) KDA 递归状态实际是 fp32，每层 96×128×128×4 B = 6.0 MiB；(b) 短卷积的 cache 形状是 [N, D, kernel_size]，kernel_size = short_conv_kernel_size = 4（不是 3），每层 3×12288×4×2 B = 288 KiB。69 层合计应为 69×(6 MiB+288 KiB) = 433.4 MiB ≈ 0.423 GiB，页面所写 0.216 GiB 只有约一半；表中「合计（参考口径）」列亦随之上移约 0.207 GiB。｜引文依据：fla `ops/common/chunk_delta_h.py` L720/L723 `final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)`；fla `ops/kda/fused_recurrent.py` L279/L281 `final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)`；fla `ops/kda/chunk.py` L434 `assert initial_state.dtype == torch.float32, "initial_state must be in float32."`；fla `modules/conv/short_conv.py` L139 文档「Previous cache tensor of shape [N, D, W], where W is the kernel size」、L226 `cache = x.new_zeros(N, D, W)`（`W = self.kernel_size[0]`）、L251 `state_size = self.hidden_size * self.kernel_size[0]`。｜修复要求：把 §5 的 KDA 状态改为 fp32 口径（递归状态 6.0 MiB/层 + 短卷积 288 KiB/层，69 层 ≈0.423 GiB），同步改表中「KDA 状态 × 69 层」列（4K/32K/128K/256K/1M 均为 0.423 GiB）与「合计（参考口径）」列（6.048 / 45.423 / 180.423 / 360.423 / 1440.423 GiB），并把 §6 的「KDA 69 层固定状态 0.216 GiB」改为 0.423 GiB；若坚持 bf16 口径，须改写为明确标注的假设，不得写成「按官方 config 精确计算」。｜修复：｜复验：

- [轻微·技术] 页面导语 + §1 表「层分布」行｜「第 0–90 层按 [3 KDA + 1 MLA] 分块（块首 0,4,…,84；MLA 在 3,7,…,87）」中，块首列表只覆盖第 0–87 层（22 块），与所述范围「0–90 层」（91 层）相差 3 层：0-based 88、89、90 是 KDA，且 0-based 88 本身是 4 的倍数（应为块首）。｜引文依据：config.json `text_config.linear_attn_config.kda_layers` = [1,2,3,5,6,7,…,89,90,91]（1-based，共 69 个，含 89/90/91 → 0-based 88/89/90）；`full_attn_layers` = [4,8,…,88,92,93]（1-based，共 24 个 → 0-based 3,7,…,87,91,92）。｜修复要求：使范围与块首列表一致，例如改为「第 0–87 层按 [3 KDA + 1 MLA] 分块（块首 0,4,…,84）」并补一句「第 88–90 层为 3 个连续 KDA」；或把块首写为「0,4,…,88」并说明 91 属该块的第 4 层。｜修复：｜复验：

- [轻微·技术] §1 表「其他」行（MXFP4 量化豁免清单）｜页面把「latent MoE 投影」「MoE 路由器」列入豁免范围，但 config.json 的 `quantization_config.ignore` 未列出这两项，其中 latent MoE 的 `routed_expert_down_proj` / `routed_expert_up_proj` 是 `nn.Linear` 且不被任何 ignore 正则命中（按 config 会被量化）；技术报告 §4.1.4 则称 latent MoE 投影保持高精度。两个官方来源在此分歧，页面未写明（同页 MTP 行已写明同类分歧，处理方式不一致）。｜引文依据：config.json `text_config.quantization_config.ignore` = ["re:.*self_attn.*","re:.*shared_experts.*","re:.*mlp\\.(gate|up|gate_up|down)_proj.*","re:.*lm_head.*","re:.*vision_tower.*","re:.*mm_projector.*"]；报告 §4.1.4「all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision」；`modeling_kimi_linear.py` L804/L807（两个投影均为 Linear）与 L898（MoE 挂在 `block_sparse_moe` 下，路径不含 `mlp.`）。｜修复要求：在「其他」行写明 config 的 ignore 表与报告 §4.1.4 在豁免范围上的分歧，或按 ignore 表修正豁免清单（并说明 AttnRes 投影 `self_attention_res_proj` / `mlp_res_proj` 亦不在 ignore 表内）。｜修复：｜复验：

- [轻微·表述] §2 的 `<noscript>` 段首句｜「脚本不可用时，各视图的节点、维度与公式逐一列在下面」属以页面为主语的下文引导语（元话语），与 dataflow 规范「不写引导语，直接进入路径」相抵触。删去后读者仍可由随后的小标题「整体总览 (93 层)」等直接进入内容。｜引文依据：不适用｜修复要求：删去该引导语，直接以小标题给出无脚本回退内容（如「无脚本时的视图内容」），不出现「列在下面」一类指向本页正文的措辞。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 3
- 处置：修复（阻断项为 §5/§6 的 KDA 状态数字与官方实现不符，须先关闭；3 项轻微可同轮并入修复）
- 机械项：`.dojo/scripts/validate.py wiki/kimi-k3-dataflow/index.html` 通过；7 个内链目标（moonvit-v2 / nope / rope / kda / mla / kv-cache / qwen3-8-flash-next-dataflow）均存在；页面仅引用 `research/measured.md`（存在），未指向已移除的实测产物路径；无「待生成」占位；数学符号均由 KaTeX 渲染。