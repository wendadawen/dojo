<!-- review-meta
round: 5
page: wiki/kimi-k3-dataflow/index.html
reviewed_content_sha256: a7d2539da78bf879
-->
# Kimi K3 前向数据流审查记录（第 5 轮）

- 页面版本：2e001ebf2a8013f73adc919b745b080a9fae9582
- 审查时间：2026-09-13 21:16
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：1. 关键规格；2. 交互式数据流（含 noscript 全部八张表：整体总览、Dense FFN 内部、Block AttnRes、KDA 层内部、MLA 层内部、Stable Latent MoE 内部、视觉编码器、多模态融合）；3. 要点；4. 视觉编码器与多模态融合（4.1–4.5）；5. 长上下文开销的来源；6. 核对方式；来源与范围说明。另逐条读完全部 VIEWS 节点 tooltip（公式与说明）。

来源获取方式：官方 config.json 与 modeling_kimi_k3.py / modeling_kimi_linear.py / configuration_kimi_k3.py / kimi_k3_vision_processing.py / media_utils.py / preprocessor_config.json（huggingface.co/moonshotai/Kimi-K3）；技术报告 arXiv:2607.24653v1 HTML 全文；fla（flash-linear-attention）short_conv.py / causal_conv1d / triton ops / ops/kda。

## 本轮核对通过的主要条目（供复验，不列为问题）

- 层分布：config `full_attn_layers`（1-based 4,8,…,88,92,93）= 24 层 MLA，`kda_layers` 69 层；0-based 层 0 KDA+Dense、0–87 按 [3 KDA+1 MLA]、88–90 连续 KDA、91–92 连续 MLA，与页面「69 KDA / 24 MLA」「块首 0,4,…,84」一致。
- 规格表全部数值逐项对 config：hidden 7168、96 头、head_dim 128、896 专家 top-16、num_shared_experts 2、routed_expert_hidden_size 3584、moe_intermediate_size 3072、intermediate_size 33792、q_lora_rank 1536、kv_lora_rank 512、qk_nope 128 / qk_rope 64 / v 128、attn_res_block_size 12、short_conv_kernel_size 4、gate_lower_bound −5、use_full_rank_gate true、situ beta 4 / linear_beta 25、vocab 163840、tie_word_embeddings false、rms_norm_eps 1e-5、routed_scaling_factor 1.0、moe_renormalize true、num_nextn_predict_layers 0、media_placeholder_token_id 163605。稀疏度 896/16=56 ✓。
- 公式回源：KDA `logα=−5σ(e^{A_log}(g+dt_bias))` 与 fla `fla/ops/kda/gate.py` safe_gate「g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))」逐字一致；delta 规则与报告 Eq.1 一致；输出缩放 1/√128 = fla `scale = K ** -0.5`；SiTU-GLU 与报告 Eq.12 一致，上界 4×25=100 ✓；AttnRes 打分/softmax 与 `_apply_attn_res` 逐行一致（打分权重 = norm.weight×proj.weight，softmax 作用在 9 个候选上，加权用原始 v）；MoE 门控与 `KimiMoEGate.forward` 一致（偏置只进 topk 排序、权重用原始分数 renormalize×1.0）。
- 参数量复算：视觉塔 27×14,682,112 + 602,112 + 4,194,304 + 1,024 = 401,214,464（对报告 Table 1「401M」）；merger 4096×4096 + 4096×7168 + 7168 = 46,144,512；447.4M/2.78T = 0.016% ✓。
- 开销表复算：MLA 60 KiB/token/层 × 24 × 1M = 1440 GiB；潜压缩 1.125 KiB → 27 GiB；4K/32K/128K/256K 各行与合计列（分项和 = 合计）逐一复算一致；93 层全 MLA 5.45 TiB、24/93=25.8% ✓；KDA 6.0 MiB/层（96×128×128 fp32，fla `assert initial_state.dtype == torch.float32`）✓。
- 4.4 视觉 token 表五行（1024/4096/9216/9360 patch 与 256/1024/2304/2340 token）逐格复算一致；融合两例（6→10、5→8）与 `_merge_input_ids_with_image_features` 的 cumsum−1 位置映射逐位复算一致。
- 2D RoPE 槽位（偶数槽编码宽 x_pos=flat%W、奇数槽编码高）与 `_precompute_freqs_cis` 一致；max_height/width=512 与 `Rope2DPosEmbRepeated(..., 512, 512)` 一致；docstring 与实现相反属实。
- 表述维度：全文（含折叠块与图注）通读，未发现「本页将…」「下面来看…」类元话语、无「本页」主语自我指代、无我/我们/你类会话指代、无调试与复现踩坑叙事、无临场评价；「本机实测」类表述是 dataflow 规范要求标注的实测来源，合规。alt 属性仅 lightbox 空 alt，无 `$...$`。站内链接（kda/mla/kv-cache/moonvit-v2/nope/rope/qwen3-8-flash-next-dataflow）均真实存在；`research/measured.md` 存在。`.dojo/scripts/validate.py` 通过。

## 问题

- [阻断·技术] 第 5 节（「长上下文开销的来源」段首与段末 KDA 侧）：「MLA KV 为 bf16，KDA 递归与短卷积状态为 fp32」与「短卷积状态 288 KiB（q/k/v 三个 12288×4 滑窗，核长 4）」互相矛盾，且与官方 fla 实现不符｜引文依据：fla `fla/modules/conv/short_conv.py` 的 `ShortConvolution.step`：`cache = x.new_zeros(N, D, W)`（N/D/W = 序列数/通道数/核长，dtype 取自输入 x）；`fla/modules/conv/triton/ops.py`：`final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)`；同文件文档「Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size」、`state_size = self.hidden_size * self.kernel_size[0]`。即短卷积 cache 是模型 dtype（bf16），不是 fp32；bf16 下 3×12288×4×2 B = 294,912 B = 288 KiB（与页面 288 KiB 相符），fp32 下应为 576 KiB（3×12288×4×4 B），69 层合计会变成 0.442 GiB 而非 0.423 GiB｜修复要求：把该括注改为「MLA KV 与 KDA 短卷积状态为 bf16（fla 以模型 dtype 保存），KDA 递归状态为 fp32」，288 KiB 与 0.423 GiB 保持不变；若坚持 fp32 口径，则须把 288 KiB 改为 576 KiB 并重算表内合计列（6.048/45.423/180.423/360.423/1440.423 → 随之变化），二者不得并存｜修复：｜复验：
- [重要·技术] 第 1 节规格表「其他」行：MXFP4 量化范围写为「范围=路由专家的 w1/w2/w3 权重」，与同一格随后给出的 ignore 表语义（以及本页 AttnRes 备注）自相矛盾｜引文依据：config.json `quantization_config = {config_groups.group_0.targets: ["Linear"], ignore: ["re:.*self_attn.*","re:.*shared_experts.*","re:.*mlp\\.(gate|up|gate_up|down)_proj.*","re:.*lm_head.*","re:.*vision_tower.*","re:.*mm_projector.*"]}`；而 `modeling_kimi_linear.py` 中 latent 投影属性名为 `routed_expert_down_proj` / `routed_expert_up_proj`（KimiSparseMoeBlock.__init__），AttnRes 三个投影为 `self_attention_res_proj` / `mlp_res_proj` / `output_attn_res_proj`（`layers.N.mlp_res_proj` 不含子串 `mlp.`），均不匹配任何 ignore 正则 → 按 ignore 表语义它们同样是量化目标，故量化范围不止路由专家的 w1/w2/w3（页面自己写的「AttnRes 的 self_attention_res_proj / mlp_res_proj 也不在 ignore 表内」正推出这一点）；另一侧，`KimiMoEGate.weight` 是裸 `nn.Parameter`（非 Linear），本就不在 targets 内，因此「MoE 路由器保持高精度」与 config 并不构成分歧｜修复要求：把「范围」改为按 config ignore 表推出的完整集合（路由专家 w1/w2/w3 + latent 下/上投影 + 三个 AttnRes res 投影），或把「范围=路由专家 w1/w2/w3」明确限定为技术报告的表述（报告 §4.1.4「quantize the MoE expert weights …」），并把「分歧」只限在 latent MoE 投影一项（路由器不在 Linear 目标内，与报告不冲突）｜修复：｜复验：
- [轻微·技术] 4.1 与视觉编码器视图 `px` 节点：「像素边长非 14 整数倍时由 processor 先 resize」把 resize 的触发条件说错、且未提补 padding｜引文依据：`media_utils.py::navit_resize_image`（kimi_k3_vision_processing.get_resize_config 调用）：`s1 = sqrt(in_patch_limit/(…))`、`s2 = patch_limit_on_one_side*patch_size/width`、`s3 = 同/height`、`scale = min(1.0, s1, s2, s3)`，只有超出 patch 预算才缩放（in_patch_limit=65536、patch_limit_on_one_side=512，见 preprocessor_config.json），否则 new_w/new_h 不变；随后 `factor = merge_kernel_size * patch_size`（=2×14=28），`pad_height = (factor - new_h % factor) % factor`——小图（如 100×100）是补 pad 到 28 的倍数，不是 resize｜修复要求：改为「processor 按 patch 预算（in_patch_limit=65536、单边 512 patch）决定是否缩放，并把尺寸补齐到 merge_kernel_size×patch_size=28 的倍数；模型侧 h、w 需为 28 的倍数（2×2 合并按整除切分，余数会被丢弃）」｜修复：｜复验：
- [轻微·技术] 4.5 第三个边界行为：「段多于占位符在 occupation 表 broadcast 处即报错，占位符多于段在填充校验处抛 ValueError」——后一句只在段数为 1 时成立｜引文依据：`modeling_kimi_k3.py` 992–997：`_token_occupation_table[input_ids.flatten() == image_token_index] = torch.tensor(feature_lengths, …)`；1068–1072：`if image_to_overwrite.sum() != image_features.shape[:-1].numel(): raise ValueError(…)`。段数 ≥2 而占位符更多时（如 3 个占位符、2 段），赋值处即因形状不匹配抛 RuntimeError，走不到第 5 步的 ValueError；只有段数 =1（可 broadcast）时才会到填充校验处抛 ValueError｜修复要求：把规则限定为「段数 =1 时占位符多于段在填充校验处抛 ValueError；段数 ≥2 时无论哪一侧不匹配都在 occupation 表赋值处报形状错误」，或直接写成实测覆盖的两个具体组合｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 2
- 处置：修复（阻断与重要问题按上述要求改后重新核对 fla dtype 与 config ignore 语义，轻改两处表述即可；改后需重跑 `.dojo/scripts/validate.py`，本轮已确认其当前返回 validation ok）