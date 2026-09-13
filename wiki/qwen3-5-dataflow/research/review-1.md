<!-- review-meta
round: 1
page: wiki/qwen3-5-dataflow/index.html
reviewed_content_sha256: 5942a046fa41f92f
-->
# Qwen3.5-397B-A17B 前向数据流 审查记录（第 1 轮）

- 页面版本：1453ff485789a71455a9b51d41b8bd8dab996ab5
- 审查时间：2026-09-13 19:00
- 审查者：独立子代理（未参与写作，未读取 research/ 内规划、修复或前序审查记录）
- 页面类型：dataflow（`meta name="dojo:type"` = dataflow），规范文件 `guides/model-dataflow.md`
- 已完整阅读章节：1. 关键规格 / 家族变体 / 2. 交互式数据流（含七个视图与全部节点悬停内容）/ 3. 要点（整体结构、GDN、全注意力层、MoE 路由、MTP、长上下文开销）/ 4. 视觉编码器与多模态融合（4.1–4.8）/ 5. 与 Qwen3.8-Flash-Next 的架构对比 / 6. 核对方式 / 来源与范围说明
- 来源获取方式：本页 `research/official/` 目录不存在（research/ 下仅有 measured.md、prereq-audit.md），官方来源改由外部实拉核对：HF `Qwen/Qwen3.5-397B-A17B` 的 config.json、model.safetensors.index.json、94 个分片中的 4 个 safetensors 文件头（HTTP Range）、README.md；transformers commit 36deb0b5 的 `qwen3_5/`、`qwen3_5_moe/`、`qwen3_next/` 源码；vLLM `qwen3_next_mtp.py`；家族六型号 config；Qwen3.8-Flash-Next 的 config/README/LICENSE。

## 独立复核通过的项（供复验参照）

- 总参数 403,397,928,944 与实验值精确一致：分片 total_size 合计 806,795,875,168 B；扣 F32 张量多出的 2 字节/元素 × 8640 元素（45×A_log[64] + 45×norm.weight[128]）后 /2 = 403,397,928,944。
- 语言主干 396,346,350,336、视觉塔 456,010,480（333 个 `model.visual.*` 张量按文件头逐张量求和 = 456,010,480）、MTP 6,595,568,128 与 1553 个张量，三者相加严格等于总参数。
- 层型：config `layer_types` 与张量头一致，主干 `self_attn` 恰为 0 起层号 3,7,…,59 共 15 层，`linear_attn` 恰 45 层；`mlp.experts.gate_up_proj` 为打包 3D 张量（60 个），MTP 专家为逐张量 gate/up/down。
- 关键张量形状：q_proj[16384,4096]、k/v_proj[512,4096]、o_proj[4096,8192]、q/k_norm[256]、in_proj_qkv[12288,4096]、in_proj_z[8192,4096]、conv1d[12288,1,4]（无 bias）、A_log F32[64]、norm F32[128]、dt_bias BF16[64]、gate[512,4096]、shared_expert_gate[1,4096]、mtp.fc[4096,8192]、merger.norm[1152]、merger.linear_fc1[4608,4608]、merger.linear_fc2[4096,4608]、patch_embed.proj[1152,3,2,16,16]。
- 公式与代码语义：GDN `g=-A_log.exp()*softplus(a+dt_bias)`、`beta=b.sigmoid()`、`in_proj_qkv` 宽 `key_dim*2+value_dim`、L2 归一化、`repeat_interleave`、块/递归两条路径；Qwen3.5 注意力 `q_proj` 无条件切出 gate 并 `attn_output*sigmoid(gate)`（`attn_output_gate` 未被读取）；`Qwen3_5MoeTopKRouter` 硬编码 `/= sum`（无 norm_topk_prob 开关），qwen3_next 侧确有 `norm_topk_prob` 字段；GDN 输出门 `Qwen3_5MoeRMSNormGated.activation="silu"`；vLLM MTP `torch.cat([inputs_embeds, hidden_states])`（embedding 在前）；视觉 `nn.LayerNorm(eps=1e-6)`、`use_postshuffle_norm=False`、`align_corners=True`、`current_pos += max(grid_thw[1],grid_thw[2])//spatial_merge_size`。
- 家族变体表六型号（层数/hidden/Q,KV 头/GDN 值头/专家与 FFN）与各官方 config 逐项一致；4B tie_word_embeddings=true。
- 激活量 16,331,922,176 / 17,349,040,896 可复算一致。
- 页内 16 个前置概念链接全部指向存在的 `wiki/<name>/index.html`。

## 问题

- 级别：重要｜来源：页面自身（两处实测值相互矛盾）｜位置：§3「MoE 路由」第 2 条 与 视图 4（MoE 内部）aux 节点｜问题：同一实验（"随机路由 6 token"）给出两个互不相容的辅助损失实测值，读者无法判断哪一个成立。｜引文依据：§3 正文"本机实测随机路由下 6 token 的损失为 0.014310，完全均匀分布下界 0.001000"；视图 4 aux 节点"本机实测随机路由 6 token 的损失 0.051200，均匀分布下界 0.001000"。｜修复要求：回到实测记录确认唯一取值，把 §3 与视图 4 节点改为同一数字；若两次实验条件不同（张量/随机种子/是否含系数）必须在两处分别写明条件，不得共用"随机路由 6 token"这一描述。

- 级别：重要｜来源：官方 config.json + 官方 checkpoint 张量头 + 页面自身（内部不一致）｜位置：视图 2（GDN 层）kv 节点 与 §3「全注意力层」第 3 条、长上下文开销表｜问题：kv 节点把每 token 每层 KV 字节数写成 1024 B，与同页其它位置的 2048 B 及长上下文表中 256K=7.500 GiB 所依赖的 2048 B/token/layer 相差一倍；节点公式 `2·2·256·2 B` 本身算得 2048，结果却写 1024。｜引文依据：节点 `2 \cdot 2 \cdot 256 \cdot 2\,\mathrm{B} = 1024\ \mathrm{B/token/layer}`；§3"每 token 每层 KV cache 仅 $2\times2\times256=1024$ 个元素（bf16 下 2048 字节）"；config num_key_value_heads=2、head_dim=256，即 2×2×256×2 B = 2048 B；262144×15×2048/1024³ = 7.500 GiB（与表中 7.500 一致）。｜修复要求：把 kv 节点结果改为 2048 B/token/layer（公式保留 2·2·256·2 B），并复核该节点"15 层 256K 合计 7.500 GiB / 60 层 30.000 GiB"仍成立。

- 级别：重要｜来源：官方 checkpoint 张量头（patch_embed.proj.weight 与 patch_embed.proj.bias）｜位置：§4.1「从像素到 patch」｜问题：把权重元素数写成 1,770,624 并称"恰等于 1152×1536"，两处都不成立：该权重实为 1,769,472 个元素，1,770,624 是权重加 bias 的模块总数。｜引文依据：文件头 `model.visual.patch_embed.proj.weight [1152, 3, 2, 16, 16]`、`model.visual.patch_embed.proj.bias [1152]`；1152×3×2×16×16 = 1,769,472，1152×1536 = 1,769,472，加 bias 1152 得 1,770,624。｜修复要求：改为"元素数 1,769,472 恰等于 1152×1536"，或改为"权重 1,769,472 加 bias 1,152 共 1,770,624"；§4.8 表中"patch_embed（Conv3d）1,770,624"须与之口径一致（若该行含 bias 需在表头或脚注写明）。

- 级别：重要｜来源：官方 config.json + transformers 源码 `qwen3_5/modeling_qwen3_5.py::compute_default_rope_parameters`｜位置：§3「全注意力层」第 4 条｜问题：MRoPE 最低频写成 1.54×10⁻⁶，与所标 θ=10⁷ 不符；该值实为 θ=10⁶ 的结果。｜引文依据：config rope_parameters.rope_theta=10000000、partial_rotary_factor=0.25、head_dim=256；源码 `dim=int(head_dim*partial_rotary_factor)`=64，`inv_freq=1/(base**(arange(0,64,2)/64))`，最低频 = 1e7^(−62/64) = 1.655×10⁻⁷（1e6^(−62/64)=1.540×10⁻⁶）。｜修复要求：按 θ=10⁷、rotary_dim=64 重算并改写为 1.66×10⁻⁷（或删去该数值，只保留"θ=10⁷"）；"支持 262144 位置"的因果说明若保留需给出依据。

- 级别：重要｜来源：仓库现状（本页 research/ 目录）＋规范 `guides/model-dataflow.md`「实测」｜位置：§6 核对方式 第 3 段｜问题：页面称实测脚本与原始输出"存于 research/ 目录"并逐一列出文件名，但该目录现存只有 measured.md 与 prereq-audit.md（其余 264 个文件已在 commit 13ead44 删除），指向的是已被移除的文件路径，且规范要求正文引用实测结论时只写"实测得到"、清单记在 research/measured.md。｜引文依据：页面"实测脚本与一次性跑完全部脚本的原始输出（392 行）存于 research/ 目录：read_headers.py … measured-output.txt 为原始输出，verify_page_numbers.py…"；`find wiki/qwen3-5-dataflow -type f` 仅得 index.html、research/measured.md、research/prereq-audit.md；规范"实测脚本与运行输出不随仓库分发，清单记在 research/measured.md；页面正文引用实测结论时写清「实测得到」而不是指向已移除的文件路径"。｜修复要求：删去正文中"存于 research/ 目录"及逐个脚本文件名的表述，改为说明各结论为"实测得到"；脚本清单改指向 research/measured.md（并核对该清单确实登记了这些脚本），同时把"来源与范围说明"表中"脚本与原始输出见 research/"一并改写。

- 级别：重要｜来源：外部来源 HF `Qwen/Qwen3.8-Flash-Next`（config.json、README.md front-matter、LICENSE）｜位置：§5 对比表「发布」行｜问题：把 Qwen3.8-Flash-Next 的许可写成 MIT，与实际不符。｜引文依据：该仓库 README front-matter `license: other`、`license_name: qwen-community-1.0`、`license_link: LICENSE`；LICENSE 首行"Qwen Community License 1.0"。（发布日期 createdAt 为 2026-08-24，与表中 2026-08-26 亦不一致。）｜修复要求：把 MIT 改为 Qwen Community License 1.0（或删去许可一栏）；发布日期核对为仓库/博客实际日期后再写。

- 级别：轻微｜来源：不适用（表述）｜位置：§2「交互式数据流」首段｜问题：元话语，用"本页"指向自身叙事。｜引文依据："下图是实际前向路径，也是本页的主体。"｜修复要求：删去"也是本页的主体"，直接说明图的含义。

- 级别：轻微｜来源：不适用（表述）｜位置：视图 1（整体总览）Layer 2 节点说明｜问题：以"注意……"引导，属规范明列的"需要注意的是"式元话语。｜引文依据："第 2 层仍为 GDN。注意本代没有 N-gram 查表注入——那是 Qwen3.8-Flash-Next 才有的结构。"｜修复要求：改为直陈事实（如"本代无 N-gram 查表注入"），去掉"注意"。

- 级别：轻微｜来源：不适用（表述）｜位置：§1 表后 note 块｜问题：以"容易被……误导的点"起头的引导腔，属 AI 拼接腔的加壳。｜引文依据："两个容易被字面值误导的点。其一，……其二，……"｜修复要求：直接陈述两个事实（attn_output_gate 不被读取、full_attention 即真全注意力），删去"容易被字面值误导"的框架语。

- 级别：轻微｜来源：不适用（表述：无来源判断写成结论）｜位置：§5 末段｜问题：把对两代路线的评价（"都是在……这一个方向上的推进""更保守稳妥的稀疏化路线"）写成结论，属无来源支持的主观判断。｜引文依据："对照着看，Qwen3.8-Flash-Next 的四项增量……都是在「长上下文推理成本」这一个方向上的推进；而 Qwen3.5 旗舰把 45/60 的层做成固定状态、KV 头压到 2 个、配 MTP，走的是更保守稳妥的稀疏化路线。"｜修复要求：改为只陈述可核对的机制差异，删去"更保守稳妥""这一个方向上的推进"一类评价；若保留评价须标注为分析而非结论。

- 级别：轻微｜来源：不适用（表述：无来源定性）｜位置：§5 首段｜问题：对 Qwen3.8-Flash-Next 的"定位"给出无来源的定性。｜引文依据："但 Qwen3.8-Flash-Next（定位 Qwen4 的架构验证版）在其上叠加了三件 Qwen3.5 没有的东西，也去掉了一件："｜修复要求：删去"（定位 Qwen4 的架构验证版）"或给出可定位依据。

- 级别：轻微｜来源：官方 README（Context Length: 262,144 natively）｜位置：§4.6 末段｜问题：把自身推断"max_position_embeddings 约束的是位置轴上限而非 token 数上限"写成结论；官方材料只表述为原生 262,144 token 上下文。｜引文依据：页面"因此 max_position_embeddings=262144 约束的是位置轴上限而非 token 数上限——序列长度预算与位置轴预算是两笔账"；Qwen3.5-397B-A17B README"Context Length: 262,144 natively and extensible up to 1,010,000 tokens"。｜修复要求：标注为推断（如"按该位置分配规则推得"），或删去"约束的是位置轴上限而非 token 数上限"的断言。

- 级别：轻微｜来源：官方 config.json + 官方 checkpoint（张量头）｜位置：视图 4（MoE 内部）sum 节点说明｜问题："全量 6,442,450,944"标注含糊：该值只是 512 个路由专家之和，并非该层 MoE 的全部参数（还应含共享专家、路由器与门）。｜引文依据：节点"单层 MoE 激活 140,513,280（……），全量 6,442,450,944，激活占 2.18%"；512×12,582,912 = 6,442,450,944，而该层 MoE 全部参数为 6,442,450,944+12,582,912+2,097,152+4,096 = 6,457,135,104。｜修复要求：把"全量"改为"512 路由专家合计"，或补上共享专家/路由器/门后的该层 MoE 总参数；激活占比 2.18% 若改分母需重算（用 6,457,135,104 时为 2.176%）。

## 结论

- 统计：阻断 0 / 重要 6 / 轻微 7
- 处置：修复（重要问题逐条修正后重新核对来源；轻微问题按修复要求改写表述；§6 的 research/ 引用按规范改为"实测得到"并指向 measured.md，随后运行 `.dojo/scripts/validate.py wiki/qwen3-5-dataflow/index.html`）
