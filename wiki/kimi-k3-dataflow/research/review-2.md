<!-- review-meta
round: 2
page: wiki/kimi-k3-dataflow/index.html
reviewed_content_sha256: 4ff578b425684ea3
-->
# Kimi K3 前向数据流审查记录（第 2 轮）

- 页面版本：72164e2bc65938ca68691d933d41b60d64083ef5
- 审查时间：2026-09-13 19:00
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 适用规范：`guides/model-dataflow.md`（head `dojo:type=dataflow`），表述与记录格式参照 `guides/concept/check.md`
- 已完整阅读章节：1. 关键规格 / 2. 交互式数据流 / 3. 要点（整体结构、KDA、Gated MLA、Stable Latent MoE、Block AttnRes、Dense FFN）/ 4. 视觉编码器与多模态融合（4.1–4.5）/ 5. 长上下文开销的来源 / 6. 核对方式 / 来源与范围说明；并逐条读完交互视图 8 个视图（overview / dense / kda / mla / moe / attnres / vision / fusion）的全部节点标签、输入输出维度、悬停公式与描述文本。
- 机械核对：`.dojo/scripts/validate.py wiki/kimi-k3-dataflow/index.html` → `validation ok`。站内前置概念链接（moonvit-v2 / nope / rope / kda / mla / kv-cache / situ-glu / block-attnres / stable-latent-moe / qwen3-8-flash-next-dataflow）目标页均存在。
- 来源核对方式：官方 `config.json`（huggingface.co/moonshotai/Kimi-K3）逐字段比对；官方/镜像 `modeling_kimi_linear.py`、`modeling_kimi_k3.py` 及 `fla/layers/kda.py` 比对；数字算式逐条复算。

## 问题

- [阻断·技术] index.html:169（§4 导语）与 index.html:190（§4.3 表「参数占比」行）：视觉塔参数占比写大 10 倍，且「其余 99.84%」随之一并错｜引文依据：页面写「加 merger 的 46,144,512 共 447.4M，占全模型约 0.16%」，表格行「447.4M（0.16%）｜其余 99.84%」；复算：401,214,464 + 46,144,512 = 447,358,976，447,358,976 / 2.78×10^12 = 1.61×10^-4 = **0.0161%**（0.16% 对应的绝对量是 4.45B，非 447M）｜修复要求：两处均改为「约 0.016%」与「其余 99.984%」，改后重新复算。

- [阻断·来源] index.html:226（§6 核对方式）、index.html:235/236（来源表）、index.html:86（page-meta）：正文指向已被移除的 `research/` 文件路径，实测来源无法定位｜引文依据：页面写「原始输出见 `research/measured-output.txt`」「…快照存 `research/`」「脚本与原始输出见 `research/`（probe1~3、measured-output.txt）」；`wiki/kimi-k3-dataflow/research/` 实际只剩 `measured.md` 与 `review-*.md`，且 `research/measured.md` 自述「本页的实测产物原先存放在本目录下，现已从仓库移除」——`probe1_vision.py`/`probe2_fusion.py`/`probe3_memcost.py`/`measured-output.txt` 均已不存在。规范明确：「页面正文引用实测结论时写清『实测得到』而不是指向已移除的文件路径」｜修复要求：删除全部已移除文件的具体路径，改为「本机实测得到」并写明脚本名与运行条件（结构参数与官方 config 对、张量形状与权重索引对、算子语义与源码对）；如仍需给出处，指向 `research/measured.md` 的登记。

- [重要·技术] index.html:441–664（交互视图脚本）：8 个视图的全部内容写在 `<script>` 的 `var VIEWS = {...}` 对象里，无 HTML 承载，脚本失效时视图不可读｜引文依据：`var VIEWS = { overview:{...`；规范「脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」，发布前检查「交互视图在无脚本时仍可读」｜修复要求：把每个视图的路径/形状/结论以 HTML 结构写入正文（或加 `<noscript>` 兜底，逐视图列出节点—维度—公式），脚本仅负责显隐与绘制；保证禁用脚本后仍能读到各视图关键信息。

- [重要·来源] index.html:234/237（来源表）：报告来源仅一处标注 Table 1，其余报告论断未标注节次/行号｜引文依据：来源表「技术报告（2026-07-28，47 页）｜…（架构动机、Quantile Balancing、SiTU、NoPE、Block AttnRes、MoonViT-V2 401M）」整行无节次；仅「总参数 2.78T / 激活 104.2B」注明「技术报告 Table 1」。规范「从报告读到的数字标注报告的节次或行号」，反之按 §2.2「无法给出片段的条目视为未核对」｜修复要求：为每条报告来源的论断（Quantile Balancing 更新规则、SiTU 有界设计、NoPE、Block AttnRes 动机与 O(L·d)→O(N·d)、MoonViT-V2 401M）补报告章节/表号。

- [轻微·技术] index.html:631（vision 视图 tpool 节点 tooltip）：分辨率笔误｜引文依据：「一张 448×444 无论 1 帧还是 4 帧都是同样多 token」；全文其余处均用 448×448（32×32=1024 patch → 256 token），且 444 不是 patch=14 的整数倍｜修复要求：改为「一张 448×448」。

- [轻微·技术] index.html:85（page-lead）与 index.html:101（关键规格表「位置编码」行）：把「全模型 NoPE」当成无条件表述，与页内 §4.2/§4.3 视觉塔「2D RoPE + 可学习方格」并置易生歧义｜引文依据：导语「全模型无位置编码（NoPE）」、规格表「全模型 NoPE（MLA 不施加 RoPE，rotary_emb=None）」；§4.2「语言主干全模型 NoPE（rotary_emb=None），视觉塔反而用满 2D RoPE」｜修复要求：两处「全模型」限定为「语言主干（93 层）」，或补一句「视觉塔除外」。

- [轻微·表述] 元话语与临场措辞｜引文依据：KDA 视图 x 节点 tooltip「DecoderLayer 里 self_attn 的输入,已过 input_layernorm。**下面分六路投影。**」（index.html:521）；§2 导语「**机制解释见下方要点。**」（index.html:109）；三处「**注意**此 $\beta=4$ 是激活尺度,与 KDA 写入强度 $\beta_w$ 是不同符号」（index.html:483、534、602）；「是 Stable **补丁**之一」（index.html:602、604）｜修复要求：删除「下面分六路投影」「机制解释见下方要点」等引导语，直接给结论；「注意…」改为陈述句；「补丁」改为「机制/改动」等中性表述。

- [轻微·格式] index.html:547、587（cytoscape 边标签）：图中标签使用 Unicode 数学字符且未经 KaTeX 渲染，与正文 LaTeX 写法不一致｜引文依据：`['bproj','kda','β_w']`、`['x','gate','g=σ(xWg)']`｜修复要求：统一为纯文本记号（如 `beta_w`、`sigmoid`），或说明图中标签不采用 LaTeX 记号。

## 结论

- 统计：阻断 2 / 重要 2 / 轻微 4
- 处置：修复

## 已核对通过项（备查，无问题）

- 结构参数与官方 `config.json` 逐项一致：num_hidden_layers=93、hidden_size=7168、num_attention_heads=96、head_dim=128、first_k_dense_replace=1、moe_intermediate_size=3072、routed_expert_hidden_size=3584、num_experts=896、num_experts_per_token=16、num_shared_experts=2、latent_moe_use_norm=true、moe_renormalize=true、routed_scaling_factor=1.0、q_lora_rank=1536、kv_lora_rank=512、qk_nope_head_dim=128、qk_rope_head_dim=64、v_head_dim=128、mla_use_output_gate=true、mla_use_nope=true、short_conv_kernel_size=4、gate_lower_bound=-5.0、use_full_rank_gate=true、attn_res_block_size=12、activation_situ_beta=4.0、activation_situ_linear_beta=25.0、vocab_size=163840、tie_word_embeddings=false、num_nextn_predict_layers=0、rms_norm_eps=1e-5、media_placeholder_token_id=163605、intermediate_size=33792、moe_router_activation_func=sigmoid；`linear_attn_config` 的 full_attn_layers=24、kda_layers=69 与页面层分布一致。
- 视觉配置一致：vt_num_hidden_layers=27、vt_hidden_size=1024、vt_num_attention_heads=12、qkv_hidden_size=1536、patch_size=14、init_pos_emb_{height,width}=64、init_pos_emb_time=4、pos_emb_interpolation_mode=bilinear、merge_type=sd2_tpool、merge_kernel_size=[2,2]、activation_func=gelu_pytorch_tanh、text_hidden_size=7168。
- 算式复算通过：patch_embed 参数 1024×3×14×14=602,112；位置表 64×64×1024=4,194,304；单层 4,718,592+1,572,864+2×4,194,304+2×1024=14,682,112；27×14,682,112+602,112+4,194,304+1,024=401,214,464（与报告 401M 吻合）；merger 4096×4096+4096×7168+7168=46,144,512；SiTU 输出上界 β·β_l=4×25=100。
- 第 5 节开销表复算通过：参考口径 60 KiB/token/层（96×(192+128)×2B）；潜压缩口径 576×2B=1.125 KiB/token/层；4K→5.625 GiB、1M→1440 GiB、潜压缩 24 层 1M→27 GiB 均正确；KDA 层状态 3.0 MiB+216 KiB，69 层=0.216 GiB，与序列长度无关；「93 层全 MLA 需 5.45 TiB、层型 24/93 降为 25.8%」正确；448×448 图 256 token × 24 层参考口径 = 360 MiB 正确。
- KDA 机制与官方实现一致：`f_a_proj(hidden→head_dim)`、`f_b_proj(head_dim→projection_size)`、`A_log` 形状 [num_heads]（每头一尺度）、`dt_bias` 形状 [num_heads*head_dim]（每头每维）、`b_proj(hidden→num_heads)`、`g_proj(hidden→projection_size)`（全秩门）；safe_gate 公式 `log_g = gate_lower_bound * sigmoid(exp(A_log)*(g+dt_bias))`，值域 (-5,0)，`exp(-5)≈0.0067` 下限——页面公式与「不是 softplus 形式」的说明均正确。
- MoE 机制与实现一致：sigmoid 打分、`e_score_correction_bias` 只影响 top-16 排序、选中权重按原始分数 renormalize × routed_scaling_factor；共享专家 `intermediate_size = moe_intermediate_size * num_shared_experts = 3072×2 = 6144`；稠密层用 config.intermediate_size=33792。
- AttnRes 描述与报告/模型一致：block 大小 12、93 层分 7×12+1×9 共 8 块、块快照在 attention 前存入、每 decoder 层在 attention 前与 MLP 前各加权一次、末尾 `output_attn_res_norm/proj` 第三次加权、候选数 9；开销由逐层 O(L·d) 降到逐块 O(N·d)。
- 术语符号全文一致，无会话指代（未出现「我/我们/你」）；未发现「（待生成）」占位或失效站内链接。