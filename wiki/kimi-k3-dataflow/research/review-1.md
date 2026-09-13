<!-- review-meta
round: 1
page: wiki/kimi-k3-dataflow/index.html
reviewed_content_sha256: 4ff578b425684ea3
-->
# Kimi K3 前向数据流 审查记录（第 1 轮）

- 页面版本：`72164e2bc65938ca68691d933d41b60d64083ef5`（`git hash-object wiki/kimi-k3-dataflow/index.html`）
- 审查时间：2026-09-13
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 页面类型与规范：`<head>` 中 `dojo:type = dataflow`，据此采用 `guides/model-dataflow.md`；「表述」维度另按该规范「表述」小节与 `guides/concept/check.md` 2.2 第 12 项执行
- 已完整阅读章节（按顺序）：
  1. `<head>`（`description` / `dojo:summary` / `dojo:type` / `dojo:topics` / `dojo:tag` 与本地资源引用）
  2. `1. 关键规格`（18 行规格表逐行）
  3. `2. 交互式数据流`（含三个内联 `<script>` 块：页面 UI 脚本、`var VIEWS` + cytoscape 脚本、站内链接脚本）
  4. `3. 要点`（整体结构 / KDA / Gated MLA / Stable Latent MoE / Block AttnRes / Dense FFN 六节全部列表项）
  5. `4. 视觉编码器与多模态融合`（4.1–4.5 含两张表）
  6. `5. 长上下文开销的来源`（开销表逐行复算）
  7. `6. 核对方式`
  8. `来源与范围说明`（含表与文末核对说明段）
  9. `var VIEWS` 八个视图的全部节点、边与工具提示文案（overview 15 节点/14 边、dense 6/6、attnres 7/6、kda 23/23、mla 24/24、moe 12/13、vision 11/10、fusion 6/7）

核对所用来源（均为本次实时抓取，非页面自带快照）：
- 官方 `config.json`：huggingface.co/moonshotai/Kimi-K3/raw/main/config.json（逐字段比对）
- 官方源码 `modeling_kimi_linear.py`、`modeling_kimi_k3.py`（逐函数比对）
- 官方权重索引 `model.safetensors.index.json`（497,220 个张量）
- 官方模型卡 `README.md`
- 官方技术报告 arXiv:2607.24653（HTML 版）
- 外部算子库 fla-org/flash-linear-attention：`fla/ops/kda/gate.py`、`fla/ops/kda/chunk.py`

## 机械验证结果

1. `.dojo/scripts/validate.py wiki/kimi-k3-dataflow/index.html` → `validation ok: wiki/kimi-k3-dataflow/index.html`，退出码 0。
2. 三个内联 `<script>` 块抽取为 `/tmp/k3df_js_{0,1,2}.js` 后 `node --check` → 三块均 OK（6905 / 29866 / 349 字节）。
3. 页内 7 个前置概念链接（`moonvit-v2`、`nope`、`rope`、`kda`、`mla`、`kv-cache`、`qwen3-8-flash-next-dataflow`）均存在 `index.html`；无「（待生成）」占位。
4. 页内无 `<details>` 折叠块、无图注；`$...$` 定界符外的数学 Unicode 字符检查由 `validate.py` 判定通过（该脚本显式豁免 `×`、`→`、`≤`、`≠`、`Σ` 等中文排版字符）。

## 已核对通过项（摘要，均给出对照位置）

- 规模与层型：93 层、69 KDA + 24 Gated MLA、hidden 7168、96 头、head_dim 128、1M 上下文、vocab 163840、`tie_word_embeddings=false`、`num_nextn_predict_layers=0`，与 `config.json` 同名字段逐个一致；`full_attn_layers=[4,8,…,88,92,93]`（1-based，共 24）= 页面所述 0-based `3,7,…,87` 加 `91,92`，`kda_layers` 69 项含 1-based 1，与「第 0 层为 KDA」一致。
- 第 0 层 Dense FFN：`first_k_dense_replace=1`；`modeling_kimi_linear.py` 中 `layer_idx >= config.first_k_dense_replace` 才建 `block_sparse_moe`，否则建 `KimiMLP`（`intermediate_size=33792`）。
- MoE 门控：`sigmoid` 打分、`scores_for_choice = scores + e_score_correction_bias` 仅用于选择、权重取 `scores.gather` 后按 `moe_renormalize` 归一再乘 `routed_scaling_factor=1.0`，与 `KimiMoEGate.forward` 逐行一致；Quantile Balancing 的「取 margin 的 1−k/n 分位数」与技术报告「b^j(t+1) ← −quantile₁₋ₖ/ₙ(s₍:,j₎ − α⁽ᵗ⁾)」一致。
- Stable LatentMoE：down 7168→3584、`routed_expert_norm`、up 3584→7168、共享专家 `moe_intermediate_size × num_shared_experts = 6144` 的 `KimiMLP`、`w1/w2/w3 = gate/down/up`，与 `KimiSparseMoeBlock` / `KimiBlockSparseMLP` 一致。
- KDA：`f_a_proj` 7168→128、`f_b_proj` 128→12288、`b_proj` 7168→96、`dt_bias` 12288、`A_log` (96,)、`use_full_rank_gate` 的 `g_proj` 7168→12288、`o_norm=FusedRMSNormGated(head_dim, sigmoid)`、`use_qk_l2norm_in_kernel=True`，与 `KimiDeltaAttention` 一致；safe_gate 公式 `log α = −5·σ(e^{A_log}(g+dt_bias))` 与 `fla/ops/kda/gate.py` L68/L92 完全一致，非安全版为 `−e^{A_log}·softplus(...)`（L36/L54），页面「不是 softplus 形式」的辨析成立。
- MLA：`q_a_proj` 1536、`q_b_proj` 18432（96×192）、`kv_a_proj_with_mqa` 576（512+64）、`kv_b_proj` 96×256、`scaling = q_head_dim^{-0.5} = 192^{-0.5}`、`rotary_emb = None`、`assert self.use_nope`、输出门 `g_proj().sigmoid()`，与 `KimiMLAAttention` 一致。
- Block AttnRes：块边界 `layer_idx % attn_res_block_size == 0`（0,12,…,84，共 8 处）在 attention 前把「进入该层时的输入」存入 `block_residual` 并把 `prefix_sum` 置空，从 attn 输出重新累计；每层 attention 前（`self_attention_res_*`）与 MLP 前（`mlp_res_*`）各加权一次，模型末尾 `_apply_output_attn_res` 第三次；打分 `norm.weight * proj.weight.squeeze(0)` 逐维相乘求和、softmax 后加权**原始 v**（非归一化后的 k），候选 = 8 快照 + 当前残差流 = 9，与 `_forward_attn_residual` / `_apply_attn_res` 逐行一致。技术报告「partition its layers into 8 blocks … and 9 total blocks when counting the embedding layer」与「第 1 个快照即 embedding」口径相容。
- SiTU-GLU：`β·tanh(g/β)·σ(g)`（β=4）与 `β_l·tanh(u/β_l)`（β_l=25）、坐标上界 100，与 `SituAndMul` 及 `activation_situ_beta/activation_situ_linear_beta` 一致。
- 视觉塔：`patch_embed` Conv2d k=s=14 无 bias（权重 602,112）、`divided_fixed` 64×64 可学习表（4,194,304）按 (h,w) 双线性插值、`t>1` 时 2D 嵌入重复 t 份再加 `time_weight[0:t]`（`persistent=False`、`init_pos_emb_time=4`）、2D RoPE `theta_base=10000` 且网格上限 512×512、偶数槽编码宽 / 奇数槽编码高（源码 docstring 标注与实现相反，页面已注明）、`MoonViTEncoderLayer` 结构（RMSNorm→wqkv 1024→4608→RoPE→cu_seqlens 打包双向注意力→wo 1536→1024；MLP2 1024→4096→1024 gelu tanh；无 bias）、`tpool_patch_merger` 的时间维 `.mean(dim=0)` + `(kh*kw)` 折叠、`PatchMergerMLPV2` 4096→4096→7168 无 bias + `post_norm=RMSNorm(7168, eps=1e-5)` 且 V1 的 `pre_norm` 已移除、塔参数 401,214,464 与 merger 46,144,512（逐项复算：27×14,682,112 + 602,112 + 4,194,304 + 1,024；4096² + 4096×7168 + 7168），与 `modeling_kimi_k3.py` 及 `vision_config` 一致。
- 融合：occupation 表、`cumsum − 1`、`T' = T − #PH + Σn_i`、文本/视觉交错落位、`ignore_index = −100`、占位符多于段数抛 `ValueError`、段多于占位符在 occupation 赋值处 broadcast 报错、纯文本由 `pixel_values is not None and len(...)>0` 守卫、`_apply_attn_res` 等，与 `_merge_input_ids_with_image_features` 及 `forward` 一致。
- 第 5 节开销表逐行复算一致：参考口径 96×(192+128)×2 B = 60 KiB/token/层 ×24 = 1440 KiB/token → 4K 5.625 GiB / 32K 45 / 128K 180 / 256K 360 / 1M 1440 GiB，+ KDA 0.216 GiB 的「合计」列逐行相加无误；潜压缩 576×2 B×24 = 27 KiB/token → 1M 27 GiB、4K 0.105 GiB；KDA 状态 (96×128×128 + 3×12288×3)×2 B×69 = 232,316,928 B = 0.2164 GiB；93/24 → 5.45 TiB 与 25.8% 复算通过。
- `4.4` 视觉 token 示例表 5 行（(1,32,32)→1024 patch→256 token；(1,64,64)→4096→1024；(1,96,96)→9216→2304；(1,72,130)→9360→2340；(4,32,32)→4096→256）逐行复算一致。

## 问题记录

级别：阻断｜来源：官方 `config.json` 的 `quantization_config`、官方权重索引 `model.safetensors.index.json`、Kimi K3 技术报告 MXFP4 小节｜位置：`1. 关键规格`表「其他」行；`4.3 27 层 block：与语言主干的差别`表「量化」行；`vision` 视图 `vtok` 节点说明｜问题：把 MXFP4 量化范围写成「路由专家权重+embedding」，来源不支持该表述——embedding 未被量化，量化范围只有路由专家的 w1/w2/w3；两处表格各错一次，且量化范围是「关键规格」的组成项｜引文依据：`config.json` 中 `quantization_config.targets = ["Linear"]`，`ignore = ["re:.*self_attn.*","re:.*shared_experts.*","re:.*mlp\\.(gate|up|gate_up|down)_proj.*","re:.*lm_head.*","re:.*vision_tower.*","re:.*mm_projector.*"]`，其中没有任何 embedding 项（`nn.Embedding` 也不在 `Linear` 目标内）；权重索引中 `language_model.model.embed_tokens.weight` 仅有一个键，而同表的路由专家是成对出现「`language_model.model.layers.12.block_sparse_moe.experts.895.w3.weight_packed` + `…w3.weight_scale`」；技术报告原文「we quantize the MoE expert weights — which dominate the model's parameter memory — to MXFP4 … while all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision」｜修复要求：两处改为「范围＝路由专家（w1/w2/w3）权重」，并把 embedding 与 lm_head、self_attn、共享专家、视觉塔、mm_projector 一并列入豁免名单；若坚持 embedding 被量化，必须补出支持它的官方材料位置。

级别：重要｜来源：官方 `config.json` 与模型卡 `README.md`｜位置：`4. 视觉编码器与多模态融合`引言（「共 447.4M，占全模型约 0.16%」）；`4.3` 表「参数占比」行（「447.4M（0.16%）」「其余 99.84%」）｜问题：视觉侧占比算错一个量级。447.4M / 2.78T = 1.61e-4，应为约 0.016%，页面写作 0.16%；据此推出的「其余 99.84%」同步错误｜引文依据：`447,358,976 / 2,780,000,000,000 = 0.000161`；模型卡「Total Parameters 2.8T」「Parameters of Vision Encoder 401M」；页面自身的 447.4M = 401,214,464 + 46,144,512 复算无误，仅比值一步出错｜修复要求：两处改为「约 0.016%」，并把「其余 99.84%」改为「其余 99.98%」。

级别：重要｜来源：`guides/model-dataflow.md`「实测」小节｜位置：页头 `page-meta`（「（含 research/ 本机实测）」）；`6. 核对方式`（「两个文件与 config 的快照存于 research/」「原始输出见 research/measured-output.txt」）；`来源与范围说明`表（「快照存 research/」「脚本与原始输出见 research/（probe1~3、measured-output.txt）」）｜问题：正文四处指向本页 `research/` 下的具体产物，而该目录实际只剩 `measured.md` 一个文件，读者按路径查不到任何内容；规范明确禁止在正文里指向已移除的文件路径｜引文依据：规范原文「实测脚本与运行输出不随仓库分发，清单记在 research/measured.md；页面正文引用实测结论时写清「实测得到」而不是指向已移除的文件路径」；`wiki/kimi-k3-dataflow/research/` 实际内容仅 `measured.md`，且该清单把 `modeling_kimi_k3.py` 登记为 0.1 KB、其余产物登记为 0.0 KB，与「1,317 行源码」「一份 config」的真实体量不符，登记本身也不足以支撑追溯｜修复要求：删去 `research/measured-output.txt`、`research/` 快照、`probe1~3` 等具体路径，改写为「实测得到」，只保留「清单记在 research/measured.md」这一处指引；同时补正 `research/measured.md` 中登记的体积。

级别：重要｜来源：`guides/model-dataflow.md`「表述」小节、`guides/concept/check.md` 2.2 第 12 项｜位置：页头 `page-meta` 行；`来源与范围说明`后的「核对说明」段；`dense` 视图 `act` 节点说明与 `kda` 视图 `bproj` 节点说明；`5. 长上下文开销的来源`第 1 段｜问题：正文写入写作/修订过程与自我评价——「2026-08-07 按官方源码修订 Block AttnRes」「2026-09-03 补充视觉编码器、多模态融合与长上下文开销」「数值与公式逐项核对」「config 数值与公式均以官方 config.json、官方实现源码…逐项核实」，属调试叙事与临场评价，规范要求只留结论与依据；且该自评与本次审查查出的错误（量化范围含 embedding、视觉占比 0.16%）相矛盾，构成无来源支持的判断。同类元话语还有工具提示中的「注意此 β=4 是激活尺度」「注意与 SiTU 激活的 β=4 是不同符号」与「两个口径分开陈述。」｜引文依据：规范「不写「本页将展示」「下面我们来看」一类的引导语，直接进入路径」「不把调试过程写进正文；只留结论与依据」；check.md 2.2-12「元话语（「本页将…」「下面来看…」「需要注意的是」）…调试叙事与临场评价」｜修复要求：`page-meta` 只保留「更新于 <日期>」与来源指向；删除「核对说明」段的修订流水与自我评价，仅保留覆盖范围与不在范围内的说明；工具提示与正文中的「注意此…」「两个口径分开陈述」改为直接陈述（如「β=4 为激活尺度，β_w 为写入强度，两者不同」）。

级别：轻微｜来源：`guides/model-dataflow.md`「视图」小节与「发布前检查」｜位置：`2. 交互式数据流`｜问题：第 2 节的视图内容（页签、图、图例、面包屑，以及每个节点承载的输入/输出维度与公式）全部由内联脚本生成，`<div class="viz">` 内部只有空容器；禁用脚本后该节只剩一个空框，节点上的形状与公式信息无处可读。规范要求「视图内容写在 HTML 里，脚本只负责切换显隐」，并把「交互视图在无脚本时仍可读」列为发布前检查项。（本仓其余 5 个 dataflow 页同样只用 cytoscape 脚本承载视图，属全类型共性问题，修复可一并落到规范与模板）｜引文依据：规范「脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」「交互视图在无脚本时仍可读」；页内为 `document.getElementById('tabs').innerHTML = TABS.map(...)` 与 `cy.add(nodes.concat(edges))`，HTML 侧无对应内容｜修复要求：至少为 overview 视图补一份 HTML 或内联 SVG 静态主干（模块顺序 + 每步形状 + 关键公式），脚本只负责切换显隐。

级别：轻微｜来源：fla 官方算子 `fla/ops/kda/chunk.py`｜位置：`3. 要点`「KDA（Kimi Delta Attention，69 层）」第 4 条 与 `kda` 视图 `kda` 节点公式｜问题：同页两处 KDA 读出公式不一致。正文写 $o = S^\top\hat q$，图示写 $o_t = \frac{1}{\sqrt{128}}S_t^\top\hat q_t$；按正文口径复算会得到放大 $\sqrt{128}\approx 11.3$ 倍的输出，公式不可复算｜引文依据：`fla/ops/kda/chunk.py` L466-467「if scale is None: scale = K ** -0.5」，`K = head_dim = 128`｜修复要求：正文补上缩放因子，写成 $o_t = \frac{1}{\sqrt{128}}S_t^\top\hat q_t$。

级别：轻微｜来源：官方 `config.json`（`vision_config`）与本页 `4.4` 示例表｜位置：`vision` 视图 `tpool` 节点说明｜问题：写作「一张 448×444 无论 1 帧还是 4 帧都是同样多 token」。448×444 既不是 patch=14 的整数倍，也不等于同页任何示例尺寸，属笔误｜引文依据：`vision_config.patch_size = 14`，448/14 = 32、444/14 = 31.71；`4.4` 表与 `vision` 视图 `vtok` 节点均用 448×448→256 token｜修复要求：改为「一张 448×448」。

级别：轻微｜来源：官方 `config.json` 与 `modeling_kimi_linear.py`｜位置：页首 `page-lead`｜问题：「第 0 层为 KDA + Dense FFN，…；FFN 全部是 Stable Latent MoE（896 路由专家 top-16 + 2 共享专家）」在同一句里自相矛盾：第 0 层已排除在 MoE 之外，却与「全部」并置；页面 `3. 要点` 的小标题写的是「Stable Latent MoE（除第 0 层外全部 FFN）」，两处口径不同｜引文依据：`first_k_dense_replace = 1`；`modeling_kimi_linear.py` 中 `if config.num_experts is not None and layer_idx >= config.first_k_dense_replace and layer_idx % moe_layer_freq == 0:` 才建 `block_sparse_moe`，否则建 `KimiMLP`｜修复要求：导语改为「第 1 层起的 FFN 全部是 Stable Latent MoE」。

级别：轻微｜来源：`guides/concept/style-guide.md` 第 11 节（经 `guides/model-dataflow.md`「表述」小节并入）｜位置：`var VIEWS` 内节点 `io` / `label` 字段与边标签——如 `overview.fuse` 的 `文本 [·,7168] + 视觉 [Σn,7168] -> [·,T′,7168]`、`overview.fuse` 标签 `T → T′`、`kda` 视图 `kda` 节点 `io` 的 `q,k,v,g,β_w,A_log,dt_bias`、边 `['bproj','kda','β_w']`、`vision.px` 说明的 `t ≤ 4`、`vision.blk` 说明的 `头维≠hidden/heads=85.3`｜问题：这些字段经 `nl()`/`esc()` 原样注入 HTML（`nl(s)` 只做转义与换行替换，不解析 `$...$`），页面上以裸 Unicode 数学字符显示，未经 KaTeX 渲染；同一变量在页内出现两种写法——公式里是 `\beta_w`，标签与工具提示里是 `β_w`｜引文依据：style-guide 11 节「页面任何位置出现的数学变量、希腊字母、上下标、数学运算符和关系符都必须包在 `$...$` 或 `$$...$$` 中，由 KaTeX 渲染」「同一变量在页面中保持同一种写法」；`kw` 节点 `label` 由 cytoscape canvas 绘制（`'label':'data(label)'`），无法渲染 KaTeX｜修复要求：`io` 字段改走与 `d` 字段相同的 `renderD()`，把 `β_w`、`Σn`、`T′`、`≤`、`≠` 写成 `$...$`；canvas 标签内的 `β_w`、`T′` 改为不含数学含义的文字（如「写门」「扩展后长度」）。

级别：轻微｜来源：技术报告与官方 `config.json`（专家规模）｜位置：`3. 要点`「整体结构：三条信息流」第 3 条｜问题：「扩到 896 个专家而通信/显存不随专家数线性增长」是一句无来源支持的判断，且按本页自己的规格数据可算出反例——896 个路由专家 × 3072 中间维 × 3584 隐维的权重随专家数线性增长，占 2.78T 中约 97.9%，读作「显存不随专家数线性增长」会得出错误结论｜引文依据：`num_experts = 896`、`moe_intermediate_size = 3072`、`routed_expert_hidden_size = 3584`；本页 `1. 关键规格`「896 路由专家，每 token 激活 16 个」；技术报告「activates 16 out of 896 experts」｜修复要求：改写为「每 token 只激活 16 个专家，token 级通信量与激活显存不随专家数增长；专家权重仍随专家数线性增长」。

级别：轻微｜来源：`modeling_kimi_k3.py` 的 `KimiK3ForConditionalGeneration.forward`｜位置：`4.5 融合：占位符扩展，序列变长`最后一段｜问题：把 decode 分支的守卫条件写成「decode 单步（len(ids)==1 且带 KV cache）」，漏掉官方分支还要求 `pixel_values is not None`，按页面表述会以为只要单 token + 有 cache 就走该分支｜引文依据：`elif (past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1):`｜修复要求：补上「且本次仍传入 pixel_values」这一条件。
