<!-- review-meta
round: 6
page: wiki/mxfp4-qat/index.html
reviewed_content_sha256: 458dc26b325ba628
-->
# MXFP4 量化感知训练审查记录（第 6 轮）

- 页面版本：25dbcbb9c5c4614151747efe1d853c72e48511cc
- 审查时间：2026-09-13 21:51
- 审查者：独立子代理（未参与写作，未读取本页 research/）
- 已完整阅读章节：核心问题 / 常见误解 / 1. MoE 专家权重 / 2. MXFP4 编码 / 3. QAT 机制 / 4. RL 一致性 / 5. 选择性量化 / 来源与范围说明（含全部 <details> 折叠块、图注与代码块）

## 来源核对依据（check.md §2.2.3）

- K3 报告 arXiv:2607.24653 §4.1.4：核对原文片段——"we quantize the MoE expert weights … to MXFP4"，"with activations computed in MXFP8"，"all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision"，"throughout the entire post-training stage, covering both SFT and RL"，"rollout and training share the same quantization scheme — eliminating the train–inference mismatch"，"Draft fine-tuning follows the post-training QAT configuration … with MoE expert weights in MXFP4 and their input activations in MXFP8"；Abstract "a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters"；Table 1 "Total Parameters 2.78T"。与 C1/C2/C3/C4/N6/N7 逐条一致。
- config.json（HuggingFace moonshotai/Kimi-K3，curl 直取）：text_config.quantization_config 与页面折叠块逐字一致（format "mxfp4-pack-quantized"、targets ["Linear"]、group_size 32、num_bits 4、symmetric true、strategy "group"、scale_dtype "torch.uint8"、ignore 六条正则 `self_attn`/`shared_experts`/`mlp\.(gate|up|gate_up|down)_proj`/`lm_head`/`vision_tower`/`mm_projector`）；text_config 值 hidden_size 7168、num_hidden_layers 93、routed_expert_hidden_size 3584、moe_intermediate_size 3072、first_k_dense_replace 1、num_experts 896、num_experts_per_token 16、num_shared_experts 2。与 C8/N1/N2/N4 一致，正文"896 是每层专家数"成立。
- ONNX "Float stored in 4 bits"：E2M1 exponent bias 1；位模式 000–111 依次为 0、0.5、1、1.5、2、3、4、6；Cast 表 x>6→6、x<-6→-6、NaN→6、-Inf→-6。与 C9/C5 及"简化条件"一致。
- ONNX "Float stored in 8 bits" E8M0 表：Exponent bias 127；Min 00000000₂=2⁻¹²⁷；Max 11111110₂=2¹²⁷；NaN 11111111₂。与 C7 一致。（OCP MX v1.0 PDF 未直接抓取；页面并列引用的 ONNX 文档已核实同一组数值与饱和规则，视为已核对。）
- ml_dtypes float4_e2m1fn："8 distinct positive magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, and 6"，bias 1。与 C9 一致。
- 数值复算：3×3584×3072=33,030,144；896×92×33.03M≈2.7227T；BF16 5.445 TB；MXFP4（0.5 B/参数 + 1 B/32 scale）1.446 TB；5.44/1.445≈3.76×；4+8/32=4.25 bit、16/4.25≈3.76×。与 N3/F3 及正文一致。另：每专家 33,030,144 参数 × 0.53125 B/参数 = 17,547,264 B，与外部公开资料吻合。
- 代码：在本地执行页面折叠块中的 Python，输出与"预期输出"逐行完全一致（block1/block2 量化、反量化、误差，w=0.8→w_hat=0.75、偏差 0.0500、STE 传回 1.0）。
- 链接：`../moe-serving/index.html`、`../quantization-basics/index.html` 均真实存在；`python3 .dojo/scripts/validate.py wiki/mxfp4-qat/index.html` 返回 validation ok。

## 问题

- [轻微·可读性] 核心问题第 1 条解答折叠块（正文 line 122）：把"896 个路由专家的权重"与"合计约 2.72T 参数"直接并置，未注明 896 是**每层**专家数（全模型 896 × 92 = 82,432 个），读者若按总数为 896 计算会得到约 3B 参数/专家，与第 1 章"每个专家约 33.03M"相差 92 倍。｜引文依据：不适用（内部一致性问题；config.json `num_experts: 896` 为每层值，overview.html 同一事实写作"896 个路由专家（每层 896 个，92 个 MoE 层）"）｜修复要求：在该折叠块"896"处补"每层"限定，如"896 个/层 的路由专家（全模型 896 × 92 = 82,432 个）…合计约 2.72T"，与正文及 overview.html 的表述统一。｜修复：｜复验：
- [轻微·可读性] 第 2 章"本章问题"第 1 题解答（正文 line 286）："最近邻量化到 E2M1 可表示值得 $q=…$"缺标点，"可表示值"与"得"粘连成词"可表示值得"，读来不通。｜引文依据：不适用｜修复要求：改为"最近邻量化到 E2M1 可表示值，得 $q=…$"。｜修复：｜复验：
- [轻微·技术] 第 1 章末段（正文 line 191）：句末 <sup>[C3]</sup> 覆盖到"既不是显存瓶颈，也不是量化首选"，但"不是量化首选"属动机推断，并非 C3 的陈述内容；本页在"来源与范围说明 · 辅助解释与类比边界"与第 5 章本章问题 2 均把该动机标注为"本页推断"，同一论断两处归属不一致。｜引文依据：C3 原文 "all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision."（只给划分事实，未给划分理由）｜修复要求：将 <sup>[C3]</sup> 限定到"非专家组件保持高精度"这一事实（或把"也不是量化首选"改写为带"本页推断"的表述），使归属与第 5 章及来源说明一致。｜修复：｜复验：
- [轻微·格式] "来源与范围说明 · 论断与来源（C）"列表：编号顺序为 C1、C2、C3、C4、C5、C7、C9、C6、C8，C6 排在 C9 之后，编号不升序，编号与内容本身未错位。｜引文依据：不适用｜修复要求：按编号升序重排为 C1…C9（不改动任何条目内容）。｜修复：｜复验：

## 结论

- 处置：可发布（0 阻断 / 0 重要；4 项轻微不影响正确性与主线，建议顺手修复）
- 事实、公式、数字与代码均逐项回源核对通过：K3 §4.1.4 四段引文、Abstract/Table 1、config.json 全字段、ONNX E2M1/E8M0 表、ml_dtypes、算术复算与代码实跑全部一致；无定位不到、无来源不支持、无同页矛盾、无 summary/overview 数字不一致、无引文编号错位。
- 表述维度未发现元话语、会话指代（我/我们/你）、调试叙事、临场评价或 AI 拼接腔；"本页推断/本页不展开"属 style-guide 允许的自称用法。

统计：阻断 0 / 重要 0 / 轻微 4

> 本轮所列问题的处理结果见 `minor-fixes.md`。
