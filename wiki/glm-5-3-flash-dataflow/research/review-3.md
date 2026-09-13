<!-- review-meta
round: 3
page: wiki/glm-5-3-flash-dataflow/index.html
reviewed_content_sha256: 3825ea6640e47f73
-->
# GLM-5.3-Flash 前向数据流审查记录（第 3 轮）

- 页面版本：`wiki/glm-5-3-flash-dataflow/index.html` 工作树哈希 4142f61dd7a9a1afde6eef342f80f21139447178（98626 字节）。审查期间工作树被改动过一次（98641 字节 → 98626 字节，19:20），本记录以该哈希版本为准。
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下的规划、修复与前序审查记录）
- 已完整阅读章节（按顺序）：1 关键规格 / 2 整体数据流 / 3 单层内部数据流 / 4 mHC：4 路残差流怎么读写 / 5 KDA 层 / 6 DSA 层 / 7 MoE 路由 / 8 位置信息从哪来 / 9 长上下文下的实际收益 / 10 FP8 量化的覆盖范围 / 11 多模态 / 12 MTP 层 / 13 核对方式 / 来源与范围说明
- 实际打开的外部来源：`zai-org/GLM-5.3-Flash` 的 `config.json`、`README.md`、`model.safetensors.index.json`，以及 62 个 safetensors 分片的 JSON 头（HTTP Range 实取，共解析 76,108 个张量，与页面口径一致）；transformers v5.16.1 的 `modeling_glm5_next.py`、`configuration_glm5_next.py`、`src/transformers/cache_utils.py`；arXiv:2602.15763v2 全文 HTML。
- 已执行的机械核对：`.dojo/scripts/validate.py` 返回 `validation ok`。未执行无头浏览器渲染核对（本机无 chromium / playwright），公式与图示按静态审查。
- 本轮核对通过的关键数字（全部与来源精确一致，无需修改）：总参数 321,323,031,390（排除 37,338 个 `weight_scale_inv` 后 76,108 个张量按 shape 累加，逐字节相符）；单 token 激活 17,376,348,990，六项分项之和相符；KDA 单层 137,732,288 × 34 = 4,682,897,792；DSA 单层 124,914,432（indexer 7,471,872，占 5.98%）× 11 = 1,374,058,752；mHC 393,243 × 90 = 35,391,870；embed + lm_head 1,268,776,960；稠密 MLP 452,984,832；视觉塔 563,627,008（visual 命名空间实测差值为 0）；MTP 第 45 层 7,432,592,416（其中 routed 专家 7,247,757,312）；37,338 个 scale 张量全部符合 ⌈N/128⌉×⌈K/128⌉，无一例外；`index.json` 的 `total_size` 328,326,771,576 与逐张量按 dtype 累加相符。§9 长上下文表（0.204/0.264/22.89%…16.661/67.588/75.35%）、交叉点 2855 token、4288 token 与 3.54% 反算、§6 k-pool 表（候选池数、稀疏度、打分量 25.0%）均可复算且正确。Sinkhorn 迭代顺序（先列归一化 + 19 轮「行、列」）、`post=2σ(·)`、`pre=σ(·)+ε`、`hc_head` 为无权均值、MoE `noaux_tc` 公式与 2.5 权重和、SwiGLU 不对称截断、`get_placeholder_mask` 的 cumsum 判式、`_keep_in_fp32_modules_strict` 字面值、`layers\.45\.` 忽略规则、conv 状态宽度取 `conv_kernel_size`（=4 而非 3）等，均与源码 / 配置逐条吻合。

## 问题

- [重要·技术] §10 第 936 行（并见同节第 929 行）：`_keep_in_fp32_modules_strict` 的四类张量被断言「checkpoint 实测这些张量的 dtype 确实全是 F32，与声明一致」，但 checkpoint 的卷积张量是 BF16；第 929 行「其中卷积核、$A$ 与 $b_{dt}$ 更进一步留在 FP32」同错。｜引文依据：实取 62 个分片 JSON 头，`model.language_model.layers.0.self_attn.q_conv1d.weight` 的 `shape=[8192,1,4]`、`dtype=BF16`；名称含 `conv1d` 的 102 个张量（34 层 × q/k/v）全部为 BF16。同组另三类确为 F32：`e_score_correction_bias` 43 个全 F32、`dt_bias` 34 个全 F32、`A_log` 34 个全 F32。源码 `modeling_glm5_next.py` L1358 只声明加载/量化期保留 fp32，并不等于 checkpoint 以 fp32 存储；config 的 `quantization_config.modules_to_not_convert` 含 `model.layers.0.self_attn.q_conv1d` 等，其保持 BF16。｜修复要求：把第 936 行改为「checkpoint 中 `e_score_correction_bias` / `dt_bias` / `A_log` 为 F32，卷积权重为 BF16；`_keep_in_fp32_modules_strict` 约束的是加载与量化时的精度选择」，并把第 929 行括号内的「卷积核……留在 FP32」改为「卷积核未量化（BF16）」。｜修复：｜复验：

- [重要·技术] §3 第 316 行、§6 第 784 行、§13 第 985 行、[C1] 第 1008 行：以 checkpoint 为口径的计数写成 11，实际为 12（漏算 MTP 第 45 层）。第 784 行「checkpoint 里恰好有 11 组 indexer 张量，层号为 $3,7,\dots,43$」与 §12「MTP 层带完整 indexer」在同一页内互相矛盾。｜引文依据：`model.safetensors.index.json` 中 `layers.N.self_attn.indexer.wq_b.weight`（及 `indexer.wk.weight`、`indexer.weights_proj.weight`）的层号集合为 [3,7,11,15,19,23,27,31,35,39,43,45]，共 12 个；`layers.N.self_attn.kv_a_proj_with_mqa.weight` 共 24 个张量 = 12 层 ×（weight + weight_scale_inv），同样含第 45 层。主干 45 层内确为 11。｜修复要求：把第 784 行改为「主干 45 层中 11 组（层号 $3,7,\dots,43$），另有 MTP 第 45 层自带一组，checkpoint 合计 12 组」；第 985 行「11 组 indexer 的层号」与第 316 行、[C1] 的「11 个层带 `self_attn.kv_a_proj_with_mqa.weight`」同样限定为「主干 11 个层」。｜修复：｜复验：

- [重要·技术] 来源与范围说明·范围之外第 1 条（第 1052 行）：「全文未出现线性注意力或 mHC」与报告不符——报告确有线注意力表述。｜引文依据：arXiv:2602.15763v2 §2.1.2「Ablation Study of Efficient Attention Variants」原文：“Gated DeltaNet (GDN) [ 54 ] : A linear attention variant that replaces the quadratic softmax attention computation with a gated linear recurrence, reducing the computational cost of attention from quadratic to linear in sequence length.”（同节另一处亦出现 “linear attention”；「mHC」「Hyper-Connection」检索确为 0 处，这一半成立。）同段其余引用均准确：§2.1 原文“GLM-5 scales to 256 experts and reduces its layer count to 80… results in a 744B parameter model (40B active parameters)”；DSA 动机原文“proving that 90% of attention entries in long contexts are indeed redundant. DSA reduces the attention computation by roughly 1.5-2× for long sequences”。｜修复要求：删去「或线性注意力」，或改为「不含 KDA / mHC 作为其架构组件（仅在 §2.1.2 比较了 GDN 等线性注意力变体）」。｜修复：｜复验：

- [重要·格式] §5 序列长度符号与全页不一致，且与状态符号撞形：§5 第 615 行用 $L$（「$t = 1,\dots,L$」），同一段又用 $S_t$、$S_0$ 表状态矩阵；而 §6 第 741、753 行、§9 第 910 行、[F1] 第 1018 行、[N5] 第 1029 行、[N8] 第 1032 行、[N9] 第 1033 行一律用 $S$ 表序列长度。§3 第 316 行、§6 第 784 行、§13 第 985 行、[C1] 第 1008 行：以 checkpoint 为口径的计数写成 11，实际为 12（漏算 MTP 第 45 层）。第 784 行「checkpoint 里恰好有 11 组 indexer 张量，层号为 $3,7,\dots,43$」与 §12「MTP 层带完整 indexer」在同一页内互相矛盾。｜引文依据：`model.safetensors.index.json` 中 `layers.N.self_attn.indexer.wq_b.weight`（及 `indexer.wk.weight`、`indexer.weights_proj.weight`）的层号集合为 [3,7,11,15,19,23,27,31,35,39,43,45]，共 12 个；`layers.N.self_attn.kv_a_proj_with_mqa.weight` 共 24 个张量 = 12 层 ×（weight + weight_scale_inv），同样含第 45 层。主干 45 层内确为 11。｜修复要求：全页统一序列长度符号：把 §6 第 741、753 行、§9 第 910 行、[F1]、[N5]（“B=1、S=6”）、[N8]、[N9] 中的 $S$（表长度者）改为 $L$，状态一律保留 $S_t$；改后各表「序列长度」列头无需变动。｜修复：｜复验：

- [轻微·技术] §13 第 985 行：「每个约 260 KB」与实测不符，偏高约 40%。｜引文依据：实取 62 个分片头 JSON 的字节数：最小 42,424 B、中位 185,152 B、最大 191,879 B、均值 183,349 B。｜修复要求：改为「每个约 180 KB」，或写「每个 40–190 KB」。｜修复：｜复验：

- [轻微·技术] [N7]（第 1031 行）：把「`dt_bias`=0」标注为「官方 safe 分支初始化」，但官方 `_init_weights` 只把 `A_log` 置 0，`dt_bias` 是按均匀分布取对数再求反 softplus，并非 0；该标注使实测条件看起来是官方初始化，实际含一处自选构造。｜引文依据：`modeling_glm5_next.py` `_init_weights`：`if module.safe_gate_lower_bound is not None: init.zeros_(module.A_log)`；其后 `init.uniform_(module.dt_bias, a=math.log(1e-3), b=math.log(1e-1)); dt = module.dt_bias.exp().clamp_min(1e-4); init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))`。｜修复要求：改为「`A_log`=0（官方 safe 分支初始化）、`dt_bias`=0（本实测的构造选择，官方初始化为 $\mathrm{softplus}^{-1}(\exp(\mathcal{U}(\ln 10^{-3},\ln 10^{-1})))$）」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 2
- 处置：修复（4 条重要问题中的前两条为与 checkpoint / 报告不符的事实性论断，须删改后重核来源；第 4 条为符号统一；2 条轻微问题一并处理。修完须重跑 `validate.py`，并复验本轮未执行的渲染项。）
- 另记：审查期间页面工作树在 19:20 被改动（原 98641 B 版本中「本页数字的来源分三层，脚本与原始输出都在 `research/` 下」及 [C1]–[N13]、[F1]–[F3] 中指向已移除产物（`p1.out`、`verify_structure.out` 等）的引用，在本轮开始时读取的版本里仍存在，在 19:20 的改动中已被替换为「实测得到」与 `research/measured.md` 指引，重读后确认已闭合，故未计入本轮问题。本轮 6 条问题均以 4142f61d 版本复核。