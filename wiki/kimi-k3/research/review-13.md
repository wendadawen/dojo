<!-- review-meta
round: 13
page: wiki/kimi-k3/index.html
reviewed_content_sha256: 6dc5a963d89c0ae7
-->
# Kimi K3 审查记录（第 13 轮）

- 页面版本：wiki/kimi-k3/index.html（工作树当前版本，未提交）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题（含 5 个解答折叠块）、1. 三维度信息流——K3 架构总览、2. 序列维度——KDA + 混合注意力、3. 深度维度——Block AttnRes、4. 宽度维度——Stable LatentMoE、5. 原生视觉——MoonViT-V2、6. 训练——数据、scaling law、Muon、长上下文扩展、7. 后训练——SFT→RL→MOPD 三阶段、QAT、EAGLE、8. 基础设施——3T 训练 + 1M RL + 推理、9. 性能与评价（含 3 个数据表与 benchmark 条件折叠块）、10. 独立评价、来源与范围说明；含全部折叠块与图注。

## 引文编号专项核对（本轮重点）

核对版本：**arXiv:2607.24653v2**（页面 meta 与 overview 均标注 v2，2026-08-07）。已同时下载 v1 用于对照编号漂移。

本页出现两类编号，需分开处理：

1. 页面自定义标签 `[C1]–[C14]`（论断）、`[F1]–[F8]`（公式）、`[N1]–[N5]`（外部数字）。它们不指向论文文献表，而是「来源与范围说明」中逐条自定义的映射，映射目标为论文 §/Table/Fig 或 config.json。逐条比对抽查（C1→§Abstract/§1/Table 1；C3→§2.1；C4→§2.2；C5/C6→§2.3；C7→§1/§3.2/Fig.7；C9→§3.4；C10→§4.1；C11→§4.1.4；C12→§1/§6.1.4/Table 2；C13/C14→§2.4/Fig.6；F1/F2→§2.1.1；F3→§2.2；F4→§2.3/Eq.11；F5→§2.3.2/Eq.12；F6→§2.3.3/Eq.14；F7→§4.1.3/Eq.15；F8→§4.1.4/Eq.16；N1→Table 1；N2→Table 1/config；N3→Fig.7；N4→Table 2；N5→Table 5）与原文实际位置一致。

2. 真正的论文文献表编号只有两处：`[60]` 与 `[27]`。逐条回源结果：

- `[60]`（页面 §3 两处：「论文 §2.2 引 [60] 称"$N \approx 8$ recovers most of the benefit across model scales"」「论文 §2.2 引 [60] 给出 $N \approx 8$ 在多模型规模下够用的经验结论」）。
  v2 文献表条目原文：`[60] Kimi Team (2026) Attention residuals. Note: Preprint`（Cited by: §1, §2.2, §2.2, §2, §5.2.2, §5.4.2, Abstract）。
  论文 §2.2 原文：`Empirically, N ≈ 8 recovers most of the benefit across model scales [60]; for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer.`
  → 编号与文献表相符，引文内容出自该篇。**一致**。

- `[27]`（页面 §4「§2.3.3 引 [27]，需要学习率 $\gamma$ 且在 896 专家下收敛慢」；来源与范围说明 F6 注「[27] 对应 DeepSeek-V3 technical report（arXiv:2412.19437）」）。
  v2 文献表条目原文：`[27] DeepSeek-AI, A. Liu, B. Feng, ... (2024) DeepSeek-v3 technical report. External Links: 2412.19437`。
  论文 §2.3.3 原文：`The original method updates b with the fixed-step rule b_j^{(t+1)} = b_j^{(t)} + γ sign(ℓ̄ − ℓ_j^{(t)}) [27], for which γ trades off slow adaptation against load oscillation.`
  → 编号与文献表相符，§2.3.3 引 [27] 的正是该 fixed-step sign 更新。**一致**。

- 版本差异实证（说明本页为何必须按 v2 核对）：同一文献表条目「Attention Residuals」在 **v1 中编号为 [59]**，在 v2 中为 [60]；而 DeepSeek-V3 在 v1/v2 均为 [27]。即按 v1 编号引用 `[60]` 会错一位，本页显式标注 v2，两处编号在 v2 下正确。

专项结论：未发现编号与文献表不符、或引文内容与所引文献不符的条目。

## 问题

- [轻微·技术] 第 9 章 benchmark 条件折叠块（「推理努力级别」段）：`除非另有说明，K3 结果在 reasoning effort = max、temperature = 1.0 下取得（§6.1.4 表注）` 的来源定位有误。该「表注」实为 **v2 中 Table 2 的 caption，位于 §6.1.1 Benchmarks 内**；同一表述另见 §6.1.3 Evaluation Configurations。§6.1.4 是 Results 小节，其中没有该表注。
  引文依据：Table 2 caption 原文（§6.1.1）`Unless otherwise noted, Kimi K3 results are obtained with reasoning effort set to max and temperature equal to 1.0.`；§6.1.3 原文 `All Kimi K3 evaluations use reasoning effort max and temperature = 1.0.`
  修复要求：把「§6.1.4 表注」改为「Table 2 表注（§6.1.1；另见 §6.1.3）」。
  修复：｜复验：

- [轻微·技术/来源] 开篇第 2 段（首段结论句）：`并由训练与基础设施支撑约 2.5× scaling 效率`。论文把 2.5× 归因于架构、数据与训练配方的联合效果，基础设施不参与该指标的来源；同页 §1（`§1 声称这些改进共同（collectively）带来约 2.5×……明确把 2.5× 归因于架构、数据与训练配方的联合效果`）、§6（`约 2.5× scaling efficiency 是架构、数据、训练配方共同作用的结果`；`论文用 collectively 描述 2.5× 的来源，未分解架构、数据、训练各贡献多少`）、§10（`论文只给 collectively 的 2.5×，未分解架构/数据/训练各贡献多少`）以及 overview.html（`这是架构+数据+训练配方的联合效果`）均为三因素表述，唯本句多出「基础设施」，构成同页不一致。
  引文依据：§3.2 `the scaling law curves in (Fig. 7) show that these improvements collectively deliver an approximately 2.5× gain in overall scaling efficiency over Kimi K2`；§1 `These architectural advances, combined with refined data and training recipes, yield an approximately 2.5× improvement in overall scaling efficiency over Kimi K2.`
  修复要求：删去「基础设施」，改为「由架构、数据与训练配方支撑约 2.5× scaling 效率」，与同页其余四处及 overview 保持一致。
  修复：｜复验：

- [轻微·来源] `<head>` 的 description：`基于 arXiv:2607.24653v2 技术报告、官方 config.json 与源码三层核对`。页面内所有来源条目（来源与范围说明的 C1–C14、F1–F8、N1–N5）只引用论文 §/Table/Fig 与 config.json 键值，全文没有任何源码文件路径或行号，所声明的第三层「源码核对」在页面上没有对应证据；本页机制论述全部落在论文与 config 两层。
  引文依据：来源与范围说明 C1–C14（§、Table 1/2、config.json 键）、F1–F8（§与公式号）、N1–N5（Table/Fig），无源码引用。
  修复要求：将 description 改为「技术报告与官方 config.json 两层核对」，或补上一处可定位的源码引用（仓库路径 + 行号）以支撑「三层」。
  修复：｜复验：

## 已核对且无问题的项（择要）

- 架构数字：Table 1 全部 20 行与 v2 Table 1 逐字一致（61→93 ↑52%、1.04T→2.78T ↑167%、32.6B→104.2B ↑220%、7,168=、Latent MoE 3584(0.5×)、2,048→3,072 ↑50%、384→896 ↑133%、8→16 ↑100%、共享专家 1→2、头数 64→96 ↑50%、词表 160K、上下文 128K→1M 8×、69 KDA + 24 MLA、ViT 401M/27 层）；「变化」列与原文 $\Delta$ 列的差异已在表注中说明（注意力机制/激活函数/注意力层组成/Latent MoE 维度四行原文 $\Delta$ 标「–」），核对属实。
- config.json（huggingface.co/moonshotai/Kimi-K3）逐键核对：num_hidden_layers=93、num_experts=896、num_experts_per_token=16、num_shared_experts=2、attn_res_block_size=12、latent_moe_use_norm=true、activation_situ_beta=4.0、activation_situ_linear_beta=25.0、mla_use_nope=true、max_position_embeddings=1048576、first_k_dense_replace=1、hidden_size=7168、moe_intermediate_size=3072、routed_expert_hidden_size=3584、num_attention_heads=96、vt_num_hidden_layers=27、quantization_config.ignore 含 lm_head/vision_tower/mm_projector 且含 self_attn/shared_experts——页面声明全部属实。full_attn_layers 24 项（4,8,…,92,93）与 kda_layers 69 项与页面折叠块列表逐项一致。
- 可复算算式：2.78/1.04≈2.67、104.2/32.6≈3.20、896/16=56、384/8=48、896/384=2.33、16/8=2、93/12=7.75、7×12+9=93、23×4+1=93、61×1.52≈93、42.0−35.0=7、$\beta_1\beta_2=4\times25=100$，均自洽。
- 公式与结论：$\text{softcap}(x,\beta)=\beta\tanh(x/\beta)$ 与 Eq.12 一致；$|f|\le\beta_1\beta_2=100$ 与 Eq.19 一致；$O(L^2d)\to O(Nd)$、$q=mk/n$ 与 §2.2/§2.3.3 一致；LaTeX 全部由 KaTeX 渲染，summary 无 $…$，validate.py 通过（无 Unicode 数学字符、无字符框线图、无 $…$ alt、无占位符）。
- 直接引语逐条回源一致：`Maintaining balanced loads becomes more challenging as LatentMoE increases the routed expert pool to 896 per layer.`、`contrastive pre-training is unnecessary as an initialization for multimodal language models at scale`、`research-level reasoning remains a key direction for improvement`、`All maxed out on thinking effort: max or xhigh`、`the final layer always performs global attention`、`giving a partial final block and 9 total blocks when counting the embedding layer`、`Concentrating the costly long-sequence computation within a small fraction of the overall training budget…`。
- 性能表 9.1/9.2/9.3 全部单元格与 v2 Table 2 一致（含 93.5/23.4/43.5-56.0/91.1/94.3-97.8 等）；第三方段（AA Intelligence Index v4.1 = 57.1 #4/580、Vals Index 74.7% #2/39、WebDev Arena 1,678 Elo #1/99 首个开源登顶、Agent Arena 9.1 #4/37）与 §6.3/Table 5 一致；成本段（$2.03/任务、38% 成本）与 §6.4 一致；「世界首个开放 3T 级模型」（§8）核对属实。
- 基础设施数字：133 ms checkpoint、6.5× 内存超分、51,219,741 沙箱 / 1,505,678 镜像、S×K token、E/R redundant expert 上界、512-token prefix cache 粒度、WarpDecode token-centric、co-located「几百 GPU」，均与 §5 一致。
- 结构：核心问题与 10 个章节的「本章问题」均有解答折叠块且与正文结论一致；所引 15 个 wiki 前置概念页（kda/mla/nope/block-attnres/stable-latent-moe/situ-glu/quantile-balancing/moonvit-v2/per-head-muon/mopd/mxfp4-qat/eagle-speculative/flash-kda/moonep/gpu-execution-model）全部存在；overview.html 与 index.html 互链；无元话语/会话指代/调试叙事；「本文」仅出现在来源与范围说明的范围声明中（非以「本页」为主语的自我指代）。
- head 的 dojo:type=`paper` 属 ALLOWED_TYPES 词表内取值（AGENTS.md 路由「精读论文」→ guides/paper.md），validate.py 通过，不记为问题。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（3 条轻微问题按修复要求就地修正即可，不改变范围与大纲）