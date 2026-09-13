<!-- review-meta
round: 7
page: wiki/kimi-k3/index.html
reviewed_content_sha256: 73bbd8a5b47ef87e
-->
# Kimi K3 审查记录（第 7 轮）

- 页面版本：ed8347ba91c3921ec986a45c17b520c0f499ead0（index.html 工作树哈希）
- 论文版本：arXiv:2607.24653v2（2026-08-07）；另核官方 config.json（huggingface.co/moonshotai/Kimi-K3/raw/main/config.json）
- 审查时间：2026-09-13 21:45
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：封面元信息 → 核心问题（5 题含解答折叠块）→ 1. 三维度信息流——K3 架构总览 → 2. 序列维度——KDA + 混合注意力 → 3. 深度维度——Block AttnRes → 4. 宽度维度——Stable LatentMoE → 5. 原生视觉——MoonViT-V2 → 6. 训练——数据、scaling law、Muon、长上下文扩展 → 7. 后训练——SFT→RL→MOPD 三阶段、QAT、EAGLE → 8. 基础设施——3T 训练 + 1M RL + 推理 → 9. 性能与评价 → 10. 独立评价——系统性设计、开源里程碑与边界 → 来源与范围说明（含全部折叠块、表注与图注）。overview.html 已通读。

## 核对方法与结果总述

- 论文原文：`https://arxiv.org/html/2607.24653v2` 全文抓取（1.52 MB HTML → 文本）后逐段定位；config.json 完整下载后逐字段比对。
- 范围论断：Table 1 的每一行（层数 61→93、总参 1.04T→2.78T、激活 32.6B→104.2B、隐藏 7,168、Latent MoE 3,584 (0.5×)、每专家 MoE 中间维 2,048→3,072、路由专家 384→896、激活专家 8→16、共享专家 1→2、注意力头 64→96、词表 160K、训练上下文 128K→1M、注意力 MLA→Hybrid KDA–MLA、激活 SwiGLU→SiTU-GLU、注意力层组成 61 MLA→69 KDA+24 MLA、MTP 1/1、ViT 401M/27 层）与原文 Table 1 逐格一致。
- config.json 逐字段核对：`num_hidden_layers=93`、`num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`、`hidden_size=7168`、`routed_expert_hidden_size=3584`、`moe_intermediate_size=3072`、`attn_res_block_size=12`、`first_k_dense_replace=1`、`mla_use_nope=true`、`latent_moe_use_norm=true`、`activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`、`max_position_embeddings=1048576`、`vt_num_hidden_layers=27` 全部一致。页内 `full_attn_layers`（24 个）与 `kda_layers`（69 个）两条列表经脚本比对与 config 完全相同（24+69=93）。
- 公式：F1（KDA 递归 Eq.1）、F2（lower-bounded decay Eq.5）、F3（Block AttnRes Eq.8–10）、F4（Eq.11）、F5（SiTU-GLU Eq.12，softcap(x,β)=β·tanh(x/β)、|f|≤β1β2=100）、F6（QB Eq.14，q=mk/n）、F7（Eq.15 stop-gradient+clip）、F8（Eq.16 LK loss=−log Σ min(p,q)）与原文对应式一致；页内只给出文字转述与符号，公式实体在子页面，页内无算式可复算错误。
- 性能数字：Table 2（DeepSWE/ProgramBench/Terminal-Bench 2.1/FrontierSWE/SWE-Marathon/GPQA Diamond/CritPt/HLE-Full/OmniDocBench/Math-Vision/BrowseComp/AutomationBench/GDPval-AA v2/JobBench/Harvey Lab-AA）与 Table 5（AA 57.1 #4/580、Vals 74.7% #2/39、WebDev Arena 1,678 #1/99、Agent Arena 9.1 #4/37）逐一与原文核对，无一处数字不符；§6.4 成本（$2.03/任务、38% 成本、落后 4.0 分）与原文一致。
- 引用编号：C1–C13、F1–F8、N1–N5 的定义与正文使用位置逐一核对，除下列 C13 一处外均对齐。
- 概念链接：页面引用的 15 个 `../../wiki/*/index.html`（kda、mla、block-attnres、stable-latent-moe、situ-glu、quantile-balancing、moonvit-v2、per-head-muon、nope、mopd、mxfp4-qat、eagle-speculative、flash-kda、moonep、gpu-execution-model）全部真实存在，无「（待生成）」占位。
- 功能：`.dojo/scripts/validate.py wiki/kimi-k3/index.html` 返回 `validation ok`；页面无 `<img>`（不存在 alt 内含 `$...$` 的问题）；无等宽字符框线图；overview.html 与 index.html 互链正常。
- 代码：本页无「可运行代码」段，第 4 项改为静态审查（页内无代码块）。

## 问题

- [重要·技术] §9 性能与评价（三张 benchmark 表 + 其后的「补充：benchmark 条件与 harness 差异」折叠块）：三张表的列头只写模型名（`Kimi K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2`），全页未标注各模型的推理努力级别。原文 Table 2 列头为 `Kimi K3 (max) | Claude Fable 5 (max, w/ fallback) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max)`，表注另写明 "Unless otherwise noted, Kimi K3 results are obtained with reasoning effort set to max and temperature equal to 1.0"。GPT-5.5 是 xhigh 而非 max，属影响横向比较的实验条件；页面已专门讨论 harness/fallback/cyberguard 等条件，唯独漏掉努力级别，读者无法判断 K3 与他模型是否同条件。｜引文依据：原文 Table 2 表头 "Benchmark | Kimi K3 (max) | Claude Fable 5 (max, w/ fallback) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max)"，表注 "Kimi K3 results are obtained with reasoning effort set to max and temperature equal to 1.0"。｜修复要求：在 §9.1 表列头或（更集中地）在「补充：benchmark 条件与 harness 差异」折叠块内补一句：各模型推理努力级别为 K3/Fable 5/GPT-5.6 Sol/Opus 4.8/GLM-5.2 = max、GPT-5.5 = xhigh，K3 温度 1.0。｜修复：｜复验：
- [轻微·技术] 第 5 节「原生视觉」第 1 段：句末 `采用 RMSNorm 并移除所有 bias<sup>[C13]</sup>` 的编号与「来源与范围说明」中 C13 的定义错位——C13 定义为「MoonViT-V2 **从零训练**：§2.4, Fig.6」，而此处句子讲的是架构参数（27 层/0.4B/RMSNorm/去 bias）。论断本身有来源（§2.4 Architecture 小节），只是所挂编号的语义不对应；同一编号在 §10.2「视觉'从零训练匹配 SigLIP'的结论依赖特定规模」处用法正确。｜引文依据：原文 §2.4 Architecture "MoonViT-V2 is a 27-layer vision transformer with roughly 0.4B parameters that adopts RMSNorm and removes all bias terms from its linear and attention projections"；页面来源表 "C13（MoonViT-V2 从零训练）：§2.4, Fig.6"。｜修复要求：把第 5 节该句的 `[C13]` 改为指向 §2.4 架构描述的编号（可新增 C14 或改用不带编号的 `（§2.4）`），使编号与定义语义一致。｜修复：｜复验：
- [轻微·可读性] 第 1 节与第 1 节末段：存在以页面自身结构为主语的元话语——「把三个维度画在一起，K3 的信息流可以这样理解：」（流程图前引导句）与「一个具体视角可以贯穿后续每个维度章节：想象一个 token……」（第 193 行）。此类表述描述的是「本页接下来怎么写」，而非内容本身。｜引文依据：不适用｜修复要求：改为直接陈述内容，例如把「一个具体视角可以贯穿后续每个维度章节」删去、句子直接从「一个 token 从 embedding 出发……」开始；「可以这样理解」改为直接给出图注要说明的对应关系。｜修复：｜复验：

## 已核对但未构成问题的项（供复验参考）

- 「2.5×」的口径：页面「OOD 验证 loss 上达到 K2 相同水平所需 FLOPs 减半多」与原文 §3.2 "Evaluated on held-out OOD validation data, the scaling law curves in (Fig. 7) show that these improvements collectively deliver an approximately 2.5× gain" 相符；collectively 归因、未分解单因素的说法与原文一致。
- §9.4「AA …#4/580（第三如果 GPT-5.6 Sol 努力变体算一个）」不是页面臆测：原文 §6.3 即为 "ranking fourth of 580 models — third if GPT-5.6 Sol effort variants are counted as a single entry"。
- 「首个登顶 WebDev Arena 的开源模型」「首个开放权重的 3T 级模型」分别对应原文 §6.3 "the first open model to top this leaderboard" 与 §8 "the world's first open 3T-class model"。
- 8 block/12 层/末尾 9 层部分 block/含 embedding 共 9 个、N≈8、O(Ld)→O(Nd)、block 内 partial sum 从第 2 层起计入候选（Eq.10 分段）均与原文 §2.2 一致。
- 四阶段上下文课程 8K→64K（预训练）/256K→1M（cooldown）、NoPE 免 RoPE rescaling/YaRN、长上下文清洗（exact/fuzzy 去重、视频帧 perceptual hashing、质量过滤、结构验证）与原文 §3.4 一致。
- 基础设施数字 133 ms checkpoint / 49 ms resume / 6.5× 内存超分 / 51,219,741 沙箱、1,505,678 镜像 / E/R 冗余专家上界（Theorem 1，§E）/ 512-token prefix 边界（Fig.12）/ co-located「几百 GPU」均与原文 §5 一致。
- 表格「变化」列自行归类并明确声明「不等同原文 Table 1 的 Δ 列」，未冒充原文列。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（补 §9 推理努力级别条件；订正 C13 编号位置；清理两处元话语），修完即可发布
