<!-- review-meta
round: 12
page: wiki/kimi-k3/index.html
reviewed_content_sha256: c3f91bed835f1a1c
-->
# Kimi K3 审查记录（第 12 轮）

- 页面版本：c784c96c5234164f602ce0c06916d980f0a4871e（git hash-object wiki/kimi-k3/index.html）
- 论文版本：arXiv:2607.24653v2（2026-08-07，Kimi K3: Open Frontier Intelligence）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序轮次审查）
- 已完整阅读章节：核心问题 / 1 三维度信息流 / 2 序列维度 / 3 深度维度 / 4 宽度维度 / 5 原生视觉 / 6 训练 / 7 后训练 / 8 基础设施 / 9 性能与评价 / 10 独立评价 / 来源与范围说明（含全部折叠块、表注与图注）
- 已核对：`validate.py wiki/kimi-k3` → validation ok

## 已核对内容（引文依据）

外部来源：arXiv 摘要页与 HTML 正文（arxiv.org/html/2607.24653v2、ar5iv）、HuggingFace config.json 与模型卡、WebSearch 复核第三方数字。

**Table 1（原文 §3.2，已确认表号与所在小节）**——逐行与原文一致：
- 层数 61→93 ↑52%；总参数 1.04T→2.78T ↑167%；激活 32.6B→104.2B ↑220%；隐藏 7168=7168；Latent MoE 维度 –→3584(0.5×)；每专家 2048→3072 ↑50%；路由专家 384→896 ↑133%；激活专家 8→16 ↑100%；共享专家 1→2 ↑100%；注意力头 64→96 ↑50%；Dense 1=1；词表 160K=160K；上下文 128K→1M（8×）；MLA→Hybrid KDA–MLA；SwiGLU→SiTU-GLU；61 MLA→69 KDA+24 MLA；MTP 1=1；ViT –→401M；ViT 层 –→27。
- 表注称原文 $\Delta$ 列对「注意力机制、激活函数、注意力层组成、Latent MoE 维度」四行标「–」——原文确为这四行标「–」。
- 复算：2.78/1.04≈2.67×、104.2/32.6≈3.20×、3072/2048=1.5、896/384≈2.33、384/8=48、896/16=56，均与页面一致。

**config.json 逐键核对**：num_hidden_layers=93、num_experts=896、num_experts_per_token=16、num_shared_experts=2、hidden_size=7168、moe_intermediate_size=3072、routed_expert_hidden_size=3584、max_position_embeddings=1048576、mla_use_nope=true、attn_res_block_size=12、latent_moe_use_norm=true、activation_situ_beta=4.0、activation_situ_linear_beta=25.0、first_k_dense_replace=1、num_attention_heads=96、vt_num_hidden_layers=27。full_attn_layers=24（4,8,…,92 与 93）、kda_layers=69（1-3,5-7,…,89-91），与页面两处层表完全一致；quantization_config.ignore 含 self_attn/shared_experts/mlp/lm_head/vision_tower/mm_projector。

**逐条引文核对（片段）**：
- §2.2：「Empirically, N≈8 recovers most of the benefit across model scales [60]」——页面引 [60] 与文字逐字一致；「giving a partial final block and 9 total blocks」（含 embedding 层）一致。
- §2.4：「contrastive pre-training is unnecessary as an initialization for multimodal language models at scale」逐字一致；「a 27-layer vision transformer with roughly 0.4B parameters that adopts RMSNorm」「removes all bias terms」；Fig.6 图注「Vision-tower gradient norms in our pre-training ablations」与 SigLIP MoonViT-3D「persistently higher gradient norms with frequent spikes」一致。
- §2.1：「the final layer always performs global attention」；KDA「lower-bounded decay」「full-rank gate」；MLA 全层 NoPE。
- §2.3：sparsity 56；softcap(x,β)=β·tanh(x/β)，β₁=4/β₂=25，|f|≤β₁β₂=100；QB 目标负载 q=mk/n；「nearly four consecutive matrix multiplications」。
- §3.1：文本四域「Web Text, Code, Mathematics, and Knowledge」；视觉语料「captions, interleaved image–text documents, OCR, perception, video, and visual coding data」；rephrasing「style and perspective-diverse prompting」。
- §3.2：cosine decay 与 WSD 各自独立搜索、「cosine decay consistently achieves a lower final loss」；Fig.7 图注「Fitted scaling-law curves for Kimi K2 and Kimi K3」，约 40% compute 即 2.5×；OOD（held-out）验证 loss 归因 collectively（KDA/AttnRes/Stable LatentMoE + 精炼数据与训练配方）。
- §3.3：「cosine learning rate schedule with a 1% linear warmup」「Weight decay is set to 0.1」。
- §3.4：8K→64K（预训练）、256K→1M（cooldown）；清洗含 exact/fuzzy 去重、perceptual hashing over frames、质量过滤、结构验证；「permuting and concatenating multimodal documents」。
- §4.1.2：「per-problem budget control」；「override the task reward with −1」；partial rollout、resumable microVM sandboxes。
- §4.1.3：Eq.15 per-token OPD reward，含 sg(·) stop-gradient 与 Rmax clip。
- §4.1.4：MXFP4 权重 / MXFP8 激活，仅量化 MoE 专家；高精度项为 attention projections、latent MoE projections、shared experts、MoE routers；EAGLE draft「unrolled for seven steps」、融合「1st, 4th, and final AttnRes blocks」、LK loss。
- §2.5：按头维度切分动量矩阵、逐头块 Newton–Schulz 正交化（替代全矩阵正交化）。
- §5.3.1：co-located RL「within a few hundred GPUs」、partial rollouts、写回「external KV cache pool」in CPU DRAM。
- 模型卡/第三方复核：GPQA 93.5、CritPt 23.4、HLE-Full 43.5/56.0、OmniDocBench 91.1、Math-Vision 94.3/97.8；Table 2 竞赛模型全部数值（DeepSWE/FrontierSWE/Terminal-Bench/ProgramBench/SWE-Marathon/BrowseComp/AutomationBench/GDPval-AA v2/JobBench 等）逐格一致；§9.4 AA Intelligence Index v4.1=57.1 #4/580、Vals AI 74.7% #2/39、WebDev Arena 1678 Elo #1/99（首个开源登顶）、Agent Arena 9.1 #4/37；§6.4 BrowseComp 91.2% @ $2.03/任务（Sol 90.4%）、Kimi Code Bench 2.0 落后 Fable 5 4.0 分/38% 成本——均与来源一致。
- 链接与结构：全部 `../../wiki/*/index.html`（kda/mla/nope/block-attnres/stable-latent-moe/situ-glu/quantile-balancing/moonvit-v2/per-head-muon/mopd/mxfp4-qat/eagle-speculative/flash-kda/moonep/gpu-execution-model）真实存在；overview.html 与 index.html 互链；无「（待生成）」占位；`alt` 仅 lightbox 为空串，无 `$...$`；页面级「核心问题」与各章「本章问题」均有解答折叠块。表述层面：全文（含折叠块、图注）未见「本页/我们/你/下面来看/需要注意的是」等元话语与调试叙事，`×`/`↑` 仅作倍率与升降记号使用。

## 问题

- [重要·技术] 1. 三维度信息流总览末段（index.html 第 193 行）：「一个 token…在宽度维度被 896 个稀疏专家之一变换」——把每 token 路由描述为「896 个专家之一」，与同页 §4（第 333 行「router 选出 16 个路由专家；16 个专家…变换」）、Table 1（第 154 行「每 token 激活专家 … 16」）及原文（Table 1「Experts Active per Token 16」、§2.3「activates 16 of 896 routed experts per token」）矛盾。｜引文依据：原文 Table 1「Experts Active per Token｜8｜16」；§2.3「routes tokens to 16 of 896 routed experts per token」。｜修复要求：改为「被 896 个稀疏专家中选出的 16 个加权变换」或等价表述，使宽度维度描述与 top-16 机制一致，不再出现「之一」。｜修复：｜复验：
- [轻微·技术] 「来源与范围说明·简化条件」与 §1 表注（第 170 行）：页表标注为「（原文 Table 1）」，但省略了原文 Table 1 的「Patch Size of ViT（14）」与「#Attention Heads of ViT（12）」两行，且未像 §9 表（第 529/546/563 行「原文 Table 2 节选」）那样标注为节选。｜引文依据：原文 Table 1 含 ViT Patch Size=14、ViT attention heads=12 两行。｜修复要求：表注补「节选」字样，或在简化条件中列出省略的 ViT 行。｜修复：｜复验：
- [轻微·技术] 4. 宽度维度·稳定化 3（第 329 行）：将固定步长 sign 更新归因为「DeepSeek-V3 的固定步长 sign 更新」，但论文 §2.3.3 仅以引文编号 [27] 引用该更新、全节未点名任何模型或作者。｜引文依据：§2.3.3「the fixed-step rule b_j^(t+1)=b_j^(t)+γ sign(ℓ̄−ℓ_j^(t)) [27]」，且「Section 2.3.3 names no model or author; references appear only as bracket numbers」。｜说明：该归因本身与公开文献一致（[27] 即引入 auxiliary-loss-free 负载均衡的工作），属外部补充而非错误；建议在来源说明中标注 [27] 对应文献，避免把外部归因呈现为论文原文。｜修复要求：在来源说明补 [27] 的对应文献名，或改为「固定步长 sign 更新（[27]）」。｜修复：｜复验：
- [轻微·表述] 10.1 优点（第 620 行）：「这反映了对『训练效率≠部署效率』的清醒认识」——「清醒认识」是对作者动机的拟人化临场评价；该段虽属已声明的解读者判断，仍宜改为对可观察设计的陈述（QAT 贯穿 RL、Eagle draft 复用 MTP 层）。｜引文依据：不适用。｜修复要求：删去「清醒认识」一类对作者主观状态的评价，仅陈述其设计做法。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复。页面事实、数字、公式、引文与 config.json 三层核对全部通过，除上列 1 项宽度维度路由表述与原文/同页矛盾外未见其他错误；该重要项关闭后即可发布。

统计：阻断 0 / 重要 1 / 轻微 3