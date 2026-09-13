<!-- review-meta
round: 5
page: wiki/kimi-k3/index.html
reviewed_content_sha256: 4b1e6c90f6bfa6eb
-->
# Kimi K3 审查记录（第 5 轮）

- 页面版本：index.html 工作树哈希 14367fcdc090e22c5010b06f049e5e904aab8a5b
- 论文版本：arXiv:2607.24653v2《Kimi K3: Open Frontier Intelligence》（2026-08-07）；另核对官方 config.json（huggingface.co/moonshotai/Kimi-K3）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次；本轮仅读当前页面、overview.html、论文原文与官方 config.json）
- 已完整阅读章节：核心问题；1. 三维度信息流——K3 架构总览；2. 序列维度——KDA + 混合注意力；3. 深度维度——Block AttnRes；4. 宽度维度——Stable LatentMoE；5. 原生视觉——MoonViT-V2；6. 训练——数据、scaling law、Muon、长上下文扩展；7. 后训练——SFT→RL→MOPD 三阶段、QAT、EAGLE；8. 基础设施——3T 训练 + 1M RL + 推理；9. 性能与评价；10. 独立评价——系统性设计、开源里程碑与边界；来源与范围说明

## 问题

- [重要·技术] §3 深度维度（第 275 行、第 283 行两处）：页面称"论文 §2.2 引 [60] 称…"与"论文 §2.2 引 [60] 给出 $N \approx 8$…"，但论文该句的引文编号是 [58]，非 [60]。｜引文依据：论文 §2.2 原文"Empirically, N ≈ 8 recovers most of the benefit across model scales [58]; for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer."；参考文献表 [58]="Kimi Team. Attention Residuals. Preprint. 2026."，而 [60]=Kimi K2.5（§3.1"our data pipelines build on those developed for Kimi K2 [59] and refined in Kimi K2.5 [60]"）。[60] 指向 K2.5，与 AttnRes 的 block-size 结论无关，按编号回源会得到错误来源。｜修复要求：把第 275、283 行两处的 "[60]" 改为 "[58]"，与原文引文编号一致。｜修复：｜复验：
- [轻微·表述] §9 性能与评价 开头（第 512 行）：正文用独立导语句"以下是关键 benchmark 的分域表现。"作过渡，属元话语（与规范 §2.14 所列"下面来看…"同类），未承载结论。｜引文依据：不适用｜修复要求：删除该独立导语句，或与其前的定位句合并（如"……优于 Claude Opus 4.8、GPT-5.5 和开源 GLM-5.2（§6.1.4，Table 2）。分域表现如下。"），使该段只陈述内容、不描述页面结构。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 1
- 处置：修复（[60]→[58] 属来源论断（引文编号）错误，须修正后再复验；轻微表述项可一并处理）

（核对说明：本轮将论文 PDF 全文落盘后逐条回源。已核到原文的关键项包括：Table 1 全部行与"变化"归类；§2.1"Each block contains 3 KDA layers followed by 1 Gated MLA layer…An additional Gated MLA layer is placed at the end of the backbone, ensuring that the final layer always performs global attention"；§2.2 Eq.10 的 `V=[b0…b_{n−1}]`（i=1）与 `[b0…b_{n−1}, b^{i−1}_n]`（i≥2）——支持"partial sum 从块内第 2 层起计入候选"；§2.3 激活爆炸/负载失衡两失效模式与 Eq.11/12/14；§C"QB jumps directly to the exact coordinate minimizer of the same dual objective…why QB requires no learning-rate-like hyperparameter and why it equilibrates within a few update steps even for nearly 10³ experts"；§2.4 MoonViT-V2 从零训练与"contrastive pre-training is unnecessary…"；§2.5 Per-Head Muon；§3.1 四文本域；§3.2/§3.3/§3.4（cosine vs WSD、1% linear warmup、weight decay 0.1、四阶段 8K→64K→256K→1M、NoPE 直接外推）；§4.1 SFT/RL/MOPD（3 域×3 努力=9 专家、−1 奖励、Eq.15）、§4.1.4（MXFP4/MXFP8 QAT、EAGLE-3 复用 MTP 层、1/4/final AttnRes block、7 步、Eq.16）；§5.1 FlashKDA/KCP、§5.2 MoonEP（"every rank to receive exactly S×K tokens"、E/R、§E 定理 1）、§5.3（a few hundred GPUs、外部 KV cache 池、133 ms/49 ms、6.5×、51,219,741 sandboxes across 1,505,678 images）、§5.4（512-token 前缀粒度、SP/side stream、WarpDecode）；§6.1.3（"we report the best score across harnesses for all models"、DeepSWE v1.1/67.3、Fable 35% fallback、BrowseComp 300K compaction/90.4%）、Table 2 全部采用行、§6.3 Table 5（57.1 #4/580、74.7% #2/39、1678 #1/99、9.1 #4/37）、§6.4（$2.03、half the cost、4.0 points/38%）、§8"the world's first open 3T-class model"；config.json 的 num_hidden_layers=93、num_experts=896、num_experts_per_token=16、num_shared_experts=2、hidden_size=7168、moe_intermediate_size=3072、mla_use_nope=true、attn_res_block_size=12、first_k_dense_replace=1、max_position_embeddings=1048576、full_attn_layers、kda_layers、vt_num_hidden_layers=27、latent_moe_use_norm=true、activation_situ_beta=4.0、activation_situ_linear_beta=25.0 均与页面一致。算术项复核：2.78/1.04≈2.67×、104.2/32.6≈3.20×、896/16=56、384/8=48、896/384≈2.33×、42.0−35.0=+7、7×12+9=93、β1β2=4×25=100 全部成立。链接、dojo 元信息、问题块与折叠块、数学字符（`×`/`→` 为 validate.py 明示允许的普通排版字符）均通过校验；`python3 .dojo/scripts/validate.py wiki/kimi-k3/index.html` 返回 validation ok。除上述两条外，未发现回源不符、算式与结论不符、无来源支持的关键论断或指向仓库不存在路径的引用。）