<!-- review-meta
round: 6
page: wiki/kimi-k3/index.html
reviewed_content_sha256: c1701fb9989085d8
-->
# Kimi K3 审查记录（第 6 轮）

- 页面版本：index.html a2502f2cf1b67caedcfc58f47caad03919411c47；overview.html 438aa351273dd4fcc77fe0a53b396ea0b48cb0ab
- 论文版本：arXiv:2607.24653v2（v1 2026-07-27，v2 2026-08-07）；官方 config.json（huggingface.co/moonshotai/Kimi-K3/raw/main/config.json）
- 审查时间：2026-09-13 21:14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题 · 1 三维度信息流 · 2 序列维度 · 3 深度维度 · 4 宽度维度 · 5 原生视觉 · 6 训练 · 7 后训练 · 8 基础设施 · 9 性能与评价 · 10 独立评价 · 来源与范围说明（含全部折叠块、表格与图注）

## 核对说明（回源结果）

本轮把论文全文（arxiv.org/html/2607.24653v2）、官方 config.json 与页面逐条对照，主要数值全部命中：

- Table 1 全 20 行与原文一致（61/93、1.04T/2.78T、32.6B/104.2B、7,168、3,584、2,048/3,072、384/896、8/16、1/2、64/96、160K、128K/1M、61 MLA/69 KDA+24 MLA、401M、27）；百分比可复算（167%、220%、52%、133%、100%、50%、8× 均正确）。
- config.json 逐字段核对：num_hidden_layers=93、num_experts=896、num_experts_per_token=16、num_shared_experts=2、mla_use_nope=true、max_position_embeddings=1048576、attn_res_block_size=12、latent_moe_use_norm=true、activation_situ_beta=4.0、activation_situ_linear_beta=25.0、first_k_dense_replace=1、vt_num_hidden_layers=27；full_attn_layers（24 个，含 93）与 kda_layers（69 个）与页面逐项相同；quantization_config.ignore 含 self_attn / shared_experts / mlp.(gate|up|gate_up|down)_proj / lm_head / vision_tower / mm_projector。
- 公式回源：Eq.1（KDA 状态更新）、Eq.5（lower-bounded decay，g_min=-5）、Eq.8-10（AttnRes，块内第 i 层 V=[b_0..b_{n-1}, b_n^{i-1}]）、Eq.11（y=ΣE_shared+W↑RMSNorm(u)）、Eq.12（SiTU-GLU，|f|≤β1β2=100）、Eq.14（QB 更新）、Eq.15（OPD reward，sg+clip）、Eq.16（LK loss）全部对应。
- 数字回源：51,219,741 沙箱 / 1,505,678 镜像、checkpoint 133ms、6.5× 内存超分、E/R redundant expert 上界（附录 E 定理 1 与紧性定理 2）、S×K token/rank、512-token prefix-hash 粒度、300K 触发 compaction 与 1M 无压缩 90.4%、$2.03/任务与 38% 成本、Table 2 与 Table 5 全部单元格，均与原文一致。
- 内部算术核对无误：2.67×/3.20×、896/16=56、384/8=48、896/384=2.33、93/12=7.75→7 个 12 层 + 1 个 9 层、7×12+9=93、1M/128K=8。
- validate.py 通过；页面无「（待生成）」占位；全部 15 个概念页链接均真实存在；meta 中 GitHub 链接 MoonshotAI/Kimi-K3 实测存在（200）。
- 表述维度：全文（含折叠块与图注）无「本页将…」「下面来看…」「需要注意的是」，无「我/我们/你」，无自我指代、调试叙事、临场评价；未检出「场景」当术语或公文连接词堆叠。

## 问题

- [重要·技术] index.html §3「深度维度」两处：把 AttnRes 的 N≈8 经验结论标为引 [58]。论文 §2.2 该句引用的是 [60]，[60] = Kimi Team (2026) Attention Residuals，[58] = Kimi Team (2025) Kimi k2: open agentic intelligence（另一篇论文，不含该结论）。｜引文依据：论文 §2.2 "Empirically, N≈8 recovers most of the benefit across model scales [60]"；参考文献 [58] "Kimi Team (2025) Kimi k2: open agentic intelligence"、[60] "Kimi Team (2026) Attention residuals"。｜修复要求：§3「开销从 $O(Ld)$ 降到 $O(Nd)$（N=8），论文 §2.2 引 [58] 称…」与「论文 §2.2 引 [58] 给出 $N \approx 8$…」两处 [58] 均改为 [60]。｜修复：｜复验：

- [重要·技术] overview.html §3「大致怎么做」深度维度一行：与 index.html §3 正文及论文 §2.2 Eq.10 冲突。overview 写「跨 block 用注意力选择性检索前层表示，块内走标准残差」，而 index.html 正文写「块内 12 层中每一层的输入都是对「embedding ＋此前各 block 汇总表示 ＋当前 block partial sum」的 softmax 加权检索」；论文 Eq.10 规定块内第 i 层（i≥2）的 V 含 [b_0,…,b_{n-1}, b_n^{i-1}]，即块内每层都对前序 block 表征做检索，论文 §2 引言亦称 AttnRes "enable each module to selectively retrieve representations from the embedding, the current block, and preceding blocks"。overview 的表述会让读者以为块内是普通残差链、不做跨块检索。｜引文依据：论文 §2.2 Eq.10 "V=[b_0,b_1,…,b_{n-1}]^T if i=1 (first layer of block n); [b_0,…,b_{n-1},b_n^{i-1}]^T if i≥2"；论文 §2 引言 "enable each module to selectively retrieve representations from the embedding, the current block, and preceding blocks"。｜修复要求：把 overview 该行的「块内走标准残差」改为与正文一致（块内每层仍对前序 block 汇总与当前 partial sum 做选择性检索；块内各层输出按求和汇总为块级表征）。｜修复：｜复验：

- [轻微·技术] index.html §8.3：沙箱数量的适用范围被缩小——页面写「训练期间共创建 51,219,741 个沙箱（跨 1,505,678 个镜像）」，原文范围是训练与评估。数字本身正确。｜引文依据：论文 §5.3.2 "Throughout Kimi K3's training and evaluation, a total of 51,219,741 sandboxes across 1,505,678 images were created."｜修复要求：把「训练期间」改为「训练与评估期间」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 1
- 处置：修复