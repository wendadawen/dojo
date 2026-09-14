<!-- review-meta
round: 9
page: wiki/kimi-k3/index.html
reviewed_content_sha256: 79da09bbc668929e
-->
# Kimi K3 审查记录（第 9 轮）

- 页面版本：5791f4bc5bd76bc04833213fd884047178771e6d
- 论文版本：arXiv:2607.24653v2（v2 修订版 2026-08-07，含官方 config.json 与 huggingface/github 页面）
- 审查时间：2026-09-14 14:36
- 审查者：独立子代理（第 9 轮独立审查，未参与写作与前序审查）
- 已完整阅读章节：1. 三维度信息流——K3 架构总览；2. 序列维度——KDA + 混合注意力；3. 深度维度——Block AttnRes；4. 宽度维度——Stable LatentMoE；5. 原生视觉——MoonViT-V2；6. 训练——数据、scaling law、Muon、长上下文扩展；7. 后训练——SFT→RL→MOPD 三阶段、QAT、EAGLE；8. 基础设施——3T 训练 + 1M RL + 推理；9. 性能与评价；10. 独立评价——系统性设计、开源里程碑与边界；来源与范围说明（含全部折叠块与图注）

## 问题

- [阻断·技术] §3 深度维度正文两处（`wiki/kimi-k3/index.html` 第 275 行「…论文 §2.2 引 [60] 称"$N \approx 8$ recovers most of the benefit across model scales"」与第 283 行「论文 §2.2 引 [60] 给出 $N \approx 8$ 在多模型规模下够用的经验结论」）：把原文该句的引文编号写成 [60]，与所引版本 arXiv:2607.24653v2 的文献表不符。原文此句标注的是 [58]（Attention Residuals），[60] 是另一篇工作（Kimi K2.5），读者按页面标注去查会落到错误的文献上。｜引文依据：原文 §2.2「Empirically, N ≈ 8 recovers most of the benefit across model scales [58]; for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer.」；同版文献表「[58] Kimi Team. Attention Residuals. Preprint. 2026.」「[60] Kimi Team. "Kimi K2.5: Visual Agentic Intelligence". In: arXiv preprint arXiv:2602.02276 (2026).」｜修复要求：将两处「引 [60]」改为「引 [58]」；改后该句的数值与表述（N≈8、8 个 block、partial final block、9 total blocks）已核对与原文一致，无需改动。｜修复：｜复验：

- [轻微·技术] §9.4 后「benchmark 条件与 harness 差异」折叠块（`wiki/kimi-k3/index.html` 第 579 行）：把引文 "All maxed out on thinking effort: max or xhigh" 标注为「论文 §6.1.4 表注原文」。该句在原文中出现在 Figure 1 的图注（主结果图，位于 §1 区域），§6.1.4 的 Table 2 表注写的是 "Unless otherwise noted, Kimi K3 results are obtained with reasoning effort set to max and temperature equal to 1.0"，并不含该句。引文本身属实、由其得出的结论（各模型各取最高努力级别，max 与 xhigh 为不同厂商命名，见 §6.1.2「All models are evaluated at maximum reasoning effort, except GPT-5.5, which uses the "xhigh" setting」）也成立，仅出处标注有误。｜引文依据：原文 Figure 1 图注「Coding All maxed out on thinking effort: max or xhigh.」「General & Visual Agents All maxed out on thinking effort: max or xhigh.」；Table 2 表注「Unless otherwise noted, Kimi K3 results are obtained with reasoning effort set to max and temperature equal to 1.0. For HLE-Full, MMMU-Pro, CharXiv (RQ), Math-Vision, and ZeroBench, each cell reports the scores without and with tool augmentation…」｜修复要求：把该引文的出处由「论文 §6.1.4 表注原文为」改为「论文 Fig.1 图注原文为」（同一段落末句的「（§6.1.4 表注）」保留，其对应 Table 2 表注无误）。｜修复：｜复验：

## 本轮已核对（无问题项，逐条给出核对依据）

- Table 1（第 143–167 行）20 行数值与「变化」列：与原文 Table 1 逐行一致（61/93、1.04T/2.78T、32.6B/104.2B、7,168、3584 (0.5×)、2,048/3,072、384/896、8/16、1/2、64/96、1/1、160K、128K/1M 8×、401M、27 layers）；图注对 Δ 列的说明（注意力机制、激活函数、注意力层组成、Latent MoE 维度四行原文标「–」）与原文一致。
- 层配置：config.json `num_hidden_layers=93`、`full_attn_layers`（4–92 步长 4 共 23 个 + 93）、`kda_layers`（69 个）、`attn_res_block_size=12`、`num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`、`routed_expert_hidden_size=3584`、`mla_use_nope=true`、`latent_moe_use_norm=true`、`activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`、`max_position_embeddings=1048576`、`vt_num_hidden_layers=27`、`first_k_dense_replace=1` 均与页面标注一致（页面列出的 24 个 MLA 层与 69 个 KDA 层清单正确）。
- 机制与公式出处：§2.1 3:1 混合与末尾「ensuring that the final layer always performs global attention」、§2.1.2 Gated MLA 施加 NoPE、Eq.1–6（KDA）、§2.2 Eq.8–10（AttnRes，含 i≥2 时 $b_n^{i-1}$ 入候选）、§2.3 Eq.11（RMSNorm 置于专家聚合与升投影之间）、Eq.12（SiTU-GLU，Fig.4 图注给出 $|f(x)| \le \beta_1\beta_2 = 100$，$\beta_1=4$、$\beta_2=25$）、Eq.13–14（QB，目标负载 $q := mk/n$，$1-k/n$ 分位）、§4.1.3 Eq.15（MOPD stop-gradient + clip）、§4.1.4 Eq.16（LK loss）、§5.1.1 FlashKDA（CUTLASS-based）、§5.1.2 KCP（cumulative transition + local-from-zero，固定大小 all-gather + prefix scan）、§5.2.1 MoonEP（每 rank 恰好 $S\times K$ 个 token，S 为序列长度、K 为每 token 选专家数；E/R 冗余专家上界，附录 E Theorem 1）逐条定位并比对一致。
- 性能数字：§9.1/9.2/9.3 三张表共 15 个 benchmark 的 6 列数值与原文 Table 2 完全相同（DeepSWE 67.5、ProgramBench 77.8、Terminal-Bench 2.1 88.3、FrontierSWE 81.2、SWE-Marathon 42.0；BrowseComp 91.2、AutomationBench 30.8、GDPval-AA v2 1686、JobBench 54.3、Harvey Lab-AA 94.6；GPQA Diamond 93.5、CritPt 23.4、HLE-Full 43.5/56.0、OmniDocBench 91.1、Math-Vision 94.3/97.8），加粗位与「第一/落后」表述均与原文一致（SWE-Marathon 42.0 比 Fable 5 的 35.0 高 7.0）。
- §9.4 第三方与成本（Table 5、§6.3、§6.4）：Intelligence Index v4.1 = 57.1（#4/580，合并 Sol 努力级别变体则第 3）、Vals Index 74.7%（#2/39）、WebDev Arena 1,678 Elo（#1/99，首个开源模型登顶）、Agent Arena 9.1（#4/37）、BrowseComp 91.2% at $2.03/task（为 GPT-5.6 Sol 90.4% 的一半成本）、Kimi Code Bench 2.0 落后 Fable 5 4.0 分且成本为其 38%，均与原文一致。
- 实验条件：Table 2 列头努力级别（K3/Fable 5/Sol/Opus 4.8/GLM-5.2 为 max，GPT-5.5 为 xhigh）、temperature = 1.0、§6.1.3「we report the best score across harnesses for all models」、DeepSWE v1.1 与 mini-SWE-agent 67.3、SWE-Marathon H20-calibrated 分支（July 9, 2026）与 Fable 5 35% fallback、BrowseComp 300K 触发压缩与 1M 无压缩 90.4%、Fable 5 含 fallback / Sol 含 cyberguard，均与原文一致。
- 训练与后训练：§3.1 四类文本域、§3.2 OOD 验证数据与「cosine decay consistently achieves a lower final loss than WSD」、§3.3 cosine + 1% linear warmup + weight decay 0.1、§3.4 四阶段 8K→64K→256K→1M 与 NoPE 直接外推、§2.5 Per-Head Muon、§4.1 SFT→RL→MOPD（3 域 × {low, high, max} = 9 专家）、per-problem budget control 超预算 −1 奖励、§4.1.4 QAT（MXFP4 权重 + MXFP8 激活，仅量化 MoE 专家权重，非专家项保持高精度）与 EAGLE-3 draft（复用 MTP 层、展开 7 步、融合第 1/4/最终 AttnRes block 特征）逐条一致。
- 基础设施：§5.3.1 co-located RL「within a few hundred GPUs」、外部 KV cache 池（KDA 状态与 MLA KV 一起 offload/prefetch）、§5.3.2 AgentENV（Firecracker microVM，checkpoint 133 ms / resume 49 ms，pause-resume/fork/snapshot，6.5× 内存超分，51,219,741 个沙箱跨 1,505,678 个镜像）、§5.4 双缓存统一分页池、512-token 前缀哈希（Fig.12）、cache-aware affinity scheduling 与 budget-based admission control，均与原文一致。
- 可读性与形式：核心问题 5 问、各章本章问题均配有解答折叠块且答案与正文一致；折叠收起后正文仍可独立成立；自绘结构为 HTML/CSS（无等宽字符框线图），无 `<img>` 与 `$...$` alt；公式均为 KaTeX（`$O(L^2d)$`、`$O(Nd)$`、`$\ell=3584$`、`$\beta \tanh(x/\beta)$` 等），未出现 Unicode 数学字符；全部前置概念链接（kda/mla/nope/block-attnres/stable-latent-moe/situ-glu/quantile-balancing/moonvit-v2/per-head-muon/mopd/mxfp4-qat/eagle-speculative/flash-kda/moonep/gpu-execution-model）对应 `wiki/<name>/index.html` 均真实存在，无「（待生成）」占位；`overview.html` 与 `index.html` 双向互链；`python3 .dojo/scripts/validate.py wiki/kimi-k3/index.html` 返回 `validation ok`；通读全文（含折叠块与图注）未发现元话语、以「本页」为主语的自我指代、会话指代、调试叙事或临场评价（第 10 章的判断性表述均置于「属于解读者判断，而非论文结论」的声明之下）。

## 结论

- 处置：修复（关闭上述 1 条阻断后再行复验）
- 统计：阻断 1 / 重要 0 / 轻微 1