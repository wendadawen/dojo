# Kimi K3 独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照原文）
- 页面版本：ac5b744（index.html 与 overview.html 同一提交，2026-08-09）
- 论文版本：arXiv:2607.24653v1，2026-07-27
- 时间：2026-08-09
- 来源：/tmp/kimi-k3-research/k3-report.txt（全文 3082 行）+ https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json

## 问题

- [重要·技术] index.html 后训练章节（MOPD、EAGLE 两处 `<em>` 内联）+ 基础设施章节（FlashKDA、MoonEP 两处 `<em>` 内联）：页面 4 处标注"子页面 X 未生成，此处简要内联"，但 `wiki/mopd/index.html`（1177 行）、`wiki/eagle-speculative/index.html`（1390 行）、`wiki/flash-kda/index.html`（1259 行）、`wiki/moonep/index.html`（1175 行）均已存在。页面声称未生成与实际不符，且未链接到已存在概念页，读者被告知"未生成"后无法访问已有详细内容。修法：将 4 处"子页面 X 未生成，此处简要内联"替换为指向对应已存在页面的链接（`../../wiki/mopd/index.html`、`../../wiki/eagle-speculative/index.html`、`../../wiki/flash-kda/index.html`、`../../wiki/moonep/index.html`），保留一句话衔接或删除"未生成"声明；修复者须打开目标页确认内容与 K3 语境相关后再链接。｜ 修复：已确认 4 个子页面均存在且标题与 K3 语境相关（MOPD 1177 行、EAGLE-3 1390 行、FlashKDA 1259 行、MoonEP 1175 行）。将 4 处"子页面 X 未生成，此处简要内联"替换为"完整机制见 X 概念页"的链接（保留原一句话衔接），并同步修改来源说明 L1069"子页面未生成"为"简要内联并链接到对应概念页"。 ｜ 复验：

- [重要·技术] index.html 性能章节 `<details>`"benchmark 条件与 harness 差异" + overview.html"关键结论"末条："Terminal-Bench 2.1 K3 报 88.3（best across harnesses），但 Artificial Analysis 报约 85%"。来源报告未含此数字：§6.1.3 只说明 Terminal-Bench 报 best across harnesses；§6.3 第三方评估中 Artificial Analysis 仅覆盖 Intelligence Index v4.1，未涉及 Terminal-Bench。该 85% 具体数字无可定位的原文依据。overview.html 进一步去掉"约"和"以官网为准"对冲，表述更绝对。修法：提供 AA 官网具体链接与查询日期作为依据并标注；或删除"约 85%"具体数字，改为"Artificial Analysis 报更低分数（不同 harness/条件，见 AA 官网）"。｜ 修复：已删除 4 处"约 85%"/"85%"具体数字（index.html L968/L978/L1001 + overview.html L72），改为"AA 报更低分数"表述；并在 L968 details 补充说明"§6.3 第三方评估中 AA 仅覆盖 Intelligence Index v4.1，未涉及 Terminal-Bench"，标注具体数字以 AA 官网为准；overview.html L72 同步去掉绝对表述加"不同来源/条件"对冲。 ｜ 复验：

- [轻微·盲读] index.html 序列维度章节 `<details>`"config.json 层配置完整列表"末句："层号从 1 开始（layer 0 是 dense 层，first_k_dense_replace=1）"。`first_k_dense_replace=1` 指 0-indexed layer 0 使用 dense FFN 而非 MoE，该层仍含 KDA 注意力（kda_layers 含 1-indexed 位置 1 = layer 0）。"layer 0 是 dense 层"可能让小白读者误以为该层无注意力机制。修法：改为"layer 0（1-indexed 位置 1）使用 dense FFN 而非 MoE，注意力仍为 KDA"，或补一句"该层仍有 KDA 注意力，仅 FFN 为 dense"。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 1
- 处置：进入修复

### 核查覆盖说明

段 A 盲读：按页面顺序通读 index.html（三维度总览→序列→深度→宽度→视觉→训练→后训练→基础设施→性能→独立评价→来源说明）与 overview.html。主线无阻断卡点：三维度框架引入清晰，"一个 token 流经 93 层"贯穿线索有效，每个维度章节有自检问题，独立评价章用 callout 明确标注为解读者推断。术语首现多附一句话解释或概念页链接，折叠块（层配置、block 划分、benchmark 条件）为补充信息，收起后主线仍成立。

段 B 对照原文（k3-report.txt 全文 + config.json）逐项核查：

1. 数字与论断：Table 1 全部 18 行与原文 Table 1 一致（层数 61→93、总参 1.04T→2.78T、激活 32.6B→104.2B、专家 384→896、top-8→top-16、共享 1→2、头 64→96、上下文 128K→1M、MLA→Hybrid、SwiGLU→SiTU-GLU、61 MLA→69 KDA+24 MLA、ViT 401M/27 层等）；变化列复算正确（52%/167%/220%/50%/133%/100%/50%/8×）。2.5× scaling efficiency（§3.2/Fig.7，OOD 验证 loss，collectively）表述一致。cosine vs WSD 独立搜索后 cosine 更优（§3.2）、1% warmup + weight decay 0.1（§3.3）、NoPE 外推 1M（§3.4）、四阶段课程 8K→64K→256K→1M（§3.4）均一致。9 专家 = 3 域 × 3 努力（§4.1.2）、QAT 从 SFT 贯穿 MXFP4/MXFP8（§4.1.4）、EAGLE draft 复用 MTP 层 / 7 步展开 / LK loss（§4.1.4）、MoonEP E/R 上界（§5.2.1/附录 E）、KCP delta rule 分解 + prefix scan（§5.1.2）、AgentENV Firecracker 133ms checkpoint / 6.5× 内存超分 / 51,219,741 沙箱 / 1,505,678 镜像（§5.3.2）、外部 KV cache 池（§5.3.1）、512-token prefix hash（§5.4.1/Fig.12）、MoonViT-V2 27 层 0.4B 从零训练（§2.4/Fig.6）、Per-Head Muon（§2.5）均与原文一致。benchmark 表（编程/智能体/视觉推理）全部数字与 Table 2 逐格核对一致，第一标注正确。第三方（Table 5：AA 57.1 #4/580、Vals 74.7% #2/39、WebDev 1678 #1/99、Agent Arena 9.1 #4/37）与成本（BrowseComp $2.03、KCB 2.0 38% 成本）均一致。性能定位（落后 Fable 5/Sol，优于 Opus 4.8/GPT-5.5/GLM-5.2）与 §6.1.4 一致。

2. 实验覆盖：编程/智能体/视觉推理分域呈现强弱（含 CritPt、HLE-Full 短板），未选择性隐藏不利结果；harness 差异、Fable 5 fallback、Sol cyberguard、SWE-Marathon H20-calibrated + 35% fallback 等条件均在折叠块说明。

3. 公式与推导：F1/F2/F5/F6/F7/F8/F9/F10 按编号引用，推导下放概念页；SiTU-GLU 输出界 β₁β₂=100（β₁=4,β₂=25）、softcap 定义、QB 目标负载 q=mk/n、MOPD stop-gradient+clip、LK loss 接受率负对数均与原文 Eq.5/11/12/14/15/16 一致。

4. 可运行代码：页面无可运行代码块（仅 ASCII 图示与 config 字段引用），无需执行。

5. 事实与推断：解读者推断均用 `<em>解读者推断</em>` 标注（C19 2.5× 主要来自架构、block size=12 选择、去掉任一稳定化可能不稳定、harness 影响编程优势）；独立评价章有 callout 声明为解读者推断非论文结论。

6. 原图：未内联论文原图（Fig.2/6/7/12），用自绘 HTML 表与文字描述替代，教学简化说明中已声明；Figure 编号引用准确（Fig.1 注释 fallback、Fig.6 梯度范数、Fig.7 scaling 曲线、Fig.12 prefix cache）。

7. 前置知识引用：11 个概念页链接（kda/mla/nope/block-attnres/stable-latent-moe/situ-glu/quantile-balancing/moonvit-v2/per-head-muon/mxfp4-qat/gpu-execution-model）目标目录均存在，路径层级正确（`../../wiki/<name>/index.html`）。config.json 字段引用（num_hidden_layers=93、num_experts=896、num_experts_per_token=16、full_attn_layers/kda_layers、attn_res_block_size=12、latent_moe_use_norm、activation_situ_beta=4.0/linear_beta=25.0、mla_use_nope=true、num_shared_experts=2、first_k_dense_replace=1、routed_expert_hidden_size=3584、moe_intermediate_size=3072、num_attention_heads=96、vt_num_hidden_layers=27、max_position_embeddings=1048576、quantization_config.ignore）逐字段与 config.json 核对一致。

8. 教学简化：Table 1/2 自绘声明、"RNN 在时间上的瓶颈"类比边界说明、"token 流经 93 层"教学构造声明均到位。

9. 页面功能：KaTeX 加载但正文无 `$...$` 定界符（公式符号用纯文本/code，功能不受影响）；折叠块、目录锚点、overview↔index 互链、主题切换均正常。
