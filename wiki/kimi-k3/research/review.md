# Kimi K3 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照原文核查）
- 页面版本：index.html `8cc7160`；overview.html 未入 git（工作树 2026-08-09）
- 论文版本：arXiv:2607.24653v1，2026-07-27
- config.json：huggingface.co/moonshotai/Kimi-K3（2026-08-09 抓取）
- 时间：2026-08-09 09:22 CST

## 问题

- [重要·盲读] index.html 深度维度章节（"Block AttnRes 将 93 层分为 8 个 block（block size = 12）"）：正文说 8 个 block、block size=12，但 8×12=96≠93，读者会察觉算术不一致；7 个完整 block + 1 个 9 层部分 block 的解释藏在折叠 `<details>` 中，收起时无法消解矛盾：将"8 个 block（block size = 12）"改为"8 个 block（block size ≈ 12，含末尾部分 block）"或"7 个 12 层 block + 1 个 9 层部分 block = 8 个 AttnRes block"，使正文自洽 ｜ 修复： ｜ 复验：
- [重要·技术] index.html 性能→benchmark 条件 details 块（"Table 2 报 best across harnesses"）：报告 §6.1.3 仅对 Terminal-Bench 2.1 明确说"we report the best score across harnesses for all models"，对其他编程 benchmark 只说"each model is evaluated under one of three agentic harnesses"，未确认全部 Table 2 编程项均报 best across harnesses；页面将其泛化为 Table 2 通用规则，可能误导读者高估 K3 编程分数的可比性：将该句改为"Terminal-Bench 2.1 报 best across harnesses；其他编程 benchmark 各模型使用三种 harness 之一（§6.1.3 未逐一说明每项的 harness 选择）"，删除"Table 2 报 best across harnesses"的泛化表述 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 三维度→序列维度及多处（"OOD 验证 loss"）：OOD（Out-of-Distribution）首次出现未解释，是 2.5× scaling efficiency 的核心度量；读者无法判断"为什么用 OOD 而非普通验证 loss"：首次出现时加括注"OOD（分布外）验证 loss——用训练分布外的留出数据评估，衡量泛化而非记忆" ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 性能→benchmark 条件 details 块（"Kimi Code / Claude Code / Codex 三种 harness"）：harness（智能体运行框架）未解释，小白读者不知道 harness 指什么、为何会影响分数：首次出现时加括注"harness：智能体运行框架，决定工具接口、系统提示、上下文管理等，不同 harness 下同一模型分数不同" ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 基础设施→MoonEP（"每 rank 恰好收到 S×K token"）：S 和 K 在此处未定义（S=序列长度，K=每 token 激活专家数），读者无法理解完美均衡的含义：改为"每 rank 恰好收到 S×K 个 token（S=序列长度，K=每 token 激活专家数=16）" ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 三维度→宽度维度（"稀疏度 56×（896/16）"）："稀疏度"作为比率概念未解释，读者不知 56× 表示"每 56 个专家中激活 1 个"还是其他含义：改为"稀疏度 56×（896/16，即每 56 个路由专家中激活 1 个）" ｜ 修复： ｜ 复验：
- [轻微·技术] index.html 性能→benchmark 条件 details 块（"Artificial Analysis 报 85%"）：Terminal-Bench 2.1 的 85% 分数来自 Artificial Analysis，但 K3 报告 §6.3 仅列出 AA 的 Intelligence Index / GDPval-AA / AA-Briefcase / Harvey Lab-AA / APEX-Agents / SciCode / AA-LCR / CritPt，未包含 Terminal-Bench；该 85% 数字无法从提供的来源（报告 + config.json）定位：添加 AA Terminal-Bench 85% 的引用来源（AA 页面 URL + 日期），或将"AA 报 85%"改为"AA 报告的 Terminal-Bench 分数低于官方 88.3（具体数字以 AA 官网为准）" ｜ 修复： ｜ 复验：
- [轻微·技术] index.html 来源与教学说明→论文事实与解读者推断（"C21（MLA 用 NoPE 为了与 KDA 解耦）"）：来源节将 C21 列为正文标注的解读者推断，但正文序列维度章节的 NoPE 讨论全部以事实陈述（§2.1.2 / §3.4 原文依据）呈现，未标注"解读者推断"，与来源节登记不一致：从来源节的推断清单中删除 C21，或在正文 NoPE 讨论处为"MLA 用 NoPE 与 KDA 解耦"的推断性表述补标"解读者推断" ｜ 修复： ｜ 复验：
- [轻微·技术] index.html 后训练→QAT（"注意力、共享专家、MLP、lm_head、视觉塔保持高精度"）：config.json `quantization_config.ignore` 还包含 `re:.*mm_projector.*`（多模态投影器），报告 §4.1.4 还提到"MoE routers"保持高精度；页面列举的未量化组件不完整：将列表补为"注意力、共享专家、latent MoE 投影（W↓/W↑）、MoE router、lm_head、视觉塔、多模态投影器保持高精度" ｜ 修复： ｜ 复验：
- [轻微·技术] index.html 训练→数据与 scaling law（"captions、图文交错、OCR、视频、视觉编程数据"）：报告 §3.1 原文为"captions, interleaved image–text documents, OCR, perception, video, and visual coding data"，页面遗漏"perception"：补为"captions、图文交错、OCR、感知（perception）、视频、视觉编程数据" ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html `<title>` 与 `<h1>` 不一致：`<title>` 为"三维度信息流与 2.5× Scaling 效率"，`<h1>` 为"三维度扩展信息流的开源 3T 级模型"，两者聚焦不同（效率 vs 规模），且 `<h1>` 的"3T 级"与正文表格"2.78T"需读者自行关联：无需改 `<title>`，但在 `<h1>` 首次出现"3T 级"时加括注"2.78T 参数，3T 级指规模类别" ｜ 修复： ｜ 复验：
- [轻微·技术] index.html 基础设施→1M 智能体 RL（"共创建 51,219,741 个沙箱"）：报告 §5.3.2 原文为"a total of 51,219,741 sandboxes across 1,505,678 images were created"，页面省略"across 1,505,678 images"，丢失了沙箱规模的关键维度：补为"共创建 51,219,741 个沙箱（跨 1,505,678 个镜像）" ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 10
- 处置：进入修复

### 审查说明

段 A 盲读以完全小白视角按页面顺序通读 index.html，记录理解主线卡点。页面三维度框架（序列/深度/宽度）的主线清晰，贯穿例子（"一个 token 流经 93 层"）有效串联各章，自检问题辅助理解。核心卡点集中在：8×12≠93 的算术矛盾（折叠块内消解）、OOD/harness/稀疏度等术语首现未解释。无阻断级卡点——主线不依赖展开折叠块即可成立。

段 B 逐条核对 index.html 与 overview.html 的数字、公式引用、实验条件、事实/推断标记、链接有效性、教学简化。Table 1 全部 21 行与报告 Table 1 一致；Table 2 节选三表（编程/智能体/视觉推理）逐格核对通过；config.json 全部引用字段（num_hidden_layers=93, num_experts=896, num_experts_per_token=16, attn_res_block_size=12, full_attn_layers 24 项, kda_layers 69 项, mla_use_nope=true, latent_moe_use_norm=true, activation_situ_beta=4.0, activation_situ_linear_beta=25.0, routed_expert_hidden_size=3584, max_position_embeddings=1048576, vt_num_hidden_layers=27, quantization_config.ignore 等）均与页面表述一致；11 个子页面链接目标全部存在；overview.html 与 index.html 互相链接有效；首页链接 `../../index.html` 有效。解读者推断（C19/C20 及宽度/性能章节内联推断）均标注"解读者推断"，独立评价章节以 callout 整体标注。发现 2 项重要问题（8×12 算术矛盾、harness 泛化）和 10 项轻微问题，均可通过最小化修改修复，无需改变研究范围或教学大纲。
