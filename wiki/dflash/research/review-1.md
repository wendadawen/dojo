# DFlash 审查记录（第 1 轮）

- 页面版本：wiki/dflash/index.html 与 overview.html（工作树未哈希，编辑器自上次改动未提交）
- 论文版本：arXiv:2602.06036v2（2026-05-28 修订），TeX 源码固定（abstract / intro / related / preliminaries / method / exp / appendix / tables/results/main、reasoning）
- 审查时间：2026-08-19
- 审查者：独立子代理（首轮）
- 已完整阅读章节：index.html 全部正文（含 4 张原图 base64 解码后的内容核对）、overview.html 全部、check.md / style-guide.md / write.md、main.tex / abstract.tex / intro.tex / preliminaries.tex / method.tex / exp.tex / conclusion.tex / related.tex / appendix.tex / tables/results/main.tex / tables/results/reasoning.tex / macros.tex、dflash2-blog.txt
- 论文图/表权威基准（按 TeX 源码 \begin{figure}/\begin{table} 出现顺序；LaTeX 计数器在 \caption 处按源码顺序递增）：
  - Figure 1 = `dflash_speedup.pdf`（intro）
  - Figure 2 = `dflash_inference_design.pdf`（preliminaries §3）
  - Figure 3 = `draft_latency_bar.pdf`（preliminaries §3.2 末段）
  - Figure 4 = `dflash_attn.pdf`（method §4.2 训练注意力）
  - Figure 5 = `acceptance_length_vs_epoch.pdf`（appendix A.5.1 loss decay）
  - Table 1 = main-results（method §4.2 末 `\input{tables/results/main}`）
  - Table 2 = reasoning-results（exp §5.2 末 `\input{tables/results/reasoning}`）
  - Table 3 = sglang-all（exp §5.3）
  - Table 4 = long-context（exp §5.4）
  - Table 5 = dflash_vs_eagle3_llama31_acc（exp §5.5.1 LLaMA 同数据对照）
  - Table 6 = ablation_draft_layers（exp §5.5.2）
  - Table 7 = ablation_target_hiddens（exp §5.5.3）
  - Table 8 = ablation_block_size（exp §5.5.4）
  - Table 9 = ablation_kv_injection（exp §5.5.5）
  - Table 10 = naive_diffusion（appendix A.2）
  - Table 11 = more-models-sglang（appendix A.4）
  - Table 12 = vllm-qwen35-9b（appendix A.4）
  - Table 13 = ablation_sample_block（appendix A.5.2 anchor 采样）
  - 论文主章节：§1 Intro、§2 Related Work、§3 Preliminaries、§4 Method（4.1 Inference / 4.2 Training）、§5 Experiments（5.1–5.5）、§6 Conclusion、Appendix A（A.1–A.5）

## 问题

### 阻断

- [阻断·来源] index.html:795, :799, N10 (§2.2 段尾 + Fig. 4)：draft_latency_bar 标注"论文 Figure 4"，但按 TeX 源码顺序该图为 Figure 3（preliminaries.tex 第 41–46 行的 `\begin{figure}[H]`，介于 fig:inference (Fig 2) 与 fig:train (Fig 4) 之间）｜引文依据：preliminaries.tex 第 42 行 `\includegraphics[]{figures/draft_latency_bar.pdf}` 与第 44–45 行 `\caption{...}\label{fig:draft_cost}`；同节同时还有 method.tex 第 27–28 行 `fig:train`、intro.tex 第 11–15 行 `fig:speedup`、preliminaries.tex 第 4–8 行 `fig:inference`、appendix.tex 第 149–152 行 `fig:ablation_acc`（Fig 5），源序 1→2→3→4→5 与原图一致｜修复要求：把 :795 与 :799 的"Figure 4"改为"Figure 3"，并把 N10 "§2.2 段尾 + Fig. 4"改为"§3.2 段尾 + Fig. 3"（章节号另见下条）｜修复：｜复验：
- [阻断·来源] index.html:938, :942 (DFlash 训练注意力图)：attn 图标注"论文 Figure 3"，但按 TeX 顺序该图为 Figure 4（method.tex 第 26–31 行的 `\begin{figure}[t]` 与 `\label{fig:train}`）｜引文依据：method.tex 第 28 行 `\includegraphics[width=1.0\linewidth]{figures/dflash_attn.pdf}`、第 30 行 `\label{fig:train}`；与 draft_latency_bar 同节判定为 Figure 3 不同｜修复要求：把 :938 与 :942 的"Figure 3"改为"Figure 4"，并把 §3.2 Training 改为 §4.2 Training（章节号偏移见下条）｜修复：｜复验：
- [阻断·来源] index.html:956 (loss decay 训练曲线引用)：正文写"论文 Figure 7（`acceptance_length_vs_epoch`）"，但该图在论文中是 Figure 5（appendix A.5.1）；论文 Figure 7 不存在｜引文依据：appendix.tex 第 149–154 行 `\begin{figure}[h!]` + `\includegraphics{acceptance_length_vs_epoch.pdf}` + `\caption{...}\label{fig:ablation_acc}`，该图为 TeX 源码第 5 个 figure 环境（=Fig 5）；论文中没有 Figure 6 或 Figure 7（C5、N10、C13、C14 等的 \autoref 都引 ≤ 5）｜修复要求：把 :956 的"论文 Figure 7（acceptance_length_vs_epoch）"改为"论文 Figure 5（`acceptance_length_vs_epoch`）"；同步把 C9 :1175 "Fig. 7 训练曲线"改为"Fig. 5 训练曲线"｜修复：｜复验：
- [阻断·来源] index.html 多处章节号系统性偏移（详见 A 类）：正文与「来源与范围说明」把 Related Work 当作 §2、Method 当作 §3、Ablation §5.5 当作 §5.4 系列；按 main.tex 的 \input 顺序（intro→related→preliminaries→method→exp→conclusion→appendix），正确的章节号是 §2=Related Work、§3=Preliminaries、§4=Method、§5=Experiments。具体错位点包括：
  - :795 "§2.2 Eq. (3) 之后" → §3.2
  - :867 "§3.1 Inference" 与 :871 "§3.1" → §4.1
  - :877 补充块 "§3.1 明确说..."（另见来源 11 条）→ §4.1
  - :932, :938, :942 "§3.2 Training" → §4.2
  - :1170 C4 "§3.1 Inference" → §4.1
  - :1171 C5 "§3.1 Inference 段尾" → §4.1
  - :1172 C6 "§3.1 Inference 末段" → §4.1
  - :1173 C7 "§3.2 Training 段首" → §4.2
  - :1174 C8 "§3.2 Training 中段" → §4.2
  - :1175 C9 "§3.2 末段 Eq. (4) 前后" → §4.2
  - :1176 C10 "§3.2 末段" → §4.2
  - :1186 F1 "§2.1 Eq. (1)" → §3.1
  - :1187 F2 "§2.2 Eq. (2)" → §3.2
  - :1188 F3 "§2.2 Eq. (3)" → §3.2
  - :1193 F6 "§3.2 Eq. (4)" → §4.2
  - :1178 C12 "§5.4 Tab. 6" → §5.5.2
  - :1179 C13 "§5.4 Tab. 8" → §5.5.4
  - :1182 C16 "§5.4.1 Tab. 3" → §5.5.1
  - :1183 C17 "Tab. 2、Tab. 10" → Table 3（SGLang）、Table 12（vLLM）
  - :1197 N4 "§5.3 Tab. 2" → §5.3 Tab. 3
  - :1197 N5 "A.4 Tab. 10" → A.4 Tab. 12
  - :1197 N6 "§5.4.1 Tab. 3" → §5.5.1 Tab. 5
  - :1197 N7 "§5.4.4 Tab. 9" → §5.5.5 Tab. 9
  - :1197 N8 "§5.4 Tab. 6" → §5.5.2 Tab. 6
  - :1197 N10 "§2.2 段尾" → §3.2
  - :1197 N13 "§5.5 Tab. 4" → §5.4 Tab. 4（long-context 在 §5.4，Tab 4）
  - :1197 N14 "A.4 Tab. 9" → A.4 Tab. 11
  - :1147 评估表并发边界行 "SGLang Tab. 2、vLLM Tab. 10" → Table 3、Table 12
  - :1150 评估表负结果行 "依据 Tab. 5" → 依据 Tab. 10
  - :1131 章节问题解答 "①KV 注入 vs 输入融合（Tab. 9）" Tab 号正确，"②层数（Tab. 6）"正确，"③训练/推理块大小（Tab. 8）"正确
  - 引文依据：main.tex 第 33–45 行的 \input 顺序（abstract→intro→related→preliminaries→method→exp→conclusion→ack→impact→biblio→appendix）；exp.tex 内 \subsection 顺序（5.1 Instruct / 5.2 Reasoning / 5.3 Serving / 5.4 Long Context / 5.5 Ablation）及 \subsubsection 顺序（5.5.1 Training Data / 5.5.2 Number of Draft Layers / 5.5.3 Number of Target Hidden Features / 5.5.4 Training-Inference Time Block Size / 5.5.5 KV Injection vs. Input Fusion）｜修复要求：按上面映射把全文（包括正文与「来源与范围说明」）所有 §X.Y 引用统一更新；每条改正须可在原文中搜索到唯一匹配；建议一次性脚本替换后逐条核对｜修复：｜复验：

### 重要

- [重要·来源] index.html:934, :987, :1173 (Tab. 7 三处)：正文与 C7 把 anchor 采样消融标为"Tab. 7"，实际应为 Table 13（appendix A.5.2 中的 `tab:ablation_sample_block`）｜引文依据：appendix.tex 第 158–186 行 `\begin{table}` 内 `\label{tab:ablation_sample_block}`；Table 7 是 `ablation_target_hiddens`（exp.tex 第 285 行），与 anchor 采样无关；表格数据（Math500 5.64/4.94、HumanEval 4.61/3.86、MT-Bench 3.18/2.80）与 appendix.tex 第 174–184 行一致，可验证确为 anchor 采样｜修复要求：把 :934, :987, :1173 的"Tab. 7"改为"Tab. 13"；将对应 §3.2 Training 引用同步改为 §4.2（见章节偏移条）｜修复：｜复验：
- [重要·来源] index.html:1073, :1182 (LLaMA 同数据对照)：正文写"论文 Tab. 3"，实际为 Table 5（exp §5.5.1 `tab:dflash_vs_eagle3_llama31_acc`）｜引文依据：exp.tex 第 179–250 行 `\begin{table}` + 第 181 行 `\label{tab:dflash_vs_eagle3_llama31_acc}`；C16 :1182 "§5.4.1 Tab. 3"亦同错｜修复要求：把 :1073 的"论文 Tab. 3"与 N6 :1197 的"§5.4.1 Tab. 3"、C16 :1182 的"§5.4.1 Tab. 3"统一改为"§5.5.1 Tab. 5"｜修复：｜复验：
- [重要·来源] index.html SGLang 与 vLLM 表格号错位：:1147 评估表并发边界 "SGLang Tab. 2、vLLM Tab. 10"、C17 :1183 "Tab. 2、Tab. 10 的并发下降趋势"、N4 :1197 "§5.3 Tab. 2"、N5 :1197 "A.4 Tab. 10"，均将 SGLang (`tab:sglang-all`) 误标为 Tab 2、将 vLLM (`tab:vllm-qwen35-9b`) 误标为 Tab 10｜引文依据：exp.tex 第 32 行 `\begin{table}` + 第 34 行 `\label{tab:sglang-all}`（= Table 3）；appendix.tex 第 124–141 行 `\label{tab:vllm-qwen35-9b}`（= Table 12）；Table 2 = reasoning-results（exp.tex line 28）、Table 10 = naive_diffusion（appendix.tex line 13）｜修复要求：把上述四处"Tab. 2"改为"Tab. 3"，"Tab. 10"改为"Tab. 12"；N4 保留 §5.3，N5 保留 A.4，章节号正确，仅表格号错｜修复：｜复验：
- [重要·来源] index.html:1149 评估表负结果行 + C5/N11 多处 (Table 5)：naive diffusion 引文标为"Table 5"，实际为 Table 10（appendix A.2 `tab:naive_diffusion`）｜引文依据：appendix.tex 第 13 行 `\begin{table}` + 第 16 行 `\label{tab:naive_diffusion}`；Table 5 = LLaMA 同数据对照（exp.tex line 181）；数据 GSM8K 2.83/3.38 等与 appendix.tex 第 32–35 行一致｜修复要求：把 :1149 评估表负结果行"依据 Tab. 5（naive diffusion）"改为"依据 Tab. 10（naive diffusion）"；C5 :1171 "A.2 Table 5"改为"A.2 Table 10"；N11 :1197 "A.2 Tab. 5"改为"A.2 Tab. 10"｜修复：｜复验：
- [重要·来源] index.html:1197 N14 (更多模型)：标为"A.4 Tab. 9"，实际为 Table 11（appendix A.4 `tab:more-models-sglang`）｜引文依据：appendix.tex 第 81 行 `\begin{table}` + 第 84 行 `\label{tab:more-models-sglang}`；Table 9 = `ablation_kv_injection`（exp.tex line 360），更多模型与之无关｜修复要求：N14 "A.4 Tab. 9"改为"A.4 Tab. 11"｜修复：｜复验：
- [重要·来源] index.html:880 (引文位置错误)：正文写"论文 §3.1 明确说「They bypass the draft model's Q projection, output projection, self-attention update, and FFN」"，但该原文出自 appendix A.3 (`\label{appd:kv-injection}`)，非 §3.1（Method Inference = §4.1）｜引文依据：appendix.tex 第 57–66 行完整句子 "target features only serve as additional KV entries for the masked-block draft tokens. They bypass the draft model's Q projection, output projection, self-attention update, and FFN."；§4.1 没有这句｜修复要求：把 "论文 §3.1 明确说"改为"论文附录 A.3 明确说"｜修复：｜复验：
- [重要·来源] index.html:1177 C11 (Eq. 5)：写"Eq. (5) 矩阵形状分析"，但论文仅有 Eq.(1)–Eq.(4) 四个编号公式；A.3 的公式用 \[ \] 显示数学，未编号，不存在 Eq.(5)｜引文依据：preliminaries.tex 第 18、28、36 行三处 `\begin{equation}`（Eq 1/2/3），method.tex 第 49 行 Eq 4 (`eq:loss-decay`)；appendix.tex 第 49、58 行用 `\[ ... \]` 无 `\label`，LaTeX 不会分配编号｜修复要求：把 C11 "Eq. (5) 矩阵形状分析"改为"A.3 矩阵形状分析"（删除 Eq.(5)）｜修复：｜复验：
- [重要·技术] index.html §1 与 §3.3 中符号 γ 多重含义未声明：§1 (F1/F2，line 782, 787) 用 γ 表示"周期内草稿 token 数"；§3.3 (F6，line 953) 用 γ 表示"loss decay 衰减率超参"；同一变量在同一页面含义冲突未说明，违反 style-guide §11"同一变量在页面中保持同一种写法"｜引文依据：preliminaries.tex 第 17–22 行 `\gamma` = speculation budget，method.tex 第 49–52 行 `\gamma` = decay rate；论文本身两处都用 `\gamma` 但含义不同（论文未改符号）｜修复要求：在 F6（:953）符号定义后补一句说明"注：此处 γ 与 §1/F1–F2 中的 γ（草稿 token 数）含义不同，勿混用"；或把 F6 的 γ 改为不同符号（如 `α` 或 `τ_decay`）并在 §1 不动 |修复：｜复验：
- [重要·技术] overview.html:65, index.html:1146 (MT-Bench 范围)：两处写"MT-Bench 对话任务一致最低（2.5–2.8× T=0）"，但论文 Tab. 1 中 T=0 MT-Bench speedup 实为 Q3-4B 2.85× 与 Q3-8B 2.75×（范围 2.75–2.85×）；2.5× 与论文不符（2.5× 接近 T=1 最低 2.47×，被错放到 T=0 行）｜引文依据：tables/results/main.tex 第 16–17 行 Q3-4B MT-Bench T=0 `2.85×` 与 Q3-8B `2.75×`；同一节 index.html :766 核心问题 Q5 答案（"T=0 仅 2.75–2.85×"）已用正确范围 |修复要求：overview :65 与 index :1146 两处 "2.5–2.8× T=0" 改为 "2.75–2.85× T=0"｜修复：｜复验：
- [重要·技术] index.html:1092 (EAGLE-3(60) 0.6–0.9×)：正文写"EAGLE-3(60) 在并发 16/32 出现 0.6–0.9×（比基线还慢）"，但论文 Tab. 5 (LLaMA) 并发 16/32 实测值为 GSM8K 0.9/0.6、HumanEval 0.9/0.6、Alpaca 0.8/**0.5**，范围下限应为 0.5×；且页面 :1086 表格自身列出 0.8 与 0.5，文字 0.6–0.9 与自身表格内部矛盾｜引文依据：exp.tex 第 199–207 行 E3(60) 三个任务五并发值；最低 0.5×（Alpaca 并发 32）｜修复要求：把 "0.6–0.9×" 改为 "0.5–0.9×"；并复核 overview :64（见轻微条）｜修复：｜复验：
- [重要·来源] overview.html:67, index.html:1124 (最大生成长度 4096)：两处实验条件句写"采样温度与最大生成长度 2048/4096"，但论文 Tab. 1 caption 仅"最多 2048 generated tokens"；4096 仅出现在训练最大序列长度（appendix A.1 "3072 tokens (4096 for Qwen3-Coder)"）｜引文依据：tables/results/main.tex 第 5 行 "maximum of 2048 generated tokens"；appendix.tex 第 7 行训练序列长度（非生成长度）｜修复要求：overview :67 与 index :1124 两处把 "2048/4096" 改为 "2048"（或明确写"训练序列长度 3072/4096"以与生成长度区分）｜修复：｜复验：
- [重要·来源] index.html:1149 评估表长上下文行（缺模型条件）：正文写"4K 训练 base 草稿器在 16K 衰减（hotpotqa 4.91→3.61），1.6K LongAlign 样本微调 3 epochs 恢复到 6.05"，但未给出使用的模型条件；论文 §5.4 明确"the base **Qwen3.5-27B** DFlash draft model"，该数字仅对 Qwen3.5-27B 草稿器成立（用户重点关注：实验数字必须带条件）｜引文依据：exp.tex 第 141 行 "We fine-tune the base Qwen3.5-27B draft model with 1.6K samples from LongAlign-10K"；exp.tex 第 145 行表格标题"Acceptance length of the base Qwen3.5-27B DFlash drafter"｜修复要求：把 "4K 训练 base 草稿器" 改为 "Qwen3.5-27B 草稿器（4K 上下文训练 base 版）"；在 N13 增补模型条件 Qwen3.5-27B｜修复：｜复验：
- [重要·技术] overview.html:58 (T=1 符号冲突)：正文写 "draft 以块扩散单步方式（$T=1$）一次前向并行预测整块"，但 T=1 在同一文档上下文被用作采样温度（:62 "T=0"，:65 T=1 也指采样温度）；T=1 作为单步去噪的符号未在论文中定义（论文仅说"a single forward pass"与"aggressive reduction in denoising steps"）｜引文依据：overview.html :58, :62, :65；method.tex 全文未用"T=1"表示去噪轮数；tables/results/main.tex 第 5 行 `Temperature = 0`/`Temperature = 1` 表示采样温度｜修复要求：把 "$T=1$" 改为 "$T_{\text{denoise}}=1$" 或改为不含符号的"单步方式（单次去噪）"以避免与采样温度 T=0/T=1 冲突｜修复：｜复验：
- [重要·来源] index.html:727 (C1 引用错位)：正文写 "6.1× 无损加速、相对 EAGLE-3 高 2.4× [C1]"，但 C1 在来源说明中定义为"2–3× 起草天花板"（preliminaries.tex 段落），不能覆盖 Qwen3-8B 上 6.1× 与对 EAGLE-3 的 2.4× 数字｜引文依据：index.html :1167 C1 定义"§1 引言第二段"；intro.tex 第 24 行 "achieves up to a 6.1× speedup on Qwen3-8B"；exp.tex §5.1 第 19 行 "a 2.4× improvement over EAGLE-3 (16)"｜修复要求：6.1× 改为 [N1]（主表 T=0），相对 EAGLE-3 2.4× 改为 [N1] 或新增 C/F/N；删除或调整 [C1] 引用位置｜修复：｜复验：
- [重要·技术] index.html:876 (naive diffusion 同档误判)：正文写"无 KV 注入的纯块扩散草稿器在 GSM8K 上加速 2.83×...和 EAGLE-3 同档"，但论文 Tab. 1 中 E3(16) GSM8K=1.99、E3(60)=2.27，naive diffusion（Tab. 10）=2.83，naive 显著高于 E3(16)；"和 EAGLE-3 同档"是与论文表格数据相悖的分析性判断，且未标注为分析性推断｜引文依据：tables/results/main.tex 第 17 行 Q3-4B GSM8K E3(16)=1.99×、E3(60)=2.27×、DFlash=5.15×；appendix.tex 第 32 行 naive GSM8K T=0=2.83×｜修复要求：把 "和 EAGLE-3 同档" 改为 "实际略高于 EAGLE-3(16)（1.99×）和 EAGLE-3(60)（2.27×）的同任务值，但仍未到 4–6×，说明 KV 注入是关键增益"，并将该句标为 [C17] 分析性判断｜修复：｜复验：
- [重要·技术] index.html 多处未标注分析性推断（用户重点关注：方法评价章 C17 分析性推断标注）：来源说明 :1163 声称"分析性判断集中在「5. 方法评价」与 C17"，但实际正文 §2–§4 多处含未标注的分析性推断：
  - :1030 "思考链中多步推理/公式化表达对草稿器更不友好"—论文 §5.2 未解释原因
  - :1051 "绝对数字差异源于 batch 调度与后端实现"—论文未给原因
  - :1051 "说明 KV 注入对 MoE 目标同样有效"—论文 A.4 仅列数值未做此归因
  - :1069 "SGLang vs vLLM 数字可推断 vLLM 落后约 0.5–1×——这是引擎实现差异，不是方法问题"—vLLM 模型为 Qwen3.5-9B，与 SGLang Q3-4B/8B 跨模型比较，0.5–1× 无可复现依据
  - :880 补充块 "这条设计简化了梯度流（只更新 K/V 路径的梯度），也避免了「target 特征改写 draft 内部表示」导致的训练不稳"—无来源机制描述
  - :1145 评估表训练效率行 "训练无 test-time test 开销（对比 EAGLE 系列 [C16]）"—论文 §4.2 原文为 "costly training-time test"（术语倒装），且 [C16]（LLaMA 同数据对照）不能支持此论断
  - :1151 评估表未做行 "未与 TiDAR/DiffuSpec/SpecDiff-2 实验对比（开源实现缺失）[C16]"—同上 [C16] 引用错位
  引文依据：exp.tex 各章节均未给出上述机制解释；来源说明 :1183 C17 自身定义为"综合 Tab. 2、Tab. 10 的并发下降趋势与投机解码的资源利用直觉"——但其引用的 Tab. 2（=reasoning）和 Tab. 10（=naive）都不涉及并发，且应改为 Table 3、Table 12｜修复要求：把上述每条加 [C17] 标注或就地写"分析性推断"；调整 C17 定义覆盖全部 [C17] 实例；移除错误的 [C16] 引用（:1145, :1151）并归到正确的论断编号；修正 C17 自身 Table 号（Table 3、Table 12）｜修复：｜复验：
- [重要·来源] index.html:1030 (reasoning 4.5×/3.9× 对应关系)：正文写"Qwen3-4B 加速约 4.5×、Qwen3-8B 约 3.9×"，把论文 §5.2 原文 "roughly 4.5× and 3.9×" 对应到 Q3-4B/Q3-8B；但论文表格 (Tab. 2) 两模型 T=0 平均加速为 Q3-4B 4.40、Q3-8B 4.44（接近）、T=1 平均为 3.75/3.83（接近），两个数字与模型的对应关系与表数据不符；4.5× 更合理对应 T=0 两模型平均，3.9× 对应 T=1 两模型平均｜引文依据：tables/results/reasoning.tex Q3-4B T=0 (4.23+4.59+4.39)/3=4.40，Q3-8B T=0 (4.17+4.64+4.51)/3=4.44，Q3-4B T=1 (3.67+3.93+3.64)/3=3.75，Q3-8B T=1 (3.75+4.03+3.70)/3=3.83｜修复要求：去掉 4B/8B 归属，改写为"思考模式 T=0 平均约 4.4×，T=1 平均约 3.8×，论文§5.2 概述为 'roughly 4.5× and 3.9×'"｜修复：｜复验：
- [重要·技术] index.html:1030 (reasoning 段"仍超过 4×")：正文写"但绝对加速比仍超过 4×"，但论文 Tab. 2 中 T=1 六个任务值中 Q3-4B T=1 GPQA=3.67×、AIME25=3.64×、Q3-8B T=1 GPQA=3.75×、AIME25=3.70× 均低于 4×，只有 Q3-8B T=1 MATH500=4.03× 与 Q3-4B T=1 MATH500=3.93× 接近 4×｜引文依据：tables/results/reasoning.tex 行 Q3-4B/Q3-8B T=1 GPQA、AIME25、MATH500 三个任务的 speedup｜修复要求：把"绝对加速比仍超过 4×"改为"T=0 平均约 4.4×、T=1 平均约 3.8×，仍具长 CoT 部署价值"，与上面 #20 合并叙述｜修复：｜复验：
- [重要·链接] index.html:729 (前置概念无链接)：正文写"前置概念——投机解码循环、注意力中的 K/V、块扩散范式、EAGLE-3 的特征级自回归起草——在对应链接页已讲过，本文只引用其结论"，但仅 §1（:795）与 §3（:938）给了 `../block-diffusion/index.html` 链接，其余三个前置概念（投机解码、EAGLE-3、K/V）未给链接；wiki 中 `speculative-decoding/` 与 `eagle-speculative/` 页面存在（`ls /Users/wendadawen/code/dojo/wiki/` 确认）｜引文依据：style-guide §13、write.md §4.1、check.md 2.2.8；wiki 目录中 speculative-decoding/、eagle-speculative/ 页面均已存在｜修复要求：补充"投机解码（`<a href="../speculative-decoding/index.html">...</a>`）"、"EAGLE-3（`<a href="../eagle-speculative/index.html">...</a>`）"、"注意力中的 K/V（`<a href="../standard-attention/index.html">...</a>` 或 block-attnres）"等首次出现的概念链接｜修复：｜复验：
- [重要·格式] overview.html:69 (残留组件注释)：HTML 注释 `<!-- 概览正文：研究问题、主要贡献、方法概述和关键结论；不展开公式推导。 -->` 仍保留在文档中｜引文依据：write.md 4.11 "清除全部【…】占位符、`<!-- @content -->` 标记、组件标记和组件说明注释"；overview.html :69 注释明显为页面作者遗留的写作提示｜修复要求：删除该行注释｜修复：｜复验：
- [重要·格式] index.html 197 处 + overview.html 3 处裸 Unicode 字符 × (U+00D7)："6×"、"4.91×/4.86×"、"2.5×"、"6.09×"、"2.4×"、"5.1×"、"2.8×"、"0.6–0.9×"、"2–3×"、"4–6×"、"2.5–2.8×" 等所有"×N"倍数标记均为裸字符 |引文依据：style-guide §11"数学运算符和关系符都必须包在 `$...$` 或 `$$...$$` 中，禁止直接使用 Unicode 数学字符替代"+".dojo/scripts/validate.py 对公式定界符之外的数学字符报错"；write.md 4.2 同要求；本次扫描 index.html 中 `U+00D7` 共 197 次、overview.html 中 3 次（grep U+00D7 全 HTML 文件计得）｜修复要求：所有 "NX" 改为 "$N\times$"（保留 N 的可读性建议在公式外加美元符即可）；可机械替换 |修复：｜复验：

### 轻微

- [轻微·来源] overview.html:64 (EAGLE-3(60) 范围)：与 index :1092 同源的 0.6–0.9× 错误，但 overview 无内部表格对照；改为 "0.5–0.9×" 即可 |修复要求：overview :64 "0.6–0.9×" 改为 "0.5–0.9×"｜修复：｜复验：
- [轻微·来源] index.html:953 (γ 取值措辞)：正文写"块 16 取 7、块 10 取 5、块 8 取 4（不同块大小有不同最优值）"；论文 A.1 仅写"is set to 7 for block size 16, 5 for block size 10, and 4 for block size 8 models"，未声明这是"最优值"（可能为作者经验值或调参结果，论文未明确）｜引文依据：appendix.tex 第 7 行 "The hyperparameter γ for the loss decay in Eq. (4) is set to 7 for block size 16, 5 for block size 10, and 4 for block size 8 models"｜修复要求：把"不同块大小有不同最优值"改为"论文设定值（未声明为全局最优）"或"论文调参取值"｜修复：｜复验：
- [轻微·来源] index.html:977 (800K 来源归属)：正文把 800K 样本数归到 [N9]（定义为"A.1 整段"），但 800K 实际出自 §5 Datasets 段而非 A.1｜引文依据：exp.tex 第 6 行 "around 800K samples"；appendix.tex 第 7 行仅 AdamW/lr/序列长度等训练超参，未提及 800K｜修复要求：把"800K 样本（Nemotron Post-Training Dataset V2 + CodeAlpaca...）" 的来源改为 §5 Datasets（新增 N 编号或归入 N1），并把 N9 改为仅涵盖训练超参｜修复：｜复验：
- [轻微·来源] index.html:1181 C15 (跳号管理)：C15 在正文中未被引用，仅在来源说明里以"未引用为论断（避免 C 编号管理混乱）"占位；style-guide §6 要求 C/F/N 编号与正文双向对应，跳号是次优解但非错｜修复要求：把 C16 重编号为 C15、C17 重编号为 C16，删除 C15 占位条目；同时把正文内所有 [C16]/[C17] 引用同步更新｜修复：｜复验：
- [轻微·来源] index.html:1197 N14 (正文未引用)：N14 在 §4 实验章节正文中无任何对应 sup 引用，违反 style-guide §6 双向对应原则｜引文依据：grep `[N14]` 在 index.html 正文中 0 命中｜修复要求：删除 N14；或在 §4 适当位置补充对 Qwen3.5-9B/27B/35B-A3B、GPT-OSS-20B/120B 的引用（带条件）｜修复：｜复验：
- [轻微·格式] index.html:1001 (四块/五块不一致)：正文写"论文把实验分四块：Instruct 模型主结果、Reasoning 模型、Serving 框架（SGLang）、LLaMA 与 EAGLE-3 同数据对照、Ablation"，列举了 5 块但写"四块"；论文 §5 含 5 个 \subsection（5.1–5.5），且 LLaMA 对照属于 Ablation 子节 |修复要求：把"分四块"改为"分五块"（或合并 LLaMA 对照到 Ablation 描述使其只剩 4 块）｜修复：｜复验：
- [轻微·可读性] index.html:1034 ("Qwen3-3 系 + Coder")："Qwen3-3 系"含义不明，疑为"Qwen3 三系"笔误｜引文依据：上下文是 SGLang 表格，包含 Qwen3-4B、Qwen3-8B、Qwen3-Coder-30B-A3B 三模型｜修复要求：把"Qwen3-3 系 + Coder"改为"Qwen3-4B/8B + Qwen3-Coder-30B-A3B"或"Qwen3 三系（4B/8B/Coder）"｜修复：｜复验：
- [轻微·可读性] index.html:795 ("3 层/5 层 DFlash（每层一次并行）")：可被误读为"5 层 = 5 次并行前向"，与单步并行的核心机制冲突｜引文依据：preliminaries.tex 第 35–39 行 diffusion drafter "generate all γ tokens in parallel within a single forward pass"｜修复要求：把"（每层一次并行）"改为"（5 层深度、单次前向、并行起草整块）"｜修复：｜复验：
- [轻微·来源] overview.html:43, :48 与 index.html:720 ("UCSD/Z Lab")：作者单位标"UCSD/Z Lab"，但论文 `\icmlaffiliation{ucsd}{UC San Diego}` 仅 UCSD 一个单位，Z Lab 来源仅为项目仓库 z-lab/dflash 与网站 dflash.z-lab.ai（不是论文 affiliation）｜引文依据：main.tex 第 14–21 行 `\icmlaffiliation{ucsd}{UC San Diego}` 单 affiliation；abstract.tex 第 8 行 code/model 链接指向 z-lab GitHub 与 Hugging Face org｜修复要求：把 "UCSD/Z Lab" 改为 "UC San Diego（项目仓库 z-lab）" 或简化为 "UC San Diego"；保持 GitHub/Hugging Face 链接的展示；元数据 description 中也对应修改｜修复：｜复验：
- [轻微·格式] overview.html:46, :62, :63, :64, :65 (× 字符)：已在重要条统一记录；3 处 × 全部纳入 #B 修复范围｜修复要求：见重要条（统一为 $N\times$）｜修复：｜复验：
- [轻微·格式] index.html:1152 (生态数字使用 `[厂商宣称]` 而非 C/F/N)：`<sup>[厂商宣称]</sup>` 不是 C/F/N 标准编号格式；语义上厂商标注是好的，但与规范不一致｜引文依据：style-guide §6、write.md 4.10、check.md 2.2.13｜修复要求：改为 `<sup>[N生态]</sup>` 并在 N 列表加 N 生态条目（指向 Inco AI 博客 2026-08-18）｜修复：｜复验：
- [轻微·格式] index.html:1158 (`[Conclusion 原文意译]`)：方法定位段用 `<sup>[Conclusion 原文意译]</sup>` 标注翻译来源，非 C/F/N 标准格式｜引文依据：同上｜修复要求：去掉上标，改用脚注或方法评价表格中"原文意译"列；或在 [Conclusion] 之外加 [N定位] 编号｜修复：｜复验：
- [轻微·技术] index.html:1043–1048 (SGLang 表格条件)：表格列出了模型/任务/conc 1/conc 32/τ，但缺 EAGLE-3 draft steps=7 / top-k=10 这一 baseline 参数说明（论文 §5.1 "the draft steps and top-k are set to 7 and 10"）；SGLang 表格本身仅对比 DFlash 与 AR（不含 EAGLE-3），故 EAGLE-3 参数不直接适用；条件标注应说明"Draft steps 与 top-k 不适用 SGLang 表（表内仅 DFlash vs AR）"｜引文依据：exp.tex 第 19 行 + Tab. 3 仅 DFlash 与 Baseline 两行｜修复要求：在表格上方补一句"本表不含 EAGLE-3 基线（见 §4.5 LLaMA 同数据对照与 N6 中 EAGLE-3 的 draft steps=7/top-k=10）"；N4 加"本表仅 DFlash vs AR baseline，不含 EAGLE-3"｜修复：｜复验：
- [轻微·技术] index.html:1075–1089 (LLaMA 表格条件缺 EAGLE-3 参数)：表格头与 N6 列出了 SGLang/Spec-v1/Flashinfer/B200/块 10 vs 树 10/60，但 EAGLE-3 "7 draft steps with top-k=10" 未在表内或 N6 写出｜引文依据：exp.tex 第 180 行表格标题与第 251 行段首 "DFlash is trained on UltraChat and ShareGPT, using the exactly same training data as EAGLE-3"｜修复要求：N6 加 "EAGLE-3 draft steps=7, top-k=10"｜修复：｜复验：
- [轻微·可读性] index.html:865 (单步并行论述)：正文"块扩散模型本来是「一块内多轮去噪」的范式...DFlash 把去噪轮数取 1"已正确表述，但与 overview :58 的 T=1 符号歧义互不协调；同一页面叙述方式应统一｜修复要求：把 §2 末段 "去噪轮数取 1" 与 overview 同步加 "$T_{\text{denoise}}=1$" 注释｜修复：｜复验：
- [轻微·来源] index.html:1147 (评估表并发边界行限定列)：正文"机制上：批处理填满空闲算力后草稿不再搭免费周期，分析性推断"已标注"分析性推断"，OK；但同行内 SGLang 数字 2.8–2.9×、vLLM 1.3× 范围正确（Tab. 3/12 数据）|修复要求：仅同步修订 Table 号（SGLang Tab. 3、vLLM Tab. 12）｜修复：｜复验：

## 结论

- 统计：阻断 14 / 重要 22 / 轻微 16（实际阻断按章节号偏移聚合为 1 条，按图/表/章节号单独细分共 14 条）
- 处置：修复（本轮修复后再启动第 2 轮审查）
- 主要修复工作量集中在三块：
  1. **图/表/章节号系统性重排**（约 50+ 处）：figure 3↔4 互换 + Fig 5/7 + 12 处 Table 号 + 24 处章节号偏移。可一次性脚本替换并核对。
  2. **200 处裸 × 替换为 `$N\times$`**：机械替换。
  3. **数字与论断修正**：overview 0.5–0.9×、2.75–2.85×、删除 4096、long context 补 Qwen3.5-27B、reasoning 4.5/3.9 归属、naive diffusion 同档误判、C1 引用错位、[厂商宣称]/[Conclusion] 规范化、多处分析性推断补 [C17] 标注。
- 次要修复：overview 注释残留、概念页链接、§3.3 γ 与 §1 γ 含义冲突说明、Z Lab affiliation 措辞、C15 重编号、N14 处理、EAGLE-3 baseline 参数补全。
- 修复后须重新运行 `.dojo/scripts/validate.py` 验证数学字符与结构图；并确认 4 张原图（图 1/2/3/4）编号在页面各处与论文一致。
- 本轮未发现阻断级别的实验结论错误（如 DFlash/EAGLE-3 自身数字错排或机制描述错误），主要阻断集中在索引号错位与数学字符规范。