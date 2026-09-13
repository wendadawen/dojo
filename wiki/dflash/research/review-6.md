<!-- review-meta
round: 6
page: wiki/dflash/index.html
reviewed_content_sha256: 31615a03f928d890
-->
# DFlash 审查记录（第 6 轮）

- 页面版本：e7cf265b043c08685320f886a32d624bb3a22a80
- 论文版本：arXiv:2602.06036v2（2026-05-28 修订；TeX/HTML 固定版本）
- 审查时间：2026-09-13 21:10
- 审查者：独立子代理（编排者派发的独立审查者）
- 已完整阅读章节：核心问题（5 条解答折叠块）→ 1. 起草为什么慢——串行成本与浅层天花板（含本章问题、构造示例）→ 2. 推理管线——KV 注入与单步并行起草（含补充折叠块、本章问题、自绘流程 figure）→ 3. 训练——让 draft 的课堂就是考场（3.1–3.5、loss-decay 权重表折叠块、本章问题）→ 4. 实验——三个后端上的数字与条件（4.1–4.6、本章问题）→ 5. 方法评价——收益边界与适用判断（评价表、本章问题）→ 来源与范围说明（含 C/F/N 编号表、构造示例、简化条件）。所有折叠块与 4 张图注均已展开阅读。
- 核对方式：WebFetch + 直取 arXiv HTML（https://arxiv.org/html/2602.06036v2）逐表逐式核对；Figure 1/2/3/4 读原图核验；Inco AI 官方博客（2026-08-18）核验生态与 DFlash 2 描述；`overview.html` 与前置概念页交叉核对。

## 问题

- [重要·技术] `4.1 主实验`表末列「DFlash 对 EAGLE-3(16) 提升」＋该节正文，与 `overview.html`「关键结论」：同一量（T=0 相对 EAGLE-3(16) 的提升）在两份互链文件里数字不一致。index 表 T=0 行为 Qwen3-4B 2.7$\times$、Qwen3-8B 2.8$\times$，正文概括为「2.4–2.8$\times$」；overview 把 T=0 的提升写作 2.4$\times$。index 的提升列是该页自算的平均加速比之比（4.91/1.81=2.71、4.86/1.76=2.76），而论文 §5.1 正文自述 T=0 为 2.4$\times$、T=1 为 2.2$\times$，index 的「2.4–2.8$\times$」既不含 2.2，也未说明该列与论文自述值的关系，读者对照两页/论文会得到三个不同数。｜引文依据：Table 1 Q3-4B T=0 DFlash(16) 4.91$\times$/τ 6.54、EAGLE-3(16) 1.81$\times$/τ 3.05；Q3-8B T=0 DFlash 4.86$\times$/τ 6.49、EAGLE-3(16) 1.76$\times$/τ 2.96；§5.1 正文 "Under greedy decoding (temperature = 0), DFlash achieves an average speedup of 4.9× over the autoregressive baseline, corresponding to a 2.4× improvement over EAGLE-3 (16). Under non-greedy sampling (temperature = 1), DFlash maintains a 4.1× speedup over baseline and a 2.2× improvement over EAGLE-3."｜修复要求：统一 index 表/正文与 overview 的提升口径——或把 overview 的 T=0 提升改为与本页 T=0 行一致（2.7$\times$/2.8$\times$），或两处同时给出论文 §5.1 自述值（T=0 2.4$\times$、T=1 2.2$\times$），并在表头/正文注明该列是「本表平均加速比之比」还是「论文自述值」。｜修复：｜复验：
- [轻微·技术] `4.5 LLaMA-3.1-8B` 段末：写「EAGLE-3(60) … 在并发 16/32 出现 0.6–0.9$\times$」，漏掉 Alpaca 并发 32 的 0.5$\times$；同页 §5（评价表「相对 EAGLE-3 的真实优势」行写 0.5–0.9$\times$）与 `overview.html`（0.5–0.9$\times$）均不同。｜引文依据：Table 5 EAGLE-3(60) 行：GSM8K conc16 0.9$\times$ / conc32 0.6$\times$；HumanEval 0.9$\times$ / 0.6$\times$；Alpaca 0.8$\times$ / 0.5$\times$（最低 0.5$\times$ 出现在 Alpaca conc32）。｜修复要求：把 §4.5 的「0.6–0.9$\times$」改为「0.5–0.9$\times$」，与 §5 及 overview 一致。｜修复：｜复验：
- [轻微·格式] `2. 推理管线` 公式说明列表与本节末段：同一矩阵 $W_c$ 的形状写法互为转置——说明列表写 $W_c\in\mathbb{R}^{D\times 5D}$，「KV 注入的工程开销」段又写「$5D\times D$ 维」。｜引文依据：A.3 "The only extra parameterized component is the shared projection $W_c\in\mathbb{R}^{D\times5D}$."（页面 §2 说明列表与 overview 亦作 $D\times5D$）。｜修复要求：把 §2 末段的「$5D\times D$ 维」改为「$D\times 5D$ 维（约 $5D^2$ 个参数）」，与 §2 公式、论文 A.3 一致。｜修复：｜复验：

## 本轮已核对通过的主要项（供下游参考）

- 公式与来源：F1=§3.1 Eq.(1)、F2=§3.2 Eq.(2)、F3=§3.2 Eq.(3)、F4=A.3 第一式、F5=A.3 第二式组、F6=§4.2 Eq.(4)，六处均逐式比对一致；τ∈[1,γ+1]、η=L_target/L 表述与原文一致。Eq.(4) 原文用 γ 兼作衰减率与草稿预算，本页改名 γ_w 并显式说明「与 F1/F2 的 γ 不是同一个量」，属合理消歧，不判为符号冲突。
- 主表数字：Table 1 四组（Q3-4B/8B × T=0/T=1）的 DFlash、EAGLE-3(16)、EAGLE-3(60) 全部速度与 τ 逐格核对无误；提升列算术自洽（4.91/1.81=2.7、4.24/1.72=2.5、4.86/1.76=2.8、4.03/1.68=2.4）。
- Table 3（SGLang）：7 行 × 基线 tok/s、conc1/conc32、τ 全部一致；最高 5.1$\times$ 对应 §5.3 "up to a 5.1× speedup on Qwen3-8B"。Table 5（LLaMA）与 Table 8/9/10/11/12/13、Table 2（reasoning）、Table 4（long context）、Table 6（层数）逐格核对一致。
- Figure 1 逐柱读数与 Tab. 1 Q3-8B T=0 行完全一致，图中 EAGLE-3 柱确为 EAGLE-3(60)（GSM8K 2.23/Math500 2.05/AIME25 2.05/HumanEval 2.17/MBPP 1.93/LCB 1.81/MT-Bench 1.90），页面「图中 EAGLE-3 为树大小 60」判断正确；Figure 3 读数与「5 层 DFlash 起草 16 token 低于 EAGLE-3 单层起草 8 token」一致；Figure 2/4 图例（蓝=Fused Target Context Feature、绿=Mask Token、橙=Clean/Target Decode Token、白=Invisible）与页面图注一致。
- A.1 训练配置（800K 样本 Nemotron Post-Training V2 + CodeAlpaca、response 由 target 生成、6 epochs、AdamW、lr 6e-4、clip 1.0、cosine/warmup 0.04、序列 3072/4096、每序列 512 anchor、γ_w=7/5/4、online/offline）逐项一致；A.3「only extra parameterized component… 5×2048×2048×2≈42 MB」「projection input and output require about 40 MB and 8 MB… below 400 KB」「They bypass the draft model's Q projection, output projection, self-attention update, and FFN」逐句一致；§4.2「training-time test」引文属实。
- 构造示例可复算：L=(3+10)/4=3.25 ms、η=10/3.25≈3.08$\times$；loss-decay 权重表 $w_k=\exp(-(k-1)/7)$ 与 16 位数值（含 w_1=1.000、w_16=0.117、比值≈8.5）逐格复算一致。
- 表述维度：全文（含折叠块与图注）未出现「本页」「下面来看」「需要注意的是」「我们/你」，无「场景」当术语、无调试叙事与临场评价；`<meta name="dojo:summary">`、正文、`overview.html` 与 `assets/` 图片一致；alt 属性无 `$…$`；无「（待生成）」；`../speculative-decoding/`、`../standard-attention/`、`../block-diffusion/`、`../eagle-speculative/`、`../dflash2/` 五个被引页均真实存在。
- 生态与 DFlash 2 描述与 Inco AI 博客（2026-08-18）一致：集成 SGLang/vLLM/TensorRT-LLM/llama.cpp、NVIDIA Blackwell 最高 15$\times$、Google TPU 3$\times$、3.5M+ 下载、路径选择器 + 两抽头动态卷积修复块内连贯性缺口。
- `python3 .dojo/scripts/validate.py wiki/dflash/index.html` → validation ok。

## 结论

- 处置：修复
- 统计：阻断 0 / 重要 1 / 轻微 2