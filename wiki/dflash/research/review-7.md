<!-- review-meta
round: 7
page: wiki/dflash/index.html
reviewed_content_sha256: f0c64655a72f1f57
-->
# DFlash 审查记录（第 7 轮）

- 页面版本：c8eaaf888a6e6db28a12dff324879235dc87c08c（wiki/dflash/index.html 工作树哈希）
- 论文版本：arXiv:2602.06036v2（28 May 2026 修订；ICML 2026 camera-ready，页面标注一致）
- 审查时间：2026-09-13 21:49
- 审查者：独立子代理（未参与写作、未读取 research/ 下任何文件）
- 已完整阅读章节：核心问题（5 条折叠块）→ 1. 起草为什么慢（含 Figure 3、构造示例、本章问题）→ 2. 推理管线（含 Figure 2、自绘流程图、补充折叠块、本章问题）→ 3. 训练 3.1–3.5（含 Figure 4、权重表折叠块、本章问题）→ 4. 实验 4.1–4.6（含 Figure 1 与 6 张表、本章问题）→ 5. 方法评价（含三张本章问题折叠块）→ 来源与范围说明（全部小节）→ 页脚脚本与 alt 属性

## 本轮回源核对摘要（关键数值，均与 arXiv:2602.06036v2 对齐）

- Tab. 1 Qwen3-4B T=0：DFlash 4.91×/τ 6.54（MATH-500 6.09×、MT-Bench 2.85×），EAGLE-3(16) 1.81×/3.05，EAGLE-3(60) 2.08×/3.48；T=1：DFlash 4.24×/5.69、EAGLE-3(16) 1.72×、EAGLE-3(60) 1.93×；Qwen3-8B T=0 4.86×/6.49 / 1.76× / 2.02×，T=1 4.03×/5.48 / 1.68× / 1.88×；MT-Bench T=1 = 2.67×（4B）/2.47×（8B）。页面 4.1 表与「最高 6.09×」「T=0 2.75–2.85×、T=1 2.47–2.67×」「对 EAGLE-3(16) 高 2.4–2.8×」全部一致（折算列 4.91/1.81=2.7、4.24/1.72=2.5、4.86/1.76=2.8、4.03/1.68=2.4，逐行复算无误）。
- §5.1 正文自述「a 2.4× improvement over EAGLE-3 (16)」「2.2×」（T=1）、§5.2「roughly 4.5× and 3.9×」、§1「approximately 2−3×」「6.1× speedup on Qwen3-8B」「nearly 2.5× faster than…EAGLE-3」——页面转述与标注 [C1][N3] 一致。
- Tab. 3（SGLang）：Qwen3-8B Math500 基线 230 tok/s、并发 1/32 = 5.1×/2.8×、τ 8.01；Coder-30B-A3B HumanEval 3.5×/3.1×、τ 8.09——页面 4.3 表一致。
- Tab. 12（vLLM Qwen3.5-9B）1/8/16/32 = 4.0-4.6/3.0、3.2/3.4/2.2、2.5/2.7/1.7、1.9/2.1/1.3——页面 4.4 表逐格一致；Tab. 11 并发 8（3.5/3.4/2.5）与页面 4.4 末段对照数值一致。
- Tab. 5（LLaMA-3.1-8B）：GSM8K DFlash 2.4/2.2/2.1/1.8/1.6 τ4.32、HumanEval 2.8/2.6/2.5/2.1/1.8 τ4.91、EAGLE-3(60) 0.6×（并发 32）——页面 4.5 表逐格一致。
- Tab. 6（层数 4.69/4.71/4.64×、τ 5.64/5.99/6.33）、Tab. 8（b16→b16 τ6.33、b16→b8 5.09、b8→b8 5.21、b8→b16 5.02）、Tab. 9（KV 注入 vs 输入融合四行 8 个 τ、6 个加速比）、Tab. 10（无 target 特征 2.83/3.73/3.43/3.35×）、Tab. 13（Math500 5.64 vs 4.94、HumanEval 4.61 vs 3.86、MT-Bench 3.18 vs 2.80）——页面 4.6 与 2 章末段逐项一致。
- Tab. 4：hotpotqa Base 4K 4.91→16K 3.61、Long 16K 6.05；A.1（AdamW、lr 6e-4、warmup 0.04、6 epochs、序列 3072/Coder 4096、512 anchors、γ_w=7/5/4）；A.3（Qwen3.5-35B-A3B、D=2048、BF16、5×2048×2048×2≈42 MB vs ~70 GB；40 MB/8 MB、<400 KB）；§3.1 Eq.(1)「Following Sadhukhan et al. (2025)」=MagicDec、§3.2 Eq.(2)(3)、§4.2 Eq.(4) w_k=exp(−(k−1)/γ)——页面引用一致。
- 逐项复算：loss-decay 权重表 16 个值（γ_w=7）与 exp(−(k−1)/7) 一致，「位置 1 约为位置 16 的 8.5 倍」；构造示例 (3+10)/4=3.25 ms、10/3.25=3.08×——算式与结论相符。
- 直读页面资产核对图：img-01（Qwen3-8B T=0 柱值 5.15/6.08/5.62/5.14/4.65/5.51/2.75，EAGLE-3 1.81–2.23，均值 2.02 对上 Tab.1 EAGLE-3(60) 行，与图注「树大小 60」自洽）、img-04（横轴 4/8/16，EAGLE-3 单层 ~6→~26 ms，DFlash 1/3/5 层基本平直）、img-02（图例 Target Context Feature/Mask Token/Clean Token/Invisible Token 与页面图注一致）、img-03。
- 代码与功能：validate.py 通过（shell 完整性、重复 id、同页锚点、本地引用、数学字符与结构图检查）；`.dojo/scripts/validate.py wiki/dflash/index.html` → validation ok。页面引用概念页 speculative-decoding / standard-attention / block-diffusion / eagle-speculative / dflash2 均真实存在；块扩散页第 2 章确含「每个块至少要过两次模型」「一次前向并行产出所有块」两处被引内容，第 4 章为「块内去噪」。alt 属性无 `$...$`；交互内容（折叠块、自绘 HTML 流程图）无脚本时仍可读。dojo:type=paper、dojo:topics=训练与优化、dojo:tag=推理加速 均在 AGENTS.md/catalog_builder 词表内。

## 问题

- [重要·技术] §3.1 anchor 采样构块 第 2 段：「如果训练时把序列均匀切块，每块的第一个 token 也是前一块的最后一 token」｜问题：该句把标准构块写成块间共享/重叠 token，与论文对 standard 的描述不符，也与本节首段自述、本页 Figure 4 的块布局自相矛盾；块首 token 在标准构块下并不保证是干净 token，也不等于前一块末 token，读者会因此对 Tab. 13 所对比的 baseline 形成错误机制模型。｜引文依据：§4.2「in normal block-diffusion training, the response is uniformly divided into blocks and random positions within each block are masked」；本节首段（页面第 271 行）「标准块扩散训练把所有块均匀切好、每块内随机遮一些位置预测剩余」；本页 Figure 4 资产布局为「r1 <m> <m> <m> r2 <m> <m> <m> r3 …」，块首 r2 与第一块末位置（mask）不是同一 token。｜修复要求：删去「每块的第一个 token 也是前一块的最后一 token」，按论文口径改写为「标准构块下块内被遮位置随机、块首不保证是干净 token，因此训练条件与推理时『以模型产出的干净 token 为条件』不匹配」；若本意是「块首 token 紧跟前一块最后一 token」，须写成「紧跟前一块最后一 token 的下一个 token」，避免被读成块间重叠。｜修复：｜复验：
- [轻微·技术] 来源与范围说明「核心论断与来源」（第 545 行）与「外部数字与实验条件」（第 564 行）：C11、N14 只在本表出现，正文从未引用这两个编号（§2「KV 注入的工程开销」段只标 [N12]，§4.4 提到 Tab. 11 处未标 [N14]）；而本页首段称「核心论断 C1–C16 编号见正文与下表」，对 C11 不成立。｜引文依据：A.3「The only extra parameterized component is the shared projection … 5×2048×2048×2 ≈ 42 MB」（C11 与 N12 同源）；A.4 Tab. 11。｜修复要求：在 §2 KV 注入开销段与 §4.4 引用 Tab. 11 处分别补 [C11]、[N14]，或删除这两个条目并改写首段措辞。｜修复：｜复验：
- [轻微·表述] §5 方法评价「训练效率」行（第 488 行）：「训练无 test-time test 开销」术语有误，论文术语为 training-time test，同行引文本身也写作 training-time test，两处并置易生混淆。｜引文依据：§4.2「Training speculative draft models on long contexts is challenging for methods such as EAGLE-3 … training-time test」。｜修复要求：改为「训练无 training-time test 开销」。｜修复：｜复验：
- [轻微·表述] 前言（第 72 行）「…在对应链接页已讲过，本文只引用其结论。」——以「本文」为主语的页面自我指代/元话语；同类还有第 523 行「本文固定版本实验环境」（措辞含糊）。｜引文依据：不适用。｜修复要求：第 72 行改为不指代页面自身的写法，如「…——结论已在对应链接页给出，此处直接引用」；第 523 行改为「本文所依据的论文版本（v2）实验环境为 SGLang B200 + FA4 + Spec-v2」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（核心事实、数字、公式与图表编号均已逐条回源核对通过；关闭上述 1 重要 + 3 轻微后即可发布）
