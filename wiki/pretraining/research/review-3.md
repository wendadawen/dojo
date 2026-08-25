# 语言模型预训练审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 99e14710bb98314425287a8423549d3b0be14d73
- 审查时间：2026-08-25 16:23 CST
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节（按顺序）：index.html 全文——标题与主要依据、导语、核心问题（4 题）、1 语言模型是什么、2 一条文本的概率如何逐 token 分解（含「为什么联合概率能写成条件概率连乘」折叠块）、3 预训练的目标——在大语料上最小化下一 token 交叉熵、4 预训练产出基座模型，行为对齐交给后训练、来源与范围说明（论断与来源 C / 公式与来源 F / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）；overview.html 全文——它是什么 / 为什么需要它 / 核心机制 / 关键结论与边界。所有问题解答折叠块均已展开阅读。

## 来源论断核对（引文依据）

每条均打开来源定位到页面标注位置并摘录原文：

- C1（语言模型是序列上的概率分布，常表述为 next token prediction）：LLaMA §7 Related work「Language models」小节原文："Language models are probability distributions over sequences of words, tokens or characters (Shannon 1948; Shannon 1951). This task, often framed as next token prediction, has long been considered a core problem in natural language processing"。与页面 C1 引文一致。✓
- C2 / F1（链式分解及「使采样与估计可行」）：GPT-2 §2 Approach 式 (1) 原文：p(x) = ∏_{i=1}^{n} p(s_n | s_1, …, s_{n−1})，其后紧接 "This approach allows for tractable sampling from and estimation of p(x) as well as any conditionals of the form p(s_{n−k}, …, s_n | s_1, …, s_{n−k−1})"。式 (1) 编号与引文均定位属实（本地提取文本 /tmp/gpt2.txt 核对）。✓
- C3（结果以每预测单元平均负对数概率的缩放/指数化形式报告）：GPT-2 §3.1 原文："Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word." ✓
- C4（后训练从 SFT 开始：标注者示范 + 监督学习微调预训练模型）：InstructGPT §3.1 High-level methodology Step 1 原文："Step 1: Collect demonstration data, and train a supervised policy. Our labelers provide demonstrations of the desired behavior on the input prompt distribution … We then fine-tune a pretrained GPT-3 model on this data using supervised learning." 页面中文引述忠实。✓
- C5（预训练目标与用户目标错位）：指令微调综述 §1 原文："One of the major issues with LLMs is the mismatch between the training objective and users' objective: LLMs are typically trained on minimizing the contextual word prediction error on large corpora; while users want the model to 'follow their instructions helpfully and safely' …"。页面表述「语言模型通常被训练为在大语料上最小化上下文词预测误差，而用户想要的是模型『遵循指令、有帮助且安全』」一致。✓
- C6（最小化 NLL 等价于最小化交叉熵）：Goodfellow et al., Deep Learning §5.5（式 5.60–5.61 之后）原文："Minimizing this KL divergence corresponds exactly to minimizing the cross-entropy between the distributions … Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model." ✓
- C7（GPT-2 词表 50,257）：GPT-2 §2.3 Model 原文："The vocabulary is expanded to 50,257." ✓
- F2（预训练目标公式）：F1（链式分解）+ C3（按预测单元平均）+ C6（交叉熵等价）的组合，各环节均有上述原文支持，公式内部自洽。✓
- 辅助论断「实际系统的词表规模以万计」：GPT-2 §2.2 "…compared to the 32,000 to 64,000 token vocabularies often used with BPE" 与 §2.3 的 50,257 共同支持。✓
- 辅助论断「广泛的续写与语言能力」（第 4 章及 overview 对基座能力的概述性陈述）：综述 §1 "LLMs such as GPT-3, PaLM, and LLaMA have demonstrated impressive capabilities across a wide range of natural language tasks…" 支持，未扩大范围。✓

## 数值复算（Python）

- 逐 token 损失：−ln 0.25 = 1.3863，−ln 0.80 = 0.2231，−ln 0.60 = 0.5108 ✓
- 总和 2.1203，平均（÷3）0.7068 ✓
- 连乘 0.25 × 0.80 × 0.60 = 0.12 ✓；−ln 0.12 ≈ 2.1203，与逐 token 损失之和一致（「乘变加」成立）✓
- 五 token 分布表合计 0.80+0.10+0.05+0.04+0.01 = 1.00 ✓

## 机械项检查

- validate.py：`python3 .dojo/scripts/validate.py wiki/pretraining/index.html` → "validation ok"，exit 0。dojo:topics 取值「训练与优化,数学基础」被词表接受。✓
- head meta：description（纯文本）、dojo:summary（含 KaTeX 可渲染公式）、dojo:type=concept、dojo:topics、dojo:tag 均存在。注：overview.html 无上述 meta，判断发布条件该条仅约束概念主页 index.html（validate.py 仅接受并验证 index.html，且 dojo:type=concept 语义指向概念主页）。✓
- 链接存在性：../../index.html、overview.html、../../wiki/cross-entropy/index.html、../../wiki/sft/index.html 目标均存在；overview.html 与 index.html 相互链接。✓
- Unicode 数学字符：正文中未检出（唯一「·」出现在脚本内中文间隔号，非数学符号）；同一变量（θ、s_i、s_{<i}）写法全文一致。✓
- 图示：页面无结构图，亦无等宽字符框线图，条款不适用。✓
- 可运行代码：页面无可运行代码，条款不适用。✓
- 本地资源：katex.min.css/js、auto-render.min.js、prism 系列、prism-primer-*.css 均存在于 libs/。✓

## 问题

- [轻微·可读性] index.html 导语（「把这件事做到极致」句）：「给任何前文都输出整个词表上下一个 token 的概率分布」中「词表上下一个 token」为「词表上」+「下一个 token」的字面粘连，「上下」连读易被误读为「上下文」相关表述；同页 1 章同义句「输出下一个 token 在词表上的概率分布」语序无歧义。｜引文依据：不适用｜修复要求：将该句语序改为与 1 章一致，如「给任何前文都输出下一个 token 在整个词表上的概率分布」。｜已修复：导语改为「输出下一个 token 在整个词表上的概率分布」。｜复验：validate.py 通过，修复处复查一致。

## 发布条件核对（规范第 5 节）

1. 三轮审查均已完成且每轮独立执行：✗ —— research/ 下仅有 review-1.md，review-2.md 不存在，第 2 轮审查记录缺失（第 1 轮记录与本轮记录存在）。
2. 每条来源论断都有引文依据记录，无法核对的已删除或降级：✓ —— C1–C7、F1–F2 全部核对到原文位置并记录于本文件，无「未能核对」项。
3. 所有阻断和重要问题均已关闭：✓（就本轮而言）—— 本轮未发现阻断/重要问题；validate.py 通过且本轮全文重审未见遗留的核心结论错误或明显误解。
4. 遗留轻微问题具有明确的接受理由：△ —— 1 个轻微问题（导语「词表上下一个 token」连读歧义）待修复或给出接受理由。
5. 全部学习目标由正文章节完整回答：✓ —— 核心问题 4 题分别由第 1–4 章正文完整回答。
6. 核心问题与本章问题均有解答折叠块：✓ —— 页面级 4 题、章节级 2+2+2+2 题均有解答折叠块，答案独立可读且与正文一致，核心问题答案指明论证所在章节。
7. 数学符号全部 LaTeX 书写、结构图为 HTML 或内联 SVG：✓ —— 正文无 Unicode 数学字符；无结构图（亦无 ASCII 框线图）。
8. validate.py 返回成功：✓。
9. 可运行代码结果与页面描述一致：不适用 —— 页面无可运行代码。
10. 关键论断和数字已重新核对来源：✓ —— 见「来源论断核对」「数值复算」。
11. head meta 齐全且 topics 在固定大类内：✓ —— 见「机械项检查」。
12. overview.html 与 index.html 相互链接：✓。
13. 页面引用的概念链接有效或具有明确占位：✓ —— cross-entropy、sft 页面均存在。
14. 递归生成的前置概念页已完成各自质检：✓ —— wiki/cross-entropy/research/ 与 wiki/sft/research/ 均含 review-1/2/3.md。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：暂缓发布。页面内容层面达到发布质量（无阻断/重要问题，仅 1 个轻微问题建议顺手修复）；但发布条件第 1 条不满足——research/review-2.md 缺失，第 2 轮审查记录不存在。需由编排者补做第 2 轮独立全量审查（或确认第 2 轮记录去向并补录），并关闭上述轻微问题后，方可发布。

（发布结论：暂缓发布——待补第 2 轮审查记录、修复 1 处轻微问题后，若无新增阻断/重要问题，可直接发布，无需第 4 轮全量审查。）


## 发布结论

- 发布时间：2026-08-25
- 三轮独立审查（每轮独立子代理）完成，全部阻断与重要问题已修复关闭；本轮轻微问题（导语语病）已修复。
- 审查者报告的「review-2.md 缺失」经编排者核实为误判：文件存在于 research/ 目录（第 2 轮独立审查记录及其修复结果完整）。
- 修复后 `.dojo/scripts/validate.py` 通过；headless Chrome 渲染实测 KaTeX 节点正常、0 错误。
- 结论：可发布。
