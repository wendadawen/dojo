<!-- review-meta
round: 5
page: wiki/sft/index.html
reviewed_content_sha256: e075b7b7fcfd12b9
-->
# SFT（Supervised Fine-Tuning）审查记录（第 5 轮）

- 页面版本：75e7b634956f3c3c8ea3303f9aa1f6b26a69f6f7
- 审查时间：2026-09-14 17:12
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题（页面级）→ 1. 预训练目标与用户目标的错位（含本章问题与两条解答折叠块）→ 2. SFT 的数据与训练目标（含三条解答折叠块）→ 3. 手算一条 SFT 样本的损失（含「展开：本例五个位置的完整对数计算」折叠块与两条解答）→ 4. SFT 在后训练流程中的位置（含 HTML 结构图与两条解答）→ 5. 实践中的边界与误解（5.1–5.4、两组来源事实汇总表、两条解答）→ 来源与范围说明（论断与来源 C1–C8、公式与来源 F1、外部数字与实验条件 N1–N3、构造示例、辅助解释与类比边界、简化条件及其限制）

## 问题

- [阻断·技术] 第 2 章 F1 损失公式（wiki/sft/index.html:150）：该行 `$$...$$` 内写作 `y_{<t}`，`<` 未转义；本页其余 5 处（154、155、194、201、372 行）都写 `y_{&lt;t}`。HTML 解析器把 `<t}),$$` 当作文档标签起始，生成一个 tagName 为 `t}),$$`、属性名亦为 `<ul` 的未知元素，从其后的符号定义列表一直到页面结尾都被吞进该元素。后果有三：① F1 公式未被 KaTeX 渲染，读者看到的是原始文本 `$$\mathcal{L}(\theta) = -\frac{1}{|D|}\sum_{(x,y)\in D}...$$`；② 紧随的三个符号定义 `<li>`（D、y_t、p_θ）失去 `<ul>` 列表结构；③ 侧边目录、折叠按钮、j/k 快捷键均按 `body > h2` 取章节，而 3、4、5 章与「来源与范围说明」的 h2 已不在 body 直系，目录只剩 1、2 两章。｜引文依据：无头 Chrome 渲染本页后，`document.body` 直系子节点中存在 tagName 为 `T}),$$` 的元素（属性名 `<ul`），其包裹文本以「$D$：SFT 训练集…」开头；`document.querySelectorAll('body > h2').length === 2`，源文件一级 h2 共 6 个；渲染后公式处文本仍为未处理的 `$$...$$`，KaTeX 输出中 `katex-display` 仅 2 个（源文件 `$$` 公式共 3 个）。｜修复要求：把 150 行 `y_{<t}` 改为 `y_{&lt;t}`，与其余 5 处一致。｜修复：｜复验：

- [轻微·技术] 第 3 章「展开」折叠块（index.html:237）：「mask 版合计 $1.2040+0.1054=1.3093$」，按展示的 4 位小数操作数相加应为 1.3094；1.3093 是精确值 1.2039728+0.1053605=1.3093333 的四舍五入。同段「指令部分合计 $4.6052+2.9957+3.9120=11.5129$」的操作数恰好加得出所写和，两处呈现口径不一致（数值本身与结论无误）。｜引文依据：不适用（构造示例）。｜修复要求：改写为按精确值求和的表述（如 `$1.2040+0.1054\approx 1.3093$`，或注明合计按精确值四舍五入），使操作数与和自洽。｜修复：｜复验：

- [轻微·格式] 来源与范围说明 F1（index.html:372）：F1 是全文唯一的公式来源条目，但正文 150 行的 F1 公式没有对应的 `<sup>[F1]</sup>` 上标引用，不符合 style-guide §6「与来源章节双向对应」；C1–C8、N2、N3 均在正文有对应上标。｜引文依据：不适用。｜修复要求：在 150 行公式后补 `<sup>[F1]</sup>`。｜修复：｜复验：

## 来源核对（原文片段与数值）

- C1/C2/C8（Zhang et al., arXiv:2308.10792）：脚注 1「In this paper, unless specified otherwise, supervised fine-tuning (SFT) and instruction tuning (IT) are used interchangeably.」；摘要「…(instruction, output) pairs in a supervised fashion…」；§1「the mismatch between the training objective and users' objective」「LLMs are typically trained on minimizing the contextual word prediction error on large corpora」，用户侧「follow their instructions helpfully and safely」；§1 三项收益「bridges the gap between the next-word prediction objective of LLMs and the users' objective…」「SFT allows for a more controllable and predictable model behavior…」「SFT is computationally efficient and can help LLMs rapidly adapt to a specific domain…」。与页面表述一致。
- C3（Ouyang et al., arXiv:2203.02155）：§3.1「Steps 2 and 3 can be iterated continuously; more comparison data is collected on the current best policy」；§3.5「Starting from the SFT model with the final unembedding layer removed…」。与三步流程与 RM 起点表述一致。
- C4（Touvron et al., arXiv:2307.09288）：§3.1「We utilize an autoregressive objective and zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens.」；§3.3「we simply set the loss to 0 for all the tokens from the previous turns, including assistant messages.」。页面标注 §3.3 为「GAtt 多轮一致性训练的语境，非 SFT 数据的一般规则」，适用范围的收窄正确。
- C5（Llama 2 §3.1「Quality Is All You Need」）：「We found that SFT annotations in the order of tens of thousands was enough to achieve a high-quality result.」「We stopped annotating SFT after collecting a total of 27,540 annotations.」「By setting aside millions of examples from third-party datasets…using fewer but higher-quality examples from our own vendor-based annotation efforts…our results notably improved.」。与 5.2 一致。
- C6/N2（InstructGPT §3.5）：「We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2.」「our SFT models overfit on validation loss after 1 epoch」「training for more epochs helps both the RM score and human preference ratings, despite this overfitting.」。与 5.3 及表一致。
- C7（InstructGPT §1）：「outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data.」「This result holds true even when we add a few-shot prompt to GPT-3…」。与 5.4 及 1 章一致。
- N1（InstructGPT §3.2）：「The SFT dataset contains about 13k training prompts」。与表格「约 13k 训练 prompts」一致。
- N3（Llama 2 §3.1）：「a cosine learning rate schedule with an initial learning rate of 2×10−5」「a weight decay of 0.1, a batch size of 64, and a sequence length of 4096 tokens」「we fine-tune the model for 2 epochs」。与表格逐项一致。
- 构造示例复算（自然对数）：−ln0.01=4.6052、−ln0.05=2.9957、−ln0.02=3.9120、−ln0.30=1.2040、−ln0.90=0.1054；合计 12.8223、指令合计 11.5129、占比 11.5129/12.8223=89.8%、mask 平均 (1.2040+0.1054)/2=0.6547、不 mask 平均 12.8223/5=2.5645；e^{−4.6052}≈0.01、e^{−1.2040}≈0.30 互验通过。正文、折叠块、表格、overview 中出现的这些数字彼此一致（唯一差异见上「轻微·技术」）。
- 链接与结构：`../../wiki/cross-entropy/index.html`、`../../wiki/pretraining/index.html`、`overview.html`、`../../index.html` 均存在；index 与 overview 互相链接；overview 的数字（27,540 条、1 epoch 后过拟合 / 16 epochs、1.3B 偏好于 175B）与本页一致；`python3 .dojo/scripts/validate.py wiki/sft/index.html` 返回 `validation ok`（该校验为文本级，未捕获上述 HTML 解析缺陷）。
- 表述维度：逐段通读含折叠块与图注，未见第一人称复数、第二人称、调试叙事、临场评价或 AI 拼接腔；`dojo:summary`、description、alt（仅 lightbox 空 alt）无问题，无 Unicode 数学字符直接出现，标题/summary/正文/列表/表格数学符号均包在 `$...$` 中。页面出现的「本文说明…文章结构」「本页不展开 / 未展开 / 不在本页范围」为 style-guide §12 明确许可的自称用法与范围说明，不判为元话语。
- 图与交互：结构图为 HTML（`.dg-flow`）非等宽字符框线图，节点与箭头含义由节点标题与图注定义，无公式类标签；其可读性不依赖脚本。

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 2
- 处置：修复（第 150 行 `<` 转义为 `&lt;`，即可同时恢复 F1 公式渲染、符号列表结构、侧边目录与折叠/快捷键作用域；另两条轻微项按修复要求处理）
