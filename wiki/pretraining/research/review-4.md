<!-- review-meta
round: 4
page: wiki/pretraining/index.html
reviewed_content_sha256: 5550061edcac30b9
-->
# 语言模型预训练审查记录（第 4 轮）

- 页面版本：index.html `470e615cd25c66f7a8382714227b96bc162287f2`（工作树）；overview.html `1beed7d2d2c597d8fc9007d8f5c1136e59647c14`
- 审查时间：2026-09-14 17:09
- 审查者：独立子代理（未参与写作与前序轮次审查）
- 页面类型：`dojo:type=concept`，适用 `guides/concept/check.md` 与 `guides/concept/style-guide.md`
- 已完整阅读章节：核心问题（页面级 4 题及解答）→ 1. 语言模型是什么（含本章问题）→ 2. 一条文本的概率如何逐 token 分解（含「展开：为什么联合概率能写成条件概率连乘」折叠块、本章问题）→ 3. 预训练的目标——在大语料上最小化下一 token 交叉熵（含本章问题）→ 4. 预训练产出基座模型，行为对齐交给后训练（含本章问题）→ 来源与范围说明（论断与来源 C、公式与来源 F、构造示例、辅助解释与类比边界、简化条件及其限制）→ 页面脚本与资源引用。另读 overview.html 作交叉核对。

## 来源核对证据（逐条回源）

| 编号 | 页面表述 | 回源位置 | 原文片段 / 关键数值 | 判定 |
|---|---|---|---|---|
| C1 | 语言模型是词/token/字符序列上的概率分布；常表述为 next token prediction | LLaMA arXiv:2302.13971 §7 Related work（ar5iv 全文核对） | "Language models are probability distributions over sequences of words, tokens or characters (Shannon 1948; Shannon 1951)." 与紧随其后的 "This task, often framed as next token prediction, has long been considered a core problem in natural language processing" | 一致 |
| C2 | 式(1) 链式分解「使 p(x) 的采样与估计可行」 | GPT-2 论文（OpenAI 2019）§2 Approach 式(1)，PDF 文本提取核对 | "p(x) = ∏_{i=1}^{n} p(s_n｜s_1,...,s_{n−1})  (1)"；紧接 "This approach allows for tractable sampling from and estimation of p(x)" | 一致（页面把 §2 式(1) 中原文的笔误下标 n 改为 i，属教材化更正，未改变含义） |
| C3 | 语言建模结果以「每个预测单元的平均负对数概率」的缩放/指数化形式报告 | GPT-2 §3.1 Language Modeling | "Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word." | 一致 |
| C4 | 后训练从 SFT 开始：标注者在输入分布上提供示范，再用监督学习微调预训练 GPT-3 | InstructGPT arXiv:2203.02155 §3.1 Step 1（ar5iv 全文核对） | "Our labelers provide demonstrations of the desired behavior on the input prompt distribution" / "We then fine-tune a pretrained GPT-3 model on this data using supervised learning." | 一致 |
| C5 | 预训练目标（最小化上下文词预测误差）与用户目标（遵循指令、有帮助且安全）错位 | Zhang et al. arXiv:2308.10792 §1（ar5iv 全文核对） | "LLMs are typically trained on minimizing the contextual word prediction error on large corpora" / "while users want the model to 'follow their instructions helpfully and safely'" | 一致 |
| C6 | 最小化 NLL 等价于最小化交叉熵；任何负对数似然损失都是交叉熵 | Goodfellow et al., Deep Learning §5.5（MLE 小节，原文核对） | "Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by the model." | 一致（页面省略了「经验分布与模型分布之间」的限定语，属可接受的紧缩，未产生反向结论） |
| C7 | GPT-2 词表 50,257 | GPT-2 §2.3 Model | "The vocabulary is expanded to 50,257." | 一致（数值精确，千位分隔写作 `$50{,}257$` 无误） |
| F1/F2 | 链式分解式；预训练目标 $\mathcal{L}(\theta)=-\mathbb{E}_{\text{语料}}[\log p_\theta(s_i\mid s_{<i})]$ | F1 = C2 出处；F2 = F1+C3+C6 组合 | 组合关系与页面自述一致，目标函数形式为标准下一 token 交叉熵，期望下标「语料」与正文「按语料中的位置取平均」自洽 | 一致 |

算式复算（Python 实测，与页面「构造示例」小节声明一致）：

- 表 1 分布合计 $0.80+0.10+0.05+0.04+0.01=1.00$ ✓
- $-\ln 0.25=1.3862944\to1.3863$、$-\ln 0.80=0.2231436\to0.2231$、$-\ln 0.60=0.5108256\to0.5108$ ✓
- 连乘 $0.25\times0.80\times0.60=0.12$ ✓；逐 token 损失之和 $=2.1202635\approx2.1203$，除以 3 $=0.7067545\approx0.7068$ ✓
- $-\ln 0.12=2.1202635\approx2.1203$ ✓；与「乘变加」结论一致

其他机械项：

- `.dojo/scripts/validate.py wiki/pretraining/index.html` → `validation ok`
- 链接有效性：`../../wiki/cross-entropy/index.html`、`../../wiki/sft/index.html` 均真实存在，无「（待生成）」占位；`../../libs/` 下 katex / auto-render / prism / 四份 CSS 均在位；index↔overview 互链成立
- KaTeX 实测（node + 本地 `libs/katex.min.js`）：`\mathbb{E}_{\text{语料}}[\log p_\theta(s_i\mid s_{<i})]`、`p(\text{天气真})=0.25\times0.80\times0.60=0.12`、`\prod_{i=1}^{n}p(s_i\mid s_1,\dots,s_{i-1})` 均渲染成功无报错；`dojo:summary` 内的 `$p(s_1,\dots,s_n)=\prod_i p(s_i\mid s_{<i})$` 同式可渲染
- 元数据：`description` 为纯文本、`dojo:type=concept`、`dojo:topics=训练与优化,数学基础`（均在 `catalog_builder.py` 的 `ALLOWED_TOPICS` 内）、`dojo:tag=训练`（在 `ALLOWED_TAGS` 内）
- 结构：无正文 `<img>`/SVG，无交互视图，故图示与降级可读性两项不适用；问题块命名、`展开：` summary 前缀、来源小节命名均符合 style-guide

## 问题

- [轻微·技术] §3 目标函数（第 208/216/228/235/282 行）：同一「负对数似然」在 §2 一律写作 $-\ln p$（并据此得出 $1.3863/0.2231/0.5108$），在 §3 一律写作 $\log p_\theta$，全页未说明 $\log$ 取自然对数，读者可能把两处读成不同底的对数函数。｜引文依据：页面 §2 第 175 行「$-\ln 0.25=1.3863$」属实（Python 实测 1.3862944）；§3 第 208 行 `\mathcal{L}(\theta)=-\,\mathbb{E}_{\text{语料}}[\log p_\theta(...)]`；GPT-2 §3.1 原文用 "average negative log probability"，未写底数。｜修复要求：在 §3 目标函数首次出现 $\log$ 处补一句「本文 $\log$ 指自然对数，与上一章的 $\ln$ 同义」，或将 §3 全章统一为 $\ln$，使同一函数全页写法一致。｜修复：｜复验：
- [轻微·技术] §4 本章问题解答（第 261 行）：「后续再用偏好优化继续调整」超出 C4 的界定范围（C4 明确限定为 InstructGPT §3.1 Step 1，该步只描述 SFT 的示范与监督微调），且该分句无引文编号。｜引文依据：来源章节 C4 原文「Ouyang et al., "InstructGPT", arXiv:2203.02155, §3.1 Step 1」；InstructGPT §3.1 Step 1 原文只到 "We then fine-tune a pretrained GPT-3 model on this data using supervised learning."，偏好优化在 §3.2/§3.3。｜修复要求：为该分句补来源编号（指向 InstructGPT §3.2/§3.3 的奖励建模与 PPO 步骤），或删去「后续再用偏好优化继续调整」，只保留 C4 覆盖的 SFT 部分。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布。本轮未发现事实性错误：C1–C7、F1/F2 七条论断与两处公式全部定位到来源并给出原文片段；构造示例的全部数值（$0.12$、$1.3863/0.2231/0.5108$、$2.1203$、$0.7068$、$-\ln 0.12$）经独立复算一致；表 1 分布合计为 1；正文、summary、overview、本章问题解答之间的数字与结论无冲突；`validate.py` 通过，公式与 summary 公式 KaTeX 实测可渲染，链接与元数据合规。表述维度通读（含折叠块、图注）未发现会话指代、调试叙事、临场评价或 AI 拼接腔；文中的「本文/本页」自称符合 style-guide §12，不作为问题记录。两条轻微问题不影响正确性与主线理解，为提升表达一致性可修，亦具明确接受理由（$\log=\ln$ 为机器学习通行约定；偏好优化为同一被引论文的后续步骤）。

统计：阻断 0 / 重要 0 / 轻微 2