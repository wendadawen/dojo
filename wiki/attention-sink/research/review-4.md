# 注意力汇聚点审查记录（第 4 轮）

- 页面版本：index.html 工作树 blob 哈希 `0b4b269f3ad45d0f0fe4ef9b5df7f885e44585e2`
- 审查时间：2026-09-10
- 审查者：独立子代理（未参与写作与前序轮次审查）
- 已完整阅读章节：核心问题（4 题及解答）、常见误解、1. 现象、2. 成因、3. 两种做法（3.1 形态一 / 3.2 形态二）、4. 对缓存与推理意味着什么（4.1 / 4.2 / 4.3）、来源与范围说明（论断与来源（C）/ 公式与来源（F）/ 外部数字与实验条件（N）/ 构造示例 / 辅助解释与类比边界 / 简化条件及其限制），及全部 `<details>` 折叠块；另完整阅读 overview.html。

## 机械验证结果

1. **validate.py**：`/usr/bin/python3 .dojo/scripts/validate.py wiki/attention-sink/index.html` 返回 `validation ok`（exit 0）。
2. **§4.2 代码实跑**：复制 `W = 128` 份额代码到 `/usr/bin/python3` 实跑，输出与页面「预期输出」逐字符一致：
   ```
   sink= 0.0: 份额 0.0078 = 0.78%
   sink= 2.0: 份额 0.0546 = 5.46%
   sink= 5.0: 份额 0.5369 = 53.69%
   sink= 8.0: 份额 0.9588 = 95.88%
   ```
3. **KaTeX 渲染**（Chrome headless `--dump-dom` 抓取渲染后 DOM）：index.html 共 83 个行内 `.katex` 节点 + 3 个 `katex-display`（对应源文件 3 处 `$$` 公式），`katex-error` 计数为 0，无红色错误片段，`\text{sink}`、`\text{share}` 均正常渲染为文字而非制表符；DOM 末尾完整闭合 `</body></html>`。overview.html 渲染后 `katex-error` 为 0。

## 问题

本轮 4 项重点复验均通过，未发现新的阻断 / 重要 / 轻微问题。逐项核对依据如下：

- **复验 1：[F1] 公式与内核一致性（通过）**。页面 §3.2 公式现在写作可见槽位也减 $m$，即分子分母均为 $e^{q\cdot k_t/\sqrt{d}-m}$，分母再加 $e^{\text{sink}-m}$。对照 `kernel.py` 稀疏注意力内核（L310–389）：L373 `acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])`（可见槽位，`acc_s` 在 L365 GEMM 后于 L367 乘 `scale=(1/d)^0.5`，故即 $q\cdot k_t/\sqrt{d}$，再减行最大值）；L383 `sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`（sink 项，同样减 `scores_max`）。两者一致：可见槽位与 sink 项共享同一个行最大值 $m$，且 $m$ 仅来自可见槽位（L369 `T.reduce_max(acc_s, scores_max, dim=1)` 只归约 `acc_s`，sink 在循环结束后 L383 才加入）。页面「$m$ 不含汇聚点项」「它与可见槽位项共享同一个 $m$」的表述与内核一致。
- **复验 2：[F2] 份额公式成立条件（通过）**。独立验算：当 $W$ 个可见槽位分数均为 0 时 $m=0$，每个可见槽位指数项 $e^{0-0}=1$，分母为 $W+e^{\text{sink}-0}=W+e^{\text{sink}}$，sink 项份额 $\text{share}=e^{\text{sink}}/(W+e^{\text{sink}})$，成立。页面 §4.2「简化条件」已说明分数同为 $s$ 时的形式：$e^{\text{sink}-s}/(W+e^{\text{sink}-s})$（此时 $m=s$，可见项 $e^{s-s}=1$，sink 项 $e^{\text{sink}-s}$，验算成立）。数值 $0.78\%/5.46\%/53.69\%/95.88\%$ 由公式复算一致。
- **复验 3：引用编号连续性（通过）**。全页引用集合为 [C1]–[C6]、[F1]–[F2]、[N1]–[N5]，无跳号，无 [C7] 残留；[C6] 出现在 §4.3 正文（`<sup>[C6]</sup>`）并在来源节有定义，C5 之后直接是 C6。
- **复验 4：KaTeX 渲染（通过）**。见机械验证结果第 3 条，`\text` 未被写坏成制表符，无渲染错误。

本轮全文阅读与来源核对结果（其余论断无变更，仅列关键依据）：

- [C1] §3.1 现象与换行符实验：arXiv 原文「beyond the bottom two layers, the model consistently focuses on the initial tokens across all layers and heads.」「a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task」；换行符实验见 Table 1（`4"\n"+1020` → 5.60）。与页面一致。
- [C2] softmax 归一化成因：原文「The nature of the SoftMax function (Equation 1) prevents all attended tokens from having zero values…the model tends to dump unnecessary attention values to specific tokens.」「removing these initial tokens' KV will remove a considerable portion of the denominator…」「initial tokens are visible to all subsequent tokens…more easily trained to serve as attention sinks」。与页面一致。
- [C3] §3.2 保留 4 个初始位置与位置口径：原文「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV…」「Introducing four initial tokens generally suffices; further additions have diminishing returns.」「StreamingLLM focuses on positions within the cache rather than those in the original text.」。与页面一致。
- [C4] DeepSeek 实现：`model.py` L639 `self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))`；`kernel.py` 分母构造如复验 1。与页面一致。
- [C5] 解析解与边界：`verify_sink.out` 显示「汇聚点项只进分母」各 sink 取值（-inf/0/2/5）下输出与解析解逐元素差 0.00e+00；「整行 -1 的输出 = [0.0, 0.0, 0.0, 0.0]」（L929 边界描述成立，内核 L355 `T.fill(scores_max, -1e30)` 用极小有限值使 sink 项 exp 溢出为 inf、分子为 0，输出 0/∞=0）。与页面一致。
- [C6] 附录 A 边界：原文「While StreamingLLM improves the efficiency of LLMs in streaming contexts, it does not extend the models' context window or enhance their long-term memory capabilities.」。与页面一致。
- [N1] Table 1（Llama-2-13B，PG19 首本书 65K）：「0+1024」5158.07、「4+1020」5.40、「4"\n"+1020」5.60；符号「Cache config x+y denotes adding x initial tokens with y recent tokens.」。与页面一致。
- [N2] Table 2：Falcon-7B / MPT-7B / Pythia-12B / Llama-2-7B（400K 拼接 token），结论「Introducing four initial tokens generally suffices; further additions have diminishing returns.」。与页面一致。
- [N3] Table 3 与 §4.2：可学习 sink token 模型 1+1023 困惑度 18.01，未做该训练的（vanilla）模型 2+1022 为 18.05（160M 参数、Pythia-160M 配方、PG19 首样本）。与页面一致。
- [N4] 真实 checkpoint：`headers.json` 直接解析得 `attn_sink` 张量 43 个（主干层 0–39 连续 + MTP 层 0–2），形状集合 `{(64,)}`，dtype `{'F32'}`。与页面「40 个主干层与 3 个 MTP 层、形状 [64]、fp32」一致。
- [N5] 份额数值：0.78% / 5.46% / 53.69% / 95.88%，由公式复算一致（见机械验证第 2 条）。

页面级「核心问题」4 题与第 1–4 章「本章问题」均配 `解答：` 折叠块，答案独立可读并指明论证章节；overview.html 与 index.html 相互链接；前置概念链接 standard-attention / kv-cache / sliding-window-attention 均存在。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布

## 发布结论

- 审查轮次：第 1 轮（0/2/10）→ 第 2 轮（0/4/3）→ 第 3 轮（0/1/3）→ 第 4 轮（0/0/0，可发布）
- 每轮均由未参与写作、未参与前序审查的独立审查者执行
- 阻断与重要问题全部关闭；遗留轻微问题已逐条修复
- `.dojo/scripts/validate.py` 返回成功；headless Chrome 渲染实测通过（KaTeX 正常、无占位符、无标签重叠、折叠块数与问题数匹配）
- `overview.html` 与 `index.html` 相互链接；前置概念页均存在
- 结论：**可发布**（注意力汇聚点）
