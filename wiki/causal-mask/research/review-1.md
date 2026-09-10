# 因果掩码审查记录（第 1 轮）

- 页面版本：c707be8e87a8f604cb7b6f138f5d9e82cb7d6103（`git hash-object wiki/causal-mask/index.html`）
- 审查时间：2026-09-10 19:03
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节（index.html 从头到尾，含全部折叠块）：
  1. h1 因果掩码（Causal Mask）：让 decoder 只看过去的遮挡规则；reading-time；blockquote.meta；引言蓝框；「核心问题」（5 题，含解答折叠块）；「常见误解」；引言衔接两段
  2. 1. 因果掩码是什么规则，解决什么问题——decoder 必须挡住未来（含 encoder/decoder 对照表、本章问题 2 题及解答）
  3. 2. 因果掩码如何机械地实现——上三角 $-\infty$ 与 softmax 归零（含 $M$ 表、本章问题 3 题及解答）
  4. 3. 手算 $3$-token 例子——掩码、softmax 归零、重归一化（含「展开：位置 2 与位置 3 的完整 softmax 与输出」折叠块、「代码：3-token 因果掩码 softmax 的可运行验证」折叠块及预期输出、本章问题 2 题及解答）
  5. 4. 训练时并行、推理时 KV-cache 隐含（含 6-token $M$ 表、本章问题 2 题及解答）
  6. 5. 边界与 NoPE 的结构前提——打破排列对称性（含 en/decoder 与填充掩码对照表、本章问题 3 题及解答）
  7. 来源与范围说明：论断与来源（C）／公式与来源（F）／构造示例／辅助解释与类比边界／简化条件及其限制
  8. overview.html 全文（含与 index.html 互链）

## 机械验证结果

```
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/causal-mask/index.html
validation ok: wiki/causal-mask/index.html
EXIT=0
```

- 上标集合差集：正文引用的 sup 令牌 = {C1–C7, F1, F2}；来源章节定义的 sup 令牌 = {C1–C7, F1, F2}。**定义未引用 = 空；引用未定义 = 空**。
- 相邻双上标：`<sup>[X][Y]</sup>` 与 `</sup><sup>` 两种模式匹配数均为 **0**（组合引用写作 `<sup>[C1, C4]</sup>` 形式，符合规范）。
- 来源章节 h3 命名：`论断与来源（C）`、`公式与来源（F）`、`构造示例`、`辅助解释与类比边界`、`简化条件及其限制`，全部为固定命名；无内容的小节 `外部数字与实验条件（N）` 未保留（符合 style-guide §6）。
- 正文 h3 命名：5 个 `本章问题`，均为固定命名。
- 占位符扫描：`【` / `TODO` / `TBD` / `XXX` / `待生成` / `占位` / `FIXME` 命中数均为 **0**。
- 问题块配对：6 个 `<ol class="chapter-questions">` 的 li 数 = details 数 = 「解答：」折叠块数，分别为 5/5/5、2/2/2、3/3/3、2/2/2、2/2/2、3/3/3，**无只列问题未作答**。
- 可运行代码：抽取 `<pre><code class="language-python">` 块为 `/tmp/causal_mask_demo.py` 后执行，stdout 与页面「预期输出」**逐行一致**（`pos 1: weights = ['1.0000', '0.0000', '0.0000'] (sum=1.0000) output = 1.0000`；`pos 2: … output = 1.5000`；`pos 3: … output = 2.0000`；对照双向三行均 `output = 2.0000`）。
- 链接与本地资源：`wiki/standard-attention/index.html`、`wiki/nope/index.html`、`index.html`、`overview.html`、`../../index.html` 均存在；`libs/katex.min.css`、`katex.min.js`、`auto-render.min.js`、`prism.min.js`、`prism-python.min.js`、`prism-primer-light.css`、`prism-primer-dark.css` 均存在。index↔overview 双向链接均在。

来源核对方式：下载论文 HTML 版 `https://arxiv.org/html/1706.03762v7` 全文并逐句检索；下载 Figure 2 左图 `1706.03762v7/Figures/ModalNet-19.png` 目视核对；下载 `https://arxiv.org/abs/2305.19466` 核对 NoPE 摘要与会议信息。

## 问题

- [重要·技术] §1 正文第三段、§1「本章问题」第 1 题答案、来源章节 [C1]：因果掩码约束的两句直接引文标注为 §3.2.3，实际位于 §3.1；且引文把原文 "We also modify" 写成 "We need to modify"｜引文依据：论文 HTML v7 §3.1 Decoder 段（位于 "3.2 Attention" 标题之前）原文 "We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i."｜修复要求：三处出处均由 §3.2.3 改为 §3.1，引文首词还原为 "We also modify"；[C1] 中属于 §3.2.3 的另一句 "self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position." 保留 §3.2.3 标注｜修复：三处出处 §3.2.3 → §3.1（index.html:736 §1 正文第三段、:759 §1 本章问题第 1 题答案、:1061 [C1]）；[C1] 引文首词 "We need to modify" 还原为 "We also modify"（重新拉取 https://arxiv.org/html/1706.03762v7，§3.1 Decoder 段原文确为 "We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions."）；[C1] 中 "self-attention layers in the decoder allow…" 一句保留在 §3.2.3，并把原先笼统的"同节"显式改写为 "§3.2.3——"。｜复验：`grep -c "We need to modify"` = 0；`grep "§3.1——\"We also modify"` 命中 1；`grep "论文 §3.2.3 把这一约束"` = 0；`validate.py` EXIT=0。
- [重要·技术] blockquote.meta 的 "含 set to $-\infty$ 脚注" 与来源章节 [C2] 的 "§3.2.3 脚注"：把正文内嵌括号说明误称为脚注；[C2] 引号内英文与原文不符｜引文依据：§3.2.3 原文为 "We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2."（正文，非脚注）；HTML 全文脚注仅见作者脚注、§3.2.1 的 $d_k$ 方差脚注与 FLOPs 脚注，§3.2.3 无脚注｜修复要求：meta 与 [C2] 的 "脚注" 改为 "§3.2.3 正文"；[C2] 引文逐字改为 "masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections"；Figure 2 左图 "Mask (opt.)" 方框一项经目视核对成立，保留｜修复：meta（index.html:669）"含 set to $-\infty$ 脚注" → "正文含 set to $-\infty$ 说明"；[C2]（:1062）"§3.2.3 脚注" → "§3.2.3 正文"，引文按原文逐字替换为 "We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections."（原为小写误引 "we also mask out (set to $-\infty$) all entries…"）。论文 HTML 复核：§3.2.3 无脚注，该句为正文第三个 bullet。｜复验：`grep -c "脚注"` = 0；`grep "§3.2.3 正文——\"We implement this inside of scaled dot-product attention"` 命中 1；`validate.py` EXIT=0。
- [重要·技术] 来源章节 [C4]、blockquote.meta 的 "§3.1（自回归）"：引用了不存在的 §3.2.4；auto-regressive 引文实际不在 §3.1｜引文依据：论文 HTML 小节列表为 §3.1、§3.2、§3.2.1、§3.2.2、§3.2.3、§3.3、§3.4、§3.5，无 §3.2.4；O(1)/O(n) 原文在 §4 "As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations."，Table 1 "Sequential Operations" 列 Self-Attention=O(1)、Recurrent=O(n)；auto-regressive 原句 "At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next." 位于 §3 引言段（其后紧接 "3.1 Encoder and Decoder Stacks" 标题）｜修复要求：把 [C4] 的 "§3.2.4" 改为 "§4（Why Self-Attention）与 Table 1"；auto-regressive 引文出处由 §3.1 改为 §3；blockquote.meta 的 "§3.1（自回归）" 同步改为 "§3（自回归语义）、§3.1（decoder 掩码约束）"｜修复：meta（index.html:669）"§3.1（自回归）" → "§3（自回归语义）、§3.1（decoder 掩码约束）"；[C4]（:1064）"§3.2.4" → "§4（Why Self-Attention）与 Table 1"；auto-regressive 引文出处 §3.1 → §3，并按论文原文补回被省略的引用标记 [10]，改为 "At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next."（该句位于 §3 引言段、其后紧接 "3.1 Encoder and Decoder Stacks"）。论文 HTML 小节列表复核：§3.1/3.2/3.2.1/3.2.2/3.2.3/3.3/3.4/3.5，确无 §3.2.4；O(1)/O(n) 位于 §4 与 Table 1 的 Sequential Operations 列。｜复验：`grep -c "3.2.4"` = 0；`grep "§3——\"At each step the model is auto-regressive \[10\]"` 命中 1；`grep "§4（Why Self-Attention）与 Table 1"` 命中 1；`validate.py` EXIT=0。
- [重要·技术] §5 正文"深层作用：NoPE 的结构前提"段、§5「本章问题」第 2 题答案、来源章节「辅助解释与类比边界」：排列等变的定义自相矛盾且不正确，"输出也会不同"为过强断言｜引文依据：不适用（页面自身机制陈述，非来源引文）｜修复要求：删除"而每个位置上的输出值不变"及括号内"'交换输入顺序时每个位置的输出不变'"；定义改为"交换输入顺序时输出序列作同样的交换（$f(PX)=Pf(X)$），输出集合不变，因此模型无法区分输入顺序"；把"即使不加任何显式位置编码，不同位置的注意力输出也会不同"改为"该注意力函数不再是排列等变的，模型由此获得区分词序的能力（具体输出是否不同取决于输入）"；三处（§5 正文、§5 答案、辅助解释边界）改到｜修复：删除所有"而每个位置上的输出值不变"/"每个位置的输出值不变"表述，定义统一改写为"交换输入顺序时输出序列作同样的交换（$f(PX)=Pf(X)$），输出集合不变，因此模型无法区分输入顺序"；把过强断言"即使不加任何显式位置编码，不同位置的注意力输出也会不同"改为"该注意力函数不再是排列等变的，模型由此获得区分词序的能力（具体输出是否不同取决于输入）"。共四处：§5 正文（index.html:1025、:1027）、§5 本章问题第 2 题答案（:1046）、来源章节「辅助解释与类比边界」（:1077）。｜复验：`grep -c "位置上的输出值不变\|位置的输出值不变"` = 0；`grep -c "输出也会不同"` = 0；`grep -c "注意力输出也不同"` = 0；`grep -c "f(PX)=Pf(X)"` = 3（正文/答案/辅助解释各一）；`validate.py` EXIT=0。
- [轻微·技术] 来源章节 [C3]（"PyTorch nn.Transformer、Hugging Face、fairseq 等主流实现一致"）与 §5 填充掩码表格"能否叠加"行（"同一 attn_mask 张量中相加"）：框架实现断言无可定位出处，且"相加"对布尔掩码不成立（布尔掩码需按位或）｜引文依据：未核对 + 页面未给出源码路径/版本，无法按 check.md §2.2 第 1 步定位；论文 §3.2.3 无此内容｜修复要求：为框架断言补可定位出处（如 PyTorch 文档或源码路径），否则删除框架名单；"相加"改为"按同一 attn_mask 传入（浮点掩码相加、布尔掩码按位或）"｜修复：[C3]（index.html:1063）删除不可定位的框架名单（"PyTorch nn.Transformer、Hugging Face、fairseq 等主流实现一致"），改引两条可定位的 PyTorch 出处——<code>nn.Transformer.generate_square_subsequent_mask</code>（源码 <code>torch/nn/modules/transformer.py</code>）文档原文 "The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0)."；<code>nn.MultiheadAttention</code>（源码 <code>torch/nn/modules/activation.py</code>）文档 "For a float mask, the mask values will be added to the attention weight" 与 <code>merge_masks</code> 的 "combined with logical <code>or</code>"。据此把 §5 正文（:1011）、填充掩码对照表"能否叠加"行（:1020）、§5 本章问题第 1 题答案（:1039）的"可叠加为同一 attn_mask 张量"/"张量中相加"改为"可合并进同一 <code>attn_mask</code> 张量（浮点掩码相加、布尔掩码按位或）"，三处各加 <code><sup>[C3]</sup></code> 上标。｜复验：`grep -c "Hugging Face\|fairseq"` = 0；`grep -c "generate_square_subsequent_mask"` = 1；`grep -c "张量中相加"` = 0；正文 sup 令牌集合仍为 {C1–C7, F1, F2}，与来源定义双差集为空；`validate.py` EXIT=0。
- [轻微·技术] 来源章节 [C5]（"KV-cache 为自回归 Transformer 推理的标准实现"）：无可定位出处｜引文依据：未核对 + 页面未给出出处；arXiv:1706.03762v7 全文无 KV-cache 相关内容｜修复要求：为该断言补出处，或改标为"本文的工程推断"，保留其已标注的"§3.1 自回归生成语义的推理侧推论"｜修复：[C5]（index.html:1065）删除无可定位出处的"KV-cache 为自回归 Transformer 推理的标准实现"，改标为明确推断："§3 自回归生成语义（C4 引用）的推理侧推论，属本文的工程推断，非论文原文"（出处随 auto-regressive 引文移到 §3）。论文 arXiv:1706.03762v7 全文确无 KV-cache 相关内容。｜复验：`grep -c "KV-cache 为自回归 Transformer 推理的标准实现"` = 0；`grep -c "属本文的工程推断，非论文原文"` = 1；`validate.py` EXIT=0。
- [轻微·技术] 来源章节 [C7]：encoder 引文实际位于 §3.2.3，标注写成 §3.2｜引文依据：§3.2.3 原文 "Each position in the encoder can attend to all positions in the previous layer of the encoder."｜修复要求：把 [C7] 的 "§3.2" 改为 "§3.2.3"｜修复：[C7]（index.html:1067）"§3.2——" → "§3.2.3——"，引文不变。论文 HTML 复核：该句 "Each position in the encoder can attend to all positions in the previous layer of the encoder." 位于 §3.2.3 Applications of Attention in our Model 的第二个 bullet。｜复验：`grep "§3.2.3——\"Each position in the encoder"` 命中 1；`grep "§3.2——\"Each position in the encoder"` = 0；`validate.py` EXIT=0。
- [轻微·格式] §2 符号说明列表与 softmax 段落、§2 掩码矩阵公式：$d_k$ 与 $d$ 未定义；query 位置索引在 $i$ 与 $t$ 间混用；softmax 求和哑标 $i$ 与 $M_{ij}$ 的行标 $i$ 同名｜引文依据：不适用｜修复要求：§2 符号列表补 $d_k$（key 的维度）与 $d$（模型维度）定义；query 位置统一为 $t$、掩码改写成 $M_{tj}$（"第 $(t,j)$ 项""$j>t$"保持 $t$）；归一化式改为 $\alpha_j=e^{s_j}/\sum_k e^{s_k}$｜修复：§2 符号列表（index.html:783–785）补定义——$n$ 为序列长度、$d$ 为模型维度、$d_k$ 为 key 的维度；掩码式（:775）$M_{ij}$ → $M_{tj}$，条件 $j>i$/$j\le i$ → $j>t$/$j\le t$；$M$ 表行标（:794–796）$i=1,2,3$ → $t=1,2,3$；softmax 归一化式（:788）$\sum_i e^{s_i}$ → $\sum_k e^{s_k}$，消除哑标 $i$ 与行标 $i$ 同名；§2 本章问题第 1 题答案（:813）与 §5 填充掩码对照表"屏蔽对象"行（:1017）残留的 $j>i$ 一并统一为 $j>t$，全页 query 位置索引统一为 $t$。｜复验：`grep -c "M_{ij}"` = 0；`grep -c "j>i"` = 0；`grep -c '\sum_i e^{s_i}'` = 0；`grep -c "M_{tj}"` = 1；`grep -c '\sum_k e^{s_k}'` = 1；`validate.py` EXIT=0；Unicode 数学字符扫描 = 0，TAB 字符 = 0。
- [轻微·格式] §3 正文"这一点会在'边界与 NoPE 结构前提'一章用到"与来源章节「辅助解释与类比边界」"只服务'边界与 NoPE 结构前提'一章的对称性讨论"：引用章节标题漏字，与 h2 标题"边界与 NoPE 的结构前提——打破排列对称性"不一致｜引文依据：不适用｜修复要求：两处均改为"边界与 NoPE 的结构前提"（§5 本章问题答案中已写作"边界与 NoPE 的结构前提"，以此为准）｜修复：§3 正文（index.html:862）与来源章节「辅助解释与类比边界」（:1077）的"边界与 NoPE 结构前提" → "边界与 NoPE 的结构前提"，与 h2 标题（:1005）及 §5 本章问题第 2 题答案（:946）对齐。｜复验：`grep -c "边界与 NoPE 结构前提"` = 0；`grep -c "边界与 NoPE 的结构前提"` = 5（h2、核心问题第 5 题答案、§3 答案、§3 正文、辅助解释），无漏字残留；`validate.py` EXIT=0。

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 5
- 处置：修复

## 第 1 轮修复复验记录（2026-09-10）

- 修复方式：以 Write 工具生成 `/tmp/fix_causal_mask_review1.py`（24 处替换，每处断言命中次数 = 1）后运行，避免 heredoc 对 LaTeX 反斜杠的解释；另以单点脚本统一最后一处查询索引 `$j>i$` → `$j>t$`。
- `validate.py`：
  ```
  $ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/causal-mask/index.html
  validation ok: wiki/causal-mask/index.html
  EXIT=0
  ```
- 引用闭环（`/tmp/verify_causal_mask.py`）：正文引用令牌 = {C1–C7, F1, F2}，来源定义令牌 = {C1–C7, F1, F2}，**定义未引用差集 = 空、引用未定义差集 = 空**。
- 相邻双上标：`</sup><sup>` 匹配数 0，`<sup>[X][Y]</sup>` 模式匹配数 0（页面内 3 处 `][` 全在 Python/JS 代码里）。
- Unicode 数学字符（公式外）：0；TAB 字符：0。
- 可运行代码：从 `<pre><code class="language-python">` 抽出的代码块实跑，stdout 与页面「预期输出」**逐字符一致**（`diff` 为空，EXIT=0）。
- 无法修复 / 删除的条目：无。9 条全部改到与来源一致；其中 2 条（[C3] 框架断言、[C5] KV-cache）来源不支持原论断，按规范改为「补可定位出处」与「降级为明确标注的工程推断」，未保留原模糊表述。
