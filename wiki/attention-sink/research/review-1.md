# 注意力汇聚点审查记录（第 1 轮）

- 页面版本：index.html `c05846211a7d786850d8ed16ad54c084f815a46d`；overview.html `63d28e1ee2247f59ba69759892f6bf40929c2ea7`
- 审查时间：2026-09-10 16:18
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节：标题与元信息（description / dojo:summary / blockquote.meta）→ 开篇 callout → 核心问题（4 题，含折叠解答）→ 常见误解 → 1. 现象（含「解读/预测/实测」对比表、本章问题 2 题折叠解答）→ 2. 成因（含公式、符号表、补充折叠块、本章问题 2 题折叠解答）→ 3. 两种做法（3.1 形态一；3.2 形态二含公式、符号表、对比表、解析解折叠块；本章问题 3 题折叠解答）→ 4. 对缓存与推理意味着什么（4.1、4.2 含公式与代码折叠块、4.3；本章问题 2 题折叠解答）→ 来源与范围说明（全部 6 个 h3）→ overview.html 全文
- 核对说明：StreamingLLM 依据 arXiv:2309.17453v3 HTML 全文逐句定位；DeepSeek 依据 `official/inference/kernel.py`、`model.py`、`config.json` 与 `ckpt/headers.json`（实测 43 个 `attn_sink`、形状 `{(64,)}`、dtype `F32`、主干 0–39 连续 + MTP 0–2）；页面可运行代码块已复制执行，输出与页面「预期输出」逐字符一致；`verify_sink.py` 因本机无 `torch` 无法执行，改为静态审查 `verify_sink.out`；`python3 .dojo/scripts/validate.py wiki/attention-sink/index.html` 返回 `validation ok`。

## 问题

- [重要·技术] index.html:941（第 3 章「本章问题」第 2 题解答）：解答称把 sink 计入行最大值「等价于人为改变了两者的相对权重」，与同段前半句「分子分母同时缩放、结果不变」自相矛盾；精确算术下 $m$ 的任一公共平移不改变比值，该归因不成立，来源只支持「实现里 sink 项在累积循环之后加入分母、行最大值只对可见槽位求 max」｜引文依据：kernel.py L369 `T.reduce_max(acc_s, scores_max, dim=1, clear=False)`（acc_s 仅由可见槽位构造，无效槽位置 `-inf`，见 L364），L383 `sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`，L355 `T.fill(scores_max, -1e30)`；把 $m$ 换成 $m+c$ 时 $o=\sum_j e^{s_j-m}v_j/(\sum_j e^{s_j-m}+e^{\text{sink}-m})$ 的分子分母同乘 $e^{-c}$，比值不变｜修复要求：删除「可见槽位的指数项被压得过小，等价于人为改变了两者的相对权重」，改为与 kernel.py 一致的实现事实（sink 项在 online softmax 累积循环之后加入分母；行最大值只取可见槽位分数），并明确「公共平移不改变比值，因此该选择来自实现结构而非数学必要性」｜修复：｜复验：
- [重要·技术] index.html:805（第 1 章第 2 段）与 index.html:825（第 1 章「本章问题」第 1 题解答）：把换行符实验表述为「把原来的内容放回去，困惑度也只是回到同一水平」，方向与来源不符；来源是「把这 4 个换行符位置作为初始位置保留」即可把困惑度恢复到与保留原始 4 个 token 相当的水平，正因如此才说明起作用的是位置而非内容｜引文依据：arXiv:2309.17453v3 §3.1「wherein the first four tokens are substituted with the linebreak token "\n". The observations indicate that the model still significantly emphasizes these initial linebreak tokens. Furthermore, reintroducing them restores the language modeling perplexity to levels comparable to having the original initial tokens.」；同节 Table 1 行「4"\n"+1020 → 5.60」「4+1020 → 5.40」｜修复要求：两处统一改为「换成换行符后，用这 4 个换行符位置作为初始位置保留，困惑度（5.60）与保留原始 4 个 token（5.40）相当」｜修复：｜复验：
- [轻微·格式] index.html:912（3.2 对比表「数量」行）：表格单元格直接出现 Unicode 乘号「×」（「43 层 × 64 头」），未由 KaTeX 渲染｜引文依据：不适用（style-guide.md §11：表格单元格内数学符号必须包在 `$...$` 中）｜修复要求：改为 `43 层 $\times$ 64 头`，或改写为不含乘号的表述｜修复：｜复验：
- [轻微·格式] index.html:1021、index.html:1030（来源与范围说明下 h3）：标题为「核心论断与来源」「核心公式与来源」，偏离固定命名｜引文依据：style-guide.md §1「来源章节（`来源与范围说明`）下的 h3 使用固定命名：`论断与来源（C）`、`公式与来源（F）`……不编号」｜修复要求：改为「论断与来源（C）」「公式与来源（F）」｜修复：｜复验：
- [轻微·格式] index.html:887、index.html:903、index.html:959（正文组合引用）：组合引用写成相邻的两个上标 `<sup>[C3][N2]</sup>`、`<sup>[C4][N4]</sup>`、`<sup>[C2][C3]</sup>`，与规范给出的组合写法不一致｜引文依据：style-guide.md §6「可组合：`<sup>[C7, F2, N2]</sup>`」｜修复要求：统一改为 `<sup>[C3, N2]</sup>` 形式｜修复：｜复验：
- [轻微·可读性] index.html:841-847（第 2 章公式与符号表）、index.html:895-901（3.2 公式与符号表）：第 2 章公式的输出符号 $o$ 未列入符号表；3.2 公式复用 $q$、$k_t$、$v_t$、$d$、$\mathcal{T}$ 未指明首次定义位置｜引文依据：不适用（style-guide.md §11「公式后紧跟 `<ul>` 逐项定义每个符号」；content-examples.md A2/A4）｜修复要求：第 2 章符号表补 $o$（注意力输出）的定义；3.2 公式后注明「$q$、$k_t$、$v_t$、$d$、$\mathcal{T}$ 含义见第 2 章」｜修复：｜复验：
- [轻微·技术] index.html:917-926（3.2 解析解折叠块）、index.html:1026-1027、index.html:1037（来源条目）：解析解与「整行无效输出全零」这一断言所在折叠块没有任何上标引用，[C5]、[C6] 无正文引用；[N3] 在正文中无任何引用，来源编号与正文未完全双向对应｜引文依据：verify_sink.out「最大差 0.00e+00」（sink = −inf/0/2/5 四行）与「[2] 边界: 整行 -1 的输出 = [0.0, 0.0, 0.0, 0.0]」；arXiv Table 3「Learnable Sink 1+1023 → 18.01」「Vanilla 2+1022 → 18.05」｜修复要求：折叠块补 `<sup>[C5]</sup>`；[N3] 在正文相应位置引用，或从来源章节删除｜修复：｜复验：
- [轻微·可读性] index.html 全文（正文首次依赖处）：正文首次依赖「KV cache」「滑动窗口」「标准 softmax 注意力」时均未给出前置概念页链接，主页面无法跳转到前置页；overview.html 已提供这三个链接｜引文依据：不适用（check.md 2.2.6「前置概念链接有效」；overview.html:69-71 已链接 `../standard-attention/`、`../kv-cache/`、`../sliding-window-attention/`，三个页面均存在）｜修复要求：在正文首次依赖处补上对应概念页链接，与 overview.html 使用同一份前置概念映射｜修复：｜复验：
- [轻微·技术] index.html:887（3.1）：把论文称为 crucial 的设计写成副作用——「这条路线还顺带改变了位置口径」，并给出论文未陈述的因果（「由于要保留的位置与最近的窗口在原文中相距很远」）｜引文依据：arXiv §3.2「When determining the relative distance and adding positional information to tokens, StreamingLLM focuses on positions within the cache rather than those in the original text. This distinction is crucial for StreamingLLM's performance.」与「This method of assigning positional embedding within the cache is crucial to StreamingLLM's functionality」｜修复要求：去掉「顺带」，改为陈述该位置口径是论文强调的关键设计，不把「相距很远」写成论文给出的原因｜修复：｜复验：
- [轻微·技术] index.html:999（4.3）：「要真正『记得更远』，需要窗口之外的另一路可见性（例如压缩后的全局 KV）」是对具体实现机制的断言，页面未引用来源｜引文依据：model.py L775-778 `if self.compress_ratio:` … `kv = torch.cat([kv, compress_kv], dim=1)` 可支持该例，但页面未给出引用｜修复要求：补 `<sup>[C4]</sup>` 引用，或改写为明确标注的推断｜修复：｜复验：
- [轻微·技术] index.html:973（4.2）：「也就是说，训练可以把汇聚点调成『几乎不参与』或『主导注意力』两种极端」由 $W=128$、槽位等权这一构造示例推出，页面未读取真实权重，结论未标注为推断｜引文依据：verify_sink.out 第 [3] 段为公式直接计算；headers.json 仅给出形状 `(64,)`、dtype `F32`，无权重数值｜修复要求：改写为「在等权槽位假设下，份额随标量取值可覆盖 0.78%–95.88%；真实取值由训练决定」，去掉对训练结果的断言｜修复：｜复验：
- [轻微·技术] index.html:6（`description`）：称页面含「份额实测」，与正文把份额数值标为构造示例的定位不一致｜引文依据：index.html:1039「[N5] 份额数值（0.78% / 5.46% / 53.69% / 95.88%）：构造示例，$W=128$、槽位等权」｜修复要求：把「份额实测」改为「份额公式与数值示例」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 10
- 处置：修复
