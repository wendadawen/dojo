<!-- review-meta
round: 5
page: wiki/attention-sink/index.html
reviewed_content_sha256: ccbc8a07754fd12d
-->
# 注意力汇聚点审查记录（第 5 轮）

- 页面版本：50b64786e0452583c5d2f2ab2c649d3d439a0eba（git hash-object wiki/attention-sink/index.html）
- 审查时间：2026-09-13 19:35
- 审查者：独立子代理
- 已完整阅读章节：核心问题、常见误解、1. 现象、2. 成因、3. 两种做法（3.1 保留初始位置的 KV、3.2 每个头一个可学习标量、本章问题，含全部折叠块）、4. 对缓存与推理意味着什么（4.1–4.3，含代码与解析解折叠块）、来源与范围说明

外部核对：StreamingLLM（arXiv:2309.17453）v1/v3/v4 正文与 Table 1/2/3、附录 A 逐条抓原文；DeepSeek-V4.1-Flash 官方参考实现镜像 README（wiki/deepseek-v4-1/research/official/）；份额公式与页面代码本机复算；`python3 .dojo/scripts/validate.py wiki/attention-sink/index.html` 返回 ok。

已核对通过（无问题）的主要事实项，供复验参考：Table 1「0+1024」5158.07、「4+1020」5.40、「4"\n"+1020」5.60；Table 2「Introducing four initial tokens generally suffices; further additions have diminishing returns.」；Table 3 Learnable Sink「1+1023」18.01、Vanilla「2+1022」18.05；附录 A「it does not extend the models' context window or enhance their long-term memory capabilities.」；§3.1「a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task」；§3.2 位置口径「StreamingLLM focuses on positions within the cache rather than those in the original text.」；份额公式复算 e^0/(128+e^0)=0.0078、e^2/(128+e^2)=0.0546、e^5/(128+e^5)=0.5369、e^8/(128+e^8)=0.9588，与页面代码输出一致；解析解 (v1+v2)/(2+e^2) 分母 9.389；W=128 与官方配置 window_size=128 一致；43=40 主干+3 MTP、每头一个 [64] fp32 与官方配置 n_heads=64 一致。

## 问题

- [阻断·来源] 来源与范围说明 [C4]、[F1] 及文末 meta「主要依据」段：引用 DeepSeek-V4.1-Flash 官方参考实现 `official/inference/kernel.py`（稀疏注意力分母）与 `official/inference/model.py`（`attn_sink` 参数定义），作为全文「主要依据」之一。这两个文件已从仓库移除，页面给出的位置无法定位。｜引文依据：`git ls-files 'wiki/deepseek-v4-1/research/official/*'` 只返回三个文件——`wiki/deepseek-v4-1/research/official/README.md`、`.../official/inference/README.md`、`.../official/encoding/README.md`；`git log --oneline --all -- wiki/deepseek-v4-1/research/official/inference/kernel.py` 的最后一次变更即 commit 13ead44「research/ 只保留 md：实测产物先登记进 measured.md 再删除（264 个文件 / 36.5 MB）」。｜修复要求：删除 `kernel.py`/`model.py` 的路径引用，或改为指向仓库中仍然存在的等价记录位置（例如登记了 KRN 行号与「不可选槽位（索引 -1）分子分母都不贡献」的 source 记录），使读者能定位 [C4]/[F1] 的依据。｜修复：｜复验：
- [阻断·来源] 来源与范围说明 [C5]、[F2]、[N5]：三处把「解析解与边界」和「份额数值」的复算脚本写作 `research/verify_sink.py`（[C5]「复算脚本 `research/verify_sink.py`」、[N5] 同）。该文件已从本页 research/ 目录移除，复算依据无法定位。｜引文依据：`git log --oneline --all -- wiki/attention-sink/research/verify_sink.py` 返回 13ead44（删除）与 f2a06c0（新增）；`git ls-files 'wiki/attention-sink/research/*'` 现存仅为 `draft-check.md`/`evidence.md`/`glossary.md`/`measured.md`/`outline.md`/`scope.md`/`review-1..4.md`，目录内已无 `.py` 与 `.out`。｜修复要求：删除该路径，或改指向登记该脚本输出的现存文件（如 `wiki/attention-sink/research/measured.md` 中对应条目）。｜修复：｜复验：
- [阻断·来源] 来源与范围说明 [N4]：把「43 个张量、形状 (64,)、F32」这一数字的来源写作 `wiki/deepseek-v4-1/research/ckpt/headers.json`（并称由 HTTP Range 读取 48 个分片头）。该文件已从仓库移除，位置无法定位。｜引文依据：`git ls-files 'wiki/deepseek-v4-1/research/ckpt/*'` 返回空；同模块 `wiki/deepseek-v4-1/research/measured.md` 第 26 行仍把 `ckpt/headers.json` 登记为归档中间数据（0.0 MB）。｜修复要求：把该来源改指向现存的记录条目（measured.md），或删除该路径及仅由它支撑的形状描述。｜修复：｜复验：
- [轻微·表述] 全文含三处元话语/临场评价式引导句，属规范 2.2.12 所列「下面来看…」「需要注意的是」同类：第 1 章首句「先看观察到的事实。」（L115）、第 2 章「这个式子有一个常被忽略的后果：」（L162，含「常被忽略」的临场评价）、第 4.3 节开头「需要明确的是，」（L315）。｜引文依据：不适用｜修复要求：改写为直接陈述——L115 直接从「StreamingLLM 在多个自回归语言模型上发现…」起句；L162 改为「分母是可见槽位指数之和；指数函数的值恒大于零，因此…」；L315 删去「需要明确的是，」，以「保住汇聚点解决的是稳定性，不是能力。」起句。｜修复：｜复验：

## 结论

- 统计：阻断 3 / 重要 0 / 轻微 1
- 处置：修复

补充说明（不计入问题）：（a）页面多处以「本页」自称（L66、L129、L289、L310、L330、L360、L366–367），均为范围/边界声明且符合 style-guide §12「自称使用『本页』或『本文』」，不构成会话指代，未计为问题；全文无「我们/你们/你/笔者」等会话人称。（b）正文多处以「第 N 章」「第 4.2 节」交叉引用，与同仓库其他概念页（sliding-window-attention、kv-cache、deepseek-v4-1）写法一致，未计为问题。（c）页面内链 `../sliding-window-attention/index.html`、`../kv-cache/index.html` 均真实存在，无「（待生成）」占位；公式全部由 KaTeX 书写，脚本/样式/代码块外无 Unicode 数学字符（`↑ ⌂ ◐ ☀` 为 UI 图标）。（d）overview.html 与 index.html 相互链接，head 的 description/dojo:summary/dojo:type/dojo:topics/dojo:tag 齐备，topics 取值「注意力机制」在词表内。