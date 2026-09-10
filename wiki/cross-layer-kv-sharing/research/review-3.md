# 跨层 KV 复用审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 `f1df025dee8f2fa504ab263bb437b81359e97753`（overview.html 同为未跟踪新文件）
- 审查时间：2026-09-10 16:52
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次审查）
- 已完整阅读章节（按顺序）：核心问题（learning-goals）→ 常见误解（misconceptions）→ 1. 复用的是哪一层的 KV → 2. 省下多少：层维度被消掉 → 3. 两级复用：共享 KV 与复用索引 → 4. 前提与边界：同组才能共用 → 来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制）→ overview.html 全文

## 机械验证结果

- `validate.py`：`validation ok`（退出码 0）。
- 引用编号双向闭合：正文 `<sup>` 引用集合 = 来源章节定义集合 = `{C1–C7, F1–F4, N1–N5}`，两个差集均为空。
- 相邻双上标：`</sup><sup>` 出现 0 次，全部为合并形式（如 `<sup>[C2, F2]</sup>`）。
- 定界符外 Unicode 数学字符：全文仅 1 处 `×`（位于 [N1] 逐字英文引文「…by about 80× for 65B models.」内，属允许范围）；`·` 仅出现在 JS 阅读时间模板字符串，非数学符号。
- 来源章节 h3 固定命名：`论断与来源（C）`、`公式与来源（F）`、`外部数字与实验条件（N）`、`构造示例`、`辅助解释与类比边界`、`简化条件及其限制` 全部正确，含（N）后缀。
- h2 编号 1–4 连续，各章末 `本章问题`、页面级 `核心问题` 命名正确，两级问题均配 `解答：` 折叠块且末尾指向对应章节。
- `<head>` 元数据齐全：description / dojo:summary / dojo:type=concept / dojo:topics / dojo:tag 均存在；validate.py 已通过词表校验。
- overview.html 与 index.html 相互链接；前置概念链接（kv-cache、standard-attention、mla、dsa）均存在。

## 问题

- [轻微·技术] 来源与范围说明 [N5]：`§4.5 有同义重述` 位置标注错误。技术报告无 §4.5（第 4 章仅有 4.1/4.2/4.3），890 字节的同义重述实际位于 §6 Conclusion（tech_report.txt 行 1968–1970：「cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) and FP4 KV caching reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash.」）。主引文（行 16–19）与引文内容正确，仅佐证小节的编号写错。｜修复要求：将「§4.5 有同义重述」改为「§6 Conclusion 有同义重述」。｜修复：[N5] 章节归属改为 §6 Conclusion｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·技术] 来源与范围说明 [F2]：`脚注说明 $N, L, D$ 分别为序列长度、层数、隐藏维` 归属不准确。该符号定义来自 YOCO Table 2 的表注（「Table 2: Inference memory complexity of KV caches. $N, L, D$ are the sequence length, number of layers, and hidden dimension.」），不是脚注；脚注 1 讲的是「once」的限定。复杂度对照本身（O(LND) vs O((N+L)D)）与 Table 2 一致。｜修复要求：将「脚注说明」改为「Table 2 表注说明」。｜修复：[F2] 归属改为「Table 2 表注说明 N, L, D 分别为序列长度、层数、隐藏维」｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·可读性] overview.html「核心机制」第 5 条：`压缩比不同的层无法共用同一批条目，因此共享发生在按压缩比划分的组内` 与 index.html 第 4 章的修正口径存在张力。index.html 明确「压缩比一致只是必要条件、不是分组依据——层 2–19 压缩比全为 2 却被拆成三个组，真正分组由 kv_source_layers 声明」；overview 的「按压缩比划分的组内」可能被读作「分组由压缩比决定」。overview 第 3 条「关键结论与边界」已给出正确分组（2–7 / 8–13 / 14–19 / 20–39）并标注「成立条件：组内压缩比一致」，未形成错误结论，但措辞应统一。｜修复要求：将「按压缩比划分的组内」改为「压缩比一致的组内」或「组内压缩比一致」，与 index.html 的必要条件口径一致。｜修复：overview 改为「组内压缩比一致」，与 index 第 4 章口径一致｜复验：已复跑 validate.py 通过并核对修改位置｜

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布

满足 check.md §5 全部发布条件：三轮审查完成且本轮为独立审查者；每条来源论断均有引文依据记录（本文核对的原文片段与关键数值见下）；阻断与重要问题均为 0；3 条轻微问题均有明确接受理由——[N5]/[F2] 为主引文正确、仅佐证/符号定义的小节编号或「脚注/表注」用词写错，不影响任何结论；overview 措辞虽与 index.html 存在轻微张力，但同页第 3 条已给出正确分组与必要条件口径，主页面 index.html 表述精确。学习目标由第 1–4 章完整回答；两级问题均有解答折叠块；数学符号全部 LaTeX；结构图为 HTML；validate.py 成功；可运行代码（verify_csa2_modes.py）输出与页面分组描述一致；关键论断与数字已逐条重核来源；`<head>` 元数据齐全；overview 与 index 互链；前置概念链接有效。

### 关键来源核对摘要（按 check.md §2.2 四步）

- [C1] YOCO Abstract/§2：核对到「a cross-decoder stacked upon a self-decoder. The self-decoder efficiently encodes global key-value (KV) caches that are reused by the cross-decoder via cross-attention.」「YOCO is stacked with L layers, where the first L/2 layers are self-decoder while the rest modules are cross-decoder.」一致。
- [C2] §2.3：核对到「the number of caches is O(N+CL)…about O(N) caches are required, i.e., you only cache once.」「YOCO roughly saves L times GPU memory」一致。
- [C3] §2.3/Table 1/3：核对到「we can exit early before entering the cross-decoder during the prefill stage.」「at least half prefilling latency reduction」一致。
- [C4] 技术报告 §2.3.1（行 490–545）：Full/Reindex/Reuse 三段引文逐字一致；model.py `Indexer.__init__` 的 `owns_k`/`is_candidate_source`/`uses_candidates`（行 500/502/503）与 `Attention._compress_kv` 发布-读取顺序（行 739–763）一致。
- [C5] verify_csa2_modes.out：核对到「每个压缩层读到的 cache 都来自其组内 source 层: True」「源层读到自己刚发布的 cache: True」，分组 2–7 / 8–13 / 14–19 / 20–39，与页面一致。
- [C6] config.json：`kv_source_layers=[2,8,14,20]`；`compress_ratios` 层 2–19=2、层 20–39=1；`Compressor.__init__` 按 `compress_ratios[layer_id]` 取压缩比（model.py 行 439）一致。
- [C7] YOCO 脚注 1：核对到「The word "once" refers to global KV cache…」一致。
- [F1] YOCO Eq.(2)：「K̂=LN(X^{L/2})W_K, V̂=LN(X^{L/2})W_V」「where W_K, W_V ∈ R^{d×d} are learnable weights」一致。
- [F2] Table 2：「Transformer O(LND)」「YOCO O((N+L)D)」一致。
- [F3] Table 3：「Transformer O(LN²D)」「YOCO O(LND)」一致。
- [F4] 技术报告 §2.2（行 389–396）：核对到「the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H_{L/2}, using layer-dependent projection weights (W_l^{KV} and W_l^{Z})」，确认「CED 共享的是输入 H_{L/2} 而非投影结果」的区分正确；跨层共享派生缓存属 CSA2（§2.3.1）的行为。
- [N1] 80×：核对到「the memory of KV caches can be reduced by about 80× for 65B models」位于 §1 Introduction；「YOCO can serve 128K tokens with 1GB GPU memory…at 65B model size」位于 §4.4，与页面标注一致。
- [N4] 技术报告 §2.1（行 327–330）：核对到「This nearly halves prefill computation」一致。
- [N5] 890 字节：核对到行 16–19 引文一致；主引文正确，佐证小节编号见上方问题 1。

## 发布结论

- 审查轮次：第 1 轮（0/4/2）→ 第 2 轮（0/2/3）→ 第 3 轮（0/0/3，可发布）
- 每轮均由未参与写作、未参与前序审查的独立审查者执行
- 阻断与重要问题全部关闭；遗留轻微问题已逐条修复
- `.dojo/scripts/validate.py` 返回成功；headless Chrome 渲染实测通过（KaTeX 正常、无占位符、无标签重叠、折叠块数与问题数匹配）
- `overview.html` 与 `index.html` 相互链接；前置概念页均存在
- 结论：**可发布**（跨层 KV 复用）
