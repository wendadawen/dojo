# KV cache 审查记录（第 3 轮）

- 页面版本：index.html aff624f68ab8bbf6e5afdf12c77b9864b9ecc488；overview.html cb4d7c41a3c1f4b3b493ac9af054dca6e0d500c6
- 审查时间：2026-08-20 11:39 CST
- 审查者：独立子代理（reviewer-kv-cache-3）
- 已完整阅读章节：index.html——核心问题（4 题）、1 注意力为什么需要缓存（含 4 token 例子表与"逐步 K/V 计算"折叠块）、本章问题 ×2、2 prefill 与 decode（含流程图示）、本章问题 ×2、3 缓存有多大（含"完整代入与两个检查"折叠块）、本章问题 ×2、4 为什么显存成为瓶颈（含对照表）、本章问题 ×3、来源与范围说明（核心论断与来源／核心公式与来源／外部数字与实验条件／构造示例／辅助解释与类比边界／简化条件及其限制）；overview.html——全部（这是什么／为什么需要它／核心机制／关键结论与边界）

## 数值口径复算（按页面声明 1 KB=1,024 B、1 GB=2^30 B）

- 2.44 GB（20,000 token）：20,000 × 131,072 B = 2,621,440,000 B ÷ 2^30 = 2.4414 ≈ 2.44 GB ✓
- 16 GB（128K token）：131,072 token × 128 KB = 17,179,869,184 B = 16 × 2^30，恰为 16 GB ✓
- 0.33M（40 GB）：40 × 2^30 ÷ 131,072 = 327,680 token = 0.3277M ≈ 0.33M ✓
- 4 token 例子：1+2+3+4 = 10 组 vs 4 组 ✓；n(n+1)/2、20,000 token 时 2 亿组 vs 2 万组 ✓
- GQA：H_kv/H_q = 8/32 = 1/4；MHA 化为 512 KB/token（128×4）✓
- 128 KB/token 代入链：2×32×8×128×2 = 131,072 B = 128 KB ✓；OPT-13B：2×5120×40×2 = 800 KB ✓
- 例外：40,000 token 一处复算失败，见问题 1。1.6 GB（OPT-13B 单请求）按二进制口径为 1.5625 GB，页面已明确标注为"vLLM 论文给出的原始数字"（原文即 1.6 GB，本轮已核对），属引用原文数字，不列为问题。

## 来源核对抽查（C/F/N 各 1 条，另附 2 条顺带核对）

- C5（既有系统请求完成即丢弃缓存）→ SGLang §1 原文："In existing inference engines, the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls and significantly slowing down the execution."；vLLM §4.2 原文："Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests."。页面表述与两处原文一致 ✓
- F1（每 token 字节数公式）→ vLLM §3 原文："the KV cache of a single token demands 800 KB of space, calculated as 2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)"。页面 MHA 形式与之一致；GQA 推广形式页面已标注推断并经 Llama-3.1-8B 复算 ✓
- N1（OPT-13B 800 KB/token、2048 token 1.6 GB）→ vLLM §3 原文："Since OPT can generate sequences up to 2048 tokens, the memory required to store the KV cache of one request can be as much as 1.6 GB." ✓
- 顺带核对 N3（65%/30%）→ vLLM §1 原文："Approximately 65% of the memory is allocated for the model weights… Close to 30% of the memory is used to store the dynamic states of the requests." ✓
- 顺带核对 C4（显存分布）同上，一致 ✓
- 材料限制说明：C2、N4 标注来源为 Strata 论文（§2.1/§1），本轮允许材料不含 Strata 论文文本，无法直接核对原文；N4 已通过页面自身复算（0.33M）验证一致，C2 两阶段描述与 vLLM §2.2 原文（"The prompt phase takes the whole user prompt… computes the probability of the first new token"／"The autoregressive generation phase generates the remaining new tokens sequentially"）内容一致。

## 机械项检查

- $ 配对：index.html $$ 6 次（偶数）、单 $ 314 个（偶数）、逐行无奇数行；overview.html 单 $ 36 个（偶数）✓
- math 字符位置：两页 $…$ 之外仅 en-dash（区间号"128K–1M""第 2–4 章"）与 overview 导航"← 返回"、标题间隔号"·"，均为标点/UI 字符，非数学符号 ✓
- 链接有效性：../../index.html、overview.html、../standard-attention/index.html、../prefix-caching/index.html、../paged-attention/index.html、../strata/index.html 及 libs/ 下 8 个本地资源文件全部存在；overview 与 index 相互链接 ✓
- 问题块：核心问题 4 题、各章本章问题（2/2/2/3 题）均有解答折叠块，答案独立可读、与正文一致，核心问题答案均指明所在章节 ✓
- meta：description、dojo:summary、dojo:type=concept、dojo:topics、dojo:tag 齐全 ✓

## 问题

- [重要·技术] index.html 第 3 章折叠块「展开：128 KB/token 的完整代入与两个检查」："40,000 token 手册即 5.12 GB"复算失败：按页面声明口径 40,000 × 131,072 B ÷ 2^30 = 4.8828 GB，应为约 4.88 GB；且与同句"token 数翻倍则总量翻倍"自相矛盾（2.44 × 2 = 4.88 ≠ 5.12）。5.12 系 40,000 × 128 KB = 5,120,000 KB 误按千进制换算所得。｜引文依据：页面口径声明"1 KB=1,024 字节、1 GB=2^30 字节"（第 3 章正文）；同折叠块 2.44 GB 与 16 GB 均按该口径复算通过。｜修复要求：将该处"5.12 GB"改为"约 4.88 GB"，使线性性检查与 2.44 GB 翻倍自洽。｜修复：｜复验：
- [轻微·可读性] index.html 第 4 章正文首现"40 GB HBM"（"量级判断：40 GB HBM 对 Llama-3.1-8B 大约只够缓存 0.3M token"）：HBM 缩写未在此处随文解释，唯一解释位于核心问题 4 的解答折叠块内（"40 GB HBM（GPU 高带宽显存）"）；按"折叠内容收起后正文仍能建立完整结论"要求，收起折叠块的读者在正文与对照表中两次遇到未解释缩写。｜引文依据：不适用。｜修复要求：在第 4 章正文 HBM 首现处加括注"（GPU 高带宽显存）"，或复用核心问题折叠块中的既有解释写法。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 1
- 处置：修复
- 说明：核心数字（2.44 GB、16 GB、0.33M、10 组 vs 4 组、GQA 1/4）全部按统一口径复算通过；C/F/N 抽查 3 条（另附 2 条）均与 vLLM/SGLang 原文一致；机械项全部通过。仅存的数值错误是折叠块内"5.12 GB"一处（应为 4.88 GB），修复并复验后即可发布。C2/N4 的 Strata 原文本轮材料不可得，未能直接核对，处置沿用前两轮核对结论并以 vLLM §2.2 交叉印证与页面复算佐证。


## Round 3 修复记录

| 编号 | 问题 | 修复 | 复验 |
|---|---|---|---|
| 重要 #1 | 折叠块"40,000 token 即 5.12 GB"复算错 | 改"约 4.88 GB（$2.44\times 2$）" | 复算一致 |
| 轻微 #1 | 第 4 章正文 HBM 首现无注 | 加"（GPU 高带宽显存）" | 验证 |
