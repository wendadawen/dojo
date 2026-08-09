# 位置编码基础独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 强推理模型对照来源）
- 页面版本：index.html `bdee39aaf81568330a984d927ed154024f15096b`；overview.html `9810c6309d63542484bf347309a068a30bfbe6e6`
- 时间：2026-08-09

审查范围：`wiki/positional-encoding/index.html` + `overview.html`，按 `guides/concept/check.md` 执行段 A 盲读 + 段 B 对照来源（WebSearch "position encoding transformer Vaswani 2017" + "T5 relative position bias"）。未读取 `research/` 目录、未读取其他概念页内容（仅用 Glob 验证前置链接文件存在性）、未修改两份文档。

## 段 A 盲读卡点与学习目标闭环

盲读主线通顺：排列等变 → 正弦公式与手算 → 可学习权衡 → T5 相对 bias → 四类对比与 NoPE。F1 公式符号（pos、i、d_model、10000、ω_i）首次出现处均有定义；d=4 手算表与折叠块逐步代入可复算；F3 线性性质有和角公式推导与 d=4 验证；T5 bias 机制（分桶、每头独立各层共享、clamp 外推）有图示与本质区别两点。

学习目标逐题核对：
1. 为什么需要位置编码 / 注意力什么性质 → 第一章（排列等变）回答 ✓
2. 写出 F1、解释符号、手算 d=4 pos=1,2、说明多频率 → 第二章回答 ✓（手算数值经复算正确）
3. 可学习与正弦的区别与权衡 → 第三章回答 ✓
4. T5 bias 机制与绝对方案本质区别 → 第四章回答 ✓
5. 四类+NoPE 对比表与 K3 选 NoPE 理由 → 第五章回答 ✓

前置链接有效性（Glob 验证，未读内容）：standard-attention、nope、rope、kimi-k3、mla 的 index.html 均存在；首页 ../../index.html 存在；overview.html 与 index.html 互相链接（overview nav "深度教学 →" 链 index，index nav "快速阅读" 链 overview）✓。

## 问题

- [重要·技术] index.html 第三章「可学习绝对位置编码」折叠块「Vaswani 2017 Table 3 row (e) 的完整对比数字」表格：表格设「EN-DE BLEU（big）」列并填入 sinusoidal 28.4 / learned 27.72，标题称其为「Table 3 row (e) 的完整对比数字」。Vaswani et al. 2017 Table 3 是 base model 消融表，row (e) 仅报告 base 的 sinusoidal 27.3 vs learned 26.74；big model 的 28.4 出自 Table 2（最终大模型结果），论文未在 Table 3 row (e) 报告 big 的 learned 消融。WebSearch 全部来源（iclr-blogposts 2025、naokishibu、dongkwan-kim、diqiuzhuanzhuan、meagmohit）均只提"nearly identical / nearly the same"，无一给出 big learned = 27.72 的可定位依据。把 big 数字并入"Table 3 row (e)"表且为 big learned 填 27.72 缺乏来源支持，易使读者误以为论文对 big 也做了 learned 消融并得到 27.72。修法：删除「big」列，只保留 base 的 27.3 / 26.74；若保留 big 列，须为 28.4 单独标注来源（Table 2 最终大模型），并删除无来源的 big learned 27.72，同时在正文明确"big 未做 learned 消融，仅 base 做了对比"。｜ 修复：删除表格 big 列，只保留 base 的 sinusoidal 27.3 / learned 26.74。正文加注"Table 3 row (e) 仅对 base 模型做了 learned 消融；big 模型（Table 2 最终结果 BLEU 28.4）只用正弦，未做 learned 对比"。N1 同步改为只列 base 数字并注明 big 28.4 出自 Table 2。 ｜ 复验：
- [轻微·技术] index.html 第二章「绝对正弦位置编码」频率说明段："$i$ 大（如 $i=d_{model}/2-1$，$d_{model}=512$ 时 $i=255$）：$\omega_i$ 小（$\omega_{255} = 1/10000$）"。由 $\omega_i=1/10000^{2i/d_{model}}$，$i=255,d_{model}=512$ 时 $\omega_{255}=1/10000^{510/512}\approx 1/9622\neq 1/10000$；仅当 $i=d_{model}/2$（不在 $i\in[0,d_{model}/2-1]$ 取值范围内）时才精确等于 $1/10000$。该近似源自 Vaswani §3.5 原文"波长从 $2\pi$ 到 $10000\cdot 2\pi$"（来源本身如此表述，页面未扩大结论），但页面将其写成精确等式 $\omega_{255}=1/10000$，与同段"约 62832 个位置转一圈"的近似口径不一致。修法：将 $\omega_{255}=1/10000$ 改为 $\omega_{255}\approx 1/10000$（或注"近似"），使精确度口径统一。｜ 修复： ｜ 复验：
- [轻微·技术] index.html 第四章折叠块「补充：T5 分桶函数的方向性说明」："邻近距离（如 $|i-j|\le 8$）每距离一桶、远距离按对数合并"。T5 默认 `relative_attention_num_buckets=32`，`max_exact=num_buckets//2=16`，精确桶范围为 $|i-j|<16$（即 $\le 15$），而非 $\le 8$（依据：T5 源码 `_relative_position_bucket`；deepwiki T5 Implementation「Half the buckets for exact small distances」、harshamusunuri「Offsets 0, ±1, ..., ±7」对应 16 桶半区的正负两侧）。修法：将"$|i-j|\le 8$"改为"$|i-j|\le 15$"（或注"默认 32 桶时精确桶范围为 $|i-j|\le 15$，随桶数变化"），与 T5 默认配置一致。｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 开头 callout："自注意力只看内容匹配、不看顺序，交换 token 顺序后每个位置的输出值不变"——"每个位置的输出值不变"在首屏易被小白理解为"整个输出序列完全不变"。第一章后续已准确修正为"输出跟着挪了位置，而每个位置上的输出值不变"（排列等变的正确含义）。修法：将 callout 该句改为"交换顺序后输出序列只是跟着挪位置、每个位置上的值不变"，与第一章表述一致，避免首屏误解。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：进入修复。无阻断；1 个重要问题（big learned 27.72 来源缺失）需修复后复验；3 个轻微问题建议一并修复。学习目标闭环、前置链接有效、互相链接成立、d=4 手算与 F3 推导经复算正确、T5 bias 机制与来源一致。修复完成后需重跑 `.dojo/scripts/validate.py` 并重新对照外部来源。
