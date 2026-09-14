<!-- review-meta
round: 6
page: wiki/sliding-window-attention/index.html
reviewed_content_sha256: cd818003c5adc4a6
-->
# 滑动窗口注意力审查记录（第 6 轮）

- 页面版本：index.html 工作树哈希 9dd29cd2e232243ccb6bb862ae022f9b17df1454（sha256 bd66c159084e206e1061748b7496f03dce94dd518f4cab89f6328fbdf7dc6745）
- 审查时间：2026-09-14 17:14
- 审查者：独立子代理（未参与写作，也未参与前序轮次的审查与修复；未读取本页 research/ 下任何文件）
- 已完整阅读章节：核心问题；常见误解；1. 窗口把可见集切成「最近 $W$ 个位置」（含 1.1、本章问题、$-1$ 槽位折叠块）；2. 省下的是打分次数与缓存字节——以及精度取舍（含 2.1–2.3、本章问题、40 层账目折叠块）；3. 层堆叠接力：感受野 $k\times W$ 与它的天花板（含 3.1–3.3、本章问题、StreamingLLM 对照数字折叠块）；4. 环形缓冲：槽位、覆盖与位置编码（含 4.1–4.3、本章问题、环形缓冲代码折叠块）；来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）；两处图注与全部折叠块。

## 来源核对依据（本轮实际比对，供复验）

- [C1] Mistral 7B（arXiv:2310.06825）§2 原文：「the hidden state ... h_i, attends to all hidden states from the previous layer with positions between i-W and i. Recursively, h_i can access tokens from the input layer at a distance of up to W×k tokens.」；图 1 图注原文：「each token can attend to at most W tokens from the previous layer (here, W=3)」与「At each attention layer, information can move forward by W tokens. Hence, after k attention layers, information can move forward by up to k×W tokens.」——页面 §1、§3.1、[C1] 的转述与取值裁定（论文正文写 [i-W, i]、图注写最多 W 个，实现取 4872–4999）一致。
- [C2] 同论文 §2 Rolling Buffer Cache 原文：「the keys and values for the timestep i are stored in position i mod W of the cache」「when the position i is larger than W, past values in the cache are overwritten, and the size of the cache stops increasing」——与 §2.2、§4.1、[F2] 一致。
- [N1]/[N2]：Mistral 7B window_size=4096、32 层；4096×32=131 072；原文「On a sequence length of 32k tokens, this reduces the cache memory usage by 8x」——§3.1「约 131 072 个 token」与 [N2]「32768/4096=8」一致。
- [C3]/[N3]：StreamingLLM（arXiv:2309.17453）§3.1 原文「removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function」；Table 1（Llama-2-13B，PG19 首本书 ≈65K token）：0+1024=5158.07、4+1020=5.40、4"\n"+1020=5.60；Table 2 图注「Window attention (0+y) has a drastic increase in perplexity」——表号与数值逐项吻合（5158.07 在 Table 1，图注在 Table 2）。
- [N4]：StreamingLLM Table 3 三个对比模型均为 160M 参数、PG19 首样本；learnable sink 1+1023=18.01、vanilla 2+1022=18.05——与页面 §3.3 补充块一致。
- [C5]：StreamingLLM §3.2 原文「StreamingLLM focuses on positions within the cache rather than those in the original text.」——与 §4.3 一致。
- 字节账复算：128×512×1 B=65 536 B=64 KiB；40×64 KiB=2560 KiB=2.50 MiB；3×64 KiB=192 KiB≈0.19 MiB，合计 2.6875≈2.69 MiB；1 048 576×512×1 B=512 MiB，×40=20 GiB；512 MiB/64 KiB=8192（1/8192）。打分表：4096²=16 777 216 与 4096×128=524 288（32×）；131072²=17 179 869 184 与 131072×128=16 777 216（1024×）；1048576²=1 099 511 627 776 与 1048576×128=134 217 728（8192×）——三行全部吻合。
- 感受野复算：40×128=5120；5120/1 048 576=0.488%≈0.49%；999 999−5120=994 879，可达 5121/1 048 576≈0.49%，不可达≈99.5%——§3.2 的 0.49% 与 99.5% 彼此自洽。
- 与仓库内 deepseek-v4-1 页交叉核对：window=128、KV 头 $1\times512$、SWA KV 保持 FP8、主 KV FP4、n_mtp_layers=3、compress_ratios 前两层为 0（纯 SWA）、报告 §2.4.4「We retain FP8 for the SWA KV cache due to its sensitivity to quantization」——本页取值与之一致；「3 个 DSpark 草稿层各注册一份窗口缓存」与 compress_ratios 尾部 0×3（草稿层压缩比为 0，即纯 SWA）相容，不构成矛盾。
- 代码：按页面原文照抄执行（W=8、序列长 20），输出与页面「预期输出」逐字符一致（末步槽位序列 [4,5,6,7,0,1,2,3]、映射回位置 [12,13,14,15,16,17,18,19]、全程 assert 通过）。
- 机械项：`.dojo/scripts/validate.py` 返回 validation ok；无 【】/TODO 等占位；[C1–C5]/[F1–F3]/[N1–N7] 全部在正文被引用且在来源章节有定义（双向对应）；六个前置概念页（standard-attention、causal-mask、kv-cache、rope、attention-sink、deepseek-v4-1）均真实存在；index 与 overview 双向链接；`alt`/`aria-label` 中无 `$...$`；图内无 `<text>` 公式（公式都在 HTML/foreignObject 外）；`×`、`–`、`→` 属 validate.py 明示豁免的普通排版字符，非未渲染公式。

## 问题

- [轻微·可读性] 第 1 章图 1 图注（index.html 第 148 行）：末句「本图用 6 个位置示意，与实际模型中的 $W=128$ 无关」与图上实际绘制的 10 个位置不符——同一图注前文已列出「位置 0–3 在本层不可见」与「位置 4–9 可见」，合计 10 个方格。此处「6」指窗口大小（图内标签「可见窗口（示意 6 个位置）」与「构造示例」小节的「第 1 章图中 6 个位置的窗口示意」都已明确为窗口口径），但该句主语为「本图」，字面读作本图只用 6 个位置。｜引文依据：不适用（图注文字与图形要素对照）｜修复要求：把该句改成窗口口径，例如「本图的窗口取 6 个位置示意，与实际模型中的 $W=128$ 无关」或「本图用 10 个位置示意，其中窗口 6 个」；「构造示例」小节表述正确，无需改动。｜修复：｜复验：

## 结论

- 处置：可发布（唯一一条轻微为图注措辞，建议顺手修改；不影响正确性与主线理解）
- 统计：阻断 0 / 重要 0 / 轻微 1