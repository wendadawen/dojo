<!-- review-meta
round: 4
page: wiki/sliding-window-attention/index.html
reviewed_content_sha256: 32f7cd070f3e10d5
-->
# 滑动窗口注意力审查记录（第 4 轮）

- 页面版本：bdc7869948d29d43fcf90805aeb42e4efe6b9d31（index.html 工作树哈希）
- 审查时间：2026-09-13 19:51
- 审查者：独立子代理（未参与写作与前三轮审查）
- 已完整阅读章节：head（description / dojo:summary / title）、开头 callout、核心问题（4 题及答案）、常见误解、第 1 章（1.1、$-1$ 槽位折叠块、本章问题）、第 2 章（2.1、2.2、2.2 折叠块、2.3、本章问题）、第 3 章（3.1、3.2、3.3、3.3 折叠块、本章问题）、第 4 章（4.1、4.1 代码折叠块、4.2、4.3、本章问题）、来源与范围说明（C/F/N 各条、构造示例、辅助解释与类比边界、简化条件及其限制）、overview.html

## 核对依据（本轮实际打开过的来源）

- Mistral 7B（arXiv:2310.06825）HTML 全文 §2：图注原文「each token can attend to at most W tokens from the previous layer (here, W=3)」；正文「The hidden state in position i of the layer k, h_i, attends to all hidden states from the previous layer with positions between i−W and i.」；「Recursively, h_i can access tokens from the input layer at a distance of up to W×k tokens」；「At each attention layer, information can move forward by W tokens. Hence, after k attention layers, information can move forward by up to k×W tokens.」；Figure 2 图注「The cache has a fixed size of W=4. Keys and values for position i are stored in position i mod W of the cache. When the position i is larger than W, past values in the cache are overwritten.」；「On a sequence length of 32k tokens, this reduces the cache memory usage by 8x.」；Table 1「Model architecture」n_layers=32、window_size=4096；「we have a theoretical attention span of approximately 131K tokens」。均与页面 [C1][C2][N1][N2] 及正文一致。
- StreamingLLM（arXiv:2309.17453）HTML 全文：§1「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV」；Table 1（Llama-2-13B，首本书 65K token）0+1024=5158.07、4+1020=5.40、4"\n"+1020=5.60；Table 2 caption「(1) Window attention (0+y) has a drastic increase in perplexity.」；§3.1「it results in an exceedingly high language modeling perplexity」与「removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function in attention computation.」；Table 3（160M 参数模型，PG19 首样本）Vanilla 2+1022=18.05、Learnable Sink 1+1023=18.01；§3.2「StreamingLLM focuses on positions within the cache rather than those in the original text.」。均与页面 [C3][C5][N3][N4] 及正文一致。
- 页面 4.1 节 Python 代码：本地 python3 实际执行，输出与「预期输出」逐字一致（`W = 8  序列长度 = 20  全程一致` / `槽位序列 : [4, 5, 6, 7, 0, 1, 2, 3]` / `映射回位置 : [12, 13, 14, 15, 16, 17, 18, 19]` / `期望因果窗 : [12, 13, 14, 15, 16, 17, 18, 19]` / `位置 % W : [4, 5, 6, 7, 0, 1, 2, 3]`）。
- 复算：4096²=16 777 216、131072²=17 179 869 184、1048576²=1 099 511 627 776、4096×128=524 288、131072×128=16 777 216、1048576×128=134 217 728、比值 32/1024/8192；128×512×1 B=65 536 B=64 KiB、40×64 KiB=2560 KiB=2.50 MiB、3×64 KiB=0.19 MiB、合计 2.69 MiB、1048576×512×1 B=512 MiB、40×512 MiB=20 GiB、5120/1048576=0.488%≈0.49%、999 999−5120=994 879。全部与页面一致。
- `.dojo/scripts/validate.py wiki/sliding-window-attention/index.html` → `validation ok`。
- 页面内 11 个本地链接（standard-attention、causal-mask、kv-cache、rope、attention-sink、deepseek-v4-1、index.html、4 个 libs）目标文件均存在；无「（待生成）」占位。
- DeepSeek-V4.1-Flash：仓库内可读的官方材料只剩 `wiki/deepseek-v4-1/research/official/README.md` 与 `official/inference/README.md`（见下条）。

## 问题

- [阻断·技术] 来源与范围说明 [N5]/[N6]/[N7] 与开篇「主要依据」把复算与实测的证据指向仓库中已不存在的文件，本轮无法按 check.md 2.2 第 1 步「打开来源，定位到页面标注的位置」核对｜引文依据：行 450「复算脚本：`research/verify_swa_accounts.py`」；行 452「复算脚本同上；官方函数对照存档见 `wiki/deepseek-v4-1/research/ckpt/verify_sparse_attn_window.out`」；行 62「字节账与索引规则的复算脚本见文末来源说明」。实查：`ls wiki/sliding-window-attention/research/verify_swa_accounts.py` → No such file or directory；`ls wiki/deepseek-v4-1/research/ckpt` → No such file or directory；`ls wiki/deepseek-v4-1/research/official/inference/` → 只有 README.md（[C4] 引的 `official/inference/config.json`、[C5] 引的 `official/inference/model.py` 同样不在仓库，全仓库 `find -name config.json`、`find -name model.py` 均为空）。可佐证的部分仅限 README：`official/README.md`「a 40-layer Transformer」「support for contexts of up to one million tokens」「SWA Bounded Replay」「FP4 main KV caching」「DSpark speculative decoding」，与页面的 40 层、1M、FP4 全局 KV、DSpark 相符；但 `window_size=128`、`head_dim=512`、DSpark 草稿层各注册一份窗口缓存这几个具体键值在仓库内无对应来源可核对｜修复要求：把 [N5]/[N6]/[N7] 的证据引用改为当前可定位的记录（例如本页 `research/measured.md` 中登记该项，或删去指向已移除脚本的路径）；[C4]/[C5] 若继续引用 config.json / model.py，须标明其为官方发布物（非本仓库镜像），并把无法在仓库内复核的具体键值（window_size、head_dim）保留在明确标注为「按官方配置，未在本仓库复核」的口径下。完成后行 450/452 与行 62 不得再出现指向不存在文件的 <code> 路径。｜修复：｜复验：

- [轻微·表述] 正文存在元话语，与 check.md 2.2 第 12 项列举的「要注意/需要注意」式提示语与章节自述同型｜引文依据：不适用｜位置与原文：1 章首段「先明确一个 query 到底能看到什么。」（行 115）；1 章次段「要注意窗口不是「把文本切成固定段」。」（行 117）；2.2 折叠块「注意这里只算窗口 KV，不含另一路压缩全局 KV 与索引器缓存」（行 238）；3 章首段「这一章算清这个接力能走多远，以及它什么时候不够用。」（行 265）；4 章首段「前两章把「缓存固定为 $W$ 条」当作结论使用，这一章说明它怎么实现，以及实现中最容易出错的两处：索引与位置。」（行 343）；3 章本章问题解答「注意这是「上限」而非「保真范围」」（行 329）｜修复要求：把上述句子改成直接陈述命题的写法——行 117 改为把「窗口不是截断」写成正面判断（如「窗口随位置滑动：位置 4999 的窗口是 4872–4999，位置 5000 的窗口是 4873–5000」），去掉「要注意」；行 115 去掉「先明确……什么。」，直接以「全注意力里……」起句；行 265/343 去掉「这一章算清/说明」的章节自述，保留「前两章已得结论→本章待解决问题」的实质过渡；行 238/329 去掉「注意」引导词，把限定条件写成陈述句。修复后全文不得再出现「要注意」「先明确一个 query 到底能看到什么」「这一章算清」「这一章说明」的字样。｜修复：｜复验：

- [轻微·技术] 2.3 节在已核对的来源句之后追加了一句机制解释，把窗口 KV 写成全局 KV 的「副本」，措辞可能让读者以为两路 KV 存在复制关系，而同一页 1.1 节又说明两路槽位是拼进同一次调用的两条独立路径｜引文依据：页面行 243「窗口 KV 是最近位置保留 FP8 的较高精度副本（比全局 KV 的 FP4 少一道压缩）」；对照行 155「窗口槽位与另一路「压缩全局 KV」的槽位拼进同一次注意力调用[C4]」；技术报告口径（经 [C3] 同源引用核对）为「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」，只说明保留 FP8 及其依据，未把窗口 KV 描述为压缩全局 KV 的副本｜修复要求：删去「副本」这一表述，改为与「两路独立」一致的写法（如「窗口 KV 直接保存最近 $W$ 个位置的键值并以 FP8 存储，不经压缩全局 KV 的池化与 FP4 量化这两步」），保留「窗口 KV 对量化更敏感」这一有来源的依据。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 2
- 处置：修复（阻断项为证据引用指向已移除文件，须先关闭）

### 本轮已核对通过、不作为问题的项（供复验参考）

- 全部外部来源引文（[C1][C2][C3][C5] 及 [N1]–[N4]）逐条与 arXiv 原文比对一致，含 Mistral 图注「each token can attend to at most W tokens from the previous layer」与页面行 181 的引用一致；未发现数字与官方材料不符。
- 全部可复算数值（打分次数三行、倍数、64 KiB / 2.50 MiB / 0.19 MiB / 2.69 MiB / 512 MiB / 20 GiB / 1-8192、5120 与 0.49%、999 999−5120=994 879）重算无误，分项之和与合计一致。
- 公式 $n_{\text{score}}=N\cdot W$、$\text{span}(k)=k\times W$、$\text{slot}(i)=i\bmod W$ 符号全文单义（$N$ 序列长度、$W$ 窗口、$i$ 位置、$k$ 层数、$r$ 压缩比），$\times$ 等符号经 style-guide 第 11 节与 validate.py 明确列为不列入的普通排版字符，不构成 Unicode 数学字符违规。
- 4.1 节代码实际执行，输出与「预期输出」完全一致。
- 正文无第一人称/第二人称（我、我们、你）；在来源与范围说明中的「本页」符合 style-guide 第 12 节「自称使用「本页」或「本文」」，不作为问题。
- 页面级核心问题 4 题、四个章节的本章问题各 2 题均有解答折叠块，答案与正文结论一致，核心问题答案均指明完整论证所在章节。
- head 中 description（纯文本）、dojo:summary、dojo:type=concept、dojo:topics、dojo:tag 齐备，validate.py 通过；overview.html 与 index.html 相互链接。
