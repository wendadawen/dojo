# 审正文 · 第 3 轮

## 事实（content-fact 1）

- 位置标错｜短页·KV cache 随层数线性增长｜「各层各存一份，缓存总量随层数线性增长」不在所标 Introduction 内（Introduction 无「层数」），原话在 Inference Advantages L257｜改：依据补 Inference Advantages。
- 位置标错｜长页开头｜「层数越多，这部分缓存越大」同理在 Introduction 内无原话，在 L257-L258｜改：依据补 Inference Advantages。
- 其余 4 段「位置对、推得出」；数字与专名出处唯一且一致，无未记冲突。

## 事实（content-fact 2）

- 位置标错｜长页·缓存与 prefill 各省下多少＋折叠块第 4 条｜「每层对 N 个位置各做一次固定窗口的注意力」「下段每层是固定窗口的注意力」把高效注意力说成固定窗口；Inference Advantages 只写「efficient attention modules ... usually fast」「constant caches」，没有「固定窗口」；固定窗口只在 Self-Decoder / Sliding-Window Attention，且默认是 gated retention｜改：依据补 Self-Decoder、Sliding-Window Attention，文字改为「常数 cache 的高效注意力」。
- 其余 3 段「位置对、推得出」；无未记的出处冲突。

## 复算（content-recompute）

- 无。复算缓存复杂度约省 L 倍、折叠块第 2 条拆解、第 4 条 prefill 复杂度、$L=4$ 例子、两个量级数字、公式，均一致；$L=4$ 属自行构造。

## 文风（content-style）

- 四类均无。

## 读者（content-reader）

- 未解释：短页「推理」「显存」（现写「GPU 显存」）；长页「KV cache」（「每层各存一份 KV 的代价」首次出现处未解释、未链接）；折叠块记号 $X^l$、$W_Q^l$、$D$；折叠块「GPU」「显存」。
- 读不通｜短页·分成两段，上段复用｜先说全局 KV「与层数无关」，紧接着给含 $L$ 的 $\mathcal{O}((N+L)D)$，$L$ 从哪来没交代。
- 读不通｜长页·缓存与 prefill 各省下多少｜「复用后是 O(LND)」与前文缓存复杂度 O((N+L)D)、基线缓存 O(LND) 同形不同义，不知道这里说的是 prefill 计算量。

## 处理

- 事实 1 两条位置标错：采纳。短页第 2 节、长页开头依据各补 paper/src-v2/main.tex Inference Advantages。
- 事实 2 位置标错（固定窗口）：采纳。两处「固定窗口的注意力」改为「只存常数 cache 的高效注意力（每层计算量随 N 线性增长）」；《缓存与 prefill 各省下多少》与折叠块第 4 条依据各补 Self-Decoder。
- 读者未解释（显存/推理/GPU）：采纳。短页第 2 节改「GPU 显存（跑模型用的内存）」；折叠块「80GB 显存的 GPU（跑模型的加速卡）」。通用释义，不标依据。
- 读者未解释（KV cache）：采纳。《每层各存一份 KV 的代价》首现处改「存下来的 key、value（[KV cache](wiki/kv-cache)）」。
- 读者未解释（折叠块记号）：采纳。折叠块第 2 条补「$D$ 是隐藏维度」，第 3 条补「$X^l$ 是第 $l$ 层的隐状态，$W_Q^l$ 是一个 $D\times D$ 矩阵」。
- 读者读不通（短页 $L$ 来源）：采纳。短页第 3 节括注补「多出的 $L$ 来自下段各层各留一份固定大小的局部 cache」。
- 读者读不通（O(LND) 同形）：采纳。长页《缓存与 prefill 各省下多少》写成「计算量上……prefill 是 O(LN²D)；复用后是 O(LND)，随序列长度线性增长」，与缓存复杂度分开叙述。
- 未处理意见：无。
