# 冲突记录

- 位置：切块的开销（首块 KV 被重复读取的次数）。两处说法：v1/4-design.tex L28 写「the first chunk's KV cache is loaded N times, the second chunk's KV cache is loaded N-1 times」；v2/4-design.tex L35 写「the first chunk's KV-cache is loaded N-1 times, the second chunk's KV-cache is loaded N-2 times」。页面采用：v2 的说法（越靠前的块被后续块重复读取的次数越多，第一块读到 N-1 次）。
- 位置：块的划分口径。两处说法：v1/0-abstract.tex L10 写「splits a prefill request into equal sized chunks」、v1/1-intro.tex L12 写「equal compute-sized chunks」（等大小/等计算量）；v2/0-abstract.tex L9 写「splits a prefill request into near equal sized chunks」（近似等大小）。页面采用：v2 的说法（大小相近的块）。
- 位置：生产提示的长度。两处说法：v1/4-design.tex L14 写「the size of prefill is reasonably large, ranging from 1K -- 4K in production workloads」；v2/4-design.tex L13 写「the median prompt size in sharegpt and arxivsum datasets is 1730 and 7059 respectively」（7059 已超出 4K）。页面采用：v1 的「常在 1K 到 4K 个 token」。
- 位置：每批里的预填充块数。两处说法：v1/0-abstract.tex L10 写「constructs a batch using a single prefill chunk」、v1/4-design.tex L37 写「a single prefill chunk」（每批恰一块）；v2/1-intro.tex L39 写「coalescing all the on-going decodes with one (or more) prefill chunks」、v2/0-abstract.tex L19 写「prefill chunk(s)」（每批可含一块或几块）。页面采用：v1 的说法（每批一块，即页面写「批里有一个预填充块」）。
- 位置：每块 512 个 token 时预填充的端到端损失。两处说法：v1/5.3-eval-ablation.tex L13 写每块 256 与 512 的损失「within 20% and 10% respectively」；v2/5.3-eval-ablation.tex 写同一块长 512 的额外开销「at most ~25%」（两处消融的模型与配置不同）。页面采用：v1 的 20% 与 10%。



