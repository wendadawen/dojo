# 冲突

- 位置：长页「适用条件与边界」MQA 替换哪些注意力层。两处说法：MQA 论文「we replace all of the attention layers in the model to multi-query attention. This includes the encoder-self-attention, decoder-self-attention and encoder-decoder-attention layers.」（paper/src-mqa/main.tex L339）；GQA 论文「We note that GQA is not applied to the encoder self-attention layers; encoder representations are computed in parallel, and memory bandwidth is therefore generally not the primary bottleneck.」（paper/src-gqa/main.tex L140）、「We apply MQ and GQA to decoder self-attention and cross-attention, but not encoder self-attention.」（paper/src-gqa/main.tex L148）。页面采用：MQA 页按 MQA 论文自身的做法写，替换包括编码器自注意力在内的全部注意力层；GQA 不施加在编码器自注意力层，这一处写进 GQA 页。
- 位置：长页「质量与训练稳定性」折叠块表里 1.424、1.439 是什么量。两处说法：正文写「we also list per-subword-token perplexity on the dev set」（paper/src-mqa/main.tex L349），表头列名却是「ln(PPL)」（paper/src-mqa/main.tex L370、L374、L376）。页面采用：按表头写，标为「dev 对数困惑度」。

