# 冲突

- 位置：长页「适用条件与边界」是否替换编码器自注意力。两处说法：GQA 论文「We note that GQA is not applied to the encoder self-attention layers」（paper/src-gqa/main.tex L140）；MQA 论文「we replace all of the attention layers in the model to multi-query attention. This includes the encoder-self-attention, decoder-self-attention and encoder-decoder-attention layers.」（paper/src-mqa/main.tex L339）。页面采用：分两句写，GQA 不施加在编码器自注意力层，MQA 替换了包括编码器自注意力在内的全部注意力层。
- 位置：长页「质量与速度的取舍」每样本推理时间的单位。两处说法：主结果表的列头标秒（paper/src-gqa/tables/main_results.tex L8「s」）；性能-时间图横轴标毫秒（paper/src-gqa/figures/results.tex L10「Time per sample (ms)」）。页面采用：秒，与时间-组数图的纵轴单位一致（paper/src-gqa/figures/time_vs_groups.tex L9「Time per sample (s)」）。
- 位置：长页「质量与速度的取舍」同一个 XXL 规模的两套推理时间。两处说法：主结果表给多头 XXL 1.51 秒、MQA-XXL 0.24 秒（paper/src-gqa/tables/main_results.tex L11-L12）；时间-组数图给多头 2.531 秒、MQA 0.489 秒（paper/src-gqa/figures/time_vs_groups.tex L29、L38、L42-L43）。页面采用：两套都写，并加一句说明两者是在不同输入、输出长度下测的。
