# 冲突记录

- 位置：长页「适用条件与边界」的激活显存代价。两处说法：正文 app.tex Memory Footprint 写「For $n=2$, this contributes less than $15\%$」（不到 15%），同节的表 tab:memory_footprint_comparison 写「OLMo-1B-SHC$\times$2 … +15.7\%」「OLMo-1B-DHC$\times$2 … +15.7\%」。页面采用：表 12 的数字，写 $n=2$ 时约多 15.7%。
- 位置：长页「适用条件与边界」的算力增量。两处说法：正文 app.tex Memory Footprint 只说「HC 计算代价低」，未给百分比；同节的表 tab:flops_comparison 给出 $n=4$ 时各模型总 FLOPs 增加 +0.127% 到 +0.208%。页面采用：笼统写「每 token 的算力增加不到百分之一」，不写具体百分比。
- 位置：长页「适用条件与边界」静态超连接的额外参数量。两处说法：app.tex Parameters 正文写「$P_{\texttt{extra}}=4\times (4+2) \times 2 \times 16=768$」（OLMo-1B-SHC$\times$4），同节的表 tab:params_comparison 该行写 HC Params 0.0000077B（约 7700）；表里 +0.00007% 的占比与 768 相符。页面采用：正文公式的 768 及其占比 0.00007%。

