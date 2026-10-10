# FP8 块量化

- 描述：FP8 块量化把权重和激活按固定大小分组，每组各自算一个缩放因子，元素缩放后存成 8 位浮点 E4M3，反量化时乘回该组的缩放因子。
- 摘要：8 位浮点 E4M3 能表示的正规数最大只有 $448$，动态范围窄。块量化把张量切成 $128\times128$（权重）或 $1\times128$（激活）的小块，每块用自己的 $s=\max|W|/448$ 缩放，组内元素共享指数位，再存成 E4M3；逐张量缩放则会被少数离群值主导，量化精度大幅下降。
- 主题：训练与优化
- 标签：量化

## 长页

- 开头｜总览：是什么、为什么、怎么工作；一句说明张量是多维数组、权重是模型参数、激活是流动的数值；8 位浮点存储省显存，块指分组各配一个缩放因子｜520｜paper/fp8-formats-2209.05433v2.txt L127-L128；paper/deepseek-v3-2412.19437v2.txt L862；paper/deepseek-v3-2412.19437v2.txt L878-L879
- ### E4M3：可表示范围与特殊值取舍｜E4M3 是 8 位浮点：1 符号、4 指数、3 尾数，偏置 7；最大正规数 448、最小 2^-6、次正规 2^-9；不表示无穷、只留一个 NaN；手算折叠｜520｜paper/fp8-formats-2209.05433v2.txt L127；paper/fp8-formats-2209.05433v2.txt L137-L142；paper/fp8-formats-2209.05433v2.txt L153-L155；paper/fp8-formats-2209.05433v2.txt L157-L167；paper/fp8-formats-2209.05433v2.txt L173
- ### 缩放：把张量对准可表示区间｜动态范围窄，直接转换会溢出或下溢，先乘缩放因子搬进可表示区间；逐张量缩放对离群值敏感、精度大幅下降；离群值拉大缩放因子的演示折叠｜620｜paper/fp8-formats-2209.05433v2.txt L121；paper/deepseek-v3-2412.19437v2.txt L899-L900
- ### 分块：权重 128×128、激活 1×128｜权重与激活按 GEMM 的 K 维分块，各一个缩放因子；每块在线取 max 定 s；存 round(x/s) 为 E4M3，反量化乘回；roundtrip 折叠｜900｜paper/deepseek-v3-2412.19437v2.txt L902；paper/deepseek-v3-2412.19437v2.txt L906-L907；paper/deepseek-v3-2412.19437v2.txt L935；paper/deepseek-v3-2412.19437v2.txt L940-L941；paper/fp8-formats-2209.05433v2.txt L128；repo/deepseek-v3/inference/kernel.py L26-L35；repo/deepseek-v3/inference/kernel.py L83-L86；repo/deepseek-v3/hf-config.json L1-L60
- ### 配套：E4M3 全用与累加精度｜惯例梯度用 E5M2、这里 E4M3 全用，理由块内共享指数位；反量化在累加中完成、按块升 FP32；14 位与 K=4096 误差折叠｜520｜paper/deepseek-v3-2412.19437v2.txt L933-L935；paper/fp8-formats-2209.05433v2.txt L128；paper/deepseek-v3-2412.19437v2.txt L916-L918；paper/deepseek-v3-2412.19437v2.txt L924-L929
- ### 适用条件与边界｜成立前提（块内最大值有代表性、元素按块聚集）、代价（每块另存缩放因子、反量化开销）、常见误解（当逐张量缩放）；激活梯度按块量化发散反例折叠｜460｜paper/deepseek-v3-2412.19437v2.txt L3702-L3707；paper/deepseek-v3-2412.19437v2.txt L902-L903；repo/deepseek-v3/inference/kernel.py L52-L54；paper/deepseek-v3-2412.19437v2.txt L863

## 短页

- ### 把张量切成小块，每块一个缩放因子｜是什么：张量是多维数组；元素按固定大小分组，每组各算一个缩放因子，缩放后存成 8 位浮点 E4M3（1 符号、4 指数、3 尾数）｜320｜paper/deepseek-v3-2412.19437v2.txt L902；paper/fp8-formats-2209.05433v2.txt L127-L128
- ### 8 位浮点装不下一个张量的全部动态范围｜为什么：E4M3 最大正规数只有 448，逐张量缩放时少数离群值主导缩放因子，量化精度大幅下降｜320｜paper/fp8-formats-2209.05433v2.txt L153-L155；paper/deepseek-v3-2412.19437v2.txt L899-L900
- ### 每块取最大绝对值定缩放，再缩放存数｜怎么做：每块运行时取 max|x|，令 s=max|x|/448，存 round(x/s) 为 E4M3，用的时候乘回 s｜320｜paper/deepseek-v3-2412.19437v2.txt L940-L941；repo/deepseek-v3/inference/kernel.py L26-L35；repo/deepseek-v3/inference/kernel.py L83-L86
