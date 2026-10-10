# 量化基础

- 描述：量化基础是把模型里的浮点权重和激活按一个缩放因子映射到低精度整数网格上的方法，用少量的量化误差换取存储、访存和计算的下降。
- 摘要：把权重和激活从浮点压到 8 位或 4 位整数，误差来自网格太粗、舍入和裁剪；缩放因子的取法、粒度和是否重训决定误差能压到多小。
- 主题：训练与优化, 推理系统
- 标签：量化

## 长页

- 开头｜一句话讲量化是什么：把张量里的浮点值换成分辨率更低、位数更少的整数值，取最近的可表示值；顺带用一两句说明权重与激活分别指模型里的什么｜230｜paper/jacob2018-src/main.tex Quantization scheme；paper/gholami2021-src/_s0_chapter.tex Uniform Quantization；paper/gholami2021-src/_s0_chapter.tex Problem Setup and Notations
- ### 量化要解决什么｜模型规模增长使权重与激活的存储、延迟和能耗成为部署瓶颈；量化把每个数从浮点压到 8 位、4 位，讲动机与压缩比｜260｜paper/gholami2021-src/_s0_chapter.tex Introduction；paper/gholami2021-src/_s0_chapter.tex Problem Setup and Notations
- ### 均匀量化：缩放因子与整数网格｜均匀量化的定义，缩放因子把实数范围分成等间距的级、舍入到最近整数、反量化乘回缩放因子；用一两句对照非均匀量化｜0｜paper/gholami2021-src/_s0_chapter.tex Uniform Quantization；paper/gholami2021-src/_s0_chapter.tex Non-Uniform Quantization
- #### 仿射映射与反量化｜实数与整数的仿射关系、缩放因子与零点的含义、从观测范围求二者的算法，以及范围在推理前定还是每个输入现算，手算折叠｜460｜paper/jacob2018-src/main.tex Quantization scheme；paper/jacob2018-src/main.tex Learning quantization ranges；paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/gholami2021-src/_s0_chapter.tex Range Calibration Algorithms: Static vs Dynamic Quantization
- #### 对称量化｜零点取 0 的特例、8 位整数取 $[-127,127]$ 而不取 $-128$ 的原因、缩放因子由最大绝对值定出，以及权重常取对称、输出非负的激活常取非对称｜360｜paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/jacob2018-src/main.tex Learning quantization ranges；paper/jacob2018-src/appendix.tex ARM NEON details
- ### 量化误差从哪来｜三个来源——离散级太粗、舍入、越界值的裁剪；离群值把缩放因子拉大后正常值被舍掉，手算折叠｜340｜paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/jacob2018-src/main.tex Training with simulated quantization
- ### 用粒度把误差限制在局部｜四种粒度各自让一个缩放因子覆盖多少元素，先说明层、通道指模型的什么；越细精度越好、元数据越多｜340｜paper/gholami2021-src/_s0_chapter.tex Quantization Granularity
- ### 训练后量化与量化感知训练｜PTQ 不重训、量化后直接部署；QAT 在训练每步先量化再反量化算前向、反向用直通估计器，代价是重训｜420｜paper/gholami2021-src/_s0_chapter.tex Fine-tuning Methods；paper/gholami2021-src/_s0_chapter.tex Post-Training Quantization；paper/gholami2021-src/_s0_chapter.tex Quantization-Aware Training；paper/jacob2018-src/main.tex Training with simulated quantization
- ### 适用条件与边界｜均匀仿射量化成立的条件、压缩换来的代价、这个概念的边界｜300｜paper/gholami2021-src/_s0_chapter.tex Uniform Quantization；paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/gholami2021-src/_s0_chapter.tex Quantization Granularity

## 短页

- ### 低精度网格上的最近值｜一句话定义量化：把张量里位数更多的浮点值换成位数更少的整数值，取最近的可表示值｜240｜paper/jacob2018-src/main.tex Quantization scheme；paper/gholami2021-src/_s0_chapter.tex Uniform Quantization
- ### 位宽、存储与吞吐｜权重与激活占的存储和访存按位宽下降；专用硬件上低精度计算吞吐更高、能耗更低，代价是精度损失｜280｜paper/gholami2021-src/_s0_chapter.tex Introduction；paper/gholami2021-src/_s0_chapter.tex Problem Setup and Notations；paper/gholami2021-src/_s0_chapter.tex Simulated and Integer-only Quantization
- ### 怎么把浮点换成整数网格｜怎么做：先说清缩放因子与零点，再从观测范围或最大绝对值取缩放因子（对称或非对称），按最近值舍入并裁剪越界值，按粒度决定每个缩放因子的覆盖范围｜290｜paper/jacob2018-src/main.tex Quantization scheme；paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/gholami2021-src/_s0_chapter.tex Quantization Granularity
