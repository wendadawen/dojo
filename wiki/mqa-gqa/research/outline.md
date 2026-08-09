# MQA 与 GQA 教学大纲

## 1. 页面开头

钩子：让读者想象一个 70B 模型、32 层、128 头、序列 4096 的推理场景——单条请求的 KV cache 就要数 GB，每生成一个 token 都要把这数 GB 从显存搬一遍。问题：推理慢是因为算力不够，还是因为 GPU 在等数据？答案是后者——瓶颈在内存带宽，不在算力。

一句话定义：MQA（Shazeer 2019）让所有 query 头共享一组 K/V 把 KV cache 减到 1/h，GQA（Ainslie 2023）把 query 头分 G 组、每组共享一组 K/V，是 MHA 与 MQA 之间的插值；两者都通过减少 K/V 头数来压缩自回归推理时的 KV cache。

要解决的具体问题：自回归推理时 KV cache 随头数与序列长度线性增长，成为内存带宽与显存瓶颈；MQA 减得太狠有质量代价，GQA 找到折中点。

学习承诺（与 scope.md Q1–Q5 一致）：读完你能回答——为什么 MHA 推理受内存带宽限制；MQA 如何共享 K/V、代价是什么；GQA 如何插值与 uptraining；手算 4 头 MHA / 2 组 GQA / 1 组 MQA 的 cache；MQA/GQA 与 MLA 的根本区别。

首个具体场景：4 头 MHA 的 KV cache 手算（贯穿例子第一次出现，固定 $h=4, d_k=64$）。

与第一章的过渡：先看清瓶颈在哪，才能看清 MQA 在减什么。

页面开头组件选择：05 blockquote.meta（主要依据）、03 learning-goals（读完你能回答）、02 context-box（阅读上下文）、04 misconceptions（最容易误解，4 条）。

## 2. 章节设计

### S1 为什么 MHA 推理受内存带宽限制——KV cache 如何随头数与序列长度增长

- 主要教学问题：自回归推理时 MHA 的瓶颈在哪、为什么是带宽而非算力？
- 对应范围：Q1；C1、C2、F1、F5、N1。
- 正文要点：
  - 训练 vs 推理的对比：训练时全序列并行、KV 一次性算出；自回归推理逐 token 生成、每步都要把之前所有 token 的 K/V 加载一遍。
  - KV cache 的来源：每 token 每 layer cache $2 h d_k$（h 个 K 头 + h 个 V 头各 $d_k$ 维），$n$ 个 token、$l$ 层为 $2 h d_k n l$。引用前置页 standard-attention 的多头投影公式。
  - 手算贯穿例子第一次出现：$h=4, d_k=64, l=1, n=10$ → MHA 每 token cache = $2\times 4\times 64=512$ 元素，10 token = 5120 元素；fp16（2 字节）= 10240 字节。
  - Shazeer 2019 §2.4.1 性能分析：内存访问/算术比值 $\Theta(n/d + 1/b)$；n 接近 d 或 b 小时比值接近 1，GPU 在等内存。
  - GPU 算力增长快于带宽增长的剪刀差（辅助，简述）——说明瓶颈随硬件演进越来越严重。
- 讲解材料及职责：
  - ASCII 图示：训练并行 vs 推理串行的对比流程，展示推理每步都要加载整个 KV cache。
  - 对照表格：KV cache 元素数公式 $2 h d_k n l$ 的拆解（h、$d_k$、n、l 各贡献什么）。
  - 数字例子（贯穿）：$h=4,d_k=64,l=1,n=10$ 的 MHA cache 手算。
- 前置知识安排：引用 standard-attention 概念页链接（多头投影 $Q=XW^Q,K=XW^K,V=XW^V$ 与多头公式）。
- 完成检查：说出 KV cache 公式 $2 h d_k n l$ 与每项含义；说出推理瓶颈在带宽不在算力；算出 4 头 $d_k=64$ 10 token 1 层的 cache 元素数。
- 过渡：瓶颈定位在"每头一份 K 和 V 的全量加载"，那能不能少存几份？下一章看 MQA 的极端做法。

### S2 MQA——所有 query 头共享一组 K/V，cache 减到 1/h

- 主要教学问题：MQA 改了什么、减了多少、代价是什么？
- 对应范围：Q2；C3、C4、C5、F2、N1。
- 正文要点：
  - MQA 的定义：$P^Q,P^O$ 保持 $\mathbb{R}^{h\times d\times d_k}$ 多头，$P^K,P^V$ 变为 $\mathbb{R}^{d\times d_k}$、$\mathbb{R}^{d\times d_v}$ 全模型一组；所有 query 头读同一组 K 和 V。
  - 张量形状对照表：MHA 的 $P^K\in\mathbb{R}^{h\times d\times d_k}$ vs MQA 的 $P^K\in\mathbb{R}^{d\times d_k}$——去掉头维度 h。
  - KV cache 公式：$2 d_k$ 每 token 每 layer，减少 h 倍。
  - 手算贯穿例子第二次出现：同一 $h=4,d_k=64,l=1,n=10$ 下 MQA cache = $2\times 1\times 64=128$ 元素/token，10 token = 1280 元素，fp16 = 2560 字节；与 S1 的 MHA 5120 元素对照，减少 4 倍。
  - Shazeer 2019 §3.1 的分析结论："reduced the offensive n/d by a factor of h"——把性能比值的 n/d 项系数减为 1/h。
  - 代价：质量轻微下降（Shazeer 2019 abstract "minor quality degradation"）；大模型训练不稳定（Ainslie 2023 Appendix A）。
- 讲解材料及职责：
  - ASCII 图示：MHA（h 组独立 K/V）vs MQA（1 组共享 K/V，h 个 query 头都指向它）。
  - 对照表格：MHA vs MQA 的张量形状与 cache 公式。
  - 数字例子（贯穿）：MQA 的 cache 手算与 MHA 对照。
- 前置知识安排：依赖 S1 的 KV cache 来源。
- 完成检查：写出 MQA 的 $P^K$ 形状（$\mathbb{R}^{d\times d_k}$，无头维度）；算出 4 头 $d_k=64$ 下 MQA 的 cache（128 元素/token）；说出代价（质量下降、训练不稳定）。
- 过渡：MQA 减得太狠（h 倍）有质量代价。能不能少减一点、找个中间点？下一章看 GQA。

### S3 GQA——在 MHA 与 MQA 之间插值，G=8 是折中点

- 主要教学问题：GQA 如何插值、uptraining 如何从已有 MHA 检查点得到 GQA、实验上 G=8 为什么好？
- 对应范围：Q3；C6、C7、C8、F3、N2、N3、N4。
- 正文要点：
  - GQA 的定义：query 头分 G 组、每组共享一组 K/V；$P^K,P^V\in\mathbb{R}^{G\times d\times d_k}$。$G=h$ 等价 MHA，$G=1$ 等价 MQA，$1<G<h$ 为插值。
  - KV cache 公式：$2 G d_k$ 每 token 每 layer。
  - 手算贯穿例子第三次出现：同一 $h=4,d_k=64,l=1,n=10$ 下 GQA-2 cache = $2\times 2\times 64=256$ 元素/token，10 token = 2560 元素，fp16 = 5120 字节；介于 MHA 5120 与 MQA 2560 之间（注意 fp16 字节数翻倍）。
  - uptraining 两步：（a）把 MHA 检查点里 h 个 K/V 投影矩阵按组均值池化为 G 个（Ainslie 2023 §2.1 指出优于选第一个或随机初始化，Figure 4 消融）；（b）用原预训练计算量的 α=5% 继续预训练（§3.1，Figure 5 显示 5% 已显著提升、10% 后递减）。
  - Ainslie 2023 Table 1 的 T5-XXL 实验：MHA-XXL 1.51s/47.2、MQA-XXL 0.24s/46.6、GQA-8-XXL 0.28s/47.1——GQA-8 质量接近 MHA、速度接近 MQA。
  - G=8 是 Figure 6 消融后选定的折中点（§4 "We selected 8 groups as a favorable middle ground"）。
  - GQA 的额外好处：训练稳定（Ainslie 2023 Appendix A，与 MQA 对照）。
- 讲解材料及职责：
  - ASCII 图示：MHA（h 组）→ GQA（G 组）→ MQA（1 组）的连续谱，展示 G 是旋钮。
  - 对照表格：MHA/GQA/MQA 的 K/V 头数、cache 公式、质量、速度对照（含 N2 实验数据）。
  - 数字例子（贯穿）：GQA-2 的 cache 手算与 MHA/MQA 三者对照。
- 前置知识安排：依赖 S1（KV cache）、S2（MQA 极端）。
- 完成检查：写出 GQA 的 cache 公式 $2 G d_k$；说出 GQA-1=MQA、GQA-h=MHA；说出 uptraining 两步（均值池化 + 5% 继续预训练）；说出 GQA-8 在 T5-XXL 上的质量与速度结论。
- 过渡：手算例子已经分别算过 MHA/GQA/MQA，但分散在三处。下一章把它们并排放一起，看清连续谱。

### S4 手算对比——4 头 MHA / 2 组 GQA / 1 组 MQA 的 KV cache 连续谱

- 主要教学问题：把三种机制并排手算，看清 MHA→GQA→MQA 是 K/V 头数从 h 到 1 的连续谱。
- 对应范围：Q4；F1、F2、F3。
- 正文要点：
  - 固定 $h=4, d_k=64, l=1, n=10$ 的教学数字，并排算三种机制：
    - MHA（4 个 K/V 头）：每 token = $2\times 4\times 64=512$，10 token = 5120 元素，fp16 = 10240 字节。
    - GQA-2（2 个 K/V 组）：每 token = $2\times 2\times 64=256$，10 token = 2560 元素，fp16 = 5120 字节。
    - MQA（1 个 K/V）：每 token = $2\times 1\times 64=128$，10 token = 1280 元素，fp16 = 2560 字节。
  - 连续谱关系：K/V 头数从 h=4 到 G=2 到 1，每减一个 K/V 头 cache 减 $2 d_k=128$ 元素；GQA 的 G 是谱上的旋钮，$G=h$ 端是 MHA、$G=1$ 端是 MQA。
  - 推广到真实规模：$h=128, d_k=128, l=80, n=4096$ 时 MHA cache 约 10.7 GB（fp16），MQA 约 84 MB——说明大模型上收益巨大。
- 讲解材料及职责：
  - 对照表格：三种机制的 K/V 头数、每 token cache、10 token cache、fp16 字节数并排。
  - 数字例子（贯穿汇总）：把 S1–S3 的三次手算合并成一张表。
  - ASCII 图示：连续谱数轴，MHA 在左端（G=h）、MQA 在右端（G=1）、GQA 在中间。
- 前置知识安排：依赖 S1–S3。
- 完成检查：算出三种机制在 $h=4,d_k=64,l=1,n=10$ 下的 cache 元素数（512/256/128 每 token）；说出 G 是谱上旋钮、两端分别是 MHA 与 MQA；说出大模型规模下收益量级（h=128 时减少 128 倍）。
- 过渡：连续谱清楚了。但这是"共享头"这一条路。还有另一条路——压缩 K/V 本身。下一章点出区别，作为 MLA 的前置。

### S5 边界与后续——MQA/GQA 不解决什么，与 MLA 的根本区别

- 主要教学问题：MQA/GQA 的适用边界在哪、与 MLA 的机制区别是什么、为什么说是 MLA 的前置？
- 对应范围：Q5；C9、F4、N5。
- 正文要点：
  - 适用边界：MQA/GQA 减的是 KV cache 的头数系数，不解决注意力本身的 $O(n^2)$ 算力；每个 query 头仍对全部前序 token 做 softmax 加权和。
  - 与 MLA 的根本区别（DeepSeek-V2 §2.1.4 Table 1 对照）：
    - MQA/GQA：共享 K/V 头，被缓存的每个 K/V 仍是完整的 $d_k$ 维 head 向量、只是份数减少；cache = $2 G d_k l$（G=1 为 MQA）。
    - MLA：不共享头，把所有头的 K/V 联合压成 $d_c$ 维潜向量 $c_t^{KV}$，推理时再用上投影重建各头 K/V；cache = $(d_c + d_h^R) l$。
    - 机制不同：共享头是"少存几份完整 head"，低秩压缩是"把所有 head 信息压到一个低维潜向量"。
  - 为什么是 MLA 前置：理解了"KV cache 来自每头一份 K 和 V"才能理解 MLA 在压缩什么；理解了"共享头有质量代价"才能理解 MLA 为什么另起一条路。
  - DeepSeek-V2 §2.1.4 Table 1 的四种机制 cache 公式对照（MHA/GQA/MQA/MLA 并排）。
  - MLA 页链接（已生成 wiki/mla/index.html），点出本页是其前置。
- 讲解材料及职责：
  - 对照表格：四种机制（MHA/GQA/MQA/MLA）的 K/V 头数、cache 公式、机制类别（共享头 vs 低秩压缩）并排。
- 前置知识安排：依赖 S1–S4；MLA 页链接（只引对照、不展开 MLA 机制）。
- 完成检查：说出 MQA/GQA 不解决 $O(n^2)$ 算力；说出共享头与低秩压缩的机制区别；说出四种机制的 cache 公式。
- 过渡：本页到此结束。MLA 的完整机制见 MLA 概念页。

## 3. 讲解顺序

按 S1→S2→S3→S4→S5。先讲为什么需要（S1 瓶颈），再讲极端做法（S2 MQA），再讲折中（S3 GQA），再把三者并排手算（S4 连续谱），最后点边界与 MLA 区别（S5）。一次只引入一个新变量：S1 引入 KV cache 与带宽瓶颈，S2 引入"共享 K/V"，S3 引入"分组插值"与 uptraining，S4 不引入新机制只汇总，S5 引入 MLA 对照。

## 4. 贯穿例子

固定 $h=4, d_k=64, l=1, n=10$ 的教学数字（人为构造，便于手算，不代表真实模型）。在 S1 第一次出现算 MHA cache=512 元素/token；S2 复用算 MQA cache=128；S3 复用算 GQA-2 cache=256；S4 把三者并排成连续谱表，并推广到真实规模 $h=128,d_k=128,l=80,n=4096$ 看量级。每次复用只增加一个新层次（S2 加共享、S3 加分组、S4 加推广）。

## 5. 讲解材料职责

- 公式：F1–F5 表达 cache 与性能比值的变量关系。
- 数字例子：4 头 MHA / 2 组 GQA / 1 组 MQA 的手算，展示代入与中间结果（教学构造）。
- 图示：训练 vs 推理流程、MHA/MQA/GQA 的 K/V 头共享结构、连续谱数轴。
- 对照表格：张量形状对照、cache 公式对照、四种机制对照。
- 代码：本页无可运行代码——机制讲解不需要代码验证，cache 公式手算即可验证。

## 6. 正文与折叠块分工

必须放正文：KV cache 公式与每项含义、MQA/GQA 的定义与张量形状、性能比值 $\Theta(n/d+1/b)$、uptraining 两步、T5-XXL 实验数据、4 头手算结果、MHA→GQA→MQA 连续谱关系、与 MLA 的机制区别。

可放折叠块：Shazeer 2019 性能比值的完整推导背景、uptraining 三种转换方法的消融细节（Figure 4）、α 影响的消融细节（Figure 5）、GQA 训练稳定性的 Appendix A 细节、推广到真实规模的额外数字。

折叠块全部收起时正文仍须回答全部学习目标——所有核心公式、定义、机制区别、手算结果都在正文。

## 7. 范围与证据约束

大纲只使用 scope.md 已纳入范围的内容。不展开 MLA 机制（仅引 cache 公式对照）、不展开 Flash/Linear Attention、不展开 kernel 实现。所有事实附 C/F/N 编号，与 evidence.md 一致。Shazeer 2019 的具体速度倍数因 PDF 一手数据未取到精确值，不引用二手数字（N1 标注为"未核实"不在正文引用具体倍数）。
