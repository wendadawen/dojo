# 审大纲 · 第 2 轮

## 读者（outline-reader）

- 顺序错乱、多余、短页标题｜无。
- 遗漏关键内容｜适用条件与边界｜上段各层凭什么共用同一份 K、V 还各司其职、这个改造有没有代价；此前节点都只在算缓存与计算的账｜改法：在「复用的是哪一层的 KV」补一句上段各自的 Q 来自本层，在「适用条件与边界」补一句对效果的影响。

## 对照资料（outline-sources）

- 资料里没有（说过头）｜短页·分成两段，上段复用｜「下段各层用固定大小的窗口注意力」把滑动窗口当成下段的固定机制，原文只作高效自注意力的一个例子，实际用 gated retention 或 sliding-window｜main.tex L161「efficient self-attention (e.g., sliding-window attention)」、L280「we use gated retention or sliding-window attention」｜改法：写成「下段用只存常数 cache 的高效自注意力（如固定窗口）」。
- 资料里没有（位置错）｜缓存与 prefill 各省下多少｜三个断言在 Cross-Decoder 内都没有，全在 Inference Advantages；另「适用条件与边界」依据里的 Language Modeling Evaluation 是训练配置与评测结果，不含边界内容｜改法：删该依据或改标复杂度表；边界章依据调整。
- 遗漏关键内容、抬高次要点｜无。

## 篇幅与冗余（outline-size）

- 重复｜短页「多层读同一份 KV」×「分成两段，上段复用」｜同一套机制（分两段、末层投影成一份、上段读它）两节各讲一遍｜改法：保留「分成两段，上段复用」，从「多层读同一份 KV」删去机制复述，只留定义＋KV cache 解释链接。
- 无关、预算不合理、短页不合格｜无。

## 处理

- 短页窗口注意力说过头：采纳。「写成下段用只存常数 cache 的高效自注意力（如固定窗口）」；长页边界章也同样只说「高效注意力（如滑动窗口）」。
- 缓存与 prefill 依据位置错：采纳。删 Cross-Decoder 依据，只留 Inference Advantages。
- 边界章依据 Language Modeling Evaluation 不含边界：采纳。边界章不引实验章，改引 Introduction（脚注「once」的含义）、Self-Decoder（常数 cache 前提）；「省下比例随层数增大」一句删去，避免引 Language Modeling Evaluation。
- 短页「多层读同一份 KV」复述机制：采纳。首节只留定义「上段各层共用下段末层投影出的唯一一份全局 KV」＋KV cache 解释链接，机制步骤留给第 3 节。
- 上段各层的 Q 与效果代价（读者遗漏）：采纳。「复用的是哪一层的 KV」写「上段各层各自的 Q 仍由本层投影得到」；「适用条件与边界」补代价「上段不再有各自的全局 KV，表达能力损失由训练补偿」。
- 简短记录：prefill 退出位置矛盾、描述限定、长序列依据位置三项已裁定，本轮未再提，无需处理。
