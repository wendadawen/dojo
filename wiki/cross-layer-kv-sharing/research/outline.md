# 跨层 KV 复用

- 描述：跨层 KV 复用让多层共用同一份全局 KV：上段各层不再各自计算全局 KV，而是通过交叉注意力读取下段末层投影出的唯一一份，把全局 KV 的层维度消掉。
- 摘要：前一半层算出唯一一份全局 KV $\hat K=\mathrm{LN}(X^{L/2})W_K$、$\hat V=\mathrm{LN}(X^{L/2})W_V$，后一半层各层复用它；缓存复杂度从 $\mathcal{O}(LND)$ 降到 $\mathcal{O}((N+L)D)$，prefill 还能提前退出一半的层。
- 主题：注意力机制, 内存与缓存
- 标签：KV cache

## 长页

- 开头｜它是什么、为什么需要、大概怎么工作：把 L 层分两段，上段各层复用下段算出的唯一一份全局 KV，只做概览｜350｜paper/src-v2/main.tex L60-L61；paper/src-v2/main.tex Introduction
- ### 每层各存一份 KV 的代价｜标准注意力每层各自缓存 K、V（解释并链接），随层数线性增长，长序列下成显存与 prefill 瓶颈；量级数字折叠｜300｜paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Inference Advantages
- ### 复用的是哪一层的 KV｜把 L 层分成下段自解码器与上段交叉解码器；下段末层输出投影成唯一一份覆盖整条序列的全局 KV，上段各层共用它、各自 Q 仍由本层算；含一个可跟的折叠小例子｜600｜paper/src-v2/main.tex You Only Cache Once (YOCO)；paper/src-v2/main.tex Cross-Decoder；repo/yoco/yoco/models/decoder/yoco.py L207-L235
- ### 缓存与 prefill 各省下多少｜复杂度从 O(LND) 降到 O((N+L)D)；prefill 提前退出一半层，attention 从 O(LN²D) 降到 O(LND)；推导折叠｜550｜paper/src-v2/main.tex Inference Advantages
- ### 适用条件与边界｜范围：只存一次只指全局 KV，非全模型 O(1)；条件：下段用常数 cache 的高效注意力；代价：上段各层不再各自维护一份全局 KV；两条误解各一句｜400｜paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Self-Decoder

## 短页

- ### 多层读同一份 KV｜是什么：把 L 层分成前后两段，上段各层共用下段末层投影出的唯一一份全局 KV，只给定义不展开步骤；先解释 KV cache 并链接｜250｜paper/src-v2/main.tex L60-L61；paper/src-v2/main.tex Cross-Decoder
- ### KV cache 随层数线性增长｜为什么需要：标准 Transformer 每层各存一份全局 KV，长序列下显存随层数增长，成为推理瓶颈｜250｜paper/src-v2/main.tex Introduction
- ### 分成两段，上段复用｜怎么做：下段各层用只存常数 cache 的高效自注意力（如固定窗口），末层输出投影成一份全局 KV，上段各层读它、不再自算，缓存降到 O((N+L)D)｜300｜paper/src-v2/main.tex Cross-Decoder；paper/src-v2/main.tex Inference Advantages
