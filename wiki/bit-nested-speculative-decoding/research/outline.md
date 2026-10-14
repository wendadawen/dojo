# Bit-Nested Speculative Decoding

- 描述：位嵌套自投机解码用同一份模型位表示支持草稿和目标：草稿是先猜后续 token 的低开销模型，目标验证这些候选；先量化出 W4 基底，再用残差补成 W8 目标，草稿只读基底，目标读基底加残差。
- 摘要：同一个权重先优化出 W4 基底，再用 4-bit 残差补成 W8 目标；基底和残差分成两个位平面，草稿只读 W4 平面，目标读两个平面；KV cache 按上下文在 KV8 与 KV4/KV8 之间切换。
- 主题：推理系统, 内存与缓存
- 标签：推理加速

## 长页

- 开头｜[投机解码](wiki/speculative-decoding)要草稿便宜又对齐目标，为什么还要一个嵌套的单表示方法？｜420｜paper/src-v2/iclr2027_conference.tex Methodology；paper/src-v2/iclr2027_conference.tex Introduction；paper/src-v2/iclr2027_conference.tex BitNest Weight Construction
- ### 先建基底，再补残差｜常规先建高精度再导草稿，为何反向？其余公式与边界处理折叠于小节｜0｜paper/src-v2/iclr2027_conference.tex BitNest Weight Construction；paper/src-v2/iclr2027_conference.tex Introduction
- #### 旋转与 W4 基底｜[量化基础](wiki/quantization-basics)之外，强 W4 基底怎么造、造完后为什么必须冻结？｜420｜paper/src-v2/iclr2027_conference.tex Detailed Bit-Nested Weight Construction；paper/src-v2/iclr2027_conference.tex BitNest Weight Construction
- #### 残差补成 W8 目标｜4-bit 残差怎样把已冻结的 W4 基底补成 W8？自设小数值例子折叠｜420｜paper/src-v2/iclr2027_conference.tex Detailed Bit-Nested Weight Construction；paper/src-v2/iclr2027_conference.tex BitNest Weight Construction
- ### 双平面存储与 KV 策略｜回答逻辑上少读 4 bit 为何需要专门布局；一个可寻址例子在折叠块｜0｜paper/src-v2/iclr2027_conference.tex Dual-Plane Weight Storage
- #### 基底与残差分成两个位平面｜两个 4-bit 码怎么分别成平面，让草稿只读基底？｜420｜paper/src-v2/iclr2027_conference.tex Dual-Plane Weight Storage
- #### 上下文感知的 KV 读法｜[KV cache](wiki/kv-cache) 在短、长上下文里分别怎么读、何时切换？｜480｜paper/src-v2/iclr2027_conference.tex Context-Aware KV Nesting；paper/src-v2/iclr2027_conference.tex Detailed KV-Cache Nesting
- ### 草稿读取减半，接受率保持高位｜回答收益来自哪里、有多明显；关键实验数字放折叠块｜0｜paper/src-v2/iclr2027_conference.tex Speedup for LLM inference；paper/src-v2/iclr2027_conference.tex Memory Saving
- #### 草稿读取减半，平均接受率 95.2%｜少读一个位平面换来什么，接受率和端到端加速到什么水平？｜420｜paper/src-v2/iclr2027_conference.tex Draft--Target Agreement；paper/src-v2/iclr2027_conference.tex Speedup for LLM inference
- #### 一份权重和受控 KV｜它的存储和长上下文 KV 增长控制到什么结果？｜380｜paper/src-v2/iclr2027_conference.tex Memory Saving；paper/src-v2/iclr2027_conference.tex Edge-Device Evaluation
- ### 适用条件与边界｜它何时有效、哪些收益和保真不能混为一谈、还有哪些限制？｜420｜paper/src-v2/iclr2027_conference.tex Dual-Plane Weight Storage；paper/src-v2/iclr2027_conference.tex Limitation；paper/src-v2/iclr2027_conference.tex Why retain KV8 for the target?；paper/src-v2/iclr2027_conference.tex Why retain W8 for the target?；paper/src-v2/iclr2027_conference.tex Effect of Draft KV Precision

## 短页

- ### 一个 8-bit 表示里的 W4/W8 两个视图｜[投机解码](wiki/speculative-decoding)和 [量化基础](wiki/quantization-basics)前置下，为何有两个视图？｜280｜paper/src-v2/iclr2027_conference.tex Methodology；paper/src-v2/iclr2027_conference.tex BitNest Weight Construction；paper/src-v2/iclr2027_conference.tex Context-Aware KV Nesting
- ### 省草稿存储，也不把目标压到 W4｜它为什么需要：自投机（草稿由目标自身导出）要省草稿存储，又不能让目标停在 W4？｜280｜paper/src-v2/iclr2027_conference.tex Introduction；paper/src-v2/iclr2027_conference.tex Self-Speculative Decoding；paper/src-v2/iclr2027_conference.tex Why retain W8 for the target?
- ### 先建 W4，补残差成 W8，按平面读｜它怎么做：从 W4 基底、残差补全到位平面读取是一条什么机制链？｜340｜paper/src-v2/iclr2027_conference.tex Detailed Bit-Nested Weight Construction；paper/src-v2/iclr2027_conference.tex Dual-Plane Weight Storage；paper/src-v2/iclr2027_conference.tex Context-Aware KV Nesting
