# 审大纲 · 第 3 轮

## 读者（outline-reader）

- 顺序错乱｜长页「复用的是哪一层的 KV」及开头、边界节｜「全局 KV」全篇当已知词用，没讲它指什么、与下段那份 KV 的关系；边界那句反而暗示下段还有另一份 KV｜改法：在「复用的是哪一层的 KV」点明全局 KV 覆盖整条序列、与下段只覆盖局部的 KV 相对。
- 顺序错乱｜短页·多层读同一份 KV｜用「上段各层」「下段末层」，但短页到此没说层被分成两段｜改法：定义里先点明把 L 层分成前后两段。
- 遗漏关键内容、多余、短页标题｜无。

## 对照资料（outline-sources）

- 资料里没有（自编因果）｜长页·适用条件与边界｜「上段无各自的全局 KV、由训练补偿」中的「由训练补偿」是自编因果，正文只说仍保留全局注意力、外部行为同 decoder-only｜main.tex L61「retains global attention capability」；L191 只说被复用｜改法：删「由训练补偿」，只留可查的一半。
- 遗漏关键内容、抬高次要点｜无。

## 篇幅与冗余（outline-size）

- 四类均无。核心机制节（600、550）大于背景节（350、300）与边界节（400）；「量级数字折叠」「折叠小例子」「推导折叠」均已标。

## 处理

- 「全局 KV」未界定：采纳。「复用的是哪一层的 KV」写「覆盖整条序列的全局 KV」；边界章写「只存一次只指全局 KV，非全模型 O(1)」，与下段的局部 KV 相对。
- 短页未说分两段：采纳。短页第 1 节定义里先写「把 L 层分成前后两段」。
- 「由训练补偿」自编因果：采纳。删去，边界章代价只留「上段各层不再各自维护一份全局 KV」；依据列删 Inference Advantages、Language Modeling Evaluation，只留 Introduction、Self-Decoder。

## 最后一批改动核对

作者在最后一轮之后做了三处改动，交仲裁核对：有没有引入「必须改」类的新问题、有没有违背前几轮裁定。仲裁结论：**通过**。

- 改动 1（「复用的是哪一层的 KV」改为「唯一一份覆盖整条序列的全局 KV，上段各层共用它、各自 Q 仍由本层算」）：有据。main.tex L185 生成 global KV caches、L191 reused by all the L/2 cross-decoder modules、L194 $\hat Q^l=\mathrm{LN}(X^l)W_Q^l$；repo/yoco/yoco.py CrossDecoder.forward 按整段 seqlen 计 key/value。未与裁定冲突。
- 改动 2（边界章删「由训练补偿」，依据删 Inference Advantages、Language Modeling Evaluation）：删去无出处的一句；剩余断言仍被 Introduction（L112 脚注）、Self-Decoder（L178-L180）、Introduction（L109、L191）覆盖，方向与「只存一次限定为全局 KV」一致。
- 改动 3（短页第 1 节加「把 L 层分成前后两段」）：main.tex L152 直接支撑；未提滑动窗口，不违背「短页不把滑动窗口当成下段固定机制」。
