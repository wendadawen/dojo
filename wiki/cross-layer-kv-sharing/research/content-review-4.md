# 审正文 · 第 4 轮

## 事实（content-fact 1）

- 推不出（漏条件）｜长页折叠块「长序列下的两个量级」第 1、2 条｜两条数字示例丢掉了原话里的配置条件，写成无条件的具体量级｜main.tex L101「a 65B-size language model (augmented with grouped-query attention and 8-bit KV quantization)」、L103「the 7B language model (augmented with Flash-Decoding and kernel fusion)」｜改：把条件写回。
- 其余 7 段「位置对、推得出」；数字与专名出处唯一，无未记冲突。

## 事实（content-fact 2）

- 无。3 段「位置对、推得出」；无未记的出处冲突。

## 复算（content-recompute）

- 无。复算全部公式与推导（$L/2$ 划分、$L=4$ 例、缓存复杂度比、prefill 复杂度比、少算一半层、86GB/80GB、110s/380s、折叠块合成）均一致；$L=4$ 属自行构造。

## 文风（content-style）

- 口语｜短页 L22、长页 L42｜口语词「跑」（GPU 显存（跑模型用的内存）、GPU（跑模型的加速卡））｜改：改「运行」。
- 重复｜长页 L49 × L63｜「全局 KV 与层数无关」两处｜改：删 L63 的该分句。
- 离题（量前置）｜短页 L22｜一段内解释 token、GPU 显存、prefill 三个基线外词｜改：少解释一个。

## 读者（content-reader）

- 未解释：短页、长页「高效注意力」首次出现无定义无链接。
- 前后矛盾｜长页·缓存与 prefill 各省下多少 × 「复用的是哪一层的 KV」｜prefill 说上段不必再算，与上段每层要用本层隐状态算 query、做注意力冲突；「复用后 O(LND)」又用了整层数 L。
- 前后矛盾｜短页·分成两段，上段复用｜同段先说上段各层算 query、做交叉注意力，又说 prefill 时上段不必再算。

## 处理

- 事实 1 漏条件：采纳。折叠块两条补回配置：第 1 条补「另配分组查询注意力（多个查询头共用一组 key/value 头）与 8-bit 的 KV 量化」；第 2 条补「另用 Flash-Decoding 与 kernel fusion（两种推理加速手段）」。
- 文风口语「跑」：采纳，两处改「运行」。
- 文风重复（与层数无关）：采纳，删《缓存与 prefill 各省下多少》里该分句。
- 文风量前置：采纳，短页第 2 节只保留 prefill 一个括注。
- 读者未解释（高效注意力）：采纳。短页第 3 节改「只存固定大小 cache 的高效注意力（如[滑动窗口注意力](wiki/sliding-window-attention)）」给予链接；长页《缓存与 prefill 各省下多少》与边界章并作一处「下段的注意力不随序列长度变长」，只在此处提概念一次，不再用未解释的呼名。
- 读者前后矛盾（prefill 与上段计算）：采纳。《缓存与 prefill 各省下多少》写明「prefill 时全局 KV 由下段算出，下段算完即可退出、不必进上段」；《复用的是哪一层的 KV》保留生成阶段上段各层算 query（生成阶段逐 token 解码仍要用上段），两处分别限定在 prefill 与生成，不再冲突；短页第 3 节同步改为「prefill 算完下段、拿到全局 KV 即可退出，不必进上段」。
- 未处理意见：无。
