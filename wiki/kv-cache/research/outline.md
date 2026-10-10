# KV cache

- 描述：KV cache 是自回归生成时把每层已算过的 key 和 value 张量留下来复用的推理机制，让生成每个新 token 时不必重算全部历史。
- 摘要：自回归模型逐个预测 token，每步都要用到全部历史 token 的 key 和 value。把它们按层缓存下来、每步只追加当前 token，每步注意力开销从随序列长度二次降到线性。
- 主题：推理系统, 内存与缓存
- 标签：KV cache

## 长页

- 开头｜总览：KV cache 是什么、为什么需要、大概怎么工作；先交代自回归生成、模型由多层堆叠、注意力里的 query/key/value（链接注意力页）｜480｜blog/hf-cache-explanation.md How caching works；blog/hf-kv-cache.md Cache strategies
- ### 每步只算当前 token，历史 key、value 复用｜先交代因果注意力；第 t 步用 q_t 和全部历史 k 点积、取全部历史 v；token 处理后其 k、v 不随未来 token 改变，可缓存复用；推导折叠｜550｜blog/hf-cache-explanation.md Attention matrices；blog/hf-cache-explanation.md Cache class
- ### 缓存按层各存一份，每步追加一个 token｜每层各存一份 key 和一份 value，逐层独立计算；每步把新的 k、v 拼到过去之后；拼接公式折叠｜480｜blog/hf-cache-explanation.md Attention matrices；blog/hf-cache-explanation.md Cache storage implementation；blog/hf-cache-explanation.md Cache class
- ### 缓存省下多少计算｜不缓存时每步重算全部历史 K 和 V，每步注意力开销随序列长度二次；缓存后开销随长度线性；两种情形的对比折叠｜400｜blog/hf-cache-explanation.md L58-L63
- ### 缓存有多大｜交代多头注意力与批大小；缓存形状 [batch_size, num_heads, seq_len, head_dim]，显存随长度、批大小线性增长；公式手算折叠｜420｜blog/hf-cache-explanation.md Cache storage implementation
- ### 适用条件与边界｜只在推理时用，训练时启用可能引发意外错误；长上下文里缓存可能成为瓶颈；一条易混：缓存省的是计算不是存储｜400｜blog/hf-cache-explanation.md L29-L30；blog/hf-kv-cache.md Cache offloading；blog/hf-cache-explanation.md L58-L63

## 短页

- ### 把每层算过的 key 和 value 存下来｜是什么：先交代自回归生成与模型由多层堆叠，链接注意力页说明 query/key/value，再讲 KV cache 存什么｜320｜blog/hf-cache-explanation.md How caching works；blog/hf-kv-cache.md Cache strategies
- ### 不存就要每步重算全部历史｜为什么：每步预测都要用全部历史的 key、value，不存就每步重算；序列长度指已生成 token 的个数，开销随它二次增长｜330｜blog/hf-cache-explanation.md Attention matrices；blog/hf-cache-explanation.md L58-L63
- ### 每步只算新 token，追加进缓存｜怎么做：每步只算当前 token 的 key、value 拼到过去之后，最新 token 的 query 读全部缓存｜320｜blog/hf-cache-explanation.md Cache class；blog/hf-cache-explanation.md Cache storage implementation
