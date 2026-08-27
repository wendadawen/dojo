# Qwen3.8-Flash-Next 数据流页：前置概念盘点

盘点方法：从页面正文实际术语频次出发（非先验清单），按机制归属列出前置依赖，
逐项用全站 grep 验证候选页面的真实覆盖内容（不看目录名，看章节标题与正文命中）。
盘点时间：2026-08-27。上一轮盘点（13 处链接）的方法缺陷：先想概念再找页，
漏掉了 N-gram、低秩、哈希、深度可分离卷积等页面核心机制的前置。

## 一、已链接的 14 处（13 处上一轮 + 1 处本轮补充）

| 概念 | 链接目标 | 锚点位置 | 验证 |
|---|---|---|---|
| Gated DeltaNet | ../gated-deltanet/ | lead 首现 | 页面存在，主题吻合 |
| 线性注意力 | ../linear-attention/ | lead 首现 | 同上 |
| MoE | ../deepseek-moe/ | lead 首现 | 同上 |
| 残差（连接） | ../residual-connection/ | lead「残差不是单条通路」 | 同上 |
| ViT | ../vit/ | lead「27 层独立 ViT」 | 同上 |
| 稀疏注意力 | ../dsa/ | 1 节 note「带块选择的稀疏注意力」 | 同上 |
| 位置编码 | ../positional-encoding/ | 1 节表格 | 该页讲位置编码分类 |
| MTP 草稿层 | ../speculative-decoding/ | 1 节表格 | 投机解码背景 |
| delta rule | ../delta-rule/ | 3 节 GDN「名称来源」 | 同上 |
| DeepSeek-V4 压缩对照 | ../dsa/ | 3 节 QSA | 同上 |
| RoPE | ../rope/ | 3 节 QSA「部分 RoPE」 | 该页讲 2 维 RoPE 机制 |
| KV（cache） | ../kv-cache/ | 长上下文开销节 | 同上 |
| GQA | ../mqa-gqa/ | 4.3 表格「无 GQA」 | 同上 |
| SwiGLU | ../swiglu/ | 4.3 表格 | 同上 |
| **低秩瓶颈**（本轮补） | ../low-rank-projection/ | 3 节超连接「低秩瓶颈」 | 该页讲 SVD/LoRA/MLA 低秩投影 |

## 二、全站零覆盖的概念（缺页，按对本页重要度排序）

### 1. N-gram / n 元语法 —— 唯一的强缺口
- 全站 grep「n-gram / N-gram / n元语法」零命中（除本页自身）。
- 本页核心机制（51.2B 表、bigram/trigram、16 头哈希）全部建立在 n-gram 概念上，
  但页面对「n-gram 是什么」无任何背景解释——正文出现「前 8 个头处理 bigram、
  后 8 个处理 trigram」时，不懂 n-gram 语言模型背景的读者在此断档。
- 页内现状：讲了哈希怎么算（公式+实测），没讲 n-gram 统计本身的动机
  （马尔可夫假设、用前 n-1 个 token 预测下一个的经典背景）。
- 建议：按 concept 流程生成 n-gram 概念页（或扩展 pretraining 页），
  本页 bigram/trigram 首现处再补链接。

### 2. MRoPE / 多模态三维位置编码
- rope 页章节：2 维 RoPE 机制、内积相对性、d 维推广、远程衰减、适用边界——
  无 MRoPE、无 partial rotary、无交错排布。
- 本页 4.6–4.7 已自包含（三维分配、位置推进量、32 槽位交错实测）。

### 3. LayerNorm 与 RMSNorm
- 无 norm 专门页；clip/dsa/glu/moe-serving 等页顺带提及。
- 本页 4.3 表格对比了差别（带 bias vs 无 bias、权重形式），最小自包含。

### 4. 整数哈希（splitmix64 / 质数取模 / 异或混合）
- 无页面。grep「哈希」命中的 dualpath/prefix-caching 等是前缀缓存 key 匹配语境，
  与整数哈希函数是不同概念。
- 本页给了完整公式、checkpoint 真值核对与溢出/可逆性分析，自包含。

### 5. 双线性重采样（可学习位置方格的适配）
- positional-encoding 页无插值/重采样内容。
- 本页 4.2 给了最小说明（48×48 方格按实际网格采样、保长宽比）。

### 6. 深度可分离因果卷积
- 无概念页；gated-deltanet 页 grep depthwise/因果卷积 零命中，
  kimi-k3-dataflow 顺带提过（同为数据流页）。
- 本页覆盖最全（kernel=4、decode 跨步状态、probe6 专项验证），自包含。

### 7. 词嵌入 / embedding
- 无专门页；pretraining 页讲「语言模型是什么/逐 token 分解」但不直接讲 embedding。
- 弱依赖：页面「词嵌入」一词对目标读者基本自明。

## 三、有页面但未链接的概念（原因记录）

| 概念 | 页面 | 未链原因 |
|---|---|---|
| 因果掩码 | causal-mask | 正文无「因果掩码」字样（仅「因果可见范围」「因果卷积」），无自然锚点；强行加需改写句子 |
| 标准注意力 | standard-attention | 公式页内自足（softmax 缩放形式已写明）；无自然锚点 |
| softmax | cross-entropy | 依赖已被 deepseek-moe 链接间接覆盖（该页讲路由 softmax） |
| GELU/激活函数 | glu/swiglu | 弱依赖，SwiGLU 链接已指向 |
| CLIP/SigLIP | siglip | 页面未提及，非依赖 |
| 负载均衡/aux loss | aux-loss-free-routing | 本页未涉及（未讲训练损失） |

## 四、结论

- 依赖面 22 个概念：14 已链、1 有页未链（低秩，本轮已补）、7 全站缺页。
- 缺页中只有 N-gram 构成理解断档（核心机制 + 零背景解释），其余 6 个页内
  自包含程度足够（note 定位下的合理状态）。
- causal-mask 与 standard-attention 的未链属锚点问题，非覆盖问题。
