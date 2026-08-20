# kv-cache 内容范围

## 1. 概念歧义处理

KV cache（key-value cache，键值缓存）：LLM 推理语境下指注意力层缓存的 key/value 张量。与数据库/缓存系统语境的 "KV cache"（键值存储）、GPU 语境的 "cache"（硬件缓存）同名不同义。本页固定为 LLM 推理含义——状态：已裁定（vLLM 论文 §2 "these states consist of the key and value tensors associated with the attention mechanism, commonly referred to as KV cache"；SGLang 论文同义使用）。

## 2. 概念含义

- 名称：KV cache（key-value cache，键值缓存）
- 简要定义：推理时把注意力层为每个已处理 token 算出的 key/value 向量保存下来，后续 token 的注意力计算直接读取，避免对历史 token 的重复计算。
- 正式定义（与 vLLM 论文一致）：Transformer 推理中与前文 token 关联的 key 和 value 张量集合，表示上下文，用于按序生成新 token（vLLM §1 "the key and value tensors associated with the attention mechanism, commonly referred to as KV cache, which represent the context from earlier tokens to generate new output tokens in sequence"）。
- 本文语境：自回归 LLM 推理（decoder-only Transformer），单请求视角。
- 包括：KV cache 的内容（每层每 token 的 K/V 向量）、大小公式、prefill/decode 两阶段与缓存的产生和复用、显存压力的来源。GQA/MQA 下 KV 头少于查询头的事实（影响大小公式）。
- 不包括：分页管理（paged-attention 页）、跨请求前缀复用（prefix-caching 页）、KV cache 量化/压缩（KIVI/CacheGen 等，另一方向）、MLA（DeepSeek 的 latent 压缩，另一机制，wiki/deepseek-moe 与 mla 相关页面已另有安排，本页不涉及）。
- 相邻概念：模型权重（常驻显存、大小固定，与 KV cache 此消彼长）；激活值（计算中间产物、临时占用）；KV cache 量化（改变每元素字节数 b，本页公式保留 b 作为参数但不展开量化方法）。

## 3. 学习目标

### Q1：注意力计算为什么需要全部历史 token 的 K/V，缓存到底省了什么计算？

- 完成答案：注意力是每个新 token 对全部历史 token 的查询——$q_t$ 要与所有 $k_1..k_t$ 做点积、再用得到的权重加权所有 $v_1..v_t$；这些 K/V 只取决于已处理的 token 与模型参数，与当前在算哪个新 token 无关，所以算一次可以反复用。不缓存则每生成一个新 token 都要重算全部历史的 K/V（乘两层权重矩阵），生成 $n$ 个 token 的总代价从 $O(n)$ 次前缀计算恶化到 $O(n^2)$ 量级；缓存把历史 K/V 变成纯读取。依据：标准注意力公式（standard-attention 页）+ 该"K/V 与查询无关"的观察。
- 为什么核心：这是 KV cache 存在的理由；不理解"为什么能缓存"就会把它当成普通的内存缓存。
- 依赖内容：注意力公式（standard-attention 页）、自回归生成流程。

### Q2：prefill 与 decode 两阶段各自如何产生和使用 KV cache？

- 完成答案：prefill 一次性处理全部输入 token（并行算出其 K/V 并写入缓存），输出第一个 token；decode 逐个生成后续 token，每步只算 1 个新 token 的 q/k/v（k/v 追加进缓存），用其 q 读全部缓存做注意力。两阶段的资源特征不同：prefill 是大矩阵乘、计算密集；decode 每步计算量小但要读全部缓存，访存密集。依据：Strata 论文 §2.1 "LLM inference operates in two phases: prefill and decode"；vLLM 论文对迭代式生成的描述。
- 为什么核心：两阶段是后续一切 serving 机制（batching、缓存分层、PD 分离）的分析单位。
- 依赖内容：Q1、自回归生成。

### Q3：一份上下文的 KV cache 有多大？

- 完成答案：每 token 每层存 K、V 各 $H_{\text{kv}}\cdot d_{\text{head}}$ 个元素，全模型为 $2\cdot L_{\text{layers}}\cdot H_{\text{kv}}\cdot d_{\text{head}}\cdot b$ 字节。MHA 下 $H_{\text{kv}}=H_q$（与注意力头数相等）；GQA 下 $H_{\text{kv}}<H_q$，KV cache 显著缩小。手算：Llama-3.1-8B（32 层、8 KV 头、$d_{\text{head}}=128$、bf16）= 128 KB/token；20,000 token 文档 = 2.5 GB；对照 OPT-13B MHA（40 层、5120 hidden、FP16）= 800 KB/token（vLLM 论文数字）。40 GB 显存除去权重后能缓存的 token 数由此公式决定。
- 为什么核心：大小公式是"显存为什么不够、为什么要分层"的定量基础，也是 Strata 页容量账的直接前置。
- 依赖内容：注意力多头结构与 GQA（standard-attention 页有多头；GQA 在本页给最小定义：多组查询头共享一组 KV 头）。

### Q4：KV cache 为什么会成为显存瓶颈，常见误解有哪些？

- 完成答案：三点。①大小随 token 数线性增长且与请求并发数相乘——几十 GB 显存只够 0.1–0.3M token（vLLM 论文：13B 模型 A100 40GB 约 30% 显存给 KV cache；OPT-13B 单请求最长 1.6 GB）。②它是动态的（随生成增长），与静态权重争夺同一块显存。③误解：KV cache 是模型的一部分/部署时固定大小——错，它随负载变化；权重与缓存此消彼长是显存规划的核心矛盾。
- 为什么核心：回答"为什么需要管理"——引出 paged-attention 与 prefix-caching 两页的问题域（本页末尾一段过渡，不展开）。
- 依赖内容：Q3。

## 4. 内容分级

核心内容：K/V 与查询无关的观察（Q1）；两阶段机制与资源特征（Q2）；大小公式与 GQA 影响、Llama-3.1-8B 与 OPT-13B 手算（Q3）；动态性与显存账（Q4）。

辅助内容：MHA/GQA/MQA 头数关系的最小解释（服务 Q3 公式）；vLLM 显存分布图数字（65% 权重/30% KV cache，13B A100 例子）；KV cache 量化只改 $b$ 的一句话说明（消除"量化了就不需要管理"的误解）。

扩展内容：排除——PD 分离（另一部署形态，Strata 页提）；CUDA Graph 与 decode（无关）；具体引擎的内存池实现（paged-attention 页范围）。

## 5. 前置知识映射

| 前置概念 | 被哪些学习目标依赖 | 概念页 |
|---|---|---|
| 标准注意力（缩放点积、多头） | Q1（注意力公式）、Q3（头维度/头数） | `wiki/standard-attention/`（已有） |

## 6. 明确不展开的内容

- 分页/块管理与碎片（paged-attention 页职责；本页只以"如何高效存放下这些缓存"一句引出）。
- 跨请求前缀复用与 radix tree（prefix-caching 页职责）。
- KV cache 量化与压缩（KIVI、CacheGen 等：只说明它们改变 $b$ 或张量内容，不展开方法）。
- MLA/latent 注意力（DeepSeek 系的 K/V 压缩机制，属另一机制家族）。
- Prefill/decode 分离部署（PD disaggregation）：本页两阶段只讲计算特征，部署形态不涉及。

## 7. 常见误解和适用边界

误解：

1. "KV cache 是模型自带的缓存/部署时分配固定大小"——它是每个请求动态产生的中间数据，大小 = 并发请求 × 各自上下文长度 × 每 token 字节数，随负载变化。影响 Q4。
2. "上下文窗口 128K 说明显存放得下 128K"——128K 只是模型可接受的长度上限，能否放下由显存决定；128K token × 128 KB = 16 GB（Llama-3.1-8B 单请求），还只是 KV cache 部分。影响 Q3、Q4。
3. "GQA 只影响计算不影响缓存大小"——GQA 直接减少 KV 头数，KV cache 缩小为 MHA 的 $H_{\text{kv}}/H_q$（Llama-3.1-8B 为 8/32=1/4）。影响 Q3。
4. "缓存了 KV cache 就不用管显存了"——缓存只省计算，不省存储；显存不够时系统只能逐出（丢缓存重算）或分层（搬到 CPU/SSD），这正是后续两页的主题。影响 Q4。

适用边界：公式适用于标准多头/GQA 注意力的 decoder Transformer（Llama/Qwen/Mistral 类）；b 取决于精度（FP16/bf16 = 2 字节、FP8 = 1 字节）； sliding window attention / 层内稀疏注意力会改变"保存全部历史"的前提，本页不覆盖；MoE 不影响该公式（注意力层结构不变）。

## 8. 论断分级

- 文献已有结论：KV cache 定义与 OPT-13B 800 KB/token、1.6 GB 单请求、13B 模型显存分布（vLLM SOSP'23 §1、§3）；两阶段（Strata OSDI'26 §2.1，亦为 serving 文献通用术语）；RadixAttention 对缓存的复用动机（SGLang NeurIPS'24 §3.2）。
- 基于证据的推断：生成 $n$ 个 token 不缓存时总前缀计算 $O(n^2)$ 量级——由自回归+注意力结构直接推出（标注推断）。
- 外部事实：Llama-3.1-8B 配置（32 层、8 KV 头、head_dim 128、128K 窗口）——Meta Llama 3.1 模型卡（写页面时经 HuggingFace config 核实）；GQA 定义（分组查询头共享 KV 头）——Ainslie et al. GQA 论文/各模型文档通用表述。
