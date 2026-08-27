# GLM-5.3-Flash 数据流页：前置概念盘点

方法：从页面正文的术语频次出发（不是先想清单），逐项到全站 `wiki/` 验证覆盖情况；
验证时看候选页的 `<title>` 与 h2 章节的实际内容，不看目录名。
脚本 `prereq_audit.py`。

## 已链接（15 个，链接目标均验证存在）

| 概念 | 页面 | 本页首现位置 |
|---|---|---|
| 线性注意力 | `linear-attention` | 引言导航 |
| delta 规则 | `delta-rule` | 引言导航 + 第 5 节递推公式 |
| KDA | `kda` | 引言导航 + 第 5 节开头 |
| MLA | `mla` | 引言导航 |
| DSA | `dsa` | 引言导航 |
| DeepSeekMoE | `deepseek-moe` | 引言导航 |
| 辅助损失无关路由 | `aux-loss-free-routing` | 引言导航 + 第 7 节路由 |
| NoPE | `nope` | 引言导航 |
| 残差连接 | `residual-connection` | 第 3 节单层数据流 |
| KV cache | `kv-cache` | 第 5 节开头 |
| 低秩分解 | `low-rank-projection` | 第 6 节 MLA 潜向量 |
| SwiGLU | `swiglu` | 第 7 节专家内部 |
| 量化基础 | `quantization-basics` | 第 10 节 |
| ViT | `vit` | 第 11 节视觉塔 |
| 投机解码 | `speculative-decoding` | 第 12 节 MTP |

覆盖质量核实（抽查章节标题确认真实覆盖，非仅命中关键词）：
- `kda` 第 2–3 节正是「delta rule + channel-wise forget gate」与「lower-bounded decay」，与本页第 5 节的机制完全对应
- `dsa` 第 3–4 节讲 lightning indexer 打分与 top-k 跨头共享，本页第 6 节的 k-pool 是其变体
- `nope` 第 5 节明确讲「MLA 用 NoPE、KDA 提供位置」，与本页第 8 节的推断一致
- `aux-loss-free-routing` 第 2 节「bias 加在路由分数上——只管选谁，不管用多少」正是本页第 7 节实测的结论

## 真正的覆盖缺口：mHC / Sinkhorn 双随机投影

全站检索 `Sinkhorn`、`双随机`、`Hyper-Connection`、`超连接`：
只在 `deepseek-v4-dataflow`、`qwen3-8-flash-next-dataflow` 两个**数据流 note 页**里出现，
没有任何概念页解释这套机制。而它是本页第 4 节的核心内容（4 路残差流的读写方式）。

处理：本页第 4 节已给自包含解释——三路系数的公式与取值范围、Sinkhorn 迭代次数与
双随机偏差的实测表、退化到普通残差的检验、末端无权重均值收敛、激活显存代价。
按 note 定位（自包含可独立阅读）这是合格的。

**是否递归生成 mHC 概念页待用户决定。** 若要生成，建议范围：
超连接的动机（残差的深度瓶颈）→ 多路并行流的读写 → Sinkhorn-Knopp 投影与双随机流形
→ 与普通残差、与 DeepSeek-V4 加权收敛版的差别。相邻已有页 `block-attnres`
（Block AttnRes 把深度方向等权累加换成块级注意力检索）解决的是同一类问题，
可作为对照，但机制不同、不能替代。

## 无专页但页内自足（不构成断档）

- **RMSNorm**：全站无专页（`block-attnres`/`glu` 等多页提及但非主题）。本页只把它当已知算子使用，且给出了完整公式语义（第 3 节说明它作用在压成单路的张量上），note 定位下可接受。
- **深度可分离卷积 / depthwise conv**：`kimi-k3-dataflow`、`dflash2` 提及但无概念页。本页第 5 节说明了它的作用（提供局部相对位置）与形状（groups 等于通道数），够用。
- **FP8 E4M3 块量化的具体格式**：`quantization-basics` 讲通用量化，未覆盖 E4M3 与块量化。本页第 10 节给出了分块形状的实测验证与不量化范围，自足。

## 未加链接的相关页及原因

- `standard-attention`、`causal-mask`：正文无自然锚点（本页不重讲注意力基础）。
- `mqa-gqa`：本页未与 GQA 对照，无锚点。
- `flash-kda`（KDA 的 GPU 实现与上下文并行）、`kimi-k3`/`kimi-k3-dataflow`（KDA 的来源模型）：属延伸阅读而非前置，本页范围不含 kernel 实现。
- `moe-serving`、`latent-moe`、`quantile-balancing`：属 MoE 的服务与变体，非本页前置。
