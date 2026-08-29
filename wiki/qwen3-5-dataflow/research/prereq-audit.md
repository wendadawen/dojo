# Qwen3.5-397B-A17B 数据流页 · 前置概念盘点

生成于 2026-08-27，与页面同步完成（吸取前两代教训：从页面正文术语出发盘点，非先验清单）。

## 方法

1. 从页面正文与交互图节点说明提取全部专业术语
2. 按机制归属列出依赖概念
3. 逐个在 wiki/ 全站验证现有页面的真实覆盖（看内容，不看目录名）
4. 首现处补链接，缺页概念在页内自包含解释

## 结果：18 个依赖概念，16 处已链，0 个断档

| 概念 | 链接目标 | 首现位置 |
|---|---|---|
| Gated DeltaNet | ../gated-deltanet/ | lead |
| 线性注意力 | ../linear-attention/ | lead |
| MoE | ../deepseek-moe/ | lead |
| MTP / 投机解码 | ../speculative-decoding/ | lead |
| ViT | ../vit/ | lead |
| 残差连接 | ../residual-connection/ | §3 整体结构 |
| delta rule | ../delta-rule/ | §3 GDN |
| 深度可分离卷积 | ../depthwise-conv/ | §3 GDN |
| 位置编码 | ../positional-encoding/ | §1 规格表 |
| MRoPE | ../mrope/ | §1 规格表 |
| RMSNorm | ../rmsnorm/ | §3 全注意力 |
| KV cache | ../kv-cache/ | §3 全注意力 |
| GQA | ../mqa-gqa/ | §3 全注意力 |
| SwiGLU | ../swiglu/ | §3 MoE |
| 超连接（对比节用） | ../hyper-connections/ | §5 对比表 |
| 上代数据流页（对比节用） | ../qwen3-8-flash-next-dataflow/ | §5 对比节 |

## 页内自包含、不加链接的概念

- softmax / 因果掩码 / 标准 softmax 注意力：正文无自然锚点，公式与表格自足（与前两代数据流页处理一致）
- LayerNorm 与 RMSNorm 的差别：视觉对比表（4.3 节）直接给出，rmsnorm 页为深读入口
- aux loss / 负载均衡：§3 MoE 小节内完整给出公式与实算
- sigmoid / silu 激活：教科书级，不占链接
- 交错 MRoPE 槽位排布：mrope 概念页已完整覆盖，本页给结论与实测数字并链接

## 结论

无断档：Qwen3.5 的全部核心机制（GDN、GQA、MRoPE、MoE、MTP、ViT）在 wiki 均有专页可链，其中 mrope、rmsnorm、depthwise-conv、hyper-connections 为近两日新建。与 Qwen3.8-Flash-Next 页相比，本页无需「页内自包含但应建页」的缺口。
