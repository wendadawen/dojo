<!-- review-meta
round: 4
page: wiki/mla/index.html
reviewed_content_sha256: abb9c21402019872
-->
# MLA 审查记录（第 4 轮）

- 页面版本：162f4682594d11e70fc14cc70b5da53f9baef7de
- 审查时间：2026-09-13 21:14
- 审查者：独立子代理（编排者派发，未参与写作与前序轮次；仅读 index.html、overview.html、外部来源与 guides/concept/check.md）
- 已完整阅读章节：核心问题；1. MLA 压缩了什么——KV 联合压缩的核心机制；2. 推理时不重建 K/V——矩阵吸收；3. 为什么 RoPE 要解耦——位置编码与矩阵吸收的冲突；4. KV cache 到底减少了多少——与 MHA / GQA / MQA 对照；5. K3 的 Gated MLA——NoPE 与 full-rank output gate；来源与范围说明（含全部 details 折叠块、图注与构造示例）

机器验证：`.dojo/scripts/validate.py wiki/mla/index.html` → validation ok；公式定界符外无 Unicode 数学字符；无 img 的 alt 含 `$...$`；`../../wiki/mqa-gqa/index.html`、`../../wiki/low-rank-projection/index.html`、`../../wiki/rope/index.html` 均存在；index↔overview 互链正常。已回源核对并通过：DeepSeek-V2 摘要 93.3%（baseline=DeepSeek 67B）、§2.1.2 Eq.(9)(10)(11)、§2.1.3 Eq.(14)–(19)、§2.1.4 Table 1（MHA 2n_h d_h l / GQA 2n_g d_h l / MQA 2d_h l / MLA (d_c+d_h^R)l≈9/2 d_h l，能力列 Strong/Moderate/Weak/Stronger）、DeepSeek-V2 config（hidden 5120、60 层、128 头、kv_lora_rank 512、q_lora_rank 1536、qk_nope 128、qk_rope 64）、DeepSeek 67B config（95 层、64 头、8 KV 头）、K3 config（93 层、96 头、hidden 7168、mla_use_nope/mla_use_output_gate=true、24 层 full_attn_layers、69 层 KDA）、K3 §2.1.1 Full-rank gate 段与 §2.1.2（NoPE、Eq.(7) output gate、KDA 提供 position-sensitive recency-aware mixing）、modeling_kimi_linear.py（self.rotary_emb=None、assert self.use_nope、kv_a_proj_with_mqa 无条件执行、split [kv_lora_rank|qk_rope_head_dim]）。各构造示例手算（c_t^KV=(1,0,1)、q'_i^T c=1、拼接 (1,1,0,0,1,0)·(1,0,1,0,0,1)/√6≈0.408、gate sigmoid 逐通道值）均可复算。

## 问题

- [重要·技术] 引言（index.html「每 token 每层要缓存 $2 n_h d_h$ 个数……DeepSeek-V2 是 $128 \times 128 \times 60$（128 头、128 维、60 层），共 $1{,}966{,}080$ 个数/token」）｜问题：同句列出的乘积 $128\times128\times60=983{,}040$ 与给出的总数 $1{,}966{,}080$ 相差 2 倍——算式漏掉 K、V 两份的系数 2，读者照式子计算会得到自相矛盾的结果。overview.html 第 2 节复述同一写法，两页同错。｜引文依据：$2\times128\times128\times60=1{,}966{,}080$，$128\times128\times60=983{,}040$（N1/N2 配置 n_h=128、d_h=128、l=60）｜修复要求：把该乘积改写为 $2\times128\times128\times60$，或明示「128×128×60 指 n_h、d_h、l 三个配置量」；index.html 与 overview.html 同步｜修复：｜复验：

- [重要·技术] §5 折叠块「补充：为什么 $W^g$ 满秩重要」与「5. 本章问题」第 3 题解答（「若 $W^g$ 不满秩……$\mathrm{Sigmoid}(W^g\mathbf{x}_t)$ 在剩余 $d_h n_h - r$ 个通道上恒为常数 $\mathrm{Sigmoid}(0)=0.5$」）｜问题：该推理在数学上不成立——$W^g$ 秩亏只说明像落在 $r$ 维子空间，各门通道值仍随输入变化，只有 $W^g$ 出现整行零时对应通道才恒为常数；且本页自己把 $W^g$ 定为 $d_h n_h \times d$，其秩上界是 $\min(d_h n_h,d)=d$（K3：$96\times128=12288$ 对 $d=7168$），故「秩 $r<d_h n_h$」对真实满秩的 $W^g$ 也恒成立，与「满秩 ⇒ 不退化」自相矛盾。｜引文依据：K3 §2.1.2 仅述「The gate projection $W^g$ is full rank, matching the new parameterization used by KDA in Kimi K3.」，未给出「通道恒为 0.5」的机制；K3 config：num_attention_heads=96、hidden_size=7168、qk_nope_head_dim=128｜修复要求：删除「剩余通道恒为常数 $\mathrm{Sigmoid}(0)=0.5$」这一失实结论，改为「满秩（秩 = $d$）保证每个输出通道都是输入的非常数线性组合、门信号随输入变化」；或明确标注为本页推断并说明 $W^g$ 秩上界为 $d$。本章问题第 3 题解答同步修改｜修复：｜复验：

- [重要·来源] 来源与范围说明 N7（「社区分析 besthub.dev《Inside K3》文章指出……并称……」）｜问题：两段英文引文归给「besthub.dev《Inside K3》」，未给可解析链接；本轮在 besthub.dev 站点及网络检索均定位不到该文与这两句引文，属无法核对的来源论断。N7 后半段关于 vLLM 的陈述（vllm.models.kimi_k3 保留 qk_rope_head_dim、rotary_emb=None、NoPE fast path）可核对，前半段不可核对。｜引文依据：无法给出被引文所在片段；可核对项：vLLM 文档 vllm.models.kimi_k3.nvidia.mla 记「optional rotary embedding」且代码断言 assert mla_use_nope｜修复要求：为 besthub.dev 引文补可解析 URL；无法给出时删除这两句直接引文，改用可核对来源（官方源码/vLLM 文档）陈述同一结论，或降级为不加引号的本页推断｜修复：｜复验：

- [轻微·格式] 全页函数名写法不一致：注意力归一化在 §1、§3 展示公式写 $\mathrm{Softmax}$，在 §2 展示公式与折叠推导写 $\mathrm{softmax}$；§5 同一行并写 $\mathrm{Sigmoid}$ 与未定义的缩写 $\mathrm{Sig}$（$(\mathrm{Sig}(1),\mathrm{Sig}(0),\mathrm{Sig}(-1),\mathrm{Sig}(0))$）。｜引文依据：不适用｜修复要求：全页统一为一种写法（建议 $\mathrm{Softmax}$、$\mathrm{Sigmoid}$），删除未定义的 $\mathrm{Sig}$，直接写 $\mathrm{Sigmoid}$｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 1
- 处置：修复

## 附：本轮判定为不成立的候选（避免误报）

- 「本文/本章/下一章」等自我指代与章节过渡：guides/concept/style-guide.md §12 明确允许自称「本页」或「本文」，§8 允许章间用一至两句衔接，故不报。
- 「构造示例。」引入语：content-examples.md A4/A9 正例即用「构造示例。」，与规范一致，故不报。
- 章末「本章……下一章……」过渡句：A9 正例即此结构（先总结本章结论、再指下一步缺口），故不报。
- 正文与 overview 的 1/57、98.2%、480 GB、93.3% baseline：两页一致，且与来源一致，故不报。
