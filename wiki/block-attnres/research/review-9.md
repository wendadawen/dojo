<!-- review-meta
round: 9
page: wiki/block-attnres/index.html
reviewed_content_sha256: 8e2fdbe079a86040
-->
# Block AttnRes 审查记录（第 9 轮）

- 页面版本：5ff9070d7ba32ec56b4cd974fafa8b97802c90b2
- 审查时间：2026-09-14 14:35
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何文件，未检索前序审查记录）
- 类型确认：dojo:type=concept，依 guides/concept/check.md 审查
- 已完整阅读章节：核心问题 → 1. 标准残差在深度上的瓶颈 → 2. Full AttnRes 的公式 → 3. Block AttnRes 的分块与块间 attention → 4. K3 的具体配置 → 5. softmax kernel 中的 RMSNorm → 来源与范围说明（含全部折叠块与图注）
- 来源获取：arXiv 2607.24653v2（HTML 正文与 §2.2/§7）；HuggingFace `moonshotai/Kimi-K3` 的 `config.json` 与官方源码 `modeling_kimi_linear.py`（下载后逐行核对）；MoonshotAI/nano-kpu 仓库 `docs/architecture.md`。

## 核对依据（逐条已定位原文/数值）

- §2.2 引文全部逐字命中：
  - "compress all prior information into a single state over depth … a bottleneck reminiscent of RNNs over time."（第 1 章 blockquote，line 172）
  - "Since network depth is modest (L < 100), the O(L^2 d) arithmetic of this full form is affordable; the practical overhead is the O(Ld) memory …"（line 382）
  - "Under Block AttnRes, memory and communication overhead drop from O(Ld) to O(Nd)."（line 454）
  - "the RMSNorm prevents layers with large-magnitude outputs from dominating the weights."（line 713）
  - "Empirically, N≈8 recovers most of the benefit across model scales [60]; for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer."（line 572）
  - "The final output layer then aggregates all N block representations."（line 637, C7）
  - "Attention Residuals (AttnRes) enable each module to selectively retrieve representations from the embedding, the current block, and preceding blocks."（line 637, C8）
  - 三维修辞："three complementary dimensions: sequence length, network depth, and model width"（Figure 2 caption 作 "token, channel, and layer mixing"）→ 支撑 line 118/overview 的"token 方向 attention / channel 方向 MoE / 深度方向 AttnRes 同级"。
- 公式：Eq.(8) k_i=v_i（i=0 取 h_1，1≤i≤l−1 取 f_i(h_i)）；Eq.(9) φ=exp(q_l^T RMSNorm(k_i))、α 归一化、h_l=Σα v_i；Eq.(10) 候选集合两种情形——均与 line 267/273/399 一致。
- config.json（text_config）实测：num_hidden_layers=93、hidden_size=7168、num_attention_heads=96、attn_res_block_size=12、kda_layers=69、full_attn_layers=24。与 line 664-669 表格、N1/N3、Table 1（93 层、69 KDA + 24 MLA）逐值一致。
- 官方源码 `modeling_kimi_linear.py` 实读：`attn_res_block_size`(L907)、`self_attention_res_norm/proj`(L910/914)、`mlp_res_norm/proj`(L912/916) 在 `_forward_attn_residual` 中分别用于 input_layernorm+attention 之前(L988-992)与 post_attention_layernorm+MoE 之前(L1028-1032)；`output_attn_res_norm/proj`(L1105/1107) 用于 `self.norm`(final norm) 之前(L1216-1232)。→ 直接印证 line 628-633 的"三次加权位置"与六个参数字段名；页面将其标为"间接证据（本页未直接复核源码）"，属偏保守，非错误。
- 关键数字复算：
  - Full 6 候选：内积 0.5/0.5/1.0/0.5/0.75/0.75；exp 和 3×1.6487+2.7183+2×2.1170=11.8984；权重 0.1386/0.1386/0.2284/0.1386/0.1779/0.1779；h_6=[0.7032,0.7032] ✓
  - Block 4 候选：内积 0.5/1.5/1.25/0.75；和 11.7377；权重 0.1405/0.3818/0.2974/0.1804；h_6=[1.0586,1.2414] ✓
  - 加 RMSNorm：RMSNorm(b) 内积 0.7071/0.9487/0.9806/0.9487；和 9.8583；权重 0.2058/0.2620/0.2704/0.2620；h_6=[1.0044,1.0564] ✓；b_1 权重 0.382→0.262 ✓
  - 大值敏感式 α1/α2=exp(q^T(k1−k2))、exp(2)≈7.4、exp(5)≈148、exp((c−1)‖q‖)=exp(4)≈54.6 均正确
  - 93=7×12+9 ✓；block 边界 1-12/13-24/…/73-84/85-93=9 层 ✓；√7168≈84.7 ✓
- 章节标题与「核心问题」答案末尾所指章节逐字对应；C1-C9/F1-F5/N1-N5 均在正文出现且双向对应；无"（待生成）"；residual-connection、kimi-k3-dataflow 前置页均存在；validate.py 返回 `validation ok`；无 Unicode 数学字符越界、无 alt 含 $、无会话指代/元话语套语（grep 全部为零命中）；结构图为 HTML grid，脚本关闭时仍可读。

## 问题

- [轻微·表述/来源] line 642（第 4 章末）：「原 preprint [60] 的设计可由模型自行决定」——N4 明确记录原 preprint 未获取原文，此处却对其"设计可自行决定加权次数"作了断定，属对未获取来源的属性归属｜引文依据：K3 报告 §2.2 仅转述 "N≈8 recovers most of the benefit across model scales [60]"，未描述 preprint 是否允许模型自定加权次数；本页 N4 自述"原 preprint 原文未核对"｜修复要求：将该半句降级为明确标注的推断（如"Eq.(8)(10) 不规定加权次数，采用该公式的模型可自行决定加权位置与次数（本页推断）"），或删除该半句，保留"公式不规定加权次数 / K3 的三次是 K3 的实例化"这一有据部分｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（唯一轻微问题不影响结论与主线，可接受；建议下一轮顺手降级/删除该半句）

统计：阻断 0 / 重要 0 / 轻微 1