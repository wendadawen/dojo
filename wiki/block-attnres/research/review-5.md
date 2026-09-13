<!-- review-meta
round: 5
page: wiki/block-attnres/index.html
reviewed_content_sha256: 94da49a4a4aad713
-->
# Block AttnRes 审查记录（第 5 轮）

- 页面版本：ec20ab82bf41b9ffbfff85b9bb1d667af2c9d8a6
- 审查时间：2026-09-13 20:12
- 审查者：独立子代理
- 已完整阅读章节：核心问题；1. 标准残差在深度上的瓶颈——为什么需要 AttnRes；2. Full AttnRes 的公式——pseudo-query 如何检索前序层；3. Block AttnRes 的分块与块间 attention——把内存从 $O(Ld)$ 降到 $O(Nd)$；4. K3 的具体配置——8 块×12 层、9 个候选、加权三次；5. softmax kernel 中的 RMSNorm——为什么不能直接用内积；来源与范围说明（含全部折叠块、图注与配置表）

## 来源核对（本轮实际打开的来源）

- K3 报告 arXiv:2607.24653v2 全文 HTML（§2 第 209 行 "Attention Residuals (AttnRes) [60] enable each module to selectively retrieve representations from the embedding, the current block, and preceding blocks"；§2.2 各引文与 Eq.(8)(9)(10)；§2.2 "9 total blocks when counting the embedding layer"；§5.2.2 "The block representation is generated once at the boundary layer"、checkpointing、cache-based pipeline communication [60]；§5.4.2 "two-phase schedule: a batched inter-block pass … online-softmax merge"、SP 激活分片、side stream 重叠；§7 "Block AttnRes with a block size of two" + github.com/MoonshotAI/nano-kpu；Table 1「Attention-Layer Composition 69 KDA + 24 MLA」「Hidden Dimension 7,168」「Attention Heads 96」；参考文献 [147] B. Zhang and R. Sennrich (2019) Root mean square layer normalization）
- HuggingFace 官方 config.json（moonshotai/Kimi-K3）：attn_res_block_size=12、num_hidden_layers=93、hidden_size=7168、num_attention_heads=96、linear_attn_config.full_attn_layers=24 项、linear_attn_config.kda_layers=69 项
- 官方源码 modeling_kimi_linear.py：参数名 self_attention_res_norm/proj、mlp_res_norm/proj、output_attn_res_norm/proj 与三次加权位置
- 本地：.dojo/scripts/validate.py 返回 `validation ok`；dojo:topics=注意力机制、dojo:tag=注意力 均在词表内；overview.html 与 index.html 相互链接；../residual-connection/、../kimi-k3-dataflow/ 均存在
- 全部手算复算通过：Full AttnRes 6 候选 h6≈[0.703,0.703]（权重 0.1386/0.2284/0.1779）、Block AttnRes 4 候选 h6≈[1.059,1.241]（0.1405/0.3818/0.2974/0.1804）、加 RMSNorm 后权重 [0.2058,0.2620,0.2704,0.2620]、exp 大值敏感性 exp(4)≈54.6 / exp(2)≈7.4 / exp(5)≈148、方向余弦 0.9487<0.9806

## 问题

- [重要·技术] §3「代码：Block AttnRes 前向计算的伪代码」折叠块末行 `output ← output_attnres(block_snapshots + [h])`：循环对每个 block 结束后都执行 `block_snapshots.append(current_partial)`，N=8 时循环结束 block_snapshots=[b_0,…,b_8]（9 项），再加 `[h]` 得 10 个候选，与本页正文「9 个候选（8 块快照 + 当前流）」及 K3 源码的 9 个候选（8 个块快照 + 最终残差流）矛盾；且伪代码另设了「全层累加、不随 block 重置」的状态 `h`，与 K3 中逐 block 重置的 prefix_sum 不符，导致同一个候选集被重复计入。｜引文依据：K3 源码 `_apply_attn_res`：`v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)`，block_residual 仅在 `layer_idx % attn_res_block_size == 0` 时追加（层 0,12,…,84 共 8 次），末尾 output AttnRes 候选 = 8 + 1 = 9；报告 §2.2 "The final output layer then aggregates all N block representations."｜修复要求：使伪代码的输出候选数为 9——删除全局状态 `h`（残差流即逐 block 的 `current_partial`），输出行改为 `output ← output_attnres(block_snapshots[:-1] + [current_partial])`，并同步修改 state 注释与「用 attn_output 替代当前残差流」相关说明，使伪代码与 §3 正文、§4「9 个候选」一致｜修复：｜复验：

- [轻微·技术] §4「9 个候选来源的构成」段：「K3 报告 §2.2 以最大候选数 9 描述该配置」把推导结论写成报告原文表述。报告 §2.2 的「9」指「含 embedding 共 9 个 block」，不是「最大候选数 9」；本页的「9 个候选」是由 Eq.(10)（候选数 n+1，n=8）推得。｜引文依据：报告 §2.2 "for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer."｜修复要求：改为「报告以『含 embedding 共 9 个 block』描述该配置；由 Eq.(10)，最后一个 block 内 i≥2 层的候选数为 9」，或删除该半句｜修复：｜复验：

- [轻微·技术] §4「9 个候选来源的构成」段末括注「具体槽位分配的实现细节（预分配与屏蔽策略）」：引入报告与源码都没有的实现描述，且与源码不符——K3 用 torch.cat 动态增长候选张量，不存在预分配固定槽位与屏蔽。｜引文依据：源码 `block_residual = torch.cat([block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1)`（动态追加）｜修复要求：删除括注「（预分配与屏蔽策略）」，或改为不指定实现的中性表述（如「候选槽位的具体实现在报告中未展开」）｜修复：｜复验：

- [轻微·表达] §2「补充：softmax 对大值的敏感性」与 §5「补充：softmax 大值敏感性的形式化」两块重复同一推导：均从 $\alpha_1/\alpha_2=\exp(q^\top(k_1-k_2))$ 出发，用 $k_1=c u$、$k_2=u$ 推出 $\exp((c-1)\|q\|)$，并给出 $\|q\|=1,c=5\to\exp(4)\approx 54.6$；§5 块为超集（另含 7.4/148 与边界）。｜引文依据：不适用｜修复要求：保留 §5 的形式化推导，将 §2 折叠块删除，或收缩为一句指向 §5 的引用（如「其形式化推导见 softmax kernel 中的 RMSNorm 一章」）｜修复：｜复验：

- [轻微·格式] 各章末尾的过渡句使用同一句式：§1「本章说明了…但还没有…——下一章讲…」、§2「本章给出了…但…——下一章讲…」、§3、§4 同构，属 style-guide §8「不使用固定句式，也不为形式完整而添加过渡」所禁的模板化衔接。｜引文依据：不适用｜修复要求：按 style-guide §8，各章按前后依赖关系各写一至两句、不套用同一句式｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复
- 说明：页面核心事实论断（Eq.8/9/10、8 块×12 层、9 个候选、加权三次位置、$N\approx8$、config.json 数值、源码参数名、§5.2.2/§5.4.2/§7 细节）经报告原文、官方 config.json 与官方源码三方逐条核对，均一致；全部手算可复算；无元话语/会话指代/调试叙事等表述问题（"本页/本文"自称符合 style-guide §12）。唯一的「重要」项是伪代码输出候选数与正文及源码不符，修复后即可发布。