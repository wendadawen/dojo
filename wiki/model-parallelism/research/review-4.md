<!-- review-meta
round: 4
page: wiki/model-parallelism/index.html
reviewed_content_sha256: b544e9ae6f477e57
-->
# 模型并行审查记录（第 4 轮）

- 页面版本：c242212c4b141e362e4c7129c5b092d5f67438f7
- 审查时间：2026-09-13 19:44
- 审查者：独立子代理
- 已完整阅读章节：引言与核心问题、1. 一层怎么切：张量并行（1.1 FFN、1.2 Attention、1.3 一层两次通信、本章问题）、2. 整摞层怎么分：流水线并行与气泡（2.1 推导、2.2 摊薄与推理的约束、本章问题）、3. 两种切法的通信代价（3.1 通信结构对比、3.2 组合部署、3.3 与 DP/EP 的边界、本章问题）、4. TP 与 PP 的取舍（本章问题）、来源与范围说明
- 机械项：`python3 .dojo/scripts/validate.py wiki/model-parallelism/index.html` → `validation ok`；页面无 Unicode 数学字符；已引用前置页 standard-attention / gpu-communication / moe-serving / beyond-buzz-disaggregation / chunked-prefill / pp-load-balancing / mla 的 index.html 均存在；overview.html 与 index.html 双向链接；`dojo:topics`（并行与通信, 推理系统）与 `dojo:tag`（并行与通信）均在词表内；页面无指向 research/ 的路径。

## 来源核对记录（本轮实际打开并定位的来源）

- GPipe（arXiv:1811.06965）§2.3 原文："This bubble time is O((K−1)/(M+K−1)) amortized over the number of micro-steps M. In our experiments, we found the bubble overhead to be negligible when M≥4×K."；"GPipe also introduces low communication overhead, given that we only need to pass activation tensors at the partition boundaries between accelerators. Therefore, we can achieve efficient scaling performance even on accelerators without high-speed interconnects." → 支持 [C5][C6][C7][N1][F2]。
- Sarathi（arXiv:2308.16369）§1 原文："In servers with high bandwidth connectivity such as NVIDIA DGX A100, tensor-parallelism can enable deployment of an LLM on up to 8 GPUs…Pope et al. show that tensor parallelism can be scaled up to 256 devices on specialized TPUv4 pods."；"However, tensor-parallelism at such a large scale can result in poor performance when hyper-clusters are unavailable." → 支持 [C8]。另 Sarathi 正文有 "bubbles like PB₂ that occur due to different compute times of prefill and decode stages when one is followed by the other"，支持 §2.2 补充块对 Sarathi 的归因。
- Megatron-LM（arXiv:1909.08053）§3 原文："Another option is to split A along its columns A=[A₁,A₂]. This partitioning allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM"；"Since GeLU is a nonlinear function…this approach will require a synchronization point before the GeLU function."；"we exploit inherent parallelism in the multihead attention operation, partitioning the GEMMs associated with key (K), query (Q), and value (V) in a column parallel fashion…"；"all GEMMs in a simple transformer layer using only two all-reduces in the forward path and two in the backward path"；Eq.(1)"Y=GeLU(XA)"、Eq.(3)"[Y₁,Y₂]=[GeLU(XA₁),GeLU(XA₂)]"；§2.3 标题为 "Data and Model Parallelism in Deep Learning" → 支持 [C2][C3][C4][C10][F1][N2]，以及来源章节对 §1/§2.3/§3 的定位。
- Beyond the Buzz（arXiv:2506.05508）Figure 5 caption 原文："Prefill performance is shown for DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP ×PP = 64)."；正文："disaggregation provides the greatest benefits in prefill-heavy traffic scenarios (i.e., ISL >> OSL) and when serving larger models (e.g., >10B parameters)."；"a key source of performance gain is the ability to use different model partitioning strategies for the prefill and decode stages." → 支持 [C9] 的组合实例与第 4 章对 Beyond the Buzz 的转述。
- 站内核对：beyond-buzz-disaggregation 页 F6 确实把"KV 复制因子/唯一切分 KV 的 GPU 数"用进带宽公式，支持 §1.2 的跨页引用。
- 公式复算：气泡比 $(p-1)/(m+p-1)$ 与 GPipe 的 $O((K-1)/(M+K-1))$ 同构；对 $p$ 求导 $m/(m+p-1)^2>0$ 复算无误；$3/7\approx0.43$、$3/19\approx0.16$ 无误；$m=1$ 得 $(p-1)/p$、$p=1$ 得 $0$ 无误；$p=4,m=4$ 时间槽图逐行核对（stage 1 忙 slot 1–4、stage 4 忙 slot 4–7，每行 4 忙 3 空）与正文一致。§2 补充块称训练时填充期/排空期在前向与反向各出现一次、slot 计数结构不变——与 GPipe 全前向再全反向调度下总时长 $2(M+K-1)$、总气泡 $2(K-1)$、比值不变一致，不报问题。

## 问题

- [阻断·技术] §1「代入具体数字验证等价性」段（line 153）与「展开：上面构造示例的完整手算」段（line 157）：同一推导给出两个互不相容的合计值。line 153 写"拼接 $[XA_1, XA_2] = [3, 3, 7, 7] = XA$"，line 157 写"两段拼接恰好是 $[4,4,4,8]$"。按题设 $X=[1,1,1,1]$、$A$ 为给定的分块对角矩阵，$XA$ 的列和只能为 $[3,3,7,7]$；$[4,4,4,8]$ 既无来源，又与同页另一处结论直接冲突（同一页内两处互相矛盾）。｜引文依据：line 153"$XA = [3, 3, 7, 7]$（列和：col1=1+2=3、col2=1+2=3、col3=3+4=7、col4=3+4=7）"；line 157"两段拼接恰好是 $[4,4,4,8]$"；独立复算 $XA=[3,3,7,7]$｜修复要求：把 line 157 的 $[4,4,4,8]$ 改为与题设一致的 $[3,3,7,7]$，并逐项复算，确保该段只出现这一个合计值。｜修复：｜复验：
- [阻断·技术] §1 手算段（line 153、157）：分段结果维度错误，使同段"拼接"等式不成立（分项之和≠合计）。$A_1=A_{:,0:2}$、$A_2=A_{:,2:4}$ 均为 $4\times2$，$X$ 为 $1\times4$，故 $XA_1$、$XA_2$ 必须是 $1\times2$ 的 $[3,3]$ 与 $[7,7]$；页面写成 $1\times4$ 的 $XA_1=[3,3,0,0]$、$XA_2=[0,0,7,7]$，按所写拼接得 8 维向量，无法等于所声明的 $[3,3,7,7]$。｜引文依据：line 153/157"$XA_1 = [3, 3, 0, 0]$"、"$XA_2 = [0, 0, 7, 7]$"、"拼接 $[XA_1, XA_2] = [3, 3, 7, 7]$"；独立复算 $XA_1=[3,3]$、$XA_2=[7,7]$、拼接 $=[3,3,7,7]$｜修复要求：删去两段中的补零，改为 $XA_1=[3,3]$、$XA_2=[7,7]$，使拼接结果与 $XA$ 一致；line 159 的 $G(XA)=[G(XA_1),\ G(XA_2)]$ 论证本身成立，需保留其对该正确分段的依赖。｜修复：｜复验：
- [轻微·表述] §1.3（line 170）："反向再各 2 次"表意不清，按"每子块各 2 次"读会得反向 4 次，与所引来源冲突。Megatron-LM §3 的计数是每层前向 2 次、反向 2 次。｜引文依据：Megatron-LM §3"all GEMMs in a simple transformer layer using only two all-reduces in the forward path and two in the backward path"｜修复要求：改写为无歧义表述（如"反向同样 2 次"），或删去"各"。｜修复：｜复验：
- [轻微·表述] 引言段（line 70）："本文的学习目标包括：说明…；解释…；推导…；说明…"是对全文内容的预告式元话语，与紧随其后的「核心问题」块重复；同段"正文首次引入各术语时会说明这一历史语境"是对本页自身结构的预告，而第 1–4 章正文并未再出现该狭义/广义的历史说明。｜引文依据：不适用｜修复要求：删去"本文的学习目标包括：…"整句（前置概念链接保留）；把"正文首次引入各术语时会说明这一历史语境"删去或改为当场说清（本页取广义用法，把 TP 与 PP 都算作模型并行）。｜修复：｜复验：
- [轻微·表述] 章节过渡与范围段（line 317、378、392）：line 378"机制讲完，这一章把两种切法放回'选型'的语境"为会话式临场语；line 317"这一章把两个画像放到互联拓扑上对比"与"通信画像"把比喻当术语反复使用；line 392"本页讲的是…，不回答…""本页提供阅读该论文所需的全部切分机制基础"以"本页"为主语作自我指代，并以"全部"作无来源的绝对表述。｜引文依据：不适用｜修复要求：line 378 改为直陈本章问题；line 317 把"画像"替换为"通信特征/通信结论"等直陈说法；line 392 去掉"全部"，把以"本页"为主语的自我指代改为对内容的直陈。｜修复：｜复验：
- [轻微·表述] §1.3 与本章问题 3（line 170、192）："一个 61 层的模型…一次前向 122 次 all-reduce"未标注为构造示例，也未给出该层数的出处（61 与 Beyond the Buzz 页用的 DeepSeek-R1 层数一致，但本页未说明）。按写作示例 A4，非来源实测的数字应明确标为构造示例。｜引文依据：不适用｜修复要求：在该处标注"构造示例"，或补上 61 层对应模型的出处。｜修复：｜复验：

## 结论

- 统计：阻断 2 / 重要 0 / 轻微 4
- 处置：修复（两个阻断项均位于 §1 同一段手算内，关闭后需重新逐项复算并复验；轻微项与来源论断无关，可一并处理）