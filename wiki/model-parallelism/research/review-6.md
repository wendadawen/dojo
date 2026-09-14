<!-- review-meta
round: 6
page: wiki/model-parallelism/index.html
reviewed_content_sha256: e6cb56c1e7d7084d
-->
# 模型并行审查记录（第 6 轮）

- 页面版本：2d7b8e5a172154707d919ac1bb34a49df8e55314
- 审查时间：2026-09-13 21:14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节（含折叠块与图注）：核心问题；1. 一层怎么切：张量并行（1.1 FFN、1.2 Attention、1.3 一层两次通信、本章问题）；2. 整摞层怎么分：流水线并行与气泡（2.1 气泡占比推导、2.2 摊薄与推理的约束、本章问题）；3. 两种切法的通信代价（3.1 通信结构对比、3.2 组合部署、3.3 与数据并行/专家并行的边界、本章问题）；4. TP 与 PP 的取舍（本章问题）；来源与范围说明

## 来源核对依据（本轮回源）

- Megatron-LM（arXiv:1909.08053）§3(a)：Eq.(1) Y=GeLU(XA)；Eq.(2) 为 A 按行切（A=[A1;A2]、X=[X1,X2]）→ Y=GeLU(X1A1+X2A2)，原文称此方案"will require a synchronization point before the GeLU function"；Eq.(3) 为 A 按列切 [Y1,Y2]=[GeLU(XA1),GeLU(XA2)]，原文"removes a synchronization point"；第二段 GEMM 按行切与部分积求和只出现在正文与 Figure 3a，原文未给公式号（全文公式 (4) 在附录，为 perplexity）。§3 结论原文："two all-reduces in the forward path and two in the backward path"。§2.3 给出 data parallelism（minibatch split across workers）与 model parallelism 的定义。
- GPipe（arXiv:1811.06965）§2.3 原文："This bubble time is O((K-1)/(M+K-1))"；"we found the bubble overhead to be negligible when M ≥ 4×K"；"we only need to pass activation tensors at the partition boundaries"；"we can achieve efficient scaling performance even on accelerators without high-speed interconnects"。
- Sarathi（arXiv:2308.16369）§2.3/§3 原文："TP is preferred only within a single node connected by high bandwidth interconnects like NVLink"；"In servers with high bandwidth connectivity such as NVIDIA DGX A100, tensor-parallelism can enable deployment of an LLM on up to 8 GPUs"；"Pope et al. show that tensor parallelism can be scaled up to 256 devices on specialized TPUv4 pods"；"tensor-parallelism at such a large scale can result in poor performance when hyper-clusters are unavailable"；摘要与 §1：变动的 prefill/decode 时长在 PP 下造成 micro-batch 失衡与气泡。
- Beyond the Buzz（arXiv:2506.05508）§4：源包内确含 disaggregation_in_practice.tex，main.tex 中该文件为第 4 节（Introduction 1 / Background 2 / Design space exploration 3 / Disaggregation in practice 4）；其 Figure 5 caption 原文为 "Prefill performance is shown for DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP × PP = 64)."；结论"benefits … in prefill-heavy traffic scenarios … and when serving larger models"。
- 复算：X=[1,1,1,1]、A=B=给定块对角阵 → XA=[3,3,7,7]；XA1=[3,3]、XA2=[7,7]，拼接一致；分块乘法 Y1B1+Y2B2=YB。(p-1)/(m+p-1)：p=4,m=4→3/7≈0.43；m=16→3/19≈0.16；p=1→0；61 层×2=122。∂/∂p[(p-1)/(m+p-1)]=m/(m+p-1)²>0。时间槽图（p=4,m=4，7 slot、每 stage 忙 4 空 3）与公式一致。
- 机械项：validate.py 通过；全部前置概念链接（standard-attention、gpu-communication、moe-serving、beyond-buzz-disaggregation、chunked-prefill、pp-load-balancing、mla）与本地资源均存在；无 Unicode 数学字符；alt/aria-label 内无 $...$；overview.html 与 index.html 互链；无"（待生成）"占位。

## 问题

- [轻微·技术] 来源与范围说明 F1（第 417 行）：F1 把 FFN 切分的来源写作 "Megatron-LM §3 Eq.(2)(3)"，但原文 Eq.(2) 正是本页 §1.1 明确否定的"第一段按行切、需在 GeLU 前同步"方案，不是 F1 所描述的"列切/行切的本地计算与部分积结构"；本页采纳的方案对应 Eq.(3)（列切）与 Figure 3a（第二段行切与部分积，原文无公式号）。｜引文依据：原文 §3(a)"One option … to split the weight matrix A along its rows and input X along its columns … will require a synchronization point before the GeLU function"（Eq.2）；"Another option is to split A along its columns A=[A1,A2] … [Y1,Y2]=[GeLU(XA1),GeLU(XA2)]"（Eq.3）。｜修复要求：把 F1 公式号改为 "Eq.(3) 与 Figure 3a"（或 "Eq.(1)(3) 与 Figure 3a"），删去 Eq.(2)。｜修复：｜复验：

- [轻微·技术] §1.1 第 112 行："两卡各得 $XA_i$" 与本句随即给出的合并式 $\mathrm{GeLU}(X_1A_1+X_2A_2)$ 不一致：该切法下 $X$ 为 $1\times4$ 完整输入、$A_i$ 为 $2\times4$ 行块，$XA_i$ 维度不成立，每卡实际只能算 $X_iA_i$。｜引文依据：原文 Eq.(2) 对应的合并式为 Y=GeLU(X1A1+X2A2)。｜修复要求：把 "$XA_i$" 改为 "$X_iA_i$"。｜修复：｜复验：

- [轻微·技术] head 的 dojo:summary（第 7 行）"FFN 列切+行切使每层前向只需 2 次 all-reduce"：正文 §1.3（第 170 行）明确每层 2 次 all-reduce = attention 子块 1 次 + FFN 子块 1 次，summary 却把该数字整体归因于"FFN 列切+行切"，与正文归因不一致；概览页/卡片读者可能读成"FFN 子块需要 2 次"。｜引文依据：本页第 170 行"每个子块 1 次 all-reduce，所以 TP 之下每层前向共 2 次 all-reduce"；Megatron-LM §3"two all-reduces in the forward path"。｜修复要求：改为"列切/行切方案使每层前向只需 2 次 all-reduce（attention 与 FFN 子块各 1 次）"或等价表述。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（仅 3 处轻微，无阻断与重要；逐条修复后可发布）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
