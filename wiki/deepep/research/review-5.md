<!-- review-meta
round: 5
page: wiki/deepep/index.html
reviewed_content_sha256: cfbb1ef2c364633a
-->
# DeepEP 审查记录（第 5 轮）

- 页面版本：06c88aa949d3eace1c1dbea8afe0a71609d7260f
- 审查时间：2026-09-13 20:12 CST
- 审查者：编排者派发的独立子代理（第 5 轮，未参与写作，未参与前序轮次）
- 已完整阅读章节：引言与前置（主要依据、核心问题 5 条、前置概念）、1. 专家并行的 all-to-all 卡在哪（1.1–1.3、本章问题）、2. 高吞吐内核（2.1–2.5、本章问题）、3. 低延迟内核（3.1–3.5、本章问题，含固定槽位模拟代码折叠块全段）、4. V2 重构（4.1–4.4、本章问题）、5. DeepEP 不解决什么（5.1–5.3、本章问题）、来源与范围说明（C1–C32、F1–F4、N1–N7、构造示例、简化条件）

核对方式：外部来源逐条取原文核对（README main commit 01dc3aa、docs/legacy.md、V1 初始 commit ebfe47e 的 deep_ep/buffer.py 与 csrc/kernels/internode_ll.cu、main 的 deep_ep/buffers/elastic.py、DeepSeek-V3 技术报告 arXiv:2412.19437v2 的 §2.1.2/§3.2.1/§3.2.2/§3.3.3/§3.4.1/§3.4.2/§3.5.1 与模型超参）。页面内嵌 Python 代码已实际执行，输出与页面"预期输出"逐行一致（rank 0 发送计划、E5 槽 0/4/8/12、recv_count 全 4、combine 结果 (3.6,4.6)/(5.0,7.0)、16/16 一致、32/128 利用率）。C/F/N 编号双向对应无缺无余；`.dojo/scripts/validate.py` 返回 validation ok；引用的 11 个前置概念页均存在；`research/sources/` 与 `deepep-src-extracts.md` 实际存在。V3 报告式（12）、node-limited routing（M=4、3.2 专家/节点、最多 13 个）、20/132 SM 与四项 SM 任务、两段转发与 20 SM/10 通道、TBO 相关段落（"two micro-batches"）、32 冗余专家/64 GPU、EP320 部署、FP8 dispatch + BF16 combine 等均已定位到原文片段，与页面表述一致。

## 问题

- [重要·技术] §2.5 表后段与 §3.5 表后段：页面先声明 RDMA 最大带宽约 50 GB/s（§2.1、§2.5），随后把跨节点 43-58 GB/s 表述为"逼近 50 GB/s 的 RDMA 峰值"，并在 §3.5 表中列出 EP8 时 98 GB/s（dispatch）/127 GB/s（combine）的 RDMA 带宽——两表的多个数值超过页面自述的单卡网卡峰值，页面未说明这是含本 rank 流量的有效带宽口径（同页 §4.4 对 V2 表已作此说明），构成同一页内两处数字互相矛盾。｜引文依据：legacy.md "with each connected to a CX7 InfiniBand 400 Gb/s RDMA network card (~50 GB/s maximum bandwidth)"；normal 表 "| Internode | 32 | 58 GB/s (RDMA) | 32 | 57 GB/s (RDMA) |"；low-latency 表 "| 8 | 77 us | 98 GB/s | 8 | 114 us | 127 GB/s |"。｜修复要求：对 §2.5、§3.5 两表补上与 §4.4 相同的口径说明（带宽为含本 rank 流量的有效/逻辑带宽，可超过单卡 NIC 峰值），或删去"逼近 50 GB/s 的 RDMA 峰值"并改写该范围描述；同时删去 §2.5"按最慢一段网络折算"这一在 legacy.md 与 README 中均无定义的表述。｜修复：｜复验：
- [重要·技术] §3.5 表后段（并见 §3.2 callout、页面级核心问题 3 的解答）：结论"纯 RDMA 的带宽利用率明显更低"与页面自身表格不符。§3.5 表 EP8 的 dispatch/combine RDMA 带宽为 98/127 GB/s、EP16 为 63/74 GB/s，均高于同页 normal 内核跨节点的 43-58 GB/s；该结论只有在 EP64 及以上才成立，页面未加限定。｜引文依据：页面 §3.5 "折算带宽从 98 GB/s 掉到 39 GB/s——对比 normal 内核跨节点的 43-58 GB/s……纯 RDMA 的带宽利用率明显更低"；legacy.md low-latency 表 EP8/EP16 行 98/127、63/74 GB/s 与 normal 表 43-58 GB/s。｜修复要求：把该比较限定在可比 EP 规模（EP64 及以上，并说明与 normal 表 EP16-64 的可比性），或改写为"随 EP 增大，低延迟内核的折算带宽下探到 normal 跨节点范围之下"；§3.2 callout 与核心问题解答中的同类无条件表述一并收窄。｜修复：｜复验：
- [轻微·技术] §1.2 第四点："发送侧事先不知道本 rank 会收到多少 token" 主客体颠倒——事先未知的是接收计数，属接收侧；同一页 §3.3 表述同一限制时用的是"接收侧"。｜引文依据：legacy.md "inside the dispatch function, we may not know how many tokens to receive for the current rank. So an implicit CPU wait for GPU received count signal will be involved"。｜修复要求：改为"接收侧事先不知道本 rank 会收到多少 token"，与 §3.3 用词统一。｜修复：｜复验：
- [轻微·技术] §4.4 表行"V3 式训练 SM 用量：V1 24 / V2 4-6"与 §1.2、§2.4 的"V3 训练用 20/132 个 H800 SM"并列出现，未说明两者关系，读者会在同一页读到 20 与 24 两个值。｜引文依据：V3 报告 §3.5.1 "we allocate 20 out of the 132 SMs available in the H800 GPU for this purpose"；README "For V3-like legacy training, SM usage reduced from 24 to 4 - 6"；legacy.md 示例 "Buffer.set_num_sms(24)"。｜修复要求：补一句说明 20（论文口径）与 24（V1 示例默认配置）的关系，或把 §4.4 行名限定为"V1/V2 对比口径（示例默认配置）"。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 2
- 处置：修复
