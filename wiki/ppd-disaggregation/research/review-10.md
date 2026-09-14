<!-- review-meta
round: 10
page: wiki/ppd-disaggregation/index.html
reviewed_content_sha256: 9dceefae019ed3bc
-->
# Not All Prefills Are Equal（PPD 分离）审查记录（第 10 轮）

- 页面版本：f9946f91276fd3576f2b996e11563b7cb15ad2c63b12e030f4d35dc93888d875（sha256 of wiki/ppd-disaggregation/index.html 工作树；git hash-object 65971ebeea6bb3d2347823cf36142dfa8f44ce6d）
- 论文版本：arXiv:2603.13358v2（v1 2026-03-09，v2 2026-05-05 修订；ICML 2026）。核对用材料：arXiv 摘要页、arXiv:2603.13358v2 全文 PDF（19 页，含 Appendix A/B.1–B.6/C.1–C.5，已 pdftotext 还原逐字核对）、页面 assets/ 下 6 张原图（像素测量）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复；未读取 research/ 下任何文件）
- 已完整阅读章节：核心问题（5 题含解答）→ 术语表 → 1 多轮对话暴露 PD 分离的两个代价 → 2 full prefill 与 append-prefill 差一个数量级 → 3 没有静态最优：3060 个数据点的扫描证据 → 4 PPD：把路由变成带权重的逐请求决策 → 5 真实负载、慢网络与权重旋钮下的表现 → 6 方法评价 → 7 附：PPD 架构概念图 → 来源与范围说明（全部折叠块、图注、表格均已通读）
- 机械核对：python3 .dojo/scripts/validate.py wiki/ppd-disaggregation/index.html → validation ok（含数学字符与结构图检查）；4 个被引前置概念页（standard-attention / mqa-gqa / gpu-communication / prefix-caching）与其 index.html 均存在；8 个 ../../libs/ 本地资源均存在；无「待生成」占位；6 张 img 的 alt 内无 `$...$`；C1–C20 / N1–N21 / F1–F5 / G1–G6 编号在正文均已引用且全部可解析（正文 F16 计数系 BF16 误匹配，非真实编号）

## 问题

- [轻微·格式] 6.2 局限（第 566 行）与第 6 章本章问题解答（第 593 行）：「硬件或负载分布远离**婚线**校准集」中「婚线」应为「离线」（两处同错）｜引文依据：不适用（页面文字）｜修复要求：将两处「婚线」改为「离线」，使「远离离线校准集」与同句「决策表基于离线校准」「离线表漂移」一致｜修复：｜复验：
- [轻微·表述] 第 3 章本章问题（第 323 行）：章节问题写成口语化流行语「Replica 为什么不香？」｜引文依据：不适用｜修复要求：改为书面表述，如「Replica 为什么不是更好的选择？」或「Replica 的胜场为何集中在 TTFT？」｜修复：｜复验：
- [轻微·图注] 3.2 图（原文 Figure 1）图注（第 261 行）：「QPS=0.5 时两组曲线在 TPS 50–70 段几乎重合」与图不符——QPS=0.5 子图中 D-local（蓝）曲线最高数据点止于 TPS≈64，蓝线在 TPS 64–70 段并不存在｜引文依据：对 assets/img-05.webp 像素测量：QPS=0.5 子图 y 轴 0–100（行 53=100、行 671=0），蓝线像素跨行 276–485（TPS 63.9→30.1）、跨列对应 TTFT≈1341–10869 ms；橙线（baselines）最高到 TPS≈92（受子图裁切）｜修复要求：把范围改为与蓝线实际跨度相符的区间（如「TPS 30–64 段两组曲线接近」），或去掉数字区间改为「低 TPS 段两组曲线接近，随 TPS 升高蓝线止于其最优点、橙线继续上行」｜修复：｜复验：
- [轻微·技术] 5.5 整体成绩与泛化性（第 500 行）：「在 2–16 轮会话与 8B/14B/30B 模型上 Turn 2+ TTFT 改善稳定在约 70%」未写明主体，紧接上句「PPD 把 Turn 2+ TTFT 平均降低约 68%」后读作 PPD 的改善；论文该 ~70% 是 x=1 相对 x=0 的优势（同页 C18 亦写作「$x{=}1$ 优势」），且出自 Appendix C.3 的合成扩展实验而非真实负载｜引文依据：§4.3「Scaling experiments (see Figure 10 in Appendix) show stable ∼70% Turn 2+ TTFT improvement across 2–16 turns and three model sizes (8B, 14B, 30B)」（该小节标题为 Full AP-to-D Advantage，比较对象为 x=0→x=1）；C.3「the relative improvement remains stable at ∼70%」｜修复要求：在该句补出限定主体（写明「$x{=}1$（PPD 的极端权重情形）」），使其与来源表 C18 一致｜修复：｜复验：
- [轻微·技术] 开篇（第 72 行）与第 1 章（第 135 行）：把 DeepSeek、Gemini 与 vLLM/SGLang/TensorRT-LLM 并列称为「主流 LLM 推理引擎」，来源中 DeepSeek/Gemini 是生产部署方而非引擎，且引擎名单中的 LMDeploy、NVIDIA Dynamo 被略去｜引文依据：§2.2「It is supported by all major serving frameworks (vLLM (Kwon et al., 2023), SGLang (Zheng et al., 2024), TensorRT-LLM, LMDeploy (Contributors, 2023), and NVIDIA Dynamo (NVIDIA, 2025)) and is deployed at production scale by providers such as DeepSeek (DeepSeek-AI et al., 2025) and Gemini (Team et al., 2025).」｜修复要求：改为「已被 vLLM、SGLang、TensorRT-LLM 等主流推理引擎采用，并由 DeepSeek、Gemini 等厂商生产部署」，区分引擎与被部署方｜修复：｜复验：
- [轻微·格式] 开篇构造示例（第 70 行）：`构造示例。 一个 5 轮客服对话…` 全角句号后多出一个半角空格，与全文标点规范不一致｜引文依据：不适用｜修复要求：删除「构造示例。」后的半角空格（或改为「构造示例：一个 5 轮客服对话…」）｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：可发布（本轮未发现阻断与重要问题；6 条轻微问题不影响正确性与主线理解，修复后可直接发布）

### 核对说明（本轮已逐条回源、无问题的重要项，供复验参考）

- 数字全部与 arXiv:2603.13358v2 逐字核对一致：batch 200 处 full prefill ∼48% / append-prefill ∼2%（§4.1、Fig.2）；4 并发 +57%/+21%（App.C.1 Fig.7）；32K 时 3–4×、64K 时 <25%（App.C.1 Fig.8）；3060 = 17×18×10（§4.2）；92.2%（§4.4）；Table 2 winner 分布 63.3/0.6/0/21.3、0/38.3/4.4/14.2、3.3/33.3/27.8/21.5、27.2/15.6/38.3/27.0（§4.4）；Table 1 −57.8/−65.2/−73.3、−47.7/−51.6/−56.2、−44.3/−38.1/−24.9（§4.3）；15–25%（§6.2）；3.1 轮/会话、∼75% KV、∼3×（§6.2）；Table 3 的 10/5/12、0/13/14、4/27/27（§6.3）；143.7→170.6 ms（+18.7%）、PPD ~51 ms、E2E +4.7%（3028→3169）、PPD +0.9%、64%→70%（§6.4 Fig.5，且页面已如实标注图注文字 ~150 GB/s 与图内横轴 ~200 GB/s 的差异）；w_tpot 1/3/6→95%/50%/20%、94–96% TTFT / 7–12% TPOT（§6.5 Fig.6）；68%（abstract/§8）；<1 ms（§5/§8）；2K 上下文 ∼256 MB（§2.2）；s_kv=128 KiB、P90 5115 token→∼670 MB、4.5/27/67 ms（App.B.6）；failure 表 11/44/61/89%、6/22/44/67%、0/6/11/22%、0/0/6/11%（App.C.4 Table 5）；30s/10s（App.C.5）
- Eq.1（S 打分）与 Eq.2（带宽注入 Δt）逐字一致；页面构造示例的算术已复算：1250²=1,562,500、50×1250=62,500、比值 25 与 n/m=24 同量级、1250 token×128 KiB=156.25 MiB、S 在 w=(1,1)/(1,6)/(1,10) 与 (1,10)+Δ=(0.2,0.05) 下取 0.52/0.12/−0.20/−0.30 全部正确
- 6 张原图与正文解释一一对应（img-06=Fig.2、img-01=Fig.3、img-04=Fig.4、img-03=Fig.5、img-02=Fig.6、img-05=Fig.1）；Fig.3 中「Last Turn's KV」红线在 PD 与 PPD 两幅均绘为指向 P 的箭头，页面在第 7 章的观察属实（已裁图逐像素确认）
- 折叠块 28 个各配 28 个 summary；核心问题 5 题、第 1–6 章本章问题共 24 题全部有解答且答案含「完整论证见…一章」或等价指引；页面无「本页/我们/我/你」等自我指代与会话指代；无调试叙事；无 <pre> 外的 Unicode 数学字符（Δ/ψ/∈ 仅出现在伪代码块内，× – → 属规范明确允许的中文排版字符）；页面未声称有可运行代码（仅一段 language-text 伪代码，不触发代码执行核对）
- 参考文献编号体系（C/N/F/G）无一处指向不存在的条目，F4/F5 已明确标注为「页面构造的验证计算」，与论文实测数字区分；论文未提供官方代码仓库链接，页面 meta 如实说明
