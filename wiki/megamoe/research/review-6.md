<!-- review-meta
round: 6
page: wiki/megamoe/index.html
reviewed_content_sha256: d9973b8f3b20e9a5
-->
# MegaMoE 审查记录（第 6 轮）

- 页面版本：56ea432c749bc909274491acf377d71a555b4f3e
- 审查时间：2026-09-14 17:04
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节（含折叠块与图注）：核心问题、常见误解、1. MoE 一层的五段执行（1.1、1.2、本章问题）、2. 一个持久 kernel 里的分工（2.1、2.2、本章问题）、3. 对称内存（3.1、3.2、本章问题）、4. 一个 token 的完整旅程（4.1、4.2、4.3、本章问题、X-Stage 补充折叠块、两处代码折叠块）、5. 收益与边界（5.1、5.2、5.3、本章问题）、来源与范围说明（C/F/N 表、构造示例、辅助解释与类比边界、简化条件及其限制）
- 本轮独立的来源核对（不读取 research/）：DeepGEMM README@559d79f、PR #304、PR #316 正文与评论、kernel 源码 sm100_fp8_fp4_mega_moe.cuh / sm100_fp8_fp4_mega_moe.hpp / heuristics/mega_moe.hpp / scheduler/mega_moe.cuh / layout/sym_buffer.cuh / layout/mega_moe.cuh / csrc/apis/mega.hpp / tests/test_mega_moe.py（均按 commit 559d79f 取原文）、arXiv:2607.23264 正文与摘要。据此确认：五段定义、grid=num_sms 与 __launch_bounds__、三组线程 128/128/128–256 与寄存器 48/40/208（或 96/88/160，上限 64512）、SymBuffer base+offsets[72] 与 map 一次加法、dispatch "Round-robin rank selection via iterative min-peeling"、kPullThreshold=4096、L1/L2 同一缓冲注释、SwiGLU 截断与乘路由权重（.cuh 行 1071）、remote store 注释、combine 归约注释、三处 NVLink barrier tag、BlockPhase 四相与 L1 预热防死锁、共享专家不经 dispatch、get_block_config_for_mega_moe 的 N_exp 公式与六档 block_m（16…192，batch 512/8192/1 分别算得 64/1024/0.125 对应 96/192/16）、tmp 表 1.50–1.96x、8 rank 平均与"每 rank token 数"口径、无共享专家 torch.equal 逐位一致/有共享 <1e-8、FP8 路径 128 整除、top-k≤32 与专家数被 rank 数整除、CUDA_ARCH>=1000 守卫与 sm_100f 断言、UMMA/2-SM cluster/TMEM、README 三步 API、默认 8 进程、DG_COMM_KERNEL_DEBUG、PR #304 八位 Mega MoE 贡献者与 04-17 合并、PR #316 04-24 合并、N5 zheanxu 04-27 口径确认、N9 yiakwy 08-12 Hopper 评论、X-Stage §1/§2.2/§2.3 的 expert wave、约 1.5x 上限、实测 1.56x、1.18x 几何均值/1.62x 最高/84 组。全部可定位、数值与原文一致；所有行号定位抽查命中；[C]/[F]/[N] 引用编号与来源表双向一致（N11 已注明"正文未引用，备考"）。

## 问题

- [重要·技术] 5.1「官方基准：测了什么、快了多少」三个读数第一点：对"batch 为 1 时加速最大"给出无来源的因果解释——"token 越少、每段工作越薄，串行执行的启动与等待占比越高，融合的相对收益就越大……这正是延迟敏感场景（低并发推理、强化学习采样）的形态"｜引文依据：PR #316 表一/表二 batch=1 行原文为「1｜56.5｜5｜1311｜1｜1.96x」与「1｜108.1｜7｜1758｜1｜1.61x」（第三列起为 Time/Compute/Global Mem/Interconnect/Speedup），即 batch=1 时互联带宽仅 1 GB/s、通信量最小；两表仅给出「Speedup (vs legacy)」数值，来源未对"为何小 batch 收益最大"作任何机制说明｜修复要求：为"小 batch 收益最大"的归因补来源，或改写为明确标注的推断（如"一个可能的解释是……"），不得保留为无来源的因果结论｜修复：｜复验：
- [轻微·表述] 4.2 首句"tile 大小不是拍脑袋定的。"——口语化措辞｜引文依据：不适用｜修复要求：替换为中性表述｜修复：｜复验：
- [轻微·可读性] 4.X「补充：X-Stage 论文对调度的再分析与"expert wave"术语」折叠块："把发布的时间线当作'Combine 是一段完整通信'会低估加速"句式不成立（"把［名词］当作［完整句子］"），含义难以解析｜引文依据：arXiv:2607.23264 §2.3："We call this the completion-coupled interpretation."、"the model yields an estimated speedup ceiling of about 1.5× over the serial baseline."｜修复要求：改写为通顺表述，如"若把 Combine 视为一整段必须完成的同步通信来估算调度时间线，会低估加速（保守上限约 1.5 倍，低于实测 1.56 倍）"｜修复：｜复验：
- [轻微·技术] overview.html「关键结论与边界」第二条："衡量这类系统的关键指标是重叠效率，不是通信体积"——无来源的一般性判断写成结论｜引文依据：不适用｜修复要求：删除该判断，或改写为其论据的陈述（如"加速来自把通信藏进计算空闲，而非减少通信量"）｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复