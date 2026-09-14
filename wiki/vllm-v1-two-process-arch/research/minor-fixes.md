# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-4.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**4** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**14** 条
- 有理由不改：**14** 条

## 修复与判定说明

清理 vllm-v1-two-process-arch 历轮积压的轻微问题：逐轮读完 research/review-1..4.md，汇总 18 条轻微级问题（含跨轮重复项），其中 14 条为历轮已修复或副作用导致「已不存在」；对当前仍存在的 4 条按修复要求改掉——§5 五要素「两个队列」补为「输入、输出与 abort 三类队列」（对齐 aborts_queue）、§6 类表 WorkerBase 删去错误的「（ABC）」标注（源码无 ABC 继承/abstractmethod）、§4 首段「所有调用」限定为「执行类调用」（check_health/shutdown 在 UniProcExecutor 下直调）、§4.2「N 卡场景下」改「N 卡时」。仅改被指出处，未动章节结构与文风，校验通过。
## 不改的条目及理由

- R1-§2「_process_input_queue 从 ZMQ 输入队列」：已不存在——现文为「从输入队列取消息（该队列由 ZMQ socket 线程 process_input_sockets 投递）」，已被历轮改对。
- R1-§2「post_step 仅在使用推测解码时更新草稿 token」条件缺失：已不存在——现文已补全「同步调度、使用推测解码、本回合执行了模型且 batch_queue_size == 1」，并注明异步调度与批队列下在别处刷新。
- R1-§1 职责表「Detokenizer」类名不存在：已不存在——现文为 IncrementalDetokenizer（按请求，由 OutputProcessor 持有）。
- R1-§3「跑手」口语译名 / 「完全不感知卡数」/「容易混淆」：已不存在——现文分别为「forward 执行者」「不区分卡数」「多个类名后缀相近」。
- R1-§3 段末「分层的设计收益」无来源评价：已不存在——现文已降级为「由此可推，三层职责彼此隔离」。
- R1-§7 环境版本串与仓库检出不符、trace 数字无留档：已不存在——现文注明实测环境版本串已无对应检出，trace 数字声明「不可再复核」。
- R2-行331 以「本页」为主语的自我指代：已不存在——现文为「源码核对基线为本地检出 v0.20.0-718-gee4ed6d71d」。
- R2-行224「注意与 EngineClient 是两个东西」元话语：已不存在——现文直陈「是两个不同的接口」。
- R2-行7 summary「WorkerWrapper」：已不存在——head dojo:summary 现作 WorkerWrapperBase。
- R2-三张 SVG 未用 foreignObject：已不存在——三张内联 SVG 的代码标识符均已用 foreignObject 承载（R4 复核确认）。
- R3-行138 pre/post_engine_step 未标注来自 hcf_mixin：已不存在——现文已注明「由本地 fork 的 vllm.hcf_mixin（vllm/hcf_mixin/v1/engine/core_mixin.py）注入，上游 vLLM 的同名流程中只有 post_step」。
- R3-三张 SVG foreignObject（重复项）：已不存在——同上。
- R3-行224/327/331/334 元话语/自我指代/复现叙事：已不存在——「以下为」「本页」「已用最小用例验证」均已删除。
- R3-行7 summary「WorkerWrapper」（重复项）：已不存在——同上。
