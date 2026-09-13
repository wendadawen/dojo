<!-- review-meta
round: 1
page: wiki/vllm-framework-map/index.html
reviewed_content_sha256: d2b9fb23df601238
-->
# vLLM 框架全景审查记录（第 1 轮）

- 页面版本：`6d761762ace960125cc4566e0722a0be20d6175b`（wiki/vllm-framework-map/index.html 工作树哈希）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 页面类型：note（规范：guides/note.md）
- 已完整阅读章节：1. 进程架构（含进程拓扑 SVG 与图注）→ 2. 五层代码地图（入口/引擎/执行/模型/编译）→ 3. 跨层子系统地图（含五层与子系统全景 SVG）→ 4. 关键设计决策 → 5. 官方设计文档索引 → 6. 走读路线与进度基准 → 来源与范围说明
- 核对来源：本地 vLLM 源码 `/Users/wendadawen/code/github/github-wendadawen/vllm`（`git describe` = v0.23.1rc0-1383-g95e073e17，HEAD 提交日 2026-07-22）与 `/Users/wendadawen/code/github/github-community/vllm`（v0.26.1rc0-454）；官方设计文档 `docs/design/`（arch_overview.md 全文、multiprocessing.md 等）；`guides/note.md`
- 机械验证：`python3 .dojo/scripts/validate.py wiki/vllm-framework-map/index.html` → `validation ok`；被引姊妹页 ../vllm-v1-two-process-arch、../vllm-mm-image-two-stage、../vllm-mm-unified-embeds-cudagraph 均存在且页面标题与链接文字一致；本页无 research/ 目录（无实测产物登记）
- 说明：页面声明版本 "vLLM 0.22.2.dev"，但环境内不存在该版本的可读代码库；本审查以唯一活跃的 vLLM 检出（v0.23.1rc0）为主，并与 v0.26.1 检出交叉复核。所有模块计数在两份检出间完全一致，故计数差异不能归因于版本漂移。

## 问题

- [阻断·技术] 来源：页面自身 §5 表格；本地源码 docs/design/｜位置：行 230（§5 首句）、行 286（来源说明）、行 233–238（§5 表格）｜问题：页面称仓库 docs/design/ "共 30 篇设计文档"，但同一节的表格逐行只列出 29 个文档名（架构总览 1 + 核心机制 5 + 编译 5 + 子系统 10 + 分布式 3 + 扩展 5 = 29），源码目录 `docs/design/` 下也确实只有 29 个 .md。正文数字与本节表格、与来源三者互相矛盾，且"30"在正文与来源说明中重复出现两次｜引文依据：页面行 230「仓库 docs/design/ 共 30 篇设计文档，按主题归类：」；行 286「…等 30 篇」；`ls docs/design/*.md | wc -l` = 29（arch_overview.md … vllm_ir.md），且表格所列 29 个名字与目录一一对应，无遗漏项｜修复要求：把行 230、行 286 的"30 篇"改为"29 篇"；若"30"另有口径，须在来源说明中写明计数依据并使其与表格一致

- [重要·技术] 来源：vllm/model_executor/layers/quantization/__init__.py｜位置：行 7（dojo:summary"28 种量化"）、行 162（§2 模型层"约 28 种量化"）、行 285（来源说明"28 种量化方法"）｜问题：页面给"28 种量化"，源码 `QuantizationMethods` 的 Literal 实际为 31 项（含 6 个 online 简写 fp8_per_tensor / fp8_per_block / fp8_per_channel / int8_per_channel_weight_only / nvfp4_per_token / mxfp8），剔除这 6 个简写后独立方法为 25 项；两种口径都不是 28。v0.23.1 与 v0.26.1 两份检出均为 31｜引文依据：`QuantizationMethods = Literal[...]` 内字符串条目计数 = 31；`QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))`｜修复要求：改为与源码一致且可复算的数字，并说明口径（例如"25 种量化方法，另有 6 个 online 简写"），同步 summary 与来源说明

- [重要·技术] 来源：vllm/model_executor/model_loader/__init__.py、vllm/model_executor/model_loader/ 目录｜位置：行 163（§2 模型层）、行 285（来源说明"约 10 种加载器"）｜问题：页面把 gguf 列为加载器（"default / sharded_state / runai_streamer / tensorizer / gguf / bitsandbytes / dummy"），但源码中不存在任何 GGUF 加载器：`find vllm -iname "*gguf*"` 无结果，`LoadFormats` 列表不含 gguf，全仓 "gguf" 仅出现于与加载无关的 `qwen2_moe.py`、`lora/layers/utils.py`。此外 `_LOAD_FORMAT_TO_MODEL_LOADER` 去重后加载器类共 7 个（DefaultModelLoader / BitsAndBytesModelLoader / DummyModelLoader / ModelExpressModelLoader / RunaiModelStreamerLoader / ShardedStateLoader / TensorizerLoader），"约 10 种"偏高｜引文依据：`LoadFormats = Literal["auto","hf","bitsandbytes","dummy","fastsafetensors","instanttensor","mistral","modelexpress","npcache","pt","runai_streamer","runai_streamer_sharded","safetensors","sharded_state","tensorizer"]`（无 gguf）；`ls model_loader/ | grep -i gguf` 无输出｜修复要求：删除 gguf；示例改为源码中真实存在的加载器类，并把"约 10 种"改为与 `_LOAD_FORMAT_TO_MODEL_LOADER` 去重类数一致的数字

- [重要·技术] 来源：vllm/v1/attention/backends/registry.py｜位置：行 154（§2 执行层"约 22 种后端"）、行 285（来源说明"约 22 种 attention 后端"）｜问题：页面给"约 22 种后端"，但 `AttentionBackendEnum` 的成员在 v0.23.1 为 39 行、v0.26.1 为 42 行（去重重复的 CUSTOM 后约 38 / 41 项），与"约 22"相差约 16；若改按后端模块文件计（backends/ 下非 __init__/registry/utils/fa_utils 的模块）也只有约 19–20 个。两种可复算口径都对不上 22｜引文依据：registry.py 中 `FLASH_ATTN / TRITON_ATTN / ROCM_ATTN / ROCM_AITER_* / FLASHINFER* / *_MLA* / MAMBA1 / MAMBA2 / LINEAR / GDN_ATTN / …` 成员列表（`grep -cE "^    [A-Z0-9_]+ = "` = 39）｜修复要求：改为与 `AttentionBackendEnum` 一致的数字并注明口径（"枚举成员数"），或删去具体数字，仅陈述列举了 flash_attn / flashinfer / triton / mla / mamba 等多个后端

- [重要·结构] 来源：guides/note.md｜位置：行 241–281（§6 走读路线与进度基准，含"三段式路线"表与"进度基准表"）、行 7（summary）｜问题：note.md 规定"每篇记录只处理一个中心结论""只记录已经验证的事实和明确确认的判断""内部环境信息不写入页面"。§6 的"生命周期主线""三段式路线""进度基准表"（✅/🟡/⬜ 完成度、"深度""对应笔记""下一站为…"）是个人走读计划与进度台账，既非已验证事实，也属内部环境信息；页面整体定位为"总索引 + 进度基准"，不围绕单一中心结论组织，且作为索引页不满足 note.md"机制说明应覆盖输入、状态、关键参数、主要步骤和输出"｜引文依据：note.md「每篇记录只处理一个中心结论」「只记录已经验证的事实和明确确认的判断」「内部环境信息不写入页面」；页面行 266「本表随走读推进更新，作为个人的进度仪表盘：」；行 281「下一站为 _preprocess 剩余三函数」｜修复要求：将 §6 的个人计划与进度表移出页面（另存为个人台账，不入 wiki 正文），本页只保留可核实的框架事实（进程/分层/子系统/设计决策/设计文档索引）并围绕单一中心结论组织

- [重要·表述] 来源：guides/note.md §表述｜位置：行 7（dojo:summary）、行 266｜问题：note.md 要求"使用正式书面语，删除会话指代、临场评价和口语化过渡""正文直接陈述内容"。页面存在自我指代"本页""本表"与个人指代"个人的"，属会话性/自指表述｜引文依据：行 7「本页是源码走读的总索引与进度基准。」；行 266「本表随走读推进更新，作为个人的进度仪表盘：」｜修复要求：改为直陈（例如"该页汇总 vLLM 的进程、代码分层与跨层子系统结构"），删除"本页 / 本表 / 个人的"等自指与个人指代

- [轻微·技术] 来源：vllm/v1/spec_decode/ 目录｜位置：行 176（§3）、行 285（来源说明"14 个推测解码模块"）｜问题：页面给"约 14 个模块"，源码 `v1/spec_decode/` 下非 __init__ 的 .py 为 16 个（custom_class_proposer、dflash、draft_model、eagle、extract_hidden_states、gemma4、llm_base_proposer、medusa、metadata、metrics、ngram_proposer、ngram_proposer_gpu、step3p5、suffix_decoding、utils、vocab_mapping），另有 dynamic/ 子目录；两检出一致｜引文依据：`ls vllm/v1/spec_decode/*.py | grep -v __init__ | wc -l` = 16｜修复要求：改为 16，或写明计数口径

- [轻微·技术] 来源：页面自身；本地源码 model_executor/models/｜位置：行 7（summary）、行 160（§2 模型层）｜问题：同一对象（模型规模）在两处口径不同：summary 写"200+ 模型"，正文写"约 300 个模型文件"。正文数字与源码相符（`models/` 下 290 个 .py，另有 transformers/ 子目录），summary 的"200+"与之不一致｜引文依据：行 7「模型层（200+ 模型、28 种量化、10 种加载器）」；行 160「约 300 个模型文件 + registry 注册表（架构名 → 类）」；`ls vllm/model_executor/models/*.py | wc -l` = 290｜修复要求：两处统一为同一表述（如"约 290 个模型文件"）

- [轻微·来源] 来源：本地代码库｜位置：行 91（page-meta）、行 285（来源说明）｜问题：页面标注版本 "vLLM 0.22.2.dev"，但环境内不存在该版本代码库——唯一活跃检出为 v0.23.1rc0-1383-g95e073e17（HEAD 2026-07-22，晚于 0.22.2），另有 v0.20.0-718 与 v0.26.1rc0-454，均非 0.22.2.dev；据此页面所声明的"本地代码库扫描（vLLM 0.22.2.dev）"无法在环境中复现定位｜引文依据：行 91「类型：框架总览（vLLM 0.22.2.dev）」；行 285「本地代码库扫描（vLLM 0.22.2.dev）」；`git -C ... describe --tags` = v0.23.1rc0-1383-g95e073e17 / v0.20.0-718-gee4ed6d71d / v0.26.1rc0-454-g5ac268497｜修复要求：核实并给出可定位的版本号（与所扫描检出的 `git describe` 一致），或在来源说明中写明扫描所用的具体提交

## 结论

- 统计：阻断 1 / 重要 5 / 轻微 3
- 处置：修复。阻断项（§5 设计文档"30 篇"与该节表格 29 个名字、与源码 29 个文件矛盾）须先把正文与来源说明统一为 29。重要项分别：把量化方法数改为源码可复算值（31 项含 6 简写 / 25 项不含）、删除不存在的 gguf 加载器并订正加载器数、订正 attention 后端数、把 §6 个人进度台账移出页面、删除"本页/本表/个人的"自指与个人指代。计数改动后同步 summary 与来源说明，并重跑 validate.py。
- 已核对无误的项（留档）：
  - 进程架构与公式与 arch_overview.md 逐条一致：Worker = DP×PP×TP（即卡数）、EngineCore = DP、API Server 默认 = DP（可配）、DP Coordinator 于 DP>1 时 1 个；4 卡 TP=4 = 6 进程、8 卡 TP=2 DP=4 = 17 进程均与原文示例相同（arch_overview.md §V1 Process Architecture / Process Count Summary）。
  - §4 三个设计决策与 arch_overview.md "Class Hierarchy" 三点逐一对应；405B 全量 bf16 ≈ 810GB、16 卡每卡 50GB 与原文"roughly 810GB weights""every GPU should only load 50GB"一致；模型构造签名 `__init__(self, *, vllm_config, prefix="")` 与原文 note 一致。
  - §1 通信描述与源码相符：EngineCore→Worker 走 `distributed/device_communicators/shm_broadcast.py`（共享内存环形缓冲）+ ZMQ（含 XPUB），请求/结果走 ZMQ，`v1/utils.py` 提供 ipc（local_only）/ tcp 地址。
  - §2/§3 目录核对无误：entrypoints/（openai、anthropic、grpc_server、mcp、cli、pooling、speech_to_text、llm.py 等）、renderers/、multimodal/、v1/engine、v1/core/sched、v1/core（block_pool、kv_cache_manager、kv_cache_coordinator、kv_cache_utils）、v1/executor（uniproc / multiproc / ray）、v1/worker、v1/attention、v1/sample（sampler、rejection_sampler、thinking_budget_state）、v1/spec_decode、v1/structured_output、v1/kv_offload、distributed/kv_transfer（mooncake、nixl、lmcache、hf3fs、moriio、flexkv 六者均存在）、distributed（eplb、stateless_coordinator、device_communicators）、lora/、v1/pool/、platforms/、compilation/（piecewise_backend、cuda_graph、passes/ir）。§3 的"4 种 grammar 后端（xgrammar / guidance / outlines / lm_format_enforcer）"属实。
  - §5 表格所列 29 个设计文档名在 docs/design/ 中全部存在。
  - 被引姊妹页均存在且标题与链接文字一致：../vllm-v1-two-process-arch（"vLLM V1 框架总览"）、../vllm-mm-image-two-stage（"多模态图片的两阶段处理"）。
  - validate.py 通过；两处图为内联 SVG（非 ASCII 字符图）。

- 修复：｜复验：