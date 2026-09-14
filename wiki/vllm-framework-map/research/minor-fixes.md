# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-4.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**2** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**9** 条
- 有理由不改：**9** 条

## 修复与判定说明

通读 research/ 下 4 轮审查记录，汇总 11 条轻微级问题。实修 2 条：(1) R2 §4 第 1 条「散装参数」口语化 + 无来源支撑的评价性补语「这是应对快速演进的扩展性设计」，改为「而非逐项参数；新增功能只需在 VllmConfig 增加字段，任何层直接读取，无需层层修改构造函数」；(2) R4 §6 括注把模型对象方法 embed_input_ids 误列为 gpu_model_runner.py 的 _preprocess「所含」函数，改为「含 _gather_mm_embeddings、_prepare_mm_inputs 等函数，并调用模型对象的 embed_input_ids」。其余 9 条：8 条经历轮修复后当前页面已不存在（spec_decode 数 16、模型数统一约 290、版本号统一完整检出号、APIServer 拼写、「显存开销最小」、cudagraph 链接、两种版本口径），1 条 KV connector「14」经第 4 轮裁定口径成立（16 处注册含 2 个示例连接器）故不改。全站无 overview.html/summary 需同步；validate.py 通过。文件：/Users/wendadawen/code/dojo/wiki/vllm-framework-map/index.html
## 不改的条目及理由

- R1 轻微·技术（spec_decode 模块数）：已不存在。§3 与来源说明现均为「约 16 个模块」/「16 个推测解码模块」，与 v1/spec_decode/ 顶层 .py（除 __init__）实测 16 个一致，第 2/4 轮已复验。
- R1 轻微·技术（模型规模 summary 200+ vs 正文约 300）：已不存在。dojo:summary 与 §2 模型层现统一为「约 290 个模型文件」，两处口径一致，无可复算矛盾。
- R1 轻微·来源（版本标注 0.22.2.dev）：已不存在。page-meta（行 90）与来源说明（行 246）现均为「v0.23.1rc0-1383-g95e073e17」，可定位复现。
- R2 轻微·技术（14 个 v1 侧 KV connector 与工厂 16 处注册不符）：不改。kv_connector/factory.py 共 16 处 register_connector，其中 ExampleConnector、ExampleHiddenStatesConnector 为示例连接器，其余 14 个为生产连接器，页面同一处以「及 mooncake/nixl 等子目录实现」另行交代子目录；第 4 轮审查已明示该口径成立、不报，故保留 14。
- R2 轻微·技术（约 290 个模型文件计数偏低且口径未说明）：已不存在。该值即第 3、4 轮逐一实测复核的「vllm/model_executor/models/ 顶层 .py = 290」，口径为顶层模型模块文件数，页面无第二口径与之冲突。
- R3 轻微·来源（page-meta 写 v0.23.1rc0 而来源写完整检出号，口径不同）：已不存在。两处现统一为完整检出号 v0.23.1rc0-1383-g95e073e17。
- R3 轻微·表述（APIServer 与 API Server 拼写不一致）：已不存在。全文已无「APIServer」，统一为「API Server」。
- R3 轻微·技术（「显存开销最小」把来源比较级写成无条件最值）：已不存在。§4 第 3 条现为「显存开销显著更小」，与 arch_overview.md 的 much smaller 强度一致。
- R3 轻微·链接（正文纯文字提及「inputs embeds 与 cudagraph」未给链接）：已不存在。§6（行 242）与来源说明（行 249）两处均已补上指向 ../vllm-mm-unified-embeds-cudagraph/index.html 的链接。
