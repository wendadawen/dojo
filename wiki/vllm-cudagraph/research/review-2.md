<!-- review-meta
round: 2
page: wiki/vllm-cudagraph/index.html
reviewed_content_sha256: ea0ee03c7969ae96
-->
# torch.compile 图捕获与 CUDA Graph 审查记录（第 2 轮）

- 页面版本：af9aa99e92fe3d7133709251113295332c33f4d1（git hash-object wiki/vllm-cudagraph/index.html）
- 审查时间：2026-09-13 19:02
- 审查者：独立子代理（编排者派发的独立审查者）
- 适用规范：`guides/note.md`（该页 `dojo:type=note`），问题分级与记录格式依 `guides/concept/check.md` 第 3 节
- 已完整阅读章节：head/meta 与导语 →「1. torch.compile 的两阶段流程」→「2. Dynamo 图捕获的处理边界」→「3. CUDA Graph 的录制与重放」→「4. 捕获尺寸的确定」→「来源与范围说明」→ 页内全部脚本。全文 347 行已逐段通读。
- 复核方式：vLLM 源码以 `raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/compilation.py`、`vllm/config/vllm.py`、`vllm/v1/cudagraph_dispatcher.py` 为准；Dynamo 行为在本机 torch 2.8.0 上实测（`torch.compile` 默认模式 + `torch._dynamo.explain` + `fullgraph=True`）。

## 问题

- [阻断·技术] 2. Dynamo 图捕获的处理边界（表格第 3 行 + 紧随其后的段落）：把 C 扩展写成了「硬性失败」。实际默认行为是 graph break 并继续执行，抛异常只在 fullgraph 下发生。原文「C 扩展（如 hashlib.sha1）｜无追踪规则，标记为 skipped 并抛异常｜图捕获中断」，以及段落「而 C 扩展被标记为 skipped 时是硬性失败，Dynamo 直接抛出 Unsupported 异常终止捕获。」｜引文依据：本机实测（torch 2.8.0，默认 `torch.compile`）对含 `hashlib.sha1` 的函数，只发出 `UserWarning: Dynamo does not know how to trace the builtin `_hashlib.openssl_sha1`.`，随后正常返回结果 `result: tensor([ 2.0409, 0.1995, -0.4376])`，未抛任何异常；仅当 `torch.compile(f, fullgraph=True)` 时才抛 `Unsupported: Attempted to call function marked as skipped`。即「抛 Unsupported 终止捕获」是 fullgraph 模式（非默认）的行为。同时页面把该警告称为「典型报错信息」也不准确——默认路径下它是警告而非报错。｜修复要求：将表格第 3 行与段落改为默认模式下的真实行为（graph break + `UserWarning`，执行继续；只有 fullgraph/nopython 模式才以 `Unsupported` 失败）；若要保留「硬性失败」，必须显式限定为 fullgraph 模式。删除或改写「典型报错信息为」一句，明确其为警告。｜修复：｜复验：

- [阻断·技术] 2. Dynamo 图捕获的处理边界（表格第 2 行）：把 `for`、`len` 一律写成 graph break，与实测不符。原文「Python 原生控制流（if、for、len）｜graph break，回落到 Python 执行｜可继续，性能下降」。｜引文依据：本机实测 `torch._dynamo.explain`：`for i in range(3)` 的循环「FOR breaks: 0 ops: 1」；`len(x)`（x 为张量）「LEN breaks: 0」；仅 data-dependent 的 `if`（`if x.sum() > 0`）「IF(tensor) breaks: 1」。即 Dynamo 会展开 `for`、会追踪 `len`，二者不产生 graph break，只有数据相关的分支才 break。｜修复要求：按实测重写该行——张量运算与 `for`/`len` 等可静态展开的控制流均正常编译，仅依赖张量取值的数据相关分支（如 `if tensor.sum()>0`）触发 graph break；不得把 if/for/len 并列为一类。｜修复：｜复验：

- [阻断·技术] 4. 捕获尺寸的确定（公式）：默认捕获尺寸公式缺少上界 `+1`，既与来源不符，也与本页上一句给出的「最大捕获尺寸」自相矛盾。原文「默认捕获尺寸按 `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max, 16))` 生成，最大捕获尺寸默认取 `min(max_num_seqs * 2, 512)`」——按 `range(256, 512, 16)`，末元素为 496，永远取不到 512，与「最大捕获尺寸 = 512」直接冲突。｜引文依据：vLLM 源码 `vllm/config/compilation.py:707-708` 与 `vllm/config/vllm.py:2346` 均为 `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))`（含 `+ 1`）；`compilation.py:1145` 另有断言 `assert self.cudagraph_capture_sizes[-1] == self.max_cudagraph_capture_size`，即来源保证末元素等于最大捕获尺寸。｜修复要求：把公式改为 `list(range(256, max, 16))` → `list(range(256, max + 1, 16))`（或等价写法），使末元素与所述最大捕获尺寸一致；并复核全文对该公式的引用位置。｜修复：｜复验：

- [重要·技术] 4. 捕获尺寸的确定：把最大捕获尺寸写成无条件公式，漏掉来源中的两个条件因子（uniform decode 长度与 Blackwell 平台上限）。原文「最大捕获尺寸默认取 `min(max_num_seqs * 2, 512)`」；同段「总 token 数近似等于并发请求数（num_seqs，即 batch size）」。｜引文依据：`vllm/config/vllm.py::_set_cudagraph_sizes`：`default_max_graph_size = 1024 if current_platform.is_device_capability_family(100) else 512`，`max_cudagraph_capture_size = min(max_num_seqs * decode_query_len * 2, default_max_graph_size)`，其中 `decode_query_len = self.uniform_decode_query_len`；`compilation.py:710-711` docstring 明写「capped at 512 by default, or 1024 on data center Blackwell GPUs」。即 `×2` 前的系数实为 `decode_query_len`（= num_speculative_tokens + 1，非投机时为 1），上限在 Blackwell 为 1024。｜修复要求：将公式写成 `min(max_num_seqs × decode_query_len × 2, default_max_graph_size)` 并注明 `decode_query_len` 含义与非投机时取 1、上限 512（Blackwell 为 1024）；「总 token 数等于 num_seqs」需限定为非投机 decode（decode_query_len=1）。｜修复：｜复验：

- [重要·来源] 来源与范围说明（来源列表）：把捕获尺寸默认生成规则的来源记为 `vllm/config/compilation.py`，但该计算实际不在该文件。原文「vLLM 源码：`vllm/config/compilation.py`（捕获尺寸默认生成规则）、`vllm/v1/cudagraph_dispatcher.py`（batch 到捕获尺寸的映射与 padding）」。｜引文依据：`vllm/config/compilation.py` 仅在 docstring 中描述尺寸规律，字段 `max_cudagraph_capture_size` 默认值为 `None`；`min(max_num_seqs * decode_query_len * 2, default_max_graph_size)` 的实现位于 `vllm/config/vllm.py::_set_cudagraph_sizes`（`vllm/config/vllm.py:2210-2227`）。｜修复要求：把「捕获尺寸默认生成规则」的来源补正为 `vllm/config/vllm.py::_set_cudagraph_sizes`（可同时保留 compilation.py 的 docstring 描述）；`cudagraph_dispatcher.py` 一条经核属实可保留。｜修复：｜复验：

- [重要·技术] 3. CUDA Graph 的录制与重放（适用阶段条）：对 prefill 的论断过宽，与 vLLM 实际设计不符。原文「而预填充（prefill）单次处理大量 token、计算形状动态变化，不适用于固定形状的图录制」。｜引文依据：`vllm/config/compilation.py:1525` 注释「# restrictions on PIECEWISE (prefill) cudagraphs.」，即 vLLM 确实为 prefill 捕获（分段式 PIECEWISE）CUDA Graph；`compilation.py` 中 cudagraph 模式含 PIECEWISE（面向 prefill/混合批）与 FULL（面向 uniform decode）。｜修复要求：改为「FULL（整图）重放要求形状固定，因而用于 uniform decode；prefill 批形状变化大，通常走 PIECEWISE 分段图或落到 eager」之类的限定表述，删除「prefill 不适用图录制」的一刀切结论。｜修复：｜复验：

- [重要·技术] 4. 捕获尺寸的确定：用「seq_len 在 decode 中恒定」解释 seq_len 不参与尺寸组合，该依据不成立。原文「decode 中每个请求每步仅产生一个 token，故总 token 数近似等于并发请求数（num_seqs，即 batch size）；seq_len 在 decode 中恒定，不参与尺寸组合」。｜引文依据：decode 每步为每个请求追加一个 token，其上下文长度（seq_len）逐步增长而非恒定；seq_len 之所以不进入捕获尺寸，是因为注意力经分页 KV cache + block table 读取，与上下文长度解耦（`vllm/v1/cudagraph_dispatcher.py` 的 `BatchDescriptor` 只含 `num_tokens`/`num_reqs`，不含 seq_len）。｜修复要求：删除「seq_len 在 decode 中恒定」这一错误依据，改用「捕获尺寸以单步总 token 数为单位、经分页 KV cache 解耦于上下文长度」的表述。｜修复：｜复验：

- [轻微·技术] 3. CUDA Graph 的录制与重放（开销特征条）：录制订为「量级可达几十至数百毫秒」，属无来源的定量判断，且未被「来源与范围说明」的推断声明覆盖。原文「录制开销较大（需逐次执行、记录地址与依赖、分配内存池，量级可达几十至数百毫秒）」；来源说明仅声明「CUDA Graph 显存占用的具体量级，属于推断」，未涵盖录制耗时。｜引文依据：不适用｜修复要求：删除该具体量级，或将其纳入「属于推断、未逐一实测」的标注范围（同段内存量级已在来源说明中标为推断，两条应对齐）。｜修复：｜复验：

- [轻微·内容] head 的 `dojo:summary`：承诺了正文未覆盖的主题。原文 `dojo:summary` 含「页面同时梳理 Python 控制流、C 扩展与动态形状的处理边界」，但正文四节中没有专门讨论动态形状（dynamic shapes）的内容，仅第 3 节一句「计算形状动态变化」。｜引文依据：不适用｜修复要求：从 summary 删除「动态形状」或补足对应正文；summary 与 description 的承诺须与正文章节一致。｜修复：｜复验：

- [轻微·表述] 4. 捕获尺寸的确定（末句）：出现调试/走读叙事腔。原文「这一对照在一次真实走读中表现为 decode 命中 FULL 重放、983 token 的 prefill 落到 eager」。｜引文依据：不适用｜修复要求：去掉「在一次真实走读中」这类临场叙事，改为直接引用被链页面的结论（如「据〈多模态统一输入与 cudagraph〉的走读数据，……」，并保留来源链接），或删除该句。｜修复：｜复验：

- [轻微·来源] 来源与范围说明：来源未给出可定位标识。原文「PyTorch torch.compile 官方文档：两阶段（图捕获 / 图编译）与 Inductor 后端说明。」；两条 vLLM 源码路径亦未给出 commit/版本。｜引文依据：不适用｜修复要求：为 PyTorch 文档条目补充 URL（或文档小节名），为 vLLM 源码条目补充 commit hash 或版本号，使论断可定位复核。｜修复：｜复验：

- [轻微·格式] head/页眉元信息：可视标注与 `dojo:type` 不一致。原文页内 `<p class="page-meta">更新于 2026-08-06 · 类型：代码机制</p>`，而 head 为 `<meta name="dojo:type" content="note">`。｜引文依据：不适用｜修复要求：统一可视「类型」标注与 `dojo:type`（note/学习记录），避免两类元信息冲突。｜修复：｜复验：

## 结论

- 统计：阻断 3 / 重要 4 / 轻微 5
- 处置：修复。第 2 节 Dynamo 边界表与「软性/硬性失效」结论、第 4 节捕获尺寸公式均存在必须改正的事实错误（已在本机 torch 2.8.0 与 vLLM main 源码上复核）。修复后需重新核对来源并重跑 `.dojo/scripts/validate.py`（本轮 `validate.py` 已通过：`validation ok: wiki/vllm-cudagraph/index.html`）。
- 未决事项：页面类型为 note，`guides/note.md` 未要求 overview.html，「核心问题/本章问题」问题块亦非 note 规范要求项，本轮不作缺项认定。
