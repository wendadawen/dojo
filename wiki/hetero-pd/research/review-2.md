<!-- review-meta
round: 2
page: wiki/hetero-pd/index.html
reviewed_content_sha256: e7e443f8ca6fd852
-->
# 异构 PD 分离全链路 审查记录（第 2 轮）

- 页面版本：index.html df754d80f98cc25fd72af2771fde85dc19654200
- 审查时间：2026-09-13 19:00
- 审查者：独立子代理
- 页面类型：note（head `dojo:type=note`），适用规范 `guides/note.md`
- 已完整阅读：导语/meta、第 1–7 节、复现要点、来源与范围说明，含 3 张内联 SVG 图与全部图注。本页为 note，规范未要求「核心问题/问题块」，该维度不适用。
- 已核对来源：**无可核对来源**。页面无任何外部链接（`grep -o 'href="[^"]*"'` 仅得 `../../index.html`、`../../libs/*.css` 与 6 个站内 `../*/index.html`）；仓库内无本页的 `research/measured.md`、`research/official/`（本页 `research/` 下仅有 `review-1.md`），而相邻页面（`wiki/kv-cache-layout`、`wiki/mrope` 等）均有 `measured.md`/`evidence.md`；页面引用的 vLLM 分支文件（`flexible_connector.py` / `hpc_attn.py` / `hyvl_int8_attn.py`）在仓库内不存在（全仓 grep `kv_meta_addr|flexible_connector|K_SCALE_BYTES|prep_xfer_dlist|compat_manifest` 仅命中本页与 review-1.md）。故全部事实性论断按「未核对」处理。
- 机械验证：`.dojo/scripts/validate.py wiki/hetero-pd/index.html` → `validation ok`；无头 Chrome 实测渲染：KaTeX 26 处（含 2 个 `$$` display）渲染正常，图 3 的 foreignObject 内 KaTeX 亦渲染，SVG 图 3 张，目录生成 12 项。

## 问题

- **[阻断·数字]** ｜来源：P 侧 `[Prefill Worker] KV send done` 日志的内建速率（页面自载，无外部可查来源）｜位置：第 281 行（并见第 234 行）｜问题：第 281 行断言「日志中 GB/s 按 $10^9$ 基数计算」，但同段与第 234 行给出的三个速率只在 $2^{30}$ 基数下成立，按 $10^9$ 应分别为 1.18 / 1.84 / 0.46——页面数字与自身基数断言互相矛盾。｜引文依据：8110080 B ÷ 6.9 ms = 1.175×10⁹ B/s（$10^9$ → 1.18 GB/s）＝ 1.095 GiB/s（$2^{30}$ → 1.09）；8110080 ÷ 4.4 ms = 1.843×10⁹（$10^9$ → 1.84）＝ 1.717 GiB/s（$2^{30}$ → ≈1.70）；8110080 ÷ 17.8 ms = 0.456×10⁹（$10^9$ → 0.46）＝ 0.424 GiB/s（$2^{30}$ → 0.42）。页面第 234 行原文「6 blocks / 8110080 B / 6.9 ms / 1.09 GB/s」、第 281 行「约 1.70 GB/s」「约 0.42 GB/s」，三值均与 $2^{30}$ 吻合。｜修复要求：回到实测日志及其速率格式化代码确认基数。若为 $2^{30}$，把第 281 行改为「日志中 GB/s 按 $2^{30}$（GiB）基数计算，日志沿用 GB/s 字样」，并与第 234 行统一；若确为 $10^9$，则三个速率重算为 1.18 / 1.84 / 0.46 并同步正文。

- **[阻断·数字]** ｜来源：D 侧 decode 计时与 token 计数（页面自载，无外部可查来源）｜位置：第 338 行（并见第 341 行）｜问题：表格同一行三项互不相等——「64 token @ 17.3 tok/s」与 3.48 s 不符；第 341 行「PD 20.3 tok/s」与本节合计时间也对不上。｜引文依据：64 ÷ 17.3 = 3.699 s ≠ 3.48 s；反之 3.48 s ⇒ 64 ÷ 3.48 = 18.4 tok/s。本节三项 150 + 4.4 + 3480 = 3634.4 ms ⇒ 64 ÷ 3.634 ≈ 17.6 tok/s（接近 17.3，但与该行标注的 decode 3.48 s 归属不符）；第 341 行 20.3 tok/s ⇒ 64 ÷ 20.3 = 3.15 s，与本节 3.63 s 相差 0.48 s（第 235 行缓存命中一轮端到端 2.64 s ⇒ 24.2 tok/s，亦非 20.3）。｜修复要求：明确 17.3 tok/s 的测量口径（是 decode 段 token/时间，还是端到端 64/3.63 s）；若为端到端则不得挂在「decode 3.48 s」行；同步核对第 341 行 20.3 / 23.5 tok/s 的口径与数值，使同一节内速率与时间自洽。

- **[阻断·来源]** ｜来源：页面「来源与范围说明」所称「内部实测记录（异构 PD 部署，2026-08）」与 vLLM 分支源码｜位置：第 353–361 行及全文代码级论断（第 154–176、289–302、308–317、328 行）｜问题：页面无任何外部链接，所引内部实测记录与 vLLM 分支在本仓库均不可达，本页 `research/` 下也没有 `evidence.md`/`measured.md`，审查者无法为任何事实性论断给出引文依据（按 check.md §2.2「无法给出片段的条目视为未核对」）。｜引文依据：见上「已核对来源」（href 清单、`git ls-files wiki/hetero-pd/` 仅 index.html、`find . -name measured.md` 无本页、全仓 grep 无分支源码）。页面第 357 行自述来源为 `flexible_connector.py` 的 `node_info`/`compat_manifest`/`export_local_agent()`/`prep_xfer_dlist()`/`import_remote_agent()`。｜修复要求：按 check.md §2.2 与 note.md「每条事实有来源支撑，或已标记为推断」，二选一：(a) 在本页 `research/measured.md` 登记可核对片段（日志原行、config 关键字段、代码符号所在分支/commit），正文据此继续写「实测得到」；(b) 对不能登记的代码级机制（第 155 行「15 字段」、第 163 行「compat_hash 只收集不强制」、第 175 行 Mooncake 三条件筛选与 `MC_TE_FILTERS`、第 289 行 `K_SCALE_BYTES=4`、第 309 行 `VLLM_TRANSFER_TIME_OUT` 默认 120 s 等）逐条删除或显式标注「未核实/推断」。

- **[重要·技术一致性]** ｜来源：`hyvl_int8_attn.py`（`K_SCALE_BYTES` / `split_hyvl_kv_cache()`，不可达）｜位置：第 246、247 行与图 3 第 260 行｜问题：K 的 scale 粒度在同一处按两种互斥方式描述——「per-token-per-head 量化」（block 内应为 64 个）与「每 32 个 token 一组 fp32」（应为 2 个）并存；第 247 行又写「后 2 行是 scale（64/32=2 组）」，而字节账支持 per-token 的 64 个。｜引文依据：第 246 行「K 的 scale 是动态值（per-token-per-head 量化，每 32 个 token 一组 fp32）」；第 247 行「后 2 行是 scale（64/32=2 组）」；图 3 第 260 行「K scale · fp32 · 2 行（每 32 token 一组，动态）」；第 242 行「132 = 128 + 4」。每行 4 B 尾部 × 64 行 = 256 B = 2 × 128 B = 64 个 fp32（对照 (64,132)→(66,128)：8448 = 66×128）。｜修复要求：统一 K scale 粒度的单一表述，并让「132 = 128 + 4」「(64,132)→(66,128)」「后 2 行」与之一致；若为 per-token 则删去「每 32 个 token 一组 fp32」，若为每 32 token 分组则补上 4 B/行尾部与 256 B 尾部区如何容纳 2 组 fp32 的说明。

- **[轻微·表述]** ｜来源：不适用（表述维度）｜位置：第 348、350 行｜问题：把内部环境信息与临场运维叙事写入正文，note.md 明列「内部环境信息不写入页面」，并属 check.md 第 12 项「临场评价/祈使句」。｜引文依据：第 348 行「启动脚本经 ssh bash -l 拉起、只透传白名单变量……另一侧的启动脚本 source 的平台环境文件里硬编码 VLLM_ENABLE_PREFIX_CACHING="1" 且只读」；第 350 行「清残留进程时不要用 pkill -f "VLLM::" 全局名字匹配——会误伤平台管控进程（实测曾把一台节点打成不可达），应先定位 vllm serve 主进程再按进程树 kill」。｜修复要求：改写为与内网无关的通用要点（如「重启前确认两侧显存归零以避免 OOM；清理残留进程时按进程树精确定位，避免误伤管控进程」），删除 ssh/白名单/平台环境文件与事故细节。

- **[轻微·公式书写与图示]** ｜来源：不适用（书写规范）｜位置：第 172、298 行；第 230、231、308、309、311、341 行；图 3 第 255、273 行 `<text>`｜问题：正文/代码/图内 `<text>` 直接出现 Unicode 数学符号（×、→），与页面自身 KaTeX 公式（第 242、296 行用 `\times`）写法不一致；图 3 的算式以 `<text>` 直排，未按 note.md「图内公式……用 foreignObject 承载」处理（同图「6 blocks × 64 slot = 384 $\geq$ 362」已用 foreignObject + KaTeX）。｜引文依据：第 172 行 `<code>addr[i] = base_addr + i × 1351680</code>`；第 298 行「elems × block_size 必须能被 head_size 整除」；第 230/231/308/309/311/341 行多处 `→`；图 3 第 255 行「1 层 × 1 KV head = 16896 B（分配视角 2 × 64 × 132）」、第 273 行「1 个 block = 80 层 × 16896 B = 1351680 B」。｜修复要求：图 3 的算式移入 `<foreignObject>` 用 KaTeX 承载，与同图既有写法统一；正文/代码内算子写法说明统一（`<code>` 内可保留记法，正文算式应走 KaTeX 或改用一致记法）。

- **[轻微·事实核对]** ｜来源：config.json（页面列为来源，不可达）｜位置：第 245 行｜问题：「64 个 Q head 共享 8 个 KV head（8:1）」与同页流宽 4096 存在张力：64 × head_size 128 = 8192 ≠ 4096；仅当该模型 head_dim×heads 不等于 hidden（q_proj 升维）时才成立，需按 config 核对，而 config 不可查。｜引文依据：第 222 行「ViT 把图像编码为 [340, 4096]，拼进文本得到 [362, 4096]」；第 245 行「64 个 Q head 共享 8 个 KV head，8:1」；第 242、289 行 head_size = 128。｜修复要求：按 config.json 的 `num_attention_heads`/`num_key_value_heads`/`head_dim`/`hidden_size` 核对；若 Q head 实为 32，同步 GQA 比为 4:1；若确为 64，补一句说明 head_dim 与 hidden 不等（q_proj 升维），避免读者按 4096/128=32 反推。

## 核对结论（通过项）

- 公式复算（第 4 节）：1351680 = 2 × 64 × 1 × 132 × 80 = 16896 × 80 ✓；16896 = 2 × 64 × 132 ✓；362 token → ⌈362/64⌉ = 6 blocks，6 × 64 = 384 ≥ 362 ✓；8 rank × 8110080 B = 64,880,640 B = 61.875 MiB ≈ 61.9 MiB ✓；V scale 空置 256 B / 8448 B = 3.03% ≈ 3% ✓。
- 公式复算（第 5 节）：head_size=128、block_size=64 时 gcd=64 → unit=2 → elems = ⌈4/2⌉×2 = 4，与 D 侧 `K_SCALE_BYTES`=4 一致（132 = 128+4）✓；反例 head_size=192、block_size=64 时 gcd=64 → unit=3 → elems = ⌈4/3⌉×3 = 6 ≠ 4 ✓；reshape (64,132)→(66,128) 元素守恒：8448 = 66×128 ✓。
- 第 7 节占比：150/3634.4 = 4.13% ≈ 4%，4.4/3634.4 = 0.121% ≈ 0.12%，3480/3634.4 = 95.8% ≈ 96%，与导语「仅占端到端 0.12%」一致（除上一「阻断·数字」所述的行内速率外）。
- 符号一致性：`block_size`/`head_size`/`num_hidden_layers`/`total_num_kv_heads`/`tp_size`/`compat_manifest`/`capabilities`/`block_stride` 等全页写法一致，未见同义异写。
- 元数据与结构：head 含纯文本 `description`、`dojo:summary`、`dojo:type=note`、`dojo:topics=推理系统`、`dojo:tag=推理系统`（validate.py 接受）；标题与导语职责不同；含「来源与范围说明」；3 张图均为内联 SVG，图外公式用 KaTeX，无 ASCII 字符图。
- 页面链接：`../kv-cache-layout/`、`../increase-kv/`、`../beyond-buzz-disaggregation/`、`../kv-cache/`、`../gpu-communication/`、`../model-parallelism/` 六个站内页均真实存在，无「（待生成）」占位。
- 页面功能：无头 Chrome 实测——KaTeX 26 处（含 2 个 `$$` display）渲染正常，图 3 的 foreignObject 内 KaTeX 亦渲染，SVG 图 3 张，目录生成 12 项；`validate.py` 通过。
- 来源路径：页面未指向任何已移除的 `research/` 文件路径（`grep "research/"` 无命中）。

## 结论

- 统计：阻断 3 / 重要 1 / 轻微 3
- 处置：修复。第 1、2 条（阻断·数字）须回到实测日志与速率计算代码重算并统一；第 3 条（阻断·来源）须登记可核对来源，或把代码级论断逐条降级/删除；第 4 条统一 K scale 粒度表述；轻微项按各自要求处理。修复后重跑 `.dojo/scripts/validate.py`，下一轮从修复后的完整页面重审。