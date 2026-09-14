<!-- review-meta
round: 8
page: wiki/hetero-pd/index.html
reviewed_content_sha256: 5f409380d3869363
-->
# 异构 PD 分离全链路 审查记录（第 8 轮）

- 页面版本：f8dbc0ef6982（工作树，`git status` 显示 M wiki/hetero-pd/index.html，含第 8 轮未提交改动）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序审查）
- 规范判定：`dojo:type = note` → 适用 `guides/note.md`
- 已完整阅读：head 元数据与 summary、导语、第 1–7 节、来源与范围说明、三张内联 SVG、页内两段脚本；无 overview.html（符合 note 单文档设计）
- 约束：本页 `research/` 未读取

## 核对依据（本轮实测/回源）

- **渲染实测**：无头 Chrome 渲染页面后 dump DOM，正文无残留 `$...$`，KaTeX 生成 9 个 `.katex` 节点（含图 3 三处 foreignObject 内公式），`auto-render` 配置正常；`python3 .dojo/scripts/validate.py wiki/hetero-pd/index.html` 返回 `validation ok`。
- **数值复算（全部自洽）**：2×64×1×132×80 = 1351680；16896×80 = 1351680；6×1351680 = 8110080；6×64 = 384 ≥ 362；64×132 = 8448；8448/128 = 66；8×8110080 B = 61.875 MiB ≈ 61.9 MiB；64/3.63 = 17.63 ≈ 17.6 tok/s；4.4 ms / 3.63 s = 0.121% ≈ 0.12%。
- **速率与耗时同口径**：按页内自述基数 2^30，8110080/6.9 ms = 1.094 GiB/s（页述 1.09）、8110080/4.4 ms = 1.717 GiB/s（页述「约 1.70」，落在 4.4 ms 的舍入区间 [1.697, 1.736] 内，非矛盾）、8110080/17.8 ms = 0.424 GiB/s（页述 0.42）。
- **公式 5 复算**：unit = head_size / gcd(block_size, head_size)，elems = ⌈raw/unit⌉×unit；代入 (64,128) → gcd=64、unit=2、elems=4；反例 (64,192) → gcd=64、unit=3、elems=6 ≠ 4，与「elems×block_size 须被 head_size 整除」（256/128、384/192）一致。
- **时序复算**：15:54:03.902 → 15:54:06.545 = 2.643 s（页述 2.64 s）；04.017 − 03.904 = 0.113 s，与 ttft 0.126 s 差 13 ms，页内已注明口径差异，属自洽。
- **图内几何像素测量**（getBoundingClientRect + 截图目视）：图 2、图 3 无文字越界或重叠；图 1 存在一处标签与图形相撞（见问题）。
- **站内链接与资源**：`../kv-cache-layout/ ../increase-kv/ ../beyond-buzz-disaggregation/ ../kv-cache/ ../gpu-communication/ ../model-parallelism/ ../../index.html` 及 katex/prism/dojo-note.css 等 libs 全部存在；链接文本与目标页主题一致（kv-cache-layout 确含 NHD/HND 排法，increase-kv 确为「D 侧命中块不再从 P 侧重传」）。
- **外部事实回源**：Mooncake 公开仓库确有 `MC_TE_FILTERS`（transfer-engine 设备白名单，读 `/sys/class/infiniband`）与 `Worker: Process failed for slice` 错误串（`gh search code` 命中 kvcache-ai/Mooncake 源码与官方 troubleshooting 文档）；vLLM 公开仓库确有 `kv_transfer_params`。页内声明的其余符号（`flexible_connector.py`、`VLLM_TRANSFER_TIME_OUT`、`VLLM_KV_META_BASE_PORT`、`VLLM_PD_LAYERWISE_THRESHOLD`、`VLLM_ENABLE_PREFIX_CACHING`、`kv_send_start`/`kv_recv_done`）在公开 vLLM/Mooncake 仓库检索不到，页面来源表已自述为「内部 vLLM 分支，公开仓库不可核对」，按 note.md「无法核实的内容标记为未核实或推断」口径不再逐条追责。
- **表述维度**：全文无「我们/你/读者」等会话指代，无「本页/本文」主语式自我指代，无调试叙事、临场评价与 AI 拼接腔；三处推断（agent_metadata 内部结构、热/冷路径区分、Moe/大并发收益）均在句内标注。

## 问题

- [重要·机制矛盾] 第 150 行「请求到来之前，两侧要把 RDMA 通路建好。整个过程没有中心注册——P 的地址由 proxy 在请求里带给 D」与第 98 行「（proxy）在 `kv_transfer_params` 里带上 P 的控制面地址，D 由此知道去哪里等 KV」、图 1 中 D 节点框内的 `kv_transfer_params: remote_prefill=P:33981`、以及第 217 行「建连完成后，一次请求在数据面走五步」互相冲突：若 P 的控制面地址随请求下发，则「建连」只能发生在请求到达之后，不可能发生在「请求到来之前」；而第 3 节明确把建连排除在请求时序之外（时序表首行即为请求到达）。页面未提及任何启动期预热请求或其它 bootstrap 通道，两处说法无法同时成立。｜依据：页内第 98、150、153、171、217、223 行与图 1/图 2 自证；第 165 行亦以「scheduler 只做转发与配对」确认控制面连接已存在。｜修复要求：二选一并写实——(a) 若确有启动期握手（例如预热请求或地址预置），在第 2 节开头补一句说明 bootstrap 来源，并删去「P 的地址由 proxy 在请求里带给 D」或限定其适用范围；(b) 若无预热，则把「请求到来之前，两侧要把 RDMA 通路建好」改为「首个请求触发建连，之后复用」，并相应调整第 3 节「建连完成后」的措辞。

- [轻微·图示] 图 1（第 100–138 行）中标签「① D 连 P:33981 建连」置于 SVG x=532（`text-anchor="end"`，y=128），实测其包围盒起点 x=581.9 px 与 proxy 框右边界 583.4 px 重叠约 1.5 px，渲染截图中「①」压在 proxy 圆角边框上；同一条由 proxy 指向 P 的实线箭头（`M410,130 C450,120 460,90 494,76`）穿过该标签的字符区域。｜依据：无头 Chrome 逐元素 `getBoundingClientRect` 测量 + 截图目视（window 1100 宽下 SVG 缩放 1.3643）。｜修复要求：将标签整体右移（或改为 `text-anchor="start"` 并起始于 x≥545），使其避开 proxy 框右边界与请求箭头。

- [轻微·图注] 图 3 图注（第 281 行）「K、V 半块各 64 行 × 132 B（128 B 数据 + 4 B 行尾 scale）」把「4 B 行尾 scale」同时赋予 V 半块，而图内 V 框自带文案为「每行 128 B 数据 + 4 B scale 槽位 / 静态值在 checkpoint · 槽位空置」，第 247 行亦写明 V 的 scale 是静态 per-head 值、不进 cache。图注与图内读数不一致。｜依据：第 247、267、281 行。｜修复要求：图注改为「每行 128 B 数据 + 4 B 行尾（K 存动态 scale，V 槽位空置）」，与图内两框文案对齐。

- [轻微·时序] 图 2 的 aria-label 与页面表述均为「三层握手时序」，但标注③「RoCE 网卡自动选路：创建 agent 前筛出 8 张数据面网卡」被排在标号②「NIXL 元数据交换（8 对 agent）」之后；而③自身文案「创建 agent 前」与第 165 行「worker 调 `export_local_agent()` 导出…」共同表明该扫描先于 agent 创建（即先于②，实则先在①之前于进程内完成）。在一个以时间自上而下推进的时序图里按②→③编号，会把「先做的前置条件」显示为「最后一步」。｜依据：第 165、174、178–213 行。｜修复要求：把③移出时序主列（如置于图顶作为前置条件带，或标注「（建链前置）」），或改用不含时序含义的排列。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（重要 1 条为机制表述自相矛盾，定点改写 1–2 句即可关闭；3 条轻微分别为图示标签碰撞、图注与图内读数不一致、时序图编号含义不清，均不触及数字与核心结论）。

本轮说明：页面全部可复算的数字、公式与派生量（1351680 / 8110080 / 16896 / 8448 / 66 / 384 ≥ 362 / 61.9 MiB / 17.6 tok/s / 0.12% / 4.4 ms / 6.9 ms / 17.8 ms / 1.70、1.09、0.42 GB/s / elems=4 与反例 elems=6）均经本轮独立复算通过，正文、`dojo:description`、`dojo:summary` 与三张图注之间的同一数字全部一致；KaTeX 在无头浏览器实测渲染无残留 `$`；站内链接与 libs 资源全部可达；Mooncake 的 `MC_TE_FILTERS`、`Process failed for slice` 与 vLLM 的 `kv_transfer_params` 等公开可核对项与资料相符，其余符号页面已自述为内部不可核对来源。故不报告阻断，仅保留上述 1 条重要与 3 条轻微。