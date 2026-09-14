<!-- review-meta
round: 5
page: wiki/hetero-pd/index.html
reviewed_content_sha256: 8f7b7fd85d5b84f0
-->
# 异构 PD 分离全链路审查记录（第 5 轮）

- 页面版本：`wiki/hetero-pd/index.html` 工作树哈希 `ef4d9e910b22683e16faa72738142870242c9d33`
- 审查时间：2026-09-14 17:40
- 审查者：独立子代理（未参与写作，未参与前序轮次审查；按要求未读取本页 `research/` 下任何文件）
- 页面类型：`note`（head `dojo:type=note`），适用规范 `guides/note.md`；记录格式按 `guides/concept/check.md` 第 3 节
- 已完整阅读章节：1 入口分发：proxy 按 request_id 双投递 / 2 建连：三层握手（第 1、2、3 层）/ 3 一次请求的完整时序 / 4 KV 的字节构成：1351680 的推导 / 5 异构可行的条件与脆弱性 / 6 失败路径与可观测点 / 7 实测结论：传输不是瓶颈 / 复现要点 / 来源与范围说明；含 3 张 SVG 图与图注、全部表格

## 来源获取与核对情况

- 页面**无任何外部链接**（`href` 仅 `../../index.html`、4 个本地 libs、6 个站内 `../*/index.html`）。来源表自述：数字来自「内部实测记录（异构 PD 部署，2026-08）」，机制来自内部代码走读。
- 可外部核对部分逐条查证（vLLM `main` 分支、Mooncake 公开仓库与文档）：
  - ✅ 可定位且与页面一致：`MC_TE_FILTERS` 在 Mooncake 文档中确实存在，定义为「逗号分隔的 RDMA 网卡白名单（allow-list），未设置时发现全部网卡」，与第 174–175 行描述一致；`kv_load_failure_policy` 取值 `recompute|fail`、默认 `fail`，与第 322 行「P 配了 recompute 但 D 没给该参数、实际生效 fail」一致（vLLM `vllm/config/kv_transfer.py`）。
  - ❌ 定位不到：来源表第 368 行归因的 `vLLM flexible_connector.py` 及 `node_info`/`compat_manifest`/`export_local_agent()`/`kv_meta_addr` 等符号，在 vLLM 公开仓库中不存在（详见问题 2 的引文依据）。同页 `VLLM_KV_META_BASE_PORT`、`VLLM_TRANSFER_TIME_OUT`、`VLLM_PD_LAYERWISE_THRESHOLD`、`VLLM_ENABLE_PREFIX_CACHING` 在 `vllm/envs.py`（main，2404 行）命中数均为 0，判定为内部分支实现，无法外部核对。
- 页内可复算项全部复算通过（无一处数字错误）：
  - `2×64×1×132×80 = 1351680`；`6×1351680 = 8110080`；`2×64×132 = 16896`；`66×128 = 64×132 = 8448`；`8×8110080 = 61.875 MiB`（图 3「61.9 MiB」）。
  - 占比：`0.150 + 0.0044 + 3.480 = 3.634 s`，KV `0.0044/3.634 = 0.121%`、decode `95.75%`、prefill `4.13%`，与正文 0.12% / 96% / 4% 及 summary「0.12%」一致。
  - 速率（按第 292 行声明的 2^30 基数）：`4.4 ms → 1.717 GiB/s`（写 1.70）、`17.8 ms → 0.424 GiB/s`（写 0.42）、`6.9 ms → 1.095 GiB/s`（写 1.09），全部吻合。
  - 第 237 行注解：`0.126 − (04.017 − 03.904) = 0.013 s = 13 ms`，与正文「相差 13 ms」一致。
  - 第 307 行公式与第 309、311 行代入：`gcd(64,128)=64, unit=2, elems=4`；`head=192, block=64 → gcd=64, unit=3, elems=6 ≠ 4`，复算一致；`elems×block_size` 可被 `head_size` 整除的约束推导正确。
  - 版式/渲染（Chrome headless 实测）：`validate.py` 返回 `validation ok`；KaTeX 在正文两个 display 公式与 3 个图内 `foreignObject` 均成功渲染（每处生成 3 个 `katex` span）；`alt`/`aria-label` 内无 `$...$`；无脚本时正文完整可读。
- 未发现：元话语、以「本页」为主语的自我指代、会话指代、调试叙事、临场评价、口语化过渡（按标记词全文检索无命中）；站内 6 个概念链接目标文件均存在，且标题与链接文字相符（`kv-cache-layout` 讲 NHD/HND、`increase-kv` 讲增量传输）。

## 问题

- **[阻断·技术一致性]** ｜来源：`hyvl_int8_attn.py` 的 `split_hyvl_kv_cache()` / `K_SCALE_BYTES`（内部实现，公开仓库不可达；本轮按页内算式与 reshape 语义核对）｜位置：第 247、248、289 行与图 3（第 263–267 行）｜问题：同页两处互相矛盾，读者无法据此得出唯一机制。第 247 行说「132 = 128 + 4：**每行** 128 字节数据加 4 字节尾部」，即 4 B scale 逐行内嵌在每个 token 行的尾部；第 248 行说把 `(64, 132)` **reshape** 成 `(66, 128)` 后「**前 64 行是 KV 数据**，每行尾部的 4 B scale 收拢成后 2 行（64 个 fp32，共 256 B），**不搬数据**」，图 3 图注同样写「64 行数据 + 2 行 scale 位」。在行主序连续张量下二者不能同时成立：reshape 不改变底层字节顺序，`(64,132)→(66,128)` 的后 2 行恰是扁平数组的最后 256 个元素（原第 62 行第 8 字节起至第 63 行末），而不是散落在各行的 64 个 scale；把逐行尾部收拢成连续 256 B 必须跨步搬运，与「不搬数据」直接冲突。｜引文依据：① 最小同构复算：`[[0,1,2],[3,4,5]].reshape(3,2)` 的最后一行为 `[4,5]`（扁平尾部），而逐行尾部是 `[2,5]`——reshape 取得的是**列序尾部**，不是**逐行尾部**；② `64×132 = 66×128 = 8448`，故 reshape 尺寸合法，分歧只在 scale 落在哪里；③ 第 309 行把同一 `(64,132)→(66,128)` reshape 解释为「元素总数必须守恒 ⇒ `elems×block_size` 可被 `head_size` 整除」，该约束只保证 reshape 合法，并不产生「前 64 行全为数据」的效果。｜修复要求：回到 `split_hyvl_kv_cache()` 与 `_fp8_per_head_scale_elems_padded()` 确定 4 B scale 的真实存放位置，二选一并同步全文：**(a) 若 scale 逐行内嵌**（即 shape `(nb,2,64,1,132)` 的行确实每行 128 B 数据 + 4 B scale），删去第 248 行与图 3 图注中「reshape 成 (66,128) ⇒ 前 64 行是数据 / 后 2 行是 scale」的表述，改写成与实现一致的读取描述（如按 stride 132 逐行取末 4 B）；**(b) 若 scale 连续存放于 block 尾部**，则改正第 247 行「每行 128 字节数据加 4 字节尾部」及第 300–302 行 shape「`head_size + K_SCALE_BYTES`」的归属。改后须保证「132 = 128 + 4」「(66,128)」「64 个 fp32 = 256 B」「每半块 8448 B」四处可相互复算，且第 309 行的 reshape 解释与第 248 行同源同向。｜修复：｜复验：
- **[重要·来源]** ｜来源：来源与范围说明表（第 368 行）｜位置：第 368 行（并牵涉 summary 与第 6 行 description 的「FlexibleConnector」命名）｜问题：页面把握手与传输机制归因到「vLLM `flexible_connector.py`」，未注明是内部 fork/分支，读者按字面去 vLLM 公开仓库无法定位该来源。已核：vLLM `main` 的 `vllm/distributed/kv_transfer/kv_connector/v1/` 目录下无 `flexible_connector.py`；`kv_connector/factory.py` 注册的 15 个连接器为 ExampleConnector / ExampleHiddenStatesConnector / LMCacheConnectorV1 / LMCacheMPConnector / NixlConnector / NixlPullConnector / NixlPushConnector / MultiConnector / HiSparseConnector / MoRIIOConnector / OffloadingConnector / DecodeBenchConnector / MooncakeConnector / MooncakeStoreConnector / FlexKVConnectorV1，无任何 Flexible* 名称；`VLLM_KV_META_BASE_PORT`、`compute_target_rank`、`kv_meta_addr` 在 vLLM 中检索不到。｜引文依据：`vllm/distributed/kv_transfer/kv_connector/factory.py` 注册表（无 Flexible*）；`vllm/envs.py`（main，2404 行）中 `VLLM_KV_META_BASE_PORT` / `VLLM_TRANSFER_TIME_OUT` / `VLLM_PD_LAYERWISE_THRESHOLD` / `VLLM_ENABLE_PREFIX_CACHING` 命中数均为 0（对照：同页 `MC_TE_FILTERS` 在 Mooncake 文档、`kv_load_failure_policy` 在 vLLM `vllm/config/kv_transfer.py` 均可定位且语义一致，说明其余代码级论断有真实来源，仅此条归因不可达）。｜修复要求：在第 368 行写明该实现所属的 fork/仓库与 commit（如「内部 vLLM 分支 `<repo>@<commit>`，文件 `flexible_connector.py`」），或把该行及 summary/description 中的 `FlexibleConnector` 统一降级标注为「内部实现，公开仓库不可核对」；不得保留「vLLM `<文件名>`」这种无版本、无仓库的归因。｜修复：｜复验：
- **[轻微·图示]** ｜来源：不适用（无头浏览器渲染实测）｜位置：第 265–267 行（图 3 内 `foreignObject x="72" y="87" width="286" height="16"`）｜问题：该 foreignObject 高 16 px、字号 11 px/1.45（行高约 16 px），只容一行；文字「K scale · fp32 · 2 行（64 个，每 token 一个，动态）」一行放不下，第二行被裁掉。Chrome headless（`--force-device-scale-factor=3`）实测渲染结果为「K scale · fp32 · 2 行（64 个，每 token 一个，」，句末「动态）」不可见；同一图 V 侧标签「V scale 槽位 · 空置（静态值在 checkpoint）」完整显示，可作对照。｜引文依据：高分辨率截图（3×）中 K 侧虚线框内文字止于「一个，」，框内无第二行；`<div>` 文本原文含「动态）」。｜修复要求：缩短该行文字（如「K scale · fp32 · 2 行（每 token 一个）」）或把该 foreignObject 的 `height` 增至 30 并同步下移/收紧相邻元素，改后重跑无头渲染确认 K、V 两侧标签均无裁切、不与虚线框重叠。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 1
- 处置：修复（阻断 1 项与重要 1 项须关闭后方可发布；轻微 1 项随本轮一并处理）
