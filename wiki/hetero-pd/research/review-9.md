<!-- review-meta
round: 9
page: wiki/hetero-pd/index.html
reviewed_content_sha256: 8a2e0a8bd6c410ee
-->
审查对象：/Users/wendadawen/code/dojo/wiki/hetero-pd/index.html（head dojo:type=note，适用 /Users/wendadawen/code/dojo/guides/note.md）。本轮按 note 规范逐项核对。

一、事实与算式复算（均自洽）
- 1351680 = 2×64×1×132×80 = 16896×80（16896 = 2×64×132）；半块 64×132 = 8448 = 66×128；K/V 半块各 8448 B。
- 8110080 = 6×1351680；8 rank 合计 64,880,640 B = 61.875 MiB ≈ 61.9 MiB（图 3 内 «8 rank 并行共 61.9 MiB» 成立）。
- 6 blocks × 64 slot = 384 ≥ 362（362 prompt → ceil(362/64)=6）；与第 3、7 节一致。
- 三处速率换算：6.9 ms → 8110080 B = 1.09 GiB/s ✓；4.4 ms ↔ 1.70（GiB 基数）✓（1.70 GiB/s 反算 4.44 ms，四舍五入即 4.4 ms）；17.8 ms → 0.42 GiB/s ✓。口径说明（基数 2^30）与数值自洽。
- 占比：150 ms/4.4 ms/3.48 s 对端到端 3.63 s；64/3.63 ≈ 17.6 tok/s ✓；D decode 63 token + P 首 token = 64 ✓。
- 公式可复算：unit = head_size/gcd(block_size,head_size)，block_size=64、head_size=128 → gcd=64、unit=2、elems=ceil(4/2)×2=4，与 D 侧 K_SCALE_BYTES=4 一致；反例 head_size=192 → unit=3、elems=6≠4 ✓。
- 数字在 description / dojo:summary / 正文 / 图注 / 图内文本（SVG foreignObject）之间一致：1351680、0.12%、6 blocks、8110080、端口 8000/8021/33981；P 侧 ttft 150 ms（关缓存）与 0.126 s（命中）两处口径区分明确，第 3 节时钟偏移注（ms 级不可跨机比较）成立。
- 引用路径核对：../kv-cache-layout、../increase-kv、../beyond-buzz-disaggregation、../kv-cache、../gpu-communication、../model-parallelism 六处 index.html 均存在。
- .dojo/scripts/validate.py wiki/hetero-pd/index.html → validation ok。
- 无头浏览器（playwright/chromium）本机不可用，未能独立实测渲染；页内公式为标准 KaTeX（含 foreignObject 内 $...$），且 KaTeX 默认 ignoredTags 含 pre/code，第 6 节 grep 块的 $L 不会被误渲染。

二、表述与规范
- 未发现元话语、以「本页」为主语的自我指代、会话指代、调试叙事、临场评价、章节过渡固定句式；aria-label 无 $...$。
- 推断项均已显式标注：agent_metadata 的 GID/QP/rkey（未实证）、hot/cold 传输区分、架构收益（第 7 节「该收益为推断」）；内部实现（flexible_connector.py / hpc_attn.py / hyvl_int8_attn.py）在「来源与范围说明」中标注为公开仓库不可核对，与 note 规范「无法核实的内容标记」一致。
- 本页无图表类交互视图，不涉及「无脚本不可读」。

三、轻微问题
1. 图 2（第 178–212 行）图层级命名与图注/正文不一致：正文标题「2. 建连：三层握手」分 第 1/2/3 层（第 3 层＝RoCE 网卡自动选路），图 aria-label 与图注亦称「三层握手」，但图内只有「建链前置 · RoCE 网卡自动选路」方框与 ①（TCP）、②（NIXL 元数据）两处编号，无 ③，第 3 层被标为「前置」，读者无法按编号把「第 3 层」对应到图内。
2. 第 7 节表占比四舍五入后分项之和为 4% + 0.12% + 96% = 100.12%，略超 100%（无 合计 行；精确值 4.13% + 0.12% + 95.75% = 100%，属舍入残差）。

未报告项说明：head_size=192 反例为公式推导而非来源事实；「恰好/正好」用于描述字节级一致这一事实本身，非临场评价；图内代码标识符用 SVG <text> 承载为全站混用约定，非本页特有。

统计：阻断 0 / 重要 0 / 轻微 2