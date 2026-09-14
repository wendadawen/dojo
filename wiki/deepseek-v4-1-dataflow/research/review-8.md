<!-- review-meta
round: 8
page: wiki/deepseek-v4-1-dataflow/index.html
reviewed_content_sha256: 9080c7946b9e76ae
-->
# deepseek-v4-1-dataflow 审查记录（第 8 轮）

- 页面版本：wiki/deepseek-v4-1-dataflow/index.html，工作树干净，git blob 90996de82939222d1a921ecb4a0eda8ba9bace2e（sha256 前缀 10abdccc219b37be）
- 审查时间：2026-09-14 18:21
- 审查者：独立子代理（未参与写作，未读本页 research/）
- 适用规范：`dojo:type=dataflow` → guides/model-dataflow.md；表述项按 guides/concept/check.md §2.2 与 style-guide.md
- 已完整阅读：全文 759 行（head 元数据、第 1–7 章正文、noscript 六视图静态表、VIEWS 交互数据与脚本）
- 来源核对版本：官方模型卡 README（wiki/deepseek-v4-1/research/official/README.md，197 行，2026 版）；「报告行 N」指向 DeepSeek_V41_Tech_Report.pdf 的抽取文本，仓库未镜像，未能逐行打开（其支持的数值已逐条改用模型卡核对）

## 已核对通过（关键项）

- 890 B/token：模型卡 CSA2 段原文「reduce the global KV cache footprint to 890 bytes per token」，与页面一致；页内分解逐项复算成立：288 B/条目 → r=2 三层各 288/2=144、合计 432，r=1 层 288，主 KV 720，加索引器 K 170 得 890。
- 288 B/条目 = 512 维 FP4（256 B）+ 每 16 通道 1 个 E4M3 scale（32 B）；索引器 K 68 B = 128 维 FP4（64 B）+ 每 32 通道 1 个 E8M0（4 B）。与模型卡「E2M1 format, one E4M3 scale per 16 channels」口径一致，且只有 68 才能凑出官方 890。
- 552B 主干 + 196B Engram、40 层＝20 编码器＋20 解码器、8B prefill / 16B decode、384 路由专家取 top-6 + 1 共享专家、1M 上下文、FP4 主 KV、SWA 有界重放：均能在模型卡找到对应句。
- 分项之和：7.8929+7.5758=15.4687；432+288=720；720+170=890；2048×8=16384；24×256×25600+2×4×5120=157,327,360≈157.33M；3×2304×5120=35,389,440≈35.39M；Engram 两表 196.61B＋投影 0.31B＝196.93B；65536×16=1,048,576=1M；128×512×1 B=64 KiB。全部自洽。
- 同页一致性：正文、spec 表、summary/meta description、总览与六个交互视图数据之间，同一数字（288/890/512/128/16384/890/24 等）无冲突；VIEWS 与 noscript 静态表的数字多重集机器比对一致（差异仅来自 JS 颜色/布局常量）。
- 公式：KaTeX 以 throwOnError:true 全部解析成功（117 条正文公式 + dojo:summary 内 $H_{L/2}$、$2048\times8=16384$）。
- 交互视图：无脚本时由 noscript 提供六视图静态表，页面仍可读；视图切换不改变正文结论；drill 节点为黄色双边框，与正文明示一致。
- validate.py 返回 validation ok。

## 问题

- [轻微·可读性] §4 CED：公式 $C_l=H_{L/2}W_l^{KV}$、$Z_l=H_{L/2}W_l^{Z}$ 中的 $Z_l$ 首次出现且全页仅此一处，页面未说明它指什么（随后的解释句只覆盖「键值条目」，未区分 $C$ 与 $Z$），也未在别处使用｜引文依据：不适用（仅涉及符号定义完整性）｜修复要求：一句话说明 $Z_l$ 的含义，或删去不使用的符号｜修复：｜复验：
- [轻微·来源] §7 核对方式末句「实测清单（含形状明细）见 wiki/deepseek-v4-1/research/measured.md」：被指向的文件实际内容为已移除实测产物的登记表（「本页的实测产物……现已从仓库移除……下表是它们被移除时的登记」），不含任何形状明细，故此括号描述与该文件实际内容不符｜引文依据：measured.md 首段原文「本页的实测产物原先存放在本目录下，现已从仓库移除……下表是它们被移除时的登记」；表中条目的体积列均为 0.0 KB｜修复要求：删去「（含形状明细）」或改写为与该文件实际内容相符的措辞｜修复：｜复验：

## 结论

- 未发现事实性错误、来源不支持或推断被包装成来源结论的条目；未发现分项之和≠合计；未发现正文/summary/overview/图注/交互视图数据之间的数字冲突。
- 未发现元话语、会话指代、调试叙事、AI 拼接腔等表述问题（关于「本页」的两处用法属 style-guide §12 允许的自称）。
- 处置：可发布，上述 2 条轻微问题可接受或后续合并修复。

统计：阻断 0 / 重要 0 / 轻微 2