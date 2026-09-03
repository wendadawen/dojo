# HiSparse 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 全部完成；论文 TeX 全文固定为 arXiv:2608.07009v1 预印本（NeurIPS 2026 投稿）

## 大纲落实

- 章节：7 章 + 评价 + 来源说明全部在 `index.html` 第 1–7 章与方法评价/来源与范围说明 h2 下展开
- 核心问题：5 题（Q1–Q5）全部列入页面级「核心问题」块，每题均带 `<` 解答折叠块，与各章节对应（Q1→第 1 章、Q2→第 2 章、Q3→第 3 章、Q4 两半→第 4 章与第 5 章、Q5→第 6 章）
- 前置知识：kv-cache / standard-attention / dsa / strata / vllm-cudagraph / paged-attention / prefix-caching / gpu-communication 全部 wiki 内已有并内联链接；LRU / Bélady / TTFT / TPOT / pinned memory 等以内联最小含义处理，理由见 scope.md
- 贯穿示例：8 个逻辑位置、$k=2$、$B=4$ 的五步选择 {2,5}/{2,6}/{3,6}/{2,5}/{4,5}；含 LRU 演化、Resolve 五阶段手算、第 5 章 4 层 prefetch 扩展；hit 之间按选择集内顺序排位是构造约定（论文未规定同一步内多个 hit 的相对顺序）
- 误解与边界：「论文没做什么」7 项、4 条常见误解、适用边界写在 scope.md；正文中条件与代价随文标注
- 评价章节：第 7 章方法评价覆盖精确性 / indexer 无关性 / 适用边界 / 与邻近工作的关系 / 未来工作
- 过渡：每章开头 1–2 句衔接（why-now）、章末 1 句过渡到下章；与跨章例子的引用已加链接锚点

## 代码运行

- `/tmp/hisparse-research/lru_sim.py`：`python3 lru_sim.py`，退出码 0。真实输出：Swap-vanilla (B=k=8) miss 1048/1600 = 65.5%；LRU (B=2k=16) 546/1600 = 34.1%；LRU (B=4k=32) 63/1600 = 3.9%。页面第 3 章 `<details><summary>可运行代码</summary>` 折叠块中给出原命令与完整输出
- 核心变量含义与正文公式对应：selection 集合、$B$ cache 容量、LRU 排位（hit 排在新 fetch 的 miss 之上）、victim 从未选中条目最旧端逐出
- 与论文数字（30% / 13.4% / 6.7%）作定性对照（trace 模型差异：构造的热点漂移 vs GLM-5.1 LongBenchV2），不当作论文复刻

## 原图

- Figure 1 (motivation.png, 200 DPI) → 第 1 章；显示正常
- Figure 2 (hisparse_overview2.png, 200 DPI) → 第 2 章；显示正常
- Figure 6 (topk_miss_rate_trace.png, 200 DPI) → 第 3 章；显示正常
- Figure 3 (swap_kernel.png, 200 DPI) → 第 4 章；显示正常
- Figure 8 (prefetch_sweep.png, 200 DPI) → 第 5 章；显示正常
- Figure 5 (peak_throughput_comparison_paper.png, 200 DPI) → 第 6 章；显示正常
- Figure 4（DeepSeek-V4-Flash 扫描，关键数字 N3 以表格完整呈现，基线未选择性删减）、Figure 7（kernel 分解，关键数字 N9 以文字呈现）未纳入图片，原因记录在 index.html 来源与范围说明中

## 机械检查

- `python3 .dojo/scripts/validate.py wiki/hisparse/index.html` → `validation ok`
- `python3 .dojo/scripts/validate.py wiki/hisparse/overview.html` → `validation ok`
- 修复路径：初稿 validate 报三处 unrendered math（μ、≈）在 index.html 第 1380、1448 与表格 `<td>≈1×</td>`，已统一改为 KaTeX（$\mu\text{s}$、$\approx$、$\approx 1\times$），校验两次通过

## 公式渲染与交互

- Headless Chrome (`/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --headless --disable-gpu --no-sandbox --virtual-time-budget=8000 --dump-dom`)，探针脚本注入后把 `.katex` 节点数、`.katex-error` 数、`img.naturalWidth>0` 数、SVG `<text>` 矩形两两重叠检测写入 `document.title`，再 grep `<title>` 提取
- index.html：271 个 .katex 节点、0 errors；6 张论文图全部加载；SVG `<text>` 标签两两重叠 0 对；21 个 `<details>` 折叠块、1 个 SVG（预取时间线）；模板灯箱占位 `<img id="lightboxImg" src="">` 计入 FAIL 属预期，非真图
- overview.html：17 个 .katex 节点、0 errors；无图片、无 SVG（overview 不放图）
- 截图 `/tmp/hisparse-render/overview-shot.png` 与 `index-shot-top.png` 视觉目检：标题层级、术语速查表、核心问题块、Figure 1 子图均正常显示，无截断/压线

## 写作偏差

- 无