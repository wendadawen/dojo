# MegaMoE 初稿检查

## 输入版本

- scope.md / evidence.md / outline.md / glossary.md：2026-09-03 完成，规划完成条件逐项满足（歧义已裁定、5 个学习目标各有完成答案、C1–C30/F1–F4/N1–N12 全部定位到存档源码、章节单一职责、贯穿示例与材料职责齐备）。
- 外壳：.dojo/templates/concept/index.html 与 overview.html（拷贝于 2026-09-03）。
- 证据源：DeepGEMM main @ 559d79f（本地克隆 /tmp/megamoe-research/repo），关键文件拷贝至 research/sources/（11 个源码/文档文件：9 个源码 + bf16 变体 + README，另 3 个 PR/论文摘录）。

## 大纲落实

- 章节结构：第 1–5 章 + 来源与范围说明，与 outline.md 一致，无增删重排。
- 学习目标：Q1–Q5 与 scope.md 措辞一致，页首「核心问题」5 题每题配解答折叠块，答案末尾指明论证章节。
- 前置知识：deepseek-moe、moe-serving、deepep、swiglu、gpu-execution-model、gpu-communication、fp8-block-quant、mxfp4-qat 八个链接在首次依赖处给出；deepep 页面由并行会话生成中，本页正文自写了 dispatch/combine 的最小含义并给链接。
- 贯穿示例：EP2 mini 配置（4 专家、top-k 2、每 rank 2 token）在第 1 章建立，第 3 章（地址翻译）、第 4 章（拉取/写回/归约）逐层复用，无参数漂移。
- 误解与边界：误解 1、2 在页首 misconception 块；误解 4 在 4.1 的 yellow callout；误解 3 在 5.3 的 yellow callout。
- 过渡：每章末尾有到下一章的逻辑缺口句。

## 目标覆盖检查

- Q1（五段与空转）：第 1 章正文（1.1 五段定义+F1+示例走查；1.2 双空闲+图 1+通信量手算）完整回答 ✓
- Q2（融合与重叠）：第 2 章正文（持久 kernel 形态、三组线程、六角色、寄存器再分配）完整回答 ✓
- Q3（对称内存）：第 3 章正文（布局约定、map 公式、图 3、为什么必须要、API 三步）完整回答 ✓
- Q4（数据流）：第 4 章正文（dispatch 三步、调度与环形缓冲、F2 手算、L1/L2 epilogue、归约、图 4、伪代码）完整回答 ✓
- Q5（收益与边界）：第 5 章正文（基准表、正确性、六条边界）完整回答 ✓
- 两级问题块：核心问题 5 题、各章 2–3 题，全部配「解答：」折叠块，答案独立成段。

## 代码运行

- 本页无可运行代码：Mega MoE 需要 sm100 GPU + 8 进程对称内存环境，无法在本机执行。页面中的官方 API 代码块标注"来自 DeepGEMM README、本页不运行"；第 4 章伪代码标注"依据源码整理、非可运行代码"。符合 write.md"代码只有在能帮助理解且实际跑通时才加入"的约定（故未加入可运行代码）。

## 机械检查

- `python3 .dojo/scripts/validate.py wiki/megamoe/index.html` → validation ok（首次报 1 处未渲染数学字符 ∈，已修复为 LaTeX 写法后通过）
- `python3 .dojo/scripts/validate.py wiki/megamoe/overview.html` → validation ok（首次报模板标记残留，删除后通过）
- 残留占位符/组件标记：无（脚本复查）
- 站内概念页链接：8 个全部存在
- 双向引用：正文 sup 引用与来源章节条目双向核对一致（C1–C30、F1–F3、N1–N11；N12 为来源章节备考条目，正文未引用且已标注）

## 公式渲染与交互

- headless Chrome 实测（探针脚本注入页面副本，结果经 document.title 读回）：
  - index.html：KaTeX 渲染 215 处，其中 SVG foreignObject 内 34 处；3 个 SVG 图共 74 个标签两两重叠检测为零重叠（首次检测出 2 处重叠：面板标题压 dispatch/combine 小标签、combine 标签压"下一专家"文字——已通过移位与块内短标签修复，复测为零）；窄屏 375px 下复测重叠为零。
  - overview.html：KaTeX 渲染 2 处（top-$k$；首次为纯文本写法，已统一为 LaTeX）。
  - 折叠块 19 个、目录锚点、明暗主题切换由模板脚本提供，渲染实测正常。
- 探针临时文件已删除。

## 写作偏差

- 无返回规划级的偏差。写作中的局部决定：① overview 的公式写法统一为 top-$k$；② 图 1 的"下一专家的计算接续"长标签因重叠改为块内 L1/L2 短标签并把说明并入图注——均为局部修正，不改变大纲。


## 三轮独立审查结果（补记）

- 第 1 轮：0 阻断 / 2 重要 / 8 轻微（贯穿示例计数错误、来源定位指向未存档文件等），全部修复并复验。
- 第 2 轮：0 阻断 / 0 重要 / 5 轻微（单位口径、SVG ASCII 标签、PDL 释义等），全部修复并复验。
- 第 3 轮：0 阻断 / 3 重要 / 6 轻微（对称内存示例方向颠倒、调度 warp 归属粒度、符号不一致等），全部修复并复验。
- 最终状态：可发布。详见 review-1.md、review-2.md、review-3.md（每轮记录含核对引文、修复与复验）。
- 过程教训：同一文件的多个编辑并行发送会被后写覆盖（两处修复因此未生效、被后续审查轮重新抓出）；修复必须单脚本顺序执行并逐条 grep 复验后再更新审查记录。
