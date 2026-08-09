# GPU 执行模型与 kernel 调度 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均为 2026-08-07 定稿，规划完成条件逐项满足（3+2 个学习目标、核心论断全部有来源定位、无证据不足项、无无法消歧项）。

## 大纲落实

- 章节：6 章 + 文末「来源与教学说明」，标题与顺序同 outline.md §2 —— 落实
- 学习目标：learning-goals 组件 5 条与 scope.md Q1–Q5 一一对应 —— 落实
- 前置知识：无 wiki 级概念页依赖；moe-serving 为占位提示（无链接）；vllm-cudagraph 为扩展阅读链接（页面存在）；未链接不存在的 expertplex 页面 —— 落实
- 贯穿例子：大活/小活两 GEMM 在第 1–6 章逐章推进（HBM→CTA 铺满→tile 刻度→流水线→启动开销→四种机制）—— 落实
- 误解：开头放 2 条（线程切换、tile 大小），第 5 章处理「persistent kernel 是 API」，第 6 章处理 stream 优先级与 MIG 档位 —— 落实
- 过渡：每章末尾总结已得结论并指出下一步问题 —— 落实
- 材料职责：ASCII 图 ×3（硬件层级/启动流水线/tile 流水）、手算例子 ×2（4×4 读取次数、tile 容量估算）、伪代码折叠块 ×1（persistent kernel）、对照表 ×3（读取次数、MIG 档位、四机制）—— 与 outline §5 一致

## 学习目标闭环

- Q1（启动层级）：第 1 章（硬件）+ 第 2 章（线程层级、启动流程、CTA 跑完才释放）正文完整回答 —— 通过
- Q2（tile 与输入长度）：第 3 章手算 + 容量估算 + 「数量变、单 tile 时间不变」结论，正文完整 —— 通过
- Q3（启动开销与两种消除手段）：第 5 章正文完整，伪代码在折叠块但机制与代价在正文 —— 通过
- Q4（四件协作工具）：第 4 章正文完整 —— 通过
- Q5（四种共享机制边界）：第 6 章正文 + 对照表完整 —— 通过
- 折叠块（4×4 清单、persistent 伪代码）均不独占任何学习目标答案 —— 通过

## 代码运行

- 无可运行代码（机制依赖真实 GPU，按 outline 全部用手算与伪代码代替；persistent kernel 伪代码块标记为 language-text 且注明不可运行）

## 机械检查

- `python3 .dojo/scripts/validate.py wiki/gpu-execution-model/index.html` → validation ok（退出码 0）
- `python3 .dojo/scripts/validate.py wiki/gpu-execution-model/overview.html` → validation ok（退出码 0）

## 公式渲染与交互

- 用仓库本地 `libs/katex.min.js` 在 node 22 中对 index.html 全部公式（1 个 display + 88 个 inline 匹配项）逐一 `renderToString`：失败 0；唯一警告来自模板自带阅读时间脚本中的模板字符串（位于 `<script>` 内，auto-render 不处理 script 标签），非页面公式
- 交互（目录、折叠、主题切换、复制按钮）由模板脚本统一提供，与既有已发布页面共用同一实现

## 写作偏差

- 无。outline §6 中「MIG 表格若过长则留正文」经评估表短，放正文，与大纲一致。
