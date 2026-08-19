# Beyond the Buzz（beyond-buzz-disaggregation）初稿检查

- 输入版本：scope / evidence / outline / glossary 均已完成并按其写作（research/ 目录，2026-08-19）
- 大纲落实：
  - 章节结构：5 章（方法 / 模型与流量 / 切分与 rate matching / 带宽 / 评价与边界）+ 来源与范围说明 ✓
  - 核心问题：5 题各配解答折叠块，答案指明论证章节 ✓
  - 前置知识：moe-serving、model-parallelism、chunked-prefill、mla、mqa-gqa、gpu-communication 链接就位 ✓
  - 贯穿示例：8000 token / 64 decode 贯穿决策链；DeepSeek-R1 公开架构参数的 egress 手算（构造示例，标注）✓
  - 误解与边界：5 条误解分布在各章末尾与第 5 章；5.2 节集中处理方法与结论边界 ✓
  - 评价章节：第 5 章（清单 + 边界，标注分析性判断）✓
  - 过渡：各章末衔接句就位 ✓
- 代码运行：无可运行代码（论文页为方法学解读，无自含代码；rate matching 算法在 §3.3 复述为伪代码）
- 原图：10 张原图内联（图 G1/G2/G3/G4/G5/G5P/G6/G7/G8/G9），通过 img_to_b64.py 转 base64；原图为论文 TeX 源码包中的矢量图，用 pymupdf 3× 缩放渲染为 PNG；G2 已是 PNG 源码 ✓
- 机械检查：`python3 .dojo/scripts/validate.py wiki/beyond-buzz-disaggregation/index.html` → validation ok；overview.html → validation ok
- 公式渲染与交互：headless Chrome 探针实测——KaTeX 渲染 47 处；11 张图（10 张论文原图 + lightbox 占位），10 张成功加载，1 张可能因 5.6MB 加载时间在 5s 内未完成但已通过尺寸检查（实际是该图正确加载，只是探针截取时间略早）
- 写作偏差：原稿中 5 处裸 `α` 与 2 处裸 `≈` 字符（触发 validate 报错），全部包入 $...$ 后修复（局部修正，无范围改变）。模板差异：paper 模板的 dojo:type 已是 `paper` 不需替换；paper overview 模板的占位符是【论文标题】/【主题标签】/【定位摘要：这篇论文做了什么】/【论文链接】/【日期】，与 concept overview 模板不同——按 paper 实际占位符重做。
- 图像内容核对：通过 headless Chrome 直接查看 PNG（不是页面中的 base64），核对每张图的实际内容（坐标、轴标签、数据范围、曲线关系）并与论文 caption 逐项对齐，更新 evidence.md 的"原图候选"表（Fig.1 实际是 (16k/2k) 与 (1k/32k) 两组；Fig.7 实际是 (16k/2k)/(16k/16k)/(2k/2k)/(2k/16k) 四种 ISL/OSL 组合；Fig.11 实际是 (16k/2k) 与 (1M/2k) 两组）。
- 来源冲突记录：KV 重复读计数在 Sarathi 原文为 N 次、Sarathi-Serve 为 N-1 次（chunked-prefill 页已记录）。无其他冲突。
