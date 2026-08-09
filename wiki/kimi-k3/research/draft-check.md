# Kimi K3 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成，状态为已确认（核心论断 C1-C23，公式 F1-F10，数字 N1-N13，冲突已标注）
- 大纲落实：
  - 章节：S1 三维度总览 / S2 序列维度 / S3 深度维度 / S4 宽度维度 / S5 原生视觉 / S6 训练 / S7 后训练 / S8 基础设施 / S9 性能与评价 / S10 独立评价 / 来源与教学说明——全部落实
  - 核心问题：Q1（S2 完整回答）、Q2（S3 完整回答）、Q3（S4 完整回答）、Q4（S6 完整回答）、Q5（S9 完整回答）——全部落实
  - 前置知识：kda/mla/nope/block-attnres/stable-latent-moe/situ-glu/quantile-balancing/moonvit-v2/per-head-muon/mxfp4-qat/moe-serving/gpu-execution-model 共 12 个子页面链接——全部落实，路径 ../../wiki/<name>/index.html
  - 贯穿例子："一个 token 流经 K3 的 93 层"在 S1 引入，S2/S3/S4 各推进一次——落实
  - 误解和边界：scope.md 记录 5 个误解 + 适用边界，正文在相关章节标注——落实
  - 评价章节：S10 独立评价，全部解读者推断，开头有 callout 声明——落实
  - 过渡：每章末有完成检查 + 下一章衔接——落实
- 代码运行：无可运行代码
- 原图：无原图内联（用自绘 HTML 表格替代 Table 1/Table 2，用 ASCII 图示替代 Fig.2，用文字描述替代 Fig.6/Fig.7）——读者应查阅原文图获取完整信息
- 机械检查：`python3 .dojo/scripts/validate.py wiki/kimi-k3/index.html` → `validation ok`，退出码 0
- 公式渲染与交互：页面无 KaTeX 公式（机制细节由子页面承载，正文只给一句话衔接）；折叠块（details）正常；目录自动生成；章节折叠按钮正常
- 写作偏差：无偏差。按 outline.md 落实，未增删核心章节、未增加核心问题、未更换贯穿例子。MOPD/EAGLE/FlashKDA/MoonEP 按大纲简要内联并标注"子页面未生成"。
- 子页面未生成的占位：MOPD、EAGLE、FlashKDA、MoonEP 在正文简要内联，标注"子页面未生成，此处简要内联"。这些是独立工作，本页只给角色级描述，不影响核心问题回答。
- 折叠块全部收起时正文仍能回答全部核心问题：是——S2-S9 每章正文已完整回答对应核心问题，折叠块只补充细节（层配置列表、block 划分计算、benchmark 条件）
