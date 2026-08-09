# ExpertPlex 初稿检查

- 输入版本：scope.md（完成，5 核心问题 Q1–Q5）、evidence.md（C1–C20/F1–F4/N1–N4，全部已确认）、outline.md（7 章 + 贯穿例子 + 评价章）、glossary.md（38 术语）均就绪
- 大纲落实：
  - 页面开头（一句话/问题/承诺/元信息/贯穿场景/过渡）✓
  - 第 1 章 Q1（PDD 粗粒度 + colocation 固定切分 + Figure 2 + C2/C3/C4/C5）✓
  - 第 2 章 Q2（权重占比 C1 + 架构 C6 + Figure 3 + 误解 2）✓
  - 第 3 章 Q3（五性质表 + tile 调度 C7 + Figure 4 + F4 + 误解 3）✓
  - 第 4 章 Q4（死锁 + 一侧通信 C8/C11 + Figure 5 + 分层路径 C9 + C20 + 误解 4）✓
  - 第 5 章（F1 goodput + F3 tile 足迹 + Figure 6 + 搜索主干 + 在线 q'）✓
  - 第 6 章 Q5（设置表 N1–N4 + 四组端到端 C12–C14 + 三个微基准 C15–C18 + Figure 11 + 误解 1）✓
  - 第 7 章独立评价（全章推断，优点/局限/适用）✓
  - 来源与教学说明（C/F/N 定位 + 原图对应 + 事实/推断区分 + 教学示例）✓
  - 前置知识引用：moe-serving、gpu-execution-model 概念页链接 ✓
  - 每章末完成检查 ✓
- 代码运行：无可运行代码（系统论文，机制无法本地复现；APK 调度循环未放伪代码，按 outline 用五性质表与机制叙述代替）
- 原图：Figure 2/3/4/5/6/11，获取途径 TeX 源码 figures/*.pdf → pdftoppm 220dpi PNG → base64 内联；均已在页面中标注 Figure 编号
- 机械检查：python3 .dojo/scripts/validate.py wiki/expertplex/index.html → validation ok（退出码 0）；overview.html → validation ok
- 公式渲染与交互：F1/F3/F4 用 $$ display KaTeX；待浏览器实际检查（交独立审查阶段在 node + 本地 katex 验证）
- 写作偏差：无。outline 选 6 张原图全部落实；goodput 手算为教学示例已标注
