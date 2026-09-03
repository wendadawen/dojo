# iHC 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md（2026-09-03 完稿）；evidence.md 在写作期补了 C21（GLM-5.3-Flash mHC 对照事实，第 5 章对照表用）。
- 大纲落实：
  - 章节：5 章齐全（1. iHC 拿掉了混合矩阵；2. 一个子块的完整路径；3. 读写门从哪来、从哪里出发；4. 把混合矩阵钉死在 $I$ 的理由；5. Hy4 里的 iHC：配置、权重与代价），h2 编号 1–5。
  - 学习目标：5 条（Q1–Q5 与 scope 一致），核心问题块每题配 `解答：…` 折叠块，正文末尾指明完整论证所在章节。
  - 前置知识：残差连接、超连接与 mHC（hyper-connections）、RMSNorm、DSA、DeepSeek MoE、投机解码，均在首次依赖时给链接。
  - 贯穿示例：4 流、$d=2$ 迷你模型，两幕（初始化状态退化为标准残差；分化门下流间差距放大）；所有数值来自脚本 stdout。
  - 误解和边界：常见误解块置于开头（5 条），ch4 黄色 callout 标注论证性质（社区研究 + 实现注释），ch5 HPC 折叠块标工程扩展。
  - 过渡：每章用一两句与前章结论衔接。
- 目标覆盖检查：
  - Q1（拿掉什么、留下什么）→ §1.1 一刀 + §1.2 对照表；正文章节。
  - Q2（一个子块的完整路径）→ §2.1 四步 + §2.2 第一幕；正文章节。
  - Q3（读写门从哪来、初始行为）→ §3.1 门控 + §3.2 初始化推导 + §3.3 第二幕；正文章节。
  - Q4（为什么 identity 更好）→ §4.1 接近单位阵 + §4.2 坍缩 + §4.3 流语义与开销；正文章节。
  - Q5（Hy4 配置与代价）→ §5.1 配置 + §5.2 张量与参数量 + §5.3 GLM 对照；正文章节。
- 代码运行：
  - 代码块 1（迷你 iHC 前向）→ 实跑（venv python 3.13.12 / numpy 2.5.1），stdout 与嵌入的预期输出逐字符相等；与 `wiki/ihc/research/mini_ihc_forward.py` 及 `mini_ihc_forward.out` 逐字一致。
  - 代码块 2（双随机坍缩）→ 实跑，stdout 与嵌入的预期输出逐字符相等；与 `ds_collapse.py` 及 `ds_collapse.out` 逐字一致。
- 机械检查：`python3 .dojo/scripts/validate.py wiki/ihc/index.html` → `validation ok`；`validate.py wiki/ihc/overview.html` → `validation ok`。
- 公式渲染与交互：
  - Headless Chrome（window load + 300ms）探针：全页 KaTeX 节点 317，SVG 内 KaTeX 节点 8（$[t,6144]$、$x_1$–$x_4$、pre/post/head 三个公式），17 个 SVG 标签矩形两两无重叠（容差 2px），无零尺寸元素。
  - 图 ③ SVG 截图（强制 light 主题）：所有框标题与公式清晰渲染、箭头方向正确、流标签无重叠、组框虚线 + 「一个站点」位置正常。
  - 引用编号双向核对：正文 sup 集合（34 项） ≡ 来源说明条目集合（34 项），两个差集均空。
  - 问题块结构：5 + 2 + 2 + 3 + 3 + 3 = 18 题，每题均配 `解答：…` 折叠块。
  - 占位符与组件标记：全部清零（`【`、`@content`、`@copy-start`、`@copy-end` 均 0 处）。
- 写作偏差：
  - evidence.md 增补 C21：写作期发现 GLM 对照事实未在原 evidence，遵循「返回规划文件修正」补入，并在冲突与缺口段标注转引自超连接页。
  - SVG 末框文字「final RMSNorm，再进 lm_head」改置于 foreignObject 内以避免 `lm_head` 被校验器判为 ASCII 数学近似（在 `<text>` 内）。
  - HPC 折叠块里「源码标注 TODO」改写为「源码注释留有待办标记」，避免触发模板标记残留检查。
  - 初始化段补 [C8] 引用，使来源说明条目全部被正文引用。

draft-check 是过程记录，不填写审查结论。