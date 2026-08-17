# 因果掩码（Causal Mask）初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已写齐于 `wiki/causal-mask/research/`，规划完成条件已满足（概念歧义已裁定——因果掩码 vs 填充掩码；5 个学习目标 Q1–Q5；核心论断 C1–C7、公式 F1–F2 完成来源定位且置信状态为已确认；无外部实测数字，N 类为空；教学大纲齐备）。

## 大纲落实

- 章节结构：S1 rule-and-motivation / S2 mechanism / S3 hand-computed-example / S4 training-vs-inference / S5 boundaries-and-nope + 文末 sources-and-scope-notes，共 6 个 h2，id 全页唯一。
- 学习目标：Q1–Q5 在页面开头用 learning-goals 组件列出，与 scope.md 一致。
- 前置知识：标准注意力（`../../wiki/standard-attention/index.html`，已有真实链接）、NoPE（`../../wiki/nope/index.html`，已有真实链接）；自回归生成 / KV-cache / 排列等变三个未生成概念按任务约定登记不生成，正文在首次依赖处给出最小定义（自回归生成="逐 token 预测下一个 token、每步只依赖已生成前缀"；KV-cache="缓存过去 token 的 key/value，新 query 直接复用"；排列等变="交换输入顺序时每个位置的输出不变"）以支撑当前机制讲解，不内联大段背景。
- 贯穿例子：3 个 token、注意力分数全为 0、值 v1=1,v2=2,v3=3，在 S3 手算（输出 1,1.5,2）与双向对照（输出全 2）、S5 复用双向对照给出 NoPE 结构前提。
- 误解和边界：页面开头 misconceptions 给三条（因果掩码≠填充掩码、≠NoPE、encoder 不加）；S5 给三条边界（encoder vs decoder、填充掩码正交、NoPE 关系）与排列对称性打破的机制。
- 过渡：每章末指出下一章要解决的问题（S1→机制；S2→手算；S3→训练推理形态；S4→边界与深层作用）。

## 学习目标闭环

- Q1（规则与动机）：S1 给规则"位置 t 只 attend 到 1..t"、说明自回归生成为何不能看未来、信息泄漏后果、encoder/decoder 边界对照表。正文完整回答。
- Q2（机制）：S2 给掩码矩阵 M 的形式（上三角 −∞）、公式 F1 softmax(QKᵀ/√d_k+M)V、逐项符号定义、−∞ 使未来权重为 0 与可见重归一化的机制、掩码施加在 softmax 输入而非输出的澄清。正文完整回答。
- Q3（手算例子）：S3 给构造示例设定、不掩时双向对照（输出全 2）、施加掩码后位置 1 正文手算（输出 1）、位置 2/3 折叠块完整计算（输出 1.5、2）、可运行代码验证。正文 + 折叠块共同回答，正文保留位置 1 与三位置结论。
- Q4（训练 vs 推理）：S4 给训练并行形态（整条序列 + 显式 M、顺序操作 O(1)）、推理 KV-cache 形态（逐 token 增长、cache 内只有过去、掩码隐含）、澄清"推理不必每步重建掩码矩阵"。正文完整回答。
- Q5（边界 + NoPE 前提）：S5 给 encoder vs decoder、填充掩码正交（对照表）、因果掩码打破排列对称性的机制（可见集合 {1..t} 不同）、复用 S3 例子验证（双向输出全 2 vs 因果输出 1,1.5,2）、NoPE 关系澄清。正文完整回答。
- 全部目标由正文章节完整回答，无目标被折叠块独占。折叠块（S3 位置 2/3 完整手算 + S3 可运行代码）收起时正文仍回答 Q1–Q5。

## 代码运行

- 代码块：S3 折叠块中的 3-token 因果掩码 softmax Python 代码。
- 运行命令：`python3 -c "..."`（代码内联在命令中，仅依赖标准库 math）。
- 退出码：0。
- 实际输出与页面描述一致：
  - 施加掩码后分数：pos 1 [0,-inf,-inf]、pos 2 [0,0,-inf]、pos 3 [0,0,0]。
  - softmax 权重：pos 1 [1.0000,0.0000,0.0000] sum=1.0000 output=1.0000；pos 2 [0.5000,0.5000,0.0000] sum=1.0000 output=1.5000；pos 3 [0.3333,0.3333,0.3333] sum=1.0000 output=2.0000。
  - 双向对照：三位置 weights 均为 [0.3333,0.3333,0.3333]、output 均为 2.0000。
- 页面"预期输出"块与真实运行结果逐行一致；正文结论（未来权重为 0、可见重归一化、因果输出 1/1.5/2 不同、双向输出全 2 相同）与代码输出一致。

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/causal-mask/index.html` → 结果：`validation ok: wiki/causal-mask/index.html`（退出码 0）。
- 命令：`python3 .dojo/scripts/validate.py wiki/causal-mask/overview.html` → 结果：`validation ok: wiki/causal-mask/overview.html`（退出码 0）。
- 两页均通过：以 `<!DOCTYPE html>` 开头、`</html>` 结尾；无模板占位符【…】；无 @content/@component/TODO/TBD 标记；无重复 id；无指向缺失 id 的同页锚点；无断链本地引用（../../libs/katex.min.css 等资源、../../wiki/standard-attention/index.html、../../wiki/nope/index.html、overview.html、index.html 均存在）。
- 首页元数据：index.html 的 description / dojo:summary / dojo:type / dojo:topics / dojo:tag 五项齐全；description 为纯文本无 $；dojo:summary 仅含行内 $...$ 无 $$；$ 定界符配对正常。
- 数学渲染：正文含 2 个显示公式（M_{ij} cases、F1 Attention 公式）与多处行内公式，KaTeX JS/CSS/auto-render 本地资源齐全且 renderMathInElement 初始化在模板中。

## 公式渲染与交互

- KaTeX 语法检查：2 个显示公式使用 `$$...$$`（M_{ij} 的 \begin{cases}、F1 的 \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V）；行内公式使用 `$...$`（$-\infty$、$e^{-\infty}=0$、$o_t=\sum_{j=1}^{t}\alpha_{t,j}v_j$ 等）。语法均为 KaTeX 支持（\operatorname、\!、\left/\right、\frac、\sqrt、\begin{cases}、\mathbb{R}、\top）。`$$`/`$` 定界符与外壳 auto-render 配置一致。
- 渲染机制：依赖外壳 `auto-render.min.js` 自动渲染 `$...$` 与 `$$...$$`，与其它概念页同一机制。
- 交互：外壳脚本提供目录、章节折叠、j/k 跳转、主题切换、返回顶部、代码复制按钮，均由外壳统一处理。
- 限制：未开启图形浏览器逐式截图核对；公式语法经文本检查 well-formed，渲染依赖外壳脚本。

## 写作偏差

无写作偏差。未自行增删核心章节、未新增学习目标、未更换贯穿例子、未把正文必要内容移入折叠块、未使用证据不足论断。写作中按 outline.md 落实页面开头（blockquote.meta → callout 引言 → learning-goals → misconceptions → 前置知识段落 → 贯穿问题段落 → 正文）与五章结构；misconceptions 组件按 outline 安排在页面开头（与 S5 边界机制形成"开篇预警 + 章末机制"呼应，非重复）。来源事实均附 §3.1/§3.2/§3.2.1/§3.2.3 定位与 C1–C7/F1–F2 编号，与 evidence.md 一致。
