# Stable LatentMoE 初稿检查

## 输入版本

- scope.md：已写，含概念歧义处理（已裁定）、5 个学习目标 Q1-Q5、内容分级、前置知识映射、6 项不展开内容、6 条常见误解与适用边界。
- evidence.md：已写，含 C1-C13 论断、F1-F4 公式、N1-N8 数字，全部来自 K3 报告 §2.3-2.3.3、官方 config.json、DeepSeekMoE 论文，置信状态均为已确认。
- outline.md：已写，含页面开头、S1-S6 章节、文末来源说明、讲解顺序、贯穿例子、讲解材料职责、正文与折叠块分工。
- glossary.md：已写，含 33 个术语/符号的首次出现位置与定义。

## 大纲落实

- 页面开头：blockquote.meta 来源摘要 ✓；learning-goals Q1-Q5 简化版 ✓；callout-blue 前置概念提示 ✓；钩子问题 ✓。
- S1 扩大专家池的代价：对照表（普通 MoE vs LatentMoE）✓；公式 $z = W_\downarrow x$ ✓；完成检查 ✓；过渡到 S2 ✓。
- S2 层结构：DeepSeekMoE 组织引用 ✓；Eq. 11 完整公式 + 全符号定义 ✓；K3 配置表（$d, \ell, N, k, N_s$ 等）✓；ASCII 数据流图 ✓；共享 vs 路由对照表 ✓；完成检查 ✓；过渡到 S3 ✓。
- S3 两个失败模式：激活爆炸（矩阵乘法链 ASCII 图）✓；负载失衡（无辅助损失路由背景）✓；两个失败对照表 ✓；完成检查 ✓；过渡到 S4 ✓。
- S4 三件稳定化：Normalized LatentMoE（RMSNorm 位置）✓；SiTU-GLU（公式 F3 + 上界 100，引用 wiki/situ-glu/）✓；QB（公式 F4 + 分位数结论，占位）✓；三件稳定化对照表 ✓；"各管一段不可互替"说明 ✓；完成检查 ✓；过渡到 S5 ✓。
- S5 RMSNorm 位置必要性：$u$ 尺度来源 ✓；不插 RMSNorm 的后果 ✓；插 RMSNorm 的效果 ✓；三个错误位置分析 ✓；原 vs Normalized 对照表 ✓；教学示例折叠块（$\ell=2$，手算）✓；完成检查 ✓；过渡到 S6 ✓。
- S6 适用边界：成立条件 ✓；依赖概念列表（含 wiki/moe-serving/、wiki/situ-glu/ 链接，QB/DeepSeekMoE/RMSNorm/LatentMoE 占位）✓；不能推出的结论（5 条）✓；完成检查 ✓。
- 文末来源与教学说明：核心论断与来源 ✓；核心公式与来源 ✓；外部数字与实验条件 ✓；教学示例 ✓；教学解释与类比边界 ✓；教学简化及其限制 ✓。

## 学习目标闭环

- Q1（为什么 LatentMoE 要把路由专家放进隐空间）：S1 完整回答——扩大专家池的代价 + LatentMoE 的对策 + 代价（多两次投影）。✓
- Q2（层结构与各组件位置）：S2 完整回答——Eq. 11 公式 + 全符号 + ASCII 数据流 + K3 配置数字 + 共享 vs 路由对照。✓
- Q3（两个失败模式与三件稳定化的对应）：S3 + S4 完整回答——两个失败定位 + 三件稳定化各自修复哪个 + 协同关系。✓
- Q4（RMSNorm 位置的必要性）：S5 完整回答——$u$ 尺度来源 + 不插的后果 + 插的效果 + 三个错误位置 + 教学示例。✓
- Q5（适用边界与不能推出的结论）：S6 完整回答——成立条件 + 依赖概念 + 5 条不能推出的结论。✓

所有目标由正文章节完整回答，无目标被折叠块独占（S5 教学示例折叠块只是辅助，正文已完整论证）。

## 代码运行

无可运行代码。页面只含 ASCII 图示（`<pre class="diagram">`）和 HTML 表格，无 Python/JS 代码块。教学示例的手算已用 Python 独立验证（见下"机械检查"后的验证记录）。

## 机械检查

命令与结果：

```
$ python3 .dojo/scripts/validate.py wiki/stable-latent-moe/index.html
validation ok: wiki/stable-latent-moe/index.html

$ python3 .dojo/scripts/validate.py wiki/stable-latent-moe/overview.html
validation ok: wiki/stable-latent-moe/overview.html
```

两页均通过：无 `<!DOCTYPE html>` 缺失、无 `</html>` 缺失、无占位符 `【…】` 残留、无 `@content`/`@component`/`TODO`/`TBD` 标记残留、无重复 id、无指向缺失 id 的锚点、无指向不存在文件的本地引用。

公式标记配对检查（去除 `<script>`/`<pre>`/`<code>` 后）：
- index.html：`$$` 12 个（6 对，偶数），`$` 428 个（214 对，偶数）✓
- overview.html：`$$` 0 个，`$` 40 个（20 对，偶数）✓

教学示例手算验证（Python 独立计算）：
- $u_A = (1,1)$，RMS $= 1$，RMSNorm$(u_A) = (1,1)$ ✓
- $u_B = (10,10)$，RMS $= 10$，RMSNorm$(u_B) = (1,1)$ ✓
- 尺度比 $10$ 倍 ✓
- SiTU-GLU 上界 $4 \times 25 = 100$ ✓
- 稀疏度 $896/16 = 56$ ✓

## 公式渲染与交互

已用 `open wiki/stable-latent-moe/index.html wiki/stable-latent-moe/overview.html` 在默认浏览器打开两页（退出码 0）。CLI 环境无法直接确认渲染像素，但已做以下静态检查：
- KaTeX 行内 `$…$` 与块级 `$$…$$` 标记全部配对（见上）。
- 公式中的 KaTeX 命令（`\sum`、`\mathbb{R}`、`\mathrm{}`、`\text{}`、`\tanh`、`\odot`、`\tilde`、`\alpha`、`\ell`、`\beta`、`\gamma`、`\operatorname` 未使用、`\sqrt`、`\frac`）均为 KaTeX 标准支持命令。
- ASCII 图示用 `<pre class="diagram">`，节点与箭头方向在正文定义。
- 外壳脚本的目录生成、章节折叠、暗/亮模式切换、阅读时间估计均由模板自带脚本保证，未改动脚本。
- 折叠块（S5 教学示例）使用标准 `<details>` 标签，外壳样式已覆盖。

浏览器中的实际像素渲染待编排者 check 阶段在浏览器中确认。

## 写作偏差

无。大纲的全部章节、学习目标、前置知识、完成检查、过渡均已按 outline.md 落实，无新增/删除核心章节、无更换贯穿例子、无改变前置知识映射、无把正文必要内容移入折叠块、无使用证据不足论断。
