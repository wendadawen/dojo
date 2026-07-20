# 概念教学文档写作规范

写 `concepts/` 下的概念学习材料时使用。目标是：单个概念一个目录，主入口叫 `index.html` 或 `index.md`，方便从首页链接。

## 目录约定

```text
concepts/
├── workflow.md
├── training-landscape.md
└── backward/
    └── index.html
```

以后新增概念时：

```text
concepts/<concept-name>/
└── index.html
```

例如：

```text
concepts/sft/index.html
concepts/rlhf/index.html
concepts/attention/index.html
```

## HTML 依赖原则

1. **默认单 HTML 文件**：CSS / JS 优先写在同一个 HTML 里。
2. **KaTeX 默认跟 `papers/` 保持一致，用 CDN**：目录更干净，不单独放 `assets`。
3. **如果 CDN 再次导致公式无法渲染**：再单独讨论是否引入本地 KaTeX，不默认创建 `assets/`。
4. **不用 auto-render 写复杂教学动画**：优先自己用 `katex.render` 控制渲染，避免局部公式不渲染时难排查。

HTML 头部默认写法：

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
```

## 数学渲染规则

- 变量和公式都用 LaTeX，不要一半数学符号、一半工程变量名。
- JS 字符串里的 LaTeX 命令必须写双反斜杠：`\\frac`、`\\partial`。
- 不用 `\mathbb{1}`，KaTeX 对数字黑板体支持不好；用 `\delta_{ky}` 表示 Kronecker delta。
- 公式里的箭头用 `\to` 或 `\rightarrow`。

## 交互偏好

- 固定底部导航：上一页 / 下一页位置不要随内容变化。
- 不要自动播放。
- 支持键盘 ← / → 翻页。
- 不要“播放 / 暂停 / 速度”控件。

## 视觉偏好

- 深色背景：深蓝渐变 + 径向光晕。
- 标题渐变文字：蓝 → 紫 → 粉。
- 毛玻璃卡片：`backdrop-filter: blur(10px)`。
- 矩阵格子可以高亮、发光、轻微缩放。
- 数学约定用卡片网格，不要堆成一长串公式。

## 写作注意

- 不要批量替换公式和变量，容易引入次生 bug。
- 改 HTML / Markdown 后立即读回检查，避免代码块或公式被清空。
- 每个阶段最好包含：标题、直觉解释、公式、数值例子、提示。
- 教学文档要先解释“这一步在干什么”，再给公式。

## 常见 bug

| 现象 | 原因 | 解决 |
|------|------|------|
| `$...$` 原样显示 | KaTeX 没加载 | 先检查 CDN；必要时再考虑本地 KaTeX |
| `renderMathIn is undefined` | auto-render 没挂到全局 | 不依赖 auto-render，直接用 `katex.render` |
| `rac{...}` 乱码 | JS 把 `\f` 当 form feed | JS 字符串里写 `\\frac` |
| `k=y` 或 indicator 显示异常 | `\mathbb{1}` 不适合 | 改成 `\delta_{ky}` |
| `\neq` 渲染成方块 | 字体没加载完整 | 检查 KaTeX CSS / fonts 是否加载成功 |
| 公式部分渲染部分不渲染 | 批量替换破坏字符串 | 手动按上下文逐处改 |
