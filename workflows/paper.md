# 论文解读文档写作流程

**产物**：纯 HTML + KaTeX（本地库）+ 图片 base64 内联。每篇论文目录只保留 HTML、可选问答页和 PDF。

**读者画像**：聪明的外行——有技术基础但不懂这个领域。30 秒能抓要点、3 分钟读摘要、10 分钟读全文。

**模板**：`templates/paper/` 目录下三套 HTML 骨架（主文档 / 简要总结 / 常见问答），拷贝改。

**首页**：新增内容后改 `content.json`，再运行 `python scripts/generate_index.py`。

---

# 一、流程

### 1. 收集证据

- 下载论文 PDF；arXiv 论文同时下载 TeX 源码包，解压到临时目录
- 优先读 TeX 源码，没有则用 `pdftotext` 提取
- 搜 GitHub 找官方源码，clone 或浏览核心文件
- 收集图片：TeX 源码 `figs/` > GitHub 仓库 > arXiv HTML > PDF 截图（最后手段）
- 每个要写的论断记下出处（§X / Table X / 代码文件:行号）

### 2. 判断论文是否适合本格式

本结构适合“有方法 + 有实验”的论文。检查：

- 论文有明确可命名的方法吗？
- 论文有可量化的实验结果吗？
- 论文的核心贡献能用一句话讲清吗？

三项都“是” → 继续。任何一项“否” → 考虑改成概念文档或简化结构。

### 3. 设计文档骨架

回答三个问题：

1. **一句话本质**：这篇论文用 X 解决了 Y 问题
2. **三个关键数字**：哪三个数字最能证明“解决了”
3. **一个核心直觉**：用什么直觉把方法讲清楚

### 4. 准备目录

```bash
mkdir -p content/papers/<venue>/<method-name>
cp templates/paper/index.html content/papers/<venue>/<method-name>/index.html
cp templates/paper/summary.html content/papers/<venue>/<method-name>/summary.html
```

如果需要问答页，再拷贝：

```bash
cp templates/paper/qa.html content/papers/<venue>/<method-name>/qa.html
```

目录结构：

```text
content/papers/<venue>/<method-name>/
├── index.html       主文档
├── summary.html     3 分钟摘要
├── qa.html          常见问答（可选）
└── paper.pdf        论文原文
```

所有中间产物（插图 `figs/`、`*.b64`、TeX 源码、`pdftotext` 文本等）内联/归档完成后删除，不提交。

### 5. 写主文档 index.html

按“规范”部分写。模板已实现默认结构，改占位符即可。

### 6. 写简要总结 summary.html

300-500 字，4 段：问题 / 方案 / 效果 / 启示（可选）。数字从主文档复制，不重新编写。

### 7. 插图

脚本输出 data URI 字符串，粘进 `<img src="...">` 的引号里：

```bash
python scripts/img_to_b64.py figs/arch.png
```

每张图三段结构：引导句 → 图片 → 解释段。主文档 3-5 张，覆盖问题/方法/实验至少各一张。

### 8. 更新首页

在 `content.json` 的 `items` 增加一条：

```json
{
  "id": "method-name",
  "section": "papers",
  "group": "ICLR 2026",
  "tag": "method-name",
  "title": "论文标题：一句话概括",
  "desc": "完整解析 · 术语速查 · 实验",
  "path": "content/papers/iclr2026/method-name/"
}
```

运行：

```bash
python scripts/generate_index.py
python scripts/validate_content.py
```

---

# 二、规范

## 主文档结构

严格按此顺序，H1 标题用结论式：

```text
顶部 callout（30 秒层）
  问题 → 方案 → 效果，各一句话。

论文元信息 blockquote
  论文名 / 作者 / 单位 / 链接 / 代码仓库 / 发布时间。

这篇论文一句话
  把一句话本质展开成 2-3 句。

术语速查表
  只列正文反复出现的术语。

一、问题（H1，结论式标题）
  本章你将知道：<一句话>
  先讲痛点，再引出本文要解决什么。

二、方法（H1，结论式标题）
  本章你将知道：<一句话>
  先讲直觉，再给公式，最后讲性质。

三、实验（H1，结论式标题）
  本章你将知道：<一句话>
  只突出 3 个关键数字，其余进表格。

四、总结（H1，结论式标题）
  本章你将知道：<一句话>
  方法本质回顾 + 局限 + 启发。

证据边界声明（H1）
  本章你将知道：每个论断的出处。
  分两段：论文明确结论 + 解读者推断。
```

每个 H1（含证据边界声明）下面都要有“本章你将知道”。

## 写作要点

- 每个论断必须有出处，出处集中在“证据边界声明”
- 方法章节先讲直觉，再给公式
- 实验对比展示所有相关基线，不只挑有利数字
- 局限只写论文明确观察到的范围限制，自己的判断放“解读者推断”
- 写完先删 20%，只保留读者理解必需内容

---

# 三、核查清单

### 外行友好度

- [ ] 只读 callout + H1 + “本章你将知道”，能不能复述论文做了什么？
- [ ] 术语表有没有用未解释术语解释术语？
- [ ] 方法章节第一段是不是直觉？
- [ ] 最技术的一句话，外行能不能读懂？

### 事实准确

- [ ] 每个论断有出处，集中在“证据边界声明”
- [ ] 实验对比展示完整相关基线
- [ ] 局限不脑补
- [ ] 论文事实和解读者推断分开
- [ ] 无临时标记：`未核对` / `待确认` / `待补出处`

### 格式

- [ ] 每个 H1 都有“本章你将知道”
- [ ] KaTeX 无红色 ParseError
- [ ] 宽表用 `.table-scroll`
- [ ] 关键数字加粗
- [ ] 本地资源路径指向 `../../../../libs/`
- [ ] `python scripts/validate_content.py` 通过

---

# 四、铁律

1. **证据优先**：找不到出处就删。
2. **直觉先行**：方法章节先讲直觉，再给公式。
3. **不选择性对比**：表格里有几行，正文就覆盖几行。
4. **关键假设不藏折叠块**：主路径必须看到。
5. **首页 manifest 驱动**：不要手改首页卡片，改 `content.json` 后生成。
