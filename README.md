# Dojo

一个与 AI 对练的通用学习仓库：读论文、学概念、问问题、读 PR，并在需要时把已经学清楚的内容整理成可浏览的 HTML。

## 使用方式

把仓库交给 Claude Code、Codex、CodeBuddy、WorkBuddy 等能读取项目文件的 AI，然后直接说你的需求：

- “帮我精读这篇论文并生成页面：<链接>”
- “我想系统学习 DPO，并生成概念页”
- “为什么 NCCL 还需要依赖网络传输库？”
- “一步一步带我读懂这个 PR：<链接>”
- “把我们刚才学清楚的内容整理成笔记”

AI 会从 `AGENTS.md` 识别场景，并加载 `guides/` 下对应的场景指南。

## 目录

```text
guides/       每个学习场景一份指南
wiki/         学过的内容生成的可浏览页面（wiki/<name>/index.html）
.dojo/        HTML 模板、首页资源、目录构建器与校验器
index.html    GitHub Pages 首页应用
```

## 首页与知识地图

GitHub Pages 发布时会扫描 `wiki/*/index.html`，自动生成文章目录和页面链接 Graph。新增文章只需提供页面元数据，不需要维护独立清单或重写 `index.html`。

## 在线访问

https://wendadawen.github.io/dojo/

GitHub 仓库、本地项目目录与站点品牌均使用 `dojo`。
