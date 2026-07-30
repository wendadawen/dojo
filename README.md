# Dojo

一个与 AI 对练的通用学习仓库：读论文、学概念、问问题、读 PR，并在需要时把已经学清楚的内容整理成可浏览的 HTML。

## 使用方式

把仓库交给 Claude Code、Codex、CodeBuddy、WorkBuddy 等能读取项目文件的 AI，然后直接说你的需求：

- “帮我精读这篇论文并生成页面：<链接>”
- “我想系统学习 DPO，并生成概念页”
- “为什么 NCCL 还需要依赖网络传输库？”
- “一步一步带我读懂这个 PR：<链接>”
- “把我们刚才学清楚的内容整理成笔记”

AI 会从 `AGENTS.md` / `CLAUDE.md` / `CODEBUDDY.md` 识别场景，并加载 `prompts/` 下对应的独立提示词。

## 目录

```text
prompts/           五种学习场景的提示词
workflows/         论文与概念页面的现有生产流程
content/papers/    论文解读页面
content/concepts/  概念学习页面
content/notes/     对话后按需生成的结构化笔记
templates/         三类 HTML 页面模板
content.json       首页内容清单
```

## 在线访问

https://wendadawen.github.io/dojo/

GitHub 仓库、本地项目目录与站点品牌均使用 `dojo`。
