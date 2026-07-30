# Dojo

这是一个通用 AI 学习仓库。不同任务差异很大，因此没有统一总工作流；先识别用户意图，再只读取对应的场景提示词。

## 场景路由

| 用户意图 | 读取并执行 |
|---|---|
| 读论文、精读论文、把论文生成 HTML | `prompts/paper.md` |
| 系统学习概念并生成 HTML | `prompts/concept.md` |
| 提一个问题、希望通过对话弄懂 | `prompts/explain.md` |
| 带我读懂一个 PR / diff | `prompts/pr.md` |
| 把已经学清楚的内容整理留档 | `prompts/note.md` |

如果意图不明确，只问一个足以判断场景的问题，不要一次发问卷。场景进行中以用户最新要求为准；用户从“解释”切到“留档”时，再加载 `prompts/note.md`。

## 项目约束

- 可浏览内容统一生成 HTML，并登记到 `content.json`，再运行 `scripts/generate_index.py` 更新首页。
- 论文、概念、笔记分别进入 `content/papers/`、`content/concepts/`、`content/notes/`。
- 生成页面后运行 `scripts/validate.py <页面路径>`。
- 不自动 commit、push、重命名 GitHub 仓库或部署，除非用户明确要求。
