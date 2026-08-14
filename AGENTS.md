# Dojo

一个工具无关的 AI 学习仓库：读论文、学概念、问问题、读 PR，并在需要时把学清楚的内容留成可浏览页面。不同任务差异很大，因此没有统一总工作流——先识别用户意图，再只读取对应的场景指南。

## 目录结构

```text
guides/       每个学习场景一份指南（AI 的操作入口）
wiki/         学习产出的可浏览页面，按需创建（wiki/<name>/index.html）
.dojo/        构建素材与脚本（模板、首页资源、目录构建器、校验器）
index.html    GitHub Pages 首页应用
```

## 场景路由

| 用户意图 | 读取并执行 |
|---|---|
| 读论文、精读论文、把论文生成 HTML | `guides/paper.md` |
| 系统学习概念并生成 HTML | `guides/concept.md` |
| 提一个问题、希望通过对话弄懂 | `guides/explain.md` |
| 带我读懂一个 PR / diff | `guides/pr.md` |
| 把已经学清楚的内容整理留档 | `guides/note.md` |

如果意图不明确，只问一个足以判断场景的问题，不要一次发问卷。场景进行中以用户最新要求为准；用户从「解释」切到「留档」时，再加载 `guides/note.md`。

## 项目约束

- 可浏览内容统一生成 HTML，放入 `wiki/<name>/`。不要手工登记首页；GitHub Pages 构建会扫描 `wiki/*/index.html` 自动生成目录与 Graph 数据。
- 新页面的 `<head>` 必须包含 `description`、`dojo:type`、`dojo:topics`、`dojo:tag` 元数据；页面关系自动来自 HTML 内部链接。
- `index.html` 是稳定首页应用，不运行旧的 `.dojo/scripts/generate_index.py`，不因新增文章而重写。
- 页面用相对路径引用 `libs/`（`wiki/<name>/` 深两层，即 `../../libs/...`）。
- 生成页面后运行 `.dojo/scripts/validate.py <页面路径>`。
- 全站通过 GitHub Pages 静态发布；不引入运行时后端或在线 API。
- 不自动 commit、push、重命名 GitHub 仓库或部署，除非用户明确要求。
