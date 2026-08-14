# Dojo

一个工具无关的技术学习仓库，用于整理论文解析、概念说明、问题解答、PR 分析与学习记录。不同任务分别遵循对应的操作指南。

## 目录结构

```text
guides/       各类学习任务的操作指南
wiki/         学习产出的可浏览页面，按需创建（wiki/<name>/index.html）
.dojo/        构建素材与脚本（模板、首页资源、目录构建器、校验器）
index.html    GitHub Pages 首页应用
```

## 任务路由

| 任务类型 | 读取并执行 |
|---|---|
| 读论文、精读论文、把论文生成 HTML | `guides/paper.md` |
| 系统学习概念并生成 HTML | `guides/concept.md` |
| 针对具体问题进行解释与推导 | `guides/explain.md` |
| 分析 PR / diff | `guides/pr.md` |
| 将已确认的学习结论整理归档 | `guides/note.md` |

需求不明确时，只提出一个能够确定任务类型的必要问题。任务类型发生变化时，再读取对应指南。

## 项目约束

- 可浏览内容统一生成 HTML，放入 `wiki/<name>/`。不要手工登记首页；GitHub Pages 构建会扫描 `wiki/*/index.html` 自动生成目录与 Graph 数据。
- 新页面的 `<head>` 必须包含 `description`、`dojo:type`、`dojo:topics`、`dojo:tag` 元数据；页面关系自动来自 HTML 内部链接。
- `index.html` 是稳定首页应用，不因新增文章而重写。
- 页面用相对路径引用 `libs/`（`wiki/<name>/` 深两层，即 `../../libs/...`）。
- 生成页面后运行 `.dojo/scripts/validate.py <页面路径>`。
- 全站通过 GitHub Pages 静态发布；不引入运行时后端或在线 API。
- 未获得明确授权时，不执行 commit、push、仓库重命名或部署。
