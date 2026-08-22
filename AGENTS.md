# Dojo

技术学习仓库：整理论文解析、概念说明、问题解答与学习记录。

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
| 将已确认的学习结论整理归档 | `guides/note.md` |

需求不明确时，只提出一个能够确定任务类型的必要问题。任务类型发生变化时，再读取对应指南。

## 项目约束

- 产出统一为 HTML 页面，写入 `wiki/<name>/`；目录与 Graph 由 GitHub Pages 构建扫描 `wiki/*/index.html` 生成。
- 页面 `<head>` 包含 `description`（纯文本）、`dojo:summary`（可含 `$...$` 公式）、`dojo:type`、`dojo:topics`、`dojo:tag`。
- 页面以 `../../libs/` 引用共享库。
- 生成页面后运行 `.dojo/scripts/validate.py <页面路径>`。
- 未获明确授权，不执行 commit、push、仓库重命名或部署。
