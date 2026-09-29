# Dojo

技术学习仓库：整理论文解析、概念说明、问题解答与学习记录。

## 目录结构

```text
guides/       各类学习任务的操作指南，示范在 guides/examples/
wiki/         学习产出的可浏览页面，按需创建（wiki/<name>/index.html）
.dojo/        通用构建设施（模板、首页资源、通用脚本），不含任何页面专属内容
index.html    GitHub Pages 首页应用
```

`.dojo/` 下只放通用设施，不出现具体模型名或页面名。某页专属的内容（生成脚本、
图片）放进该页自己的 `wiki/<name>/` 目录。分组说明见 `.dojo/scripts/README.md`。

## 任务路由

| 任务类型 | 读取并执行 |
|---|---|
| 读论文、精读论文、把论文生成 HTML | `guides/paper.md` |
| 了解某个概念、系统学习概念、把概念写成页面 | `guides/concept.md` |
| 针对具体问题进行解释与推导 | `guides/explain.md` |
| 将已确认的学习结论整理归档 | `guides/note.md` |
| 推导模型前向数据流并生成 HTML | `guides/model-dataflow.md` |

执行任何任务前，先完整读取对应指南，再开始动手。任务类型能从用户的话里确定时，直接做到交付，不要再问用哪种做法、要不要恢复旧页。「我要了解某某概念」走概念页，按 `guides/concept.md` 做到交付；目录里已有该页也要重写，不在对话里讲解，也不复述旧稿。需求不明确时，只提出一个能够确定任务类型的必要问题。任务类型发生变化时，再读取对应指南。

## 项目约束

- 产出统一为 HTML 页面，写入 `wiki/<name>/`；目录与 Graph 由 GitHub Pages 构建扫描 `wiki/*/index.html` 生成。
- 页面 `<head>` 包含 `description`（纯文本）、`dojo:summary`（可含 `$...$` 公式）、`dojo:type`、`dojo:topics`、`dojo:tag`。
- `dojo:topics` 只能从固定大类中选：注意力机制、模型结构、推理系统、内存与缓存、并行与通信、训练与优化、多模态、数学基础；需要新大类时先改 `.dojo/scripts/ci-01-validate.py` 的 `ALLOWED_TOPICS` 再使用。
- `dojo:tag` 取封闭词表中的**单一**值，供首页按技术筛选；词表见 `.dojo/scripts/ci-01-validate.py` 的 `ALLOWED_TAGS`。它与 `dojo:topics` 的分工是「粗分类」对「这篇讲什么技术」：技术栈名（vLLM、llama.cpp）属于实现细节，文档形态（速查、论文）由 `dojo:type` 承担，都不进词表。新增取值前先确认它至少有 3 个页面。
- 页面以 `../../libs/` 引用共享库。
- 页面图片存入 `wiki/<name>/assets/`，用相对路径 `assets/...` 引用。
- 生成页面后运行 `.dojo/scripts/ci-01-validate.py <页面路径>`。
- 未获明确授权，不执行 commit、push、仓库重命名或部署。
