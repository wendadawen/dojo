# Dojo

技术学习知识库。

```text
guides/       任务指南
wiki/         页面，每页一个目录 wiki/<name>/
.dojo/        模板、首页源码、脚本
libs/         页面共用的前端库
index.html    首页
```

## 任务

用户想了解或学习某个概念，读 `guides/concept.md`，按它的六步做完，每步开始前读 `guides/concept/` 下对应的文档。需求不清楚时，只问一个能确定是哪个概念的问题。

## 约束

- 页面写入 `wiki/<name>/`，用 `guides/concept.md` 里的脚本生成，不手写 HTML。
- 主题和标签只能取 `.dojo/scripts/site/vocab.py` 里的词表，写在 `content.md` 开头。都不合适，主题用「其他领域」，标签用「通用概念」。
- 未获明确授权，不执行 commit、push、仓库重命名或部署。
