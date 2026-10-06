# Dojo

技术学习仓库：整理论文解析、概念说明、问题解答与学习记录。

## 目录结构

```text
guides/       各类学习任务的操作指南
wiki/         学习产出的可浏览页面，按需创建（wiki/<name>/index.html）
.dojo/        通用构建设施（模板、首页资源、通用脚本），不含任何页面专属内容
index.html    GitHub Pages 首页应用
```

`.dojo/` 下只放通用设施，不出现具体模型名或页面名。某页专属的内容（生成脚本、
图片）放进该页自己的 `wiki/<name>/` 目录。分组说明见 `.dojo/scripts/README.md`。

## 任务路由

| 任务类型 | 读取并执行 |
|---|---|
| 了解某个概念、系统学习概念、把概念写成页面 | `guides/concept.md` |

目前只维护概念页。论文解析、学习记录、问题解答、模型数据流的指南已移到 `guides/archive/`，对应的 `wiki/` 页面在元数据里标为 `dojo:status=outdated`，首页默认不显示。这几类任务等概念页稳定后再恢复，恢复前不要按归档指南生成新页面。

执行任何任务前，先完整读取 `guides/concept.md`，再开始动手，按它从第 1 步做到交付。需求不明确时，只提出一个能够确定是哪个概念的必要问题。

## 项目约束

- 产出统一为 HTML 页面，写入 `wiki/<name>/`；目录与 Graph 由 GitHub Pages 构建扫描 `wiki/*/index.html` 生成。
- 页面 `<head>` 包含 `description`（纯文本）、`dojo:summary`（可含 `$...$` 公式）、`dojo:type`、`dojo:topics`、`dojo:tag`。
- `dojo:topics` 是粗分类，一个页面可以有多个，只能从这些大类里选：注意力机制、模型结构、推理系统、内存与缓存、并行与通信、训练与优化、多模态、数学基础、算法与数据结构、计算机系统、概率与统计、其他领域。概念不属于前面任何一类时用「其他领域」。需要新大类时，改 `.dojo/scripts/ci-01-validate.py` 的 `ALLOWED_TOPICS`，并同步本文件列出的大类，再使用。
- `dojo:tag` 只取一个值，写这篇讲什么技术，词表见 `.dojo/scripts/ci-01-validate.py` 的 `ALLOWED_TAGS`。首页按类型和主题筛选，标签参与搜索。新领域的头几个概念先用「通用概念」，同一领域攒够 3 个再新增专属标签，并把已有页面改过去。
- 页面以 `../../libs/` 引用共享库。
- 页面图片存入 `wiki/<name>/assets/`，用相对路径 `assets/...` 引用。
- 生成页面后运行 `.dojo/scripts/ci-01-validate.py <页面路径>`。
- 未获明确授权，不执行 commit、push、仓库重命名或部署。
