# Dojo

一个用于整理技术学习材料的静态知识库，覆盖论文解析、概念说明、问题解答、PR 分析与学习记录。

## 使用方式

在支持读取项目文件的开发工具中打开仓库，并按任务提供必要输入：

- 论文链接、标题或 PDF：生成论文解析页面
- 概念名称：生成概念说明页面
- 具体技术问题：进行解释与推导
- PR 链接、编号或 diff：分析改动目的与实现
- 已确认的材料或结论：整理为学习记录

`AGENTS.md` 负责任务路由，`guides/` 保存各任务的执行规范。

## 目录

```text
guides/       各类学习任务的操作指南
wiki/         学过的内容生成的可浏览页面（wiki/<name>/index.html）
.dojo/        HTML 模板、首页资源、目录构建器与校验器
index.html    GitHub Pages 首页应用
```

## 首页与知识地图

GitHub Pages 发布时会扫描 `wiki/*/index.html`，自动生成文章目录和页面链接 Graph。新增文章只需提供页面元数据，不需要维护独立清单或重写 `index.html`。

## 在线访问

https://wendadawen.github.io/dojo/

GitHub 仓库、本地项目目录与站点品牌均使用 `dojo`。
