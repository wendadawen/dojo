# Paper Reading

论文解读笔记。每篇论文一个目录，纯 HTML + KaTeX 单文件，离线可读。

## 目录结构

```
.
├── index.html       # 首页，论文列表
├── workflow.md      # 写作流程与规范
├── template/        # HTML 骨架模板
│   ├── index.html   # 主文档模板
│   ├── summary.html # 3 分钟摘要模板
│   ├── qa.html      # 常见问答模板
│   └── img_to_b64.py
├── hsd/             # HSD：分层概率账本提高草稿接受率
├── memagent/        # MemAgent：固定长度记忆 + RL
└── pless/           # p-less Sampling：超参-free 截断采样
```

每个论文目录固定三个文件：

- `index.html` — 主文档（问题 / 方法 / 实验 / 总结 / 证据边界声明）
- `summary.html` — 3 分钟摘要
- `paper.pdf` — 论文原文

## 写新笔记

1. 读 `workflow.md` 了解完整流程与规范
2. 拷贝 `template/` 对应骨架到 `<method-name>/`
3. 图片用 `python template/img_to_b64.py figs/xxx.png` 转 base64 内联
4. 写完按 `workflow.md` 末尾的核查清单逐项检查
5. 在根 `index.html` 论文列表加条目

## 在线访问

GitHub Pages 首页：打开 `index.html` 或访问仓库地址。
