# Paper Reading

论文解读和概念学习笔记。

## 目录结构

```text
.
├── index.html                 # 首页
├── README.md
├── papers/                    # 论文解读
│   ├── workflow.md            # 论文解读写作流程
│   ├── template/              # 论文 HTML 骨架模板
│   │   ├── index.html
│   │   ├── summary.html
│   │   ├── qa.html
│   │   └── img_to_b64.py
│   └── iclr2026/              # ICLR 2026 论文
│       ├── hsd/
│       ├── memagent/
│       └── pless/
└── concepts/                  # 概念学习
    ├── workflow.md            # 概念教学 HTML 写作规范
    ├── training-landscape.md  # 训练 / RL / post-training 学习地图
    └── backward/              # 线性层 backward 推导
        └── index.html
```

## 写新论文笔记

1. 读 `papers/workflow.md` 了解完整流程与规范
2. 拷贝 `papers/template/` 对应骨架到 `papers/<venue>/<method-name>/`
3. 图片用 `python papers/template/img_to_b64.py figs/xxx.png` 转 base64 内联
4. 写完按 `papers/workflow.md` 末尾的核查清单逐项检查
5. 在根 `index.html` 论文列表加条目

## 写新概念学习材料

1. 读 `concepts/workflow.md` 了解公式、动画和交互规范
2. 每个概念单独建目录，例如 `concepts/sft/`、`concepts/attention/`
3. 主入口统一命名为 `index.html` 或 `index.md`
4. 在根 `index.html` 的概念学习列表加条目

## 在线访问

GitHub Pages 首页：打开 `index.html` 或访问仓库地址。
