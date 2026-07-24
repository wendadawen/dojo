# Paper Reading

论文解读和概念学习笔记。

## 目录结构

```text
.
├── index.html                 # 首页，由 scripts/generate_index.py 从 content.json 生成
├── content.json               # 最小内容 manifest：控制首页条目、顺序和描述
├── content/                   # 最终可阅读内容
│   ├── papers/                # 论文解读页面
│   │   └── iclr2026/
│   │       ├── hsd/
│   │       ├── memagent/
│   │       └── pless/
│   └── concepts/              # 概念学习页面和学习地图
│       ├── training-landscape.md
│       ├── backward/
│       ├── dpo/
│       ├── rlhf/
│       ├── grpo/
│       ├── mooncake/
│       └── vllm-pd/
├── templates/                 # HTML 骨架模板，不是发布内容
│   ├── paper/
│   └── concept/
├── workflows/                 # 写作流程和质量标准
│   ├── paper.md
│   └── concept.md
├── scripts/                   # 生产工具
│   ├── generate_index.py
│   ├── validate_content.py
│   └── img_to_b64.py
└── libs/                      # 前端本地依赖：KaTeX、Prism、字体
```

## 写新论文笔记

1. 读 `workflows/paper.md` 了解完整流程与规范
2. 拷贝 `templates/paper/` 到 `content/papers/<venue>/<method-name>/`
3. 图片用 `python scripts/img_to_b64.py figs/xxx.png` 转 base64 内联
4. 写完按 `workflows/paper.md` 的核查清单逐项检查
5. 在 `content.json` 增加条目
6. 运行：

```bash
python scripts/generate_index.py
python scripts/validate_content.py
```

## 写新概念学习材料

1. 读 `workflows/concept.md` 了解概念教学结构、公式、代码和证据规范
2. 拷贝 `templates/concept/index.html` 到 `content/concepts/<concept-name>/index.html`
3. 写完按 `workflows/concept.md` 的核查清单逐项检查
4. 在 `content.json` 增加条目
5. 运行：

```bash
python scripts/generate_index.py
python scripts/validate_content.py
```

## 内容边界

- `content/` 只放读者会看的最终内容和必要附件
- `templates/` 只放可复制的 HTML 骨架
- `workflows/` 只放写作规则，不再单独维护重复的 prompt 文件
- `scripts/` 只放可复用生产工具
- 临时素材、草稿、抓取结果和一次性 prompt 不入库

## 在线访问

GitHub Pages 首页：打开 `index.html` 或访问仓库地址。
