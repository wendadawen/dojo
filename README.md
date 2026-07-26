# Paper Reading

论文解读和概念学习笔记。

## 目录结构

```text
.
├── index.html                 # 首页，由 scripts/generate_index.py 从 content.json 生成
├── content.json               # 最小内容 manifest：控制首页条目、顺序和描述
├── AGENTS.md / CLAUDE.md / CODEBUDDY.md   # AI CLI 发现入口，指向 workflows/concept.md
├── content/                   # 最终可阅读内容
│   ├── papers/                # 论文解读页面（content/papers/<venue>/<name>/）
│   └── concepts/              # 概念学习页面（content/concepts/<name>/）
├── templates/                 # HTML 骨架模板，不是发布内容
│   ├── paper/
│   └── concept/               # index.html 外壳 + components.html 组件库
├── workflows/                 # 写作流程和质量标准
│   ├── paper.md
│   ├── concept.md             # concept 工作流编排入口
│   └── concept/               # concept 各阶段规则（step-1~5）与写法示例库
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

concept 工作流按五阶段进行：研究范围 → 教学大纲 → 生成初稿 → 独立审查 → 修复与发布。研究范围、教学大纲和最终发布各设一个人工确认门槛，未确认不得进入后续阶段。

1. 用一句话启动，例如："执行 concept 工作流，概念是 PPO"
2. AI 依次阅读 `workflows/concept.md`（编排入口）和 `workflows/concept/` 下的阶段文件执行
3. 在三个确认点审阅确认材料，明确回复通过或退回意见
4. 最终确认通过后，AI 更新 `content.json`，并运行 `python3 scripts/generate_index.py` 生成首页

产物：`content/concepts/<name>/index.html`；研究材料、确认记录和审查报告等过程证据保存在 `content/concepts/<name>/research/`。

## 内容边界

- `content/` 只放读者会看的最终内容和必要附件
- `templates/` 只放可复制的 HTML 骨架
- `workflows/` 只放写作规则，不再单独维护重复的 prompt 文件
- `scripts/` 只放可复用生产工具
- 临时素材、草稿、抓取结果和一次性 prompt 不入库

## 在线访问

GitHub Pages 首页：打开 `index.html` 或访问仓库地址。
