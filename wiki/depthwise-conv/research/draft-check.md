# 深度可分离卷积 draft-check

## 输入版本
- scope/evidence/outline/glossary 已落地（见同目录 research/）
- 模板换头部占位符 + content 插入
- 渲染实测通过（见 research/ 探针输出）

## 大纲落实
- 引言 + 核心问题 5 题 + 常见误解 + 5 个章节 + 来源与范围说明 —— 按 outline 全部进入
- 每章末尾「本章问题」2 题带解答折叠块
- 贯穿示例：D_K=3,M=N=64 成本计数、$1/N+1/D_K^2$ 推导验证、因果 1D depthwise 手算 等

## 目标覆盖检查
- Q1-Q5 全部由正文章节回答：✓
- 每个核心问题与章节问题都有解答折叠块，答案独立可读：✓

## 代码运行
- research/depthwise-conv_page_code.py：已运行通过
- 输出在 research/depthwise-conv_page_code.out
- 与页面声称的数字一致

## 机械检查
- python3 .dojo/scripts/validate.py depthwise-conv/index.html: ok
- headless Chrome 渲染实测：0 处未渲染 $...$、0 对标签重叠、0 处越界

## 公式渲染与交互
- KaTeX 全部渲染（详见 [N] 数字标注）
- 折叠块默认收起时正文仍能回答所有问题
