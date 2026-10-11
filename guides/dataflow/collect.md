# 收集

只做一件事：把这次前向要用到的官方资料收到 `wiki/<name>/research/sources/`。不分析，不判断哪份有用，收完再说。

## 收什么

- 源码：优先用 transformers 里的实现，把所用版本或 commit 写进资料。transformers 里没有这个模型时，用官方仓库的源码。
- `config.json`：模型仓库里的那份，存一份副本到 `sources/config.json`。
- 权重文件头：拿得到时按 HTTP Range 只取 `safetensors` 头部解析张量形状**与 dtype**，写成 `sources/shapes.json`。值的写法：只有形状，写成维度数组 `[5120, 2560]`；**记得到 dtype 时写成 `{"shape": [5120, 2560], "dtype": "BF16"}`**（safetensors 头的 `dtype` 字段原样记，如 `BF16`、`F32`）。同名形状一致时合并成 `<名字>.{i}` 一条。
- 模型卡：模型仓库的 `README.md`，存一份副本到 `sources/model-card.md`。**介绍面板要写的参数量、每 token 激活量、上下文长度一律出自它**——不收它，那几个数就只能凭印象写。
- 论文或技术报告：有就收。

和其他资料对不上时，以这一步选定的源码为准。

## 怎么存

按类型放进 `sources/` 下的子目录：`repo/` 放源码，`paper/` 放论文的 PDF 和 LaTeX 源码（有的话）。`config.json` 与 `shapes.json` 直接放在 `sources/` 下，不建子目录。

每份资料的来源、版本、取回日期和一句话说明写进 `dataflow.md` 的「资料」一节（见 `plan.md`），不再单独建索引文件。
