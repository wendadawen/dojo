# 收集

只做一件事：把这次前向要用到的官方资料收到 `wiki/<name>/research/sources/`。不分析，不判断哪份有用，收完再说。

## 收什么

- 源码：优先用 transformers 里的实现，把所用版本或 commit 写进资料。transformers 里没有这个模型时，用官方仓库的源码。
- `config.json`：模型仓库里的那份，存一份副本到 `sources/config.json`。
- 权重文件头：拿得到时按 HTTP Range 只取 `safetensors` 头部解析张量形状，写成 `sources/shapes.json`，键是张量名、值是维度数组。同名形状一致时合并成 `<名字>.{i}` 一条。
- 论文或技术报告：有就收。

和其他资料对不上时，以这一步选定的源码为准。

## 怎么存

按类型放进 `sources/` 下的子目录：`repo/` 放源码，`paper/` 放论文的 PDF 和 LaTeX 源码（有的话）。`config.json` 与 `shapes.json` 直接放在 `sources/` 下，不建子目录。

`sources/index.md` 每份资料一行：

```markdown
- <文件或目录名>｜<类型：源码/论文/配置/权重头>｜<来源链接>｜<版本或 commit>｜<取回日期>｜<一句话说明它讲什么>
```
