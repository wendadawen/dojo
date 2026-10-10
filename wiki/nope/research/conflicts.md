# 冲突记录

- 1.3B 预训练模型的规模叫法：正文 Discussion 的 Scaling up to 1B models 一段写「three 1B variants post-submission」（main.tex L506），其后的说明与附录标题、表格写「1.3B models」「1.3B parameters」（main.tex L509、L1169、L1175）。页面采用 1.3B（随附录正文与表格）。
- NoPE 在 1.3B 代码模型上的外推长度：正文 Discussion 写「up to almost twice their training context size」（main.tex L508，训练上下文 1024 的近两倍），附录 Generalization Evaluation 写「up to 1800」（main.tex L1189）。页面折块数字采用 1800，正文只说「能外推到训练上下文之外的一定长度」，不写倍数。
- NoPE 外推长度的数字与表格：附录 Generalization Evaluation 写「generalize to larger context sizes up to 1800」（main.tex L1189），但同附录的困惑度表（tables/pretraining_ppl.tex）里，长度桶 [2000,2250] 起各桶的 NoPE 在上下文 1800 处困惑度已是 25 到 42。页面折块据正文写「约 1800」，并写「再长时两者困惑度都上升」。
