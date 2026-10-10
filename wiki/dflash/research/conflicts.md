# 资料冲突

- 加速倍数口径｜src-v2/sections/intro.tex 图 1 标题写 "more than 2.5× higher speedup than EAGLE-3"（泛指更高）；arXiv 摘要页 https://arxiv.org/abs/2602.06036 写 "up to 2.5× higher speedup"（最高）｜页面不引用这一比例，不涉及
- 无目标特征草稿器的加速幅度｜src-v2/sections/method.tex Inference 写 "the resulting speedups are modest, typically around 2–3×"；src-v2/sections/appendix.tex Diffusion Drafter without Target Feature 的表 tab:naive_diffusion 给出 2.83/3.73/3.43/3.35（温度 0）与 2.76/3.31/2.66/2.65（温度 1），最大 3.73 越出 2–3｜页面按表格写「典型在 2 到 3 倍左右，个别任务到 3.7 倍」，折叠块列出全部行
- $W_c$ 与投影激活的字节数舍入｜src-v2/sections/appendix.tex KV Injection Mechanism and Memory Overhead 同页写 "$5\times2048\times2048\times2\approx42$ MB" 与 "the projection input and output require about 40 MB and 8 MB"，两个量同为 5·2048·2048·2 字节却分别舍入成 42 与 40｜页面照写（前者是权重、后者是激活输入，是两个量），不改
