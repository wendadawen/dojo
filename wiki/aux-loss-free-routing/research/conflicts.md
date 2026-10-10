# conflicts

- 位置：原始论文（src-lfb/iclr2024_conference.tex）3B 实验的训练 token 数。§4.1 Experimental Setups 写「we train the 1B model on 100B tokens and the 3B model on 200B tokens」，Table 1（tab:main_results）3B 行对应 200B；Conclusion 一节却写「trained on 100B and 300B tokens respectively」。页面采用 200B（与正文和表格一致）。
- 位置：bias 更新步长的记号。原始论文（src-lfb/）Algorithm 1 与 §3 记为 update rate $u$；DeepSeek-V3 报告（src-dsv3/main.tex §2.1.2）记为 bias update speed $\gamma$。不是数值冲突，仅记号不同；页面统一用 $\gamma$，并在首次出现时说明原始论文记作 $u$。
- 位置：门控权重是否归一化，以及 $P_i$ 用的是归一化前后的分数。原始论文（src-lfb/iclr2024_conference.tex L164、L184、L223）的门控不归一化，$g_{i,t}=s_{i,t}$，$P_i=\frac{1}{T}\sum_t s_{i,t}$（未归一化的路由分数）；DeepSeek-V3 报告（src-dsv3/main.tex L342、L356、L382-L383）的门控要归一化，$g_{i,t}=g'_{i,t}/\sum_j g'_{j,t}$，$P_i=\frac{1}{T}\sum_t s'_{i,t}$，$s'$ 是归一化后的亲和度。页面讲模型门控时依 DeepSeek-V3 报告（归一化），讲辅助损失 $L_{bal}$ 时依原始论文（$P_i$ 用未归一化的路由分数）。两处各自照其依据写，页面上不讨论。
