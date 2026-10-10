# 冲突记录

- 位置：论文里 KDA 转移矩阵的因子次序，table/parallel.tex L57-L59 与 6-discuss.tex L20 写作 $\prod_j\operatorname{Diag}(\boldsymbol{\alpha}_j)(\mathbf{I}-\boldsymbol{k}_j\boldsymbol{k}_j^\top)$（对角门在前），而 3-kda.tex L4（主式 eq:recurrent_KDA）与 appendix/proof.tex L35 写作 $(\mathbf{I}-\beta_j\boldsymbol{k}_j\boldsymbol{k}_j^\top)\operatorname{Diag}(\boldsymbol{\alpha}_j)$（修正项在前）。两处说法不一致。
- 页面采用：按正文（3-kda.tex 主式 eq:recurrent_KDA）与附录证明，取 $(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top)\operatorname{Diag}(\boldsymbol{\alpha}_t)$，即修正项在前、对角门在后。全页（短页更新式、状态、修正、门控、分块、RoPE 对照）都按这一次序写。
