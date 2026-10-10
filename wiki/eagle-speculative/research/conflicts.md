# 冲突记录

- 位置：前作 EAGLE 复用 target 的哪一层特征（页面「只用最深的那一层」一段的背景）。两处说法：论文 paper/src-v3/paper.tex L121、L157 写 EAGLE「reuses the top-layer features of the target model (the features before the LM head)」，即最顶层；官方仓库 repo/EAGLE/README.md 写 EAGLE「extrapolating the second-top-layer contextual feature vectors」，即倒数第二层。页面采用论文的说法（最深一层）。依据：paper/src-v3/paper.tex L121；repo/EAGLE/README.md L34。
- 位置：草稿取哪一层出分布（页面「一步草稿」一段说「草稿直接借 target 的 LM head，不另训」）。两处说法：论文 paper/src-v3/paper.tex L183、L209 写「uses the LM head of the target model」「we input a_I into the LM head」，即借用 target 的 LM head；官方仓库 repo/EAGLE/eagle/traineagle3/cnets.py L531 给草稿单独建 `lm_head = nn.Linear(hidden_size, draft_vocab_size)` 并在 L851 用它，即草稿另训一个词表较小的输出层。页面采用论文的说法（借 target 的 LM head）。依据：paper/src-v3/paper.tex L183；repo/EAGLE/eagle/traineagle3/cnets.py L531。

