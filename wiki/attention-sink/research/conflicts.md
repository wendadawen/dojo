# 资料冲突

- Pythia 模型规模｜paper/src-v4/text/1_introduction.tex L24 写 Pythia-[2.9,6.9,12]B，paper/src-v4/text/4_experiment.tex L8 写 Pythia-[2.8,6.9,12]B｜页面只写 Pythia 的 12B（两处都认同的规模）
- 训练的模型个数｜paper/src-v4/text/4_experiment.tex L12 写 trained two language models，paper/src-v4/text/3_method.tex L52 与 paper/src-v4/figtab/pretrain_fix.tex L10-L14 是三个模型（Vanilla、Zero Sink、Learnable Sink）｜页面采用三个（按正文和表格），训练条件不写
- Falcon-7B 在 4+2044 的困惑度｜paper/src-v4/figtab/cache_size.tex L16 写 12.84，paper/src-v4/figtab/start_size_and_linebreak.tex L35 写 12.12；两张表的说明都写在拼起来的 PG19 测试集 400K token 上测，同一配置，MPT-7B（14.99）和 Pythia-12B（12.09）两表一致，只有 Falcon-7B 对不上｜页面采用 12.84（与正文「4+1020 最低」的叙述同出一表），折叠表里不再列 Falcon-7B 的初始 token 个数消融
