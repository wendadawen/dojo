# 监督微调

- 描述：监督微调（SFT）是在预训练语言模型之上，用人工写好的「指令—回答」标注数据继续训练，让模型从「接着续写」变成「按指令作答」。
- 摘要：用人工写好的示范数据继续训练预训练模型，只对回答部分算损失；模型由此学会按指令作答，并成为后续偏好优化的起点。
- 主题：训练与优化
- 标签：训练

## 长页

- 开头｜图景：文本先切成 token（模型读写文字的基本单位）；没按指令训练过的预训练模型收到指令只会续写；SFT 用示范数据继续训练，把「按指令作答」写进参数｜250｜paper/instructgpt/src/neurips_2021.tex L121；paper/llama2/src/templateArxiv.tex L343-L392
- ### 目标错位｜预训练目标是预测下一个 token，用户目标是遵循指令，两者错位；在输入里给几个例子（提示）为什么不够｜250｜paper/instructgpt/src/neurips_2021.tex L121；paper/instructgpt/src/neurips_2021.tex L125-L126
- ### 数据与损失｜样本是指令加回答；沿用下一 token 交叉熵、整条序列一次送入；交叉熵与因果掩码链接前置页｜250｜paper/llama2/src/templateArxiv.tex L343；paper/llama2/src/templateArxiv.tex L391
- #### 样本形态｜一条样本的组成与拼接；多轮对话只监督最后一轮回答｜200｜paper/llama2/src/templateArxiv.tex L343；paper/llama2/src/templateArxiv.tex L391；paper/llama2/src/templateArxiv.tex L680
- #### 交叉熵与掩码｜逐位置交叉熵与序列总损失公式（折叠）；掩码把指令 token 置零、只在回答 token 上算损失；用小词表手算掩码与否两种损失（折叠）｜400｜paper/llama2/src/templateArxiv.tex L391
- ### 在训练流程中的位置｜SFT 是后训练的第一步；之后用人类对多个回答的比较数据训练一个给回答打分的模型（奖励模型），再按它的打分做偏好优化｜180｜paper/instructgpt/src/neurips_2021.tex L191-L195
- ### 数据量与质量｜Llama 2 弃用数百万第三方数据、改用少量自标注；数万条够用，数字折叠；标注平台差异明显｜300｜paper/llama2/src/templateArxiv.tex L378-L384
- ### 训练设置｜两篇的轮数、学习率、批大小、序列长度、dropout，条件与数字折叠；dropout 只有 InstructGPT 报告｜200｜paper/llama2/src/templateArxiv.tex L389-L392；paper/instructgpt/src/neurips_2021.tex L296；paper/instructgpt/src/neurips_2021.tex L1196；paper/instructgpt/src/neurips_2021.tex L1202
- ### 适用条件与边界｜验证损失过拟合不等于该停训，SFT 模型按奖励模型打分选；两条与直觉相反的训练要点｜250｜paper/instructgpt/src/neurips_2021.tex L296；paper/instructgpt/src/neurips_2021.tex L1202

## 短页

- ### 在示范数据上继续训练｜在预训练模型上继续下一 token 交叉熵训练（前置[交叉熵](wiki/cross-entropy)）；token 是文本切出的单位｜250｜paper/llama2/src/templateArxiv.tex L343-L392；paper/instructgpt/src/neurips_2021.tex L121
- ### 续写不是回答｜为什么需要：预训练的下一 token 目标与用户要的按指令作答错位；SFT 把「看到指令就作答」变成默认行为｜250｜paper/instructgpt/src/neurips_2021.tex L121；paper/instructgpt/src/neurips_2021.tex L125-L126
- ### 损失只算在回答上｜怎么做：样本由指令和回答拼成、拼接成固定长度，指令段损失置零、只监督回答｜400｜paper/llama2/src/templateArxiv.tex L391
