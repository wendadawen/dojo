# 语言模型预训练 术语表

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| 语言模型（language model） | 标题/开头 | token 序列上的概率分布；实际接口为给定前文预测下一 token 分布 |
| token | 开头 | 文本的最小处理单元 |
| 词表（vocabulary） | 第 1 章 | 全部 token 的集合 |
| 下一 token 预测（next token prediction） | 第 1 章 | 给定前文输出下一 token 分布的任务表述 |
| 条件概率 $p(s_i \mid s_{<i})$ | 第 2 章 | 给定前文 $s_{<i}$ 时下一个 token 为 $s_i$ 的概率 |
| 链式分解 / 自回归分解 | 第 2 章 | 联合概率 = 各步条件概率连乘 |
| 联合概率 | 第 2 章 | 整条序列一起出现的概率 |
| 交叉熵 | 第 2 章 | 逐 token 的损失 $-\log p_\theta(\cdot)$；见交叉熵概念页 |
| 似然（likelihood） | 第 3 章 | 参数给定时数据出现的概率 |
| 负对数似然（NLL） | 第 3 章 | 似然取负对数，与交叉熵等价 |
| 语料（corpus） | 第 3 章 | 大规模文本集合 |
| 预训练（pretraining） | 第 3 章 | 在语料上以下一 token 交叉熵目标训练 |
| 自监督（self-supervised） | 第 3 章 | 监督信号来自数据自身（下一 token 即标签），无需人工标注 |
| 无标注（unlabeled） | 第 3 章 | 不含人工标注的数据 |
| 预测单元（canonical prediction unit） | 第 3 章 | 平均损失所除以的单位（token/字符/字节） |
| 基座模型（base model） | 第 4 章 | 预训练完成、未经后训练的模型 |
| 续写（continuation） | 第 4 章 | 基座模型按分布生成后续文本的默认行为 |
| 采样（sampling） | 第 4 章 | 按分布随机抽取 token；不展开 |
| 后训练（post-training） | 第 4 章 | 预训练之后的对齐训练统称（SFT 起） |
| SFT（监督微调） | 第 4 章 | 在人类示范上微调预训练模型；见 SFT 概念页 |
| 掩码语言模型（masked LM） | 术语对照 | BERT 式预训练范式；本页不展开 |
| decoder-only / 自回归 | 术语对照 | 只用前文条件、逐 token 生成的架构范式；本页语境 |
| 缩放定律（scaling laws） | 不使用 | —（本页排除） |
