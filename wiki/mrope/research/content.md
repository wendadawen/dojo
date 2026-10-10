# MRoPE

- 描述：MRoPE（多模态旋转位置编码）把旋转位置编码（给每个位置一个编号、按编号旋转向量的位置编码）的一个位置编号拆成时间、高、宽三份，让文本、图像、视频各按自己的排布取这三份编号。
- 摘要：MRoPE 把 RoPE 的一个位置编号拆成时间、高、宽三份，分别驱动向量的一段维度旋转。文本 token 三份相同、退化成 1D-RoPE；图像 token 时间编号固定、高宽按行列取；视频再让时间编号逐帧增加。
- 主题：多模态, 模型结构
- 标签：位置编码

## 资料

- Qwen Team. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191v2, 2024. [https://arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191)
- Qwen Team. Qwen2-VL: official implementation. [https://github.com/QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
- Qwen Team. Qwen2-VL blog. [https://qwenlm.github.io/blog/qwen2-vl/](https://qwenlm.github.io/blog/qwen2-vl/)

## 短页

### 位置编号分成三份

旋转位置编码给每个位置一个编号，按编号把查询和键（[自注意力](wiki/standard-attention)里做点积的两个向量）各转一个角（见[旋转位置编码](wiki/rope)）。模型逐单元处理输入，处理的最小单元叫 token——文本、图像、视频都由 token 组成。多模态指输入里的数据不止文本，还含图像、视频等多种。MRoPE（多模态旋转位置编码）把一个位置编号拆成时间、高、宽三份，分别驱动向量的一段维度去旋转。文本 token 三份取同一个编号，退化成普通的一维编号；图像 token 的时间编号不随时间变化、保持固定，高、宽编号按它所在的行、列取；视频再让时间编号逐帧加一。
依据：paper/src-v2/template.tex L363-L370；blog/qwen2-vl-blog.txt L131

### 一维编号表达不了图和视频

文本是一维的，一个 token 接一个 token，一个编号就够用：靠前的编号小、靠后的编号大。图像 token 排在高×宽两维的网格上，视频按帧展开、每个 token 还落在一帧上。用一维编号，只能把网格按某种顺序拉平排成一条线，编号也就只反映这条线上的先后：行、列这些方向区分不出来。一维位置编码限制了对三维空间（行、列、帧三个方向）和时间动态的建模，就要按行、列、帧这些方向分别给编号。
依据：paper/src-v2/template.tex L290；paper/src-v2/template.tex L366

### 按网格取三份编号再旋转

视觉 token 按它的帧、行、列取时间、高、宽三个编号，转法和旋转位置编码一样；一段输入里多张图或多段视频时，后一段的编号从前一段用过的最大编号加一开始。
依据：paper/src-v2/template.tex L367-L371；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L336-L367；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L453-L455

## 长页

旋转位置编码（RoPE）给序列里每个位置一个编号，按编号把注意力的查询和键各转一个角，使两个位置点积里的位置信息只由它们的相对位置决定（见[旋转位置编码](wiki/rope)；查询和键是注意力里做点积的两个向量，见[自注意力](wiki/standard-attention)）。模型逐单元处理输入，处理的最小单元叫 token，文本、图像、视频都由 token 组成。多模态指输入里的数据不止文本，还含图像、视频等多种。MRoPE（多模态旋转位置编码）把 RoPE 的一个位置编号拆成时间、高、宽三份，让每个 token 按自己的排布取这三份编号来旋转查询和键。
依据：paper/src-v2/template.tex L253；paper/src-v2/template.tex L363-L367

### 为什么一维编号不够

一维编号把 token 按在序列里的先后排成一条直线：位置越靠后，编号越大，任意两个 token 的距离就是两个编号之差。这对文本正合适——文本本身是一维的，前后关系就是一条链。图像和视频不是一条链。图像 token 排在高×宽两维网格上，一个 token 的位置要由它所在的行和列共同确定；视频按帧展开，还要加上它落在第几帧。用一维编号，只能把二维网格、三维视频按某种顺序拉平排成一条线，编号只反映这条线上的先后，分不出 token 是在同一行、同一列还是同一帧。一维位置编码因此限制了对三维空间（行、列、帧三个方向）和时间动态的建模。
依据：paper/src-v2/template.tex L290；paper/src-v2/template.tex L366

### 文本、图像、视频各取一组编号

MRoPE 把位置编号拆成时间、高、宽三份，每份是一个整数，每个 token 拿到一组三个编号。

- 文本 token：三份取同一个编号，就是它在一维序列里的位置，MRoPE 在纯文本上退化成原来那种一维的旋转位置编码（1D-RoPE）。
- 图像：时间编号不随位置变化，每个视觉 token 的时间编号都取同一个值；高编号按它所在的行取，宽编号按它所在的列取。
- 视频：把它看成若干帧的序列，一帧是一张图。帧内 token 的高、宽编号和图像一样取，时间编号逐帧加一。
依据：paper/src-v2/template.tex L367-L370；blog/qwen2-vl-blog.txt L131

::: 视频编号示例（自设）
取一段 2 帧的视频，每帧排成 3 行 2 列，共 $2\times3\times2=12$ 个视觉 token。把 token 按帧、行、列的顺序排成一串，给每个 token 写一组（时间, 高, 宽）三个编号。
依据：自行构造

1. 时间编号：第 0 帧的 6 个 token 都是 0，第 1 帧的 6 个都是 1，得到 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1。
依据：自行构造
2. 高编号：每帧 3 行，行号取 0 到 2，每行 2 个 token，得到 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2。
依据：自行构造
3. 宽编号：每帧 2 列，列号取 0 到 1，逐列交替，得到 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1。
依据：自行构造
4. 三串对齐，第 0 帧第 0 行的两个 token 是 $(0,0,0)$ 和 $(0,0,1)$，高编号相同，靠宽编号区分；第 0 帧第 0 列的三个 token 是 $(0,0,0)$、$(0,1,0)$、$(0,2,0)$，宽编号相同，靠高编号区分。
依据：自行构造
:::

### 三份编号怎么接回旋转

旋转位置编码把向量的分量两两配对，每一对按位置编号转一个角。MRoPE 沿用这个旋转，只把配对用的编号从一份换成三份：把向量的维度分成时间、高、宽三段，时间编号驱动第一段，高编号驱动第二段，宽编号驱动第三段，每段各占多少维是预设的。段内还是旋转位置编码的转法——按这一段的编号算出余弦和正弦，把这一对分量交叉相乘再相加（见[旋转位置编码](wiki/rope)）。

三段各自独立，同一帧、同一行、同一列的编号分别相同，转出的三段分量角也就分别相同。查询和键都这样转，值不转（值也是注意力里的向量，见[自注意力](wiki/standard-attention)），和旋转位置编码一样。
依据：paper/src-v2/template.tex L365-L367；paper/src-v2/template.tex L368-L370；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L336-L367；repo/qwen2-vl/qwen-vl-finetune/qwenvl/train/trainer.py L135-L137

### 一次输入里多段怎么编号

一段输入里可能既有文字又有图像、视频，还可能有不止一张图或不止一段视频。编号按出现的先后一段一段地接：每一段先记住前面已经用过的最大编号，从它加一开始给这一段编号，段与段的编号不重叠。段内视觉 token 的时间、高、宽三份都从 0 起各自算（时间按帧、高按行、宽按列），再统一加上这一段的起始编号作为偏移；三份编号的偏移相同。段的起始编号只随这一段出现的先后增大。
依据：paper/src-v2/template.tex L371；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L158-L164；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L453-L480

### 编号变小，序列变长

编号不再随 token 个数一直增大，同一份编号会被同一行、同一列或同一帧的多个 token 共用。于是同样一段图像或视频，MRoPE 用到的最大编号比一维编号小，模型因此能外推到比训练更长的序列。
依据：paper/src-v2/template.tex L369-L370；paper/src-v2/template.tex L373

::: 与 1D-RoPE 的对比和长度外推
在图像和视频的下游任务上比较两种编码（1D-RoPE 是本概念所替代的一维做法），MRoPE 总体更好。长度外推上，训练时每个视频最多 16384 个 token，推理拉到 80K 个 token 时表现仍稳。
依据：paper/src-v2/template.tex L772-L773；paper/src-v2/template.tex L787-L795
:::

### 适用条件与边界

让这套做法成立的几条：一段图像或视频要先知道它排成几行几列、视频有几帧，才编得出编号；纯文本时 MRoPE 不比 1D-RoPE 多做一步。一个位置从一份编号变成三份，三份要各自维护。
依据：paper/src-v2/template.tex L368；paper/src-v2/template.tex L370；paper/src-v2/template.tex L373；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L322-L326
