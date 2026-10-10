# MRoPE

- 描述：MRoPE（多模态旋转位置编码）把旋转位置编码的一个位置编号拆成时间、高、宽三份，让文本、图像、视频的 token 各按自己的排布取这三份编号来旋转查询和键。
- 摘要：MRoPE 把 RoPE 的一个位置编号拆成时间、高、宽三份，分别驱动向量的一段维度旋转。文本 token 三份相同、退化成 1D-RoPE；图像 token 时间编号固定、高宽按行列出；视频再让时间编号逐帧增加。
- 主题：多模态, 模型结构
- 标签：位置编码

## 长页

- 开头｜前置链接 wiki/rope；token 是模型逐单元处理的文本、图像、视频单元，每个 token 一个位置编号；MRoPE 把 RoPE 的编号拆成三份｜300｜paper/src-v2/template.tex L253；paper/src-v2/template.tex L363-L367
- ### 为什么一维编号不够｜文本 token 自身带一维先后，一个编号正好；图像 token 排成高×宽两维，视频按帧展开多出时间维，一维编号表达不了这种二三维排布｜320｜paper/src-v2/template.tex L290；paper/src-v2/template.tex L366
- ### 文本、图像、视频各取一组编号｜三份编号怎么取：文本取同一编号，图像时间固定、高宽按行列，视频时间逐帧加一；视频编号示例折叠｜420｜paper/src-v2/template.tex L367-L370；blog/qwen2-vl-blog.txt L131
- ### 三份编号怎么接回旋转｜三份编号各驱动向量一段维度，每段按对应编号旋转；为什么要分三段、每段转哪个编号｜360｜paper/src-v2/template.tex L365-L367；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L336-L367
- ### 一次输入里多段怎么编号｜一段输入含多张图或多段视频时，后一段的编号从前一段的最大编号加一开始，各段顺序相接、编号不重叠，文本接着视觉之后｜280｜paper/src-v2/template.tex L371；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L161-L163；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L453-L455
- ### 编号变小，序列变长｜同一帧同一行列的 token 编号重复，最大编号比逐位置递增小得多，超出训练长度时更不易取到没见过的值、可外推更长序列；消融与外推折叠｜340｜paper/src-v2/template.tex L373；paper/src-v2/template.tex L768-L795
- ### 适用条件与边界｜纯文本时退化成 1D-RoPE；要靠视觉 token 的网格形状才编得出号；一个位置从一份编号变三份｜340｜paper/src-v2/template.tex L368；paper/src-v2/template.tex L369；paper/src-v2/template.tex L373；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L322-L326

## 短页

- ### 位置编号分成三份｜是什么：token 是文本、图像、视频的单元；RoPE 的一个编号拆成时间、高、宽三份驱动向量三段，文本三份相同、图像视频各不同；前置链接 wiki/rope｜300｜paper/src-v2/template.tex L363-L370；blog/qwen2-vl-blog.txt L131
- ### 一维编号表达不了图和视频｜为什么需要：文本有一维先后，图像排在高宽两维、视频再加一个时间方向，一维编号压不下这种排布｜300｜paper/src-v2/template.tex L290；paper/src-v2/template.tex L366
- ### 按网格取三份编号再旋转｜怎么做：视觉 token 按帧、行、列取三个编号，分别转向量三段；文本 token 三份取同一个编号｜300｜paper/src-v2/template.tex L367-L371；repo/qwen2-vl/qwen-vl-finetune/qwenvl/data/rope2d.py L336-L367
