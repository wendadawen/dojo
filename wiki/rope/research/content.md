# RoPE

- 描述：RoPE（旋转位置编码）是一种位置编码：把同一序列里每个位置的向量按该位置旋转一个角度，使两个位置点积里的位置信息只由它们的相对位置（两个位置之差）决定。
- 摘要：位置 $p$ 的向量按一组预设角频率 $\theta_i$ 旋转，每对分量转 $p\theta_i$。旋转是正交的，两个向量点积后角度差只剩相对位置 $m-n$；角频率取 $10000^{-2(i-1)/d}$ 时，点积随相对距离增大而衰减。
- 主题：模型结构, 注意力机制
- 标签：位置编码

## 资料

- Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864v5, 2023. [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- ZhuiyiTechnology. RoFormer: official implementation. 2022. [https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)
- Jianlin Su. bert4keras. 2024. [https://github.com/bojone/bert4keras](https://github.com/bojone/bert4keras)

## 短页

### 旋转出来的位置

位置编码是给模型补上「谁在前、谁在后」这层信息的手段。RoPE 是其中一种，作用在注意力的查询和键上（两者是注意力里的向量，见[自注意力](wiki/standard-attention)）。它把向量的分量两两配对，第 $i$ 对按一个预设角频率 $\theta_i$ 旋转——位置 $m$ 的向量转 $m\theta_i$，位置 $n$ 的向量转 $n\theta_i$。两个向量都转完再做点积，各自的角度里都带上自己的位置，相减后只剩 $(m-n)\theta_i$；点积里的位置信息因此只由相对位置 $m-n$ 决定，与它们处在序列的哪一段无关。
依据：paper/src-v5/roformer_arxiv.tex L110；paper/src-v5/roformer_arxiv.tex L262-L288

### 分数要看出先后

注意力分数是查询与键的点积，它对次序不敏感：把两个位置的先后对调，点积不变，所以模型要区分词序，只能依赖位置编码。这样的编码最好让分数里直接带上两个位置的差，并且差得越远、连接越弱。绝对位置编码给每个位置配一个向量、加到该位置的向量上，两个位置的差没有直接进入分数；RoPE 不往向量上加位置，把查询和键各转一个位置角，让差直接出现在点积里。角频率取 $10000^{-2(i-1)/d}$ 时（$d$ 是向量的维数），点积还随相对距离增大而衰减。
依据：paper/src-v5/roformer_arxiv.tex L104；paper/src-v5/roformer_arxiv.tex L199；paper/src-v5/roformer_arxiv.tex L283；paper/src-v5/roformer_arxiv.tex L305

### 按位置转，逐元素算

做法是：把向量的分量两两配对，每对看成一个平面上的点，位置 $m$ 处转 $m\theta_i$ 角，$\theta_i=10000^{-2(i-1)/d}$ 是一组预设角频率（$d$ 是向量的维数），随组号变大而变小。查询和键都这样转，值不转（见[自注意力](wiki/standard-attention)）。实现时不必造出旋转矩阵，按分量算出余弦、正弦，同组的两个分量交叉相乘再相加即可。
依据：paper/src-v5/roformer_arxiv.tex L262-L288；paper/src-v5/roformer_arxiv.tex L435-L478

## 长页

RoPE 用在注意力里。注意力在位置之间传信息：位置 $m$ 的查询与各位置的键做点积得到分数，再对值加权求和，得到该位置的输出（见[自注意力](wiki/standard-attention)）。这一步的分数只由查询与键的向量算出，把同一条序列内位置的先后次序对调，分数不变，所以次序必须靠额外的位置编码告诉模型——位置编码就是给模型补上「谁在前、谁在后」这层信息的手段。
依据：paper/src-v5/roformer_arxiv.tex L104；paper/src-v5/roformer_arxiv.tex L127-L136

位置编码大致分两类：一类是绝对式，给每个位置配一个向量、加到该位置的输入向量上，再送进注意力（设成一组可训练的向量——这些向量是参数、在训练中调整，或用正弦函数直接算出来，都属这类，位置是加进去的）；另一类是相对式，让注意力分数里带上两个位置的差。RoPE 属于相对式：它不往向量上加位置，而是把查询和键各按自己的位置转一个角度，名字里的 Rotary 即指这个旋转。
依据：paper/src-v5/roformer_arxiv.tex L104；paper/src-v5/roformer_arxiv.tex L110；paper/src-v5/roformer_arxiv.tex L146-L160

### 点积里为什么放得下位置

要让位置以相对形式进入，先把它写成一条对点积的要求：位置 $m$ 的查询与位置 $n$ 的键做点积，结果只由这两个向量和它们的位移 $m-n$ 决定，与它们各自的绝对位置无关。
依据：paper/src-v5/roformer_arxiv.tex L206；paper/src-v5/roformer_arxiv.tex L218-L222

把这个点积记成 $g(\mathbf{x}_m, \mathbf{x}_n, m-n)$，其中 $\mathbf{x}_m$、$\mathbf{x}_n$ 是两个位置的输入向量，函数除它们之外只再收一个参数——两位置的差。位置只以 $m-n$ 的形式出现，而两个向量的内容照旧各自进入：换掉 $\mathbf{x}_m$ 或 $\mathbf{x}_n$，点积就跟着变。要做的就是找一对函数 $f_q$、$f_k$，把输入向量和它的位置分别映射成查询和键，使这条要求成立。
依据：paper/src-v5/roformer_arxiv.tex L206；paper/src-v5/roformer_arxiv.tex L218-L222

### 二维：位置就是一个旋转角

每个向量只有两个分量时，把二维向量看成平面上一支从原点出发的箭，用它的长度和与横轴的夹角就能确定这个向量。要求一对函数分别作用在查询和键上，使点积只含 $m-n$；在这一情形里有一类解：函数就是把箭按位置转一个角。位置 $m$ 的查询转 $m\theta$，位置 $n$ 的键转 $n\theta$，$\theta$ 是预设的非零常数。两支箭各自转完，夹角里两个位置角相减，只剩 $(m-n)\theta$，相对位置因而进入了点积。
依据：paper/src-v5/roformer_arxiv.tex L227-L260

::: 二维解的推导与一个例子
（把箭按角旋转时，长度不变、与横轴的夹角加上这个角。）
依据：自行构造

1. 设位置 0 时查询、键分别是 $\mathbf{q}$、$\mathbf{k}$。把两者看成平面上的箭：查询长 $r_q$、与横轴夹角 $\alpha_q$；键长 $r_k$、夹角 $\alpha_k$。
依据：自行构造
2. 让位置 $m$ 的查询在 $\mathbf{q}$ 上再转 $m\theta$，位置 $n$ 的键在 $\mathbf{k}$ 上再转 $n\theta$。两条箭的长度没变，夹角变成 $(\alpha_k+n\theta)-(\alpha_q+m\theta)$。
依据：自行构造
3. 点积等于两长度之积乘夹角余弦，即 $r_qr_k\cos[(\alpha_k-\alpha_q)-(m-n)\theta]$。两个向量的内容（长度与原本夹角）各自进入，位置只剩 $m-n$，要求满足。
依据：自行构造

例子：取查询 $\mathbf{q}=(1,0)$、键 $\mathbf{k}=(1,0)$，$\theta=0.5$ 弧度，$m=2$、$n=1$。
依据：自行构造
1. 查询转 $2\times0.5=1$ 弧度，得到 $(\cos1,\sin1)$，约 $(0.5403,0.8415)$。
依据：自行构造
2. 键转 $1\times0.5=0.5$ 弧度，得到约 $(0.8776,0.4794)$。
依据：自行构造
3. 点积约等于 $0.5403\times0.8776+0.8415\times0.4794\approx0.8776$，正是 $\cos0.5$，只由 $m-n=1$ 与 $\theta=0.5$ 决定，与 $m$、$n$ 各自是多少无关。
依据：自行构造
:::

### 高维：每两个分量一组旋转

把二维的结论推广到 $d$ 维（$d$ 为偶数）：把 $d$ 个分量两两配对，分成 $d/2$ 组二维子空间，每组各自转一个角。第 $i$ 组配一个预设角频率 $\theta_i$，位置 $m$ 的这一组转 $m\theta_i$。全部 $d$ 个分量的旋转写成一个分块对角矩阵，每块是一个二维旋转，共用位置 $m$、各用各的 $\theta_i$。查询和键各自乘上这个矩阵，点积展开后每组的角差都是 $(m-n)\theta_i$，加总只剩相对位置。
依据：paper/src-v5/roformer_arxiv.tex L262-L288

写成公式：$\mathbf{q}_m=\mathbf{R}^d_{\Theta,m}\mathbf{W}_q\mathbf{x}_m$，$\mathbf{k}_n=\mathbf{R}^d_{\Theta,n}\mathbf{W}_k\mathbf{x}_n$。$\mathbf{W}_q$、$\mathbf{W}_k$ 是查询、键的投影矩阵——把输入向量乘上它们就得到查询、键；$\mathbf{R}^d_{\Theta,m}$ 是位置 $m$ 的旋转矩阵，$\Theta=\{\theta_i\}$ 是那组预设角频率。旋转矩阵是正交的，满足 $\mathbf{R}^{d\top}_{\Theta,m}\mathbf{R}^d_{\Theta,n}=\mathbf{R}^d_{\Theta,n-m}$，查询与键的点积于是写成 $\mathbf{x}_m^\top\mathbf{W}_q^\top\mathbf{R}^d_{\Theta,n-m}\mathbf{W}_k\mathbf{x}_n$，位置在里面只以两位置的差出现。
依据：paper/src-v5/roformer_arxiv.tex L266-L289

### 角频率与长程衰减

角频率按 $\theta_i=10000^{-2(i-1)/d}$ 取：$i$ 越大，$\theta_i$ 越小，而且是几何级数地减。于是各组分量转向快慢不同——靠前的组每前进一个位置就转过一个较大的角，靠后的组要前进很多位置才转过一个较小的角。这个取法让点积具备长程衰减的性质。
依据：paper/src-v5/roformer_arxiv.tex L283；paper/src-v5/roformer_arxiv.tex L305

::: 角频率的取值
1. 取 $\theta_i=10000^{-2(i-1)/d}$，$i=1,2,\dots,d/2$。指数里是 $i-1$，所以 $i=1$ 时 $\theta_1=10000^0=1$，是最大的一档。
依据：paper/src-v5/roformer_arxiv.tex L283
2. $i$ 每加 1，指数减 $2/d$，$\theta_i$ 就乘上 $10000^{-2/d}$，是一个小于 1 的固定倍数，所以随 $i$ 几何递减。
依据：paper/src-v5/roformer_arxiv.tex L283
3. 最小的一档在 $i=d/2$，$\theta_{d/2}=10000^{-(d-2)/d}$，随 $d$ 增大趋近 $1/10000$。
依据：自行构造
:::

#### 远距离的相关度会衰减

相对距离变大时，点积有一层随距离减弱的上界。把查询和键按刚才的分组两两配对，每组给点积贡献一项：这一项的大小是这组两个二维向量长度之积，角度随 $(m-n)\theta_i$ 变。可以证明这层上界随相对距离增大而衰减：相对距离远的位置，连接更弱。
依据：paper/src-v5/roformer_arxiv.tex L304-L305；paper/src-v5/roformer_arxiv.tex L487-L504

::: 衰减上界怎么来的
（把每组两个二维分量看成一个复数：分量一取实部、分量二取虚部。按位置转 $m\theta_k$ 就是乘 $e^{im\theta_k}$；两个二维向量的点积等于其中一个复数乘另一个的共轭后取实部。）
依据：自行构造

1. 第 $k$ 组两个二维向量各自转了 $m\theta_k$、$n\theta_k$ 后，对点积的贡献是这一组的复数乘积的实部 $\operatorname{Re}[c_k\,e^{i(m-n)\theta_k}]$，其中 $c_k=\mathbf{q}_{[2k-1:2k]}\mathbf{k}^*_{[2k-1:2k]}$（$\mathbf{q}_{[2k-1:2k]}$ 是查询的第 $2k-1$、$2k$ 两个分量，$k=1,2,\dots,d/2$；星号表示共轭）。
依据：paper/src-v5/roformer_arxiv.tex L487-L489
2. 记 $h_k=c_k$、$S_j=\sum_{k=1}^{j-1}e^{i(m-n)\theta_k}$，并约定 $h_{d/2+1}=0$、$S_1=0$。整串和对 $k$ 做一次分部求和：$\sum_k h_k(S_{k+1}-S_k)=-\sum_k S_{k+1}(h_{k+1}-h_k)$（最右一项 $S_{d/2+1}h_{d/2+1}$ 因 $h_{d/2+1}=0$ 消失）。
依据：paper/src-v5/roformer_arxiv.tex L489-L492
3. 取绝对值放缩，点积的模不超过 $\max_k|h_{k+1}-h_k|$ 乘 $\sum_k|S_{k+1}|$。
依据：paper/src-v5/roformer_arxiv.tex L497-L503
4. $S_{k+1}=\sum_{j=1}^{k}e^{i(m-n)\theta_j}$ 是一串单位复数的和，每一项的相位是 $(m-n)\theta_j$。$\theta_j$ 互不相同、随 $j$ 从 1 递减到接近 $10^{-4}$，$m-n$ 越大，这些相位铺开得越广、越接近均匀分布在圆周上，方向和相抵，部分和 $S_{k+1}$ 的模就越小。
依据：paper/src-v5/roformer_arxiv.tex L504
:::

### 用逐元素乘加算

用 $d\times d$ 的旋转矩阵乘向量，开销大，因为矩阵是稀疏的，绝大多数位置是零。有等价的逐元素写法：把每个分量和它相邻的一个分量配成一对，每对 $(x_{2i-1}, x_{2i})$ 做一次二维旋转——分量分别乘这组的 $\cos m\theta_i$，交换位置、变号后乘 $\sin m\theta_i$，再相加。这样只用相邻分量的乘加，不必构造旋转矩阵，也就避开了稀疏矩阵乘的开销。
依据：paper/src-v5/roformer_arxiv.tex L289；paper/src-v5/roformer_arxiv.tex L435-L478

::: $d=4$、$m=1$ 的例子
取 $d=4$，这组角频率按公式是 $\theta_1=10000^0=1$、$\theta_2=10000^{-1/2}=0.01$；位置取 $m=1$，输入向量取 $(1,0,1,0)$。
依据：自行构造

1. 第一组 $(1,0)$，$\theta_1=1$，转角 $1\times1=1$ 弧度：$1\times\cos1-0\times\sin1=\cos1$，$1\times\sin1+0\times\cos1=\sin1$，得约 $(0.5403,0.8415)$。
依据：自行构造
2. 第二组 $(1,0)$，$\theta_2=0.01$，转角 $1\times0.01=0.01$ 弧度：得约 $(1.0000,0.0100)$。
依据：自行构造
3. 首尾相接，位置 1 的向量转成 $(0.5403,0.8415,1.0000,0.0100)$。
依据：自行构造

```python
import math

def rope(x, m, thetas):
    out = []
    for i, th in enumerate(thetas):
        c, s = math.cos(m * th), math.sin(m * th)
        a, b = x[2 * i], x[2 * i + 1]
        out.append(round(a * c - b * s, 4))
        out.append(round(a * s + b * c, 4))
    return out

thetas = [1.0, 0.01]
print(rope([1.0, 0.0, 1.0, 0.0], 1, thetas))
```

```text
[0.5403, 0.8415, 1.0, 0.01]
```
:::

### 加在自注意力的哪一步

RoPE 加在算分数的前一步：查询和键各自做完投影、还没和对方做点积时，先按自己的位置旋转；值不旋转，后面的加权求和也不动。位置信息就这样只经由查询和键的点积进入分数。它不引入新的可训练参数：$\theta_i$ 是预设的常数，绝对位置编码里那批可训练的位置向量在这里不存在。
依据：paper/src-v5/roformer_arxiv.tex L152；paper/src-v5/roformer_arxiv.tex L206；paper/src-v5/roformer_arxiv.tex L283；paper/src-v5/roformer_arxiv.tex L286-L289；repo/bert4keras/bert4keras/layers.py L519-L520

### 适用条件与边界

让这套做法成立的几条：$d$ 取偶数，分量才配得成对；位置按绝对位置旋转、点积里出现的是 $m-n$，$\theta_i$ 是固定常数，位置只以两位置的差进入分数、与序列总长无关，所以同一套旋转在比训练时更长的序列上也用得下去（这叫作序列长度灵活）。
依据：paper/src-v5/roformer_arxiv.tex L93；paper/src-v5/roformer_arxiv.tex L263；paper/src-v5/roformer_arxiv.tex L267-L289

代价在计算：查询和键各要按自己的位置做一遍旋转，用逐元素乘加来算。旋转是正交的、只改方向不改向量长度，而且只做乘法，所以它也能装进把注意力写成先各自变换、再相乘的形式（线性注意力）里。
依据：paper/src-v5/roformer_arxiv.tex L289；paper/src-v5/roformer_arxiv.tex L307-L327

最容易和它混淆的是绝对式位置编码。绝对式把位置向量加到查询、键的表示上，是加法；RoPE 不往表示上加位置向量，而是把查询和键乘以旋转，是乘法。
依据：paper/src-v5/roformer_arxiv.tex L292
