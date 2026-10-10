# 投机解码

- 描述：投机解码用一个更高效的近似模型先猜出接下来若干候选，再由目标模型一次并行验证这些候选，接受其中能保证输出分布不变的部分，从而用更少的串行次数生成与目标模型同分布的结果；它不改模型架构、不需要重新训练，可直接用在现成模型上。
- 摘要：自回归解码每生成一个 token 都要跑一次大模型。先用一个更高效的近似模型猜 $\gamma$ 个候选，再让目标模型一次并行验证并接受，接受率越高、一次串行产出的 token 越多，而输出分布不变。
- 主题：推理系统, 模型结构
- 标签：推理加速

## 资料

- Yaniv Leviathan, Matan Kalman, Yossi Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023, arXiv:2211.17192v2, 2023. [https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)

## 短页

### 用一个近似模型猜、目标模型一次验

自回归生成指用已经生成的内容去预测下一个 token，每一步只生成一个 token，token 是文本被切分后的最小单位。投机解码在一轮里先用一个更高效的近似模型猜出接下来 $\gamma$ 个 token，$\gamma$ 是每轮要猜的候选个数；再把前缀和这些候选一起交给目标模型做一次前向（把当前输入跑一遍模型），为这几个位置各算出一个分布，据此逐个接受。被接受的部分就是能保证分布不变的那些候选，被拒的位置由目标模型补一个。一轮至少产出 1 个 token，最多 $\gamma+1$ 个，结果与只跑目标模型同分布。
依据：paper/src-v2/main.tex L143-L151；paper/src-v2/main.tex L193-L196

### 自回归一次只生成一个 token，串行前向慢

逐个生成 $K$ 个 token 要 $K$ 次串行前向：每步都要等上一步算完，而大模型的单次前向又慢。推理（用训练好的模型算出输出）通常受限于访存与通信，而非算术运算，算力往往没有用满，硬件上有闲置算力。投机解码用这份闲置算力做并行验证，把串行次数降下来。
依据：paper/src-v2/main.tex L163-L175

### 猜、验、接受、补一个

近似模型自回归地猜 $\gamma$ 个候选，同时留下每一步的分布 $q_i$；目标模型对前缀、前缀加第 1 个候选、……、前缀加前 $\gamma$ 个候选，共 $\gamma+1$ 段输入一次并行前向，得到一组分布 $p_1,\dots,p_{\gamma+1}$。对第 $i$ 个候选 $x$，以概率 $\min(1,p_i(x)/q_i(x))$ 接受，直到第一个被拒或全部接受。若 $\gamma$ 个候选全被接受，从目标模型最后一个位置的分布 $p_{\gamma+1}$ 采一个 token；若某一个被拒，则从修正分布补一个 token，修正分布是把目标分布减去近似分布、截掉负值再归一化的结果。
依据：paper/src-v2/main.tex L195-L196；paper/src-v2/main.tex L202-L204；paper/src-v2/main.tex L216-L234

## 长页

自回归模型一次生成一个 token，token 是文本被切分后的最小单位：它每生成一个 token，就接在已有序列后面，再据此生成下一个。把一段输入跑一遍模型、得到下一个 token 的分布，叫一次前向；这样逐个生成 $K$ 个 token，要 $K$ 次串行前向，每次都得等上一次算完。大模型的单次前向比小模型慢，生成整段文本的耗时随 token 数增长。
依据：paper/src-v2/main.tex L143-L151；paper/src-v2/main.tex L163

推理（用训练好的模型算出输出）的瓶颈通常不在算术运算，而在访存与通信，算力往往没有用满，硬件上因此常有闲置的算力。
依据：paper/src-v2/main.tex L173-L175

投机解码用这份闲置算力换更少的串行次数：先用一个更高效的近似模型猜出接下来若干 token，再由目标模型一次并行验证这些猜测，只接受能保证输出分布不变的部分。一轮里目标模型只跑一次，却可能产出多个 token。
依据：paper/src-v2/main.tex L193-L196

### 一步投机解码：猜、验、纠

#### 猜：近似模型自回归给出 $\gamma$ 个候选

模型对下一个 token 给出的是词表上的一整个概率分布，词表是模型可能生成的全部 token；按一个分布的概率抽一个 token 叫采样。取概率最大的那个（argmax）、按温度（调整分布陡峭程度的参数）调整分布后再采样等做法，都等价于对某个调整后的分布做标准采样。目标模型分布 $p$ 与近似模型分布 $q$ 就指按所选采样方式调整后的分布。
依据：paper/src-v2/main.tex L198-L200

近似模型 $M_q$ 是一个更高效的模型，对同一任务给出它自己的分布 $q$。以当前前缀起步，它自回归地重复采样 $\gamma$ 次：第 $i$ 步在「前缀加此前 $i-1$ 个候选」上采出第 $i$ 个候选 $x_i$，同时留下这一步的分布 $q_i$。$\gamma$ 是每轮要猜的候选个数，是一个正整数。
依据：paper/src-v2/main.tex L195-L196；paper/src-v2/main.tex L216-L220

#### 验：目标模型一次并行打分

把前缀本身、以及「前缀加前 $i$ 个候选」这 $\gamma+1$ 段输入一起交给目标模型 $M_p$，一次前向算出 $\gamma+1$ 个分布 $p_1,\dots,p_{\gamma+1}$。$p_i$ 是目标模型在第 $i$ 个位置上的分布。
依据：paper/src-v2/main.tex L206；paper/src-v2/main.tex L221-L223

这 $\gamma+1$ 段能一次并行算完，是因为它们的输入在算之前就都已知：候选由近似模型先给出，每段输入不依赖别的段算出的输出，于是可以合成一批一起前向。这样做的前提是并发执行这些前向不增加墙钟时间（实际经过的时间）。
依据：paper/src-v2/main.tex L221-L223；paper/src-v2/main.tex L320

#### 判与纠：接受判据与修正分布

只凭目标模型分布 $p$ 采样一个 $x$，可以改成：先从 $q$ 采样一个 $x$，若 $q(x)\le p(x)$ 就保留；若 $q(x)>p(x)$，以概率 $1-\frac{p(x)}{q(x)}$ 拒绝它，改从修正分布 $p'(x)=norm(\max(0,p(x)-q(x)))$ 重采一个。$norm$ 指把一串非负数归一化成概率分布，即每项除以它们的和；$p'$ 就是 $p$ 减去 $q$、截掉负值再归一化的结果。按这套规则抽出的 $x$，分布恰好是 $p$。
依据：paper/src-v2/main.tex L202-L204

推广到一轮出多个 token：对 $i=1,\dots,\gamma$ 各取一个均匀随机数 $r_i$，$x_i$ 在 $r_i\le\frac{p_i(x_i)}{q_i(x_i)}$ 时接受，否则拒绝。接受数 $n$ 是被连续接受、直到第一个被拒为止的候选个数；全部接受时 $n=\gamma$。
依据：paper/src-v2/main.tex L224-L227

接着补出最后一个 token：若 $n=\gamma$，$\gamma$ 个候选全被接受，从目标分布 $p_{\gamma+1}$ 采一个 $t$；若 $n<\gamma$，第 $n+1$ 个候选被拒、被丢弃，改从修正分布 $norm(\max(0,p_{n+1}-q_{n+1}))$ 采一个 $t$。返回前缀、$n$ 个被接受的候选与补出的 $t$。被拒处的这次修正保证每轮至少产出一个 token，与逐个生成时一样。
依据：paper/src-v2/main.tex L228-L234；paper/src-v2/main.tex L196

#### 一个手算例子

::: 一轮 $\gamma=1$ 的手算过程
词表只有 A、B、C 三个 token。
依据：自行构造

1. 近似模型的分布 $q=(\mathrm{A}\,0.2,\mathrm{B}\,0.5,\mathrm{C}\,0.3)$，采样得候选 $x_1=\mathrm{B}$。
依据：自行构造
2. 目标模型的分布 $p=(\mathrm{A}\,0.5,\mathrm{B}\,0.3,\mathrm{C}\,0.2)$，在 $\mathrm{B}$ 处 $p_1(\mathrm{B})/q_1(\mathrm{B})=0.3/0.5=0.6$。
依据：自行构造
3. 取随机数 $r_1=0.7>0.6$，拒绝 $\mathrm{B}$，接受数 $n=0$。
依据：自行构造
4. 修正分布：$p-q=(0.3,-0.2,-0.1)$，取 $\max(0,\cdot)$ 得 $(0.3,0,0)$，归一化得 $p'=(\mathrm{A}\,1,\mathrm{B}\,0,\mathrm{C}\,0)$。
依据：自行构造
5. 从 $p'$ 补出 token，只能是 A。这一轮产出 $1$ 个 token，就是 A。
依据：自行构造
:::

#### 为什么输出分布不变

要证的是：按上面的规则抽出的 token，分布与直接对 $p$ 采样相同。做法是把「抽到某个 token」拆成「猜被接受」和「猜被拒后改采」两种情形，分别算概率再相加。
依据：paper/src-v2/main.tex L649-L657

::: 逐项证明
1. 修正分布是归一化的：$\sum_x\bigl(p(x)-\min(p(x),q(x))\bigr)=1-\beta$，$\beta$ 是接受概率，于是 $p'(x)=\dfrac{p(x)-\min(q(x),p(x))}{1-\beta}$。
依据：paper/src-v2/main.tex L651
2. 抽到 $x'$ 的概率拆成两项：$P(x=x')=P(\text{猜被接受},x=x')+P(\text{猜被拒},x=x')$。
依据：paper/src-v2/main.tex L655-L657
3. 第一项：$q(x')\min\!\bigl(1,\frac{p(x')}{q(x')}\bigr)=\min(q(x'),p(x'))$。
依据：paper/src-v2/main.tex L661-L663
4. 第二项：$(1-\beta)p'(x')=p(x')-\min(q(x'),p(x'))$。
依据：paper/src-v2/main.tex L667-L669
5. 两项相加：$\min(p(x'),q(x'))+p(x')-\min(p(x'),q(x'))=p(x')$。
依据：paper/src-v2/main.tex L671-L675
6. 于是输出分布与只跑目标模型相同。这一步对任意 $p$、$q$ 都成立，与近似模型的选取无关。
依据：paper/src-v2/main.tex L649；paper/src-v2/main.tex L444
:::

### 接受率与加速比

#### 接受率 $\alpha$ 衡量近似模型有多准

单步接受概率 $\beta$ 指：给定前缀，按投机解码的接受判据接受从 $q$ 采出的那个候选的概率。$\alpha$ 是 $\beta$ 的期望，衡量近似模型与目标模型的接近程度：$\alpha$ 越大，候选越常被接受。$\alpha$ 是模型与任务本身的性质，与硬件和软件实现无关。
依据：paper/src-v2/main.tex L244-L249；paper/src-v2/main.tex L328

#### $\alpha$ 与两个分布的接近程度

定义一个分布对之间的距离 $D(p,q)=\sum_x|p(x)-M(x)|$，其中 $M(x)=\frac{p(x)+q(x)}{2}$ 是 $p$ 与 $q$ 的平均分布。可以推出 $D(p,q)=1-\sum_x\min(p(x),q(x))$。
依据：paper/src-v2/main.tex L270-L276

由此得到 $\beta=1-D(p,q)$，进而 $\alpha=E(\min(p,q))$。$p=q$ 时每一项 $\min$ 等于该处概率、求和为 1，$\alpha=1$，候选全被接受；两分布支撑集（概率非零的 token 集合）不相交时 $\alpha=0$。
依据：paper/src-v2/main.tex L287-L289；paper/src-v2/main.tex L294-L295；paper/src-v2/main.tex L310

::: $D$ 到 $\alpha$ 的推导
1. $D(p,q)=\sum_x|p(x)-M(x)|$，代入 $M=\frac{p+q}{2}$，得 $\sum_x\frac{|p-q|}{2}$。
依据：paper/src-v2/main.tex L280
2. 再写成 $1-\sum_x\frac{p+q-|p-q|}{2}$，而 $\frac{p+q-|p-q|}{2}=\min(p,q)$，故 $D=1-\sum_x\min(p,q)$。
依据：paper/src-v2/main.tex L280
3. 接受概率 $\beta=E_{x\sim q}\min\!\bigl(1,\frac{p(x)}{q(x)}\bigr)=\sum_x\min(p(x),q(x))$，于是 $\beta=1-D$。
依据：paper/src-v2/main.tex L297-L302
4. 对前缀取期望：$\alpha=E(\beta)=1-E(D)=E(\min(p,q))$。
依据：paper/src-v2/main.tex L310
:::

#### 一次串行产出几个 token

在假设各步的 $\beta$ 相互独立、同分布时，一轮的产出是一个上限为 $\gamma+1$、停止概率为 $1-\alpha$ 的截断几何变量，期望产出
$$
E(\#\ \text{产出 token})=\frac{1-\alpha^{\gamma+1}}{1-\alpha}.
$$
$\alpha$ 越大这个数越接近 $\gamma+1$；$\alpha=0$ 时取 1。
依据：paper/src-v2/main.tex L249-L254

::: 期望产出的推导
1. 把一轮拆成 $\gamma+1$ 次「是否继续产出」的判断：前 $\gamma$ 次对应 $\gamma$ 个候选，最后一次是补出的那个。
依据：自行构造
2. 每次接受以概率 $\alpha$ 发生，且各步独立；产出 token 数等于接受的次数加 1，直到首次被拒为止，上限 $\gamma+1$。
依据：paper/src-v2/main.tex L249-L254
3. 这个期望等于 $\sum_{k=0}^{\gamma}\alpha^k$：产出数至少为 1，再乘以每多接受一个候选的 $\alpha$ 累加，套几何级数求和公式得 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$。
依据：paper/src-v2/main.tex L251-L254
:::

#### 墙钟加速与最优 $\gamma$

定义代价系数 $c$ 为近似模型单次前向耗时与目标模型单次前向耗时之比。记目标模型单次代价为 $T$，一轮要跑 $\gamma$ 次近似模型和一次目标模型，代价为 $Tc\gamma+T$，产出 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 个 token。用它和逐个生成（每个 token 代价 $T$）相比，墙钟加速的期望为
$$
\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}.
$$
依据：paper/src-v2/main.tex L323-L337

只要 $\alpha>c$，就存在让加速大于 1 的 $\gamma$，此时加速至少为 $\frac{1+\alpha}{1+c}$，$\gamma=1$ 就取到这个下界。这个加速假设生成足够长，否则会被每轮产出的 token 数封顶。
依据：paper/src-v2/main.tex L341-L349

给定 $c$ 与 $\alpha$，最优 $\gamma$ 是使加速公式最大的那个整数，可在较小的范围里数值求出。
依据：paper/src-v2/main.tex L387-L389

::: 不同 $\alpha$、$\gamma$ 下的运算量与加速
下表假设近似模型自身的耗时与运算量都可忽略。
依据：paper/src-v2/main.tex L395-L396

| $\alpha$ | $\gamma$ | 总算术运算量增幅 | 加速 |
|---|---|---|---|
| 0.6 | 2 | 1.53X | 1.96X |
| 0.8 | 5 | 1.63X | 3.69X |
| 0.9 | 10 | 1.60X | 6.86X |
依据：paper/src-v2/main.tex L402-L411
:::

#### 多花算力、少读访存

一轮里目标模型对 $\gamma+1$ 个位置并行计算，并发算术运算量是原来的 $\gamma+1$ 倍。总算术运算量则可能上升：候选被接受时这份并发不额外增加总运算量，被拒时那次计算就浪费了。总运算量的期望增幅为 $\frac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}$，$\hat c$ 是近似模型与目标模型每 token 算术运算量之比。
依据：paper/src-v2/main.tex L354-L365

::: 总运算量增幅的推导
1. 记标准解码生成一个 token 的运算量为 $\hat T$。一轮里近似模型跑 $\gamma$ 次，每次 $\hat c\hat T$；目标模型对 $\gamma+1$ 个位置并行算一次，运算量 $\hat T(\gamma+1)$。一轮合计 $\hat T\hat c\gamma+\hat T(\gamma+1)$。
依据：paper/src-v2/main.tex L369
2. 一轮平均产出 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 个 token，用它去除一轮的运算量，得每 token 的运算量。
依据：paper/src-v2/main.tex L369
3. 再除以标准解码的 $\hat T$，得增幅 $\frac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}$。
依据：paper/src-v2/main.tex L365
:::

与运算量相反，读内存的次数会下降：目标模型的权重（模型参数）与 [KV cache](wiki/kv-cache) 一轮只读一次，读它们的访存次数缩为原来的 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 分之一。
依据：paper/src-v2/main.tex L373-L375

当近似模型的代价可忽略（$c\approx0$）时，加速上界是 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$，它不超过 $\frac{1}{1-\alpha}$，$\gamma$ 越大越接近这个上界。
依据：paper/src-v2/main.tex L449

### 近似模型怎么选

投机解码对近似模型 $M_q$ 没有任何限制：$M_q$ 换成什么都不影响输出分布，它只影响速度。
依据：paper/src-v2/main.tex L444

选与目标模型同一种架构、比目标小约两个数量级的模型，通常最好，因为它在接受率 $\alpha$ 与代价 $c$ 之间取得平衡：$\alpha$ 随近似模型变大而上升，但 $c$ 随之上升。
依据：paper/src-v2/main.tex L445-L447；paper/src-v2/main.tex L475

代价可忽略的近似模型（$c\approx0$）更便于部署。一种是无参数的 n-gram 模型：按前几个 token 的统计直接给出下一个 token 的候选，取用只相当于查表。
依据：paper/src-v2/main.tex L449-L450；paper/src-v2/main.tex L452-L454

::: $\alpha$ 与加速的经验数字
$M_p$ 取 11B 的模型（T5-XXL），任务为英文到德语翻译，温度取 0（argmax）。
依据：paper/src-v2/main.tex L470；paper/src-v2/main.tex L484

| $M_q$ | $\gamma$ | $\alpha$ | 加速 |
|---|---|---|---|
| T5-small（77M） | 7 | 0.75 | 3.4X |
| T5-base（250M） | 7 | 0.8 | 2.8X |
| T5-large（800M） | 7 | 0.82 | 1.7X |
依据：paper/src-v2/main.tex L486-L490

把 $M_q$ 换成 2-gram（只看前一个 token 的统计）模型，$\alpha$ 约为 0.2，取 $\gamma=3$ 得到 1.25 倍加速。
依据：paper/src-v2/main.tex L450
:::

### 适用条件与边界

投机解码的前提是有额外算力：它靠并发换延迟，并发需要硬件上还有余量。算力本来就紧张时，无法用并发换延迟下降，这个方法不带来收益。
依据：paper/src-v2/main.tex L606；paper/src-v2/main.tex L608

它不改模型架构、不需要重新训练，可以直接用在现成的模型上。
依据：paper/src-v2/main.tex L187；paper/src-v2/main.tex L190

一个容易混淆的相近做法是拒绝采样：它也先从 $q$ 采样，但要按 $\max_x\frac{p(x)}{q(x)}$ 定一个常数上界，接受概率可能低得多；投机解码的接受概率是 $\alpha=E(\min(p,q))$，更高。
依据：paper/src-v2/main.tex L681-L689

它用更多算术运算换更少的串行次数与更少的读访存，省的是延迟，不是总运算量。
依据：paper/src-v2/main.tex L373-L375；paper/src-v2/main.tex L608

::: $\alpha$ 的经验范围
在多种目标模型、近似模型与采样设置上，比目标小约两个数量级的近似模型通常给出 0.5 到 0.9 的 $\alpha$。
依据：paper/src-v2/main.tex L522
:::
