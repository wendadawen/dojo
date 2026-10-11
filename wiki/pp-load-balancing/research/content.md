# PP Load Balancing

- 描述：大语言模型（LLM）按已有内容逐个生成文本时用流水线并行（PP）部署，把流水线每一级（stage）每一步的执行时间拉齐、压小设备空等（气泡）的 token（词元）切分与 batch（一次处理的一组请求）配平做法。
- 摘要：流水线的节拍由最慢的 stage 决定。沿 token 维度把一条长请求切成片段送进流水线，或者在 batch 维度把并发请求在各 micro-batch 间的计算量配平，都能让各 stage 等长，把气泡压下去。
- 主题：并行与通信, 推理系统
- 标签：并行与通信

## 资料

- Zhuohan Li, Siyuan Zhuang, Shiyuan Guo, Danyang Zhuo, Hao Zhang, Dawn Song, Ion Stoica. TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models. ICML 2021. [https://arxiv.org/abs/2102.07988v2](https://arxiv.org/abs/2102.07988v2)
- Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. arXiv:2407.00079. [https://arxiv.org/abs/2407.00079](https://arxiv.org/abs/2407.00079)
- Tianyu Guo, Xianwei Zhang, Jiangsu Du, Zhiguang Chen, Nong Xiao, Yutong Lu. gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling. arXiv:2504.14775. [https://arxiv.org/abs/2504.14775v2](https://arxiv.org/abs/2504.14775v2)
- Shangming Cai. Pipeline Parallelism in SGLang: Scaling to Million-Token Contexts and Beyond. LMSYS Org Blog, 2026. [https://lmsys.org/blog/2026-01-15-chunked-pipeline/](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)

## 短页

### 把各 stage 每步执行时间拉齐

PP（pipeline parallelism，流水线并行）把模型的层按顺序切成若干段，每段放在一组设备上，这一组叫一个 stage，前一个 stage 算完把中间结果传给下一个。一条请求的输入先被切成 token（词元，模型处理文本的最小单位）串，逐段流过流水线。流水线里各段设备等输入、等下游的空档叫气泡。PP 负载均衡就是让各 stage 每一步的执行时间保持相近、把气泡压小的切分与调度做法。它不改变模型本身的计算，只改请求怎么切、每一步各排多少。
依据：paper/src-2407.00079/data/prefill.tex L37-L40；paper/src-2504.14775v2/chapters/2.background_motivation.tex L81-L85；paper/src-2504.14775v2/chapters/4.design.tex L18-L22

### 节拍由最慢的 stage 决定

一次前向（模型从输入算到输出的一遍计算）的总时间由最慢的 stage 决定，只要一个 stage 比别人久，其余 stage 就在等它，气泡由此出现。推理把这种空等放大两次：单条很长的请求整体进入流水线时，输入到 128K 甚至 1M token，第一个 stage 要算很久，后面的 stage 长时间空等；注意力的计算量还随位置变化，位置越靠后要与越多的历史 token 交互（见 [标准注意力](wiki/standard-attention)），同样大小的片段越往后越慢，各 stage 的步长（每一步的执行时间）越拉越开。并发请求多时，每一步里各 micro-batch（微批次，一次迭代中并行推进的最小批次部分）的计算量又会有波动。
依据：paper/src-2504.14775v2/chapters/2.background_motivation.tex L106-L113；paper/src-2102.07988v2/terapipe.tex L305-L305；blog/sglang-pp-blog.txt L111-L117；blog/sglang-pp-blog.txt L139-L141

### 按执行时间对齐，不是按条数

均衡的对象是各 stage 的执行时间，不是 token 的条数。token 维度：把一条请求的输入序列切成若干片段分给不同 stage，切点前长后短，靠后的片段装更少 token，以抵消它更重的注意力。chunk（切成一段的 token 块，见 [chunked prefill](wiki/chunked-prefill)）若取固定大小，会随前缀（已处理的历史 token）变慢，各 stage 的步长越差越多；改用随前缀缩小的 chunk，就能让各 stage 的节拍对齐。batch 维度：把并发的 decode token（生成阶段每一步新产出的 token）按 token 数均摊到各 micro-batch，让每一步的计算量接近。
依据：paper/src-2102.07988v2/terapipe.tex L305-L305；blog/sglang-pp-blog.txt L151-L160；paper/src-2504.14775v2/chapters/4.design.tex L90-L96

## 长页

推理服务常把大模型用 PP（pipeline parallelism，[流水线并行](wiki/pipeline-parallelism)）部署：模型按层切成若干 stage 分到不同设备，一条请求的输入先被切成 token（词元，模型处理文本的最小单位）串，逐段流过流水线。这样只在相邻 stage 之间传中间结果，通信量小。但只要各 stage 的步长不整齐，就有设备在空等，这些空等时段叫气泡，长上下文和并发都会把气泡放大。此时决定速度的是最慢的那个 stage：一次前向（模型从输入算到输出的一遍计算）的节拍由它定，那就设法让各 stage 每一步的执行时间相近。
依据：blog/sglang-pp-blog.txt L111-L117；paper/src-2504.14775v2/chapters/2.background_motivation.tex L81-L85；paper/src-2504.14775v2/chapters/2.background_motivation.tex L43-L52；paper/src-2102.07988v2/terapipe.tex L305-L305

一条沿 token 维度切：把同一条请求的输入序列切成片段，让不同片段同时出现在不同 stage。另一条沿 batch 维度配平：把并发请求在这些 micro-batch（一次迭代里并行推进的最小批次部分）间摊匀。
依据：paper/src-2504.14775v2/chapters/2.background_motivation.tex L81-L85；paper/src-2504.14775v2/chapters/4.design.tex L90-L96；paper/src-2407.00079/data/prefill.tex L37-L43

### GPU 为什么空转

一类等待是 inter-stage 依赖：一个 stage 必须等前一个 stage 把这一步算完才能开始，依赖链上越靠后的 stage 起步越晚，流水线填充与排空期因此有空档。另一类是 inter-batch 依赖：一次能有多少个 micro-batch 同时在流水线里推进，受流水线深度（流水线的级数，也是能同时在流水线里推进的 micro-batch 数上限）限制，填不满的部分也是气泡。负载失衡相应分两类：inter-stage 失衡是各 stage 之间的计算量分布不均，inter-batch 失衡是不同 micro-batch 之间的计算量不同。
依据：paper/src-2504.14775v2/chapters/2.background_motivation.tex L81-L85

空等在推理里被两处放大。把 128K 到 1M token 的输入当作一个整体批次送进流水线，下游 stage 会长时间空等，同一时刻一次前向还要为整条序列保存和传递中间结果，峰值显存随之抬高；位置越靠后的 token 又要与越多的历史交互（见 [标准注意力](wiki/standard-attention)），同样大小的片段越往后处理越慢。两处都让各 stage 的步长越拉越开。
依据：blog/sglang-pp-blog.txt L111-L117；blog/sglang-pp-blog.txt L139-L141

设流水线有 $K$ 个 stage，一次送入 $M$ 个 micro-batch，气泡比率是 $\frac{K-1}{K-1+M}$。式子里的 $K-1$ 是流水线填不满时多出的步数，$M$ 越大，这部分固定开销摊得越薄；各步等长时按它估计就够了。步长不齐时，实测气泡比率会高于这个理论值。
依据：blog/sglang-pp-blog.txt L76-L88；blog/sglang-pp-blog.txt L149-L149；paper/src-2102.07988v2/terapipe.tex L305-L305

::: 气泡比率的来源
1. 把流水线看成 $K$ 级排成一列，每一步每一级处理一个 micro-batch，并把结果交给下一级。
依据：自行构造
2. 第一个 micro-batch 进入后要走完 $K$ 级才离场，它占用的时段是第 1 步到第 $K$ 步。
依据：自行构造
3. $M$ 个 micro-batch 依次进入，最后一个在第 $M$ 步进入、第 $M+K-1$ 步离场，所以整条流水线从第一个进入到最后一个离场一共 $M+K-1$ 步。
依据：自行构造
4. 每一级实际干活的步数是 $M$，比它多出来的 $K-1$ 步是没填满时的空转。
依据：自行构造
5. 气泡比率等于空转步数除以总步数。
依据：自行构造
:::

### 把一条请求切进流水线（token 维度）

一条请求能不能拆进流水线，取决于模型是不是自回归（按顺序逐个生成，每一步的输入只来自已生成的部分）。decoder-only 模型（只有解码器、逐 token 生成下一 token 的语言模型）用因果掩码，位置 $t$ 的计算只用到位置 $\le t$ 的 token，不看未来（见 [因果掩码](wiki/causal-mask)）。因此同一条序列的不同片段之间只有单向依赖：第 4 个片段在某个 stage 上计算时，第 3 个片段可以同时在下一个 stage 上计算，不必等整条序列先在一个 stage 上算完。这就是序列内部的流水线。
依据：paper/src-2102.07988v2/terapipe.tex L282-L285

但沿 token 维度切不能均分。注意力里位置 $t$ 要与它前面全部 $t$ 个位置交互，越靠后的位置计算越重，所以片段越靠后单位 token 的成本越高。均分序列会让第一段最轻、最后一段最重，而流水线的总时间由最慢的 stage 决定，最重的那段把节拍拖住。反过来，靠前的片段装更多 token、靠后的片段装更少，各片段的执行时间才接近，这一形态叫前长后短。
依据：paper/src-2102.07988v2/terapipe.tex L305-L305

把这件事写成式子是：序列总长 $L$、切成 $N$ 个片段，第 $i$ 个片段长 $l_i$、处理时间 $t_i$，目标是

$$T^*=\min_{l_1,\dots,l_N}\Big\{\sum_{i=1}^{N} t_i + (K-1)\cdot\max_{1\le j\le N} t_j\Big\}.$$

第一项是单个 stage 把全部片段算一遍的总前向时间；第二项是流水线开销，等于最慢片段时间乘以 $K-1$。$t_i$ 既取决于第 $i$ 段自己的长度 $l_i$，也取决于它前面所有片段的总长 $\sum_{j< i} l_j$——前缀越长，这一段每算一个位置要交互的历史越多。
依据：paper/src-2102.07988v2/terapipe.tex L325-L339

::: 12 token 请求、3 段切分的手算
题设：序列长 $L=12$，stage 数 $K=3$。成本代理：位置 $t$ 的注意力要交互 $t$ 个位置，记位置 $t$ 的成本为 $t$；一个片段的成本是段内各位置成本之和。只计注意力交互，略去其余计算与通信。本手算固定切成 3 段，只为比较切法；片段细到极限会压低 GPU 利用率。
依据：自行构造
1. 整条请求不切分，即 $N=1$：一个片段覆盖位置 1 到 12，成本 $1+2+\dots+12=78$；每个 stage 都要把这一段算一遍，代入目标函数得 $78+2\times 78=234$。
依据：自行构造
2. 均分 4+4+4：三段成本是 (10, 26, 42)，三段之和仍是 78，最慢段 42，代入目标函数得 $78+2\times 42=162$。
依据：自行构造
3. 前长后短 7+3+2：三段成本是 (28, 27, 23)，最慢段降到 28，总时间 $78+2\times 28=134$。
依据：自行构造
4. 为什么 28 已是最小：在三段的前提下，三段之和固定为 78；若要最慢段不超过 27，第一段最多装前 6 个位置（前 7 个位置之和已是 28），这时后两段合计 $78-21=57>2\times 27$，装不下。所以最慢段最小是 28，由 7+3+2 取到。
依据：自行构造
:::

### 切片怎么求、推理里怎么落地

切点不是试出来的。可以用动态规划求最优切分：把最慢片段时间 $t_{\max}$ 当作枚举参数，对每个 $t_{\max}$ 求一个子问题——在每段都不超过 $t_{\max}$ 的前提下，让单 stage 总时间 $\sum_i t_i$ 最小；再按目标函数加上流水线开销 $(K-1)\cdot t_{\max}$ 得到这个 $t_{\max}$ 下的总时间，对全部 $t_{\max}$ 取最小就是全局最优。子问题有最优子结构，可以按前缀长度逐段递推。片段处理时间 $t_{\text{fwd}}$ 不必逐一实测，拆成无前缀时的前向时间加一段上下文开销，后者用长度与前缀的线性模型估计。
依据：paper/src-2102.07988v2/terapipe.tex L336-L346；paper/src-2102.07988v2/terapipe.tex L374-L378

::: DP 递推与复杂度
1. 固定 $t_{\max}$，子问题是让单 stage 总时间 $S^*(L;t_{\max})$ 最小，且每个片段的时间不超过 $t_{\max}$。
依据：paper/src-2102.07988v2/terapipe.tex L336-L339
2. 子问题有最优子结构：长为 $i$ 的前缀，最优值等于在末尾接一段不超时的片段 $k$、再由更短前缀转移而来，即 $S^*(i;t_{\max})=\min_{1\le k\le i}\{S^*(i-k;t_{\max})+t_{\text{fwd}}(k,i-k)\}$。
依据：paper/src-2102.07988v2/terapipe.tex L342-L346
3. 递推填一张表用 $O(L^2)$ 时间，再从表里回溯出各段长度；枚举所有 $t_{\max}$ 最坏 $O(L^4)$，配合提前终止与枚举间隔下限 $\varepsilon$，实测可在一分钟内完成。
依据：paper/src-2102.07988v2/terapipe.tex L370-L372
4. $t_{\text{fwd}}(i,j)$ 拆成 $t_{\text{fwd}}(i,0)$ 与上下文开销 $t_{\text{ctx}}(i,j)$，后者用线性模型 $a_0+a_1 i+a_2 j+a_3 i j$ 拟合，相对预测误差小于 2%。
依据：paper/src-2102.07988v2/terapipe.tex L374-L378
:::

推理里把这套 token 切分落地的一种做法是 chunked pipeline parallelism（分块流水线并行）：把一条请求的输入切成若干不超过一定长度的 chunk，同一请求的不同 chunk 由不同节点同时处理，从而缩短首 token 时延；跨节点通信只发生在各 stage 的边界，可以和计算重叠。
依据：paper/src-2407.00079/data/prefill.tex L37-L43

### 固定 chunk 为什么错配、动态 chunk 怎么对齐

chunk 切成固定大小后还有一层错配。固定大小意味着每个 chunk 装的 token 一样多，而位置越靠后注意力越重，因而越靠后的 chunk 处理时间越长，各 stage 的步长随进度漂移。这种错配会沿流水线往后传：在前面的 stage 只是小幅等待，越往后的 stage 累积得越多。
依据：blog/sglang-pp-blog.txt L139-L141

对齐的办法是让 chunk 大小随前缀变化。把累计运行时间对前缀长度 $p$ 拟合为一个二次函数——注意力的计算量本来就随前缀增长；在线时，对当前前缀 $p$ 解方程

$$\mathrm{Runtime}(p+\Delta L)-\mathrm{Runtime}(p)=\mathrm{Runtime}(\text{初始 chunk})$$

得到下一个 chunk 的大小 $\Delta L$。$\mathrm{Runtime}(\text{初始 chunk})$ 是从空前缀开始、算完初始 chunk 的用时。前缀越长，这一段单位 token 的处理时间越长，$\Delta L$ 随 $p$ 增大逐步缩小，使各 stage 每一步的用时接近。预测值向下取整到 $\max(\text{page size},\,64)$ 的倍数，其中 page size 是 KV cache（见 [KV cache](wiki/kv-cache)）按页分配时一页的 token 数，对齐页边界便于按页管理显存。
依据：blog/sglang-pp-blog.txt L151-L160

预测不可能精确，于是用一个平滑因子控制每步允许缩小多少，默认 0.75：取 1 是完全跟随模型，取 0 退回固定 chunk，推荐的取值区间是 0.6 到 0.85。初始 chunk 取先按固定大小试出的最优值的 2 到 3 倍，可以减少 chunk 总数、避免尾 chunk 太小；对极长的输入建议取到 4 倍，且预测器保证后续 chunk 不小于初始值的四分之一。
依据：blog/sglang-pp-blog.txt L160-L174

::: 固定 chunk 的漂移怎么累积
题设：4 个等大 chunk，处理时间依次 1、3、5、7（随前缀递增，便于手算），流水线有 4 个 stage。
依据：自行构造
1. 前几个 chunk 把流水线填满后，每一级每一步处理一个 chunk，靠后的级要等前一级交出同一个 chunk 才能开工。
依据：自行构造
2. 相邻 chunk 的时间差是 2；从第 1 级交到第 $r$ 级要多经 $r-1$ 次传递，每步等待约 $(r-1)\times 2$ 个时间单位。
依据：自行构造
3. 于是第 2 级每步约等 2、第 4 级每步约等 6，越靠后的 stage 累积等待越多。
依据：自行构造
4. 换成更长的序列：chunk 时间 1、3、5、7、9（5 个 chunk、4 个 stage），单 stage 总时间 $1+3+5+7+9=25$，加流水线开销 $3\times 9=27$，总时间 52；其中第 4 级每步仍等 6。
依据：自行构造
:::

### 把并发请求配平（batch 维度）

另一条路线不看单条请求，而看并发的多条请求怎么拼成每一步的 micro-batch。常见做法是 chunked prefill 加一个固定的 token budget（每一步允许处理的 token 总量上限，见 [chunked prefill](wiki/chunked-prefill)）：每一步先排 decode token（生成阶段每一步新产出的 token），再在不超过这个上限的前提下排 prefill token（预填充，模型读完输入、算出首个输出 token 的阶段）。波动来自两处，一是 decode token 未必能和 prefill token 排进同一步，二是 decode token 在各步之间分布不均。把 token budget 调小理论上能减小波动，却会不成比例地压低 prefill 速率，拖累总吞吐。
依据：paper/src-2504.14775v2/chapters/1.introduction.tex L26-L32；paper/src-2504.14775v2/chapters/2.background_motivation.tex L29-L30

一种做法把 prefill 与 decode 的调度拆开，各自按全局状态定这一步排多少 token。decode 侧把正在运行的 decode token 总数 $\#RD$ 均摊到全部在流水线里推进的 micro-batch，每个排 $\#D=\#RD/\#PP_{\text{depth}}$ 个，$\#PP_{\text{depth}}$ 是流水线深度；剩下的不足 $\#D$ 时就把剩下的都排上。prefill 侧把等待 prefill 的 token 数 $\#WP$ 按计划摊到 $\#T$ 次迭代，得到一轮该发多少，再用 KV cache 空闲率 $KV_{\text{free}}$ 压一个上限，空闲越少 prefill 越少。两个量合并成

$$\#P=\max\Big(\min\big(\tfrac{\#WP}{\#T},\ \#MaxP\cdot\tfrac{KV_{\text{free}}-KV_{\text{thresh}}}{1-KV_{\text{thresh}}}\big),\ \#MinP\Big),$$

其中 $\#MaxP$、$\#MinP$ 是每轮 prefill 的上下限，$KV_{\text{thresh}}$ 是空闲率阈值。当 $KV_{\text{free}}$ 低于 $KV_{\text{thresh}}$ 时，调度器暂停 prefill（此时不再按上式排入），避免 KV cache 溢出导致正在进行的 decode 被抢占、事后重算。
依据：paper/src-2504.14775v2/chapters/4.design.tex L24-L64；paper/src-2504.14775v2/chapters/4.design.tex L90-L96

::: 一步里 decode 与 prefill 各排多少
题设：流水线深度 $\#PP_{\text{depth}}=4$，正在跑的 decode token 共 $\#RD=96$；等待 prefill 的 token 共 $\#WP=6000$，计划分 $\#T=5$ 次迭代消化；prefill 上下限 $\#MaxP=1000$、$\#MinP=16$；KV cache 空闲率 $KV_{\text{free}}=0.6$，阈值 $KV_{\text{thresh}}=0.2$。
依据：自行构造
1. decode：$\#D=96/4=24$，每个在流水线里推进的 micro-batch 排 24 个 decode token。
依据：自行构造
2. prefill 第一路：$6000/5=1200$。
依据：自行构造
3. prefill 第二路（KV 上限）：$1000\times\frac{0.6-0.2}{1-0.2}=1000\times 0.5=500$。
依据：自行构造
4. 合并：$\max(\min(1200,500),16)=500$，这一步排 500 个 prefill token。
依据：自行构造
5. 校验阈值：$0.6>0.2$，不触发暂停。
依据：自行构造
:::

### 两条路线怎么配合

两条路线补的是不同的空等，可以一起用：先按 token 维度把长请求切开，再在 batch 维度决定每一步排入的 token 数。
依据：blog/sglang-pp-blog.txt L113-L117；paper/src-2504.14775v2/chapters/1.introduction.tex L25-L32；paper/src-2407.00079/data/prefill.tex L37-L43；paper/src-2102.07988v2/terapipe.tex L442-L446

### 适用条件与边界

token 维度切分依赖单向自回归：位置只依赖历史，双向注意力模型（每个位置能看到两侧的模型，如编码器）要让每个位置看到两侧，切开就会用到未来信息，不适用。切分粒度有下限：片段太小会压低 GPU 利用率，在 GPT3-1B 的单层上，单个 token 与 256 个 token 的一次前向时间可以相同，这也是 chunk 不能无限缩小的原因。短上下文下它不带来显著额外开销：请求不长时整体进流水线也不会长时间空等，收益主要在长输入。
依据：paper/src-2102.07988v2/terapipe.tex L282-L283；paper/src-2102.07988v2/terapipe.tex L158-L158；paper/src-2102.07988v2/terapipe.tex L300-L302；blog/sglang-pp-blog.txt L113；paper/src-2407.00079/data/prefill.tex L42-L42

两条做法各有代价：一是跨机通信：流水线只在 stage 边界传张量，这条通信路径要和计算重叠才不会拖慢节拍。二是调参：动态 chunk 的平滑因子和初始大小都要按模型与硬件调，没有一组通用的最优值。
依据：paper/src-2407.00079/data/prefill.tex L42-L42；blog/sglang-pp-blog.txt L160-L166

常见误解之一是 PP 度数越大越好——度数（用多少个 stage 做流水线）越大，同样 chunk 数下气泡比率越高，纯高 PP 通常不推荐，应把不产生气泡的并行方式放在同一台机器内。另一是 chunk 越小越好——chunk 太小会让总数变多、尾 chunk 过小，反而使性能下降。
依据：blog/sglang-pp-blog.txt L88；blog/sglang-pp-blog.txt L160-L166

batch 维度配平只针对 inter-batch 失衡；各 stage 之间计算量分布不均（如层划分不同）造成的 inter-stage 失衡，不在本页两条做法的范围。
依据：paper/src-2504.14775v2/chapters/2.background_motivation.tex L81-L85
