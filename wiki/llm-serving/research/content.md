# LLM Serving

- 描述：LLM 服务是把大语言模型跑起来、按请求回答的在线系统，一次请求分 prefill 与 decode 两个阶段执行，快慢拆成首字延迟 TTFT 与每 token 生成耗时 TPOT 两个指标。
- 摘要：一次请求先 prefill、再 decode：prefill 并行处理整段输入、产出首字，decode 每步只生成一个新 token。输入较长时 prefill 偏计算、decode 偏访存，服务快慢也因此拆成首字延迟 TTFT 与每 token 生成耗时 TPOT 两个指标，对两者各设 SLO，用满足 SLO 的每卡最大请求速率 goodput 衡量；把两阶段放到同一批卡上会互相干扰，分到两组卡上各配资源能消除干扰，代价是各存一份权重并搬运中间结果。
- 主题：推理系统, 并行与通信
- 标签：推理系统

## 资料

- DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv:2412.19437, 2024. [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- Yinmin Zhong 等. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. OSDI 2024. [https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)

## 短页

### 一次请求走两个阶段

大语言模型（LLM）按请求一问一答，回答是逐个 token（模型读写文本的最小单位）生成的。一次请求分两个阶段：prefill 阶段一次性并行处理用户输入的全部 token，为每个输入位置算出并保存 key 和 value（模型为每个位置存下的两个向量，供后续步骤复用），并产出第一个 token；decode 阶段每步只处理一个新 token，复用前缀已缓存好的 key 和 value，逐 token 往下生成，直到出结束符。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex LLM Inference

### 两个阶段的计算特性不同

输入不短时，prefill 要一次算完的 token 多，偏向把算力用满，是计算受限；decode 每步要搬动的模型权重（模型里可训练的数字）和缓存，和 prefill 一次要搬的差不多，可它每步只算一个 token，时间几乎花在把权重和缓存从显存（GPU 上的内存）读进来，是访存受限。两个阶段合在同一批里，一个 prefill 步比一个 decode 步长得多，decode 的每个 token 都被 prefill 拖慢；反过来，decode 的加入也把 prefill 拉长。
依据：paper/src-distserve/sections/background.tex LLM Inference；paper/src-distserve/sections/background.tex Problems and Opportunities

### 服务好不好：TTFT、TPOT 与 goodput

等第一个 token 出来的时间和后续每个 token 出来的速度是两个不同的量，要分成两个指标。TTFT（time to first token，首字延迟）是 prefill 阶段的时长；TPOT（time per output token，每 token 输出时间）是 decode 阶段平均生成一个 token 的时间，首 token 不计。一次请求的总延迟等于 TTFT 加上 TPOT 乘以 decode 阶段生成的 token 数。SLO（service-level objective，服务级目标）是对 TTFT 与 TPOT 分别设的上限；达成率是要满足这个上限的请求占比；goodput 是在满足达成率的前提下、每张卡每秒最多能服务的请求数。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Background and Motivation

## 长页

大语言模型（LLM）按请求一问一答。服务系统把模型跑在 GPU 上，收到一个请求就逐 token 生成回答。模型不是一次把整段回答算出来，而是每算一步产出一个 token，下一步再基于已经有的全部 token 算下一个。这里的 token 是模型读写文本的最小单位，文本先被切成 token 再送进模型。模型里可训练的数字叫参数，也叫权重。被服务的模型可能是稠密模型（每个 token 都过全部参数的普通模型），也可能是混合专家（见 [MoE](wiki/moe)）。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex LLM Inference；paper/src-dsv3/main.tex Introduction
模型算得快不快，直接决定这个请求等多久；系统还要在一批 GPU 上尽量多接请求、少花资源。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Background and Motivation

### prefill 和 decode：一次请求的两个阶段

服务把一次请求的执行分成两个阶段。prefill 阶段一次性并行处理用户输入的全部 token，为每个输入位置算出并保存 key 和 value 张量（张量是向量和矩阵的推广，这就是 [KV cache](wiki/kv-cache)），并产出第一个 token。输入不简短时，prefill 一次要算的 token 多，倾向于计算受限。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex LLM Inference

decode 阶段每步只处理一个新 token，复用前缀已缓存好的 key 和 value，逐 token 往下生成，直到出结束符。它每步要搬运的模型权重和缓存与 prefill 相当，却只算一个 token，因此受限于访存。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex LLM Inference

### 服务好不好怎么量：TTFT、TPOT 与 goodput

等第一个 token 出来的时间和后续每个 token 出来的速度是两个不同的量，要分成两个指标。TTFT（time to first token，首字延迟）是 prefill 阶段的时长；TPOT（time per output token，每 token 输出时间）是 decode 阶段平均生成一个 token 的时间，首 token 不计。一次请求的总延迟等于 TTFT 加上 TPOT 乘以 decode 阶段生成的 token 数（也就是总生成 token 数减 1）。不同应用对两者的要求不同：实时对话看重 TTFT，TPOT 只要快过人的阅读速度就够用。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Background and Motivation

SLO（service-level objective，服务级目标）是对 TTFT 与 TPOT 分别设的上限，以及要满足这个上限的请求占比，叫达成率。goodput 是在满足达成率的前提下、每张卡每秒最多能服务的请求数；每张卡的 goodput 越高，摊到每个请求上的成本越低。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Background and Motivation

::: 一条请求的总延迟怎么算
设一次请求的 TTFT 为 200 毫秒，TPOT 为 50 毫秒，decode 阶段生成了 5 个 token。
依据：自行构造

1. 代入总延迟公式：$\mathrm{TTFT} + \mathrm{TPOT} \times 5 = 200 + 50 \times 5 = 450$ 毫秒。
2. 若把 TPOT 压到 40 毫秒、TTFT 不变，总延迟降到 $200 + 40 \times 5 = 400$ 毫秒。
3. 长回答（decode 生成的 token 多）时总延迟里 TPOT 这一项占比大，更值得压 TPOT；短回答时更值得压 TTFT。
依据：自行构造
:::

### 合设还是分离：两阶段放不放在同一批卡上

prefill 和 decode 共用同一份模型权重，所以默认把两个阶段放在同一批 GPU 上，这叫合设。合设省显存（GPU 上的内存），也不用来回搬 key 和 value，但两个阶段会互相干扰：一个 prefill 步比一个 decode 步长得多，它们排在同一批里，decode 的每个 token 都被 prefill 拖慢；反过来，decode 的加入也把 prefill 拉长。合设还把两个阶段的资源与并行方式绑在一起：prefill 偏计算、适合把算子（模型里的一个计算步骤）切到多张卡上算（张量并行，见 [张量并行](wiki/tensor-parallelism)）来缩短单步时间，decode 偏访存、适合另一种配法，绑在一起就得按更紧的那一个配，容易多配资源。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Problems and Opportunities；paper/src-distserve/sections/background.tex LLM Inference

另一个做法是把两个阶段分到不同的 GPU 上，叫分离：prefill 用一组卡、decode 用另一组卡，干扰就没有了，两组卡还能各自按所属阶段的特性配置资源和并行方式。分离的代价是每组卡都要各存一份完整的模型权重，prefill 算完要把 key 和 value 传给 decode 那组卡，并且扩容时以承载一整份模型权重的一组卡为一个最小单位，不能只加其中一张。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Problems and Opportunities

### 适用条件与边界

两个阶段的划分和两套指标，前提是模型逐 token 生成回答。模型的类型、输入输出长度、部署规模都只是具体取值，换一份配置它们就变，而两阶段与 TTFT/TPOT 的度量方式不随之改变。一条容易混的地方：goodput 不等于吞吐；吞吐指每秒生成的 token 数，不看延迟是否达标，goodput 只算满足 SLO 的那部分请求。
依据：paper/src-distserve/sections/background.tex Background and Motivation；paper/src-distserve/sections/intro.tex Introduction
