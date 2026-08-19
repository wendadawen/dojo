# DFlash 2 核心论断与证据

来源固定版本：
- BLOG = Inco AI, "DFlash 2: Keep Drafting Parallel", 2026-08-18, https://inco.ai/blog/dflash2/（全文快照 /tmp/dflash-research/dflash2-blog.txt）
- MC = Hugging Face 模型卡 incoai/Qwen3.8-27B-DFlash2（2026-08-18 版本，全文 /tmp/dflash-research/qwen38-modelcard.md）
- DF = DFlash 论文 arXiv:2602.06036v2（背景机制，经 ../dflash/index.html 引用）

## C 论断

- C1：DFlash 2 于 2026-08-18 由 Inco AI 发布，形态为博客 + 两份开源草稿模型（Qwen3.8-27B-DFlash2、Muse-Glimmer-30B-DFlash2），无论文；官方引用条目为指向博客的 `@misc`。定位：BLOG 页首日期与文末 Citation；GitHub z-lab/dflash README 引用块。置信：已确认。
- C2：DFlash 2 是草稿器而非独立语言模型，运行于投机解码服务内。定位：MC "It is not a standalone language model: it runs inside a speculative decoding server and drafts tokens for the target model to verify."。置信：已确认。
- C3：无损性：greedy 输出与 target 逐字节一致，采样保持 target 分布。定位：MC "Decoding is lossless: greedy output matches the target model exactly, and sampling preserves its distribution."；BLOG "with the output provably unchanged"。置信：已确认。
- C4：选择空间的证据——五层 Qwen3-4B DFlash 在 GSM8K 上（条件于前文全部正确）：首位 Recall@1 85.4%、Recall@16 99.5%；oracle 从 top-16 中选路可将接受长度从 4.27 提到 6.79。定位：BLOG Table 1 及正文。适用条件：五层 Qwen3-4B、GSM8K、含验证器 bonus token 的接受长度口径。置信：已确认。
- C5：连贯性主要是局部的——候选的适配主要取决于前一个 token，因此给相邻对打分就够。定位：BLOG "Coherence is mostly local: a candidate's fit depends mainly on the token just before it, so scoring neighboring pairs should be enough."。适用条件：论文式的设计论证（作者论证），正文标注为设计依据。置信：已确认（作为作者论证引用）。
- C6：选择器打分全并行（一次算完所有相邻对、无额外骨干或 LM head 前向），唯一串行工作是沿预计算分数走路径（greedy 或采样）；拒绝采样恢复 target 精确分布。定位：BLOG "Scoring stays fully parallel ... The only sequential work is the final walk over precomputed scores ... rejection sampling restores the exact target distribution."。置信：已确认。
- C7：后缀衰减是骨干问题：oracle 也从首位 99.5% 衰减到末位 87.8%，候选池本身在后段枯竭，选择器修不了。定位：BLOG "Even the oracle decays ... No selector can fix that, because the candidates themselves are running out. We call this suffix decay, and it is a backbone problem."。置信：已确认。
- C8：深度帮助后段但无差别（indiscriminate）：3/5/15 层在首位几乎相同、沿块拉开；15 层 3× 参数、+15.2% 循环延迟。定位：BLOG Figure 2 及数据表（DFlash 3L/5L/15L 首位 85.21/85.39/86.42%，末位 64.97/72.86/78.73%）。置信：已确认。
- C9：注意力的块内份额沿层下降（第 1 层约 30% → 第 5 层约 8%，且集中于少数头），说明「读上下文」与「建模块内依赖」两份工作失衡。定位：BLOG "the block's share of attention falls from 30% in Layer 1 to 8% in Layer 5, and what remains concentrates in a shrinking handful of heads" + Figure 3 热图。置信：已确认。
- C10：卷积设计——在每层每个 attention 与 FFN 子层前后插两抽头动态深度卷积；系数 = 学习基核 + 当前隐状态的小修正、每 16 通道共享一个修正；首位读上一个已验证 token 的表示；块内局部、无状态，不改 attention、LM head 或验证流程。定位：BLOG "we insert this two-tap dynamic depthwise convolution before and after each attention and feed-forward sublayer" + "The first position reads the last verified token's representation" + "The convolution is block-local and stateless, so it drops into DFlash without changing attention, the LM head, or verification."。置信：已确认。
- C11：五层 + 卷积接近十五层效果：+16.5M 参数（3%）、+0.7% 循环延迟；第 4–5 层块内注意力平均份额从 9.4% 降到 0.5%（卷积吸收局部工作）；结论「够到一个位置的卷积买回十个额外层的大部分收益，后缀衰减主要是局部问题」。定位：BLOG "With only 16.5M added parameters (3%), five-layer DFlash with convolution comes close to 15-layer DFlash" + "Average within-block attention across Layers 4 and 5 also falls from 9.4% to 0.5%"。置信：已确认。
- C12：两个组件合计 +1.3% 循环延迟（选择器 +0.6%、卷积 +0.7%）。定位：BLOG "the selector and the convolution together add only 1.3% to the five-layer DFlash draft–verify cycle latency."。置信：已确认。
- C13：Qwen3.8-27B 与 Muse Glimmer 两款草稿器已发布，模型卡按任务与并发分解加速。定位：BLOG "Two Drafters, Out Today"；MC Evaluation 各表。置信：已确认。
- C14（分析性推断，正文标注）：高并发下收益趋近 1 的机制——批处理填满空闲算力后，draft-verify 不再搭空闲周期，错误草稿变成挤占算力的浪费。依据：MC 并发表的趋势（DFlash 2 从 3.43× 掉到 1.01×；MTP 掉到 0.77×）+ 投机解码资源直觉。置信：推断（标注）。
- C15（厂商宣称，正文标注）：DFlash（一代）生态——SGLang/vLLM/TensorRT-LLM/llama.cpp 运行、Hugging Face 下载超 350 万次（2026-08）、NVIDIA 在 Blackwell 上测得最高 15×、Google 在 TPU 上 3×。定位：BLOG 开头段落。置信：厂商宣称（标注，不作为机制论断）。
- C16：vLLM 与 llama.cpp 的 DFlash 2 支持位于 PR 分支（vLLM PR #52816、llama.cpp PR #27342），非稳定版本。定位：BLOG Run It Now 代码块（`vllm @ git+...refs/pull/52816/head`、`git fetch origin pull/27342/head`）。置信：已确认。

## F 公式

- F1：相邻对打分 $S_t(a,b)=U_t(b)+\langle A(a)\odot H(h_t),B(b)\rangle$。定位：BLOG "A Lightweight Path Selector" 节公式。符号：$a$=前驱候选、$b$=当前候选、$U_t(b)$=DFlash 自身对 $b$ 的 logit、$A/B$=每 token 的 256 维嵌入、$H(h_t)$=上下文门（决定匹配的哪些部分算数）。置信：已确认。
- F2：两抽头动态卷积 $\operatorname{Conv}_{k}(x)_t=k_{t,0}\odot x_t+k_{t,1}\odot x_{t-1}$。定位：BLOG "A Lightweight Local Convolution" 节公式。符号：$x_t$=当前位置表示、$x_{t-1}$=前一位表示（首位取上一个已验证 token）、$k_{t,0},k_{t,1}$=动态系数（学习基核+隐状态修正）。置信：已确认。

## N 数字

- N1：选择器单项（无卷积、五层 Qwen3-4B、GSM8K）接受长度：DFlash 4.27（T=0）/3.78（T=1）；+DSpark 修正头 4.49/4.08（+77.8M 参数、+9.6% 延迟）；+路径选择 4.61/4.25（+2.0M、+0.6%）。定位：BLOG Table 2。条件：五层 Qwen3-4B、GSM8K。置信：已确认。
- N2：逐位置 Recall（条件于前文全对，五层 Qwen3-4B、GSM8K）：Recall@1 从 85.4% 衰减到 72.9%（位置 0→6）；Recall@16 从 99.5% 到 87.8%；oracle 接受长度 6.79。定位：BLOG Table 1。置信：已确认。
- N3：层数与卷积对照（Recall@1，位置 0→6）：3L 85.21→64.97%；5L 85.39→72.86%；15L 86.42→78.73%（3× 参数、+15.2% 延迟）；5L+卷积 85.83→77.61%（+3% 参数、+0.7%）。定位：BLOG Figure 2 数据表。条件：Qwen3-4B、GSM8K、T=0、无选择器。置信：已确认。
- N4：Qwen3.5-4B 五项基准接受长度（thinking 开启、T=1.0、top-p 0.95、top-k 20、presence penalty 1.5、无损拒绝采样）：均值 MTP 4.54、DFlash 4.92、DSpark 5.49、DFlash 2 5.97；逐项 GSM8K 4.78/4.99/5.69/6.20、MATH-500 5.04/5.42/6.20/6.76、HumanEval 4.84/5.43/5.80/6.28、MBPP 4.16/4.49/4.96/5.41、MT-Bench 3.90/4.26/4.77/5.20。定位：BLOG Table 3。置信：已确认。
- N5：MATH-500 逐位置条件接受率（Qwen3.5-4B、同上采样）：DFlash 2 首位 88.3%、末位（位置 14）86.48%；MTP 末位 77.85%、DFlash 末位 77.48%、DSpark 末位 79.86%。定位：BLOG Figure 5 数据表。置信：已确认。
- N6：Qwen3.8-27B 接受长度（块 8、官方默认采样）：均值 MTP 4.28、DSpark 3.62、DFlash 2 4.80；GSM8K 5.02/4.36/5.46、MATH-500 4.72/3.92/5.28、HumanEval 3.91/3.30/4.39、MBPP 3.99/3.51/4.79、MT-Bench 3.74/3.01/4.10。定位：BLOG Table 4 + MC Acceptance Length 表（两者一致）。置信：已确认。
- N7：Muse Glimmer 接受长度（块 16、默认采样）：均值 DFlash 4.44、DSpark 4.48、DFlash 2 5.70；GSM8K 5.43/5.45/6.57、MATH-500 5.39/5.01/6.56、HumanEval 4.11/4.33/5.66、MBPP 3.74/4.02/5.30、MT-Bench 3.52/3.59/4.42。定位：BLOG Table 5。置信：已确认。
- N8：吞吐（Qwen3.8-27B、SGLang、单卡 H200、FA3、块 8=每步 7 草稿 token、Qwen3.8 官方推荐采样 + xhigh reasoning、最大 4096 新 token）：并发 1：GSM8K 236.1 tok/s（3.43×）、MATH-500 3.34×、HumanEval 3.11×、MBPP 3.29×、MT-Bench 2.67×；并发 8：2.84×/2.85×/2.67×/2.78×/2.27×；并发 32：1.45×/1.30×/1.16×/1.25×/1.01×。MTP 同条件并发 32：1.04×/0.94×/0.84×/0.87×/0.77×。定位：MC Throughput 各表（含绝对 tok/s）。置信：已确认。
- N9：Muse Glimmer 吞吐区间 3.1–4.6×（batch size 1）。定位：BLOG "That translates into 2.7–3.4× ... on Qwen3.8-27B, and 3.1–4.6× on Muse Glimmer."。置信：已确认。
- N10：发布物：incoai/Qwen3.8-27B-DFlash2（含 GGUF Q4_K_M 镜像）与 incoai/Muse-Glimmer-30B-DFlash2，Apache 2.0。定位：MC frontmatter（license: apache-2.0）+ BLOG。置信：已确认。
- N11：对 DFlash 的接受长度增益按基准为 16–25%。定位：BLOG "Across benchmarks the gain runs 16–25%."。置信：已确认。
