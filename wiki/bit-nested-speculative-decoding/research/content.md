# Bit-Nested Speculative Decoding

- 描述：位嵌套自投机解码用同一份模型位表示支持草稿和目标：草稿是先猜后续 token 的低开销模型，目标验证这些候选；先量化出 W4 基底，再用残差补成 W8 目标，草稿只读基底，目标读基底加残差。
- 摘要：同一个权重先优化出 W4 基底，再用 4-bit 残差补成 W8 目标；基底和残差分成两个位平面，草稿只读 W4 平面，目标读两个平面；KV cache 按上下文在 KV8 与 KV4/KV8 之间切换。
- 主题：推理系统, 内存与缓存
- 标签：推理加速

## 资料

- Chence Yang, Ningxi Cheng, Arash Akbari, Qitao Tan, Qingchan Zhu, Ci Zhang, Changdi Yang, Yanzhi Wang, Wei Niu, Jinhui Wang, Jin Lu, Geng Yuan. BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration. arXiv:2610.02800v2, 2026. [https://arxiv.org/abs/2610.02800](https://arxiv.org/abs/2610.02800)
- YCC-DAVID. BitNest 官方仓库. commit 5020cbe541a527ec2ae9adacaaa4b4f2217bc890, 2026. [https://github.com/YCC-DAVID/BitNest](https://github.com/YCC-DAVID/BitNest)

## 短页

### 一个 8-bit 表示里的 W4/W8 两个视图

每个权重（模型里的可学参数）的实数值先量化（舍到少数整数码）成 4 位基底；4 位残差补它的误差，合成 8 位目标。W4 和 W8 指每个权重分别存 4 位、8 位整数代码的两种视图。位嵌套自投机解码就用同一份模型位表示支持两个模型角色。

[投机解码](wiki/speculative-decoding)让一个低开销的草稿模型先猜几个 token（模型处理文本的最小单位），再由一个目标模型验证。位嵌套自投机解码的草稿由目标模型自身导出，不用另一份草稿模型。
依据：paper/src-v2/iclr2027_conference.tex Methodology；paper/src-v2/iclr2027_conference.tex BitNest Weight Construction

### 省草稿存储，也不把目标压到 W4

只用一份低精度表示能省存储，但最终模型能力也被压到低精度；额外保留低精度草稿又会和目标权重、[KV cache](wiki/kv-cache) 共同占用内存。KV cache 是生成时保存历史 key 和 value 的缓存，key 用来和当前位置比较，value 用来加权求和。位嵌套先建一个 W4 草稿基底，再用残差恢复 W8 目标，因此既不用多存一份草稿，也不把验证时的目标停在 W4。
依据：paper/src-v2/iclr2027_conference.tex Introduction；paper/src-v2/iclr2027_conference.tex Self-Speculative Decoding；paper/src-v2/iclr2027_conference.tex Why retain W8 for the target?

### 先建 W4，补残差成 W8，按平面读

构造时先做等价旋转（保持模型函数不变的变换），再用 GPTQ（一种训练后量化算法）建 128 个连续权重一组的 W4 基底并固定；残差按基底缩放的 $1/16$ 编码，合成码为 $16q_4+q_r$。基底码和残差码分别连续打包成两个 4-bit 位平面；草稿只读基底平面，目标模型读两个平面。[KV cache](wiki/kv-cache) 按 BitNest 策略，短上下文（输入 token 序列长度短）中草稿和验证都读 KV8（8 位 key/value 视图），长上下文草稿读 KV4（4 位 key/value 视图）、验证读 KV8。草稿猜出候选后，目标模型仍按投机解码的规则验证。
依据：paper/src-v2/iclr2027_conference.tex Detailed Bit-Nested Weight Construction；paper/src-v2/iclr2027_conference.tex Dual-Plane Weight Storage；paper/src-v2/iclr2027_conference.tex Context-Aware KV Nesting

## 长页

[投机解码](wiki/speculative-decoding)用草稿模型猜候选、用目标模型验证。自投机把草稿从目标模型自身导出，避免另配独立草稿模型。每个权重（模型里的可学参数）可量化成低精度整数码。4 位基底和 4 位残差合成 8 位目标视图：基底指先造出的 W4 视图，残差是补基底误差的低位码，目标视图指目标模型读取的 W8 权重视图。常规量化部署先从高精度模型导出低精度副本；这个方法的实现叫 BitNest，它反过来先建将来当草稿的 W4 基底，再固定基底补出 W8 目标视图。它取 $b_W=r_W=4$，$b_W$ 是基底位宽，$r_W$ 是残差位宽；同一思路也用于 [KV cache](wiki/kv-cache)。
依据：paper/src-v2/iclr2027_conference.tex Methodology；paper/src-v2/iclr2027_conference.tex Introduction；paper/src-v2/iclr2027_conference.tex BitNest Weight Construction

### 先建基底，再补残差

#### 旋转与 W4 基底

BitNest 先把两组函数保持变换 $R_1,R_2$ 融合进原权重 $W$，得到 $W'$；融合指把变换吸收进权重本身。隐藏维度是每个 token 的向量宽度，$R_1$ 作用于这个维度；$R_2$ 作用于每个注意力头的 head 维度。函数保持指模型表示变了而输入输出不变。旋转属这类变换，能压低或重排激活（逐层流动的中间结果）里的离群值，使等价模型表示更适合低精度量化。基底由 GPTQ（一种训练后量化算法）生成，以组大小 $g=128$ 处理连续权重，$g$ 是共用一个缩放的权重个数。缩放是把整数码还原成实数的实数步长。GPTQ 返回整数码 $q_4$ 和组缩放 $s_4$，记号 $(q_4,s_4)=\mathrm{Quantize}_{4,g=128}(W')$ 就表示这一映射，其中 $q_4\in[-8,7]$，草稿权重为 $W_{\mathrm{draft}}=q_4s_4$。此后 $q_4$ 和 $s_4$ 固定，后续残差在这个基底上构造。
依据：paper/src-v2/iclr2027_conference.tex Detailed Bit-Nested Weight Construction；paper/src-v2/iclr2027_conference.tex BitNest Weight Construction；paper/src-v2/iclr2027_conference.tex LLM Quantization and Rotation-Based PTQ；repo/BitNest/optimize_rotation.py L84-L93；repo/BitNest/eval_utils/rotation_utils.py L123-L147

#### 残差补成 W8 目标

固定基底后的剩余权重误差是 $\Delta W=W'-W_{\mathrm{draft}}$。残差缩放取 $s_{\mathrm{ref}}=s_4/16$；4 位有 16 个码位，4 位基底码和 4 位残差码按数值关系 $16q_4+q_r$ 合成 8 位码。合成码的可达范围是 $[-128,119]$：$q_4=-8$ 的分支把残差下界抬到 $0$，避免低于 $-128$；$q_4=7$ 时最大合成码是 $119$。clip 表示把数值裁剪到给定范围。对 $q_4>-8$ 的权重，残差码是

$$
q_r=\operatorname{clip}\left(\operatorname{round}\left(\frac{\Delta W}{s_{\mathrm{ref}}}\right),-8,7\right).
$$

$q_4=-8$ 时使用刚才的下界分支。合成目标为

$$
W_{\mathrm{target}}=q_4s_4+q_rs_{\mathrm{ref}}=(16q_4+q_r)s_{\mathrm{ref}}.
$$

目标在不改基底的情况下增加精化信息；W4 视图因此是 W8 表示的严格子集。
依据：paper/src-v2/iclr2027_conference.tex Detailed Bit-Nested Weight Construction；paper/src-v2/iclr2027_conference.tex BitNest Weight Construction

::: 一个 4 权重组的手算例子

设 $s_4=0.16$，所以 $s_{\mathrm{ref}}=0.01$；变换后权重 $W'$ 的四个分量为 $0.63, -1.21, 0.14, 1.19$。

1. 为突出嵌套关系，这组设 GPTQ 的补偿结果就是最近格点：$q_4=(4,-8,1,7)$，草稿权重为 $0.64,-1.28,0.16,1.12$。
2. 残差依次是 $-0.01,0.07,-0.02,0.07$，除以 $0.01$ 后最近取整得 $q_r=(-1,7,-2,7)$。
3. 第 2 个权重 $q_4=-8$，残差 $7$ 在允许范围 $[0,7]$；合成码是 $16q_4+q_r=(63,-121,14,119)$。
4. 乘 $0.01$ 得 W8 目标视图 $0.63,-1.21,0.14,1.19$；每个值都由基底码加残差码恢复。

依据：自行构造

:::

### 双平面存储与 KV 策略

#### 基底与残差分成两个位平面

两个 4-bit 码不把同一个权重的高低 nibble 放进一个字节。nibble 是 4 位二进制。BitNest 把所有基底码连续打包成一个基底平面，把所有残差码连续打包成一个精化平面；$N$ 为偶数个权重时，每个平面占 $N/2$ 字节。草稿读取时执行读取和乘加的程序只取基底平面，对应每个权重的 4 bit 整数码读取量；验证时再取精化平面恢复 W8。残差缩放 $s_{\mathrm{ref}}=s_4/16$ 由基底缩放导出，所以精化平面不用另存缩放。
依据：paper/src-v2/iclr2027_conference.tex Dual-Plane Weight Storage

::: 一个可寻址的位平面例子

设 4 个权重的基底码和残差码分别是 $(4,-8,1,7)$、$(-1,7,-2,7)$。

1. 若按权重交错进字节，基底和残差都在相邻 nibble 里；草稿即使只要基底码，字节加载仍会带上残差码。
2. 有符号 4-bit 码先加 $8$ 变成无符号码。位平面 A 连续打包基底 $(12,0,9,15)$，位平面 B 连续打包残差 $(7,15,6,15)$；每个码都落在 4-bit 的 $[0,15]$。
3. 草稿内核只读平面 A 的 $N/2$ 字节；目标内核读平面 A 和 B，共 $N$ 字节，再分别减 $8$ 恢复有符号码。

依据：自行构造

:::

#### 上下文感知的 KV 读法

[KV cache](wiki/kv-cache) 保存历史 key 和 value；key 用于和当前位置比较，value 用于加权求和。token 是文本最小单位；[注意力](wiki/standard-attention)为当前位置计算每个历史位置的匹配分数，再按分数加权求和 value，head 是注意力里的一个通道组。上下文指输入 token 序列的长度。BitNest 策略是：短上下文里草稿和验证都读 KV8（8 位 key/value 视图），长上下文里草稿读 KV4（4 位 key/value 视图）基底、验证读两个平面恢复 KV8；采用这一策略是因为短上下文里 KV 流量小、KV4 会降低草稿与目标一致性，长上下文才切到低精度草稿读取。写入 KV 时，一个 key 或 value 向量 $x$ 按每个 token、每个 KV head 对称量化；对称量化围绕 0 取有符号码，用同一个缩放还原。$d=128$ 是每个 head 的维度。缩放 $s=\max_i|x_i|/7$，基底为 $q_4=\operatorname{clip}(\operatorname{round}(x/s),-8,7)$；BitNest 策略对所有 $q_4$ 取残差码 $q_r=\operatorname{clip}(\operatorname{round}((x-q_4s)/(s/16)),-8,7)$，KV8 目标为 $\hat x_{\mathrm{target}}=(16q_4+q_r)s/16$。

草稿为新候选写的 KV 状态先临时保留；验证时目标模型重算这些状态，接受的前缀留在持久缓存，被拒后缀丢弃。
依据：paper/src-v2/iclr2027_conference.tex Context-Aware KV Nesting；paper/src-v2/iclr2027_conference.tex Detailed KV-Cache Nesting

### 草稿读取减半，接受率保持高位

#### 草稿读取减半，平均接受率 95.2%

固定基底和共享缩放使目标视图由同一批基底码生成，两个精度视图保持对齐。在四个 7B–8B（参数量 7 billion 到 8 billion）语言模型（已训练用于生成文本的模型）的六个工作负载上，生成时平均接受率（被接受的草稿候选比例）是 95.2%，四个模型的平均值在 94.8%–95.7%；相对同一引擎的 16-bit 自回归基线（每生成一个 token 跑一次模型的对照），端到端解码加速是 1.48–1.61 倍。
依据：paper/src-v2/iclr2027_conference.tex Methodology；paper/src-v2/iclr2027_conference.tex BitNest Weight Construction；paper/src-v2/iclr2027_conference.tex Draft--Target Agreement；paper/src-v2/iclr2027_conference.tex Speedup for LLM inference

::: 接受率、加速和模型质量的关键数字

1. 2K（2048 个 token）前缀、batch size（一次同时处理的序列数）1、生成 256 token 的 NVIDIA RTX A6000 GPU 设置下，四个公开开源语言模型（LLaMA 和 Qwen 系列）LLaMA-2-7B、LLaMA-3-8B、Qwen2-7B、Qwen2.5-7B 的六任务平均接受率分别为 95.7%、95.3%、95.0%、94.8%。
2. 各模型六个工作负载的加速倍数如下：LLaMA-2-7B 为 1.50、1.59、1.59、1.61、1.60、1.57 倍；LLaMA-3-8B 为 1.53、1.55、1.59、1.59、1.56、1.49 倍；Qwen2-7B 为 1.48、1.53、1.54、1.55、1.53、1.52 倍；Qwen2.5-7B 为 1.52、1.58、1.57、1.58、1.59、1.57 倍。
3. 质量接近 W8 目标视图：BitNest 在 WikiText-2（文本基准）上的困惑度（语言建模误差，越低越好）为 6.2256，W8A8 为 6.1617，FP16（16 位浮点）为 6.1389；这些是 LLaMA-3-8B 的结果。

依据：paper/src-v2/iclr2027_conference.tex Draft--Target Agreement；paper/src-v2/iclr2027_conference.tex Speedup for LLM inference；paper/src-v2/iclr2027_conference.tex Why retain W8 for the target?

:::

#### 一份权重和受控 KV

嵌套权重（基底加残差得到的 W8 目标视图）的持久存储就是 W8 目标视图本身：LLaMA-2-7B 语言模型为 6.22 GiB，和单独 W8 模型相同；不用再为 W4 草稿保留另一份表示。上下文到 32K 时，KV cache 占内存的比例上升，BitNest 峰值内存为 21.0 GiB。在一类 16 GB 边缘计算板上，LLaMA-2-7B-32K（上下文扩展版）的解码约 1.5 倍于 FP16（16 位浮点）自回归。
依据：paper/src-v2/iclr2027_conference.tex Memory Saving；paper/src-v2/iclr2027_conference.tex Edge-Device Evaluation；paper/src-v2/iclr2027_conference.tex Additional Edge-Device Results

::: 存储和边缘端数字的条件

1. 6.22 GiB 是 LLaMA-2-7B 的嵌套权重存储；单独 W8 目标加独立 W4 草稿需 9.38 GiB，差距来自后者要另存草稿表示。
2. 21.0 GiB 是 32K 上下文下的 BitNest 峰值内存。
3. 边缘结果在 NVIDIA Jetson Orin NX16（16 GB 边缘计算板），batch size 1、贪心解码（每步取最高分 token）、只测解码段：LLaMA-2-7B-32K 为 9.0–9.3 tok/s（每秒生成 token 数）；FP16 AR（FP16 自回归）为 6.0 tok/s。

依据：paper/src-v2/iclr2027_conference.tex Memory Saving；paper/src-v2/iclr2027_conference.tex Edge-Device Evaluation

:::

### 适用条件与边界

它的速度收益依赖系统能按权重平面访问和执行：正常运行下草稿只发生基底平面的低开销读取，验证用两个平面恢复 W8 目标视图。低精度视图主要承担起草：LLaMA-3-8B 语言模型上 W4A8（4 位权重、8 位激活）的 GSM8K（数学题基准）为 41.4%，读回精化平面的 BitNest 目标恢复到 49.8%；KV4 正常只给草稿。KV4 草稿下，目标模型的 KV 从 KV8 改用 KV4 时，Qwen2.5-7B 语言模型的 GSM8K 从 81.6% 降到 78.4%。草稿与目标的一致性用接受率衡量，目标保真用最终输出质量衡量；两边都用 KV4 时接受率仍达 93.3%，但 GSM8K 比只用 KV4 草稿的 81.6% 更差。

KV 精化的速度收益随 KV 访问占比增大。GQA（分组查询注意力，多个查询 head 共享 key/value head）已经降低 KV 流量，KV4 草稿带来的吞吐差距小。它目前是自投机，不使用能并行生成候选 token 块的独立草稿模型。$\gamma$ 是每轮草稿候选数；$\gamma$ 增大会降低单槽（单个候选位）接受率，在 2 到 6 的测试范围里吞吐在中间 $\gamma$ 处最高。
依据：paper/src-v2/iclr2027_conference.tex Dual-Plane Weight Storage；paper/src-v2/iclr2027_conference.tex Limitation；paper/src-v2/iclr2027_conference.tex Why retain KV8 for the target?；paper/src-v2/iclr2027_conference.tex Why retain W8 for the target?；paper/src-v2/iclr2027_conference.tex Effect of Draft KV Precision；paper/src-v2/iclr2027_conference.tex Effect of Speculation Length；paper/src-v2/iclr2027_conference.tex Experimental Setup

::: 边界里的消融条件

1. 93.3% 是 Qwen2.5-7B、W8A8 权重、KV4 draft 到 KV4 target、$\gamma=4$ 的接受率。
2. 32K 下，LLaMA-2-7B-32K（MHA，多头注意力，32 个 KV head）草稿用 BF16（16 位浮点）KV、KV8、KV4 的吞吐是 23.4、28.5、34.3 tok/s；Qwen2.5-7B（GQA）对应吞吐是 50.7、51.1、51.4 tok/s。LLaMA-2-7B-32K 草稿用 KV4、KV8 的接受率分别为 0.875、0.944。
3. LLaMA-2-7B-32K 草稿用 KV4 时，4K 下 $\gamma=2$ 到 6 的吞吐是 57.2、56.8、59.3、56.1、55.6 tok/s，31K 下是 33.0、33.6、34.3、34.0、33.0 tok/s。

依据：paper/src-v2/iclr2027_conference.tex Additional Target-Precision Analysis；paper/src-v2/iclr2027_conference.tex Effect of Draft KV Precision；paper/src-v2/iclr2027_conference.tex Effect of Speculation Length

:::
