<!-- review-meta
round: 4
page: wiki/depthwise-conv/index.html
reviewed_content_sha256: 1ae5f0d9f7927965
-->
# 深度可分离卷积审查记录（第 4 轮）

- 页面版本：5cf48e21f4971a85c72cb9d4235ef3d15b716ec6
- 审查时间：2026-09-13 20:13 CST
- 审查者：编排者派发的独立审查者（独立子代理）
- 已完整阅读章节：核心问题 / 常见误解 / 1. 标准卷积——滤波与组合一步完成 / 2. 拆成两步——depthwise 与 pointwise / 3. 省了多少——比值推导 / 4. 大模型里的用法——KDA 的短深度卷积 / 来源与范围说明

## 核对依据（回源片段）

- MobileNets（arXiv:1704.04861）§3.1：标准成本 `DK⋅DK⋅M⋅N⋅DF⋅DF`；depthwise 成本 `DK⋅DK⋅M⋅DF⋅DF`；pointwise 成本 `M⋅N⋅DF⋅DF`；合计 `DK⋅DK⋅M⋅DF⋅DF+M⋅N⋅DF⋅DF`；缩减比 `1/N+1/DK²`；原句 “MobileNet uses 3×3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy”。页面 [F1]–[F5]、[C1]–[C3]、[N1] 与之一致。
- MobileNets §4.1 “Model Choices”，Table 4 “Depthwise Separable vs Full Convolution MobileNet”：Conv MobileNet 71.7% / 4866M Mult-Adds / 29.3M Params；MobileNet 70.6% / 569M / 4.2M。页面“简化条件及其限制”中 §4 Table 4 的定位与四个数字均对上；4866/569=8.55、71.7−70.6=1.1 与页面所述一致。
- GLM-5.3-Flash 官方 config.json（zai-org/GLM-5.3-Flash）：`text_config.hidden_size=4096`、`linear_attn_config={num_heads:64, head_dim:128, short_conv_kernel_size:4}`、`kda_layers` 34 项 / `full_attn_layers` 11 项。→ qkv_dim=64×128=8192、conv_dim=3×8192=24576、kernel=4 均对上。
- transformers `models/glm5_next/modeling_glm5_next.py`（main）：`600 self.qkv_dim = self.head_dim * self.num_heads`；`606-608 q/k/v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)`（4096→8192）；`610 self.conv_dim = self.qkv_dim * 3`；`611-618 self.conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1)`。→ 页面 [C4] 的「通道 24576、kernel 4、groups=通道数、无偏置、因果」全部成立。
- checkpoint 张量头：`...layers.{0..33}.self_attn.{q,k,v}_conv1d.weight` 各为 BF16 `[8192, 1, 4]`（34×3=102 个张量）。→ 页面 [N3]「checkpoint 拆三个 [8192,1,4]」成立。
- 实跑两段代码块（python3 / torch 2.x）：输出与页面「预期输出」逐行一致（标准 9,437,184；depthwise 147,456；pointwise 1,048,576；合计 1,196,032；比值 0.126736 / 8.2 位公式 0.126736；7.9x；因果卷积 `[-1.0, -2.5, -3.5, -3.5, -3.5]`；groups=2 通道 0 `[1.0, 3.0, 5.0, 7.0]`、通道 1 `[10.0, 20.0, 30.0, 40.0]`）。`.dojo/scripts/validate.py` 返回 `validation ok`。

## 问题

- [重要·技术] 3. 省了多少·本章问题解答（L279）末句「这也解释了收益上限：比值 $1/N+1/D_K^2$ 恒大于其中较大的一项，故收益不超过相对较小项的倒数。」由「比值 > 较大项」只能推出「收益 = 1/比值 < 1/较大项」，即上限是**较大项的倒数**（也就是 $N$ 与 $D_K^2$ 中**较小者**的值）；句子却写成「相对较小项的倒数」（= $N$ 与 $D_K^2$ 中较大者），推理链与被推出的对象不一致，给出的「收益上限」是无意义的那个。以本页参数 $N=64$、$D_K^2=9$ 计：页面写出的界是 $1/(1/N)=64$，而有意义的界是 $9$（实际收益 $7.89$）。｜引文依据：MobileNets §3.1 缩减比 `1/N+1/DK²`；$1/(1/64+1/9)=7.89$，$\min(64,9)=9$，$\max(64,9)=64$。｜修复要求：改为「比值 $1/N+1/D_K^2$ 恒大于其中较大的一项，故收益 $=1/(1/N+1/D_K^2)$ 不超过较大项的倒数，即不超过 $N$ 与 $D_K^2$ 中较小者的值（$N=64,D_K^2=9$ 时收益 $<9$）」。｜修复：按修复要求改写 L279——「故收益 $=1/(1/N+1/D_K^2)$ 不超过较大项的倒数，即不超过 $N$ 与 $D_K^2$ 中较小者的值（$N=64,D_K^2=9$ 时收益 $<9$）」。｜复验：已复跑 validate.py 通过并核对修改位置｜
- [轻微·技术] 来源说明 [C4]（L362）与 4. 大模型里的用法·本章问题解答（L350）两处均引 GLM 的 `causal_conv1d_fn`（L404-410），并称其「用 $F.\mathrm{conv1d}(\mathrm{padding}{=}k{-}1)$ 后取 $[:, :, :\mathrm{seq\_len}]$」。核对当前 transformers main 同名文件：`def causal_conv1d_fn` 在 **L396**，`padding = weight.shape[-1] - 1` 在 **L404**，`F.conv1d(... groups=hidden_size)` 在 **L406-411**，而 `[:, :, :seq_len]` 在 **L412**——所引 L404-410 恰好未包含其声称的那一行头部切片；且页面未注明所依据的源码版本，main 上游行号会漂移，无法据该行号定位。[C4] 的 conv1d 配置引用 L603-680 与当前 main（类 `Glm5NextTextLinearAttention` 起于 L587、`self.conv1d` 在 L611-618）相符，无需改。｜引文依据：`396:def causal_conv1d_fn(`；`404: padding = weight.shape[-1] - 1`；`406-411: out = F.conv1d(hidden_states.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias, padding=padding, groups=hidden_size)`；`412: )[:, :, :seq_len]`。｜修复要求：把 `causal_conv1d_fn` 的行号改为覆盖 L396-412（至少到 L406-412），并在 [C4] 注明所依据的 transformers 版本/commit；或删去行号只留函数名。｜修复：L350 与 [C4]（L362）的 `causal_conv1d_fn` 行号改为 L396-412；[C4] 补注所依据版本「（main，行号随上游漂移）」。｜复验：已复跑 validate.py 通过并核对修改位置｜
- [轻微·表述] 引言末句（L65）「本文讲清拆分的原理、省算的账，以及「只见半边」的 KDA 用法。」以「本文」为主语宣告文章将要讲什么，与规范列举的「本页将…」同型，属元话语式自我指代；引言前两段已说明问题与范围，该句不承载信息。｜引文依据：不适用（表述类）。｜修复要求：删除该句，或改为不以自指为主语的陈述（如直接写范围：「范围：depthwise / pointwise 的拆分、缩减比推导，以及 KDA 只取 depthwise 半边的用法」）。｜修复：删除该句（L65 末句「本文讲清拆分的原理、省算的账，以及「只见半边」的 KDA 用法。」），引言前两段已说明问题与范围。｜复验：已复跑 validate.py 通过并核对修改位置｜
- [轻微·可读性] 4. 大模型里的用法·代码块「观察重点」（L333）「而 $t\geq2$ 时左侧补零给出的 $x[-1]=0$ 恰等于等差数列的外推值，故此后输出恒为 $-3.5$。」左侧补零进入滑窗只发生在 $t=2$（窗口首元素为补零 $x[-1]=0$）；$t\geq3$ 的窗口不含任何补零（$t=3$ 为 $[1,2,3,4]$、$t=4$ 为 $[2,3,4,5]$），它们恒为 $-3.5$ 是「核系数和为 0 + 等差输入」的结果（同句前半已给出该机制），与补零无关。把 $t\geq3$ 也归因于「左侧补零」不准确。｜引文依据：实跑 torch 输出 `[-1.0, -2.5, -3.5, -3.5, -3.5]`；核 $(1,0.5,-0.5,-1)$，系数和 $1+0.5-0.5-1=0$。｜修复要求：改为「$t=2$ 时窗口首元素是补零 $x[-1]=0$，恰等于等差数列的外推值，使该窗口也成为等差窗口；$t\geq3$ 的窗口本就无补零、同为等差窗口，故 $t\geq2$ 起输出恒为 $-3.5$」，或删去对补零的单独归因。｜修复：L333 改为「$t=2$ 时窗口首元素是补零 $x[-1]=0$，恰等于等差数列的外推值，使该窗口也成为等差窗口；$t\geq3$ 的窗口本就无补零、同为等差窗口，故 $t\geq2$ 起输出恒为 $-3.5$」；并删去随之失效的「来自两点」归纳，改为「$-3.5$ 恒定：核系数之和为零…与位置 $t$ 无关」。｜复验：已复跑 validate.py 通过并核对修改位置｜

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（重要 1 须关闭；三条轻微不影响主线，建议同轮一并处理）
