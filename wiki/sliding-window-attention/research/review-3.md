# 滑动窗口注意力审查记录（第 3 轮）

- 页面版本：`de346beb943861cdff8f91906a8791ff984c7daa`（index.html 工作树哈希）
- 审查时间：2026-09-10 16:39
- 审查者：独立审查者（第 3 轮独立上下文，未参与写作与前两轮审查）
- 已完整阅读章节：h1 标题与 reading-time；blockquote.meta；开篇 callout-blue；核心问题（learning-goals，4 题及解答）；常见误解（misconceptions）；「1. 窗口把可见集切成最近 W 个位置」（1.1、补充:-1 槽位、本章问题）；「2. 省下的是打分次数与缓存字节——以及精度取舍」（2.1、2.2、展开:40 层窗口缓存、2.3、本章问题）；「3. 层堆叠接力：感受野 k×W 与它的天花板」（3.1、3.2、3.3、补充:StreamingLLM 对照数字、本章问题）；「4. 环形缓冲：槽位、覆盖与位置编码」（4.1、代码:环形缓冲、4.2、4.3、本章问题）；「来源与范围说明」（论断与来源 C、公式与来源 F、外部数字与实验条件、构造示例、辅助解释与类比边界、简化条件及其限制）；overview.html 全文。

## 机械验证结果

**1. validate.py（实际执行）**

```
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/sliding-window-attention/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/sliding-window-attention/index.html
EXIT=0
```

**2. 第 4.1 节可运行代码块（实际执行）**

将页面中的 Python 代码原样复制运行，输出与页面「预期输出」逐字符一致：

```
W = 8  序列长度 = 20  全程一致
最后一步 p = 19
  槽位序列   : [4, 5, 6, 7, 0, 1, 2, 3]
  映射回位置 : [12, 13, 14, 15, 16, 17, 18, 19]
  期望因果窗 : [12, 13, 14, 15, 16, 17, 18, 19]
  位置 % W   : [4, 5, 6, 7, 0, 1, 2, 3]
```

（断言全程通过，无异常。）

**3. 关键数字独立复算（不引用页面脚本）**

| 数字 | 复算 | 结论 |
|---|---|---|
| 64 KiB | $128\times512\times1=65\,536$ B $=64$ KiB | 一致 |
| 2.50 MiB | $40\times64$ KiB $=2560$ KiB $=2.50$ MiB | 一致 |
| 1/8192 | $65\,536/(1\,048\,576\times512\times1)=65\,536/536\,870\,912=1/8192$ | 一致 |
| 5120 | $40\times128=5120$ | 一致 |
| 0.49% | $5120/1\,048\,576=0.4883\%\approx0.49\%$ | 一致 |
| 打分次数对照表三行 | $4096^2=16\,777\,216$、$4096\times128=524\,288$、$32\times$；$131\,072^2=17\,179\,869\,184$、$131\,072\times128=16\,777\,216$、$1024\times$；$1\,048\,576^2=1\,099\,511\,627\,776$、$1\,048\,576\times128=134\,217\,728$、$8192\times$ | 全部一致 |

**4. 来源核对（逐条四步核对，关键片段如下）**

- [C1] Mistral 7B arXiv:2310.06825 §2「Sliding Window Attention」：原文逐字核对到「The hidden state in position i of the layer k, h_i, attends to all hidden states from the previous layer with positions between i−W and i. Recursively, h_i can access tokens from the input layer at a distance of up to W×k tokens」及「At each attention layer, information can move forward by W tokens. Hence, after k attention layers, information can move forward by up to k×W tokens.」与页面引用一致；Figure 1 caption「each token can attend to at most W tokens from the previous layer」支持页面「图注取最近 W 个」的表述。
- [C2] 同上 §2「Rolling Buffer Cache」：「The cache has a fixed size of W, and the keys and values for the timestep i are stored in position i mod W of the cache. As a result, when the position i is larger than W, past values in the cache are overwritten, and the size of the cache stops increasing.」逐字一致。
- [C3] StreamingLLM arXiv:2309.17453：§1「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV」逐字一致；§3.1「removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function (Equation 1) in attention computation」逐字一致（注意：同条中「Window attention (0+y) has a drastic increase in perplexity.」实际出自 Table 2 caption，见问题 2）。
- [C4] DeepSeek-V4.1-Flash：config.json `"window_size": 128`、`"head_dim": 512`、`"n_layers": 40`、`"n_mtp_layers": 3`；model.py 第 702 行注释「The K stays fp8, quantized over the whole post-RoPE vector, RoPE tail included.」、第 707 行 `act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)`、第 777 行 `kv = torch.cat([kv, compress_kv], dim=1)`，均与页面引用一致。
- [C5] 原序列位置：model.py 第 767 行 `freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]` 一致；StreamingLLM §3.2「StreamingLLM focuses on positions within the cache rather than those in the original text. This distinction is crucial for StreamingLLM's performance.」逐字一致。
- [F2] model.py 第 718 行 `self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)` 支持 $\text{slot}(i)=i \bmod W$。
- [N1] Mistral Table 1：`window_size 4096`、`n_layers 32`；§2「At the last layer, using a window size of W=4096, we have a theoretical attention span of approximately 131K tokens.」一致。
- [N2] Mistral §2「On a sequence length of 32k tokens, this reduces the cache memory usage by 8x」一致。
- [N3] StreamingLLM Table 1（Llama-2-13B，PG19 首本书 65K）：0+1024=5158.07、4+1020=5.40、4"\n"+1020=5.60，逐项一致。
- [N4] StreamingLLM Table 3（160M 参数模型）：Learnable Sink 1+1023=18.01、Vanilla 2+1022=18.05，一致。
- [N5]/[N6]/[N7] 与官方 config.json 及复算脚本 verify_swa_accounts.py/.out、verify_sparse_attn_window.out 的存档输出一致（W=128 序列长 300 处 0 不符；W=8 序列长 20 官方函数对照一致）。
- headers.json 佐证「单 KV 头 512 维」：`layers.X.attn.wkv.weight` shape 为 `[512, 5120]`，40 层各一份；`attn_sink` shape `[64]`（64 query 头）；`mtp.0/1/2.*` 对应 3 个草稿层（DSpark 层存于 mtp.* 命名空间）。

**5. 说明项（非问题）**

「技术报告 §2.4.4：We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」不在本次允许的本地材料与可核对外部来源内，未能直接取到该报告原文片段；但「窗口 KV 用 FP8（1 B/元素）」这一论断由 config.json `"dtype": "fp8"` 与 model.py 第 702 行注释「The K stays fp8」独立佐证。页面已在「简化条件及其限制」中披露参考实现 `act_quant(..., True)` 的 inplace=True 为 quant+dequant 回 BF16（缓冲区实际 2 B/元素，128 KiB/层），与本页按部署口径算出的 64 KiB 不同——该披露经 kernel.py 第 42/52/105/111 行核实属实。此项不构成内容错误。

## 问题

- [轻微·格式] 来源与范围说明 > 「外部数字与实验条件」h3：标题写作「外部数字与实验条件」，缺少 style-guide §1 固定命名要求的「（N）」后缀。｜引文依据：style-guide 规定来源章节 h3 固定命名为「外部数字与实验条件（N）」；index.html 该处为 `<h3>外部数字与实验条件</h3>`。｜修复要求：将该 h3 标题改为「外部数字与实验条件（N）」。｜修复：h3 改为「外部数字与实验条件（N）」｜复验：已复跑 validate.py 通过并核对修改位置｜
- [轻微·可读性] 来源与范围说明 > 论断与来源（C）[C3]：「Window attention (0+y) has a drastic increase in perplexity.」被标注为出自 §3.1，但该句实际出自论文 Table 2 的 caption。｜引文依据：arXiv:2309.17453v1 全文中 "drastic" 仅出现一次，位于 Table 2 caption「Table 2: Effects of reintroduced initial token numbers on StreamingLLM. (1) Window attention (0+y) has a drastic increase in perplexity.」；§3.1 中对应表述为「While the window attention technique offers efficiency during inference, it results in an exceedingly high language modeling perplexity.」。｜修复要求：[C3] 来源列表已含「Table 2」，只需把该引文前的「§3.1」标签改为「Table 2」（或「§3.1、Table 2」），使引文位置标注与原文一致。｜修复：[C3] 标签改为「§1、Table 2 与 §3.1」，并在引文前标明该句出自 Table 2 caption（附 §3.1 的对应表述）｜复验：已复跑 validate.py 通过并核对修改位置｜

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布

核对说明：全部来源论断（C1–C5、F1–F3、N1–N7）均已给出引文片段或关键数值；关键数字（64 KiB、2.50 MiB、1/8192、5120、0.49%、打分次数对照表三行）独立复算一致；4.1 节代码实际运行输出与页面「预期输出」逐字符一致；validate.py 返回成功；overview.html 与 index.html 相互链接；5 个前置概念页面均存在；两级问题块均有「解答：」折叠块且答案独立可读。剩余 2 条轻微问题不影响正确性与主线理解，处置为可发布。

## 发布结论

- 审查轮次：第 1 轮（0/2/8）→ 第 2 轮（0/2/4）→ 第 3 轮（0/0/2，可发布）
- 每轮均由未参与写作、未参与前序审查的独立审查者执行
- 阻断与重要问题全部关闭；遗留轻微问题已逐条修复
- `.dojo/scripts/validate.py` 返回成功；headless Chrome 渲染实测通过（KaTeX 正常、无占位符、无标签重叠、折叠块数与问题数匹配）
- `overview.html` 与 `index.html` 相互链接；前置概念页均存在
- 结论：**可发布**（滑动窗口注意力）
