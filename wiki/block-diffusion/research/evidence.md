# 块扩散（block-diffusion）核心论断与证据

来源固定版本：
- BD = Arriola et al., "Block Diffusion: Interpolating between Autoregressive and Diffusion Language Models", ICLR 2025, arXiv:2503.09573（TeX 源码，本地 /tmp/dflash-research/bd-tex/iclr2025_conference.tex）
- DF = Chen, Liang, Liu, "DFlash: Block Diffusion for Flash Speculative Decoding", ICML 2026, arXiv:2602.06036v2（TeX 源码，本地 /tmp/dflash-research/tex/）

核心论断（C）、公式（F）、数字（N）：

- C1：块扩散在块间自回归、块内做离散扩散去噪。定位：BD §3 开头 "autoregressively modeling blocks of tokens and performing diffusion within each block"。适用条件：无。置信：已确认。
- C2：单一 transformer 用 block-causal 注意力掩码实现所有块的并行训练；块 $b$ 的 token attend 块 $1..b$。定位：BD §3.1 Model Architecture "we parameterize $\x_\theta$ using a transformer with a block-causal attention mask ... tokens in block $b$ attend to tokens in blocks 1 to $b$"。适用条件：掩码按块划分构造。置信：已确认。
- C3：K/V 是模型每块输出的一部分，生成后续块时作为缓存复用。定位：BD Eq.(6) 模型签名 $\x_\text{logits}^b, \mathbf{K}^b, \mathbf{V}^b \gets \x^b_\theta(\x^b_t, \mathbf{K}^{1:b-1}, \mathbf{V}^{1:b-1})$ 及正文 "we define $\x_\theta$ to support these as input and output"。适用条件：无。置信：已确认。
- C4：块扩散克服纯扩散的两个限制：支持任意长度生成、推理可用 KV cache 与并行 token 采样。定位：BD abstract "Block diffusion overcomes key limitations of both approaches by supporting flexible-length generation and improving inference efficiency with KV caching and parallel token sampling"。适用条件：与全并行扩散 LM 对比。置信：已确认。
- C5：训练时每个块至少过模型两次：一次带噪（计算该块去噪损失），一次干净（作为后续块的条件）。定位：BD §3.2 开头 "observe that denoising $\x_t^b$ requires a forward pass on this noisy input, while denoising the next blocks requires running $\x_\theta$ on the clean version $\x^b$. Thus every block has to go through the model at least twice."。适用条件：单一大模型同时承担所有块的参数化。置信：已确认。
- C6：块大小为 1 时扩散目标在期望上等价于 AR 似然，但梯度方差高。定位：BD §3 开头 "We show that for a block size of one, the diffusion objective suffers from high variance despite being equivalent to the autoregressive likelihood in expectation"；§4.2 Case Study: Single Token Generation。适用条件：块大小退化为 1 的极限。置信：已确认。
- C7：采样逐块进行：每块从（含噪声的）初始状态出发迭代去噪，去噪轮数 $T$ 从高到低逐步推进，完成后把该块 K/V 并入缓存进入下一块。定位：BD 附录 Algorithm "SAR Inference"（外层 for 循环 over blocks，内层 for 循环 $t=T$ to $1$，块完成后 "Update kv cache after sampling block $i$"）。适用条件：离散去噪步数 $T$ 的标准采样器。置信：已确认。
- C8：BD3LM 采用掩码扩散的高效采样器，token 一旦揭开不再重新遮住，生成步数上界为序列长度。定位：BD §6.2（实验）"adopts an efficient sampler from masked diffusion, where the number of generation steps (NFEs) is upper-bounded by $L$ since tokens are never remasked"。适用条件：掩码扩散简化目标对应的采样器。置信：已确认。
- C9：DFlash 把块内去噪简化为单次前向：所有被遮位置一次并行预测。定位：DF §3.1 "DFlash predicts the next token block using a block-level diffusion process. All masked positions within a block are decoded in parallel in a single forward pass."；DF §1 "the entire block, every position, predicted in parallel"。适用条件：作为草稿器、有目标模型验证兜底。置信：已确认。
- C10：扩散起草的延迟对块大小不敏感（一次前向的并行成本远低于同规模模型逐 token 的串行成本）。定位：DF §2.2 "$t_{\text{parallel}} \ll \gamma \cdot t_{\text{step}}$ for models of comparable size. For moderate block sizes, $T_{\text{draft}}$ is therefore largely insensitive to $\gamma$"。适用条件：中等块大小、可比模型规模。置信：已确认。
- C11：现有开源扩散 LM 通常生成质量逊于自回归模型，且维持质量往往需要很多去噪步，拖慢原始推理速度。定位：DF §1 引述（"current open-source dLLMs typically underperform their autoregressive counterparts ... maintaining acceptable output quality often necessitates a high number of denoising steps"）；BD abstract "lag in likelihood modeling ... limited to fixed-length generation"。适用条件：以质量相当的 AR 模型为参照。置信：已确认。
- F1：自回归分解 $\log p_\theta(\x)=\sum_{\ell=1}^{L}\log p_\theta(x^\ell\mid \x^{<\ell})$。定位：BD Eq.(1)。置信：已确认。
- F2：块扩散分解 $\log p_\theta(\x)=\sum_{b=1}^{B}\log p_\theta(\x^{b}\mid \x^{<b})$。定位：BD Eq.(4)。置信：已确认。
- F3：模型签名 $\x_\text{logits}^b,\mathbf{K}^b,\mathbf{V}^b\gets \x^b_\theta(\x^b_t,\mathbf{K}^{1:b-1},\mathbf{V}^{1:b-1})$。定位：BD Eq.(6)。置信：已确认。
- F4：训练目标（NELBO 按块求和）$\mathcal{L}_\text{BD}(\x;\theta):=\sum_{b=1}^{B}\mathcal{L}(\x^{b},\x^{<b};\theta)$。定位：BD Eq.(5)。置信：已确认（本页只概述用途，不推导）。
- N1：DFlash 草稿器块大小 16（LLaMA-3.1 为 10）、层数 5（Qwen3-Coder 为 8）。定位：DF §5 Implementation。适用条件：Qwen3/LLaMA 系列实验配置。置信：已确认。
- N2：BD3LM 预训练用最大块大小 $L'=L$，再按不同 $L'$ 微调；与 SSD-LM 对比时用每块 $T=25$ 步达到可比 NFE。定位：BD §6 "We pre-train a base \algo{} using the maximum block size $L'=L$"；§6.2 "$T=25$ where NFEs are comparable across methods"。适用条件：LM1B/OWT 实验设置。置信：已确认（页面仅作轮数权衡的例证，不展开实验）。

原文图候选：本页为概念页，结构图自绘（注意力掩码网格、采样循环），不使用论文原图。
