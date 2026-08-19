# DFlash 核心论断与证据

论文固定版本：arXiv:2602.06036v2（TeX，本地 /tmp/dflash-research/tex/）。定位缩写：§=section，Eq.=equation，Tab.=table，Fig.=figure，A.=appendix。

## C 论断

- C1：现有先进投机解码（EAGLE-3）仍依赖自回归起草，串行且易误差累积，把实际加速限制在约 2–3×。定位：§1 第 2 段 "state-of-the-art methods like EAGLE-3 still rely on autoregressive drafting. This serial drafting process is not only inherently inefficient but also susceptible to error accumulation, which effectively caps achievable speedups at approximately 2-3×"。适用条件：以 2026 年初已部署方法为参照。置信：已确认（论文声称）。
- C2：LLM 隐藏特征隐含多个未来 token 的信息（论文设计前提）。定位：§1 "As observed by samragh2025, large autoregressive LLMs' hidden features implicitly contain information about multiple future tokens"。适用条件：引自外部文献。置信：已确认（文献已有结论，引用标注）。
- C3：prefill 时从 target 第 2 层到倒数第 3 层之间均匀取 5 层隐藏态，拼接后经轻量投影融合为 target 上下文特征。定位：§3.1 "we extract hidden representations from a fixed set of layers uniformly sampled from shallow to deep. These hidden states are concatenated and passed through a lightweight projection layer"；§5 Implementation "extracted from 5 layers uniformly selected between the second layer and the third-to-last layer"。置信：已确认。
- C4：EAGLE-3 把 target 特征只融合在 draft 输入（与 token embedding 拼接过 FFN），draft 越深信息越稀释，加层收益递减；DFlash 把融合特征注入每一层的 K/V，接受长度随层数有效扩展。定位：§3.1 "they fuse these features with the draft model's token embeddings and feed them only as inputs ... As the draft model depth increases, the information from target model becomes more and more diluted"；"This design provides strong and consistent conditioning throughout the draft model, enabling acceptance length to scale effectively with the number of draft layers"。适用条件：同规模对照（A.6 消融）。置信：已确认。
- C5：无 target 条件的 5 层块扩散草稿器加速仅 2–3×，原因是缺上下文、等于从零预测未来 token。定位：§3.1 + A.2 Table 5（naive diffusion：GSM8K 2.83×/Math500 3.73×/AIME25 3.35×，T=0）。置信：已确认。
- C6：DFlash 块内所有被遮位置一次前向并行预测（单步，无迭代去噪）。定位：§3.1 "All masked positions within a block are decoded in parallel in a single forward pass"。置信：已确认。
- C7：训练构块方式：从 response 随机采样 anchor 作块首、遮其后 block_size−1 个位置，训练并行预测；随机化 anchor 同时提供数据增广。定位：§3.2 "We randomly sample anchor tokens from the response, use each anchor as the first position of a block, and mask the remaining positions"。消融 Tab. 7（Sample 4.69× vs Standard 4.13× Math500）。置信：已确认。
- C8：训练时所有块拼接成单序列、块间注意力禁止（稀疏掩码、Flex Attention 实现），一次前向反向训多个块。定位：§3.2 "all blocks are concatenated into a single sequence and processed jointly using a sparse attention mask ... Tokens attend bidirectionally within the same block ... attention across different blocks is disallowed"。置信：已确认。
- C9：块内早期位置的预测错误使后续全部作废，因此训练对早期位置加权。定位：§3.2 "Errors at early positions within a draft block invalidate all subsequent tokens ... we apply an exponentially decaying weight"。置信：已确认。
- C10：draft 与 target 共享 token embedding 与 LM head 且训练中冻结，只更新 draft transformer 层。定位：§3.2 "the draft model shares the token embedding layer and language modeling head with the target model and keeps them frozen during training"。置信：已确认。
- C11：KV 注入只增加共享投影 $W_c\in\mathbb{R}^{D\times 5D}$，Qwen3.5-35B-A3B（$D=2048$，BF16）约 42 MB，相对约 70 GB 的 target 可忽略。定位：A.3。置信：已确认。
- C12：5 层 draft 在综合加速上最优（8 层 $\tau$ 更高但延迟更高）。定位：§5.4 Tab. 6（3L 4.69×/5L 4.71×/8L 4.64×，Math500）。置信：已确认。
- C13：训练块大小可向下泛化（b16 训练、b8 推理接近 b8 训练），反向不成立；块 8 模型 35.7% 的块被整块接受，说明块 8 常未被用满。定位：§5.4 Tab. 8 与正文。置信：已确认。
- C14：DFlash 在 SGLang 上启用 Spec-v2 调度重叠，B200 单卡、FA4 后端。定位：§5.3 "All experiments are conducted on a single B200 GPU with the FlashAttention-4 (FA4) backend. We enable Spec-v2 scheduling overlap"。置信：已确认。
- C15：长上下文：4K 训练的 base 草稿器在 16K 上接受长度明显衰减（hotpotqa 4.91→3.61），用 LongAlign-10K 的 1.6K 样本微调 3 epochs 恢复到 6.05。定位：§5.5 Tab. 4。置信：已确认。
- C16：与 EAGLE-3 在 LLaMA-3.1-8B 上用完全相同训练数据（UltraChat+ShareGPT）对比，DFlash 全任务全并发领先两种树大小配置。定位：§5.4.1 Tab. 3（GSM8K conc1：DFlash 2.4× vs EAGLE-3(10) 1.6×、EAGLE-3(60) 1.9×）。适用条件：SGLang Spec-v1（Spec-v2 不支持 EAGLE 树验证）。置信：已确认。
- C17（分析性推断，正文标注）：高并发收益缩水的机制解释——批处理填满空闲算力后，draft+verify 不再搭在空闲周期上，错误草稿变成挤占有效请求的浪费计算。依据：SGLang 表中加速随并发下降的趋势（Tab. 2）+ 投机解码的资源直觉。置信：推断（标注入正文）。

## F 公式

- F1：每 token 平均延迟 $L=(T_{\text{draft}}+T_{\text{verify}})/\tau$，$\tau\in[1,\gamma+1]$ 含 bonus token。定位：§2.1 Eq.(1)。置信：已确认（引 sadhukhan2025 MagicDec）。
- F2：自回归起草成本 $T_{\text{draft}}=\gamma\cdot t_{\text{step}}$。定位：§2.2 Eq.(2)。置信：已确认。
- F3：扩散起草成本 $T_{\text{draft}}=t_{\text{parallel}}$，中等块大小下对 $\gamma$ 不敏感（$t_{\text{parallel}}\ll\gamma\cdot t_{\text{step}}$）。定位：§2.2 Eq.(3)。置信：已确认。
- F4：target 特征融合 $H_t=\mathrm{RMSNorm}(W_c[H^{(l_1)};\ldots;H^{(l_5)}])$。定位：A.3 第一式。置信：已确认。
- F5：第 $i$ 层 draft 的投影 $Q_i=W_i^Q H_d$，$K_i=[W_i^K H_t;\,W_i^K H_d]_{\mathrm{seq}}$，$V_i=[W_i^V H_t;\,W_i^V H_d]_{\mathrm{seq}}$。定位：A.3 第二式组。附注：target 特征只作 KV 条目，绕过 draft 的 Q 投影、输出投影、自注意更新和 FFN。置信：已确认。
- F6：块内位置 $k$ 的损失权重 $w_k=\exp(-(k-1)/\gamma)$。定位：§3.2 Eq.(4)。置信：已确认（$\gamma$ 为衰减率超参，非草稿长度）。

## N 数字

- N1：Transformers 后端、Qwen3-4B/8B、T=0：DFlash 平均 4.91×/4.86×，EAGLE-3(16) 1.81×/1.76×，EAGLE-3(60) 2.08×/2.02×；最高 6.09×（Math500，Q3-4B）。定位：Tab. 1。条件：thinking 关闭、最多 2048 新 token、块 16、greedy。置信：已确认。
- N2：同设置 T=1：DFlash 4.24×/4.03×，对 EAGLE-3(16) 提升 2.2×。定位：Tab. 1 下半 + §5.1。置信：已确认。
- N3：思考模式（thinking 开启）：加速约 4.5×（4B）与 3.9×（8B）。定位：§5.2（reasoning 表）。置信：已确认。
- N4：SGLang B200+FA4+Spec-v2：最高 5.1×（Qwen3-8B Math500 conc1）；Qwen3-4B Math500 conc32 2.9×；Qwen3-Coder-30B-A3B LCB conc1 2.6×；平均 $\tau$ 最高 8.09。定位：Tab. 2。置信：已确认。
- N5：vLLM、Qwen3.5-9B：conc1 为 4.0×/4.6×/3.0×（Math500/HumanEval/MT-Bench），conc32 为 1.9×/2.1×/1.3×。定位：A.4 Tab. 10。置信：已确认。
- N6：LLaMA-3.1-8B（同数据）：DFlash(10) GSM8K 2.4×/HumanEval 2.8×/Alpaca 2.2×（conc1），EAGLE-3(10) 1.6×/2.0×/1.5×。定位：Tab. 3。条件：SGLang Spec-v1、Flashinfer、B200。置信：已确认。
- N7：KV 注入 vs 输入融合消融（Qwen3-4B、5 层、块 8）：GSM8K $\tau$/加速 = KV 4.2/3.3× vs 输入融合 3.5/2.9×；EAGLE-3-5L（输入融合、自回归）4.2/2.1×；DFlash-AR（KV、自回归）4.8/2.4×。定位：Tab. 9（§5.4.4）。置信：已确认。
- N8：层数消融（块 16、5 特征层）：Math500 $\tau$ = 3L 5.64 / 5L 5.99 / 8L 6.33；加速 4.69×/4.71×/4.64×。定位：Tab. 6。置信：已确认。
- N9：训练配置：800K 样本（Nemotron Post-Training V2 + CodeAlpaca，response 由 target 生成）；6 epochs、AdamW、lr $6\times10^{-4}$、序列长 3072（Coder 4096）、每序列采 512 个 anchor；loss decay $\gamma$：块 16 取 7、块 10 取 5、块 8 取 4。定位：§5 + A.1。置信：已确认。
- N10：5 层草稿器生成 16 token 的延迟低于 EAGLE-3 单层生成 8 token（5 层并行一次前向 vs 单层 8 次串行）。定位：§2.2 + Fig. 4（draft_latency_bar）。置信：已确认。
- N11：无 target 特征的 5 层块扩散草稿：T=0 加速 GSM8K 2.83×、Math500 3.73×、AIME24 3.43×、AIME25 3.35×。定位：A.2 Tab. 5。置信：已确认。
- N12：KV 注入显存：$W_c$ 为 $D\times 5D$，Qwen3.5-35B-A3B $D=2048$、BF16 下约 42 MB（vs target 约 70 GB）；bs1、seq2048 时投影输入/输出约 40 MB/8 MB，块 16 解码时临时激活 <400 KB。定位：A.3。置信：已确认。
- N13：长上下文（Qwen3.5-27B 草稿器）：hotpotqa 16K 时 base 3.61、微调后 6.05；qasper 16K base 3.57、微调 6.00。定位：Tab. 4。条件：LongAlign-10K 1.6K 样本、3 epochs。置信：已确认。
- N14：更多模型（SGLang、B200、conc8，$\tau$/加速）：Qwen3.5-4B DFlash 7.1/3.0× vs MTP 6.5/1.5×（Math500）；Qwen3.5-27B 7.7/3.8×；GPT-OSS-120B 5.4/1.6×。定位：A.4 Tab. 9。置信：已确认。

## 原图候选

- Fig. 2（dflash_inference_design.pdf）：KV 注入推理设计——首选，对应第 2 章。
- Fig. 3（dflash_attn.pdf）：训练注意力掩码与 anchor 构块——对应第 3 章。
- Fig. 4（draft_latency_bar.pdf）：1/3/5 层 DFlash 与 1 层 EAGLE-3 起草成本对比——对应第 1 章。
- Fig. 1（dflash_speedup.pdf）：与 EAGLE-3 的加速对比——对应第 4 章。
获取途径：TeX 源码 figures/ 目录 PDF，转 PNG 后用 img_to_b64.py 内联。
