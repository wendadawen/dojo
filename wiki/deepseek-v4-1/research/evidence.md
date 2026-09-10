# DeepSeek-V4.1-Flash 核心论断与证据

来源代号：
- **R** = 技术报告 `DeepSeek_V41_Tech_Report.pdf`（仓库内，`research/official/tech_report.txt` 为 pdftotext -layout 提取的文本，行号即该文件行号）
- **CFG** = `official/inference/config.json`（推理代码实际读取的配置，`generate.py` 中 `ModelArgs(**json.load(f))`）
- **CFG-HF** = `official/config.json`（HF 仓库 transformers 格式配置，`text_config` 段）
- **SRC** = `official/inference/model.py` 行号
- **KRN** = `official/inference/kernel.py` 行号
- **HDR** = `ckpt/headers.json`（HTTP Range 读到的 48 个分片头，96085 个张量）
- **MEAS** = 本机实测脚本（`research/*.py` 与其 `ckpt/*.out` 存档）

## 核心论断（C）

### C1：主干 40 层 = 20 层编码器 + 20 层解码器，前两层只用 SWA
- 论断：语言主干 40 层，组织为 20 层因果编码器 + 20 层解码器；除前两层只有滑动窗口注意力外，每层同时有全局注意力与 SWA。
- 来源定位：R 行 314-319；CFG `compress_ratios` = `[0,0,2×18,1×20,0×3]`（前 2 层 ratio 0 = 纯 SWA）。
- 引文依据：R 行 314-319「Its language backbone comprises 40 causal Transformer layers, organized into a 20-layer causal encoder followed by a 20-layer decoder. Each layer incorporates both global attention and sliding window attention (SWA), except for the first two layers, which use SWA only.」
- 适用条件：指语言主干；MTP 层（40-42）ratio 0，不参与全局 KV。
- 置信状态：**已确认**（报告 + 配置 + MEAS `verify_csa2_modes.py` 层表一致）

### C2：CSA2 三模式与跨层共享
- 论断：每个 ratio>0 的层静态属于 Full / Reindex / Reuse 之一。Full 自算主 KV 与索引器 K 并产出新 Top-K；Reindex 复用最近的主 KV 与索引器 K、自算 Top-K；Reuse 复用主 KV 与最新 Top-K，不算索引分。主 KV 与索引器 K 按 ratio 组跨层共享。
- 来源定位：R §2.3.1（行 490-520）；SRC L496-503（`owns_k` / `is_candidate_source` / `uses_candidates`）、L653-661、L722-737。
- 引文依据：R 行 500-505「Full Mode. The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices.」；SRC L498-500 注释「the index keys are derived from the compressor's latent, so only a layer that compresses its own KV can produce them; every other indexer reads them from that layer's cache」。
- 适用条件：主 KV 共享要求同组 ratio 相同（CFG 中组内 ratio 一致）。
- 置信状态：**已确认**（报告 + 源码 + MEAS `verify_csa2_modes.py`：Full=[2,8,14,20]、Reindex=[24,28,32,36]、Reuse=30 层、SWA=[0,1]，且运行期每个压缩层读到的 cache 都属于组内 source 层）

### C3：主 KV 用 MXFP4，SWA KV 保持 FP8
- 论断：主 KV 量化到 E2M1（每 16 通道一个 E4M3 scale，无二级 global scale），在 RoPE 之后量化；SWA KV 保持 FP8；索引器 Q/K 也是 FP4。
- 来源定位：R §2.4.4（行 678-700）；SRC L760 `fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`、L707 `act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)`（窗口 KV）、L546/L552（索引器 K/Q）。
- 引文依据：R 行 691-694「we select E2M1 with one E4M3 scale per 16 channels, following NVFP4, but omitting its second-level global scale」；R 行 698-700「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」
- 适用条件：主 KV 指压缩后的全局 KV；窗口 KV 不压缩且为 FP8。
- 置信状态：**已确认**（报告 + 源码；HDR 中 `compressor.wkv.weight` 为 BF16 存储，代码 ratio>1 时提升 fp32）

### C4：Engram 插在层 1 与 14，每 token 查 24 行
- 论断：Engram 是 n-gram 哈希查表的条件记忆，只插在层 1 与层 14；阶数取 2/3/4，8 个哈希头，每 token 查 $(4-1)\times 8 = 24$ 行；查表结果经稠密投影写入残差流，门控由流与键的归一化点积决定。
- 来源定位：CFG `engram_layer_ids=[1,14]`、`engram_max_ngram_size=4`、`engram_n_heads=8`、`engram_head_dim=256`；SRC L328-365（Engram.forward）、`engram.py`；HDR 层分布实测 `engram 层: [1, 14]`。
- 引文依据：R 行 340「Engram (Cheng et al., 2026b) adds sparsely accessed conditional memory」；SRC L344 `n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads`。
- 置信状态：**已确认**（配置 + 源码 + MEAS `probe_engram_layout.py`）

### C5：DSpark 3 个 MTP 块、block size 5、目标层 37-39
- 论断：DSpark 是块式推测解码头，3 个 MTP 块，每块一次前向产出 5 个草稿位置，噪声 token id 128799，输入取自层 37/38/39 的注意力输入（按 hc 维取均值后拼接），带 Markov 头与置信头，路由 128 专家取 3。
- 来源定位：CFG `n_mtp_layers=3`、`dspark_block_size=5`、`dspark_target_layer_ids=[37,38,39]`、`dspark_n_routed_experts=128`、`dspark_n_activated_experts=3`；SRC L1100-1156。
- 引文依据：SRC L1131-1132 `draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)`；SRC L1265-1266 `if i in self.target_layer_ids: main_hiddens.append(h.mean(dim=2))`。
- 置信状态：**已确认**（配置 + 源码 + MEAS `run_mini.py` DSpark 段跑通：draft_ids 形状 $[1,6]$、logits $[1,5,V]$、confidence $[1,5]$，缩小配置 block size 3）

### C6：CED——解码器全局 KV 由 $H_{L/2}$ 投影
- 论断：全局注意力上，解码器层（$l>L/2$）的 KV 条目不来自本层隐状态，而由第 $L/2$ 层隐状态 $H_{L/2}$ 用逐层投影权重得到；与 CSA2 结合后，只有解码器中被指派 Full 模式的层（层 20）计算全局 KV，其余解码器层共用。
- 来源定位：R §2.2 行 386-397（Eq. 1）；R §2.3.1 末段行 536-540；SRC：层 20 是 Full 层，其压缩器输入即层 19 的输出。
- 引文依据：R 行 388-393「For global attention, CED treats the bottom L/2 layers of the Transformer as the causal encoder. For the upper half layers (i.e., the decoder, l > L/2), the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H_{L/2}, using layer-dependent projection weights」；R 行 537-540「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer, i.e. the last layer of the causal encoder.」
- 适用条件：全局 KV；SWA KV 仍逐层从本层隐状态计算。
- 置信状态：**已确认**（报告两处 + 源码结构）

### C7：分层稀疏索引器只在解码器使用，候选池 2048 块 × 8 位置
- 论断：解码器的第一个 Full 层为每个 query 打分全部因果可见的主 KV 位置，同时按块打分（块分 = 块内最大索引分）选出 2048 个块，块内 8 个位置进入候选池（16384 个候选位置）；后续 Reindex 层只在这个池内搜索 Top-512。
- 来源定位：R §2.3.2（行 542-570）；CFG `candidate_source_layer=20`、`candidate_topk_blocks=2048`、`candidate_block_size=8`；SRC L583-610 `select_candidate_blocks`。
- 引文依据：R 行 565-567「For example, selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions.」；SRC L604-605「the block with this query's newest position is only partly filled, so pin it in」。
- 适用条件：仅在解码器（$l \ge 20$）使用；编码器层不做候选限制。
- 置信状态：**已确认**（报告 + 配置 + MEAS `verify_csa2_modes.py`：建池层 20、池内搜索层 [24,28,32,36]）

### C8：MoE 路由用 sqrtsoftplus + noaux_tc 偏置
- 论断：384 路由专家 + 1 共享专家，每 token 激活 6 个路由专家；打分 $\sqrt{\mathrm{softplus}(xW^\top)}$；无辅助损失的修正偏置只参与选择、不缩放权重；权重在 top-6 内归一化后乘 route scale 1.5；图像 span 的 token 使用另一套偏置。
- 来源定位：CFG `n_routed_experts=384`、`n_activated_experts=6`、`score_func="sqrtsoftplus"`、`route_scale=1.5`、`norm_topk_prob=true`；SRC L809-827。
- 引文依据：SRC L821-823「the bias picks experts but does not scale them: weights come from the raw scores」；SRC L817 `scores = F.softplus(scores).sqrt()`。
- 置信状态：**已确认**（配置 + 源码 + MEAS `verify_gate_formula.py`：独立重实现 indices 完全一致、权重最大差 0；扰动 bias 后权重不被缩放）

### C9：Single-Pass mHC——4 份残差流、Sinkhorn 20 轮
- 论断：残差流按 hc_mult=4 复制；一次投影同时给出 pre/post/comb 三套系数；comb 经 softmax 后再做 19 轮行列交替归一化（合计行归一化 20 次、列归一化 20 次，末步在列方向），得到近似双随机矩阵。
- 来源定位：CFG `hc_mult=4`、`hc_sinkhorn_iters=20`；KRN L406-462；SRC L948-966。
- 引文依据：KRN L450-458（`for _ in T.serial(sinkhorn_iters - 1):` 内先 `reduce_sum(dim=1)` 再 `reduce_sum(dim=0)`）。
- 置信状态：**已确认**（源码 + MEAS `verify_sinkhorn.py`：独立 float64 重实现最大差 8.9e-08，行/列各 20 次、末步列方向，行和列和均为 $1-\epsilon$）

### C10：参考实现的 prefill/decode 在 ratio>1 的 kv_source 层上读取到别的层的索引键
- 论断：`shared_attn.index_k = self.k_cache` 的发布挂在 `if self.owns_k and latent is not None:` 分支内，而读取 `index_k = shared_attn.index_k[...]` 无条件执行；解码时 ratio>1 的层在组未填满的步不发布，于是读到上一次前向中最后发布者（真实配置下是 ratio=1 的层 20）的 cache。
- 来源定位：SRC L537-548 与 L554；CFG `kv_source_layer_ids=[2,8,14,20]`（层 2/8/14 ratio 2、层 20 ratio 1）。
- 引文依据：SRC L536-548（发布语句在 `if` 内）与 L554（读取语句在 `if` 外）；MEAS `ckpt/debug_consistency7.out`：偶数解码步 layer 2 读到的行 0 首 4 值 `[-1.0,-0.5,-1.0,+0.0]` = layer 4 的 k_cache 行 0，奇数步读到 `[-2.0,-0.75,-1.5,+0.25]` = 自己的。
- 适用条件：解码步中该层压缩组未填满（ratio>1 时每隔 ratio 步出现一次），且栈内存在更晚的 kv_source 层。
- 置信状态：**已确认**（本机实测三重证据；仅描述该参考实现的行为，不外推到服务实现）

### C11：参考实现不包含 prefill 半栈跳过与 SWA 有界重放
- 论断：`Transformer.forward` 对任何 prefill 都逐层跑满全部 40 层，代码中没有"只跑编码器"或"SWA 重放窗口"的分支；报告描述的 prefill 减半与 Bounded Replay 属服务实现。
- 来源定位：SRC L1261-1267（`for i, layer in enumerate(self.layers)` 无跳过条件）；`inference/` 目录无 replay 相关标识。
- 引文依据：SRC L1261-1267 循环体；R 行 404-410 描述 SWA replay 与 Decoder SWA Bounded Replay 属服务端行为。
- 置信状态：**已确认**（源码通读；此条是"报告 vs 参考实现"的边界说明）

## 核心公式（F）

### F1：压缩器池化与组位置
$$C_j = \mathrm{RMSNorm}\!\Big(\sum_{t=jr}^{(j+1)r-1} \mathrm{softmax}_{t'}\!\big(s_{t'}\big)\, v_t\Big),\qquad \text{RoPE 位置} = j\cdot r$$
- 符号：$r$ 为该层压缩比；$v_t = W_{kv} h_t$、$s_t = W_{gate} h_t$ 为逐 token 的 fp32 投影；softmax 在组内 $r$ 个 token 上取。
- 来源定位：SRC L458-485；MEAS `verify_compressor_pooling.py`（独立重实现逐位一致，最大差 0.000e+00）。
- 适用条件：$r>1$ 时门控与池化在 fp32；池化结果先转回 bf16 再 RMSNorm。$r=1$ 时退化为 $\mathrm{RMSNorm}(W_{kv}h)$，无门控。
- 置信状态：**已确认**

### F2：索引器打分
$$I_{q,j} = \sum_{h} w_{q,h}\cdot \mathrm{ReLU}\big(q_h \cdot k_j\big)$$
- 符号：$q_h$ 为第 $h$ 个索引头的 fp4 量化 query；$k_j$ 为压缩条目 $j$ 的 fp4 索引键；$w_{q,h}$ 由 $W_{proj}h_q$ 给出并乘常数 $\text{softmax\_scale}\cdot n_{\text{heads}}^{-1/2}$。
- 来源定位：SRC L550-557；MEAS `ckpt/debug_consistency4.out`（分数原值）。
- 置信状态：**已确认**

### F3：压缩条目可达性
$$n_{\text{reach}}(i) = \left\lfloor \frac{i+1}{r}\right\rfloor \ (\text{prefill}),\qquad \left\lfloor \frac{s+1}{r}\right\rfloor \ (\text{decode})$$
- 符号：$i$ 为 query 位置（0 起），$s$ 为解码步；超过可达数的压缩条目在打分阶段置 $-\infty$。
- 来源定位：SRC L563-567。
- 置信状态：**已确认**（prefill 与 decode 两条公式在 MEAS 中逐位置一致）

### F4：候选块打分
$$\text{block}(b) = \max_{j \in b} I_{q,j},\qquad \text{钉住最新块},\qquad \text{top-}2048$$
- 来源定位：SRC L596-610。
- 置信状态：**已确认**

### F5：MoE 路由
$$\text{score} = \sqrt{\mathrm{softplus}\big(xW^\top\big)},\quad \text{idx} = \mathrm{topk}_{6}\big(\text{score}+b\big),\quad w = \frac{\text{score}_{\text{idx}}}{\sum \text{score}_{\text{idx}} + 10^{-20}}\times 1.5$$
- 来源定位：SRC L811-826。
- 置信状态：**已确认**（MEAS 独立重实现一致）

### F6：mHC 系数
$$\text{pre} = \sigma(m s_0 + b) + \epsilon,\qquad \text{post} = 2\sigma(m s_1 + b),\qquad \text{comb} = \text{Sinkhorn}_{20}\big(\mathrm{softmax}(m s_2 + b) + \epsilon\big)$$
- 来源定位：KRN L426-460。
- 置信状态：**已确认**（MEAS 行/列各 20 次、末步列方向）

### F7：CED 全局 KV（报告 Eq. 1）
$$C_l = H_{L/2}W_l^{KV},\qquad Z_l = H_{L/2}W_l^{Z},\qquad l > L/2$$
- 来源定位：R 行 388-393。
- 适用条件：与 CSA2 结合后仅解码器 Full 层实现；参考实现中 $H_{L/2}$ 即层 20 的注意力输入。
- 置信状态：**已确认**

### F8：YaRN 频率
$$\text{freq}_i' = \frac{\text{freq}_i}{\text{factor}}\cdot \text{ramp}_i + \text{freq}_i\,(1-\text{ramp}_i),\qquad \text{ramp}_i = \mathrm{clamp}\!\Big(\frac{i-\text{low}}{\text{high}-\text{low}},0,1\Big)$$
- 符号：$\text{freq}_i = \text{base}^{-2i/d}$；$\text{low},\text{high}$ 由 $\text{corrected\_dim}(\text{rotations}) = d\ln\!\big(N_{\text{orig}}/(2\pi\,\text{rotations})\big)/(2\ln\text{base})$ 取 $\lfloor\cdot\rfloor$ / $\lceil\cdot\rceil$。
- 来源定位：SRC L369-389；MEAS `verify_rope_yarn.py`（独立重实现最大差 7.5e-09；base 160000 时 low=15、high=25）。
- 置信状态：**已确认**

### F9：稀疏注意力
$$o = \frac{\sum_{t\in\mathcal{T}} e^{\,q\cdot k_t/\sqrt{d}}\, v_t}{\sum_{t\in\mathcal{T}} e^{\,q\cdot k_t/\sqrt{d}} + e^{\,\text{sink}-m}},\qquad m = \max_t \frac{q\cdot k_t}{\sqrt{d}}$$
- 符号：$\mathcal{T}$ 为选中槽位集合；$m$ 只由选中槽位的分数取得，sink 只进分母；不可选槽位（索引 $-1$）分子分母都不贡献。
- 来源定位：KRN L310-403；MEAS `verify_sparse_attn_window.py`（与解析解逐元素差 0.00e+00）。
- 置信状态：**已确认**

## 外部数字与实验条件（N）

| 编号 | 数值 | 来源 | 实验/成立条件 |
|---|---|---|---|
| N1 | 全局 KV cache **890 B/token**；约 DeepSeek-V4-Flash 的 1/4 | R 行 18-19、行 133-140 | 1M 上下文、FP4 主 KV + 跨层共享 + 压缩比 2/1；MEAS `verify_cache_size.py` 按 CFG+源码常量推导得 720+170=890（精确一致） |
| N2 | 每 token 激活 **8B（prefill）/ 16B（decode）** | R 行 320-322 | MEAS `verify_active_params.py`：编码器半栈 7.8929B、解码器半栈 7.5758B（HDR 全量张量 + CFG 专家数） |
| N3 | **552B** backbone + **196B** Engram 参数 | R 行 320 | MEAS `count_params.py`：backbone 551.57B、engram 196.93B（HDR 96085 张量，fp4 按两值一字节折算） |
| N4 | 窗口 128、索引 Top-512、候选 2048×8=16384、索引头 32×128、KV 头 1×512 | CFG；R §2.3.2 | 与 HDR 形状一致 |
| N5 | 主 KV FP4 相对误差 9.5%（FP8 2.65%，3.58×）；可表示上界 448×6=2688 | R §2.4.4（行 691-700） | MEAS `verify_fp4_error.py`：512 维 RMSNorm 输出（L2 范数 22.6274 = √512）上测得；报告称训练观察最大幅值约 10 |
| N6 | 单层每 token 激活 377M（attn 126.6M / gate 1.97M / 共享 35.4M / top-6 路由 212.3M / mHC 0.98M） | MEAS `verify_active_params.py` | 用 HDR 真实形状 + CFG 专家数；层 20 因压缩器/索引器 attn 为 134.7M |
| N7 | 参考实现 prefill(16) vs prefill(8)+decode(8)：原始最大差 8.9e-3（layer 2 起、偶数位置）；修正后 bf16 残差 5.0e-3、fp32 1.5e-8~3e-8 | MEAS `ckpt/debug_consistency3~7.out` | 缩小模型（dim 64、7 层、保留全部结构特征），同一初始化；fp32 对照需给 engram 查表补 dtype 保持 |
| N8 | 压缩器池化两条路径：fp32 层面对照差 4.5e-6（相对 2.2e-6）；bf16 后 5/4096 个元素落在不同 ulp | MEAS `verify_compressor_pooling.py` | 真实维度（dim 5120、head 512）随机权重 |
| N9 | 约 **1/4**（vs DeepSeek-V4-Flash）、**1/437**（vs DeepSeek-V1）、持久化 **1/8** | R 行 35-38、行 139-140 | **登记为报告宣称**：缺少对方模型的 config 与缓存结构，无法独立推导，本页只陈述宣称本身 |

## 无法独立核对、需降级处理的条目

- N9 的三个倍数关系：只作"报告宣称"陈述，不给推导。
- 报告的准确率类结论（"no measurable decrease in accuracy"）：属训练实验结论，本页只陈述其依据（幅值上界余量 268.8×，N5），不声称已复核精度。

## 构造示例与辅助解释

- **构造示例**：$r=2$ 的压缩条目位置账（组 $j$ 覆盖 token $2j,2j+1$，RoPE 位置 $2j$）、890 B/token 的加总过程、单层激活参数的加总过程。全部标"构造示例"，数字来自 CFG 与 HDR 的真实取值。
- **辅助解释**：把索引器比作"先在块级别缩小搜索范围、再在位置级别选 Top-512"的两级筛选；边界：它不改变可见性规则，只影响"选得准不准"。

## 简化条件

- 字节账按"每个 source 层各存一份主 KV 与一份索引器 K"计算，不含运行时缓冲（窗口环形缓冲 128×512、候选池掩码、压缩器残态）；这些随 batch 与实现变化，不影响 per-token 主账。
- 激活参数按"任意 6 个路由专家"计算（同一层内 384 个专家形状相同），不反映具体 token 的实际选择。
- prefill/decode 一致性实测使用等比缩小模型（dim 64），验证的是计算规则而非特定权重输出。
