<!-- review-meta
round: 4
page: wiki/deepseek-v4-dataflow/index.html
reviewed_content_sha256: 00fd464eb426360b
-->
# DeepSeek-V4-Pro 前向数据流 审查记录（第 4 轮）

- 页面版本：d596b3d5e6575de1461f71a0bddff1bb827a6086
- 审查时间：2026-09-13 20:13
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：1 关键规格；2 交互式数据流（整体总览、CSA、HCA、压缩器、Indexer、MoE、mHC、MTP 八个视图的全部节点、边与悬停说明）；3 要点（整体结构、CSA、HCA、两种层共有的四个细节、mHC、MoE 路由、长上下文效率的来源）；4 本机实测与核对；来源与范围说明

## 来源获取

- 官方 config.json（huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json）
- 官方 inference/model.py、inference/kernel.py（raw 路径，逐行核对执行路径与算子顺序）
- model.safetensors.index.json 与 model-00001/00002/00004/00063/00064 分片的 safetensors 文件头（HTTP Range 读取，核对张量名、dtype、形状）
- 技术报告 arXiv:2606.19348v1 全文 HTML：摘要、2.2、2.3.1–2.3.4

## 已核对无误（关键项）

- config.json 与规格表逐条相符：61 层 + 1 MTP、hidden 7168、heads/KV 128/1、head_dim 512 / qk_rope_head_dim 64、sliding_window 128、index_topk 1024 / index_n_heads 64 / index_head_dim 128、n_routed_experts 384 / top 6 / shared 1 / moe_intermediate_size 3072、scoring_func sqrtsoftplus / routed_scaling_factor 2.5、num_hash_layers 3、hc_mult 4 / hc_sinkhorn_iters 20、max_position_embeddings 1048576 / YaRN factor 16 / 原生 65536、expert_dtype fp4、quantization fp8 e4m3、q_lora_rank 1536、o_groups 16 / o_lora_rank 1024、rope_theta 10000 / compress_rope_theta 160000、swiglu_limit 10.0、norm_topk_prob true、hc_eps 1e-6、vocab 129280、tie_word_embeddings false。
- compress_ratios 共 62 项，前 12 项 [128,128,4,128,4,128,4,128,4,128,4,128]，末项 0；据此 HCA 31 / CSA 30。checkpoint 索引：30 层含 attn.indexer.*（layers 2..60 偶数层）；3 层含 layers.{0,1,2}.ffn.gate.tid2eid（I64 [129280,6]）；gate.bias 共 59 个 = 主干 58 个（layers 3..60，缺 0/1/2）+ mtp.0 一个。
- 张量头复核全部相符：compressor.wkv HCA BF16 [512,7168] / CSA BF16 [1024,7168]；indexer.compressor.wkv BF16 [256,7168]；compressor.ape HCA F32 [128,512] / CSA F32 [4,1024]；attn_sink F32 [128]；wo_a F8_E4M3 [16384,4096]；wo_b F8_E4M3 [7168,16384]；wq_a F8_E4M3 [1536,7168]；wq_b F8_E4M3 [65536,1536]；wkv F8_E4M3 [512,7168]；indexer.wq_b F8_E4M3 [8192,1536]；indexer.weights_proj BF16 [64,7168]；路由专家 w1 I8 [3072,3584] + scale F8_E8M0 [3072,224]；共享专家 w1 F8_E4M3 [3072,7168]；hc_attn_fn F32 [24,28672]；mtp.0.hc_head_fn F32 [4,28672]；mtp.0.e_proj / h_proj F8_E4M3 [7168,7168]；embed / head BF16 [129280,7168]；norm BF16 [7168]。
- 算术独立复算：注意力参数按各层张量逐项加总得 19.465B（30×CSA 331.29M + 31×HCA 307.30M），与页面 19.47B 相符；路由专家 61×384×66.06M=1547.40B；共享专家 4.03B；mHC 61×2×688128=0.084B；每 token 激活 61×6×66.06M=24.18B，加非专家 24.68B = 48.86B ≈ 官方 49B；整表合计经 MTP 内部构成复算 = 1598.84B。KV 4.355 GiB / 244 GiB（BF16 GQA8 head_dim 128，K+V 计）= 1.79%；每条目 576/1024 = 56.25%；HCA 候选 4K 160 / 64K 640 / 1M 8320；Indexer KV 1.875 GiB、FP4 折算 0.469 GiB。
- 报告回源：摘要「27% of single-token inference FLOPs … compared with DeepSeek-V3.2」；2.3.4「approximately 2% times of that baseline … BF16 GQA8 … with a head dimension of 128」「BF16 … RoPE dimensions, while FP8 … remaining dimensions … reduces the KV cache size by nearly half」「attention computation within the lightning indexer is performed in FP4 precision」；2.2 式(6)(7) A_l=σ(Ã_l)、C_l=2σ(C̃_l) 与「B_l∈M」双随机流形；2.3.3「avoids exploding attention logits」；4.2.1「We set the number of Transformer layers to 61 … For the first two layers, we use HCA」；kernel.py:415-423、model.py:342/344/367/418-421/427/498/506/514/534/571-583/660-686/805 与页面描述逐条相符。引用位置（2.3.3、2.3.4、Eq.6/Eq.7）定位正确。
- 页面链接 ../hyper-connections/index.html、../deepseek-moe/index.html 均存在；正文所指 research/measured.md 存在；所引官方文件 inference/model.py、inference/kernel.py、inference/config.json 在 HF 仓库中均存在；`.dojo/scripts/validate.py` 返回 validation ok；页面无声称可运行的代码块（正文不含代码，仅公式），无需执行核对。

## 问题

- [阻断·技术] 第 2 节「mHC 内部」视图 sink 节点（index.html 第 616 行）：节点公式与同节点说明、以及第 3 节「mHC」要点（第 184 行）自相矛盾。node 的 f 写 `M^(t) = T_r(T_c(M^(t-1)))`，按复合顺序最外层是行归一化 T_r，据此结论应是「行和精确为 1」；而同一节点 d 写「softmax(-1)+ε → 列归一化 → 19 轮(行归一化→列归一化)」、要点写「最后一步是列归一化，因此列和精确为 1、行和只是近似」。｜引文依据：kernel.py:415-423 `for _ in T.serial(sinkhorn_iters - 1):` 体内先 `T.reduce_sum(comb_frag, row_sum, dim=1)` 除 row_sum、再 `T.reduce_sum(comb_frag, col_sum, dim=0)` 除 col_sum，每轮最后一步是列归一化；报告 2.2 式(8) `M^(t) = 𝒯_r(𝒯_c(M^(t-1)))`，原文「where 𝒯_r and 𝒯_c denote row and column normalization, respectively」——可见 f 抄的是报告式(8)（最后一次为行归一化），与页面自述的官方 kernel 顺序相反，且与「列和精确为 1」的结论不符。｜修复要求：将 f 改为 kernel 的实际顺序 `M^(t) = T_c(T_r(M^(t-1)))`；或保留报告式(8)但在 f 旁标注其为报告写法并指出与 kernel 顺序的差别（与第 184、243 行口径一致）。｜修复：｜复验：

- [重要·技术] 第 2 节「CSA 层」视图上下文长度自相矛盾：压缩器按 seqlen=512 计（comp 节点 io `[1,512,7168] -> [1,128,512]`，128=512/4），Indexer 侧却按上下文 ≥4096 计（idx 节点 io `... -> [1,512,1024]`、catidx `[1,512,128] ++ [1,512,1024] -> [1,512,1152]`、attn 节点 `idx[1,512,1152]`），而 attn 节点同时把 kv 写成 `kv[1,640,512]`。seqlen=512 时 topk 取 min(index_topk, seqlen/4)=128，idx 宽度应为 128+128=256；要得到 1152 需上下文 ≥4096，此时 kv 也不可能是 640。即同一节点内 1152 个索引无法从 640 条 KV 中 gather，与本节实测表 ratio=4 行 `(1, 512, 144)` 的口径也不同。｜引文依据：model.py:427 `topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]`；页面第 210 行实测表「1｜4｜(1, 640, 128)｜(1, 512, 144)｜144 = 128 + min(16, 512/4)」；第 213-216 行 HCA 视图则全用 seqlen=512 口径（516、132），两视图口径不一致。｜修复要求：把 CSA 视图统一到同一上下文——要么按 seqlen=512 写（comp 128 条、idx/catidx/attn 索引宽 256），要么按长上下文写（压缩条目数与 kv 长度同步放大，idx 宽 1152），并在图注中写明该视图取哪个上下文。｜修复：｜复验：

- [轻微·技术] 第 4 节参数量表分项显示值之和与合计不符：1547.40+19.47+4.03+0.93+0.93+0.17+0.08+25.84 = 1598.85，合计栏写 1598.84；占比列 96.8+1.2+0.3+0.1+0.1+0.0+0.0+1.6 = 100.1%。经独立复算，真实合计确为 1598.836B（≈1598.84），差异纯属两位小数显示取整，不影响结论。｜引文依据：第 220-230 行表格。｜修复要求：统一取整口径（例如分项多留一位小数，或合计写 1598.8B），使分项可按显示值相加。｜修复：｜复验：

- [轻微·技术] MTP 行标签漏列共享专家：第 228 行「MTP（1 层：路由专家 + 注意力 + 融合投影 + mHC）」计 25.84B，但按 MTP 张量清单复算（路由专家 384×66.06M + 注意力 0.300B + e_proj/h_proj 0.103B + mHC 0.0015B + gate 0.0028B）只有 25.774B，差额 0.066B 恰为 MTP 共享专家一份（66.06M）；含共享专家时正好 25.840B，合计也对齐 1598.84B。｜引文依据：model.py:624-627 MoE 恒含 shared_experts；checkpoint 中 mtp.0 分片同时存在 ffn.experts.* 与 ffn.shared_experts.*。｜修复要求：MTP 行标签补上「共享专家」，或注明该行含共享专家。｜修复：｜复验：

- [轻微·技术] HCA 视图 attn 节点公式漏 attention sink 项：f 写 `softmax(qK^T/√d)V`，而该节点 label 为「sparse_attn MQA + sink」、d 亦写「MQA + sink」，且 CSA 视图同名节点 f 含 `+Exp(z_h')` 分母项；两视图描述的是同一个 sparse_attn 调用。｜引文依据：kernel.py:346 `sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`；页面第 494 行 CSA attn 节点 f 含 sink 项。｜修复要求：HCA 视图 attn 节点 f 补上 sink 项或注明此处省略。｜修复：｜复验：

- [轻微·可读性] 实测指标未给度量定义：第 152 行「相对差异 0.76」、第 174 行「输出变化 1.57」未说明是何种范数、相对什么量，读者无法判断量级含义（同段其他实测数字如 2.4e-7、0.83、1.1e-6 均可解读）。｜引文依据：不适用。｜修复要求：补一句度量定义（如「压缩条目与均值池化输出的 RMS 相对差」）。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：修复