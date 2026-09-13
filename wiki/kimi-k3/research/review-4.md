<!-- review-meta
round: 4
page: wiki/kimi-k3/index.html
reviewed_content_sha256: 985061a0f2dbfffa
-->
# Kimi K3 审查记录（第 4 轮）

- 页面版本：4f573c8a591e1f6ce5b53f94423d198cf0f198c5
- 论文版本：arXiv:2607.24653v2（Kimi Team, Moonshot AI，2026-08-07）
- 审查时间：2026-09-13 19:43
- 审查者：独立子代理（未参与写作，也未参与前序轮次）
- 已完整阅读章节（含全部折叠块与图注）：1. 三维度信息流——K3 架构总览；2. 序列维度——KDA + 混合注意力；3. 深度维度——Block AttnRes；4. 宽度维度——Stable LatentMoE；5. 原生视觉——MoonViT-V2；6. 训练——数据、scaling law、Muon、长上下文扩展；7. 后训练——SFT→RL→MOPD 三阶段、QAT、EAGLE；8. 基础设施——3T 训练 + 1M RL + 推理；9. 性能与评价；10. 独立评价——系统性设计、开源里程碑与边界；来源与范围说明；另通读页首 核心问题 5 题与各章 本章问题 全部折叠解答
- 来源核对方式：arXiv:2607.24653v2 正文逐节定位（Fig.2/6/7、Eq.8–10/11/12/13/14/15/16、Table 1/2/3/5、§2.1–§2.5、§3.1–§3.4、§4.1、§5.1–§5.4、§6.1/§6.3/§6.4、附录 C/E、§8）；HuggingFace `moonshotai/Kimi-K3` 的 `config.json`、`model.safetensors.index.json`、`modeling_kimi_linear.py`、`modeling_kimi_k3.py` 官方源码实拉核对
- 机械验证：`python3 .dojo/scripts/validate.py wiki/kimi-k3/index.html` → `validation ok`；页内 15 个前置概念链接（kda/mla/block-attnres/stable-latent-moe/situ-glu/quantile-balancing/nope/moonvit-v2/per-head-muon/mopd/eagle-speculative/mxfp4-qat/flash-kda/moonep/gpu-execution-model）全部真实存在；`overview.html` 与 `index.html` 双向互链；无指向已移除 `research/` 的链接（页内唯一 “research” 出现在引文 "research-level reasoning remains a key direction for improvement"）；`dojo:topics=模型结构`、`dojo:tag=模型架构` 均在词表内

## 已核对通过的关键项（本轮回源确认一致）

- Table 1 全部行与原文逐字一致：61→93（↑52%）、1.04T→2.78T（↑167%）、32.6B→104.2B（↑220%）、Latent MoE 3584(0.5×)、MoE hidden/expert 2048→3072、routed 384→896（↑133%）、top-k 8→16、shared 1→2、heads 64→96、dense 1=1、vocab 160K=160K、训练上下文 128K→1M（8×）、MLA→Hybrid KDA–MLA、SwiGLU→SiTU-GLU、61 MLA→69 KDA+24 MLA、MTP 1=1、ViT 401M / 27 层；四行 Δ 列原文确为「–」，与页内注解一致
- config.json 实拉确认：`num_hidden_layers=93`、`num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`、`attn_res_block_size=12`、`max_position_embeddings=1048576`、`mla_use_nope=true`、`latent_moe_use_norm=true`、`activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`、`first_k_dense_replace=1`、`vt_num_hidden_layers=27`；`full_attn_layers` 恰为 24 个（4,8,…,92,93），`kda_layers` 恰为 69 个（页内列出的 69 个层号与 config 逐字一致）
- 权重清单核对：`layers.0.mlp` 为 dense（gate/up/down_proj）、`layers.0.self_attn` 为 KDA（k_conv1d/q_conv1d/v_conv1d/f_a/f_b/g_proj/o_norm），`layers.3.self_attn` 为 MLA（kv_a_proj_with_mqa/q_a_proj/kv_a_layernorm），支持页内“第 1 层 dense FFN＋KDA”“层号从 1 起算”的说法
- QAT 量化范围核对：索引中仅 `block_sparse_moe.experts.*.w1/w2/w3.weight_packed`+`weight_scale`（MXFP4）带量化标记（247,296 个 packed 全在 experts 下，占普通 `.weight` 之外的全部），attention/共享专家/routed_expert_down_proj/up_proj/norm/MoE gate(router)/lm_head/vision_tower/mm_projector/dense.mlp 均为全精度 `weight`——支持 [C11] 与 config `quantization_config.ignore`
- Table 2（编程/智能体/视觉与推理）页内 15 个数字组与原文逐格一致（DeepSWE 67.5/70.0/73.0/…、ProgramBench 77.8、Terminal-Bench 2.1 88.3/88.8、FrontierSWE 81.2/86.6、SWE-Marathon 42.0/35.0、BrowseComp 91.2/90.4、AutomationBench 30.8、GDPval-AA v2 1686/1747/1736/1593/1491/1510、JobBench 54.3/57.4、Harvey Lab-AA 94.6、GPQA Diamond 93.5/92.6/94.1、CritPt 23.4、HLE-Full 43.5/56.0 与 53.3/63.0、OmniDocBench 91.1、Math-Vision 94.3/97.8 与 94.8/98.6），加粗位与原文 best 一致
- Table 5 与 §6.3 核对：AA Intelligence Index v4.1 57.1 #4/580（原文“third if GPT-5.6 Sol effort variants are counted as a single entry”，页内括注为忠实翻译，非杜撰）；Vals Index 74.7 #2/39；WebDev Arena 1678 Elo #1/99 且原文确有 “the first open model to top this leaderboard”；Agent Arena 9.1 #4/37
- §6.4 成本效率：BrowseComp 91.2% @$2.03/任务＝GPT-5.6 Sol（90.4%）的一半；Kimi Code Bench 2.0 落后 Fable 5 4.0 分、成本 38%——逐字对应
- §6.1.3 评测配置：三种 harness（Kimi Code/Claude Code/Codex）、Terminal-Bench 报 best across harnesses、DeepSWE v1.1 且 mini-SWE-agent 下 67.3、SWE-Marathon H20-calibrated 分支（July 9 2026）且 Fable 5 35% 触发 fallback、BrowseComp 300K 触发 compaction 且 1M 无压缩 90.4%——逐条对应
- §2.2 引文："Standard residual connections … a bottleneck reminiscent of RNNs over time"、Eq.(8) q_l=w_l、Eq.(9) softmax kernel、Eq.(10) 块内 value matrix、N≈8 引 [60]、8 块×12 层＋末尾部分块＋计入 embedding 共 9 块——引文与数值对应
- §2.3.1/2.3.2/2.3.3：Eq.(11) 上投影前插 RMSNorm、Eq.(12) SiTU-GLU β1=4/β2=25 且 |f|≤β1β2=100、Eq.(13) router sigmoid、Eq.(14) QB 目标负载 q=mk/n、附录 C “requires no learning-rate-like hyperparameter … equilibrates within a few update steps even for nearly 10^3 experts”、固定步长 sign 更新原文确引 [27]=DeepSeek-v3 technical report——逐条对应
- §2.4/2.5：MoonViT-V2 从零训练、SigLIP 初始化 MoonViT-3D 梯度范数尖峰（Fig.6）、27 层约 0.4B、RMSNorm 去 bias、“contrastive pre-training is unnecessary … at scale”；Per-Head Muon 按头切分动量矩阵做 Newton–Schulz——逐条对应
- §3.3/3.4：cosine＋1% linear warmup＋weight decay 0.1、Per-Head Muon＋weight clipping＋QB、四阶段课程 8K→64K（预训练）/256K→1M（cooldown）、NoPE 无需 RoPE rescaling 或 YaRN、长上下文清洗（exact/fuzzy 去重、视频帧感知哈希、质量过滤、结构验证）与排列拼接合成——逐条对应
- §4.1：三阶段、9 专家（3 域×3 努力）、per-problem budget 超预算 -1、MOPD Eq.(15) stop-gradient+clip、QAT 从 SFT 起、EAGLE-3 draft 复用 MTP 层/7 步展开/融合第 1、4、最终 AttnRes block 输出/Eq.(16) LK loss——逐条对应
- §5：FlashKDA（CUTLASS chunkwise，优于 Triton 参考）、KCP（delta rule 的 M_t 使简单求和不足，分解为累积转移＋零起点局部状态，固定大小 all-gather＋prefix scan）、MoonEP（每 rank 恰 S×K token、E/R 冗余专家上界且基本紧、附录 E Theorem 1、零拷贝、静态形状免 host 同步）、AgentENV（Firecracker、checkpoint 133ms、6.5× 内存超分、51,219,741 沙箱/1,505,678 镜像）、512-token hash 粒度、WarpDecode——逐条对应
- 算术复算全部正确：2.78/1.04=2.67×、104.2/32.6=3.20×、896/16=56、384/8=48、896/384=2.33×、7×12+9=93、4×23+1=93、42.0−35.0=7、β1β2=4×25=100、3584=0.5×7168

## 问题

- [阻断·技术] index.html 行 122（核心问题 2 解答）、行 293（§3「代价与边界」）、行 314（§3 本章问题 3 解答）、行 295（§3「一个 token 在深度维度」段，另见行 121 折叠标题）：把 Block AttnRes 说成“块内层仍走标准残差、不能做跨层检索，只有进入新 block 的层才能对前序 block 的汇总表示和 embedding 做选择性检索、每 12 层一次跨块信息重置”。原文与官方源码均支持相反结论。｜引文依据：原文 §2 概览 “Along the depth dimension, Attention Residuals (AttnRes) [60] enable each module to selectively retrieve representations from the embedding, the current block, and preceding blocks, extending information access beyond conventional sequential residual accumulation (§2.2)”；§2.2 Eq.(10) 的 value matrix 为 V=[b_0,…,b_{n-1}]^T（i=1，块内第一层）/ V=[b_0,…,b_{n-1}, b_n^{i-1}]^T（i≥2，块内后续层），即块内每一层的候选都含全部前序块级表征 b_0…b_{n-1}；官方 `modeling_kimi_linear.py::KimiDecoderLayer._forward_attn_residual` 对每一层都执行 `_apply_attn_res(prefix_sum, block_residual, …)`，而 `_apply_attn_res` 内 `v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)`，故非块边界的块内层同样以全部块快照为检索候选；页面自身引用的前置概念页 wiki/block-attnres/index.html 亦写“候选集合为历史块快照加上当前块 partial sum（块的第一层除外）”“9 个候选是最后一个 block 内 i≥2 层的最大候选数（8 个块快照加当前流）”。｜修复要求：按 §2 与 Eq.(10) 改写“代价与边界”，删去“只有进入新 block 的层才能…检索”“块内层不能做跨层检索”“每 12 层一次跨块信息重置”及其在核心问题 2 解答、行 295 token 视角段、行 314 本章问题 3 解答中的对应表述；正确表述为——块内 12 层中每一层的输入都是对“embedding＋此前各 block 汇总表示＋当前 block partial sum”的 softmax 加权检索（i≥2 层才把当前块 partial sum 计入候选），Block AttnRes 相对 Full AttnRes 的代价是把候选粒度从层输出降为块级表征（丢失块内单层输出的可检索性），而非块内层无法检索前序块。｜修复：｜复验：

- [轻微·格式] index.html 行 700–706「公式与来源（F）」：F1–F8 共 8 条编号在正文中从未以 [F#] 形式被引用（正文只写 §/Eq. 号）；行 707–712「外部数字与实验条件」的 N5 亦未被正文引用（N1–N4 均有内联引用，如 §9.4 第三方评估段无 [N5]）。｜引文依据：不适用｜修复要求：或在正文相应位置补 [F#]（如 §2 讲 KDA 递归、§4 讲 SiTU-GLU/QB 处）与在 §9.4 补 [N5]，或把 F/N 列表显式标注为“参考索引，不在正文内联引用”，使 C（全部内联引用）与 F/N 的处理一致。｜修复：｜复验：

- [轻微·格式] index.html 行 57–64：页内 `<style>` 定义了 `.diagram` 类（等宽字体＋深色底，含 `font-family: var(--font-mono)`），但全页无任何元素使用 `class="diagram"`（结构图只用 `.flow-diagram`），属死 CSS；该等宽框线语义也易与“自绘结构图须为 HTML/内联 SVG”的规定混淆。｜引文依据：不适用｜修复要求：删除 `.diagram` 规则块。｜修复：｜复验：

- [轻微·可读性] §7（行 383、387、405 等）及全页首次出现的 SFT、RL、QAT 三个缩写未给出中文全称（supervised fine-tuning / reinforcement learning / quantization-aware training），MOPD 在正文已展开而这三个未展开；同一情形见标题「7. 后训练——SFT→RL→MOPD 三阶段、QAT、EAGLE」。｜引文依据：不适用｜修复要求：在 §7 首次出现处补全称，如“监督微调（SFT）”“强化学习（RL）”“量化感知训练（QAT）”。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 3
- 处置：修复（阻断项须关闭并复验后方可发布）。其余核对项（Table 1/2/5 数字、公式 Eq.8–16、config.json/源码三层一致性、§2.1–§5.4 机制描述、算术复算、表述维度、链接与 head 元数据、validate.py）本轮均未发现问题。