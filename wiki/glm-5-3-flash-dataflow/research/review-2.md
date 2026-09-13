<!-- review-meta
round: 2
page: wiki/glm-5-3-flash-dataflow/index.html
reviewed_content_sha256: 3825ea6640e47f73
-->
# GLM-5.3-Flash 前向数据流审查记录（第 2 轮）

- 页面版本：037124126159548972f2e6c522c9df59d4c44c77ba7689d69365821237f3d976
- 审查时间：2026-09-13 19:20–19:25
- 审查者：独立子代理（未参与写作与前序审查；首条消息只给页面路径、来源获取方式与规范路径）
- 页面类型：dataflow（本页 head 的 `dojo:type=dataflow`）；适用规范 guides/model-dataflow.md，分级与记录格式取 guides/concept/check.md 第 3、6 节
- 已完整阅读章节：1 关键规格 / 2 整体数据流（整体前向图、45 层类型排布图）/ 3 单层内部数据流（含代码折叠块）/ 4 mHC（系数产生图、退化段、hc_head 折叠块）/ 5 KDA 层（递推公式、遗忘门取值范围、prefill/decode）/ 6 DSA 层（k-pool 表、缩小预算折叠块、打分公式、跨层共享）/ 7 MoE 路由（路由公式、每 token 激活表）/ 8 位置信息 / 9 长上下文收益 / 10 FP8 量化 / 11 多模态 / 12 MTP / 13 核对方式 / 来源与范围说明（C、F、N、简化条件、范围之外）

## 核对材料

- 官方 `config.json`（`zai-org/GLM-5.3-Flash` main，69,416 字节）：text_config / vision_config / quantization_config / 各 token id
- 官方 checkpoint 头：`model.safetensors.index.json`（`total_size`、76,108 项、37,338 个 `weight_scale_inv`、62 分片），以及分片 1/2/18/32/49 的 safetensors JSON 头（张量形状与 dtype）
- 官方源码：`huggingface/transformers` main 的 `modeling_glm5_next.py`、`configuration_glm5_next.py`、`cache_utils.py`
- 官方 README；技术报告 arXiv:2602.15763 v2 全文（HTML）
- 页面内链（residual-connection、kda、delta-rule、kv-cache、low-rank-projection、aux-loss-free-routing、swiglu、quantization-basics、speculative-decoding、vit）均存在
- 机械项：`.dojo/scripts/validate.py wiki/glm-5-3-flash-dataflow/index.html` 返回成功

已核对通过（未在此列出的条目按已核对论）：config 的 45 层 / 34 KDA + 11 DSA（`full_attn_layers=[3,7,…,43]`）/ 288 专家 top-8 + 1 共享 / 前 3 层 dense / `hidden_size=4096` / `hc_mult=4` / `hc_sinkhorn_iters=20` / `gate_lower_bound=-5.0` / 64 头×128 维 conv k=4 / MLA q 1536 kv 512 / `qk_rope_head_dim=0` / `index_topk=2048`、`index_kpool=4` / `max_position_embeddings=1048576` / vision depth 24 宽 1024 patch 14 merge 2 / FP8 E4M3 128×128 与 `modules_to_not_convert` / `image_token_id=154854`、`video_token_id=154855`；总参数 321,323,031,390 与激活 17,376,348,990 两项均按 config 结构独立复算得到完全相同的整数；每 token 激活表六项之和等于合计、各占比与合计自洽；Sinkhorn 迭代次序（先列归一化、再 `iters-1` 轮行+列，故列 20 次行 19 次、末步列向）与 `Glm5NextTextHyperConnection.forward` 一致；`fn=[24,16384]`、`base=[24]`、`scale=[3]`（checkpoint 实测形状一致）；KDA 递推式与 `recurrent_kimi_delta_attention` 逐项对应；遗忘门 safe 分支公式与 `Glm5NextTextForgetGate.forward` 一致；indexer 打分式与 `Glm5NextTextIndexer.forward` 一致（`softmax_scale=head_dim**-0.5`、`weights_proj* n_heads**-0.5`、ReLU）；k-pool 换算式、`2048+3=2051` 宽度、MoE `noaux_tc` 公式与 `Glm5NextTextTopkRouter.forward` 一致；`_keep_in_fp32_modules_strict` 四项与 `_keys_to_ignore_on_load_unexpected` 含 `layers\.45\.` 逐字一致；`mla_use_nope` 等 9 个惰性键在两份源码中命中数均为 0；长上下文表四行与交叉点 2855 / 4288 均可复算。

## 问题

- [重要·来源] §6「跨层共享的能力存在但未启用」第 784 行、§13 第 985 行、§2 第 316 行与 [C1]（第 1008 行）：`checkpoint 里恰好有 11 组 indexer 张量` 与 `11 个层带 self_attn.kv_a_proj_with_mqa.weight` 与 checkpoint 实际不符，且与同页 §12 自相矛盾（§12 明写第 45 层「带 `kv_a_proj_with_mqa` 与完整 indexer」）。｜引文依据：`model.safetensors.index.json` 中含 `.indexer.` 的层号 = 3,7,11,15,19,23,27,31,35,39,43,**45**（共 12 组，每组 7 个张量共 84 个）；含 `self_attn.kv_a_proj_with_mqa.weight` 的层号同为这 12 个（第 45 层为张量头 `[512,4096] F8_E4M3`）。｜修复要求：把计数口径限定为主干 45 层（如「主干 45 层里恰好有 11 组 indexer 张量」），或把数字改为 12 并注明含 MTP 层第 45 层；三处（§2、§6、§13、[C1]）须一致。｜修复：｜复验：

- [重要·来源] §10 第 936 行（并牵连第 929 行）：`checkpoint 实测这些张量的 dtype 确实全是 F32，与声明一致` 对 `conv1d` 不成立。｜引文依据：分片 02/18/49 的 safetensors 头中 `layers.0/1/20/40.self_attn.{q,k,v}_conv1d.weight` 形状 `[8192,1,4]`、dtype 均为 **BF16**；同批核对中 `A_log`、`dt_bias`、`e_score_correction_bias`、`hc_*_base`、`hc_*_scale` 确为 F32（与页面其余 dtype 陈述一致），说明本页 dtype 类结论取自张量头。页面对 mHC 的 `fn`(BF16)/`base`/`scale`(F32) 与 gate(BF16) 的陈述均与张量头吻合，唯独 conv1d 例外。｜修复要求：改为「`A_log`/`dt_bias`/`e_score_correction_bias` 在 checkpoint 中为 F32；`conv1d`（checkpoint 中拆为 q/k/v_conv1d）在磁盘上为 BF16，按 `_keep_in_fp32_modules_strict` 在加载时升为 FP32」，并同步修正第 929 行「卷积核…更进一步留在 FP32」的措辞。｜修复：｜复验：

- [重要·来源] 「范围之外」第 1052 行：`注意力为 MLA 加 DSA，全文未出现线性注意力或 mHC` 中「全文未出现线性注意力」与所引报告不符。｜引文依据：arXiv:2602.15763v2 §2.1.2「Ablation Study of Efficient Attention Variants」原文含 `Linear attention variants such as GDN further improve quality but at the cost of additional parameters; SimpleGDN strikes the best balance by maximally reusing pre-trained weights.`（全文 HTML 中 `linear attention` 命中 3 次，`GDN` 9 次；`mHC`/`Manifold` 命中 0 次）。｜修复要求：删去「线性注意力」或限定为「GLM-5 的注意力为 MLA+DSA，未把线性注意力用作主干机制；mHC 未出现」；不得保留「全文未出现线性注意力」这一与来源相反的原句。｜修复：｜复验：

- [轻微·技术] §13 第 985 行：`（每个约 260 KB）` 与实测不符。｜引文依据：各分片头长度实测 分片1=93,512 B、分片2=160,856 B、分片18=173,760 B、分片32=177,792 B、分片49=173,800 B；按 76,108 项 / 62 分片、每项约 140 B 估算均值约 170 KB。｜修复要求：改为「每个约 90–180 KB（均值约 170 KB）」。｜修复：｜复验：

- [轻微·表述] §7 第 871 行：`反过来看总量：…` 属轻度元话语式过渡（与 guides/concept/check.md 2.2 第 12 条「元话语」同类）。｜引文依据：不适用。｜修复要求：改为直接陈述该口径，如「按总量口径：321.32 B 里…」，去掉「反过来看」这类引导语。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 2
- 处置：修复（三条重要问题须在下一轮前关闭；本轮未发现核心结论错误、未发现算式与合计不符，正文引用实测结论处已统一改写为「实测得到」且不再指向已移除的 `research/` 文件路径，`research/measured.md` 存在，`validate.py` 通过）
- 备注：审查期间页面工作树由 4 处修订更新（§4 退化段重写、§5 符号 $S\to L$、§9/§11/§13 表述清理、[N]/[C] 条目去文件路径）；本轮结论基于上列哈希的版本，核对以该版本为准。
