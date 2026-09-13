<!-- review-meta
round: 5
page: wiki/qwen3-5-dataflow/index.html
reviewed_content_sha256: 165f0d6d13686c9e
-->
# Qwen3.5-397B-A17B 前向数据流审查记录（第 5 轮）

- 页面版本：9bdc7ebc72092c6063a1c1e8f6b726a99cbfba36（工作树）
- 审查时间：2026-09-13 21:23
- 审查者：编排者派发的独立审查者（独立上下文子代理，未参与写作与前序轮次）
- 已完整阅读章节：1. 关键规格（含「家族变体」表）；2. 交互式数据流（含七张 noscript 视图表与全部「边：」图注）；3. 要点（整体结构 / GDN / 全注意力层 / MoE 路由 / MTP 草稿层 / 长上下文开销）；4. 视觉编码器与多模态融合（4.1–4.8）；5. 与 Qwen3.8-Flash-Next 的架构对比；6. 核对方式；来源与范围说明；并逐字读了页内 `<script>` 中七份 VIEWS 数据（节点 io/f/d 文本即提示框正文）与页脚脚本。

## 本轮回源与复算（结论：页面绝大部分数字逐项成立）

- 官方 `config.json`（huggingface.co/Qwen/Qwen3.5-397B-A17B，raw 全文）：hidden_size 4096、num_hidden_layers 60、num_experts 512、num_experts_per_tok 10、moe_intermediate_size 1024、shared_expert_intermediate_size 1024、num_attention_heads 32、num_key_value_heads 2、head_dim 256、linear_num_key_heads 16、linear_num_value_heads 64、linear_key/value_head_dim 128、linear_conv_kernel_dim 4、full_attention_interval 4、attn_output_gate true、mrope_section [11,11,10]、rope_theta 1e7、partial_rotary_factor 0.25、vocab_size 248320、max_position_embeddings 262144、router_aux_loss_coef 0.001、mamba_ssm_dtype float32、mtp_num_hidden_layers 1、mtp_use_dedicated_embeddings false、image_token_id 248056 / video_token_id 248057 / vision_start 248053 / vision_end 248054、vision_config{depth 27, hidden 1152, num_heads 16, intermediate_size 4304, out_hidden_size 4096, num_position_embeddings 2304, spatial_merge_size 2, temporal_patch_size 2, deepstack_visual_indexes []}。layer_types 实为 linear,linear,linear,full 循环，与页面「全注意力位于 0 起层号 3,7,…,59」一致。全部与页面所写相符。
- `model.safetensors.index.json`：metadata total_size=806,795,875,168；2924 个张量、94 个分片；`mtp.*` 1553 个；`.mlp.gate.weight` 61 个（60 主干 + 1 MTP）；含 `.linear_attn.` 的层集 45 个、主干含 `.self_attn.` 的层集恰 {3,7,11,…,59} 15 个；无 indexer 张量、无 deepstack 张量；visual.blocks 27 个；MTP 专家为 512×3=1536 个逐张量。均与页面相符。
- 以 HTTP Range 读取全部分片 JSON 头：90 个非 BF16 张量全为 F32，且恰为本代 45 层 `A_log`[64] 与 `linear_attn.norm.weight`[128]；`dt_bias` 为 BF16[64]。总量关系 806,795,875,168 = 2×P + 2×8640 → P = 403,397,928,944 精确成立。逐张量形状核对：q_proj[16384,4096]、k_proj/v_proj[512,4096]、q_norm/k_norm[256]、in_proj_qkv[12288,4096]、in_proj_z[8192,4096]、conv1d[12288,1,4]（无 bias）、out_proj[4096,8192]、shared_expert_gate[1,4096]、experts 打包 3D、mtp.fc、mtp.pre_fc_norm_embedding/_hidden、visual.patch_embed.proj[1152,3,2,16,16]+bias、pos_embed[2304,1152]、merger.norm.weight[1152]、merger.linear_fc1.bias[4608]、attn.qkv[3456,1152]、mlp.linear_fc1.bias[4304]、LayerNorm 带 bias——全部相符。
- 复算全部成立：403,397,928,944 = 396,346,350,336 + 456,010,480 + 6,595,568,128；激活 16,331,922,176 = 45×118,022,400 + 15×104,866,304 + 60×140,513,280 + 4,096 + 1,017,118,720，含查表 17,349,040,896 = +1,017,118,720（词表查表一行 248320×4096）；MTP 6,595,568,128 与视觉分组 456,010,480（MLP 267,890,544 / attn 143,451,648 / merger 40,119,040 / pos_embed 2,654,208 / patch_embed 1,770,624 / LN 124,416）、merger 40,119,040、每层 MoE 激活 140,513,280、512 专家 6,442,450,944、60 层 386,547,056,640、2.18%、95.82%、KV 2048 B/token/layer 与 0.938/7.500/30.000 GiB、GDN 4.00 MiB×45+72 KiB×45=0.179 GiB、视觉 token 数 196/784/1,764/1,980、位置推进 max(h,w)/2 的 14/42/24、mrope 最低频 1.66e-7、辅助损失均匀下界 0.010000 全部复算相符。
- 「合计 366,184 字节」实测：94 个分片头 JSON 长度之和 365,432 + 94×8 字节前缀 = 366,184，精确相符。
- 外部源码：transformers commit 36deb0b5 下 `src/transformers/models/qwen3_5/modeling_qwen3_5.py`、`qwen3_5_moe/`、`qwen3_vl_moe/` 路径均 HTTP 200 存在；vLLM `vllm/model_executor/models/qwen3_next_mtp.py` 第 73 行注释「mirroring the Qwen3.5 MTP handling (PR #38832)」、第 133 行 `torch.cat([inputs_embeds, hidden_states], dim=-1)`（embedding 在前）、两路 `pre_fc_norm_embedding`/`pre_fc_norm_hidden`，与页面 MTP 拼接描述相符。
- 家族五型号 config 实拉全部相符：122B-A10B(48/3072/32-2/64/256 top-8/I=1024)、35B-A3B(40/2048/16-2/32/256 top-8/I=512)、27B(64/5120/24-4/48/dense 17408)、9B(32/4096/16-4/32/dense 12288)、4B(32/2560/16-4/32/dense 9216/tie_word_embeddings=true)。
- Qwen3.8-Flash-Next：config 实拉 48/2560/24-2/48 v-heads/512 top-10/I=640，模型卡 I=640、125B+51B+4B、license qwen-community-1.0，与第 5 节对比行相符。
- `.dojo/scripts/validate.py wiki/qwen3-5-dataflow/index.html` 返回 `validation ok`。

## 问题

- [重要·技术] 5. 与 Qwen3.8-Flash-Next 的架构对比（第 371 行「发布」行）：对比列把 Qwen3.8-Flash-Next 的发布日期写作 `2026-08-24`，与可查来源不符，且该对比列在「来源与范围说明」中没有任何出处标注。｜引文依据：官方模型卡 README（huggingface.co/Qwen/Qwen3.8-Flash-Next）对日期只给到 `month = {August}, year = {2026}`，无日；llm-stats「Qwen3.8-Flash-Next was released on August 26, 2026」；datalearner「Release Date: August 26, 2026」。页面同页 Qwen3.5 侧日期为 `2026-02-16`，而该侧模型卡同样只给 `month = {February}, year = {2026}`，说明该列两个日期都需落到可定位的出处处。｜修复要求：为「2026-08-24」补上可定位来源并在来源表登记该列依据；若来源实为 8 月 26 日，则据实改写，或把日期收敛到来源支持的形式（如「2026-08 下旬」）。｜修复：｜复验：
- [轻微·技术] head `dojo:summary`（第 7 行）与开篇段落（第 110 行）：分项按两位小数列出「396.35B + 0.46B + 6.60B」，三项相加为 403.41B，与同句所标合计 403.40B 不等。｜引文依据：精确值 396,346,350,336 + 456,010,480 + 6,595,568,128 = 403,397,928,944（403.398B），三段独立四舍五入（396.346→396.35、0.456→0.46、6.596→6.60）后相加得 403.41。｜修复要求：使分项与合计在展示精度上自洽——例如分项改用精确值（396.346B + 0.456B + 6.596B）或合计随分项进位写 403.41B。｜修复：｜复验：
- [轻微·可读性] 第 254 行（noscript 视图表）与第 767 行（fusion 视图 `ids` 节点说明）：`视频用时间戳逐帧分隔：t1 vs 帧1 ve t2 vs 帧2 …` 中「vs」「ve」无法从字面理解为 vision_start / vision_end。｜引文依据：同页 4.7（第 351 行）写作 `<t1> <vision_start> 帧1 <vision_end> <t2> …`，两处写法不一致。｜修复要求：把该说明改为完整写法（`<t1> <vision_start> 帧1 <vision_end> <t2> …`）或与 4.7 节统一，使单看工具提示也能读懂。｜修复：｜复验：
- [轻微·格式] 正文与表格中直接把 Unicode 数学符号当运算符使用：第 125 行「32 查询头 × 256 维」、第 126 行「= 2×(32×256)」、第 184 行「12288 通道 × 3 位置 × 2 字节」、第 189 行「递归状态 · 64 × 128 × 128」、第 318 行「64 头 × 128 × 128 … 12288×3」，以及 noscript 各视图标签里的 `→`（如第 174 行「lm_head · 4096 → 248320」）。｜引文依据：`guides/concept/style-guide.md`「页面任何位置出现的数学变量、希腊字母、上下标、数学运算符和关系符都必须包在 `$...$` 或 `$$...$$` 中…禁止直接使用 Unicode 数学字符替代」；本页 head 的 `dojo:summary` 同类位置已用 `$...$` 写法（对照 wiki/qwen3-8-flash-next-dataflow 的 `$2\times2$`）。注：同族 dataflow 页正文也存在同样写法，若判定为可接受的既有惯例，可在此条注明接受理由。｜修复要求：把作为乘法/关系运算符的 `×`、`→` 改写为 `$\times$`、`$\to$`，或改写为不含运算符的中文措辞（如「12288 通道、3 个位置」）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（事实与来源、数字自洽、表述三处轻微项一并处理；均不改变页面核心结论与数据流）
