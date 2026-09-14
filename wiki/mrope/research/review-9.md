<!-- review-meta
round: 9
page: wiki/mrope/index.html
reviewed_content_sha256: 7ef9dd91897ddfe6
-->
# MRoPE审查记录（第 9 轮）

- 页面版本：bccf703c77b6fd6fd1060a4205361495689dfe21（wiki/mrope/index.html 工作树哈希）
- 审查时间：2026-09-14 17:46
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何文件）
- 已完整阅读章节：开头（标题 / 主要依据 / 引言）；核心问题；常见误解；1. 文本有序、图像有格——一维位置轴装不下多模态；2. 位置 id 变三元组——三种模态的分配规则（含「代码：构造序列的三维位置 id 与推进量」折叠块）；3. 一个头维装三条轴——分段与交错两种槽位排布（含「补充：为什么 32 个槽位对应 64 个旋转维」折叠块）；4. 位置轴上的省账——推进量与长序列外推；来源与范围说明
- 核对所用来源版本：
  - arXiv:2409.12191**v2**（HTML 全文，逐节抓取 §1 / §2.1 / §3.3.2 与 Table 8、Figure 5 原文）
  - huggingface/transformers **commit 36deb0b5**：`src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`、`src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`（经 raw.githubusercontent 取回后按行核对；qwen4_exp 副本与工作树 /tmp/modeling_qwen4_exp.py 逐字节相同，diff 无差异）
  - HuggingFace `Qwen/Qwen2-VL-7B-Instruct` 的 config.json
  - 本机实际运行页面代码块；`.dojo/scripts/validate.py` 返回 `validation ok`
  - 页内交叉核对：`wiki/qwen3-8-flash-next-dataflow/index.html`、`wiki/qwen3-5-dataflow/index.html`

## 来源核对（逐条，均给出原文片段/数值）

- C1 三分量拆分：§2.1 「deconstructing the original rotary embedding into three components: temporal, height, and width」——支持。
- C2 文本退化：§2.1 「For text inputs, these components utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE」——支持（页面「功能等价于 1D-RoPE」措辞一致）。
- C3 图像分配：§2.1 「the temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components」——支持。
- C4 视频递增：§2.1 「For videos, which are treated as sequences of frames, the temporal ID increments for each frame」——支持。
- C5 三维动机：§1 「Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions. The use of one-dimensional position embeddings in current models significantly limits their ability to model three-dimensional space and temporal dynamics effectively.」——支持，页面未扩大适用范围。
- C6 降低位置 id 与外推：§2.1（M-RoPE 小节）「reduces the value of position IDs」…「enabling the model to extrapolate to longer sequences during inference」——支持；页面对其所在章节的归属（§2.1）与来源一致（曾被怀疑应在 §3.3.2，核对后确认在 §2.1）。
- C7 跨模态衔接：§2.1 「each modality's numbering starts from the previous modality's highest ID plus one」——支持。
- C8 分段排布：commit 36deb0b5 的 `modeling_qwen2_vl.py` L180 `def apply_multimodal_rotary_pos_emb`、L212 `mrope_section = mrope_section * 2`、L213 `cos.split(mrope_section, dim=-1)` 配 `m[i % 3]`——页面标注的 L180-216 与实际行号命中。
- C9 交错排布与 docstring：`modeling_qwen4_exp.py` L140 `def apply_interleaved_mrope`，docstring 「Reorganizes frequency layout from chunked [TTT...HHH...WWW] to interleaved [THWTHWTHW...TT], preserving frequency continuity.」——页面标注 L140-155 命中。
- C10 推进量：`modeling_qwen4_exp.py` **L2115** `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`——页面标注的 L2115 正落在该推进语句上，命中。
- F1 视觉段生成式：`get_vision_position_ids`（L1980 起）中 `position_temporal = torch.arange(llm_grid_t) * time_interval`、`position_height = torch.arange(llm_grid_h) + start_position`、`position_width = … + start_position`、`vision_position_ids[0] += start_position`——与 $T=\mathrm{arange}(n_t)\cdot\text{interval}+s$、$H/W=\mathrm{arange}(g/\text{merge})+s$ 逐项一致；页面已声明论文正文无显式公式、取自官方实现。
- F2 交错槽位归属：`modeling_qwen4_exp.py` L150-154 `freqs_t = freqs[0]`；`length = mrope_section[dim] * 3`；`idx = slice(offset, length, 3)`。按 mrope_section=[11,11,10] 复算：H=slice(1,33,3)→{1,4,…,31} 共 11；W=slice(2,30,3)→{2,5,…,29} 共 10；T 余下 {0,3,…,30} 共 11。与页面槽位表（T 11 / H 11 / W 10，槽位 31=H、30=T）逐项一致。
- 分段例参数：`Qwen/Qwen2-VL-7B-Instruct` config.json 的 `rope_scaling` 实为 `{"type": "mrope", "mrope_section": [16, 24, 24]}`（页面写法「config.json，rope_scaling.mrope_section」准确）；hidden_size 3584 / num_attention_heads 28 → head_dim 128（页面「head_dim 128、频率槽位共 64 个」成立）；mrope_section*2=[32,48,48] 切 cos/sin 后 T[0,31]/H[32,79]/W[80,127] → 频率 T[0,15]/H[16,39]/W[40,63]，与页面一致。
- N1 消融数字：§3.3.2 Table 8，1D-RoPE → M-RoPE 为 NextQA 43.9→46.0、STAR 55.5→57.9、RWQ 54.5→53.7、InfoVQA 50.8→50.3；设定「We employ Qwen2-1.5B and ViT-L as the backbone and report the results of the pre-trained models」；措辞「compared to 1D-RoPE, using M-RoPE achieves better performance in downstream tasks, particularly in video benchmarks」——页面数字、方向、骨干与措辞均一致（含 RealWorldQA/InfoVQA「小幅下降」）。
- N2 16K/80K：§3.3.2「despite limiting the maximum tokens per video to 16K during training, the model still exhibits exceptional performance at a maximum inference length of 80K tokens」；Figure 5 图注「the model demonstrated robust performance when the inference length exceeded the maximum training length of 16384 tokens」，语境为 Qwen2-VL-72B Video-MME Medium Video——页面一致。
- N3/N4 与代码块：本机重跑页面代码，输出与「预期输出」逐行完全相同（196/14、(8,8,8)/(8,21,21)、(22,22,22) 而非 (204,204,204)、196→14、784→28、1764→42、1980→60）。
- 页内算式复核：比值=token 数/推进量=(g_h/merge·g_w/merge)/(max/merge)=min(g_h,g_w)/merge，代入四组网格得 14.0/28.0/42.0/33.0，与表格一致；209=8+196+5、1777=8+1764+5、26=22+5−1、54=8+42+5−1、1777/209≈8.5、54/26≈2.08，均成立。
- 交叉页一致性：description / dojo:summary / 正文 / overview 的 (1,28,28) 与 (1,84,84) 数字、16K/80K、消融结论一致；qwen3.8-Flash-Next 的 rotary_dim 64、mrope_section [11,11,10]、rope_theta 1e7、max_position_embeddings 262144 与 `qwen3-8-flash-next-dataflow` 一致；Qwen3.5-397B-A17B 的「与 Qwen3.8 逐项相同」与 `qwen3-5-dataflow` 一致。
- 链接与资源：`../rope/`、`../positional-encoding/`、`../vit/`、`../qwen3-8-flash-next-dataflow/`、`../qwen3-5-dataflow/index.html`、`overview.html` 均真实存在且双向互链；无「（待生成）」占位；无 Unicode 数学字符；alt 无 `$...$`；`dg-stack`/`dg-layer`/`diagram-caption` 类在 dojo-concept.css 中均存在。
- 表述：未发现元话语、「本页」被误用为主语自我指代（仅出现在规范允许的来源说明段）、会话指代、调试叙事、临场评价或 AI 拼接腔；章节衔接用「上一章已得结论 + 下一步问题」的方式，非固定句式。

## 问题

- [轻微·技术] index.html L89（核心问题第 3 条解答）与 L282（第 3 章正文）：交错排布下每个分量实际只缺 1–2 个槽位，页面写作「各自缺少数个端点附近槽位」，「数个」与实测不符｜引文依据：页内 L278 自述「$T$ 落在 $0,3,\ldots,30$，$H$ 落在 $1,4,\ldots,31$…、$W$ 的槽位从 2 到 29…（不含两端点）」；按 qwen4_exp L150-154 复算槽位归属为 T 缺槽位 31、H 缺槽位 0、W 缺槽位 0 与 31｜修复要求：把 L89、L282 两处的「各自缺少数个端点附近槽位」改为可逐项复验的数量表述，例如「$T$ 缺槽位 31、$H$ 缺槽位 0、$W$ 缺槽位 0 与 31」｜修复：｜复验：
- [轻微·格式] index.html L169（F1 公式符号表）：同一公式的符号 $s$、$n_t$ 用 `$...$` 包裹而 `interval` 裸写，与公式内的 `$\text{interval}$` 及同列表内 `$\text{merge}$` 的写法不一致，违反 style-guide §11「同一变量全页写法一致」｜引文依据：不适用｜修复要求：将 L169 的 `interval` 改为 `$\text{interval}$`（L174、L335 已用 `$\text{interval}$`）｜修复：｜复验：
- [轻微·格式] index.html L7（`dojo:summary`）：同一变量 `merge` 在 dojo:summary 写作 `$\mathrm{merge}$`，正文 6 处（L95/L107/L166/L174/L309/L332）一律写作 `$\text{merge}$`｜引文依据：不适用｜修复要求：把 dojo:summary 中的 `\mathrm{merge}` 改为 `\text{merge}`，与正文统一｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（无阻断、无重要；3 条轻微不阻断发布，建议随轮一并修掉）

统计：阻断 0 / 重要 0 / 轻微 3