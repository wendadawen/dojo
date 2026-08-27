# MRoPE审查记录（第 1 轮）

- 页面版本：eb4b08eb52a603ac8092cb376fe3a9889cd2300d（index.html 工作树哈希）
- 审查时间：2026-08-27 14:32
- 审查者：编排者派发的独立审查者（未参与写作与前序修复）
- 已完整阅读章节：引言、核心问题、常见误解、1. 文本有序、图像有格——一维位置轴装不下多模态、2. 位置 id 变三元组——三种模态的分配规则、3. 一个头维装三条轴——分段与交错两种槽位排布、4. 位置轴上的省账——推进量与长序列外推、来源与范围说明（C/F/N/构造示例/类比边界/简化条件各小节）、overview.html 全部小节；全部解答折叠块与代码折叠块已展开阅读。

## 问题

- [重要·技术] index.html §2 F1 公式（「视觉段的生成式（按官方实现）」）：公式写 $T=s$，与所引实现不符，也与本页视频规则矛盾。实现中时间分量为 `arange(t)·time_interval + s`，仅当 $t=1$（图像）时退化为 $T=s$；本页同章表格与本章问题均称视频 $t$ 逐帧递增（第 $k$ 帧 $t=s+k$），公式按「视觉段」理解即漏掉视频情形｜引文依据：qwen2_vl modeling L905/L911 `position_temporal = torch.arange(llm_grid_t, device=device) * time_interval` … `vision_position_ids[0] += start_position`；qwen4_exp modeling L2023/L2029 同构｜修复要求：把 $T=s$ 改为 $T=\mathrm{arange}(t)+s$（可注明默认 time_interval=1），或在公式处明确限定「单帧图像」并指向视频规则｜修复：｜复验：
- [重要·技术] index.html §3「两种排布的实质差别在频率覆盖：分段让 $T$ 只占低频段、$W$ 只占高频段」：频率方向颠倒。RoPE 槽位序号小对应 inv_freq 大（高频、近距离），本页自己给出的分段布局 $T$ 独占槽位 $[0,15]$（即高频端）、$W$ 独占 $[40,63]$（低频端），与该句相反；本章问题 2 答案「可以把更多低频槽位分给长程分量（如 $t$）」在同一误导下展开｜引文依据：qwen4_exp modeling L113 `inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))`（槽位 0 频率最高）；qwen2_vl L212-217 T 取 cos/sin 维度 $[0,31]$ 即频率槽位 $[0,15]$（高频段）｜修复要求：改为「$T$ 占高频段、$W$ 占低频段」并同步修正后续推理与本章问题 2 答案中依赖该方向的说法；修正后重新核对两处表述与本页槽位表一致｜修复：｜复验：
- [重要·技术] index.html §4「消融显示 M-RoPE 全面优于 1D-RoPE」：与 Table 8 不符，扩大了来源结论。RWQ 54.5→53.7、InfoVQA 50.8→50.3 下降，MMStar 36.7→36.7 持平；论文原文仅称 "achieves better performance in downstream tasks, particularly in video benchmarks"｜引文依据：论文 §3.3.2 Table 8：RWQ 1D-RoPE 54.5 / M-RoPE 53.7；InfoVQA 50.8 / 50.3；MMStar 36.7 / 36.7；NextQA 43.9 / 46.0；STAR 55.5 / 57.9｜修复要求：改为与来源一致的表述（如「总体优于、视频基准提升最明显」，或注明图像类个别基准持平或下降）｜修复：｜复验：
- [重要·技术] index.html §4「同样 262144 的位置上限，纯文本最多 26 万多个 token」：数值无 [N]/[C] 标注，且在允许来源中定位不到——qwen4_exp 与 qwen2_vl 配置默认 max_position_embeddings 均为 32768｜引文依据：configuration_qwen4_exp.py L115 `max_position_embeddings: int = 32768`；configuration_qwen2_vl.py L90 `max_position_embeddings: int = 32768`｜修复要求：为 262144 补充可定位来源（checkpoint config.json 或数据流页实测记录，并登记 [N] 编号），否则删除该数值、改为不依赖具体上限的表述；同时「max_position_embeddings 约束的是后者」宜注明为配置语义而非硬限制（本页 16K→80K 外推即为位置 id 超出训练范围的例证）｜修复：｜复验：
- [重要·技术] index.html §3「以其真实参数为例（头维 128、mrope_section $[16,24,24]$）」：数值在允许来源（modeling_qwen2_vl.py、configuration_qwen2_vl.py）中定位不到，无 [C]/[N] 标注；槽位表 $[0,15]$/$[16,39]$/$[40,63]$ 依赖该数值成立｜引文依据：允许来源中无该数值（configuration_qwen2_vl.py 未见 mrope_section 默认值；modeling L180-216 仅含机制）｜修复要求：补充可定位来源（如 Qwen2-VL-7B config.json）并在来源章节登记，或降级为明确标注的推断/构造参数；修复后逐槽位复核分段槽位表｜修复：｜复验：
- [重要·格式] overview.html：数学符号未用 LaTeX——lead 段「(t,h,w)」三元组为普通文本；「1344×1344」使用 Unicode 乘号 ×，违反 style-guide §11（数学符号一律 LaTeX、禁止 Unicode 数学字符直接出现）｜引文依据：不适用（扫描结果：overview 非 code 区域检出 ×）｜修复要求：改为 $(t,h,w)$ 与 $1344\times1344$｜修复：｜复验：
- [轻微·可读性] overview.html「关键结论与边界」：「1344×1344 的图占 1764 个序列位置」换算条件未说明，读者无法从两页复算（1344/patch 16=84 patch 网格，merge 2 后 42×42=1764）｜引文依据：configuration_qwen4_exp.py L279 `patch_size: int | list[int] | tuple[int, int] = 16`、L280 `spatial_merge_size: int = 2`｜修复要求：在 overview 或 index 补一句换算依据（84×84 patch 网格、merge 2）｜修复：｜复验：
- [轻微·格式] index.html：正文与解答共 8 处以「第 N 章」指代其他章节，未使用章节标题（style-guide §1「正文引用其他章节时使用章节标题」）：核心问题 4 处解答、§2 两处、§3 一处、§4 一处｜引文依据：不适用｜修复要求：统一改为章节标题或「第 N 章『标题』」形式｜修复：｜复验：
- [轻微·技术] index.html §3 与「简化条件及其限制」：rotary_dim 64（「只旋转前 64 维」）未在来源章节登记出处；配置默认 partial_rotary_factor=1.0，该值实际由 mrope_section $[11,11,10]$ 和为 32 槽位反推（头维 256 与 mrope_section 默认值已可定位）｜引文依据：configuration_qwen4_exp.py L225-226 `partial_rotary_factor = (self.rope_parameters or {}).get("partial_rotary_factor", 1.0); rotary_dim = int(self.head_dim * partial_rotary_factor)`；modeling L92 `self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])`｜修复要求：为 rotary_dim 64 登记可定位依据（checkpoint 配置的 partial_rotary_factor=0.25），或标注为「由 32 槽位反推」｜修复：｜复验：
- [轻微·技术] index.html §4 与来源章节 N2：「16K 训练 / 80K 推理」未注明实验主体为 Qwen2-VL-72B、任务为 Video-MME 中等时长视频，与 N1（Qwen2-1.5B + ViT-L）并列为「两个支撑数字」时易被误读为同一骨干｜引文依据：论文 §3.3.2 "Figure 5 illustrates the performance of Qwen2-VL-72B at different inference lengths"、Figure 5 图注 "on Video-MME Medium Video"｜修复要求：在 N2 处或正文补注「Qwen2-VL-72B、Video-MME 中等时长视频」｜修复：｜复验：

## 机械验证记录

- `.dojo/scripts/validate.py wiki/mrope/index.html`：通过。
- 页面 Python 代码块实际运行：输出与页面「预期输出」逐行一致（196/14、(8,8,8)/(8,21,21)、(22,22,22) 非 (204,…)、四组 grid 数字）。
- 交错槽位归属表按源码逻辑复算：T={0,3,…,30}×11、H={1,4,…,31}×11、W={2,5,…,29}×10，与页面一致；slice(1,33,3)/slice(2,30,3) 与源码 L150-154 一致。
- 链接有效：../rope/、../positional-encoding/、../vit/、../qwen3-8-flash-next-dataflow/ 均存在；../../libs/ 本地资源齐备；overview 与 index 互链。
- dojo:topics「注意力机制,多模态」均在 AGENTS.md 固定大类内。
- 其余来源论断逐条核对通过：C1-C7（论文 §1/§2.1 原文逐字比对）、C8（qwen2_vl L180-218 分段机制与 mrope_section*2、i%3）、C9（qwen4_exp L140-155 及 docstring）、C10（qwen4_exp L2115 `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`）、N1（Table 8 数字）、N2（16K/80K 原文）、头维 256（config L123）、patch 16/merge 2（config L279-280）、§4 各表数值复算无误。

## 结论

- 统计：阻断 0 / 重要 6 / 轻微 4
- 处置：修复（按上表逐条修复后复验；其中 262144 与 [16,24,24] 两条若无法补充可定位来源，按规范删除或降级为明确标注的推断，不得改写成模糊表述保留原意）
