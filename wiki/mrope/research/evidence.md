# evidence：MRoPE

来源固定：Qwen2-VL 论文 arXiv:2409.12191v2（Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution）§2.1 与 §3.3.2，https://arxiv.org/html/2409.12191v2（本机已抓取全文文本）；transformers@36deb0b5 源码（qwen2_vl 与 qwen4_exp 两实现）；本机实测（数据流页 probe8 的 J/K 节 + 本页独立复测）。

| 编号 | 论断 | 来源定位 | 适用条件 | 置信 |
|---|---|---|---|---|
| C1 | MRoPE 把旋转位置嵌入拆成时间、高、宽三个分量 | 论文 §2.1"This is achieved by deconstructing the original rotary embedding into three components: temporal, height, and width." | — | 已确认 |
| C2 | 文本输入下三个分量用相同位置 id，MRoPE 功能上等价于 1D-RoPE | 论文 §2.1"For text inputs, these components utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE." | 纯文本序列 | 已确认 |
| C3 | 图像：每个视觉 token 的 temporal id 恒定，height/width 按其在图中的位置赋值 | 论文 §2.1"When processing images, the temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components based on the token's position in the image." | 图像输入 | 已确认 |
| C4 | 视频：temporal id 逐帧递增，height/width 与图像同规则 | 论文 §2.1"For videos, which are treated as sequences of frames, the temporal ID increments for each frame…" | 视频输入 | 已确认 |
| C5 | 动机：真实环境是三维的，一维位置嵌入限制对三维空间与时间动态的建模 | 论文 §1"Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions. The use of one-dimensional position embeddings in current models significantly limits their ability to model three-dimensional space and temporal dynamics effectively." | — | 已确认 |
| C6 | MRoPE 降低图像与视频的位置 id 取值，使模型能在推理时外推到更长序列 | 论文 §2.1"M-RoPE not only enhances the modeling of positional information but also reduces the value of position IDs for images and videos, enabling the model to extrapolate to longer sequences during inference." | — | 已确认 |
| C7 | 跨模态衔接：每个模态的位置编号从前一模态最大位置 id 加一开始 | 论文 §2.1"In scenarios where the model's input encompasses multiple modalities, position numbering for each modality is initialized by incrementing the maximum position ID of the preceding modality by one." | — | 已确认 |
| C8 | 分段排布：qwen2_vl 把 cos/sin 按 mrope_section*2 切段，第 i 段用 i%3 选分量；[16,24,24] 为 Qwen2-VL 官方配置取值（Qwen/Qwen2-VL-7B-Instruct 的 config.json rope_scaling.mrope_section，本机已拉取核对） | 源码 modeling_qwen2_vl.py L180-216 apply_multimodal_rotary_pos_emb："mrope_section = mrope_section * 2; cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))]" | transformers@36deb0b5 | 已确认 |
| C9 | 交错排布：qwen4_exp 以 T 维为底，H 覆盖 slice(1, 33, 3)、W 覆盖 slice(2, 30, 3)，三维在全部 32 槽位交错；源码 docstring 称从 chunked [TTT...HHH...WWW] 改为 interleaved [THWTHW...] 且保持频率连续性 | 源码 modeling_qwen4_exp.py L140-155 apply_interleaved_mrope 及其 docstring；本页实测槽位归属 T=11/H=11/W=10 与 mrope_section=[11,11,10] 逐项一致 | mrope_interleaved=true | 已确认 |
| C10 | 位置推进量：图像块结束后 current_pos += max(grid_h, grid_w) // spatial_merge_size，而非推进视觉 token 数 | 源码 qwen2_vl 与 qwen4_exp 的 get_rope_index 均为该式（qwen4_exp 版 L2115）；与 C7 的「前一模态最大位置 id + 1」一致——视觉 token 的最大位置 id 即 start + max(h,w)/merge − 1 | — | 已确认 |
| F1 | 视觉段三维位置生成：$T=\mathrm{arange}(t/1)$，$H=\mathrm{arange}(h/2)+s$，$W=\mathrm{arange}(w/2)+s$，meshgrid 后叠为 $(3,n)$ | 源码 qwen4_exp L1980-2030 get_vision_position_ids；qwen2_vl 同构 | spatial_merge_size=2 | 已确认 |
| F2 | 交错槽位分配：H ← slice(1, 33, 3)，W ← slice(2, 30, 3)，其余归 T | 源码 L150-154；本机实测复现 | rotary_dim=64、mrope_section=[11,11,10] | 已确认 |
| N1 | 消融：Qwen2-1.5B + ViT-L 骨干上 M-RoPE 优于 1D-RoPE，尤其视频基准（如 PerceptionTest 47.4 vs 46.6、NextQA 46.0 vs 43.9、STAR 57.9 vs 55.5、MathVista 43.4 vs 39.2） | 论文 §3.3.2 与 Table 8 | 该骨干与训练配置 | 已确认 |
| N2 | 训练每视频最多 16K token，推理 80K token 仍表现稳健（Figure 5 语境为 Qwen2-VL-72B 在 Video-MME 中等时长视频） | 论文 §3.3.2"Notably, despite limiting the maximum tokens per video to 16K during training, the model still exhibits exceptional performance at a maximum inference length of 80K tokens." | Video-MME（Figure 5） | 已确认 |
| N3 | 本机实测：(1,28,28) 图接在 8 个文本 token 后，196 个视觉 token 的 T 恒为 8、H/W 在 [8,21] 遍历；后续文本从位置 22 开始而非 204 | 数据流页 probe8_vision.py J 节实测；本页独立复测 | 构造输入 | 已确认 |
| N4 | 本机实测：推进量对照 (1,28,28)→14、(1,56,56)→28、(1,84,84)→42、(1,66,120)→60；对应 token 数 196/784/1764/1980 | 同上 | — | 已确认 |

冲突与不足：论文正文未给出显式的 $(t,h,w)$ 数学公式（以 Figure 3 图示与文字描述呈现），公式 F1 取自官方实现源码，页内明确标注这一对应关系；Qwen2.5-VL 的 position alignment 细节未纳入（超范围）。
