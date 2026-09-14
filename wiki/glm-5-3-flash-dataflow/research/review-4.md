<!-- review-meta
round: 4
page: wiki/glm-5-3-flash-dataflow/index.html
reviewed_content_sha256: 51b2c049174ba814
-->
# GLM-5.3-Flash 前向数据流审查记录（第 4 轮）

- 页面版本：12f080638fbb818dae9e4a8d6eda01cb82fb9691（index.html 工作树哈希）
- 审查时间：2026-09-13 21:11
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：1 关键规格；2 整体数据流（含 45 层的类型排布）；3 单层内部数据流；4 mHC：4 路残差流怎么读写；5 KDA 层：34 层线性注意力；6 DSA 层：11 层稀疏注意力；7 MoE 路由；8 位置信息从哪来；9 长上下文下的实际收益；10 FP8 量化的覆盖范围；11 多模态：图像与视频共用 token；12 MTP 层；13 核对方式；来源与范围说明（C1–C6 / F1–F3 / N1–N13 与简化条件）。

## 回源核对

- head 的 dojo:type=dataflow → 依 guides/model-dataflow.md 审查。
- 官方 config.json（zai-org/GLM-5.3-Flash）：hidden_size 4096、num_hidden_layers 45、n_routed_experts 288、num_experts_per_tok 8、n_shared_experts 1、q_lora_rank 1536、kv_lora_rank 512、qk_nope_head_dim 256、qk_rope_head_dim 0、v_head_dim 256、index_topk 2048、index_kpool 4、index_n_heads 32、index_head_dim 128、hc_mult 4、max_position_embeddings 1048576、FP8 e4m3 block [128,128]。与页面 §1 表逐项一致。
- NVIDIA NeMo AutoModel 文档（model-coverage/vision-language-models/thudm/glm-5-3-flash）：45 层 34 KDA + 11 KPool-DSA、三 KDA 夹一 DSA 且末层 KDA、288 routed + 1 shared top-8、前 3 层 dense 后 42 层 MoE、4 路 mHC、320B total / 18B active、context 1,048,576。与页面一致；页面另以 checkpoint 累加给出 321.32 B，并在 [N2] 注明官方 README 为 320B total / 18B active，无未披露的分歧。
- arXiv:2602.15763v2 §2.1 / §2.1.1 / §2.1.2：GLM-5 为 256 专家、80 层、744B（40B 激活）、MLA + DSA，§2.1.2 比较 GDN，DSA 动机为「长上下文约 90% 注意力项冗余」「约 1.5–2× 降算」。页面「范围之外」段落的表述与之相符，且明确不把该报告当作本模型的结构依据。
- 算式复算（全部与页面一致）：MoE 单层激活 227,672,352×42=9,562,238,784；分项 34×137,732,288=4,682,897,792、11×124,914,432=1,374,058,752、embed+lm_head=154880×4096×2=1,268,776,960、3×150,994,944=452,984,832、mHC 786,486×45=35,391,870，合计 17,376,348,990；占比 55.03/26.95/7.91/7.30/2.61/0.20 加和为 100.00。参数口径 321,323,031,390 与分项（主干 313,890,070,334 含视觉塔 + MTP 7,432,961,056）闭合。
- 长上下文表复算：fp32 状态 64×128×128×4=4 MiB、conv 24576×4×2=0.1875 MiB、DSA 769×2=1.502 KiB。4096→0.204/0.264 GiB（省 22.89%）、32768→0.655/2.112（68.97%）、131072→2.204/8.448（73.91%）、1048576→16.661/67.588（75.35%）；上限 34/45=75.6%；交叉点 149,291,008/(34×1538)=2855；仅 512 维口径下 149,291,008/34,816=4288、L=4096 改判为「多占 3.54%」——均一致。
- k-pool 表（min(512,⌈L/4⌉)、稀疏度、打分占稠密恒为 25%）、视觉 token 换算（448/896/1344 与 T=4 视频，784 像素/token）、视觉塔 563,627,008、37,338 个 scale 张量计数（按结构分解=36,288+864+126+3+44+9+4）与 [N1] 76,108 张量口径、degradation 检验（L≤2048 退化因果掩码、尾巴长度=可见数 mod 4、index_topk=16 时选中集 [20..51] 四整池且不含下标 63），均自洽。
- 公式 [F1]（delta rule 递推、safe 分支 g、chunk/recurrent 等价）、[F2]（noaux_tc 路由、偏置只影响选择、权重和 2.5、SwiGLU 不对称截断、共享专家并联）、[F3]（池内逐通道门控）与页面对应段落一致；checkpoint 口径 q_conv1d×34、kv_a_proj_with_mqa×12（11 主干 + MTP）与层排布一致。
- 链接与文件：../residual-connection ../kda ../kv-cache ../delta-rule ../low-rank-projection ../aux-loss-free-routing ../swiglu ../speculative-decoding ../vit ../quantization-basics 的 index.html 均存在；research/measured.md 存在（符合 model-dataflow.md 的记载位置）。
- .dojo/scripts/validate.py wiki/glm-5-3-flash-dataflow/index.html → validation ok。
- 表述通读（含折叠块与图注）：未发现元话语（"本页将…""下面来看…""需要注意的是"）、以"本页"为主语的自指、会话指代（我/我们/你）、调试完成式叙事或 AI 拼接腔；数学符号全部 KaTeX 渲染，正文/标题/summary/表格无直接出现的 Unicode 数学字符（仅数据流箭头 → 作流向记号），alt/aria-label 无 `$...$`。

## 问题

- [轻微·技术] §9「两个口径差 64 倍」提示框与 [N11]：同句内「缓存形状为 $[B,64,L,256]$」与「每 token 每层 32768 个元素」不符，$64\times256=16384$（32768 是 K 与 V 两路之和）。｜引文依据：原文「实测缓存形状为 $[B,64,L,256]$，每 token 每层 32768 个元素即 64 KiB，是潜向量口径的 64 倍」。｜修复要求：把形状写成「K 与 V 各为 $[B,64,L,256]$」或改为 $[B,2,64,L,256]$，使形状与 32768 元素、64 KiB、「64 倍」自洽；结论方向不变。｜修复：｜复验：
- [轻微·表述] §13「公式与数据流：实跑官方算子」末句：「抽取区间的起止用断言锁定，防止上游文件行号漂移后静默抽错」属复现脚本的防护实现细节（踩坑叙事），非本页结论或运行条件。｜引文依据：「抽取区间的起止用断言锁定，防止上游文件行号漂移后静默抽错」。｜修复要求：删去该句，或压缩为「抽取区间以行号范围固定」，只保留核对范围（第 63–1330 行）这一依据。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布。核心结构、公式、数字与 checkpoint 口径全部回源核对通过且页内自洽，未发现阻断或重要问题；两条轻微不影响正确性与主线结论，可作为遗留轻微接受或随手修正。

统计：阻断 0 / 重要 0 / 轻微 2

> 本轮所列问题的处理结果见 `minor-fixes.md`。
