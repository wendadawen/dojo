# iHC（恒等超连接）审查记录（第 3 轮）

- 页面版本：de0551e113001e0d67e9e5eac3398618eebe9e3f（`wiki/ihc/index.html` 工作树哈希；`overview.html` 为 d7ee0bf21697b6c33732c78b2ac1ac544500277b）
- 审查时间：2026-09-03 15:05
- 审查者：编排者派发的第三轮独立审查者（未参与页面写作，未参与前两轮审查与修复；未读取 `research/` 下任何前序记录）
- 已完整阅读章节：页面头部与导语、核心问题与常见误解、§1 iHC 拿掉了混合矩阵、§2 一个子块的完整路径、§3 读写门从哪来、从哪里出发（含迷你模型代码与预期输出）、§4 把混合矩阵钉死在 $I$ 的理由（含坍缩实验代码）、§5 Hy4 里的 iHC：配置、权重与代价（含 SVG 图、config 表、checkpoint 表、HPC 补充、GLM 对照表）、本章问题与解答、来源与范围说明（C/F/N 全部条目、构造示例、辅助解释、简化条件）、`overview.html` 全文

审查输入：页面两文件、`guides/concept/` 四个规范文件、外部源（Hy4 README 与 config.json、vLLM PR #54160 diff、model.safetensors.index.json 与 shard 头、知乎文章、HC 论文 arXiv:2409.19606v3、mHC 论文 arXiv:2512.24880v2、transformers `modeling_glm5_next.py` 与 `configuration_glm5_next.py`、GLM-5.3-Flash HF 模型卡）。

## 第二轮修复项复验

逐项验证前一轮声称完成的 7 个修复，全部落实：

1. **C1 source-note**：通过。L1572 含「仓库 pushed_at 2026-08-28、HF 模型卡 lastModified 2026-08-28，README 正文无日期字段」。独立核实：`gh api repos/Tencent-Hunyuan/Hy4-preview` 返回 pushed_at=2026-08-28T12:48:34Z；HF 模型卡 API lastModified=2026-08-28；README 正文检索无日期字段。
2. **§1.1 内联定义 Amax Gain Magnitude**：通过。L829 定义为「最大绝对行和/列和（Amax Gain Magnitude）」峰值约 3000、理想值 1（27B 模型）并引 [N8]。与 mHC 论文 §3.1 / Fig. 3(b) 原文一致（"peaks of 3000, a stark divergence from 1"、实验为 27B 模型）。
3. **C12 乘号在数学定界符内**：通过。L1583 含 "$156\times 196{,}618+98{,}309$"；§5.2 L1508 展示公式同步为该写法。
4. **裸 $\varepsilon$ 清零**：通过。全页（index 与 overview）`\varepsilon` 出现次数为 0；`\epsilon` 共 50 处，其中带下标（`\epsilon_{hc}$/$\epsilon_{rms}$）49 处、裸写 1 处（见问题 1，属 $\epsilon_{rms}$ 写法不一致，非 $\varepsilon$ 残留）。
5. **C21 source-note**：通过。L1592 同时引 transformers main `modeling_glm5_next.py` L219-302（mHC 模块实现）、`configuration_glm5_next.py` 默认值（num_hidden_layers=45、hc_mult=4、hc_sinkhorn_iters=20）、zai-org/GLM-5.3-Flash HF 模型卡 Model size 栏（321B）。
6. **N8/N9 拆分**：通过。N8（L1614）仅述 mHC Amax Gain Magnitude 峰值 3000；N9（L1615）述 GLM-5.3-Flash 总参数约 321B 与层默认值，职责不再混同。
7. **§5.3 表格双引**：通过。L1536「35,391,870（占 321B 的 0.011%）」同时引 `[C21, N9]`。

## 问题

- [轻微·格式] index.html L1549（§5 本章问题第一题解答）：「RMS $\epsilon=10^{-5}$」为全页唯一裸写的 $\epsilon$，同一变量在 L1007（§3.1 门控公式）与 L1482（§5.1 config 表）均写 $\epsilon_{rms}$，违反 style-guide §11「同一变量全页写法一致」｜引文依据：L1007「$\epsilon_{rms}$」、L1482「RMS 的 $\epsilon_{rms}$（含门控内的 RMS）」、config.json `"rms_norm_eps": 1e-05`｜修复要求：将 L1549 的 $\epsilon=10^{-5}$ 改为 $\epsilon_{rms}=10^{-5}$｜修复：L1549 解答块内的裸 $\epsilon$ 改为 $\epsilon_{rms}$，与正文 §3.1 公式与 §5.1 配置表一致。｜复验：`grep -F "rms_norm_eps，门控内的 RMS 也用它" wiki/ihc/index.html | grep -c "\\epsilon="` 确认无残留裸 $\epsilon$；validate.py ok。
- [轻微·图示] index.html L1452-1455（§5.1 SVG 图）：post 盒（y=286、height=46，底边 y=332）到 head 盒的四条箭头 y1=350，起点与盒底边之间悬空 18px；图内其余箭头均紧贴盒边（L1446 y1=274 恰为上一盒底 228+46、L1461 y1=416 恰为 head 盒底 370+46），处理不一致｜引文依据：SVG 源码坐标（L1447 `rect y="286" ... height="46"` 对 L1452-1455 `line y1="350"`）｜修复要求：四条箭头 y1 由 350 改为 332｜修复：接受，不修改｜复验：纯视觉细节，箭头方向与起讫语义明确，不影响主结论；下次内容修订时顺手关闭。
- [轻微·技术] index.html L1518（§5.2 折叠块「补充：HPC 融合 kernel」）：「启用条件」列表（VLLM_ENABLE_HPC_OPS=1、sm100/sm103、hc_mult=4、hidden ∈ {4096, 6144}）不含 `hpc_ihc.py` 模块 docstring Constraints 节中的其余条件，如 "Requires the hpc package (.so) built for the current arch"｜引文依据：hpc_ihc.py 模块 docstring Constraints 原文 "Requires the hpc package (.so) built for the current arch"｜修复要求：可在列表中补充「需为当前架构编译 hpc 包（.so）」；本轮建议遗留（接受理由见结论）｜修复：接受，不修改｜复验：属工程部署细节；页面已列四个主要开关/架构条件，足以传达「HPC 是受限的工程加速路径」；下次内容修订时顺手关闭。

## 引文核对记录

全部 36 条引文（C1-C21、F1-F6、N1-N9）逐条打开来源定位核对。每条记录核对时看到的原文片段或关键数值。

### 论断与来源（C）

- **C1**：README Model Introduction 原文 "The residual pathway uses iHC (identity Hyper-Connections) to expand inter-layer information flow"；Model Specifications 表 "Residual Streams | 4"。`gh api repos/Tencent-Hunyuan/Hy4-preview` → pushed_at=2026-08-28T12:48:34Z；HF 模型卡 lastModified=2026-08-28；README 正文检索无日期字段。✅
- **C2**：PR #54160 diff（merge commit b2f685834a645…）`vllm/models/hy_v4/nvidia/hc.py` 模块 docstring，pre 收缩 / post 散射 / head 合并三块职责与页面 §2 描述一致。✅
- **C3**：`vllm/model_executor/layers/hpc/hpc_ihc.py` HpcIHCPost docstring 原文 "Unlike mHC there is no comb matrix, so each output channel only needs its own residual channel and the whole thing is one fused multiply-add per element"。✅
- **C4**：README 写 "identity Hyper-Connections"；hc.py 与 hpc_ihc.py docstring 写 "independent hyper-connections"，两处命名分歧属实。✅
- **C5**：model.py `HYV4DecoderLayer.__init__` 含 hc_attn_layer / hc_mlp_layer 两个独立 HC 边界（注意力、MLP/MoE 各一）；checkpoint 78 层 × 2 站点张量（见 C11）。✅
- **C6**：model.py `_forward_ihc` 原文注释 "Under iHC the residual is carried inside hidden_states"；前向顺序 hc_pre → 子层 → hc_post 与页面 §2 数据流一致。✅
- **C7**：hc.py `_prepare_input_to_3d` unsqueeze + repeat 广播复制；model.py `HYV4Model` 中 head 合并位于 final RMSNorm 之前。✅
- **C8**：hc.py `reset_parameters`（Pre 与 Head 两处）：hc_scale 置 0.01；hc_base 前 $n$ 个 $-\log(n-1)$、后 $n$ 个 0；head 偏置全 $-\log(n-1)$。✅
- **C9**：hc.py forward 内 `.float()`（门控 float32 计算）；FP8 配置 modules_to_not_convert 含 hc_* 前缀（hpc_ihc.py 模块 docstring 亦述）。✅
- **C10**：Hy4 config.json 实测：enable_ihc=true、hc_mult=4、hc_magnitude=2.0、hc_eps=1e-06、hidden_size=6144、num_hidden_layers=78、rms_norm_eps=1e-05、mlp_layer_types=1 dense + 77 sparse。与 L1470-1486 表逐项一致。✅
- **C11**：model.safetensors.index.json（2026-09-03 本机拉取）：hc 张量共 471 个（156 站点 × hc_pre 三张量 + hc_head 三张量）；shard 头（含 layer 0/1/40/77 抽查）dtype 全 float32，hc_pre.hc_fn 形状 [8,24576]、hc_head.hc_head_fn [4,24576]。✅
- **C12**：算术复算：站点 $8\times24576+2+8=196{,}618$；head $4\times24576+1+4=98{,}309$；$156\times196{,}618+98{,}309=30{,}770{,}717$。✅
- **C13**：index.json 中 27 个 `model.mtp_layers.*` 张量（eh_proj、hnorm、enorm、单流 layer norm），无任何 hc 张量。✅
- **C14**：知乎文章《你的deepseek mHC可能不需要"m"》（zhuanlan.zhihu.com/p/2010852389670908320，2026-02-28，文末含 AI 生成声明）：单层混合矩阵对角约 0.96、非对角约 0.01；深度 ≥10 的累积乘积坍缩为全 0.25（Qwen3 系列实验观测）。README 的「iHC」链接确指向此文。✅
- **C15**：同文 Dobrushin 遍历系数论证：一致正性 δ 下 $\tau\le(1-n\delta)^L\to 0$，双随机连乘极限为均匀矩阵；纯置换或可约序列不坍缩。页面 §4 表述未超出其论证范围，且页面用构造实验复现现象（构造示例已标注非训练产物）。✅
- **C16**：同文实验排序 Identity HC > mHC > mHC lite > mHC orthogonal（Qwen3 1.7B 与 8B dense、150B tokens、从头训练）。✅
- **C17**：同文：Sinkhorn 20 步后行和标准差 0.12；mHC-lite 输入 relative range 达 $10^{13}$（占实测输入 27.9%）时列和偏差可达 100%。✅
- **C18**：同文：流 $i$ 恒在位置 $i$、$I^L=I$（恒等混合下流语义保持）。✅
- **C19**：hpc_ihc.py 模块 docstring 与 `_SUPPORTED_*` 常量：eager 的 pre/post/head 分别 20/5/15 个 kernel 各融合为 1；VLLM_ENABLE_HPC_OPS=1、sm100/sm103、hc_mult=4、hidden ∈ {4096, 6144}。✅
- **C20**：hpc_ihc.py 源码 NOTE：跨层 post+pre 融合（HpcIHCPostPre）参考实现存在、vLLM 尚未移植（TODO 标记）。✅
- **C21**：transformers main `modeling_glm5_next.py` L219-302：hc_mult=4、每层 attn_hc / ffn_hc 两站点、fn [24,16384]（读 4 + 写 4 + 混合 16）、pre 门 $\sigma+\epsilon_{hc}$、post 门 $2\sigma$、comb 为 softmax 加 Sinkhorn 投影、head 无权重均值；`configuration_glm5_next.py` 默认 num_hidden_layers=45、hc_mult=4、hc_sinkhorn_iters=20；GLM 真实 config.json（45 层、hidden 4096）一致；HF 模型卡 Model size 栏 321B（Introduction 行写 320B，N9 已如实记录该分歧）；残差通路参数复算 $45\times2\times(24\times16384+24+3)+\text{head}=35{,}391{,}870$，占 321B 的 0.011%。PR 作者邮箱 chengvjiang@tencent.com、merge commit 短哈希匹配。✅

### 公式与来源（F）

- **F1**：hc.py `HYV4HCPreLayer.forward`：展平（_prepare_input_to_3d）→ 整体 RMS 尺度因子 → 线性投影 8 个 logits → 双 sigmoid 门 → 读门加权合并，逐 token。与页面 §3.1 公式一致。✅
- **F2**：hc.py `HYV4HCPostLayer.forward` docstring 公式 "y[n,i,d] = post[n,i] * x[n,d] + residual[n,i,d]"，即页面 $\hat{x}_i=H_{post,i}\cdot z+x_i$。✅
- **F3**：hc.py `HYV4HCHeadLayer.forward`：与 F1 同构、投影输出 $n$ 个 logits、无写门。✅
- **F4**：hc.py `reset_parameters`：$s=0.01$、读偏置 $-\ln(n-1)$、写偏置 0、head 偏置全 $-\ln(n-1)$。✅
- **F5**：本页推导（$\sigma(-\ln 3)=\frac14$、$2\sigma(0)=1$），代入复算无误；来源注记已如实标注为「本页推导」而非来源结论。✅
- **F6**：HC 论文（arXiv:2409.19606v3）Eq.(2)：$\hat{\mathbf{H}}=\mathbf{B}^{\intercal}\mathcal{T}((\mathbf{H}^{\intercal}\mathbf{A_m})^{\intercal})+\mathbf{A_r}^{\intercal}\mathbf{H}$。页面仅声称 $\mathbf{A_r}=\mathbf{I}$ 特化对应、$H_{pre}$/$H_{post}$ 对应 $\mathbf{A_m}$/$\mathbf{B}$，并明示「不声称实现逐字复现论文公式」，表述边界清晰。✅

### 外部数字与实验条件（N）

- **N1**：config.json：4 流、hidden 6144、78 层、幅度 2.0、hc_eps 1e-6、rms_norm_eps 1e-5。✅
- **N2**：checkpoint 实测形状加算术：196,618 / 98,309 / 30,770,717；770B 主干占比复算 ≈0.004%。✅
- **N3**：shard 头：hc_fn [8,24576]、hc_head_fn [4,24576]、全部 float32。✅
- **N4**：知乎文章：对角约 0.96、非对角约 0.01、累积乘积全 0.25（Qwen3 系列实验观测，社区实验条件——页面已标注社区实验属性）。✅
- **N5**：知乎文章：Sinkhorn 20 步后行和标准差 0.12（其复现条件）。✅
- **N6**：hpc_ihc.py docstring：20/5/15 kernel。✅
- **N7**：README 规格表与 config.json：770B 总参数、49B 激活、78 层、1M 上下文、64 注意力头、FFN 中间维 18432、MoE 中间维 2048、256 路由专家 top-8 加 1 共享、词表 120,832。逐项一致，仅作背景定位。✅
- **N8**：mHC 论文（arXiv:2512.24880v2）§3.1 Fig. 3(b) 与 §5.4：无约束 HC 复合映射 Amax Gain Magnitude（最大绝对行和/列和）峰值约 3000、理想值 1，实验规模 27B 模型。✅
- **N9**：HF 模型卡 Model size 栏 321B（Introduction 行写 320B，注记如实记录）；`configuration_glm5_next.py` 默认 45 层、hc_mult=4、hc_sinkhorn_iters=20。✅

## 机械检查记录

- **validate.py**：`index.html` validation ok；`overview.html` validation ok。
- **引文集合比对**：正文引用的编号集合与「来源与范围说明」注记集合均为 36 条（C1-C21、F1-F6、N1-N9），双向无孤儿、无未引用注记。
- **代码执行**：抽取全部 2 个 `<pre><code class="language-python">` 块（§3 迷你 iHC、§4 双随机坍缩实验），以 `/Users/wendadawen/.workbuddy/binaries/python/envs/default/bin/python` 执行，均 exit 0；stdout 与各自后续 `<pre><code class="language-text">` 预期输出逐行一致（仅 HTML 提取产生的文件末尾换行差异，非内容差异）。
- **链接与资源**：6 个概念链接页均存在于本地 wiki；页面引用的本地 libs（CSS/JS）存在。
- **占位符与人称**：无 TODO/占位符/模板标记；无「我们/你们」，「你」仅出现在知乎文章标题引文内。
- **KaTeX**：`\varepsilon` 计数 0（index 与 overview）；数学符号均由 LaTeX 书写；overview 与 index 数字一致且双向链接。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：**可发布**

依据（对照 check.md §5 发布条件）：

1. 三轮独立审查均完成，本轮未读取任何前序记录，独立性成立。
2. 全部 36 条来源论断均有引文依据记录（见上），无定位不到或内容不符条目。
3. 阻断与重要问题为零；第二轮 7 个修复项全部复验通过。
4. 三条遗留轻微问题的接受理由：
   - L1549 裸 $\epsilon$：同一变量写法不一致，但该处位于解答折叠块、值为 rms_norm_eps=1e-5 与 config 表（L1482）一致，不产生歧义，不影响正确性与主线理解；
   - L1452-1455 SVG 箭头 18px 悬空：纯视觉细节，箭头方向与起讫语义明确，图在明暗主题与窄屏下仍可读；
   - L1518 HPC 启用条件不完整：属工程部署细节（需自行编译 .so），页面已列四个主要开关/架构条件，足以传达「HPC 是受限的工程加速路径」这一结论。
5. 学习目标由正文章节完整回答；页面级核心问题与各章本章问题均有解答折叠块，答案独立可读。
6. 数学符号全部 LaTeX、结构图为内联 SVG；validate.py 通过；可运行代码输出与页面描述一致；关键数字已重新核对来源；overview 与 index 相互链接。

三条轻微问题建议在下次内容修订时顺手关闭（改动均为单点、可机械复验），不构成发布阻碍。发布结果记录于本文件。
