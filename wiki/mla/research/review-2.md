# Multi-head Latent Attention（MLA）独立审查（第二次）

- 审查者：独立上下文（AI 模拟小白读者 + 强推理模型对照来源）
- 页面版本：index.html 工作树哈希 `1f4a60c92f2fa8062aebaf9fd1920ff39f342244`；overview.html 工作树哈希 `c8e6cb160c48851921eb8f56ce27f31dba5f002b`
- 时间：2026-08-09
- 审查对象：`wiki/mla/index.html`、`wiki/mla/overview.html`
- 来源：WebSearch "DeepSeek-V2 MLA arxiv 2401.06066"；`/tmp/kimi-k3-research/k3-report.txt` §2.1.2（含 §2.1.1 上下文）；WebFetch `https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json`

## 审查说明（流程限制与来源覆盖）

**流程限制**：本次审查按用户指令禁止读取 `research/` 目录。check.md 段A要求"逐题核对 scope.md 的学习目标是否全部由正文章节完整回答"——scope.md 位于 `research/` 内，受禁令限制无法读取，故学习目标闭环核对项未执行。该限制不阻断本审查其余项，但发布门控中"学习目标闭环"需由编排者另行确认。

**arxiv 编号说明**：用户搜索关键词含 "2401.06066"，但 WebSearch 全部结果指向 DeepSeek-V2 真实编号 **arXiv:2405.04434**；2401.06066 未命中任何 MLA/DeepSeek-V2 相关内容（疑为指令噪声）。两份文档均使用 2405.04434，与来源一致，非页面问题。

**段B来源核对覆盖**：
- DeepSeek-V2 论文（arXiv:2405.04434）：arxiv 编号 ✓；KV 联合压缩 Eq.(9)(10)(11) ✓；Query 压缩 Eq.(12)(13) ✓；解耦 RoPE Eq.(14)-(19)、注意力分母 √(d_h+d_h^R) ✓；Table 1 四机制公式与 "Stronger"、(d_c+d_h^R)l ≈ 9/2 d_h l ✓；W^UK→W^Q、W^UV→W^O 吸收 ✓；RoPE 与吸收不兼容论述 ✓；93.3% / 42.5% / 5.76× 相对 DeepSeek 67B（GQA）✓。
- k3-report §2.1.2（第352-365行）：NoPE 原文及动机 ✓；output gate Eq.(7) 公式及 õ_t 定义 ✓；x_t 符号（与 DeepSeek 论文 h_t 不同）已确认。
- k3-report §2.1.1（第347-350行）：KDA full-rank gate 改动来源在此节，非 §2.1.2。
- config.json（moonshotai/Kimi-K3）：N4 全部 10 字段逐字一致（q_lora_rank=1536、kv_lora_rank=512、qk_nope_head_dim=128、qk_rope_head_dim=64、v_head_dim=128、mla_use_output_gate=true、mla_use_nope=true、num_attention_heads=96、hidden_size=7168、num_hidden_layers=93）✓；N5 full_attn_layers 含 24 项（末项 93 为额外末端 MLA 层）✓；use_full_rank_gate=true、attn_res_block_size=12 ✓。
- 前置链接：overview.html → `../../wiki/linear-attention/index.html` 目标页存在 ✓；index↔overview 互相链接 ✓；GQA/MQA、RoPE、低秩投影概念页均为"待生成"占位 ✓。
- **未直接核对**：DeepSeek-V2 的 n_h=128 / d_h=128 / l=60 / d=5120 / d_c'=1536 未从本次 WebSearch 摘要逐字定位（摘要仅含公式与 Table 1，未含完整配置表）；其中 d_c=512、d_h^R=64 经 Table 1 的 9/2 d_h l 关系间接确认，d_c'=1536 与 K3 q_lora_rank=1536 一致（页面称"DeepSeek-V2 与 K3 都是 1536"，K3 侧已确认）。建议修复者核对 DeepSeek-V2 原文 §2.1.2 末段配置表。
- **未执行**：浏览器实测公式渲染/折叠交互/目录锚点（审查环境无浏览器，静态检查 KaTeX delimiters 配置 `$`/`$$` 正常、details 结构正常、TOC 锚点由 JS 动态生成）；validate.py（审查任务禁止改文档，未运行）。

## 问题

- [阻断·技术] overview.html L52：「DeepSeek-V2 是 128 头 × 128 维 × 60 层，128K 上下文下整个 KV cache 约 256 GB（BF16）」与 index.html L653 同配置下「约 480 GB」矛盾。按公式 2·n_h·d_h·l·2bytes·128K = 2×128×128×60×2×131072/1024³ ≈ 480 GB；256 GB 恰为漏算 K/V 两份（仅算一份：128×128×60×2×131072/1024³ ≈ 256 GB）。overview 快速阅读页核心数字错误。：将「约 256 GB」改为「约 480 GB」并与 index.html 一致；若引别的 baseline 须显式标注。 ｜ 修复：已将 overview.html L52「约 256 GB」改为「约 480 GB（BF16）」，并在「$2 n_h d_h$ 个数」后补「（K 和 V 各一份）」显式标注 K/V 两份，与 index.html L653 一致。 ｜ 复验：
- [重要·盲读] index.html 学习目标 L661「为什么 MLA 要把 RoPE 解耦」与 context-box L678「解耦 RoPE 维度 $d_h^R$」在 RoPE 未讲解（S3 才讲）前即使用术语，且 context-box 位于 S1 之前，小白无法理解「解耦 RoPE 维度」含义。：在 context-box $d_h^R$ 标签后加括注「（RoPE 在 S3 讲解，此处先列维度）」；学习目标 RoPE 首现处加「（旋转位置编码，见 S3）」。 ｜ 修复：已在学习目标 L661「RoPE」首现处加「（旋转位置编码，见 S3）」；在 context-box L678「解耦 RoPE 维度 $d_h^R$」标签后加「（RoPE 在 S3 讲解，此处先列维度）」。 ｜ 复验：「K3 §2.1.2 末段说这个 full-rank 参数化与 KDA 的 output gate 一致，是 K3 相对 Kimi Linear 的一个改动」来源定位错误：k3-report §2.1.2（L352-365）仅称 "input-dependent, channel-wise full-rank output gate"，未说「与 KDA 一致」也未说「相对 Kimi Linear 改动」；这两点实际在 §2.1.1 L347（"Kimi K3 changes KDA's output gate from the low-rank parameterization used by Kimi Linear to an input-dependent full-rank projection"）。：将该句来源由「K3 §2.1.2 末段」改为「K3 §2.1.1 Full-rank gate 段」，或删去不在 §2.1.2 的「与 KDA 一致 / 相对 Kimi Linear 改动」论断。 ｜ 修复：已将 L960 details 末句来源由「K3 §2.1.2 末段」改为「K3 §2.1.1 Full-rank gate 段」，重述为「K3 把 KDA 的 output gate 从 Kimi Linear 用的低秩参数化改为 input-dependent full-rank projection，MLA 的 output gate 沿用同一参数化」；同步将 L938 主文同一引文来源由「§2.1.2 末段」改为「§2.1.1」。 ｜ 复验：
- [重要·盲读] index.html S5 L933 gate 公式及 L946 手算用 $\mathbf{x}_t$ 作输入，但全文 MLA 输入自 S1 L690 起一直用 $\mathbf{h}_t$，$\mathbf{x}_t$ 与 $\mathbf{h}_t$ 关系未说明。来源对照：DeepSeek-V2 论文用 $\mathbf{h}_t$，k3-report §2.1.2 用 $\mathbf{x}_t$——页面切符号跟随报告但未桥接，小白无法判断二者是否同一量。：在 S5 改动二公式 $\mathbf{x}_t$ 首现处加一句「$\mathbf{x}_t$ 即前文输入 $\mathbf{h}_t$（K3 报告记为 $\mathbf{x}_t$，本文沿用报告符号）」。 ｜ 修复：已在 S5 改动二 gate 公式前的引导句加「（下面公式中的 $\mathbf{x}_t$ 即前文输入 $\mathbf{h}_t$——K3 报告记为 $\mathbf{x}_t$，本文 S5 沿用报告符号以与引文一致）」。 ｜ 复验：
- [重要·盲读] index.html S5 L956「gate 在通道 0 放大（gate > 0.5，原值 1 → 0.73，相对其他通道被强调）」表述与数值方向矛盾：Sigmoid 输出恒在 (0,1)，所有通道绝对值都被缩小（1→0.73 是缩小非放大），「放大」先入为主误导，括号补「相对被强调」不足以纠正。：将「放大」改为「相对强调」或「缩小最少」，统一用「相对强调/抑制」描述门效果（因 Sigmoid 门对所有通道都是绝对缩小）。 ｜ 修复：已将 L956「gate 在通道 0 放大（gate > 0.5，原值 1 → 0.73，相对其他通道被强调）」改为「gate 在通道 0 相对强调（gate > 0.5，原值 1 → 0.73，Sigmoid 对所有通道都是绝对缩小，但此通道缩小最少，相对被强调）」，与数值方向一致。 ｜ 复验：
- [重要·盲读] index.html S3 L832「RoPE 部分 $\mathbf{k}_t^R$ 不能吸收，必须单独缓存」推理跳步：「不能吸收」到「必须单独缓存」缺一步——$\mathbf{k}_t^R$ 由 $\mathbf{h}_t$ 直接投影（论文 Eq.(15) $W^{KR}\mathbf{h}_t$，不经 $\mathbf{c}_t^{KV}$），推理时不保留 $\mathbf{h}_t$，故须缓存算好的 $\mathbf{k}_t^R$。小白无法从「不能吸收」直接推出「必须缓存」。：在「必须单独缓存」前加「$\mathbf{k}_t^R$ 由 $\mathbf{h}_t$ 直接投影（不经 $\mathbf{c}_t^{KV}$），推理时不再保留 $\mathbf{h}_t$，故 $\mathbf{k}_t^R$ 须事先算好缓存」。 ｜ 修复：已在 L832「RoPE 部分 $\mathbf{k}_t^R$ 不能吸收，必须单独缓存」改为「RoPE 部分 $\mathbf{k}_t^R$ 不能吸收。$\mathbf{k}_t^R$ 由 $\mathbf{h}_t$ 直接投影（论文 Eq.(15) $W^{KR}\mathbf{h}_t$，不经 $\mathbf{c}_t^{KV}$），推理时不再保留 $\mathbf{h}_t$，故 $\mathbf{k}_t^R$ 须事先算好单独缓存」，补全推理链。 ｜ 复验：
- [轻微·盲读] index.html L653 从「每 token 每层要缓存 $2 n_h d_h$ 个数」直接跳到「共 $1{,}966{,}080$ 个数/token」未显式说明乘以层数 60；且 $n_h, d_h$ 在公式中首现，定义要到 context-box（L675-676）才给；「BF16」首现未说明为 2 字节。：在「共 $1{,}966{,}080$」前加「× 60 层 =」；首次出现 $n_h, d_h$ 处加「（头数 / 每头维度，见下表）」；BF16 首现处加「（2 字节）」。 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html S2 L754「把 $W^{UK}$ 吸收进 $W^Q$」——S1 中 MLA 的 query 经 $W^{DQ}, W^{UQ}$，$W^Q$ 是 MHA 符号。虽与论文原文表述一致（论文亦作 "absorbed into $W^Q$"，来源一致非错误），但 MLA 语境下 $W^Q$ 指代未点明，小白需自行对应。：在「$W^Q$」后加括注「沿用论文符号，指 MLA query 的最终投影」或注明「即 $W^{UQ}$ 链的等价合并」。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html L653「必须为前序所有 token 保留 K 和 V」中「K 和 V」首次出现未展开为 key/value，完全小白可能不知缩写。：首次出现「K 和 V」改为「key（K）和 value（V）」。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html S4 L859-882 表格标题为「每 token 每层 cache」，但 L875 代入数值时直接给「乘以层数 $l$」后的总值（MHA: 2×128×128×60=1,966,080），而 MLA/MHA 比值（L881）又用单层（576/32,768）。每层与总值口径混用，小白需自行换算。：代入数值处分两步列——先「每层」（MHA 32,768 / MLA 576），再「×60 层 = 总值」（1,966,080 / 34,560），比值统一标注「单层或总值同比例」。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 1 / 重要 5 / 轻微 4
- 处置：进入修复。阻断项（overview.html L52 的 256 GB 算术错误）须先关闭方可发布；重要项均可在最小化修改范围内关闭，无需改变研究范围或教学大纲。
- 流程限制遗留：scope.md 学习目标闭环未核对（受 research/ 读取禁令限制），发布门控该项须由编排者另开上下文确认；DeepSeek-V2 完整配置表（n_h/d_h/l/d/d_c'）建议修复者对照原文 §2.1.2 末段复核；validate.py 与浏览器实测公式渲染未运行，发布门控前须补。
