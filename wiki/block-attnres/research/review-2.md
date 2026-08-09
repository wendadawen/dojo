# Block AttnRes 独立审查（第二次）

- 审查者：独立上下文（AI 模拟小白读者 / 第二次审查，未参与生成与第一次审查）
- 页面版本：index.html ac5b7442 / overview.html ac5b7442（HEAD bcb38d6，工作树无未提交修改）
- 时间：2026-08-09 17:10 +0800
- 输入：wiki/block-attnres/index.html、wiki/block-attnres/overview.html、Kimi K3 技术报告 §2.2（/tmp/kimi-k3-research/k3-report.txt L377–419）与 §2 开头（L203–217）、HuggingFace 官方 config.json（https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json）
- validate.py：`python3 .dojo/scripts/validate.py wiki/block-attnres/index.html` → exit 0（`validation ok`）

## 段 A 盲读

按页面顺序通读 overview.html 与 index.html。学习目标 5 条全部由正文章节回答（S1 回答目标1、S2 回答目标2、S3 回答目标3、S4 回答目标4、S5 回答目标5）。主线顺畅，公式—图示—手算—对照表—伪代码衔接清晰，证据层级标注（K3 报告原文 vs 源码核对 vs 推算）做得比较诚实。

主线卡点集中在符号一致性：S1 用 $F_l$（大写）表示层变换、$h_l$ 表示层输入，S2 Eq.(8) 改用 $f_i$（小写）且 $f_i(h_i)$ 中的 $h_i$ 未重申是"第 $i$ 层输入/残差流"。小白在 S1→S2 切换时会怀疑 $F$ 与 $f$ 是否同一对象、$h_i$ 是否还是 S1 的残差流。其余章节主线顺畅。

## 段 B 对照来源

K3 报告 §2.2（L377–419）逐句核对页面 C1–C7、F1–F4：标准残差瓶颈类比、Full AttnRes 公式 Eq.(8)(9)（含 $k_i=v_i$、$\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k))$、$\alpha_{i\to l}$、$h_l=\sum\alpha v$）、RMSNorm 防大值主导、Block AttnRes 分块 Eq.(10)（含 $b_n$、$b_0=h_1$、partial sum、两种候选集合）、内存 $O(Ld)\to O(Nd)$、末尾聚合、$N\approx 8$ 经验结论与 K3 的 8 块×12 层——均与原文一致。HuggingFace config.json WebFetch 直接核对：`attn_res_block_size=12`、`num_hidden_layers=93`、`hidden_size=7168`、`num_attention_heads=96`、`linear_attn_config.full_attn_layers`（24 个索引）、`linear_attn_config.kda_layers`（69 个索引）——全部一致；$93=7\times 12+9$ 的 partial block 推算与 config.json 的层索引范围吻合。手算数字逐位复算：S2 Full AttnRes 6 候选（$h_6\approx[0.703,0.703]$）、S3 Block AttnRes 4 候选（$h_6\approx[1.059,1.241]$）、S5 加 RMSNorm 重算（权重 $[0.21,0.26,0.27,0.26]$、$h_6\approx[1.004,1.056]$）——全部正确。前置概念页 residual-connection、kimi-k3-dataflow 链接目标文件存在。详见下方问题。

## 问题

- [轻微·盲读] index.html L689（S1）用 $F_l$ 表示层变换、$h_l$ 表示层输入，L749/751（S2 Eq.(8)）改用 $f_i$（小写）且未声明与 $F_i$ 的对应关系：在 S2 首次出现 $f_i$ 处补一句"$f_i$（即 S1 的 $F_i$，此处用小写以与 K3 报告 §2.2 原文一致）"或直接在 S1 统一用 $f$ ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html L751（S2 Eq.(8)）的 $f_i(h_i)$ 中 $h_i$ 未明确其含义；K3 报告 §2.2 L390 原文说明"$f_i(h_i)$ is the output of layer $i$"，隐含 $h_i$ 是第 $i$ 层输入，但页面未点出这层关系，小白可能将 $h_i$ 与 S1 的 $h_l$（残差流）断开理解：在 L751 或 L753 首次出现 $f_i(h_i)$ 处补一句"$h_i$ 是第 $i$ 层的输入（残差流），$f_i(h_i)$ 是其输出" ｜ 修复： ｜ 复验：
- [轻微·技术] index.html L1045、L1192（C8）把"每个 module（attention 模块与 MLP 模块）各加权一次"标为"K3 报告原文确认"，引用 §2 L209 "AttnRes enable each module to selectively retrieve representations"。但原文只说"enable each module to selectively retrieve representations"，并未直接断言"每个 module 各加权一次"——"各加权一次"是从"each module"推出的合理推断，具体加权次数应由源码确认（页面 L1046 也承认"具体三次位置来自源码核对"）。把推断标为"原文确认"会抬高证据层级：把 C8 中"K3 报告原文确认：每个 module 各加权一次"改为"K3 报告 §2 L209 'each module' 推断（attention 模块与 MLP 模块各一次）+ §2.2 L414 末尾聚合；具体加权次数与位置由源码核对（间接证据）"，与 L1043–1047 的证据层级表述保持一致 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html L1014（S4）"9 个候选来源（8 块快照 + 当前流）"中"8 块快照"= $b_0$（embedding）+ $b_1\dots b_7$（7 个完整 block 求和），但 K3 实际有 8 个层 block（block 1–8），block 8 因 partial 用 $b_8^{i-1}$ 而非完整快照。小白可能困惑"为什么 8 块快照不含 block 8"：在 L1014 把"8 块快照"改为"8 个历史快照（embedding + 7 个完整 block 求和；block 8 为 partial，其当前累加用 partial sum $b_8^{i-1}$ 作为'当前流'单独列出）"或类似显式说明 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：进入修复。四个轻微项均为符号一致性、证据层级标注与表述清晰度的最小化修复，不涉及核心公式、手算数字或 K3 配置的正确性。无需改变研究范围或教学大纲。
