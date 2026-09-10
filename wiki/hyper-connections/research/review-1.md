# 超连接审查记录（第 1 轮）

- 页面版本：130be2be2465cf650c210d98dbb51df8b8cde57d（`git hash-object index.html`）
- 审查时间：2026-09-10 21:05
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节：引言、核心问题、常见误解、1. 恒等映射：残差连接保住了什么、2. 超连接：把一条流加宽成 $n$ 条（含 `<details>` 代码块「n=2 的 HC 层与 n=1 退化验证」）、3. 无约束的代价与双随机约束（含代码块「无约束链与双随机链的 24 层对照」）、4. Sinkhorn-Knopp：把矩阵投影到双随机流形（含「2×2 矩阵手算两步 Sinkhorn」与「3×3 正矩阵的 Sinkhorn 收敛与谱范数」两个折叠块）、5. 落地：GLM-5.3-Flash 的 mHC、来源与范围说明（论断与来源 C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）

## 机械验证结果

命令与结果：

1. `/usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py .../hyper-connections/index.html`
   → `validation ok`（EXIT=0）

2. 可运行代码块实跑（`/usr/bin/python3`，torch 2.8.0），逐字符比对页面「预期输出」：
   - 第 2 章代码块：实测 `n=1 输出: [1.1, 2.2, 3.3, 4.4]` / `普通残差: [1.1, 2.2, 3.3, 4.4]` / `最大差: 0.0` / `n=2 输出行 0: [1.1, 2.2, 3.3, 4.4]` / `n=2 输出行 1: [1.05, 2.1, 3.15, 4.2]` / `两行是否已不同: True`——与页面完全一致。
   - 第 3 章代码块：实测 `无约束链 24 层, 复合矩阵行和绝对值最大: 1.095e+06` / `双随机链 24 层, 复合矩阵行和: [1.0, 1.0, 1.0, 1.0]`——与页面完全一致。
   - 第 4 章代码块：实测迭代 1–5 行/列和、`迭代 20: 行和最大偏差 0.0e+00, 列和最大偏差 0.0e+00`、`最大奇异值: 1.0`——与页面 12 行预期输出逐字符一致。
   - 第 4 章手算折叠块复算：非对称例列和 $(10,3)$→行和 $(1.2333,0.7667)$→行归一后列和 $(0.8601,1.1399)$→再列归一后行和 $(1.0855,0.9145)$，行和偏差 $0.2333\to0.0855$，与页面数值一致。

3. 引用编号双向闭合（程序提取 `<sup>[X]</sup>` 与 `<strong>[X]</strong>`）：
   - 正文引用 → 来源定义：空集 ✓
   - 来源定义 → 正文引用：**非空** = {F7, F8, N4}（见问题 2）

4. 相邻双上标：命中 `<sup>[C7][N1]</sup>`、`<sup>[C11][N2]</sup>`（见问题 1）

5. Unicode 数学字符（排除 `<pre>` 代码块）：命中 U+00D7「×」3 处，均在标题/正文中（见问题 3）；U+2212 未出现；`→`（U+2192）2 处出现在第 5 章 SVG 的 `foreignObject` 文字标号中，属语句连接符。

6. TAB 字符：`index.html` 0 个，`overview.html` 0 个 ✓

7. 残留占位符（待生成/TODO/TBD/占位/FIXME/待补）：0 ✓

8. `<head>` 五项元数据：`description`（纯文本）✓、`dojo:summary`（含 `$...$` 可渲染）✓、`dojo:type=concept` ✓、`dojo:topics=模型结构`（在 AGENTS.md 第 29 行固定大类表内）✓、`dojo:tag=残差结构` ✓

9. 互链：`index.html` 导航含 `href="overview.html"` ✓；`overview.html` 导航含 `href="index.html"`（「完整说明 →」）✓

10. 前置概念链接：`../residual-connection/index.html`、`../rmsnorm/index.html`、`../block-attnres/index.html` 三页均存在 ✓

11. 图示：两个 `<figure class="diagram">` 内 `<svg>` 的 `<text>` 元素数为 0，标签全部由 `foreignObject > div.dg-label` 承载并由 KaTeX 渲染 ✓

## 问题

- [重要·技术] 第 5 章及来源章节 [C11]/[C12]/[C13]/[F8] 的源码标注：引用的 `models/glm5_next/modeling_glm5_next.py`（L219-302、L286-290、L283-294、L1316-1318、L1477、L1493）在指定源码根 `/Users/wendadawen/code/dojo/wiki/deepseek-v4-1/research/official/inference/` 下不存在该文件，全仓亦无 `modeling_glm5_next.py`；该根目录只有 `model.py`（1310 行）与 `kernel.py`（591 行），被引 L1316-1318/L1477/L1493 超出文件末尾，L286-290 落在 `RMSNorm` 上而非 Sinkhorn 迭代序。因此这些行号无法按标注定位核对。｜引文依据：等价实现在指定文件内可核对且与页面机制一致——`kernel.py` L427 `pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps`；L429 `post[i, j] = 2 * T.sigmoid(...)`；L436 `# comb = comb.softmax(-1) + eps`；L445-458 先列归一一次、再 `for _ in T.serial(sinkhorn_iters - 1)` 循环「行归一+列归一」（末步为列）；`model.py` L938 `mix_hc = (2 + hc_mult) * hc_mult`，L941-946 `hc_attn_fn/ffn_fn/base/scale` 两套独立参数，L962-966 `y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`；`config.json` L40-42 `hc_mult=4 / hc_sinkhorn_iters=20 / hc_eps=1e-06`。GLM 侧数字（45 层、hidden_size=4096、$24\times16384+24+3=393{,}243$、35,391,870、0.011%、列和偏差 $1.07\times10^{-6}$、行和偏差 $1.01\times10^{-2}$）可由 `wiki/glm-5-3-flash-dataflow/research/config.json`（`num_hidden_layers=45`、`hidden_size=4096`）与 `p3.out`（§3.1/§3.5/§3.8）交叉核对。｜修复要求：使每处源码标注可定位——或注明该实现属外部 `transformers` 仓库（离线来源）并给出可在本机复现的等价证据（含文件与行号），或改为引用本机可定位实现的文件与行号。｜修复：第 5 章与 [C11]/[C12]/[C13]/[F8] 的源码标注全部改到本机实际存在的机制等价实现——`wiki/deepseek-v4-1/research/official/inference/kernel.py` L407–474（pre L427、post L429、comb L430–443、Sinkhorn 序 L445–458、分母 ε L448/L454）与 `model.py` L938、L941–946、L948–955、L962–966、L1257–1258；GLM 侧规格与数字改引 `wiki/glm-5-3-flash-dataflow/research/config.json`（L16/L17/L15、L20/L240）与 `p3.out`（§3.1–3.8）、`verify_structure.out` 参数量合计项 [10]。删除 `models/glm5_next/modeling_glm5_next.py` 及 L219-302/L286-290/L283-294/L1316-1318/L1477/L1493，并在 [C11] 注明该 transformers 源文件未收录在本机、故不引其行号；正文删去无法核对的类名 `Glm5NextTextHyperConnection`，meta 主要依据同步改写。｜复验：全页无 `modeling_glm5_next`/`Glm5NextTextHyperConnection`/`models/glm5_next` 及旧行号残留；新引行号逐一到本机文件核对命中（kernel.py L427/L429/L436/L448/L454、model.py L938/L962、config.json L15-17/L20/L240）；`validate.py` → `validation ok`。

- [轻微·格式] 相邻双上标：`<sup>[C7][N1]</sup>`（第 3 章）与 `<sup>[C11][N2]</sup>`（第 5 章）在同一 `<sup>` 内并列两个方括号，不符合 `style-guide.md` §6 给出的组合写法。｜引文依据：不适用｜修复要求：改为 `<sup>[C7, N1]</sup>`、`<sup>[C11, N2]</sup>`（或拆成前后各一个 `<sup>`）。｜修复：第 3 章 `1.6<sup>[C7][N1]</sup>` 改为 `<sup>[C7, N1]</sup>`；第 5 章 `<sup>[C11][N2]</sup>` 改为 `<sup>[C11, N2]</sup>`。｜复验：`<sup>\[[^\]]*\]\[[^\]]*\]</sup>` 与 `</sup>\s*<sup>` 两种模式全页命中 0。

- [轻微·格式] 来源定义未被正文引用：`[F7]`（Sinkhorn 数值行为）、`[F8]`（GLM 回写式）、`[N4]`（无约束链 24 层行和 $1.095\times10^{6}$）在「来源与范围说明」中定义，但正文无对应 `<sup>`（如第 5 章表格「回写式」行、两处代码块说明均未标注），双向闭合差集非空。｜引文依据：不适用｜修复要求：为 F8/F7/N4 各补一处正文上标，或删除对应来源条目，使两个差集均为空。｜修复：[F8] 补在第 5 章表格「回写式」行；[F7] 补在第 4 章「$3\times3$ 的完整收敛过程与谱范数检验」句；[N4] 补在第 3 章「观察重点」的「放大到约 $10^6$」处。｜复验：程序提取正文 `<sup>[X]</sup>` 与来源 `<strong>[X]</strong>`，两个方向差集均为空。

- [轻微·格式] Unicode 乘号 U+00D7 直接出现在标题/正文：summary「展开：2×2 矩阵手算两步 Sinkhorn」「代码：3×3 正矩阵的 Sinkhorn 收敛与谱范数」与正文「两个 2×2 矩阵手算两步」。违反 `style-guide.md` §11（数学运算符须用 LaTeX），且与「来源与范围说明」中已用 `$2\times2$`、`$3\times3$`、`$4\times4$` 的写法不一致。｜引文依据：不适用｜修复要求：三处 `2×2`/`3×3` 改写为 `$2\times2$`/`$3\times3$`，全页写法统一。｜修复：三处改为 LaTeX——details summary「$2\times2$ 矩阵手算两步 Sinkhorn」「代码：$3\times3$ 正矩阵的 Sinkhorn 收敛与谱范数」与正文「两个 $2\times2$ 矩阵手算两步」。｜复验：`×`（U+00D7）在公式定界符与 `<pre>` 之外命中 0；`validate.py` 的 bare-math 检查通过。

- [轻微·技术] 第 5 章末 Block AttnRes 段落含机制与归因陈述（「块级注意力检索（用 pseudo-query 检索前序层的输出）」「机制是检索而非线性混合」「不需要流形约束」），但无 `<sup>` 来源标注，也未列入「论断与来源」。｜引文依据：所述机制可在链接页核对（`wiki/block-attnres/index.html` meta：「每层用可学习 pseudo-query 对前序块表征做 softmax 加权检索」「块内求和、块间 attention」），内容与页面一致，仅缺引用标注。｜修复要求：补一条 `[C]` 来源并加 `<sup>`，或在句中明确该段为对链接页结论的概述。｜修复：新增来源条目 [C14]（Block AttnRes 用可学习 pseudo-query 对前序块表征做 softmax 加权检索、块内求和块间注意力、不施流形约束；来源 `wiki/block-attnres/index.html`），正文该段补 `<sup>[C14]</sup>`。｜复验：正文/来源双向引用闭合；[C14] 描述与链接页 meta（pseudo-query softmax 检索、块内求和块间 attention）一致。

- [轻微·技术] [C1] 末句「该思想在 ResNet 原始论文（he2016）确立，HC 论文 §1 引用同一思想但未使用 identity 一词」中「HC 论文 §1 引用同一思想」不可定位。｜引文依据：HC 论文全文 `identity mapping` 仅出现 1 次（Appendix H：「let $\mathcal{T}^{0}$ be an identity mapping」，语义与 §1 无关）；HC §1 讨论的是 seesaw 与「residual connections... predefine the strength of connections between the output and input within a layer」，未提及恒等映射思想。（ResNet 归属一句可由 mHC 论文 §1「early research (He et al., 2016b) revealed that the identity mapping property of the residual connection maintains stability and efficiency during large-scale training」支持。）｜修复要求：给出 HC §1 对应原文片段，或删除「HC 论文 §1 引用同一思想但未使用 identity 一词」这一从句。｜修复：删除「HC 论文 §1 引用同一思想但未使用 identity 一词」从句，[C1] 改为引 mHC 论文 §1 原文（identity mapping 定义）并保留对 He et al., 2016b 的归属。｜复验：新引文与 mHC 论文 §1 原文逐字一致（HTML 版核对）；HC 论文 §1 不再作为该论断依据。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复

（本轮未发现核心结论错误：HC 论文 Eq. 2/3/4/5 三映射式、Eq. 15/16 非可训练矩阵、mHC 论文 Eq. 1/2/4 递归展开与复合映射、Eq. 6 双随机约束、§4.1 三性质与 $n=1$ 退化、§4.2 Eq. 7/8/9 参数化与 $t_{\max}=20$、§3.1/§5.4 的 3000/12k/1.6 数字，均已逐句定位到原文并给出片段；三个可运行代码块输出与页面描述逐字符一致。）
