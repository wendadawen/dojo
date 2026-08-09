# Vision Transformer（ViT）独立审查（第二次）

- 审查者：独立上下文（AI 模拟）
- 页面版本：149ac543f66080d08262739aefb279e474f4a327（index.html 工作树哈希）
- 时间：2026-08-09

## 审查范围

- 输入：`wiki/vit/index.html`、`wiki/vit/overview.html`、`guides/concept/check.md`、`guides/concept/content-examples.md`
- 外部来源：WebSearch "ViT Dosovitskiy 2021 arxiv 2010.11929"（patch embedding 公式 Eq.1-4、Table 1 模型配置、Table 2 实验结果、"large scale training trumps inductive bias"原文、pre-LN 与 1D 位置编码选择）
- 未读取 `research/` 目录、未读取仓库中其他概念页（仅验证链接路径存在）

## 段 A 盲读小结

按页面顺序阅读，四个学习目标在正文中均得到完整回答：

1. ViT 要解决什么问题 → "ViT 要解决什么问题"章回答（CNN 的 locality + translation invariance 在小数据下是优势、大数据下成为限制；ViT 主动去除空间归纳偏置）
2. 写出 $z_0$ 公式并手算 224×224 patch 16 → 196 token、197 序列长度 → "图像如何变成 token 序列"章回答（三步合成公式 + 手算例子 + 形状对照表）
3. 写出 $z_\ell', z_\ell, y$ 三条公式并说明与标准 Transformer 的三处差异 → "Transformer 编码块"章回答（pre-LN、encoder-only、Q/K/V 来源）
4. 数据规模边界 → "数据规模边界"章回答（小数据从零训练不如 ResNet；JFT-300M 预训练后 ViT-H/14 达 88.55% 超越 BiT-L 87.54% 且用更少 TPUv3-core-days）

术语首现解释充分（归纳偏置、locality、translation invariance、patch、token、class token、位置编码、pre-LN、MSA、MLP、残差、$z_L^0$ 均有定义或最小概念说明）。推导无跳步：$N=HW/P^2$、$d_k=D/h$、注意力矩阵 $197\times 197$ 均给出代入。手算例子（224×224 patch 16 → 196 token、ViT-B/12 d_k=64）可复算。三个折叠块（BERT [CLS] 来源、1D vs 2D 位置编码、class token vs GAP）前后均能接回主线。前置链接 ../../wiki/standard-attention/index.html、../../wiki/residual-connection/index.html、../../wiki/positional-encoding/index.html、../../wiki/moonvit-v2/index.html 路径均存在；待生成概念页（CNN、矩阵乘法、LayerNorm、GELU）均有占位提示和最小概念说明，符合 C3 规则。

## 段 B 对照来源小结

1. 定义与机制：[C1] ViT 主动去除空间归纳偏置与论文 §1 "avoiding binding priors"、"large scale training trumps inductive bias" 一致；[C2] patch 切分 + 线性投影与 §3.1 Eq.(1) + Figure 1 一致；[C3] class token 借鉴 BERT [CLS] 与 §3.1 一致；[C4] 可学习 1D 绝对位置编码与 §3.1 Eq.(1) 的 $+E_{\text{pos}}$ 一致，附录 D.4 消融对照 1D vs 2D 差异不显著；[C5] pre-LN + MSA + 残差 + MLP 与 §3.1 Eq.(2)-Eq.(3) 一致，pre-LN 稳定性引用 Xiong et al. 2020 (arXiv:2002.04745)；[C6] 分类头 $y=\text{LN}(z_L^0)$ 与 §3.1 Eq.(4) 一致；[C7] 数据规模边界与 §4.3 小数据结果、§4.2 Table 2 大数据结果、§4.4 结论一致；[C8] 模型配置 ViT-B/L/H 与 Table 1 一致。
2. 公式与推导：[F1] $z_0=[x_{\text{class}};x_p^1 E;\ldots;x_p^N E]+E_{\text{pos}}$、$E\in\mathbb{R}^{P^2 C\times D}$、$E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$ 与 §3.1 Eq.(1) 一致；[F2] $z_\ell'=\text{MSA}(\text{LN}(z_{\ell-1}))+z_{\ell-1}$ 与 Eq.(2) 一致；[F3] $z_\ell=\text{MLP}(\text{LN}(z_\ell'))+z_\ell'$ 与 Eq.(3) 一致；[F4] $y=\text{LN}(z_L^0)$ 与 Eq.(4) 一致。符号 $x, P, N, E, x_{\text{class}}, E_{\text{pos}}, D, z_0, \ell, L, \text{LN}, \text{MSA}, \text{MLP}, z_L^0$ 首现处均有定义。
3. 可运行代码：页面无可运行代码块，只有 ASCII 围道图与公式，不适用。
4. 事实与推断：[N1] ViT-Base 12 层/D=768/12 头/MLP 3072/86M、ViT-Large 24 层/D=1024/16 头/MLP 4096/307M、ViT-Huge 32 层/D=1280/16 头/MLP 5120/632M 与 Table 1 一致；[N2] 224×224 patch 16 → 196 token、197 序列长度、$z_0\in\mathbb{R}^{197\times 768}$ 计算正确；[N3] Table 2 数字 ViT-L/16 (I21k) 85.30±0.02/0.23k、ViT-L/16 (JFT) 87.76±0.03/0.68k、ViT-H/14 88.55±0.04/2.5k、BiT-L 87.54±0.02/9.9k、Noisy Student 88.4-88.5/12.3k 与论文 Table 2 一致。"large scale training trumps inductive bias" 直接引自论文 §1 原文。
5. 前置知识引用：standard-attention、residual-connection、positional-encoding、moonvit-v2 四个目录均存在，链接层级正确；CNN、矩阵乘法、LayerNorm、GELU 四个待生成概念页均有占位提示和最小概念说明。
6. 教学简化：四项简化（CNN 归纳偏置只取两个、MSA 内部公式引用 standard-attention、只引用 ImageNet top-1 与 TPUv3-core-days、变体家族一句话提及）均有说明，不影响核心结论。形状对照表特别标注"$P^2 C=768$ 恰好等于 $D=768$ 是 ViT-B/16 的巧合，不是必须"——教学诚实标注到位。
7. 页面功能：KaTeX 渲染（delimiters 配置 $$...$$ 与 $...$、throwOnError:false）、details 折叠、自动生成目录锚点（h2 均有显式 id: why-vit、patch-embedding、transformer-encoder、data-scale-boundary、sources-and-teaching-notes）结构正确。

数字复算：$N=HW/P^2=224\times 224/16^2=50176/256=196$ ✓；$N+1=197$ ✓；$P^2 C=16\times 16\times 3=768$ ✓；$d_k=D/h=768/12=64$ ✓；注意力矩阵 $197\times 197$ ✓；$P=32\to N=49$、$P=14\to N=256$ ✓；2.5k/9.9k≈1/4、2.5k/12.3k≈1/5 ✓；ViT-B/L/H 参数 86M/307M/632M 与 Table 1 一致 ✓。

## 问题

（无）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布（审查范围内；发布门控的 validate.py、代码重跑等由编排者执行）
