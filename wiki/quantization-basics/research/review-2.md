# 量化基础独立审查（第二次）

- 审查者：独立上下文（AI 模拟小白读者 + 来源对照）
- 页面版本：wiki/quantization-basics/index.html（1319 行）、overview.html（83 行）
- 时间：2026-08-09
- 来源：[G] Gholami et al. 2021, arXiv:2103.13630；[J] Jacob et al. 2018 CVPR；[OCP] Rouhani et al. 2023, Microscaling Formats v1.0

## 段 A 盲读

按页面顺序阅读，记录小白读者理解主线上的卡点。

1. 开篇 callout 用"7B 模型 FP16=14GB 装不下 24GB 显卡"引入问题，直觉且具体。"量化决策的四要素"框给出位宽、对称性、粒度、训练方式四个维度，帮助读者建立全局框架。

2. S1（"量化要解决什么问题"）算清压缩比：FP16→INT8=2×、FP16→INT4=4×。"INT4 对称有 15 个等级"正确（含 0，±1 到 ±7）。"位宽越低网格越粗"的权衡清晰。

3. S2（"均匀量化怎么把浮点变成整数"）给出仿射量化公式 F1 与反量化公式。符号 $s$、$z$、$q_{\min}$、$q_{\max}$ 首现时有定义。两个验证检查（$x=0$ 映射到 $z$、网格点零误差）帮助巩固理解。对称特例 F2 清晰。

4. S2 的 ASCII 量化网格图（`<pre class="diagram">`）直观展示浮点→网格映射。但该 `<pre>` 块末尾缺少 `</pre>` 闭合标签，且以 `</p>` 结尾——这是 HTML 格式错误（见问题 1）。浏览器可能自动闭合，但 `</p>` 会以可见文本出现在图中。

5. S2 手算 $[1.2, 3.4, 5.6]$ INT4 对称：$s=0.8$，$x_q=[2,4,7]$，$\hat{x}=[1.6,3.2,5.6]$，误差 $[+0.4,-0.2,0]$。代码复算一致 ✓。舍入约定（四舍五入 vs banker's rounding）在折叠块中说明。

6. S3（"量化误差从哪来"）给出三来源：离散级粗、舍入、裁剪。离群值例子 $[1.2,3.4,5.6,50.0]$ 展示 per-tensor 的单点故障，有说服力。完整手算在折叠块中。

7. S4（"对称/非对称 与 量化粒度"）清晰对照 per-tensor/per-channel/per-block。有效位宽计算（$32×4+8=136$ bit，$136/32=4.25$ bit/元素）正确。"粒度必须与位宽一起声明"的 callout 是好的提醒。

8. S5（"PTQ vs QAT"）覆盖 PTQ 流程、QAT 伪量化+STE、MXFP4 浮点量化。STE 的"透明窗户"类比在 callout 中标注边界。MXFP4 的块结构、E8M0 scale、E2M1 元素清晰展开。

9. 逐题核对学习目标：
   - "量化要解决什么问题、为什么会引入精度损失？" → S1 ✓
   - "给定浮点权重和位宽，如何用对称均匀量化编码并反量化？" → S2 ✓
   - "对称与非对称、per-tensor/per-channel/per-block 的差别？" → S4 ✓
   - "量化误差的三个来源？离群值为什么让误差爆炸？" → S3 ✓
   - "PTQ 与 QAT 的差别？MXFP4 相对定点量化的优势？" → S5 ✓

   全部学习目标由正文章节完整回答。

## 段 B 对照来源

### 1. 定义与机制

- C1（仿射量化公式）：标准均匀量化公式。[G] 覆盖此内容（Uniform Quantization 小节）。公式 $x_q=\mathrm{clip}(\mathrm{round}(x/s)+z, q_{\min}, q_{\max})$，$\hat{x}=s(x_q-z)$ 与文献一致 ✓
- C2（对称量化 $z=0$，范围 $[-2^{b-1}+1, 2^{b-1}-1]$，$s=\max(|x|)/(2^{b-1}-1)$）：标准 AbsMax 方法。[G] 覆盖 ✓
- C3（非对称 $s=(\beta-\alpha)/(q_{\max}-q_{\min})$，$z=\mathrm{round}(q_{\min}-\alpha/s)$）：推导正确（要求 $\alpha\mapsto q_{\min}$，代入解出 $z$）✓
- C4（粒度 per-tensor → per-channel → per-block）：[G] 覆盖此分类（Layerwise/Channelwise/Sub-channelwise）✓
- C5（误差三来源：离散级粗、舍入、裁剪）：[G] 讨论 rounding 与 truncation，页面将误差组织为三来源是教学合成，合理 ✓
- C6（离群值毒化 per-tensor）：[G] 讨论离群值对 scale 的影响。emergentmind 确认"outliers no longer poison their channel" ✓
- C7（PTQ 校准 128–512 样本）：[G] §2.5 讨论校准。128–512 是常见工程值 ✓
- C8（QAT 伪量化前向 + STE 反向 $\partial\hat{w}/\partial w\equiv 1$）：[J] Jacob et al. 2018 CVPR 是 STE 的经典来源。[G] §2.5 讨论 QAT ✓
- C9（MXFP4 块结构：块 32 + E8M0 scale + E2M1 元素）：[OCP] 规范确认。zeroentropy.dev 确认"blocks of 32 elements share a single 8-bit power-of-2 scale (E8M0); each element is a 4-bit micro-float (E2M1)" ✓

### 2. 公式与推导

- F1（仿射量化与反量化）：公式标准，符号首现时有定义 ✓
- F2（对称量化特例）：$z=0$ 下的特例，推导正确 ✓
- F3（非对称 scale/zero-point）：要求 $\alpha\mapsto q_{\min}$，推导 $z=\mathrm{round}(q_{\min}-\alpha/s)$ 正确 ✓
- F4（STE 约束 $\partial\hat{w}/\partial w\equiv 1$）：标准 STE 定义 ✓

### 3. 可运行代码

S5 代码块（"对称均匀量化与离群值放大"）已实际执行。输出与页面"预期输出"逐行一致：

```
E1: scale=0.8, x_q=[2,4,7], x_hat=[1.6,3.2,5.6], error=[0.4,-0.2,0.0]
E2: scale=7.1429, x_q=[0,0,1,7], x_hat=[0.0,0.0,7.1429,50.0], error=[-1.2,-3.4,1.5429,0.0]
E3: 块A scale=0.8 恢复正常块精度, 块B scale=7.1429 隔离离群值
```

一致 ✓

### 4. 事实与推断

- N1（每参数字节：FP32=4, FP16/BF16=2, INT8=1, INT4=0.5；7B FP16≈14GB, INT8≈7GB, INT4≈3.5GB）：算术正确（$7×10^9×2=14$GB）✓
- N2（MXFP4 块 32, 元素 4bit, scale 8bit, 每块 136bit, 有效 4.25bit）：zeroentropy.dev 与 emergentmind 一致确认 ✓
- N3（PTQ 校准 128–512 样本）：常见工程值，[G] §2.5 ✓
- N4（E2M1 可表示值 $\{0,\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}$，最大绝对值 6）：zeroentropy.dev 确认此集合与最大值 6 ✓。但页面称"16 个有限值"——实际集合有 15 个不同数值（4-bit 有 16 个码字，但 -0 与 +0 数值相同），见问题 2。

### 5. 前置知识引用

- ../mxfp4-qat/index.html — 目录存在 ✓

### 6. 教学简化

- 舍入约定简化（四舍五入 vs banker's rounding）——已说明 ✓
- INT4 对称范围取 $[-7,7]$（部分实现用 $[-8,7]$）——已说明 ✓
- per-block 教学块大小 3 与 1——已说明实际 MXFP4 固定 32 ✓
- 代码未实现 QAT STE 反向——已说明 ✓

### 7. 页面功能

- KaTeX 公式渲染：正确 ✓
- 折叠块（舍入约定、离群值手算、有效位宽、伪代码、可运行代码）：收起后主线不依赖 ✓
- 目录锚点：h2/h3 带 id ✓
- S2 量化网格图 `<pre class="diagram">` 缺少闭合标签（见问题 1）

## 问题

- [轻微·技术] S2 "量化网格"图（`<pre class="diagram">`，约第 737 行）：`<pre>` 块以 `</p>` 结尾但缺少 `</pre>` 闭合标签。`</p>` 会以可见文本出现在图中，且 `<pre>` 未闭合可能导致后续元素渲染异常。修法：将行尾 `</p>` 改为 `</pre>` ｜ 修复： ｜ 复验：
- [轻微·技术] S5 MXFP4 小节：正文称"E2M1 可表示的 16 个有限值为 $\{0,\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}$"。该集合实际含 15 个不同数值（0 加 7 对正负值），4-bit E2M1 有 16 个码字但 $-0$ 与 $+0$ 数值相同。来源标注 N4 同样写"16 个"。修法：改为"15 个不同值（16 个码字，其中 $-0$ 与 $+0$ 数值相同）" ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：进入修复（2 条轻微分别为 HTML 标签闭合与数值计数，改动量小，不影响核心结论与主线理解）
