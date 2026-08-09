# 投机解码独立审查

- 审查者：独立上下文（AI 模拟小白读者）
- 页面版本：index.html cf3320f0 / overview.html 2620c8eb（工作树内容哈希，文件未入 git）
- 时间：2026-08-09

## 问题

- [重要·盲读] index.html §4「能快多少」line 941：「最优 γ ≈ 8」无推导无来源。文档给出 S 公式后直接陈述「α=0.8, c=0.05 时最优 γ≈8；α=0.5 时最优 γ 显著更小」，但未展示如何从 S 公式求最优 γ（对 γ 枚举比较或求极值），也未标注来源（Leviathan §3.5 讨论了最优 γ 选择）。读者无法复算或验证 γ≈8。：补充最优 γ 的求法说明（如「枚举 γ=5..10 计算 S，γ=8 时 S 最大」），或标注来源为 Leviathan §3.5。 ｜ 修复：已在 line 941 补充枚举推导——"把 S 代入枚举 γ=5,6,7,8,9,10 可得 S≈2.95,3.04,3.08,3.09,3.08,3.05，最优 γ≈8"，并标注来源"最优 γ 的选择见 Leviathan et al. 2023 §3.5"。 ｜ 复验：validate.py 通过
- [轻微·盲读] overview.html line 73 / index.html line 1057：「K3」首现未解释——读者不知道 K3 是模型、系统还是产品。工程实例章节以此开头，但 K3 本身无任何标识。：首次提及处加括号标识（如「K3（××的 LLM 推理系统）」）或改用通用表述。 ｜ 修复： ｜ 复验：
- [轻微·盲读] overview.html line 75 / index.html line 939：「vLLM」首现未解释——读者不知道它是开源 LLM 推理框架。：首次提及处加括号标识（如「vLLM（开源 LLM 推理框架）」）。 ｜ 修复： ｜ 复验：
- [轻微·盲读] overview.html line 73：$\mathrm{TV}(p,q)$ 在快速阅读页使用但未定义（定义仅在 index.html §3）。仅读 overview 的读者无法理解此公式。：首次使用处加一句「TV 为总变差距离，衡量两分布差异」或改用纯文字描述。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html line 674：「HBM 带宽」中 HBM 缩写未展开（High Bandwidth Memory）。虽链接了 GPU 执行模型前置页，但缩写首现处未展开。：首现处展开为「HBM（高带宽显存）」。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html line 674：「SM / Tensor Core」中 SM 缩写未展开（Streaming Multiprocessor）。：首现处展开为「SM（流多处理器）」。 ｜ 修复： ｜ 复验：

## 段 A 盲读小结

按页面顺序阅读两份文档，主线理解卡点如上。核心机制（Draft-then-Verify 五步、接受规则、残差重采样、bonus token）、单位置保分布证明、加速比公式与手算例子均可由正文独立理解，无折叠块依赖、无推理跳步（除「最优 γ」一处）。

学习目标核对（依据 index.html 内嵌「读完你能回答」5 条，scope.md 在 research/ 目录内不可读，无法交叉核对）：

1. 自回归解码慢在哪里 + 并行验证为何几乎免费 → §1 完整回答 ✓
2. Draft-then-Verify 一轮 + 接受/拒绝条件 → §2 完整回答 ✓
3. 接受/拒绝规则为何保分布 → §3 完整回答（含五步推导链与折叠块）✓
4. α 与 γ 如何决定加速比 + 何时更慢 → §4 完整回答（最优 γ 推导除外）✓
5. 贪心解码退化 + 与采样情形关系 → §3 末尾完整回答 ✓

## 段 B 对照来源核查

来源：Leviathan et al. 2023 (arXiv:2211.17192) + Chen et al. 2023 (arXiv:2302.01318)，经 WebSearch 获取摘要与正文片段。

### 1. 定义与机制

| 页面论断 | 来源对照 | 结论 |
|---|---|---|
| 两篇论文独立提出相同机制 [C1] | Chen et al. 正文：「the work in this manuscript was undertaken concurrently and independently of the work on speculative decoding from Leviathan et al. (2022)」 | 一致 ✓ |
| Leviathan 命名 "Speculative Decoding"，ICML 2023 Oral | arXiv:2211.17192 摘要 Comments: ICML 2023 Oral | 一致 ✓ |
| Chen 命名 "Speculative Sampling"，DeepMind | arXiv:2302.01318 全部作者属 DeepMind | 一致 ✓ |
| 提交日期：Leviathan 2022-11-30，Chen 2023-02-02 | arXiv 提交历史确认 | 一致 ✓ |
| Draft-then-Verify 五步流程 [C3-C6] | 两篇论文算法描述一致 | 一致 ✓ |
| 贪心退化为 argmax 匹配 [C7] | Leviathan §3 末尾讨论 greedy | 一致 ✓ |

### 2. 公式与推导

| 公式 | 来源 | 复算 | 结论 |
|---|---|---|---|
| $a(x)=\min(1,p/q)$ [F1] | Leviathan §3.1, Chen §2.1 | 三条边界（$p\geq q \Rightarrow 1$、$p<q \Rightarrow p/q$、$p=0 \Rightarrow 0$）正确 | 一致 ✓ |
| 残差分布 $p'(x)=\mathrm{norm}(\max(0,p-q))$ [F2] | Leviathan Algorithm 1, Chen Eq.(2) | 归一化常数 $Z=\beta$ 由 $\min+\max$ 恒等式推出，正确 | 一致 ✓ |
| $\Pr[\text{emit }x]=\min(q,p)+\max(0,p-q)=p$ [F3] | Leviathan Theorem 2, Chen §2.3 | 五步推导链每步可手算复算，恒等式 $\min(a,b)+\max(0,b-a)=b$ 正确 | 一致 ✓ |
| $\alpha=\sum\min(p,q)=1-\mathrm{TV}(p,q)$ [F4] | Leviathan §3 | $\min(a,b)=(a+b-\|a-b\|)/2$ 代入正确 | 一致 ✓ |
| $\mathbb{E}[L]=(1-\alpha^{\gamma+1})/(1-\alpha)$ [F5] | Leviathan Theorem 3.8 | 截断几何级数闭式正确；边界 $\alpha=1\Rightarrow\gamma+1$、$\alpha=0\Rightarrow 1$ 正确 | 一致 ✓ |
| $S=(1-\alpha^{\gamma+1})/[(1-\alpha)(1+\gamma c)]$ [F6] | Leviathan Theorem 3.8（walltime-improvement factor） | $S=\mathbb{E}[L]/(1+\gamma c)$ 正确 | 一致 ✓ |

手算复算：
- α=0.8, γ=5, c=0.05：E[L]=(1-0.8⁶)/0.2=0.7379/0.2=3.689 ✓；S=3.689/1.25=2.95× ✓（与 Leviathan 摘要 2×-3× 一致）
- α=0.2, γ=5, c=0.05：E[L]=(1-0.2⁶)/0.8≈1.25 ✓；S=1.25/1.25=1.00× ✓
- 贯穿例子（3 token 词表）：逐位 a_i、残差归一化、单位置 Pr[emit x]=p(x) 全部复算通过 ✓
- 平均 ᾱ≈0.833, γ=3：E[L]=(1-0.833⁴)/0.167≈3.11 ✓

### 3. 可运行代码

页面无声称可运行的代码块（伪代码标注为「不是 Python」）。无需执行。

### 4. 事实与推断

| 外部数字 | 来源 | 结论 |
|---|---|---|
| [N1] T5-XXL (11B)，2×-3×，identical outputs | arXiv:2211.17192 摘要原文 | 一致 ✓ |
| [N2] Chinchilla 70B，分布式，2-2.5×，不修改模型不降质量 | arXiv:2302.01318 摘要原文 | 一致 ✓ |
| [N3] T5-base (~250M) draft，α≈0.8，c≈0.05 | 社区综述引用原文；T5-base 实际 220M，「约 250M」可接受 | 间接一致（S≈2.95 与 2-3× 吻合）✓ |
| [N4] vLLM 2024-10 博客，高 QPS 下 1.4×-1.8× 减速，Llama3-70B 4×H100 | vLLM 官方博客原文确认：ShareGPT 1.4× 减速、CNN Dailymail 1.8× 减速 | 一致 ✓ |
| EAGLE-3 为真实 draft 架构 | arXiv:2503.01840 (EAGLE-3, 2025-03) | 一致 ✓ |

教学示例（S1 42ms 带宽下限、S4 α=0.2 反例、S5 3-token 词表）均标注为「教学示例」「教学计算」「纯粹构造」，未越界写成来源结论。✓

### 5. 前置知识引用

| 链接 | 目标 | 存在 | 占位提示 |
|---|---|---|---|
| `../../wiki/gpu-execution-model/index.html` | GPU 执行模型 | 存在 ✓ | 非占位 |
| `../../wiki/standard-attention/index.html` | 标准 Transformer 注意力 | 存在 ✓ | 非占位 |
| `../../wiki/eagle-speculative/index.html` | EAGLE-3 | 存在 ✓ | 标注「占位待生成」 |

注：eagle-speculative/index.html 文件存在但文档标注「待生成」——可能为 stub 或标注未更新，因禁止读其他页面无法核实内容。

### 6. 教学简化

i.i.d. 接受率假设、3-token 词表、概率取一位小数、「单次前向 γ+1 位置墙钟时间几乎相同」均在 §「教学简化及其限制」逐项说明并标注可推出/不可推出边界。类比（「开车去仓库取货」「target 觉得 draft 高估就倾向于拒绝」）均标注失效边界。✓

### 7. 页面功能

- 公式渲染：KaTeX delimiters `$$`/`$` 配置正确，`throwOnError:false`；`$i^\*$` 中 `\*` 在 KaTeX 中渲染为 `*`（已确认支持），无渲染错误。
- 折叠交互：`<details>` 块收起后正文主线可独立理解（伪代码、完整推导、完整手算均为补充，非主线依赖）。
- 目录锚点：JS 自动生成 TOC，`scroll-margin-top: 90px` 避开固定导航。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：进入修复
- 段 B 来源核查：核心论断（两篇论文独立提出、T5-XXL 2×-3×、Chinchilla 70B 2-2.5×、vLLM 1.4×-1.8× 减速）、全部 6 条公式、手算数字均与来源一致，无技术性错误。EAGLE-3 确认为真实方法（arXiv:2503.01840）。Theorem 3.8 编号经社区来源确认对应 walltime-improvement factor。
- 审查限制：scope.md 位于 research/ 目录，依审查任务包禁止读取，学习目标仅核对 index.html 内嵌 5 条，未交叉核对 scope.md 原文。
