# DeepSeekMoE 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 强推理模型对照来源）
- 页面版本：index.html `23a8ee6` / overview.html `c641f47`（工作树哈希）
- 时间：2026-08-09 15:14 CST
- 来源：Dai et al. 2024, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", arXiv:2401.06066（经 WebSearch + arxiv HTML 全文核对）

## 段 A 盲读记录

按页面顺序扮演完全小白读者（仅依赖页面提供信息，不用自身领域知识补全）：

- **开篇钩子**："16 专家 top-2 → 120 种搭配，切 4 份变 64 小专家 top-8 → 44 亿种，计算量没涨"——作为钩子可懂，"组合数 = C(N,K)"的算法在 S2 表格中首次显式给出（$\binom{4}{1}=4$、$\binom{8}{2}=28$），开篇的 120 读者可由直觉接受，不构成卡点。
- **S1 知识混合/冗余**：Python 代码 + 散文叙事的 E1 示例直观；知识冗余的 ASCII 图示（3 专家各学一份语法）清晰；"专家专门化 = non-overlapping and focused knowledge"有定义。无卡点。
- **S2 细粒度分割**：FFN 参数量公式 $2\cdot d\cdot d_{ff}$ 引用前置页；公式 F3/F4 符号首次出现处均有定义（$u_t^l$、$\operatorname{FFN}_i$、$g_{i,t}$、$s_{i,t}$、$e_i^l$、$h_t^l$、残差）；计算量守恒推导 $mK\times(1/m)=K$ 可跟上；4×1×2 教学表格可手算验证。代码块在折叠 details 内，关键结论（守恒、组合数）已在正文给出，收起状态不影响主线。无卡点。
- **S3 共享专家隔离**：公式 F5 用 underbrace 标注恒激活/稀疏两部分；F6 路由范围 $K_s{+}1..mN$、选 $mK{-}K_s$ 清晰；"三个数字连带变化"逐项写出；计算量守恒再次推导；ASCII 分工图以教学例子 N=4,K=1,m=2,K_s=1 贯穿；"为什么共享专家不参与路由"给出因果解释。负载均衡折叠块明示"不影响理解两策略，本页不展开"。无卡点。
- **S4 实验证据**：三组对比表格清晰；"三架构计算量恒定是前提"有强调；边界 callout 提醒不可外推。**唯一卡点**：S4 第二段提到"同总参数的稠密模型（Dense×16）"，"Dense×16"中的"×16"首次出现未解释——读者不知道 16 指什么（FFN 中间维度 16 倍，对应 2B 配置 N=16），虽然"同总参数的稠密模型"给出了职能描述，但"×16"这个标识符本身无法解释。
- **S5 影响与继承**：DeepSeek-V3 与 Stable LatentMoE 的继承关系只点明不展开，符合"只交代继承"的承诺。无卡点。
- **学习目标闭环核对**：
  1. 知识混合/冗余成因及对专门化的妨碍 → S1 完整回答 ✓
  2. 细粒度分割守恒 + 手算 4 专家 top-1 m=2 → S2 表格 + 代码完整回答 ✓
  3. 共享专家消除冗余 + 分工 + 完整公式 → S3 F5 + 连带变化完整回答 ✓
  4. 相同算力下相对 GShard 的性能数字 → S4 三组对比完整回答 ✓

## 段 B 对照来源核查

### 1. 定义与机制（C1–C8）

| 论断 | 页面表述 | 来源核对 | 结论 |
|------|---------|---------|------|
| C1 专门化 = non-overlapping and focused knowledge | S1 | 论文摘要 + §1 + §3 多处出现该定义 | ✓ |
| C2 知识混合成因 | S1 | §1 引言 | ✓ |
| C3 知识冗余成因 | S1 | §1 引言 | ✓ |
| C4 细粒度分割：1/m 维度、mN 专家、mK 激活、守恒 | S2 | §3.1 + Figure 2(b) | ✓ |
| C6 共享专家隔离：K_s 恒激活、不路由、各减 K_s | S3 | §3.2 + Figure 2(c) | ✓ |
| C7 三架构参数与计算恒定 | S2/S3/S4 | Figure 2 caption 原文 "the number of expert parameters and computational costs remain constant" | ✓ |
| C8 DeepSeek-V3 继承 + 改路由 | S5 | 页面标注来源为 DeepSeek-V3 Technical Report arXiv:2412.19437 §2.1.2（独立于本论文）；原始论文 softmax 亲和度 Eq.5/8/11 已核对一致。C8 的 DeepSeek-V3 侧细节未在本次指定来源（arXiv:2401.06066）中，页面已透明标注来源，未扩大原始论文结论 | ✓（来源透明） |

### 2. 公式与推导（F1–F7）

逐条对照 arxiv HTML 全文提取的 Eq.(3)–(11)：

- F3（Eq.6）$h_t^l = \sum_{i=1}^{mN}(g_{i,t}\operatorname{FFN}_i(u_t^l)) + u_t^l$ —— 完全一致 ✓
- F4（Eq.7–8）TopK 范围 $1..mN$、选 $mK$；$s_{i,t}=\operatorname{Softmax}_i({u_t^l}^T e_i^l)$ —— 完全一致 ✓
- F5（Eq.9）$h_t^l = \sum_{i=1}^{K_s}\operatorname{FFN}_i(u_t^l) + \sum_{i=K_s+1}^{mN}(g_{i,t}\operatorname{FFN}_i(u_t^l)) + u_t^l$ —— 完全一致 ✓
- F6（Eq.10–11）TopK 范围 $K_s{+}1..mN$、选 $mK{-}K_s$ —— 完全一致 ✓
- F7 计算量守恒 $mK\times(1/m)=K$ 及 $K_s\times(1/m)+(mK{-}K_s)\times(1/m)=K$ —— 由 C7 + F3/F5 直接推出，推导正确 ✓
- 符号定义：$u_t^l$、$g_{i,t}$、$s_{i,t}$、$e_i^l$、$h_t^l$ 首次出现处均有定义，全文含义一致 ✓

### 3. 可运行代码

提取页面 `<details>` 内 Python 代码，写入临时文件 `/tmp/deepseek_moe_code_check.py` 并执行。实际输出与页面"预期输出"逐行比对：

| 例子 | 页面输出 | 实际输出 | 一致 |
|------|---------|---------|------|
| 教学 4×1×2 分割前 | 参数量=4.0, 计算量=1.0, 组合数=4 | 4.0, 1.0, 4 | ✓ |
| 教学 4×1×2 分割后 | 参数量=4.0, 计算量=1.0, 组合数=28 | 4.0, 1.0, 28 | ✓ |
| 论文 16×2×4 分割前 | C(16,2)=120, 参数量=16.0, 计算量=2.0 | 120, 16.0, 2.0 | ✓ |
| 论文 16×2×4 分割后 | C(64,8)=4426165368, 参数量=16.0, 计算量=2.0 | 4426165368, 16.0, 2.0 | ✓ |
| 完整 DeepSeekMoE K_s=1 | 路由=7, 激活路由=1, 总激活=2, 参数=4.0, 计算=1.0 | 7, 1, 2, 4.0, 1.0 | ✓ |

所有守恒断言均为 True，与页面描述一致。`add_shared` 中 `params = routed_total*(1/m) + Ks*(1/m) = (mN-Ks+Ks)*(1/m) = N`、`flops = (mK-Ks)*(1/m) + Ks*(1/m) = K`，推导正确。

### 4. 事实与推断（N1–N8）

| 数字 | 页面 | 来源 | 结论 |
|------|------|------|------|
| N1: 2B vs GShard 2.9B, Pile Loss 均 1.808 | S4 表 | 摘要 + §4.3 + Table 2：DeepSeekMoE 2B=1.808, GShard×1.5=1.808 | ✓ |
| N2: 2B 接近 Dense×16 上界 | S4 | §4.3 "nearly approaches the performance of its dense counterpart... upper bound" | ✓ |
| N3: 2B 配置 9 层/d=1280/1+63/m=4/1+7=8/2.0B/0.3B | S3+S4+来源 | §4.1.3 Table 5 全部吻合 | ✓ |
| N4: 16B vs LLaMA2 7B, 74.4T vs 187.9T=39.6% | S4 表 | 摘要 + §5.2.2 + Table 4 | ✓ |
| N5: 16B 配置 28 层/d=2048/2+64/m=4/2+6=8/16.4B/2.8B/2T | S3+来源 | §5.1.2 全部吻合 | ✓ |
| N6: 16B 单卡 40GB, ~2.5× 7B 速度 | S4 | §5.2.1 "single-device deployment on a GPU with 40GB" + "nearly 2.5 times" | ✓ |
| N7: 145B vs DeepSeek 67B, 28.5%(maybe 18.2%) | S4 表 | 摘要 "28.5% (maybe even 18.2%)" | ✓ |
| N8: 组合数 120→4,426,165,368 | S2 | §3.1 原文 "$\binom{64}{8}=4,426,165,368$" | ✓ |

教学示例 E1–E3 在来源章节明确标记"构造数字"，与论文事实区分清晰。

### 5. 前置知识引用

- `../moe-serving/index.html` —— 文件存在 ✓
- `../stable-latent-moe/index.html` —— 文件存在 ✓
- `../../index.html`（首页）—— 文件存在 ✓
- overview.html ↔ index.html 互相链接 —— nav 中 "快速阅读"→overview.html，overview 中 "深度教学→"→index.html ✓

### 6. 教学简化

来源章节"教学简化及其限制"列出 5 项（FFN 单位度量、组合数无约束假设、负载均衡不展开、DeepSeek-V3 不展开、性能不可外推），每项均说明限制，未导致核心结论失真。类比"把通用知识拎出来"在"教学解释与类比边界"中声明只描述职能、不断言具体内容。✓

### 7. 页面功能

- `python3 .dojo/scripts/validate.py wiki/deepseek-moe/index.html` → 退出码 0 ✓
- KaTeX 公式分隔符配置正常（`$$`/`$`）；代码高亮 Prism.js 本地加载 ✓
- 折叠 details（代码块 + 负载均衡补充）收起后主线可读 ✓
- 目录锚点由 JS 自动生成，scroll-margin-top 避开固定导航 ✓

## 问题

- [轻微·盲读] S4 第二段"同总参数的稠密模型（Dense×16）"："Dense×16"中的"×16"首次出现未解释，读者不知道 16 指什么（FFN 中间维度放大 16 倍，对应 2B 配置 N=16），"同总参数的稠密模型"已给出职能描述但标识符本身无法解释：在"Dense×16"后补一句括注"即 FFN 中间维度放大 16 倍、与 N=16 对应的稠密模型"，或直接删去"Dense×16"仅保留"同总参数的稠密模型" ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（1 项轻微问题不阻断发布；按 check.md §4 发布门控，遗留轻微问题需写明接受理由：S4"Dense×16"标识符未解释不影响核心正确性与主线理解——"同总参数的稠密模型"已传达关键语义，"×16"仅为论文中的模型命名，读者不知道其词源不影响理解"稠密模型是 MoE 理论上界"这一核心论断）
- 复核范围说明：C8（DeepSeek-V3 继承）的 DeepSeek-V3 侧细节（1 shared + 256 routed、sigmoid+偏置、aux-loss-free）引用 DeepSeek-V3 Technical Report arXiv:2412.19437，不在本次指定来源 arXiv:2401.06066 范围内；页面已透明标注来源，原始论文侧的 softmax 亲和度（Eq.5/8/11）已核对一致，未扩大原始论文结论
- 学习目标闭环：4 项学习目标全部由正文章节（S1–S4）完整回答 ✓
- validate.py 退出码 0 ✓
- 可运行代码已重跑，输出与页面一致 ✓
- 关键论断和数字已重新对照外部来源 ✓
- overview.html 存在且与 index.html 互相链接 ✓
