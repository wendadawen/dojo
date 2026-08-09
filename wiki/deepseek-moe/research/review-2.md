# DeepSeekMoE 独立审查（第二次）

- 审查者：独立上下文（AI 模拟）
- 页面版本：23a8ee6f443a3ae21709f6f705c46dafae3e67d0（index.html 工作树哈希）
- 时间：2026-08-09

## 审查范围

- 输入：`wiki/deepseek-moe/index.html`、`wiki/deepseek-moe/overview.html`、`guides/concept/check.md`、`guides/concept/content-examples.md`
- 外部来源：WebSearch "DeepSeekMoE Dai 2024 arxiv 2401.06066"（细粒度分割 §3.1、共享专家隔离 §3.2、公式 Eq.6-11、2B/16B/145B 实验数字）+ WebFetch ar5iv.labs.arxiv.org/html/2401.06066 核对 §3.2 Eq.9-11 与 §5.1.2 16B 配置 + 实际执行页面可运行代码块
- 未读取 `research/` 目录、未读取仓库中其他概念页（仅验证链接路径存在）

## 段 A 盲读小结

按页面顺序阅读，四个学习目标在正文中均得到完整回答：

1. 知识混合与知识冗余 → "传统 MoE 的两个毛病"章回答（知识混合 = 专家少→token 涵盖多种知识→专家塞杂；知识冗余 = 不同专家各自学通用知识；两者妨碍专家专门化）
2. 细粒度分割保持计算量不变（手算）→ "把专家切小"章回答（中间维度 1/m、mN 专家、mK 激活、计算量守恒 mK×(1/m)=K；手算 N=4,K=1,m=2 → 8 小专家 top-2、参数 4、计算 1、组合 4→28）
3. 共享专家隔离消除冗余（完整公式）→ "共享专家隔离"章回答（Eq.9 完整公式、K_s 恒激活无门控、路由专家 mN-K_s、激活 mK-K_s、总激活 mK 守恒）
4. 相同算力下相对 GShard 的性能优势 → "相同算力下真的更好"章回答（2B vs GShard 2.9B Pile Loss 1.808、16B vs LLaMA2 7B 39.6% 计算、145B vs DeepSeek 67B 28.5%/18.2%）

术语首现解释充分（top-K MoE、router、专家加权和、知识混合、知识冗余、专家专门化、细粒度分割、共享专家隔离均有定义或最小概念说明）。推导无跳步：计算量守恒 mK×(1/m)=K、参数守恒 mN×(1/m)=N、共享专家引入后路由专家与激活数的连带变化均给出代入。手算例子（N=4,K=1,m=2 + K_s=1）可复算，与代码执行结果一致。可运行代码块（§2 折叠块）含全部导入与预期输出，实际执行后输出与页面预期完全一致。前置链接 ../moe-serving/index.html、../stable-latent-moe/index.html 路径均存在。

## 段 B 对照来源小结

1. 定义与机制：[C1] 知识混合 + 知识冗余妨碍专门化与论文 §1 一致；[C2] 知识混合成因与 §1 一致；[C3] 知识冗余成因与 §1 一致；[C4] 细粒度分割（中间维度 1/m、mN 专家、mK 激活、守恒）与 §3.1 + Figure 2(b) 一致；[C5] 组合数 C(N,K)→C(mN,mK)、120→4,426,165,368 与 §3.1 一致；[C6] 共享专家隔离（K_s 恒激活、不路由、路由专家与激活各减 K_s）与 §3.2 + Figure 2(c) 一致；[C7] 三架构计算量恒定与 Figure 2 caption + §3.1/§3.2 一致（"while maintaining a consistent number of expert parameters and computational cost" + "In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by K_s"）；[C8] DeepSeek-V3 继承 shared+routed 但改路由为 sigmoid + 偏置与 §2-§3 + DeepSeek-V3 技术报告一致。
2. 公式与推导：[F1] 通用 top-K MoE 层 h=Σg·FFN+u 与 Eq.(3) 一致；[F2] 通用门控与 Eq.(4)(5) 一致；[F3] 细粒度 MoE 层与 Eq.(6) 一致；[F4] 细粒度门控（TopK 范围 1..mN、选 mK）与 Eq.(7)(8) 一致；[F5] 完整 DeepSeekMoE 层 h=Σ_{1..K_s}FFN + Σ_{K_s+1..mN}g·FFN + u 与 Eq.(9) 完全一致（路由专家索引 K_s+1..mN，即 mN-K_s 个）；[F6] 完整门控（TopK 范围 K_s+1..mN、选 mK-K_s）与 Eq.(10)(11) 一致；[F7] 计算量守恒推导 mK·(1/m)=K、K_s·(1/m)+(mK-K_s)·(1/m)=K 正确。符号 u_t^l、FFN_i、g_{i,t}、s_{i,t}、e_i^l、h_t^l、K_s、mN、mK 首现处均有定义。
3. 可运行代码：§2 折叠块代码已实际执行（python3），输出与页面预期完全一致——教学例子：分割前 参数量=4.0/计算量=1.0/组合数=4，分割后 参数量=4.0/计算量=1.0/组合数=28，守恒均为 True；论文例子：C(16,2)=120→C(64,8)=4426165368，参数量 16.0/计算量 2.0 守恒；完整 DeepSeekMoE：路由专家总数=7/激活路由=1/总激活=2/参数量=4.0/计算量=1.0 守恒。代码含全部导入（import math），变量映射到公式（segment 对应 F3/F4，add_shared 对应 F5/F6），教学简化已说明（1 单位 = 原 FFN 大小，组合数假设 router 无约束）。
4. 事实与推断：[N1] 2B vs GShard 2.9B Pile Loss 1.808 与论文摘要 + §4.3 + Table 2 一致；[N2] 2B 接近 Dense×16 上界与 §4.3 一致；[N3] 2B 配置 9 层/隐藏 1280/1 shared+63 routed/m=4/激活 1+7=8/总参 2.0B/激活 0.3B 与 §4.1.3 一致（mN=64, N=16, m=4 严格自洽）；[N4] 16B vs LLaMA2 7B 39.6% 计算（74.4T vs 187.9T FLOPs/4K tokens）与摘要 + §5.2.2 + Table 4 一致；[N5] 16B 配置 28 层/隐藏 2048/2 shared+64 routed/m=4/激活 2+6=8/总参 16.4B/激活 2.8B/训练 2T tokens 与 §5.1.2 完全一致（注：mN=66 对应 N=16.5 非整数，但论文允许总专家数独立于 N 设定，页面与论文表述一致）；[N6] 16B 40GB 单卡部署、2.5× 7B 速度与 §5.2.1 一致；[N7] 145B vs DeepSeek 67B 28.5%（可能 18.2%）与摘要 + §7 一致；[N8] 组合数 120→4,426,165,368 与 §3.1 一致。
5. 前置知识引用：moe-serving、stable-latent-moe 两个目录均存在，链接层级正确。
6. 教学简化：五项简化（FFN 参数量用单位度量、组合数假设无约束、负载均衡不展开、DeepSeek-V3 只点差异、性能结论依赖实验配置）均有说明，不影响核心结论。
7. 页面功能：KaTeX 渲染、details 折叠（含 code-details 可运行代码块）、自动生成目录锚点（h2 均有显式 id: s1-two-flaws、s2-fine-grained-segmentation、s3-shared-expert-isolation、s4-evidence、s5-inheritance、sources-and-teaching-notes）结构正确。

数字复算：C(4,1)=4 ✓、C(8,2)=28 ✓、C(16,2)=120 ✓、C(64,8)=4,426,165,368 ✓；2B 配置 mN=64=1+63, N=16, m=4 严格自洽 ✓；16B 配置 mN=66=2+64, m=4→N=16.5（与原论文 §5.1.2 一致，论文允许总专家数独立设定）✓；计算量守恒 mK×(1/m)=K ✓；参数守恒 mN×(1/m)=N（2B 配置 64×(1/4)=16 ✓）；共享专家引入后路由专家 mN-K_s、激活 mK-K_s、总激活 mK 守恒 ✓；代码执行输出与页面预期完全一致 ✓。

## 问题

（无）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布（审查范围内；发布门控的 validate.py 由编排者执行）
