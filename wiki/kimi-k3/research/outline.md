# outline.md — K3 教学大纲

## 页面开头

- **一句话说明**：K3 用 KDA+AttnRes+Stable LatentMoE 三维度扩展信息流，配合训练/基础设施创新，在 2.8T 参数下实现约 2.5× scaling 效率提升。
- **论文解决的具体问题**：开源模型在预训练 scaling（第一轴）上进展缓慢，多停留在 1T 级；K3 把预训练推到 3T 级，同时扩展测试时 scaling（第二轴）到 1M 上下文。
- **学习承诺**：读完能回答 Q1-Q5（见 learning-goals 组件）。
- **论文元信息**：blockquoate.meta 组件。
- **首个具体场景**：从 Table 1 的 K2→K3 架构对比切入，让读者直观看到"变了什么"。
- **与第一章的过渡**：架构变化不只是"更大"，而是三维度信息流的系统性重新设计。

## 章节设计

### S1：三维度信息流：K3 架构总览
- **教学任务**：让读者理解 K3 的核心设计理念——在序列、深度、宽度三维度扩展信息流，并用 Table 1 量化 K2→K3 的变化。
- **核心问题**：Q1-Q5 全部的入口
- **内容**：
  - 三维度框架（sequence/depth/width）
  - Table 1 自绘表格（K2 vs K3，含 Δ 列）
  - Fig.2 架构图描述（不内联原图，用文字+ASCII 图示描述三维度）
  - 2.5× scaling efficiency 声明（引向 S6/S8）
- **前置知识引用**：首次提到 KDA/MLA/AttnRes/LatentMoE 时给子页面链接
- **贯穿例子**：引入"一个 token 流经 K3"的贯穿视角——token 从 embedding 出发，经过 93 层，每层在三个维度被处理
- **完成检查**：用 K2→K3 的 5 个关键 Δ 自检

### S2：序列维度：KDA + 混合注意力（Q1）
- **教学任务**：完整回答 Q1——为什么 3:1 混合，如何同时获得效率和全局注意力。
- **核心问题**：Q1
- **内容**：
  - KDA 一句话角色 + 链接 `../../wiki/kda/index.html`
  - MLA 一句话角色 + 链接 `../../wiki/mla/index.html`
  - 3:1 混合动机：KDA 高效长序列，MLA 全局注意力，职责分离
  - 末尾 MLA：保证最终层全局注意力
  - NoPE：MLA 层不用位置编码，KDA 隐式编码位置 + 链接 `../../wiki/nope/index.html`
  - config.json 层配置确认（69 KDA + 24 MLA，full_attn_layers 列表）
  - 贯穿例子推进：token 在序列维度的处理——3 层 KDA 状态累加，第 4 层 MLA 全局重置
- **折叠块**：config.json 的 full_attn_layers/kda_layers 完整列表
- **完成检查**：解释 3:1 而非全 KDA 或全 MLA 的原因；NoPE 如何省去外推

### S3：深度维度：Block AttnRes（Q2）
- **教学任务**：完整回答 Q2——Block AttnRes 如何避免 93 层信息稀释，代价与边界。
- **核心问题**：Q2
- **内容**：
  - 标准残差瓶颈（类比 RNN 时间瓶颈）+ 链接 `../../wiki/block-attnres/index.html`
  - AttnRes 思路：把注意力方法论用到深度
  - 全 AttnRes 的 O(L²d) 可负担但 O(Ld) 内存/通信大
  - Block AttnRes：8 block × 12 层，O(Ld)→O(Nd)
  - config.json attn_res_block_size=12 确认
  - 贯穿例子推进：token 在深度维度的处理——每层可检索前 block 的表示
  - 代价与边界：块内仍有标准残差稀释；block size 与 block 数的权衡
- **折叠块**：93/12=7.75 的 block 划分细节
- **完成检查**：解释为何 N≈8 够用；块内 vs 跨块的信息流差异

### S4：宽度维度：Stable LatentMoE（Q3）
- **教学任务**：完整回答 Q3——三件稳定化各解决什么。
- **核心问题**：Q3
- **内容**：
  - LatentMoE 结构 + 链接 `../../wiki/stable-latent-moe/index.html`
  - 两个失效模式：(1) 四连矩阵乘激活爆炸 (2) 近 10³ 专家负载均衡
  - 稳定化 1：RMSNorm 插入 W↑ 前 + 链接 stable-latent-moe
  - 稳定化 2：SiTU-GLU（β₁=4, β₂=25, 界=100）+ 链接 `../../wiki/situ-glu/index.html`
  - 稳定化 3：Quantile Balancing + 链接 `../../wiki/quantile-balancing/index.html`
  - config.json 数值确认
  - 贯穿例子推进：token 在宽度维度的处理——投影到潜在空间，选 16 专家，RMSNorm，升回
- **完成检查**：三个稳定化各解决哪个失效模式；为何不能去掉任何一个

### S5：原生视觉：MoonViT-V2
- **教学任务**：解释 K3 为何从零训练视觉编码器，及其稳定性发现。
- **核心问题**：Q4 的视觉部分
- **内容**：
  - 从零训练 vs SigLIP 初始化的动机：训练稳定性 + 链接 `../../wiki/moonvit-v2/index.html`
  - Fig.6 梯度范数对比（文字描述，不内联原图）
  - next-token prediction 让表示直接由语言建模目标塑造
  - 架构：27 层，~0.4B，RMSNorm，无 bias，时空因子化
  - "对比预训练对大规模多模态语言模型不是必要的初始化"——论文声称
- **完成检查**：为何从零训练更稳定；这个发现的意义

### S6：训练：数据、scaling law、Per-Head Muon、长上下文扩展（Q4）
- **教学任务**：完整回答 Q4——训练方法如何共同支撑 2.5× scaling 效率。
- **核心问题**：Q4
- **内容**：
  - 数据：四类文本域 + 视觉，重述/去重/质量过滤
  - scaling law：cosine decay vs WSD，独立搜索最优超参后 cosine 胜出
  - Fig.7 2.5× scaling efficiency（文字描述曲线含义）
  - Per-Head Muon：按头正交化 + 链接 `../../wiki/per-head-muon/index.html`
  - 长上下文：NoPE 外推 + 四阶段渐进（8K→64K→256K→1M）+ 链接 nope
  - 训练配方：cosine + 1% warmup + weight decay 0.1
  - 2.5× 是 collectively 的结果（C19 推断标注）
- **完成检查**：列出支撑 2.5× 的 5 个训练因素；NoPE 如何省去外推

### S7：后训练：SFT→RL→MOPD 三阶段、QAT、EAGLE draft
- **教学任务**：解释后训练如何整合多域能力到统一模型。
- **核心问题**：Q4 的后训练部分
- **内容**：
  - SFT：冷启动，XTML chat template，扩展 agentic 轨迹
  - RL：3 域 × 3 努力 = 9 专家，partial rollout，reasoning effort RL
  - MOPD：九专家蒸馏为统一模型（简要内联，子页面未生成）
  - QAT：从 SFT 开始，MXFP4 权重 + MXFP8 激活 + 链接 `../../wiki/mxfp4-qat/index.html`
  - EAGLE-3 draft：MTP 层微调为 draft model，LK loss（简要内联，子页面未生成）
- **完成检查**：三阶段各做什么；QAT 为何从 SFT 开始

### S8：基础设施（简述）
- **教学任务**：简述支撑 3T 训练 + 1M RL + 推理的基础设施创新。
- **核心问题**：Q4 的基础设施部分
- **内容**：
  - KDA 系统协同：FlashKDA（CUTLASS chunkwise kernel）、KCP（KDA Context Parallelism，固定大小 all-gather）——简要内联
  - MoonEP：完美均衡 EP，E/R redundant 上界，零拷贝通信，静态形状——简要内联
  - RL 系统：co-located，partial rollout，外部 KV cache pool，AgentENV microVM 沙箱——简要内联
  - 推理服务：KDA-aware prefix cache，高性能内核，fleet 调度——简要内联
  - 链接 `../../wiki/gpu-execution-model/index.html`（GPU 执行模型已有）
- **完成检查**：三类基础设施各解决什么挑战

### S9：性能与评价（Q5）
- **教学任务**：完整回答 Q5——性能定位与边界。
- **核心问题**：Q5
- **内容**：
  - Table 2 关键 benchmark 自绘表格（编程/智能体/视觉/推理分域）
  - 亮点：ProgramBench 77.8（第一）、SWE-Marathon 42.0（第一）、BrowseComp 91.2（第一）、Terminal-Bench 88.3（≈Sol 88.8）
  - 短板：HLE-Full 43.5/56.0、CritPt 23.4、DeepSWE 67.5
  - 第三方评估：AA 57.1（#4/580）、Vals 74.7%（#2/39）、WebDev Arena 1678（#1/99）
  - 成本效率：BrowseComp $2.03/任务，KCB 2.0 38% Fable 5 成本
  - benchmark 条件与 harness 差异（Terminal-Bench 官方 vs AA、Fable 5 fallback、Sol cyberguard、SWE-Marathon H20 校准）
- **折叠块**：完整 Table 2（更多 benchmark）
- **完成检查**：K3 相对四个闭源和一个开源的位置；harness 差异如何影响解读

### S10：独立评价（必有，全部为解读者推断）
- **教学任务**：给出解读者对 K3 的判断。
- **内容**：
  - 优点：三维度信息流设计的系统性、开源 3T 的里程碑、训练-部署协同（QAT+EAGLE）、基础设施工程深度
  - 局限：研究级推理仍落后、benchmark harness 差异使绝对分数需谨慎、2.5× 未分解单因素、视觉"从零训练匹配 SigLIP"的结论可能依赖特定规模
  - 适用场景：长上下文 agentic 任务、成本敏感部署、开源研究；不适合需要最强研究级推理的场景
  - 与相邻工作位置：K2 的直接升级；混合注意力（KDA+MLA）与纯注意力（GPT/Claude）路线对比；Stable LatentMoE 与传统 MoE 路线对比

## 贯穿问题

贯穿例子：**一个 token 流经 K3 的 93 层**。
- S1 引入：token 从 embedding 出发，三维度信息流
- S2 推进：token 在序列维度——3 层 KDA 状态累加，第 4 层 MLA 全局交互
- S3 推进：token 在深度维度——每层可检索前 block 表示
- S4 推进：token 在宽度维度——投影到潜在空间，选 16 专家
- S5 推进：若 token 是视觉 token，MoonViT-V2 先编码
- S6 推进：训练时这个 token 的 loss 如何被优化
- S9 推进：推理时这个 token 如何被服务

## 讲解材料职责

| 材料 | 职责 | 位置 |
|---|---|---|
| Table 1 自绘表格 | 量化 K2→K3 架构变化 | S1 |
| ASCII 图示 | 三维度信息流 + token 流经路径 | S1 |
| config.json 层配置 | 确认 69 KDA + 24 MLA | S2 折叠块 |
| Block 划分计算 | 93/12=7.75 | S3 折叠块 |
| Table 2 自绘表格 | 性能定位 | S9 |
| 2.5× scaling 曲线文字描述 | scaling efficiency 含义 | S6 |

## 正文与折叠块分工

### 必须放正文

- 三维度框架 + Table 1
- 3:1 混合动机 + 末尾 MLA
- AttnRes 标准残差瓶颈 + Block 划分 + O(Ld)→O(Nd)
- 三件稳定化各解决什么
- 2.5× scaling efficiency 声明 + collectively 标注
- 性能定位（亮点+短板）+ harness 差异
- 独立评价全部内容
- 所有子页面链接

### 可放折叠块

- config.json full_attn_layers/kda_layers 完整列表
- Block 划分计算细节（93/12 的部分 block）
- Table 2 完整 benchmark（正文只放关键）
- SiTU-GLU 输出界推导（引子页面）
- QB 对偶推导（引子页面）

### 折叠块全部收起时正文仍须回答

- Q1：3:1 混合 + 末尾 MLA + NoPE → 正文已完整
- Q2：Block AttnRes + 8 block + O(Ld)→O(Nd) + 代价 → 正文已完整
- Q3：三件稳定化 + 各解决什么 → 正文已完整
- Q4：数据/scaling law/Muon/NoPE/QAT + collectively 2.5× → 正文已完整
- Q5：性能定位 + harness 差异 → 正文已完整
