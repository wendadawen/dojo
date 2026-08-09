# MOPD 核心论断与证据（evidence.md）

来源优先级：Kimi K3 技术报告原文 §4.1.2、§4.1.3 为核心论断的主依据；OPD 一般理论由 WebSearch 核实的公开来源（Agarwal et al. 2023、Thinking Machines Lab 2025、Qwen3）作辅助定位，不作为 MOPD 核心论断的唯一依据。

## 核心论断

### C1：9 个专家教师 = 3 领域 × 3 努力程度

- **论断内容**：K3 通过 RL 跨三个领域、每个领域在三个努力程度级别上各训练一个专家，共 9 个专家模型。三领域为 (i) 通用任务、(ii) 通用 agent、(iii) 编码 agent；三努力程度为 {low, high, max}。
- **来源定位**：K3 技术报告 §4.1.2，原文 "we scale RL across three broad domains ... train a single expert for each domain at every reasoning effort level ... Crossing these three domain experts with three reasoning effort levels in {low, high, max} yields a total of nine expert models."
- **适用条件**：K3 的具体训练配置。
- **置信状态**：已确认。

### C2：MOPD 把九个专家的能力合并进统一学生模型

- **论断内容**：MOPD 用于把这 9 个领域/努力程度专项专家的能力合并进一个统一模型。训练时对给定领域 $d$ 和努力程度 $e$，用对应教师 $\pi_{\text{teacher}}^{(d,e)}$ 指导学生。
- **来源定位**：K3 技术报告 §4.1.3，原文 "We adopt Multi-Teacher On-Policy Distillation (MOPD) to consolidate these domain-specialized capabilities across varying reasoning efforts into a unified model ... for a given domain d and a sampled reasoning effort level e ∈ {low, high, max}, optimization is guided by the corresponding teacher model π_teacher among the nine experts."
- **适用条件**：9 个专家已训练完成。
- **置信状态**：已确认。

### C3：per-token OPD 奖励公式（Eq.15）

- **论断内容**：给定输入 $x$ 和前缀 $y_{<t}$，教师 $\pi_{\text{teacher}}^{(d,e)}$ 与学生 $\pi_\theta$ 在 token $y_t$ 上的 per-token OPD 奖励为 $r_{\text{opd}}^{d}(y_t\mid e,x,y_{<t})=\text{clip}(\text{sg}(\log(\pi_{\text{teacher}}^{(d,e)}(y_t\mid x,y_{<t})/\pi_\theta(y_t\mid e,x,y_{<t}))),-R_{\max},R_{\max})$。
- **来源定位**：K3 技术报告 §4.1.3 Eq.(15)。
- **适用条件**：学生条件化于 $e$；教师为对应 $(d,e)$ 专家；token $y_t$ 由学生采样得到。
- **置信状态**：已确认。

### C4：sg 与 Rmax 的定义

- **论断内容**：$\text{sg}(\cdot)$ 是停梯度算子；$R_{\max}>0$ 是裁剪阈值，用于约束极端 advantage 信号以稳定 RL 训练。
- **来源定位**：K3 技术报告 §4.1.3，原文 "sg(·) denotes the stop-gradient operator, and Rmax > 0 is a clipping threshold to constrain extreme advantage signals, thereby stabilizing RL training."
- **适用条件**：通用。
- **置信状态**：已确认。

### C5：稠密奖励无缝集成进 RL 框架，支持 partial rollout

- **论断内容**：该稠密奖励信号无缝集成进 RL 框架，天然支持 partial rollout 训练等基础设施级优化。
- **来源定位**：K3 技术报告 §4.1.3，原文 "This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks."
- **适用条件**：RL 框架支持 per-token 奖励输入（如 K2.5 策略优化算法）。
- **置信状态**：已确认。

### C6：top-k 蒸馏目标无优势

- **论断内容**：K3 也实验了更精细的 top-k 蒸馏目标，但在收敛速度和最终性能上都没有观察到明显优势。
- **来源定位**：K3 技术报告 §4.1.3，原文 "While we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting."
- **适用条件**：K3 的实验设置（具体数值未在报告中给出）。
- **置信状态**：已确认（结论方向）；具体数值未披露，正文不引用任何 top-k 的具体数字。

### C7：专家先训练，轨迹联合收集用于 SFT 和 MOPD

- **论断内容**：9 个专家通过 RL（含 reasoning effort budget control）训练；所有努力级别专家产生的轨迹联合收集，用于监督微调和多教师在线策略蒸馏。
- **来源定位**：K3 技术报告 §4.1.2，原文 "Trajectories produced by the resulting experts at all reasoning levels are jointly collected for supervised fine-tuning and multi-teacher on-policy distillation." 以及 §4.1.2 reasoning effort RL 段落关于 $\tau$ 退火得到 low/high/max 专家的描述。
- **适用条件**：MOPD 是两阶段流程的第二阶段。
- **置信状态**：已确认。

## 核心公式

### F1：Eq.15（per-token OPD 奖励）

- **公式**：见 C3。
- **来源**：K3 技术报告 §4.1.3 Eq.(15)。

### F2：对数比值分解（用于手算例子）

- **公式**：$\log(\pi_{\text{teacher}}/\pi_\theta)=\log\pi_{\text{teacher}}-\log\pi_\theta$。
- **来源**：由对数性质直接推出，用于解释"教师比学生更认可该 token 时比值为正"。
- **推导链**：对数除法 → 对数减法（基本性质）。

## 外部数字

### N1：专家数量

- **数字**：9 个专家（3 领域 × 3 努力程度）。
- **来源**：K3 技术报告 §4.1.2。
- **实验条件**：K3 训练配置。

### N2：努力程度集合

- **数字**：{low, high, max}。
- **来源**：K3 技术报告 §4.1.2、§4.1.3。
- **实验条件**：K3 训练配置。

无其他外部实验数字。K3 报告本节未给出 MOPD 的具体性能数字或 top-k 对比的数值结果，正文不构造或引用任何此类数字。

## 置信状态说明

所有核心论断均直接引用 K3 技术报告原文，置信状态为"已确认"。无"存在冲突"或"证据不足"项，满足进入生产阶段的条件。
