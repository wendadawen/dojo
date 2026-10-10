# 投机解码

- 描述：投机解码用一个更高效的近似模型先猜出若干候选 token，再由目标模型一次并行验证这些候选，接受其中能保证输出分布不变的部分，从而用更少的串行前向生成与目标模型同分布的结果；它不改模型架构、不需要重新训练，可直接用在现成模型上。
- 摘要：自回归解码每生成一个 token 都要跑一次大模型。先用一个更高效的近似模型猜 $\gamma$ 个候选，再让目标模型一次并行验证并接受，接受率越高、一次串行产出的 token 越多，而输出分布不变。
- 主题：推理系统, 模型结构
- 标签：推理加速

## 长页

- 开头｜自回归一次一个 token、K 个 token 要 K 次串行前向（三词各一句解释）；通常受限访存与通信、可能有空闲算力；投机解码是什么与一步流程｜280｜paper/src-v2/main.tex L143-L151；paper/src-v2/main.tex L163-L175；paper/src-v2/main.tex L193-L196
- ### 一步投机解码：猜、验、纠｜本章回答一次迭代具体做什么，只有标题，内容在五个小节｜0｜paper/src-v2/main.tex L193-L196
- #### 猜：近似模型自回归给出 $\gamma$ 个候选｜模型输出下一 token 的概率分布；各种采样都对调整后分布做标准采样，$p$、$q$ 即此分布；$M_q$ 逐步采样 γ 个 token 及各步 $q_i$｜320｜paper/src-v2/main.tex L195-L196；paper/src-v2/main.tex L198-L200；paper/src-v2/main.tex L216-L220
- #### 验：目标模型一次并行打分｜把前缀及其逐前缀加候选的 $\gamma+1$ 段一次前向，得到 $p_1,\dots,p_{\gamma+1}$；为什么这一批能并行算｜320｜paper/src-v2/main.tex L193-L196；paper/src-v2/main.tex L206；paper/src-v2/main.tex L221-L223
- #### 判与纠：接受判据与修正分布｜逐位接受测试与接受数 $n$；全接受取 $p_{n+1}$，否则取 $norm(\max(0,p-q))$；返回 $n$ 个与补的那一个｜380｜paper/src-v2/main.tex L202-L204；paper/src-v2/main.tex L224-L234；paper/src-v2/main.tex L196
- #### 一个手算例子｜自设小词表上一对具体的 $p$、$q$，走一遍猜、验、拒、纠；全过程折叠｜220｜自行构造
- #### 为什么输出分布不变｜被接受的猜与拒绝后的修正两项概率之和恰为 $p$，逐项推导折叠｜320｜paper/src-v2/main.tex L649-L677；paper/src-v2/main.tex L202-L204
- ### 接受率与加速比｜本章回答能快多少，只有标题，内容在五个小节｜0｜paper/src-v2/main.tex L238-L240
- #### 接受率 $\alpha$ 衡量近似模型有多准｜单步接受概率 $\beta$ 的定义，$\alpha=E(\beta)$，$\alpha$ 是模型与任务的内在性质｜300｜paper/src-v2/main.tex L244-L249；paper/src-v2/main.tex L328
- #### $\alpha$ 与两个分布的接近程度｜定义 $D$，由它得到 $\beta=1-D$、$\alpha=E(\min(p,q))$，$p=q$ 时取 1；推导折叠｜300｜paper/src-v2/main.tex L267-L313
- #### 一次串行产出几个 token｜在 β 独立同分布假设下期望产出 $(1-\alpha^{\gamma+1})/(1-\alpha)$；推导折叠｜300｜paper/src-v2/main.tex L249-L254
- #### 墙钟加速与最优 $\gamma$｜加速公式、代价系数 $c$、$\alpha>c$ 才有增益（生成足够长时）；最优 $\gamma$ 数值取最大；推导与数字表折叠｜320｜paper/src-v2/main.tex L316-L349；paper/src-v2/main.tex L387-L389；paper/src-v2/main.tex L395-L418
- #### 多花算力、少读访存｜并发运算涨 $\gamma+1$ 倍、总运算量可能上升；读权重与 [KV cache](wiki/kv-cache) 的访存下降；零代价 $M_q$ 的上界｜300｜paper/src-v2/main.tex L351-L375；paper/src-v2/main.tex L442-L450
- ### 近似模型怎么选｜$M_q$ 只影响速度、不影响分布；同架构、小约两个数量级通常最好；n-gram 等零代价模型（按统计直接给候选）也有非零 $\alpha$；数字折叠｜380｜paper/src-v2/main.tex L442-L458；paper/src-v2/main.tex L510-L522
- ### 适用条件与边界｜有效前提是受限于访存、有空闲算力（此处不展开机制），算力受限时无效；可套现成模型、不改架构不用重训；与拒绝采样的区别（一两句）；$\alpha$ 经验范围折叠｜300｜paper/src-v2/main.tex L604-L617；paper/src-v2/main.tex L190；paper/src-v2/main.tex L679-L689；paper/src-v2/main.tex L528-L584

## 短页

- ### 用一个近似模型猜、目标模型一次验｜是什么：token 是文本的最小单位，自回归每步只生成一个 token；近似模型猜 $\gamma$ 个候选，目标模型一次并行验证并接受，结果与目标模型同分布｜305｜paper/src-v2/main.tex L143-L151；paper/src-v2/main.tex L193-L196
- ### 自回归一次只生成一个 token，太慢｜为什么需要：K 个 token 要 K 次串行前向（一次前向＝把当前输入跑一遍模型），通常每步受限于访存而非算术、有闲置算力可用来并行验证｜300｜paper/src-v2/main.tex L163-L175
- ### 猜、验、接受、补一个｜怎么做：近似模型自回归猜 $\gamma$ 个，目标模型并行打分，逐位接受，被拒处从修正分布（一句说明）补一个；一轮最多 $\gamma+1$ 个｜330｜paper/src-v2/main.tex L193-L196；paper/src-v2/main.tex L202-L204；paper/src-v2/main.tex L216-L234
