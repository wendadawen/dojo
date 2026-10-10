# 辅助损失无关路由

- 描述：辅助损失无关路由是 MoE 训练里的一种负载均衡做法：给每个专家加一个偏置，只加到「选哪些专家」的判定上、不加到门控权重，再按专家负载用固定小步更新这个偏置，因此不用额外写辅助损失项。
- 摘要：把专家偏置 $b_i$ 加到路由分数上，只用 $s_{i,t}+b_i$ 决定 top-k 选谁，门控值仍取原始 $s_{i,t}$；每步训练后按 $b_i \leftarrow b_i + \gamma\,\mathrm{sign}(\bar c_i - c_i)$ 更新偏置，过载减、欠载增，不产生干扰梯度。
- 主题：训练与优化, 模型结构
- 标签：MoE

## 长页

- 开头｜概览：MoE 训练会负载失衡；一句说明辅助损失（鼓励均衡的额外损失项）及代价，改用不进梯度的偏置；首次出现 MoE 链接 `wiki/deepseek-moe`｜400｜paper/src-lfb/iclr2024_conference.tex Introduction；paper/src-lfb/iclr2024_conference.tex Auxiliary Loss for Load Balance；paper/src-dsv3/main.tex with Auxiliary-Loss-Free Load Balancing
- ### 为什么要绕开辅助损失｜路由坍塌的现象与后果、专家并行下失衡拖慢；辅助损失（权重 $\alpha$）把均衡写进梯度目标，$\alpha$ 过大过小的两难与干扰梯度｜700｜paper/src-lfb/iclr2024_conference.tex Mixture-of-Experts；paper/src-lfb/iclr2024_conference.tex Auxiliary Loss for Load Balance；paper/src-lfb/iclr2024_conference.tex The Dilemma Between Load Balance and Model Performance
- ### 偏置只决定选哪些专家，不改门控权重｜路由分数与带偏置的选择式：判定用 $s_{i,t}+b_i$、门控值取原始 $s_{i,t}$；MoE 层输出；两个边界核对｜880｜paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-dsv3/main.tex Basic Architecture of ；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing.
- ### 按负载偏差的符号更新偏置｜负载、均值与负载偏差的含义；更新式与方向；为何规则更新、为何用 sign、因果约束、初始化｜860｜paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-lfb/iclr2024_conference.tex L239-L258；paper/src-lfb/iclr2024_conference.tex Update rule.；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing.
- ### 手算一个 4 专家选 2 的训练步｜4 专家选 2、一批 4 个 token、初始偏置全 0、$\gamma=0.1$：正文一两句说明道理，选择与负载、偏差与更新、下一轮变化的全过程折叠｜90｜自行构造
- ### 适用条件与边界｜只做批级均衡、不保证序列内（DeepSeek-V3 另加序列级损失）；$\gamma$ 两端代价、sign 需多步逼近；误解：偏置改了加权｜680｜paper/src-lfb/iclr2024_conference.tex Update rate.；paper/src-lfb/iclr2024_conference.tex Update rule.；paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-dsv3/main.tex Complementary Sequence-Wise Auxiliary Loss.；paper/src-dsv3/main.tex Training Hyper-Parameters.

## 短页

- ### 一个只进选择的偏置｜是什么：给每个专家加一个偏置，只加到「选哪些专家」的判定、不加到门控权重；补一句 MoE 并链接 `wiki/deepseek-moe`｜270｜paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-dsv3/main.tex with Auxiliary-Loss-Free Load Balancing
- ### 不写损失项也能均衡｜为什么需要：MoE 训练会负载失衡、少数专家包揽多数 token；辅助损失混进主任务梯度损害性能、强度难调｜300｜paper/src-lfb/iclr2024_conference.tex Introduction；paper/src-lfb/iclr2024_conference.tex The Dilemma Between Load Balance and Model Performance
- ### 加偏置选专家，按负载调偏置｜怎么做：判定进 top-k 用 $s_{i,t}+b_i$、门控取原始 $s_{i,t}$；每步后按负载（分到的 token 数）符号增减偏置｜360｜paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing.
