# 大纲审查第 3 轮

## 读者

无

## 对照资料

1. 资料里没有｜### 问题形式与价值函数｜哪类状态序列｜漏条件：写成 DP 的一般问题形式，但原文把“决策步序列、价值函数、状态”限定在 mathematical optimization 语境，并说 usually｜“In terms of mathematical optimization, dynamic programming usually refers to simplifying a decision by breaking it down into a sequence of decision steps over time.”（wiki/dp/research/sources/blog/dynamic-programming.md L15）｜写清这是数学优化语境下的通常表述，不要写成所有 DP 的通例。
2. 资料里没有｜### 适用条件与边界｜误解｜找不到：依据范围内没有“误解”，sources 别处也没有概括 DP 的常见误解｜在 L7、L2238-L2240、L2352-L2353 内无对应；最接近的是 “While some decision problems cannot be taken apart this way, decisions that span several points in time do often break apart recursively.”（wiki/dp/research/sources/blog/dynamic-programming.md L7）｜删去“误解”，只保留有依据的成立条件与不适用时改用 divide and conquer 的边界；若要写误解，另补资料。
3. 资料里没有｜### 按依赖顺序求出状态的值｜先写基例｜位置错：“基例”不在所引 L18-L23、L2244-L2245、L2352-L2353 内，原文在别处｜“...with base case F1 = F2 = 1.”（wiki/dp/research/sources/blog/dynamic-programming.md L2240）｜补引 L2240，或补引 L2355-L2364 中含 “if n = 0 / return 0” 的基例。

## 篇幅与冗余

1. 重复｜长页「复杂度与状态设计」与「适用条件与边界」｜两节都重复讲时间与存储代价｜「正文只给代价由状态设计决定这一结论」；「以时间与存储说明代价」｜保留「复杂度与状态设计」中的代价结论，从「适用条件与边界」删去「以时间与存储说明代价」，该节只留成立条件、误解和不适用处理。

## 处理

1. 成立：问题形式节点改为“在数学优化语境中通常怎样把决策写成状态序列”。
2. 部分成立：删去把误解当作资料原有概括的写法；末节保留“常见误解”，但限定为“由成立条件推出的误解”，依据仍是 L2238-L2240。若正文不能自然从该依据推出，回大纲重改。
3. 成立：短页第三节依据补 L2240。
4. 成立：末节删去“以时间与存储说明代价”；代价只保留在复杂度与状态设计节。

## 最后一批改动核对

待仲裁核对：适用条件与边界节由“不适用时的处理和常见误解”改为“不适用时的处理和由成立条件推出的误解”，依据仍为 blog/dynamic-programming.md L2238-L2240。

仲裁结论：不必改。guides/concept.md 要求末章写常见误解；L2238 列出两个成立条件，L2239-L2240 展开最优子结构与子问题重叠。把误解限定为“由成立条件推出”后，位置和依据均成立，最后一批改动通过核对。
