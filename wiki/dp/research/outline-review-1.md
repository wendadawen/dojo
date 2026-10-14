# 大纲审查第 1 轮

## 读者

1. 顺序错乱｜开头｜「大致按状态递推工作」先用了“状态递推”｜读到这里我缺“状态是什么、状态之间怎样递推”这个前置知识；此前没有节点讲过它，本节也没有定义｜把“状态”和“递推”作为前置概念链接成页，或把状态定义移到这句之前。
2. 顺序错乱｜把大问题拆成会重复的小问题｜「有最优子结构」先用了“最优子结构”｜读到这里我缺“最优解为什么由子问题的最优解组成”这个前置知识；短页此前没有节点讲过它，本节只用它说明 DP 适合什么题，没有讲它的含义｜把“最优子结构”作为前置概念链接成页，或在第一节先定义它。
3. 顺序错乱｜先定状态和递推，再填表｜「从 Bellman 方程到记忆化或填表实现」先用了“Bellman 方程”｜读到这里我缺“Bellman 方程是什么、它怎样表达状态递推”这个前置知识；短页前两节都没有讲它，本节直接把它当成实现过程的起点｜把“Bellman 方程”作为前置概念链接成页，或在第三节开头先解释它。

## 对照资料

1. 资料里没有｜描述｜说过头：“保存每个子问题的解”被写成 DP 通例；原文的自底向上实现只占 O(1) 空间，不需要保存全部子问题｜“This method also uses O(n) time since it contains a loop that repeats n − 1 times, but it only takes constant (O(1)) space, in contrast to the top-down approach which requires O(n) space to store the map.”（wiki/dp/research/sources/blog/dynamic-programming.md L2353）｜改为“保存已解子问题的结果，或按依赖只保留必要值”。
2. 资料里没有｜摘要｜说过头：“生成大问题的最优解”把 DP 限定成优化问题；原文明确 DP 也能计数解｜“Dynamic programming makes it possible to count the number of solutions without visiting them all.”（wiki/dp/research/sources/blog/dynamic-programming.md L2780）｜改为“生成大问题的解（优化问题中为最优解）”，或补上计数类 DP 的口径。
3. 资料里没有｜适用条件与问题形式｜位置错：“状态和价值函数怎样刻画它”不在所引 L2238-L2240、L2245 内；状态和价值函数定义在别处｜“This is done by defining a sequence of value functions V1, V2, ..., Vn taking y as an argument representing the state of the system at times i from 1 to n.”（wiki/dp/research/sources/blog/dynamic-programming.md L16）｜补引 L15-L21，并注明这是 Mathematical optimization 语境。
4. 资料里没有｜复杂度与状态设计｜依据缺：“状态数、转移数和存储如何决定代价”不在所引 L2239-L2240、L2352-L2353 内；原文只在具体例子中给出复杂度结果｜“Notice that the above solution takes O(nk²) time with a DP solution.”（wiki/dp/research/sources/blog/dynamic-programming.md L3172-L3188）｜若保留一般规则需另补来源；若只讲示例，补引 L2353、L3157-L3168、L3172-L3188，另见“适用条件与边界”的“代价”。
5. 资料里没有｜复杂度与状态设计｜依据缺：“状态要保留足够信息”不在所引范围内；原文在别处用具体问题说明状态需要保留哪些信息｜“Imagine backtracking values for the first row – what information would we require about the remaining rows, in order to be able to accurately count the solutions obtained for each first row value?”（wiki/dp/research/sources/blog/dynamic-programming.md L2780）｜补引 L2780、L2806-L2889，或删去这一断言。
6. 资料里没有｜先定状态和递推，再填表｜位置错：“记忆化”不在所引 L14-L23、L2245 内；记忆化的原文在别处｜“This technique of saving values that have already been calculated is called memoization; this is the top-down approach...” （wiki/dp/research/sources/blog/dynamic-programming.md L2352）｜补引 L2244、L2352-L2353（如需说明条件，可另引 L2246 的 referentially transparent 限定）。

## 篇幅与冗余

1. 重复｜长页 适用条件与问题形式 ↔ 长页 适用条件与边界｜两处都在讲适用/成立条件，前者问“什么样的问题能用 DP”，后者又总结“成立条件”｜前者：“回答什么样的问题能用 DP，以及状态和价值函数怎样刻画它”；后者：“总结成立条件、代价、误解和不适用时该换成什么视角”｜删去前节“回答什么样的问题能用 DP”，只保留问题形式与状态/价值函数刻画；成立条件集中到末节。
2. 预算不合理｜长页 自顶向下：记忆化、长页 复杂度与状态设计｜两个核心机制节预算 560、520，低于背景“适用条件与问题形式”620 和边界“适用条件与边界”620｜“自顶向下：记忆化”560；“复杂度与状态设计”520；“适用条件与问题形式”620；“适用条件与边界”620｜把“适用条件与问题形式”删重复后压到 400，把“适用条件与边界”压到 500。
3. 短页不合格｜短页 把大问题拆成会重复的小问题｜一节里同时要求说明“最优子结构”和“子问题重叠”两件事｜“说明 DP 面向有最优子结构且子问题重叠的问题”｜只保留一个要点，例如保留“子问题重叠”，删去“最优子结构”。
4. 短页不合格｜短页 先定状态和递推，再填表｜一节同时讲状态、递推和“记忆化或填表”两套实现，超过一件｜“说明从 Bellman 方程到记忆化或填表实现的过程”｜删去“记忆化或填表实现的过程”，本节只保留状态与递推。

## 处理

1. 成立：开头改为“先把大问题写成状态上的递推”，状态与递推由“问题形式与价值函数”和“Bellman 方程”紧接定义。不改链接；“状态”“递推”“最优子结构”“Bellman 方程”是本概念内部核心词，不是基线外前置概念。
2. 成立：短页第一节改为“用状态表示子问题，用状态之间的递推关系合成原问题”，第二节先给出最优子结构含义，再接子问题重叠。
3. 成立：短页第三节改为“先写基例和 Bellman 方程，再用记忆化或填表求值”，在该节内部按序解释三者。
4. 成立：描述改为“保存或按依赖更新子问题的结果”，避免把保存全部子问题写成通例。
5. 成立：摘要补优化与计数两类结果，引用资料中的计数能力。
6. 成立：章节改为“问题形式与价值函数”，依据补 L15-L21；末节集中写成立条件，避免重复。
7. 成立：复杂度节改为只依据 Fibonacci 与鸡蛋掉落两个资料例子；状态设计的一般化展开删去。
8. 成立：短页第三节依据补记忆化与填表位置。
9. 成立：核心机制预算调到 600 及以上，边界节压到 500。
10. 不成立：读者第 1 条中“状态”“递推”按概念内部词处理，已通过节点顺序解决；读者第 2 条中“最优子结构”和资料对照员第 2 条不构成必须前置链接，已在页内定义或限定口径。
