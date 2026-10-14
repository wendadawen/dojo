# Dynamic Programming

- 描述：动态规划把大问题递归分解成少量重复子问题，并保存或按依赖更新子问题的结果来合成原问题的解。
- 摘要：先定义状态并写出它们之间的 Bellman 方程，再让小子问题的值按依赖顺序生成大问题的解；优化问题得到最优解，计数问题得到方案数。
- 主题：算法与数据结构
- 标签：通用概念

## 长页

- 开头｜说明 DP 是什么，以及大致先把大问题写成状态上的递推｜380｜blog/dynamic-programming.md L6-L8；blog/dynamic-programming.md L2238-L2240
- ### 问题形式与价值函数｜在数学优化语境中回答 DP 通常怎样把决策写成状态序列，以及价值函数和状态怎样刻画子问题｜400｜blog/dynamic-programming.md L15-L21；blog/dynamic-programming.md L2238-L2240
- ### Bellman 方程｜回答状态的值如何用后继状态表示，边界与回溯怎么做｜680｜blog/dynamic-programming.md L14-L23；blog/dynamic-programming.md L1306
- ### 自顶向下：记忆化｜回答递归求解时如何查表、计算并保存子问题｜600｜blog/dynamic-programming.md L2240；blog/dynamic-programming.md L2244；blog/dynamic-programming.md L2349-L2353
- ### 自底向上：表格与状态顺序｜回答如何按依赖顺序填表，折叠块给出自行构造的爬楼梯两种走法｜700｜blog/dynamic-programming.md L2245；blog/dynamic-programming.md L2353-L2362
- ### 复杂度与状态设计｜折叠块用 Fibonacci 与鸡蛋掉落例子核对状态数、转移数和存储，正文只给代价由状态设计决定这一结论｜100｜blog/dynamic-programming.md L2352-L2353；blog/dynamic-programming.md L3157-L3196
- ### 适用条件与边界｜总结最优子结构、子问题重叠等成立条件，不适用时的处理和由成立条件推出的误解｜500｜blog/dynamic-programming.md L7；blog/dynamic-programming.md L2238-L2240

## 短页

- ### 把大问题写成状态上的小问题｜说明 DP 用状态表示子问题，用状态之间的递推关系合成原问题｜280｜blog/dynamic-programming.md L6-L8；blog/dynamic-programming.md L15-L21
- ### 保存子问题结果，避免重复计算｜说明最优子结构和子问题重叠共同使存储结果能消去递归重复｜280｜blog/dynamic-programming.md L2238-L2240；blog/dynamic-programming.md L2352-L2353
- ### 按依赖顺序求出状态的值｜说明先写基例和 Bellman 方程，再用记忆化或填表求值｜280｜blog/dynamic-programming.md L18-L23；blog/dynamic-programming.md L2240；blog/dynamic-programming.md L2244-L2245；blog/dynamic-programming.md L2352-L2353
