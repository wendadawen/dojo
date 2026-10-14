# 大纲审查第 2 轮

## 读者

无

## 对照资料

1. 资料里没有｜### 自顶向下：记忆化｜查表｜位置错：“查表”不在所引 L2240、L2349-L2353 内；原文在 L2244 才写先检查表中是否已有解｜“Whenever we attempt to solve a new sub-problem, we first check the table to see if it is already solved.”（blog/dynamic-programming.md L2244）｜补引 L2244，或把依据范围扩到 L2346-L2353。
2. 资料里没有｜### 适用条件与边界｜代价｜依据缺：“代价”不在所引 L7、L2238-L2240 内；这些行只讲适用条件、子问题重叠和分治替代，没有运行时间或存储代价｜“This method also uses O(n) time since it contains a loop that repeats n − 1 times, but it only takes constant (O(1)) space, in contrast to the top-down approach which requires O(n) space to store the map.”（blog/dynamic-programming.md L2353）｜补引 L2352-L2353 或 L3172-L3328，否则删去“代价”。

## 篇幅与冗余

1. 重复｜长页 开头；长页 自顶向下：记忆化｜开头提前讲了后文的保存机制，“为什么重复子问题需要保存结果”与“递归求解时如何查表、计算并保存子问题”重合｜「为什么重复子问题需要保存结果」「递归求解时如何查表、计算并保存子问题」｜开头只留 DP 是什么和按状态递推的概览，删去“为什么重复子问题需要保存结果”；保存机制保留在「自顶向下：记忆化」。
2. 预算不合理｜长页 复杂度与状态设计｜两个例子都放在折叠块外，600 字全部计入；例子在折叠块外的预算超过 100｜「用 Fibonacci 与鸡蛋掉落两个例子说明状态数、转移数和存储如何决定代价」「600」｜把两个例子的全过程移入折叠块，折叠外只留一般结论，预算由 600 改为不超过 100。

## 处理

1. 成立：自顶向下节点依据补 L2244。
2. 成立：适用条件与边界节点依据补 L2352-L2353；“写什么”明确用时间与存储说明代价。
3. 成立：开头只保留 DP 是什么和状态递推概览，预算由 400 改为 380。
4. 成立：复杂度节改为折叠块放例子、折叠外只留结论，预算由 600 改为 100。
