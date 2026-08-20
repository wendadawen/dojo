# prefix-caching 术语表

| 术语/符号 | 首次出现 | 定义或含义 |
|---|---|---|
| prefix caching（前缀缓存） | 页面开头 | 跨请求识别公共前缀并复用其 KV cache 的机制 |
| context caching / prompt caching | 页面开头 | 同族机制在文献与商业 API 中的叫法 |
| 前缀（prefix） | 第 1 章 | 从序列第一个 token 开始的连续相同段（精确 token 匹配） |
| system prompt（系统提示词） | 第 1 章 | 所有请求共享的开头指令段 |
| few-shot 示例 | 第 1 章 | 提示中给出的示例输入输出对，多查询可共享 |
| RAG（检索增强生成） | 第 1 章 | 把检索到的文档拼进提示的用法；同文档多问题共享文档前缀 |
| radix tree（基数树） | 第 2 章 | 压缩前缀树：边标变长 token 段，根到节点路径 = 已缓存序列 |
| trie / 前缀树 | 第 2 章 | 每边单元素的朴素前缀树（radix tree 的对照） |
| 最长公共前缀 / 分叉点 | 第 2 章 | 新请求在树上能匹配到的最深节点 |
| 匹配 / 插入 / 逐出 | 第 2 章 | radix tree 三操作：找前缀、存新段、释放空间 |
| LRU（最近最少使用） | 第 3 章 | 逐出策略：先释放最久未用的叶子 |
| 引用计数（reference counter） | 第 3 章 | 节点上正在使用它的运行请求数；为 0 才可逐出 |
| 叶子 / 祖先（leaf / ancestor） | 第 3 章 | 树上无子节点的节点 / 有后代的所有上层节点 |
| 命中率（hit rate） | 第 4 章 | 命中而免算的 token 占比 |
| 按页匹配（per-page matching） | 第 4 章 | 只有整页落在共享前缀内才计命中 |
| 页命中数（F1） | 第 4 章 | $\lfloor \text{共享前缀长度}/\text{页大小}\rfloor$ |
| 哈希指纹（page hash） | 第 5 章 | hash(token IDs, 前一页指纹) 链式生成的页唯一标识 |
| HiRadixTree | 第 5 章谱系句 | Strata 对 SGLang RadixTree 的扩展（本页不展开） |
| delay hit（延迟命中） | 第 4 章过渡句 | 前缀正在计算中时同前缀请求到达引发的重复计算问题（Strata 页主题） |
