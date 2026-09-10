# 滑动窗口注意力 术语表

| 术语 | 英文/缩写 | 首次出现 | 含义 |
|---|---|---|---|
| 滑动窗口注意力 | Sliding Window Attention, SWA | 页面开头 | 每个 query 只关注最近 $W$ 个位置的注意力变体 |
| 窗口 | window | 第 1 章 | 上述 $W$ 个位置；$W$ 称为窗口大小 |
| 窗口大小 | window size, $W$ | 第 1 章 | 单层可见的最大距离；V4.1-Flash 取 128 |
| 可见集 | visible set | 第 1 章 | 某个 query 在某一层实际参与打分的槽位集合 |
| 因果掩码 | causal mask | 第 1 章 | 禁止 query 关注未来位置；与窗口互补 |
| 占位槽位 | placeholder slot | 第 1 章 | 用 $-1$ 标记"该槽位还没有内容"，不参与打分 |
| 环形缓冲 | rolling / ring buffer cache | 第 2 章 | 固定 $W$ 槽的窗口 KV 缓存，位置 $i$ 写入槽位 $i \bmod W$ |
| 感受野 | receptive field | 第 3 章 | 经 $k$ 层堆叠后，一个位置可间接访问的输入层距离，约 $k\times W$ |
| 层堆叠接力 | layer-wise propagation | 第 3 章 | 信息每层最多前移 $W$ 的累积效果 |
| 注意力汇聚点 | attention sink | 第 3 章 | 被分配大量注意力、语义上不重要的位置；详见专页 |
| 压缩全局 KV | compressed global KV | 第 3 章 | 窗口之外的另一路可见性，由压缩条目提供；详见 CSA2 主题 |
| prefill 索引 | prefill indexing | 第 4 章 | 整段前向中每个 query 各得一行可见槽位 |
| decode 索引 | decode indexing | 第 4 章 | 单步解码时唯一 query 看到整个环，槽位按最旧到最新列出 |
| 缓存内位置 | position within cache | 第 4 章 | 把位置重新编号为缓存内的相对位置（StreamingLLM 的做法） |
| 原序列位置 | position in original text | 第 4 章 | 位置取 token 在原文中的下标（V4.1 的做法） |

## 符号

| 符号 | 含义 |
|---|---|
| $W$ | 窗口大小（可见距离上限） |
| $i$ | 序列位置（0 起） |
| $k$ | 层序号（0 起，本文按"第 $k$ 层"叙述时从 1 起） |
| $N$ | 序列长度 |
| $h_i$ | 位置 $i$ 的隐状态 |
| $n_{\text{score}}$ | 每个 query 的打分次数 |

## 记号约定

- 槽位（slot）指缓存数组的下标；位置（position）指 token 在序列中的下标。二者在 decode 时不同。
- 字节用 B；窗口 KV 的字节账按单序列、单层计算。
