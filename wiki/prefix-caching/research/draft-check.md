# prefix-caching 前置概念页引用状态

## 状态记录

- 页面位置：wiki/prefix-caching/index.html 与 overview.html（已存在）
- 状态：2026-08-19 之前的会话已完成本页面（主题：跨请求复用相同前缀的 KV cache：SGLang RadixAttention 树实现 + vLLM/Mooncake 哈希指纹实现 + 页大小对命中率的影响 + PagedAttention 页级共享基础）。本任务范围内未重新生成。
- 字数：约 46 KB（index.html）+ 4.9 KB（overview.html）
- 章节结构：5 章（1 什么能复用；2 radix tree 组织与匹配；3 LRU 逐出与引用计数；4 命中率与页大小；5 两种实现——radix tree 与哈希）+ 来源与范围说明
- 元数据：description / dojo:summary / dojo:type=concept / dojo:topics / dojo:tag 全部填写
- 机械校验：python3 .dojo/scripts/validate.py wiki/prefix-caching/index.html → validation ok；overview.html → validation ok
- 站内引用：链接到 wiki/paged-attention/index.html、wiki/kv-cache/index.html、wiki/strata/index.html 等已写完成的概念页

## PPD 论文页对 prefix-caching 的引用关系

按 plan.md 2.4（前置知识映射）约束：正文里引用的概念页都要真实存在，不留"（待生成）"标注。现有 prefix-caching 页面已覆盖 PPD 引用所需的最小含义：

- 块/页级粒度匹配（前缀缓存页 F1、§4 命中率）：满足 PPD 页 §5 末尾"缓存位置割裂"概括 → 计算应该靠近哪份缓存的指路所需的机制常识
- 跨请求复用 KV cache（§1 什么能复用 + §2 匹配）：满足 PPD 页 §4.4 vLLM 实现细节中"vLLM prefix cache"指代的最小含义
- 多轮对话历史复用 + 生成结果也入缓存（§1 例子）：直接呼应 PPD 页核心场景（多轮对话 KV 缓存可复用）

唯一未在现有页面出现的 PPD 引用条款是"PD 分离下 D 节点 KV 缓存与 P 节点 prefix cache 互相不可达"——这是 PPD 论文 §2.2 揭示的部署拓扑问题，按 plan.md 2.4 规则"某个概念页还没生成完时，正文先写这个概念在当前上下文里需要的最小含义"，PPD 页正文已就地说明该现象（见 PPD §5 末尾"缓存位置割裂"段），不依赖外部概念页。

## 决策

接受现有 prefix-caching 页面作为本任务的前置概念页。本任务范围内的概念页三轮独立审查（按 concept/check.md 流程）不适用于该页面——它不是本任务生成的，且已通过 validate.py 与现有引用关系正常。

## 写作偏差

无。本任务范围内的所有正文均通过 validate.py。
