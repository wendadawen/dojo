# PagedAttention

- 描述：PagedAttention 是一种注意力算法，把每个序列的 key、value 切成固定 token 数的块存在不连续显存里，按块计算注意力，使 KV cache 能按需分配、跨序列共享，显存几无浪费。
- 摘要：把序列的 KV cache 切成固定大小的块，块可存放在不连续的显存里，注意力按块逐块计算；像操作系统的分页一样按需分配、用块表映射逻辑块到物理块，并允许序列之间共享块。
- 主题：推理系统, 内存与缓存
- 标签：推理加速

## 长页

- 开头｜总览：它是什么、为什么需要、大概怎么做；交代 KV cache 并链接它的页面，链接标准注意力，讲清 token 与生成的两阶段（预填、逐个解码）｜450｜paper/src-v1/main.tex L40-L48；paper/src-v1/main.tex L86-L109；paper/src-v1/main.tex L175-L186；paper/src-v1/intro.tex L64-L70

- ### 显存里装得下多少请求｜动机：模型权重常驻显存，每个请求的 KV cache 是动态部分、随生成增长；显存容量决定批大小与吞吐｜250｜paper/src-v1/main.tex L134-L137；paper/src-v1/intro.tex L30-L38

- ### 连续存法浪费在哪里｜按最大长度预留整块连续内存；三类浪费各指什么；有效内存低至 20.4%，且不能共享，压实不可行｜250｜paper/src-v1/main.tex Memory Management in Existing Systems；paper/src-v1/intro.tex L52-L62；paper/src-v1/intro.tex L57

- ### 把每个序列的 KV cache 切成块｜每个序列切成固定 token 数的块、块可存在不连续的显存里；块大小的记号 $B$ 与「块」指什么｜300｜paper/src-v1/main.tex L175-L178

- #### 在块上算注意力｜把标准注意力改写成按块算：块内分数行、分母对全部块求和、输出按块累加；各符号含义；取 $B=4$ 的小例子折叠｜320｜paper/src-v1/main.tex L178-L182

- ### KV cache 管理器｜虚拟内存类比：页对块、字节对 token、进程对请求；逻辑块、物理块与块表；逻辑与物理分离后按需分配、不必预留全部位置｜300｜paper/src-v1/main.tex KV Cache Manager；paper/src-v1/intro.tex L64-L70

- ### 一个序列从头生成，块怎么变｜跟一个请求走一遍：预填填入逻辑块、逐个解码写未满块、块满才分配新物理块；浪费限于一个块内、请求结束即释放；字面例子折叠｜300｜paper/src-v1/main.tex L207-L237

- ### 多个序列共用块｜提示的块由多个序列指向、引用计数；按块写时复制；缓存好的共享前缀同理；这种映射对上层隐藏，不同解码方式的请求可混在一批｜300｜paper/src-v1/main.tex Application to Other Decoding Scenarios

- ### 显存不够时怎么办｜先来先服务、优先抢占最晚到的；按序列整体换出；恢复可换到 CPU 内存（换出量有上界）或重算；择一取决于块大小、带宽与算力｜280｜paper/src-v1/main.tex L303-L319；paper/src-v1/main.tex L471-L482

- ### 省下多少显存、快多少｜并行采样（同一提示生成多个输出）让多个输出共用提示的块、省下 6.1%-9.8%；相同延迟下吞吐提升 2-4 倍；实验条件与数字折叠｜280｜paper/src-v1/main.tex L46；paper/src-v1/eval.tex L140-L141；paper/src-v1/intro.tex L75

- ### 适用条件与边界｜内核是什么、它慢 20%-26% 但端到端更快；块大小默认 16 的取舍；要自写读块内核；借用分页的前提；与 FlashAttention 区别一句｜300｜paper/src-v1/main.tex L419；paper/src-v1/main.tex L443-L446；paper/src-v1/main.tex L450；paper/src-v1/main.tex L363-L370；paper/src-v1/discussion.tex L4-L9；paper/src-v1/related_work.tex L27

## 短页

- ### 把各个序列的 key、value 切成块｜是什么：一句交代 KV cache（链接它）与标准注意力（链接），讲清 token，再讲把 key、value 切成固定 token 数的块、块可不连续、按块算｜300｜paper/src-v1/main.tex L40-L48；paper/src-v1/main.tex L175-L178

- ### 按最长预留，留下大片空闲｜为什么需要：现有系统按最长预留连续一段；三类浪费使有效内存很低、不能共享，能同时处理的请求数被压住｜330｜paper/src-v1/intro.tex L40-L62；paper/src-v1/main.tex L149-L166

- ### 切成块，用块表按需取用｜怎么做：切块；块表映射逻辑块到物理块、记已填位置；新 token 填未满块、块满才分配新块，浪费限于最后一个块；多个序列可共用块｜320｜paper/src-v1/main.tex L198-L213；paper/src-v1/main.tex L207-L237
