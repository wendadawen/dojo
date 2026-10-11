# Expert Parallelism

- 描述：专家并行（EP）把模型的专家分片到多张 GPU 上，一个 token 用到的专家不在本卡时，它的激活经卡间通信送过去算再送回来，这两步叫 dispatch 与 combine。
- 摘要：专家总数大，把专家分片到多张卡、每张卡只放一部分，整体批随卡数一起扩大，好让每个专家收到足够多的 token。一个 token 要用的专家常在别的卡上，它的激活得用 all-to-all 搬过去算、再搬回来；各卡收到的 token 数不同，最慢的一张卡会拖慢整步，需要用冗余专家与调度来均衡。
- 主题：并行与通信, 推理系统
- 标签：并行与通信

## 长页

- 开头｜概览：EP 是什么、为什么需要、怎么工作；先交代 token、专家与 all-to-all，专家属混合专家模型（链接 wiki/moe）｜360｜paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)
- ### 专家分片到多张卡｜为何分片、一个 token 走哪几步；k=每个 token 选中的专家数；dispatch/combine 的 all-to-all；搬运计数例折叠｜600｜paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)；paper/src-shazeer/main.tex Addressing Performance Challenges
- ### 负载不均：最慢的卡决定整步｜各卡收到的 token 数不同、最慢的卡成瓶颈；用冗余专家、按负载排布与负载均衡器缓解｜420｜blog/deepseek-v3-r1-inference-system-overview.md Achieving Optimal Load Balancing；paper/src-dsv3/main.tex Prefilling
- ### 通信与专家计算重叠｜先讲清「批」，再把一批拆成两个微批，用一个微批的计算盖住另一个微批的通信；通信没有消失｜380｜blog/deepseek-v3-r1-inference-system-overview.md Computation-Communication Overlapping；paper/src-dsv3/main.tex Prefilling
- ### 适用条件与边界｜k 远小于专家数才有收益；换配置数字变、机制不变；EP 只解决权重怎么摆与激活怎么搬，服务指标另见服务页｜260｜paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)

## 短页

- ### 把专家分到多张卡上算｜是什么：先交代专家、GPU、all-to-all、token、批与激活，再讲 EP 把专家分片到多张卡、每张卡只放一部分｜320｜paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)
- ### 每个专家分到的批太小｜为什么需要：分片前每个专家分到的批很小、效率低；专家铺到多张卡后整体批随卡数变大，每个专家才收到足够多的 token｜320｜paper/src-shazeer/main.tex Addressing Performance Challenges；blog/deepseek-v3-r1-inference-system-overview.md System Design Principles
- ### 搬 token、均衡负载、重叠通信与计算｜怎么做：激活用 all-to-all 的 dispatch 与 combine 在卡间搬；各卡负载不均用冗余专家均衡；把一批拆成两个微批，用计算盖住通信｜340｜paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Computation-Communication Overlapping
