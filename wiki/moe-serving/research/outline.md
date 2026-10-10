# MoE 大模型推理与服务基础

- 描述：MoE 大模型推理与服务基础，讲一种把部分层的前馈网络换成一排专家的模型：每个 token 由路由器选中少数几个专家计算，总参数随专家数增长而每 token 计算量不涨；这些专家要分片到多张 GPU 上，一次请求分 prefill 与 decode 两个阶段执行。
- 摘要：把前馈层换成一排专家，每个 token 只让其中几个参与计算，于是总参数可以很大而每 token 的计算量几乎不涨；代价是专家要分片到多张 GPU 上，token 的激活得在卡间用 all-to-all 搬运。一次请求分 prefill 与 decode 两个阶段，服务快慢要拆成首字延迟与出字速度分别度量。
- 主题：推理系统, 模型结构, 并行与通信
- 标签：MoE

## 长页

- 开头｜总览：MoE 是什么、为什么需要、大概怎么工作；先交代 token、参数、多层堆叠、每层有 attention 与 FFN；以 DeepSeek-V3 为例｜380｜paper/src-shazeer/main.tex Introduction and Related Work；paper/src-dsv3/main.tex Introduction
- ### 一排专家加一个路由器｜MoE 层怎么算：路由器给每个 token 打分、只留 top-k 个、按权重求和；层输出公式与 8 专家 top-2 小例子折叠｜760｜paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-shazeer/main.tex Gating Network；paper/src-dsv3/main.tex Architecture
- ### 存得多、算得少｜稀疏激活：总参数随专家数增长、每 token 计算量只随选中个数增长；DeepSeek-V3 的 256 选 8 与 671B/37B｜380｜paper/src-dsv3/main.tex Introduction；paper/src-dsv3/main.tex L722-L727；paper/src-shazeer/main.tex Introduction and Related Work
- ### 专家并行：把专家切到多张卡｜为什么把专家分片到多卡、dispatch/combine 搬什么；2 卡 4 专家 top-2 计数例折叠｜760｜paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)；paper/src-shazeer/main.tex Addressing Performance Challenges
- ### 负载不均：最慢的卡决定整步｜各卡收到的 token 数不同，最慢的卡成为瓶颈拖慢整体；服务侧用冗余专家与负载均衡调度缓解｜400｜blog/deepseek-v3-r1-inference-system-overview.md Achieving Optimal Load Balancing；paper/src-dsv3/main.tex Prefilling；paper/src-shazeer/main.tex Balancing Expert Utilization
- ### 通信与专家计算重叠｜先讲清「批」，再把一批拆成两个微批，用一个微批的计算盖住另一个微批的通信；通信没消失｜380｜blog/deepseek-v3-r1-inference-system-overview.md Computation-Communication Overlapping；paper/src-dsv3/main.tex Prefilling
- ### prefill 和 decode：一次请求的两个阶段｜prefill 并行处理整段输入、建 KV cache（链接该页）、产出首字，计算受限；decode 每步算一个新 token、复用缓存，访存受限｜580｜paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex LLM Inference；paper/src-dsv3/main.tex Inference and Deployment
- ### 服务好不好怎么量：TTFT、TPOT 与 goodput｜把「快」拆成首字延迟（TTFT，看 prefill）与出字速度（TPOT，看 decode）；SLO 与 goodput 各是什么｜520｜paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Background and Motivation
- ### 部署：两阶段合设还是分离｜合设省显存无 KV 传输但相互干扰；分离消除干扰但每实例一份完整副本、按部署单元（最小整副本组合）扩容｜480｜paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Problems and Opportunities；paper/src-dsv3/main.tex Decoding
- ### 适用条件与边界｜k 远小于专家数才省计算；换配置后数字失效但机制成立；两条误解：稀疏激活省显存、goodput 就是吞吐｜280｜paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-dsv3/main.tex Hyper-Parameters；paper/src-distserve/sections/background.tex LLM Serving Optimization

## 短页

- ### 一排专家，每个 token 只让几个参与计算｜是什么：先交代 token、参数与 FFN，再讲 MoE 把前馈层换成一排专家加一个路由器，每个 token 只选其中 top-k 个｜340｜paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-dsv3/main.tex Architecture
- ### 存得多、算得少｜为什么需要：稠密模型（全部参数都算的普通模型）参数翻倍则每 token 计算量也翻倍；MoE 让总参数随专家数增长而每 token 只算选中几个｜300｜paper/src-dsv3/main.tex Introduction；paper/src-shazeer/main.tex Introduction and Related Work
- ### 切到多卡、分成两段执行｜怎么做：先讲清 all-to-all 与 prefill/decode 两阶段、TTFT 与 TPOT 是什么；专家分片到多卡，权重分开放｜340｜paper/src-dsv3/main.tex Inference and Deployment；paper/src-distserve/sections/background.tex LLM Inference；paper/src-distserve/sections/intro.tex Introduction
