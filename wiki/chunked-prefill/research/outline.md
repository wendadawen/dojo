# Chunked Prefill

- 描述：Chunked Prefill 是把一条预填充请求切成若干计算量相近的块、分多次迭代算完，让正在解码的请求每轮与其中一块同批算。
- 摘要：把长预填充切成小块分次算：每块仍让 GPU 接近满载，正在解码的请求每轮与这一块合进同一个批，于是不必等整段预填充算完，解码不被打断，每轮的计算量也接近。
- 主题：推理系统
- 标签：推理加速

## 长页

- 开头｜预填充与解码两阶段的性质对比，这一概念是什么；token 在本页一句话讲清，标准注意力、KV cache 作前置链接｜200｜paper/src-v1/1-intro.tex L5-L15；paper/src-v2/1-intro.tex L28；paper/src-v2/3-motivation.tex L70；paper/src-v2/2-background.tex LLM Inference Process
- ### 切块不改结果：逐块设掩码｜块的划分与每块算一遍，逐块设置因果掩码使查询看到的键落在本块或更早块，KV cache 随块追加；因果掩码作为前置概念链接｜300｜paper/src-v1/4-design.tex L21-L28；paper/src-v2/2-background.tex L20-L22
- #### 掩码随块移动｜用一个自己设的位置编号例子说明第 1、2 块的查询各能看到哪些键，例子过程折叠｜220｜自行构造
- #### 逐块算与整段算等价｜逐块算得到的输出与整段一次算相同；推导折叠｜150｜paper/src-v1/4-design.tex L21-L22；paper/src-v1/2-background.tex The prefill and decode phases
- ### 解码与预填充块合成一批｜一块预填充与若干解码合成一个批（建立在迭代级批处理之上：请求可逐个迭代进出批），本批计算量由这块预填充决定｜300｜paper/src-v1/4-design.tex L37-L60；paper/src-v2/2-background.tex Scheduling Policies for LLM Inference；paper/src-v2/3-motivation.tex L48-L57
- #### 合并线性运算｜预填充与解码共用同一套模型权重，融合后权重只从显存取一次，解码由此转为计算受限｜200｜paper/src-v1/4-design.tex L56-L60
- #### 并入预填充几乎不增时间｜解码批内存受限，加进更多解码请求时这一轮耗时几乎不增加，所以一块预填充能顺带推进多个解码；时间数字折叠｜200｜paper/src-v2/3-motivation.tex L49-L57；paper/src-v1/tables/tbl-compute-split.tex L11-L16
- ### 为什么它同时改善吞吐和延迟｜整段预填充顶高每轮延迟、只跑解码又用不满算力，两个极端都不好；把预填充切成小块铺到多个迭代后，吞吐上去、每轮延迟被块大小限住｜250｜paper/src-v2/3-motivation.tex L66-L80；paper/src-v2/4-design.tex L24-L28
- ### 块大小与预算怎么定｜块太小降利用率、太大搭不了几个解码，块大小看预填充与解码的 token 数之比，预算按延迟目标定；块长加搭车解码数要凑成瓦片大小的整数倍｜250｜paper/src-v1/4-design.tex L68-L94；paper/src-v2/4-design.tex L30-L38
- ### 切块的开销｜越靠前的块，其 KV 被后续块重复读取的次数越多；但注意力只占整体时间小部分，所以端到端影响有限，小块则明显；不同块长的对比数字折叠｜200｜paper/src-v1/4-design.tex L25-L29；paper/src-v1/5.3-eval-ablation.tex L10-L15；paper/src-v2/4-design.tex L33-L35
- ### 适用条件与边界｜前提是提示足够长、能拆成仍足以吃满算力的块；它只提高解码侧效率、不改预填充本身；建立在迭代级批处理之上，只是把预填充摊到多个迭代里｜200｜paper/src-v1/4-design.tex L14；paper/src-v2/4-design.tex L24；paper/src-v1/5.1-eval-single-gpu.tex L34

## 短页

- ### 把预填充切成几块算｜是什么：一条预填充请求不一次算完，按 token 拆成几块、分多次迭代算；预填充、解码一句话讲清，标准注意力、KV cache 作前置链接｜220｜paper/src-v1/4-design.tex L14；paper/src-v2/4-design.tex L15；paper/src-v2/2-background.tex L20
- ### 预填充算力利用率高，解码利用率低｜为什么需要：预填充并行处理整段提示、吃满计算；解码每轮只处理一个 token、是内存受限的（本页一句讲清）；整段预填充会让在解码的请求等很久｜320｜paper/src-v1/1-intro.tex L5-L7；paper/src-v1/3-motivation.tex L7-L23；paper/src-v2/3-motivation.tex L13-L57；paper/src-v2/3-motivation.tex L69-L70
- ### 一块预填充带一批解码｜怎么做：一轮算一块，正在解码的请求搭这一轮一起算（两者放进同一个批）；不写公式｜300｜paper/src-v1/4-design.tex L21-L22；paper/src-v1/4-design.tex L37-L60；paper/src-v2/4-design.tex L24-L28
