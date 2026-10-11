# LLM Serving

- 描述：LLM 服务是把大语言模型跑起来、按请求回答的在线系统，一次请求分 prefill 与 decode 两个阶段执行，快慢拆成 TTFT 与 TPOT 两个指标，两阶段可以合设也可以分离部署。
- 摘要：一次请求先 prefill、再 decode：prefill 并行处理整段输入、产出首字，decode 每步只生成一个新 token。输入较长时 prefill 偏计算、decode 偏访存，服务快慢也因此拆成首字延迟 TTFT 与每 token 生成耗时 TPOT 两个指标，对两者各设 SLO，用满足 SLO 的每卡最大请求速率 goodput 衡量；把两阶段放到同一批卡上会互相干扰，分到两组卡上各配资源能消除干扰，代价是各存一份权重并搬运中间结果。
- 主题：推理系统, 并行与通信
- 标签：推理系统

## 长页

- 开头｜概览：LLM 服务是什么、为什么需要、大致怎么工作；说清 token；被服务的是稠密或混合专家（链接 wiki/moe）模型｜320｜paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex LLM Inference；paper/src-dsv3/main.tex Introduction
- ### prefill 和 decode：一次请求的两个阶段｜prefill 并行处理整段输入、存下 KV cache（链接该页）、产出首字，输入较长时偏计算；decode 每步出新 token、复用缓存，偏访存｜560｜paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex LLM Inference
- ### 服务好不好怎么量：TTFT、TPOT 与 goodput｜把「快」拆成 TTFT 与每 token 耗时 TPOT；总延迟=TTFT+TPOT×步数；SLO=延迟上限，goodput=满足 SLO 的每卡最大请求速率｜500｜paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Background and Motivation
- ### 合设还是分离：两阶段放不放在同一批卡上｜合设省显存、不搬中间结果，但互相干扰、资源与并行方式（链接张量并行页）被绑在一起；分离消除干扰、各配资源，代价是各存一份权重、要传中间结果、按整副本扩容｜460｜paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Problems and Opportunities；paper/src-dsv3/main.tex Decoding
- ### 适用条件与边界｜两阶段与两套指标以逐 token 生成为前提；换模型负载数字变但机制不变；一条易混：goodput 不等于裸吞吐｜260｜paper/src-distserve/sections/background.tex Background and Motivation；paper/src-distserve/sections/intro.tex Introduction

## 短页

- ### 一次请求走两个阶段｜是什么：模型逐 token 生成；一次请求分 prefill（处理整段输入、存下 KV cache、出首字）与 decode（每步一个新 token）｜320｜paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex LLM Inference
- ### 两个阶段脾气不同｜为什么需要：输入较长时 prefill 要算的 token 多、偏计算，decode 单步计算量小却要搬同样多的权重和缓存、偏访存；放一起互相拖慢｜320｜paper/src-distserve/sections/background.tex LLM Inference；paper/src-distserve/sections/background.tex Problems and Opportunities
- ### 服务好不好：TTFT、TPOT 与 goodput｜怎么做：把延迟拆成首字延迟 TTFT 与每 token 生成耗时 TPOT，各设 SLO（延迟上限），用达成率与 goodput 衡量｜340｜paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Background and Motivation
