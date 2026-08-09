#!/usr/bin/env python3
"""构建 ExpertPlex 快速阅读 overview.html。"""
import pathlib

ROOT = pathlib.Path("/Users/wendadawen/code/dojo")
SHELL = (ROOT / ".dojo/templates/paper/overview.html").read_text()

BODY = r'''
<h2>这篇论文做了什么</h2>
<p>ExpertPlex 是一个服务 MoE 大模型的系统。它让 prefill 和 decode 两个推理阶段<strong>共享同一份 MoE 专家权重</strong>、却各自独占 attention 的 GPU，并用一个能在微秒级切换的常驻 kernel（APK）管住共享，从而把有效吞吐（goodput）做到 PD 分离的 2 倍。要理解 prefill/decode、MoE、goodput 这些基础，见概念页《<a href="../moe-serving/index.html">MoE 大模型推理与服务基础</a>》；要理解 GPU kernel 与 tile，见《<a href="../gpu-execution-model/index.html">GPU 执行模型</a>》。</p>

<h2>创新点</h2>
<ul>
  <li><strong>共享专家 + 分离注意力</strong>：MoE 权重占参数 95% 以上，共享一份就消除了跨阶段冗余；attention 不足 5%，按阶段独占整卡保住本地算力。部署粒度从此和 MoE 权重解耦，不再像 PD 分离那样随模型变粗。</li>
  <li><strong>自适应常驻 kernel（APK）</strong>：在 tile 边界（实测 2.2–25.3 微秒、与序列长度无关）调度 MoE 计算，做到有界抢占和 SM 重分配，无需 CPU 介入、兼容 CUDA Graph——这是现有 GPU 共享机制（stream 优先级、MPS、MIG、Green Context）都凑不齐的五项能力组合。</li>
  <li><strong>attention 发起的一侧通信</strong>：dispatch 由 attention push、combine 由 attention pull，去掉 MoE 侧的 ring buffer 和轮询，从而消除两阶段并发的死锁，还能让一个阶段的通信与另一阶段的计算跨阶段重叠。</li>
  <li><strong>tile 到集群的跨栈联合优化</strong>：把并行度、布局、重叠策略、APK 共享策略一起搜索最大化 goodput，并用 tile 数（而非 token 数）建模 MoE 延迟。</li>
</ul>

<h2>大致怎么做</h2>
<ol>
  <li>把每个节点的 GPU 分成 prefill attention 服务器、decode attention 服务器、共享 MoE 服务器三类；前两类可为空。</li>
  <li>每块 MoE GPU 上跑一个 APK 常驻 kernel：它把两个阶段的 MoE 操作排成队列，按 CTA cluster 粒度领 tile 执行；decode 紧急时沿存储层级发一个协作决策，所有 cluster 在下一个 tile 边界一致切换。</li>
  <li>抢占上界 = 一个 tile 时间 + 一次检查，与被中断操作总长无关；单阶段就绪时用全部 SM，竞争时按公式给 decode 分配、prefill 拿剩余。</li>
  <li>通信全部由 attention 侧发起，直写 MoE 服务器的最终 buffer；prefill 的跨节点流量尽量走 prefill attention 服务器之间做分层去重，decode 直连，必要时用 IB 虚拟通道优先级隔离。</li>
  <li>离线搜索枚举布局、用 tile 感知延迟模型估迭代延迟、选 goodput 最大的配置；在线由 APK 按 tile 足迹实时调整 SM 预算。</li>
</ol>

<h2>关键结论</h2>
<ul>
  <li><strong>goodput 提升</strong>：MiniMax-M2.7 + ShareGPT 上 ExpertPlex 达 11.3 请求/秒/节点，是 PD 分离的 2.01×、PD 合设（PDMux）的 1.41×；GLM-5.1-FP8 + LooGLE 上对 PDMux 是 1.66×。</li>
  <li><strong>不是全面碾压</strong>：在 GLM-5.1-FP8 + ShareGPT 上 ExpertPlex 与 PDMux 持平在约 1.5 请求/秒/节点——PDMux 的 TP attention 对短请求 TTFT 有利。条件换了结论就变。</li>
  <li><strong>APK 微基准</strong>：相对独占执行，APK 只让 decode 慢 8%、prefill 慢 1.12×；而 CUDA stream 优先级让 decode 慢 13.79×，MPS/Green Context 让 prefill 慢 3.33×/4.07×。</li>
  <li><strong>成立条件</strong>：SGLang 系基线、H800 硬件、FP8 模型、ShareGPT/LooGLE 长度分布、Poisson 到达、论文给定 SLO；多节点验证只到 3 台；GLM 上 PD 分离因 OOM 无数据、部分基线用 16GPU 布局。</li>
</ul>
'''

out = SHELL.replace("【主题标签】", "MoE 服务系统")
out = out.replace("【论文标题】", "ExpertPlex：高吞吐的 MoE 分离式服务系统")
out = out.replace("【一句话定位：这篇论文做了什么】",
                  "让 prefill 和 decode 共享 MoE 专家、独占 attention，用微秒级 tile 调度管住共享，把 MoE 大模型 goodput 做到 PD 分离的 2 倍。")
out = out.replace("【论文链接】 · 作者 · 发表 · 更新于 【日期】",
                  'arxiv.org/abs/2607.18002 · Wu, Jin, Zhang 等 · arXiv 2026 v2 · 更新于 2026-08-07')
out = out.replace('    <!-- @content -->\n    <!-- 快速阅读正文：做了什么、创新点、大致怎么做（高层）、关键结论。5 分钟内可扫读，不展开机制与公式推导。 -->',
                  '    ' + BODY.replace('\n', '\n    '))

(ROOT / "wiki/expertplex/overview.html").write_text(out)
print("overview.html written, size:", len(out))
