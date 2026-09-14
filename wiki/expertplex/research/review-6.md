<!-- review-meta
round: 6
page: wiki/expertplex/index.html
reviewed_content_sha256: e77709046ed38b82
-->
# ExpertPlex 审查记录（第 6 轮）

- 页面版本：7bde57db49dbc0ac6fb8a14df450908bc9ad2ba4（wiki/expertplex/index.html 工作树哈希）
- 论文版本：arXiv:2607.18002v2（v1 2026-07-20，v2 2026-07-21 修订；本地核对用 v2 PDF 全文文本层）
- 审查时间：2026-09-14 16:54
- 审查者：独立子代理（编排者派发的独立审查者，未参与写作，未读取 research/）
- 已完整阅读章节：核心问题（5 题及解答）；1 两条现有路线，各自的死结（1.1、1.2、补充折叠块、本章问题）；2 ExpertPlex 的架构（含误解澄清 callout、本章问题）；3 APK：在 tile 边界上调度 GPU（3.1–3.4、两个误解澄清 callout、本章问题）；4 通信：让 attention 侧发起一切（4.1–4.3、误解澄清 callout、本章问题）；5 跨栈优化器（5.1–5.3、两个折叠块、本章问题）；6 实验（6.1–6.3、三个误解澄清 callout、本章问题）；7 独立评价（7.1–7.3、本章问题）；来源与范围说明（C/F/N 清单、原图对应、简化条件）；含全部图注、脚本与 overview.html

## 核对方法与覆盖

- 论文原文：WebFetch 取 arXiv:2607.18002v2，落盘 PDF 后取文本层与图像层逐条核对。
- 页面所引 Figure 2/3/4/5/6/11 均与 assets/img-06、img-05、img-04、img-03、img-02、img-01 逐像素比对，编号、面板结构、轴标签、图内标注文字（Larger DoP! 1.5x/2x、Prefill 2/3、Decode 1/3、Network Interference!、Blocking!、Bubble!、P/pi 与 System/Device scope、Scale-up/Scale-out、Pareto 五机制）一致。
- 关键数字全部回源：权重占比 95%/96%/98%（§2.1）；部署单元 32P+320D、176 GPU、Kimi-K2 128×H200（§1/§2.4）；EP4 下 decode 17.7–34.7 μs 对 16K prefill 1.8–2.9 ms、84–101×（§4.1，比例 2.9/34.7≈83.6、1.8/17.7≈101.7 自洽）；tile 边界 2.2–25.3 μs、GEMM <10.7 μs（§4.2/§7.6/Figure 15）；H800 NVLink 160 GB/s 对 IB 50 GB/s＝3.2×（§5.1）；11.3 req/s/node 与 5.65×/2.72×/2.01×/1.41×、LooGLE 4.12×/1.28×、GLM 3.3×/5.0×/1.5×/2.5×/持平~1.5/1.66×（§7.2）；13.79×/3.33×/4.07×/+8%/1.12×（§7.3 正文，并与 Figure 12 标签值一致）；<12%/<20 μs/<10%（§7.4）；~5%/~45 μs（§7.5）；模型规格 230GB/756GB/256 专家 top-8/7.0B/22.6B/full attention 对 DSA（§7.1）；四组 SLO 1s-50ms、10s-100ms、2s-100ms、20s-100ms（§7.1）；硬件与基线描述（§7.1）。
- 四条公式 F1–F4 与原文 Eq.(1)–(4) 逐符号比对一致并可复算；构造示例 8/0.5=16、64/(0.05×20)=64、min=16 复算正确；符号 $B_p,B_d,T_p,T_d,\bar O,G,\ell,q,q',Q_{\max},x_{\mathrm{moe}},x_{\mathrm{moe}}^\star,m_e,M_t$ 全文单义。
- 引用编号 C1–C21、F1–F4、N1–N4 的章节指向逐条抽查（C1§2.1、C2§1/§2.4、C4§2.5、C5§4.1、C19§4.1、C20§5.1、C16§7.4/Fig13、C17§7.5/Fig14、C18§7.6/Fig15、F4§6.4）均与原文位置相符。
- 页面无「本页/本文/我们/你们」等自称或会话指代（脚本计数为 0）；无调试叙事、无临场评价、无 path 指向不存在文件；alt 中无 $...$；无交互视图；validate.py 返回 validation ok；dojo:topics=推理系统、dojo:tag=MoE 均在词表内；index.html 与 overview.html 互链；两个前置概念页 ../moe-serving/、../gpu-execution-model/ 真实存在且确含所声明的先修内容（goodput/MoE 术语、Green Context、CUDA Graph、SM/CTA/cluster）。
- 无发现阻断或重要问题：未定位到任何「来源不支持」「把实验条件写成无条件论断」「推断包装成来源结论」的条目；正文/图注/summary/overview 之间的同一数字（2.01×、1.66×、11.3、2.2–25.3 μs、13.79×/3.33×/4.07×/+8%/1.12×、95%/5%）四者一致。

## 问题

- [轻微·功能] index.html:159 / 204 / 276 / 345 / 410 / 495 / 534（七处 `<h3>本章问题</h3>`）：这些 h3 无 `id` 属性，而每个都是 `body` 的直接子元素，会进入第 625–652 行的目录生成脚本；该脚本对无 id 的 `body > h3` 执行 `h.textContent.trim().replace(/[\s#?？：]/g,'-')…` 生成 id，七处文本同为「本章问题」，运行时全部得到 `id="本章问题"`。后果有三：① DOM 出现七个重复 id；② 侧边目录生成七条 `href="#本章问题"` 且 `dataset.target` 相同的链接，点击任意一章的「本章问题」只会跳到第 1 章；③ 第 656–666 行的滚动高亮按 `dataset.target` 比对，会同时点亮这七条。｜引文依据：同仓库其他论文页对同级标题均显式给出唯一 id——deepep/index.html:206 `<h3 id="ch1-all-to-all-questions">本章问题</h3>`、ultraep/index.html:214 `<h3 id="nonstationary-questions">本章问题</h3>`、sherry-ternary-quant/index.html:144 同例；本页七处均为无 id 的 `<h3>本章问题</h3>`（node 复算 `"本章问题".trim().replace(/[\s#?？：]/g,"-")…` === `"本章问题"`）。｜修复要求：为这七个 h3 各加唯一 ASCII id，沿用房屋约定，建议 ch1-two-paths-questions、ch2-arch-questions、ch3-apk-questions、ch4-comm-questions、ch5-opt-questions、ch6-eval-questions、ch7-judge-questions；改后目录中七条「本章问题」各自锚点唯一、点击跳转正确、滚动高亮只点亮一条。｜修复：｜复验：
- [轻微·表述] index.html:128「各自有不同的卡点」、155「合设还有个没解决的毛病」、231「最怕的就是那个 2 毫秒的 prefill GEMM 把 30 微秒的 decode GEMM 顶在后面」、258「等于白搞」：四处口语化措辞与全文正式语域不一致（同段其余文字均为书面语）。｜引文依据：不适用｜修复要求：改写为书面语并保持原意、C 标号与数字不变——128 改为「各自有不同的问题」；155 改为「合设还有一个未解决的问题」；231 改为「共享一张 MoE GPU 的主要风险是那个 2 毫秒的 prefill GEMM 把 30 微秒的 decode GEMM 顶在后面」；258 改为「却把微秒级流水线串行化，收益被全部抵消」。改后这四句不得引入新的事实陈述。｜修复：｜复验：

## 已核对且判定为不成立（不报为问题）

- §6.2 表 MiniMax-M2.7 + LooGLE 行的「vs PDD = —」：与同表 GLM 行「OOM 无数据」并置，读作「论文未报告该比值」（§7.2 正文确未给出该倍数），非「无数据」误述，判定可接受。
- §6.3 表 CUDA stream 行 prefill 列为「—」：论文 §7.3 正文未给该值，Figure 12 的 x 轴两处组标签与 §7.3 正文（13.79× 属 decode、3.33×/4.07× 属 prefill、ExpertPlex +8%/1.12×）互相调换；页面取正文口径且与 §7.3 的机制解释（decode GEMM 短而间歇）自洽，非页面错误。
- §7.3「论文以作者键引用这类系统」：指论文以文献条目键（StepFun / Zhexiang Zhang / Ruidong Zhu 等）而非系统名引用，且该句已明示系统名为解读者补充，非论文原文，判定可接受。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：修复（仅两处轻微：目录锚点唯一性与口语化措辞，均不影响正确性与主线理解，不阻断发布）