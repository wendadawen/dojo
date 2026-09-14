<!-- review-meta
round: 3
page: wiki/hy4-preview-lite/index.html
reviewed_content_sha256: bb8b6011521f1b18
-->
# Hy4 preview 轻量版压缩链路审查记录（第 3 轮）

- 页面版本：cf90710583f33a07ad0107923381a4bb855a91f1
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何规划、修复或前序审查记录）
- 已完整阅读章节：压缩对象与结果 / 低档格式：Sherry 稀疏三值量化 / 层间分档：MIX-STQ1_0 逐层混合精度 / 效果核对 / 异构设备联合推理 / 部署要点 / 来源（含页面导语、三张表格、SVG 图注与内联脚本）

## 来源核对摘要（本轮实际核到的原文片段/数值）

- 模型卡 huggingface.co/AngelSlim/Hy4-preview-GGUF（raw README）：三档产物 "Hy4-preview-Q4_K_M.gguf 435.20 GiB / 4.86"、"Hy4-preview-UD-IQ1_M.gguf 219.83 GiB / 2.44"、"Hy4-preview-STQ1_0.gguf 213.66 GiB / 2.38"；"The routed-expert gate/up projections run at 1.3125 bpw (STQ1_0) on 29 layers and 2.0625 bpw (IQ2_XXS) on the other 48"；ffn_down_exps "IQ3_XXS, IQ4_XS on last 3 layers"；Attention/gate/q_a "Q5_K"；MLA、DSA indexer "Q8_0"；若干分量 "F32"；编码器 'd = sum(w*sel*x)/sum(w*sel^2) instead of d = amax'、"1200 real expert rows"、"-89.7% weighted SSD"、"a further -4.1% of the remainder"、"alternate for 3 rounds"；ffn_down 精度更高理由 "writes straight into the residual stream, so its error is not attenuated by a later gate"、"2 levels higher"；"over NFS random page faults run at ~12 MB/s, turning a 1-minute load into hours"；补丁 "0001-hyv4-architecture.patch 18 files"、"0002-stq1_0-quant-and-cuda.patch 25 files"；"--jinja"、"The HY4 chat template matches no llama.cpp built-in family"；"~435 GiB (Q4_K_M) or ~214 GiB (STQ1_0)"；imatrix 对 STQ1_0 强制。
- 模型卡吞吐表（原文逐字）：英文节 "STQ1_0 | 204.56 ± 1.42 t/s | 20.47 ± 0.02 t/s"，中文节 "STQ1_0 | 204.56 ± 1.42 t/s | 19.52 ± 0.01 t/s"（预填充一致、解码两处不同）——页面「同一份模型卡两处不一致」成立。
- 官方 Hy4 preview 模型卡 tencent/Hy4-preview："770B total parameters, of which 49B are activated per token"、"256 routed experts and 1 shared expert"、SWE-bench Multilingual 82.9。
- 官方快讯（腾讯混元，凤凰科技 tech.ifeng.com/c/8w4NPbQRMoR、驱动人生、x-techcon 等转载）："上线后开源社区反映本地部署硬件门槛过高，呼吁推出更轻的版本"；"MIX-STQ1_0的路由专家权重不光评测表现更好，还比UD-IQ1_M少占了超5GiB"；异构实测 "一台4090笔记本和一台四卡A4000服务器…两端显存合计80GB、内存合计64GB，分处两个局域网…1.02 token/s，比笔记本单机offload快了约6倍"；"靠prima.cpp的异构调度把MoE层拆开分配，让计算和权重读取彼此重叠"。
- 评测数字（Tencent Hy 转载）："MCP Atlas：83.7 → 83.2；SWE-Bench multi：82.9 → 81.3；MRCR：81.3 → 81.1；IFBench：73.5 → 72.5"；三族路由专家占 "97.7%"。
- llama.cpp PR #22836：标题 "ggml-cpu: add STQ1_0 ternary quantization with ARM NEON vec_dot kernel"；256 weights/block、42 bytes、1.3125 bpw、"32 bytes of 4-bit codebook indices (qs[32])" + "8 bytes of 1-bit signs per group (sign[8])" + "2 bytes for a single fp16 scale"。
- arXiv:2601.07892（Sherry）：4 weights → 5 bits、1.25 bits/weight、3:4 稀疏。
- 复算通过：C(4,3)×2^3=32=2^5；42×8/256=1.3125；32+8+2=42；(29×1.3125+48×2.0625)/77=1.78；219.83−213.66=6.17；保留率 81.1/81.3=99.8% 至 81.3/82.9=98.1%；435.20 GiB/770B≈4.86 bpw、213.66 GiB/770B≈2.38 bpw。全篇公式符号单义，无 Unicode 数学字符直出。
- 机械验证：`python3 .dojo/scripts/validate.py wiki/hy4-preview-lite/index.html` 返回 `validation ok`。本环境无 Chrome/puppeteer（`.dojo/scripts/verify_render_diff.py` 的 CHROME_CANDIDATES 均不存在），未执行无头浏览器渲染实测，公式与 SVG 仅作静态核对，未发现标签压线或溢出迹象。

## 问题

- [轻微·表述] index.html:72（表格 STQ1_0 行「定位」列）：定位列写作「MIX-STQ1_0 逐层分档（轻量版主角）」，「主角」是口语化临场评价词，与 note.md「使用正式书面语，删除会话指代、临场评价和口语化过渡」不符。｜引文依据：不适用｜修复要求：把「（轻量版主角）」替换为中性描述（例如「本页所述压缩链路的主体档」）或直接删去该括注。｜修复：｜复验：
- [轻微·表述] index.html:70（表格 Q4_K_M 行「定位」列）：「标准 4-bit，安全默认档」中的「安全默认档」是把无来源支持的判断写成结论，官方模型卡三档只按文件、体积、bpw 列出，未给出该定性。｜引文依据：模型卡三档表仅含 file/size/bpw 三列，无「safe default」一类描述｜修复要求：改为可核对的客观描述（例如「三档中体积最大、bpw 最高」）或删去「安全默认档」。｜修复：｜复验：
- [轻微·来源] index.html:146（来源节「评测数字」条）：把 MCPAtlas 83.7/83.2、SWE-Bench 82.9/81.3、MRCR 81.3/81.1、IFBench 73.5/72.5 四组具体分数归到「凤凰科技 tech.ifeng.com/c/8w4NPbQRMoR」，但该 URL 正文只有「分差不到 1 分」「差距都在一两分以内」等概括，并未载这四组数字，示例来源与所指内容不符。｜引文依据：ifeng 原文「分差不到1分」；四组数字见 Tencent Hy 转载「MCP Atlas：83.7 → 83.2；SWE-Bench multi：82.9 → 81.3；MRCR：81.3 → 81.1；IFBench：73.5 → 72.5」｜修复要求：把示例来源换成载有这四组数字的来源，或删去不载数字的该 URL。｜修复：｜复验：
- [轻微·可读性] index.html:95、123、127（「imatrix」首现及正文使用处）：「imatrix」在首次使用前未定义，也未就近指向给出定义的位置，而它在页面里是载荷概念（29/48 层分界「由 imatrix 推导」、零位选择「按 imatrix 加权」、$w_i$ 定义为「imatrix 权重」），note.md 要求「基础定义位于首次使用之前」。｜引文依据：不适用｜修复要求：在首次出现处给出简短定义（llama.cpp 的重要性矩阵 importance matrix）或链到已定义该术语的前置页。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布。本轮未发现事实、公式、数字、复算或来源一致性问题：页面全部可回源条目（三档体积与 bpw、29/48 层分档、1.3125/2.0625/1.78 bpw、42 字节字节账、97.7%、5 GiB、6.17 GiB、四组评测分数与 98.1%–99.8% 保留率、8×H20 吞吐、补丁 18/25 文件、NFS 12 MB/s、80GB+64GB、1.02 token/s、6 倍）均与模型卡、官方快讯、llama.cpp PR #22836、arXiv:2601.07892 逐条吻合，页面自报的「模型卡英文节/中文节解码吞吐不一致」经逐字核对成立。上述 4 条均为轻微项，按「修复要求」处理后即可发布，不影响正确性与主线理解。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
