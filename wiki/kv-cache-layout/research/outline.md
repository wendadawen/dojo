# 文章大纲

页面路径：wiki/kv-cache-layout/index.html 与 overview.html。

## 页面开头

具体场景引入：推理引擎的部署文档或代码注释里出现"KV cache 使用 HND 布局"，或者环境变量 `VLLM_KV_CACHE_LAYOUT=HND`——这句话描述的不是数据内容，而是同一批 K/V 数在显存里的排列方式。场景：同一份 KV cache，两种排法，各有代价。本文回答：这两种排法是什么、各自优化哪条路径、vLLM 怎么在同一块显存上支持两者。

组件顺序：blockquote.meta（主要依据：FlashInfer 官方 kv_layout 文档、vLLM main 源码、FreeKV 论文）→ 引言（问题、范围、结构）→ learning-goals（核心问题 4 条 = scope Q1–Q4）。不使用开头 misconceptions 组件，误解分散到各章处理。

贯穿示例（全文固定，构造数据）：一页 KV cache，$p=4$ 个 token（$t_0$–$t_3$）、$H_{\mathrm{kv}}=2$ 个 KV 头（$h_0$、$h_1$）、$d=2$ 维，共 16 个数，用 0–15 表示物理槽位号。第 1 章引入排布，第 2 章写 $t_2$，第 3 章读 $h_0$ 整页，第 4 章 as_strided 代码复用同一组数。

## 第 1 章 一页 KV cache 的三个维度（回答 Q1）

- 章节问题：
  1. KV cache 的三个维度 $N/H/D$ 各自统计什么，从哪里来？
  2. NHD 与 HND 分别把哪个维度排在外层，物理内存里谁连续？
  3. 分页形式下两种布局的完整形状是什么？"HDN"是第三种布局吗？
- 完成答案要点：
  1. $N$ = token 数（序列方向），$H$ = KV 头数（GQA 下 $H_{\mathrm{kv}}$ 少于查询头数），$d$ = 每头维度；来源是每层对每 token 的 $k=xW_K$、$v=xW_V$（引用 kv-cache 页、mqa-gqa 页）
  2. 字母顺序即物理维度从外到内的顺序：NHD 下同一 token 的 $H_{\mathrm{kv}}$ 个头连续，HND 下同一头的整页 $p$ 个 token 连续；用贯穿示例的 16 个槽位写出两种排布
  3. NHD 页形状 $(n_{\text{page}}, p, H_{\mathrm{kv}}, d)$、HND $(n_{\text{page}}, H_{\mathrm{kv}}, p, d)$（F1）；HDN 不是标准写法，业界只有 NHD/HND 两种命名（C1 佐证 FlashInfer 只定义两种）
- 对应范围：Q1；C1、C5；F1
- 正文要点：维度来源 → 排列含义 → 贯穿示例排布 → 分页形式 → 拼写澄清
- 表达材料：
  - 图 1（内联 SVG，内存条带）：同一页 16 格在两种布局下的物理顺序，蓝橙两色区分 $h_0/h_1$，格内标 token 序号；职责=让"谁连续"直接可见（对应 C1/C5 的文字定义）
  - 对照表：NHD/HND 六列（页形状、连续对象、单头取一页的步长、自然路径、代表使用者、对应 vLLM 布局名）——前两列本章填，后几列留到第 3、4 章回填（写作时统一放第 3 章末或拆两张表，以避免空列：拆分——本章只放两列小表：页形状+连续对象）
- 前置知识安排：KV cache 存什么（kv-cache 页链接，首次依赖处）；GQA 头数（mqa-gqa 页链接）；分页机制（paged-attention 页链接，在分页形式小节）

## 第 2 章 写入路径：NHD 与投影输出一致（回答 Q2）

- 章节问题：
  1. 投影输出 $xW_K$、$xW_V$ 的形状是什么，为什么与 NHD 一致？
  2. decode 每步追加一个 token 时，NHD 下的写入是什么模式？HND 下会额外发生什么？
- 完成答案要点：
  1. 一批 $N$ 个 token 的隐状态 $X$ 形状 $(N, d_{\text{model}})$，$W_K$ 形状 $(d_{\text{model}}, H_{\mathrm{kv}} \cdot d)$，乘积 $(N, H_{\mathrm{kv}} \cdot d)$，reshape 成 $(N, H_{\mathrm{kv}}, d)$ 就是 NHD，零转置（C2）
  2. decode 每步一个 token：写入该 token 的 $H_{\mathrm{kv}}$ 对 $k/v$ 向量，NHD 下这 $2H_{\mathrm{kv}} \cdot d$ 个数在（K、V 各自的）连续段内；若物理布局是 HND，同一 token 的不同头散在 $H_{\mathrm{kv}}$ 个相距 $p \cdot d$ 的位置，写入 kernel 要按头算地址（C18 的 transpose 适配即是证据）；用贯穿示例写 $t_2$
- 对应范围：Q2；C2、C18（写入侧证据）
- 正文要点：投影形状推导 → 零转置论断 → decode 追加写模式 → 贯穿示例 $t_2$ 落位 → 引出"读路径另有诉求"的过渡
- 表达材料：无新图（复用图 1 定位 $t_2$ 的落位）；小段手算（$X W_K$ 形状计算，构造数字）
- 前置知识安排：无新增

## 第 3 章 读取路径：HND 把单头整页变连续（回答 Q3）

- 章节问题：
  1. attention 计算读取 KV cache 的访问单位是什么，为什么按头取整页？
  2. 两种布局下"取一个 KV 头的一整页"的传输单元各是多少？代入 $d=128$、fp16、$p=32$ 复算。
  3. HND 的收益集中在哪些场景？为什么 fp16 下差异不显著、默认仍是 NHD？
- 完成答案要点：
  1. GQA 下一个查询头组对齐一个 KV 头，attention 沿序列维度扫全部历史 token，即按 (头, 整段序列) 读取；分页下表现为按 (头, 页) 取块（引用 mqa-gqa、paged-attention）
  2. NHD 单头取一页碎成 $p$ 段、每段 $d$ 个元素；HND 一段 $p \cdot d$ 个元素（F3、C6）；$d=128$、fp16、$p=32$：256 B 对 8 KB（N1、N2），32 倍差
  3. 收益场景一：低精度（fp8/nvfp4）kernel——HND 更友好（C3），trtllm-gen 默认 HND、NHD 触发转置拷贝（C8）、TMA 16 字节盒宽约束（C9，深层机制部分标注推断）；场景二：页级大块搬运（offload/换入换出）——FreeKV 的 CPU 端 HND 与混合布局（C7）；fp16 无显著差异 + vLLM 默认偏好 NHD（C4、C15）；误解 1 在此处理
- 对应范围：Q3；C3、C4、C6、C7、C8、C9；F3；N1、N2
- 正文要点：读访问单位 → 传输单元对比与复算 → 低精度场景 → 页搬运场景 → "没有全局最优"（FreeKV 混合布局）→ 默认 NHD 的理由
- 表达材料：
  - 图 2（内联 SVG 或 HTML）：取 $h_0$ 整页的两种访问模式——NHD 下 4 段碎块（每段 $d=2$ 格）对 HND 下 1 段连续块（$p \cdot d=8$ 格），复用图 1 的颜色与编号；职责=把 F3/N1/N2 的"碎片对整段"可视化
  - 对照表（全量版放本章末）：连续对象 / 单头取一页的传输单元 / 典型收益场景 / 代表使用者
- 前置知识安排：GQA 组内共享（已引用）；无新增

## 第 4 章 vLLM 实现：逻辑形状固定，stride 置换切换布局（回答 Q4）

- 章节问题：
  1. 什么是张量 stride？同一块物理内存如何同时表达 NHD 与 HND 两种视图？
  2. vLLM 的 `KVCacheLayout` 枚举怎么定义布局，NHD/HND 别名对应哪两个成员？
  3. 布局如何在引擎里解析：后端声明、协商、`VLLM_KV_CACHE_LAYOUT` 各起什么作用？SM100 上为什么解析成 head-major？
  4. 切换布局需要复制数据吗？写入 kernel 怎么适配布局？
- 完成答案要点：
  1. stride = 下标每加一，物理位置移动的元素数；行主序 NHD 视图 stride $(H \cdot d, d, 1)$、HND $(N \cdot d, d, 1)$（F2）；`as_strided` 同一 storage 两个视图（C20，可运行代码）
  2. 逻辑形状恒 $[L, B, H, N, C]$（L 层、B 块、H 头槽、N 块内状态数、C 每格字节数，K/V 沿 C 拼接）；枚举值是 stride 置换，LBNHC=(0,1,3,2,4) 即 NHD、LBHNC=(0,1,2,3,4) 即 HND（C10、C12、C11）
  3. engine core 跑一次：各后端 `supported_kv_cache_layouts` 声明偏好列表，取交集；显式 `VLLM_KV_CACHE_LAYOUT` 必须在候选内；无声明时默认偏好 LBNHC 最先（C13、C14、C15）；SM100 trtllm-gen 只吃 head-major 块内布局 → FlashInfer 声明 (LBHNC, BLHNC)（C16、C19、N3）；翻译成 FlashInfer 名（C17）
  4. 不复制：布局是同一 storage 的视图问题（C20 验证）；写入路径 per-layer 视图 (B,H,N,2·hs) 先 transpose(1,2) 得 (B,N,H,hs) 再喂 reshape_and_cache_flash（C18）；误解 2 在此处理
- 对应范围：Q4；C10–C20；F2；N3、N4
- 正文要点：stride 最小解释 → as_strided 代码验证 → KVCacheLayout 枚举与别名 → 形状体系 $[L,B,H,N,C]$ → 协商与环境变量 → SM100 路径 → 写入适配 → 演进注记（旧版每层独立 buffer + `(num_blocks, 2, block_size, H, d)` + permute，机制同源）
- 表达材料：
  - 可运行代码折叠块（as_strided 双视图，参数与贯穿示例一致 N=4/H=2/d=2 的 16 元素版本，便于与图 1 对照；三段说明：验证机制/观察重点/简化条件）
  - 图 3（HTML 结构图 dg-flow）：逻辑形状 $[L,B,H,N,C]$ → stride 置换 (0,1,3,2,4) → 物理顺序，两行并排展示 NHD/HND 两条路径；职责=定位"枚举值=置换"这个核心机制
  - 对照表：vLLM 布局名 ↔ FlashInfer 名 ↔ 本文叫法 ↔ 页内形状（LBNHC↔NHD↔token 在外、LBHNC↔HND↔头在外，其余四种存在但不展开）
- 前置知识安排：stride 概念正文内最小解释（scope 已定，不递归生成）

## 第 5 章 来源与范围说明（固定章节）

- 小节：论断与来源（C1–C20）、公式与来源（F1–F3）、外部数字与实验条件（N1–N4）、构造示例（16 槽位贯穿示例、as_strided 代码参数）、辅助解释与类比边界（低精度友好机制的推断标注）、简化条件及其限制（贯穿示例参数极小、vLLM main 滚动版本、只讲 FlashInfer 三维主轴）

## 讲解顺序与依赖

问题场景（第 1 章开头）→ 维度与排布（第 1 章）→ 写路径（第 2 章，依赖第 1 章排布）→ 读路径（第 3 章，与第 2 章对比：写读诉求相反）→ 工程 landing（第 4 章，依赖前三章的全部结论）。无循环依赖。

## overview.html 概览

1. 定位摘要：KV cache 布局 = 同一批 K/V 数在显存里的排列方式；NHD token 在外、HND 头在外
2. 问题背景：写路径和读路径对连续性的诉求相反，布局是把显存访问模式显式化的选择
3. 核心机制 3–5 点：三维含义与分页形状；NHD 与投影输出一致（写自然）；HND 单头整页连续（读大块，256 B 对 8 KB）；低精度 kernel 与页级搬运受益；vLLM 用逻辑形状+stride 置换在同一块显存上支持两者，SM100 解析为 head-major
4. 关键结论与边界：fp16 下无显著差异、默认 NHD；布局切换不复制数据；vLLM 细节基于 2026-09 main 分支
