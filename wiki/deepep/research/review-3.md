# DeepEP 审查记录（第 3 轮）

- 页面版本：index.html `d985dcb143f6d5207c164a34c7bc36d86f7e3bd5`（修复前）
- 审查时间：2026-09-03
- 审查者：独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：overview.html 全文；index.html 引言、blockquote.meta、核心问题（5 题含解答）、第 1-5 章全部小节与本章问题（含全部折叠块、表格与图）、来源与范围说明全部小节

## 问题

- [重要·技术] index 4.1（并影响 4.4 对比表「安装期编译」行）｜"集群上本来就有 NCCL，不必再引入一套独立的通信栈"与来源矛盾：V2 仓库仍要求安装 NVSHMEM 以支持 legacy 方法｜引文依据：README "Install NVSHMEM dependency — DeepEP also depends on NVSHMEM to provide support for legacy methods."｜修复要求：删除或改写，明确"EPv2 新路径不经 NVSHMEM，但 V2 安装仍依赖 NVSHMEM 以支持 legacy 方法"｜修复：4.1 改为"EPv2 的通信路径建立在集群通常已部署的 NCCL 之上。注意这不是彻底移除 NVSHMEM：V2 安装仍依赖 NVSHMEM 以支持 legacy 方法（V1 路径）[C18]"；4.4 表格 V2 安装期编译单元格补"（仍依赖 NVSHMEM 支持 legacy 方法）"；来源章节 C18 条目补入 README 安装节依据；evidence C18 同步更新｜复验：通过（validate ok）
- [重要·技术] index 3.3 代码块注释与来源章节"构造示例"小节｜"无 token 路由到本 rank 的专家"与路由表数据不符：ROUTES[(2,1)] = [(6, 0.5), (2, 0.5)]，rank 2 的第 2 个 token 路由到本 rank 的 $E_2$｜引文依据：代码注释与来源章节声明 vs 实际 ROUTES[(2,1)] 目标含 E2｜修复要求：修改该条路由（保持每专家恰好 4 个 token 且不改 $E_5$ 行与预期输出）或删除声明；改后重新运行核对｜修复：ROUTES[(2,1)] 改为 (6, 0.5), (0, 0.5)、ROUTES[(7,1)] 改为 (2, 0.5), (6, 0.5)（交换 E0/E2 在 rank 2 与 rank 7 间的分配，各专家仍恰 4 个 token，全表无自路由；rank 0 路由与 $E_5$ 行不变）。修复中另发现并更正一处更深的简化偏差：原模拟用"token 序号 j"作段内槽位下标，而 V1 源码（internode_ll.cu 槽位偏移）的段内下标是"该来源 rank 发往此专家的到达序号"（发送 rank 本地 workspace 的按专家原子计数）——rank 4 的 $t_1$ 应落槽 8 而非槽 9。代码改为 segment_base + 段内计数实现，槽位表、观察重点（槽 0、4、8、12）、3.3 正文与第 3 章本章问题 3 解答的槽位规则表述、简化条件（补"构造中每 rank 至多一个 token 发往同一专家，段内序号均为 0"）全部同步；源码偏移摘录补入 deepep-src-extracts.md 第 7 节，来源章节 C13 与 evidence C13 补依据｜复验：通过（页面代码提取运行 exit 0、输出与预期逐字符一致；validate ok；渲染探针 138 个 KaTeX 零错误、SVG 无重叠）
- [轻微·技术] index 3.4｜decode 侧两个 micro-batch 交错表述为既定做法，来源为探索中｜引文依据：§3.4.2 "we are **also exploring** processing two micro-batches… in the decoding stage"｜修复要求：加限定｜修复：3.4 改为"prefill 侧同时处理两个 micro-batch…[C27]；decode 侧同样在探索这一交错（论文表述为 also exploring）——让一个 micro-batch 的 attention 与另一个的 dispatch+MoE+combine 重叠[C28]"｜复验：通过
- [轻微·技术] index 3.3（来源章节 C13 条目）｜"每个来源 rank 固定占 num_max 个槽、槽位地址只由来源决定"的依据未标注在 C13 所列来源位置（docstring 只给形状）｜引文依据：实际依据在 ebfe47e:csrc/kernels/internode_ll.cu 行 139-141 的偏移计算及行 231-232（combine 侧同构）｜修复要求：来源章节 C13 补 internode_ll.cu 偏移计算为依据｜修复：已补（含快照第 7 节摘录）；正文槽位规则同步改写为与源码一致的"段基址 + 段内到达序号"表述（见重要问题 2 的修复）｜复验：通过
- [轻微·技术] index 1.1｜"共享专家留在本 rank 计算、不参与通信"标注 [F1]，但式（12）-（14）不含该论断｜引文依据：F1 摘录无共享专家通信语义｜修复要求：补可定位引用或改写｜修复：该句移出公式引句，改为符号列表后的独立句"式中共享专家项不含门控值：共享专家对每个 token 都生效，留在本 rank 计算不参与通信……（共享专家的设计见 DeepSeekMoE 结构页链接）"，公式句的 [F1] 只覆盖公式本身｜复验：通过
- [轻微·格式] index 3.5 本章问题 5 解答｜"槽位总数 = 本地专家数 × 每 rank 最大 token 数 × rank 数 × $h$"中 3 处 Unicode × 出现在数学等式里｜引文依据：不适用｜修复要求：× 改为 $\times$ 或整式入 $...$｜修复：改为"槽位总数为 $\text{本地专家数}\times\text{每 rank 最大 token 数}\times\text{rank 数}\times h$"｜复验：通过（validate ok，无裸数学字符）
- [轻微·可读性] index 3.3 槽位表｜"槽 4 收到 $t_0$"等用 $t_0$ 指代 rank 2、rank 6 的 token，与贯穿定义的 $t_0$（rank 0 的 token）记号冲突｜引文依据：不适用｜修复要求：表格改用 $t_{r,j}$ 记号或加说明｜修复：表格改用 $t_{0,0}$、$t_{2,0}$、$t_{4,1}$、$t_{6,0}$，表前补"表中 $t_{r,j}$ 表示来源 rank $r$ 的第 $j$ 个 token"｜复验：通过

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 5
- 处置：7 条全部修复闭环。修复后终验：validate.py 两页通过；页面代码提取运行输出与预期逐字符一致；C/F/N 双向引用差集为空；headless Chrome 探针（138 个 KaTeX 公式零错误、正文无残留 $、SVG 公式标签 9 个全渲染、标签两两无重叠、29 个折叠块）。

## 发布评估（编排者按 check.md 第 5 节）

- 三轮审查均由未参与写作的独立子代理执行 ✓（每轮首条消息只含页面路径、来源获取方式、规范路径与指令）
- 每条来源论断都有引文依据记录；无法核对的论断已删除或降级 ✓（三轮共 1 重要 + 2 重要已修复，DualPipe"完全隐藏"补引文、NVSHMEM 依赖更正、槽位规则对齐源码）
- 所有阻断和重要问题均已关闭 ✓（三轮累计：阻断 0、重要 3、轻微 20；重要 3 条全部修复，轻微 18 条修复、2 条记录接受理由——章节引用风格（第 2 轮已实际修复）、误解分散放置）
- 遗留轻微问题具有明确的接受理由 ✓（误解分散放置系 outline 指定；无其他遗留）
- 全部学习目标由正文章节完整回答 ✓
- 页面级「核心问题」5 条与章节级「本章问题」20 条均有解答折叠块，答案独立可读 ✓
- 数学符号全部 LaTeX、结构图 HTML/内联 SVG、SVG 公式在 foreignObject 且实测渲染无重叠 ✓
- validate.py 成功 ✓；可运行代码结果与页面描述逐字符一致 ✓；关键论断和数字经三轮独立核对 ✓
- head 元数据齐全，dojo:topics 在固定大类内（并行与通信、推理系统）✓
- overview.html 与 index.html 相互链接 ✓；概念链接全部有效（8 个前置/相邻页面存在）✓
- 递归前置概念页均为已有页面，无需递归生成 ✓

**发布结论：可发布。**（首页目录与关系图由 GitHub Pages 构建自动扫描生成；未获授权不执行 commit/push。）
