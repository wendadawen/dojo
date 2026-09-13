<!-- review-meta
round: 3
page: wiki/gpu-communication/index.html
reviewed_content_sha256: ce45b66800de7bca
-->
# GPU 通信审查记录（第 3 轮）

- 页面版本：wiki/gpu-communication/index.html，内容 sha256 8a0e02cfff980c0b20b2e2ff004b2610e5a3284b8db30e3c4acc9335b3baf1e4
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查与修复）
- 适用规范：页面 head 的 dojo:type = note → guides/note.md；记录格式取自 guides/concept/check.md 第 3 节
- 已完整阅读章节（按顺序，含全部图注；页面无折叠块）：分层与主线：谁调度、谁搬运｜通信原语：搬、算、换三类动作（含「八个原语的完整定义」）｜环形 all-reduce：带宽最优的两阶段算法（含「通信量：为什么环形算法带宽最优」「其他算法与选择」）｜机内互联：总线、直连与中心交换｜跨机传输：CPU 参与与绕过 CPU 两条路径｜网卡：一个角色，多个名字与用法｜通信库：集合通信库与传输库的分工｜来源
- 机械验证：`.dojo/scripts/validate.py wiki/gpu-communication/index.html` 返回 `validation ok`；内链 ../deepep/index.html、../model-parallelism/index.html 均存在；全文无指向 research/ 的路径，无「（待生成）」占位

## 来源回源核对（逐条写出所核对的原文片段/数值）

| 论断 | 来源 | 核对结果 |
|---|---|---|
| ring all-reduce 两阶段（reduce-scatter + all-gather）、每阶段 N-1 步、每卡发送量 2(N-1)K/N | andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/ | 一致。原文 "Data Transferred = 2(N−1)K/N"，两阶段各 N-1 次迭代，与页面公式、步数表述相同 |
| 4/8/128 卡每卡发送 1.5 / 1.75 / 1.98 倍；朴素方案 N-1 倍 | 公式复算 | 2(3)/4=1.5、2(7)/8=1.75、2(127)/128≈1.984，与公式一致，且 4 卡示例的「两阶段各 3 步、每步 1/4，共 1.5MB」自洽 |
| GPUDirect RDMA 让网卡直读 GPU 显存、绕过 CPU、不落主机内存 | docs.nvidia.com/cuda/gpudirect-rdma/ | 一致。原文 "without needing to copy data to host memory"，并述 GPU 与对端设备间的 direct path、CPU bypass |
| gloo=CPU 训练、NCCL=GPU 训练 的后端选择规则 | pytorch.org/docs/stable/distributed.html | 一致（官方规则：「Use the NCCL backend … with CUDA GPU」「Use the Gloo backend … with CPU」） |
| NIXL：Xfer=Transfer、UCX 为默认后端、另支持 LIBFABRIC/GDS/UCCL | github.com/ai-dynamo/nixl（README + docs/BackendGuide.md） | 后端清单一致（UCX、GDS、LIBFABRIC、UCCL 均出现，UCX 为通用网络插件）；「默认」一词见问题 4 |
| TCCL：在开源 NCCL 基础上扩展、API 与 NCCL 完全兼容、重编 PyTorch 或装插件接入、常用模式性能提升约 40% | cloud.tencent.com/document/product/1573/93190 与星脉网络官方披露 | 一致。原文 "API 与 NCCL 完全兼容"；接入方式列有三条（重编 PyTorch / 装通信插件 / 装 NCCL 插件），与页面相符；40% 出于星脉网络披露「AllReduce、AllGather、ReduceScatter 等常用通信模式下约 40% 通信性能提升」，页面归因正确。腾讯云同页另有「约 50% 带宽利用率」表述，口径不同，页面未采用，不构成矛盾 |
| PCIe 与 NVLink 带宽量级「数倍」 | NVIDIA 公开规格 | 页面已注明「数量级对比，具体数值随代际变化」，属已标注的近似，可接受 |

其余论断（原语三分类与 8 原语前后状态表、三层架构、PCIe/NVLink/NVSwitch 分工、socket/TCP/IP/以太网分段、IB 信用流控无损 vs RoCE 需 PFC/ECN、MAC 48 位、bond 聚合与冗余、NIC 存在的两点理由）逐项复算与常识核对，未发现数字错误或内部矛盾；8 卡两两直连 C(8,2)=28 条正确，且已明确标为「拓扑示意」。

## 问题

- [轻微·可读性] 第 142 行：「reduce_scatter 是归约与切分的组合，先算后切——它的两个阶段分别对应 all-reduce 的前半段。」｜引文依据：不适用｜问题：reduce_scatter 是单一集合操作，等价于 all-reduce 的第一阶段；本页 147–150 行也把 all-reduce 定义为「reduce-scatter + all-gather」。「它的两个阶段分别对应 all-reduce 的前半段」把两个对象对应到一个对象，语法与逻辑均不成立，会被读成 reduce_scatter 自身含两个阶段。｜修复要求：改写为单义陈述，如「reduce_scatter 就是 all-reduce 的第一阶段（reduce-scatter）」｜修复：｜复验：
- [轻微·来源] 第 231 行：「HCA 里的 Adapter 指「转接两种规格」：主机内部总线与 IB 网络」｜引文依据：来源栏未收录 HCA 词源；IBTA/NVIDIA 材料只给出 HCA = Host Channel Adapter 这一名称，无「Adapter 指转接两种规格」之说｜问题：对缩写语义的解释性推断被写成事实，无来源支撑。｜修复要求：删除该半句，或按 note 规范标注为推断｜修复：｜复验：
- [轻微·来源] 第 141 行：「换……是唯一能实现「维度互换」的原语」｜引文依据：来源栏未覆盖该判断｜问题：「唯一」是无条件论断，无来源支撑。｜修复要求：删去「唯一」，或补来源／降级为推断｜修复：｜复验：
- [轻微·来源] 第 252 行与第 262 行：「UCX 是默认传输后端」｜引文依据：github.com/ai-dynamo/nixl 的 README 与 docs/BackendGuide.md 均未出现 "default" 字样（原文只将 UCX 描述为通用网络插件、"primary transport for AMD GPU memory"）；「默认」明确写于 NVIDIA Dynamo / TensorRT-LLM 文档（NIXL with UCX, default）｜问题：来源栏所引页面本身不含该表述，引用位置与论断不匹配。｜修复要求：来源栏改引 NVIDIA Dynamo 或 TensorRT-LLM 文档｜修复：｜复验：
- [轻微·格式] 第 32–33 行：`@media (prefers-color-scheme: dark) {  }` 为空规则，`/* ---------- 页面自定义 ---------- */` 后另有一行残留缩进空行｜引文依据：不适用｜问题：CSS 抽取残留，无功能作用，降低维护性；同位置在 .dojo/templates/note/index.html 中为非空规则。｜修复要求：删除该空规则与残留空行｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：可发布。核心结论（三分类原语、环形 all-reduce 两阶段与 2(N-1)/N 通信量、CPU 参与与否的跨机分界、PCIe/NVLink/NVSwitch 分工、IB 与 RoCE 取舍、GPUDirect、网卡三层面、通信库与传输库分工）逐条回源核对一致，无数字错误、无内部矛盾、无实验条件被写成无条件结论；5 项轻微问题留待接受或随手修正，均不影响正确性与主线理解。