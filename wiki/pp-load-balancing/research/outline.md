# PP 负载均衡

- 描述：LLM 推理用流水线并行（PP）部署时，把各 stage 每一步的执行时间拉齐、压小 GPU 空闲气泡的 token 切分与 batch 配平做法。
- 摘要：流水线的节拍由最慢的 stage 决定。沿 token 维度把一条长请求切成片段送进流水线，或者在 batch 维度把并发请求的计算量在各 micro-batch 间配平，都能让各 stage 等长，把气泡压下去。
- 主题：并行与通信, 推理系统
- 标签：并行与通信

## 长页

- 开头｜长请求进 PP 流水线时的空等与计算量波动；本页目标与 token、batch 两条路线；交代 stage、micro-batch、气泡，PP 链接前置页｜550｜blog/sglang-pp-blog.txt L111-L117；paper/src-2504.14775v2/chapters/2.background_motivation.tex L81-L85
- ### GPU 为什么空转｜两类依赖与两个放大器；气泡比率公式及其两项，推导折叠｜550｜paper/src-2504.14775v2/chapters/2.background_motivation.tex L81-L85；paper/src-2504.14775v2/chapters/2.background_motivation.tex L106-L113；blog/sglang-pp-blog.txt L111-L117；blog/sglang-pp-blog.txt L76-L88；paper/src-2102.07988v2/terapipe.tex L305-L305
- ### 把一条请求切进流水线（token 维度）｜链接自回归前置页；回指放大器二说明均分为何前轻后重、最优切分前长后短；目标函数两项；贯穿示例手算折叠｜800｜paper/src-2102.07988v2/terapipe.tex L282-L285；paper/src-2102.07988v2/terapipe.tex L305-L305；paper/src-2102.07988v2/terapipe.tex L325-L339
- ### 切片怎么求、推理里怎么落地｜链接 prefill 与 chunk 前置页；TeraPipe 的 DP 递推与拟合折叠；Mooncake 用 CPP 做推理 prefill 的机制｜750｜paper/src-2102.07988v2/terapipe.tex L342-L380；paper/src-2407.00079/data/prefill.tex Multi-node Prefill
- ### 固定 chunk 为什么错配、动态 chunk 怎么对齐｜一句回指 token 节的注意力增量；错配沿流水线传播在靠后的 stage 复合放大；二次模型方程定下一个 chunk 大小；漂移手算折叠｜750｜blog/sglang-pp-blog.txt L139-L160
- ### 把并发请求配平（batch 维度）｜链接 decode/KV 前置页；token budget 的波动来源；gLLM 解耦后 decode 均摊、prefill 随积压与 KV 空闲率调；手算折叠｜750｜paper/src-2504.14775v2/chapters/1.introduction.tex L26-L32；paper/src-2504.14775v2/chapters/4.design.tex L18-L64；paper/src-2504.14775v2/chapters/4.design.tex L90-L96
- ### 两条路线怎么配合｜两维各解决哪类失衡、能否同时用；gLLM 只处理 inter-batch，inter-stage 留后续｜450｜paper/src-2504.14775v2/chapters/2.background_motivation.tex L81-L85；paper/src-2407.00079/data/prefill.tex Multi-node Prefill；blog/sglang-pp-blog.txt L186-L198
- ### 适用条件与边界｜成立条件（单向自回归、chunk 下限、短上下文收益小）、代价（跨机通信、调参）、两条常见误解｜400｜paper/src-2102.07988v2/terapipe.tex L282-L283；paper/src-2407.00079/data/design_and_implementation.tex L32-L32；blog/sglang-pp-blog.txt L113；blog/sglang-pp-blog.txt L88；blog/sglang-pp-blog.txt L160-L166；paper/src-2504.14775v2/chapters/2.background_motivation.tex L85-L85

## 短页

- ### 把各 stage 每步执行时间拉齐｜定义：PP 部署时让各 stage 每步时间相近的切分与调度；交代 PP、stage、气泡，PP 链接前置页｜230｜blog/sglang-pp-blog.txt L111-L117；paper/src-2504.14775v2/chapters/2.background_motivation.tex L81-L85
- ### 节拍由最慢的 stage 决定｜代价：一次迭代由最慢 stage 定；长请求整体进入、注意力随前缀变重两个放大源，注意力一句解释并链接前置页｜330｜paper/src-2504.14775v2/chapters/2.background_motivation.tex L106-L113；paper/src-2102.07988v2/terapipe.tex L305-L305；blog/sglang-pp-blog.txt L111-L117；blog/sglang-pp-blog.txt L139-L141
- ### 按执行时间对齐，不是按条数｜均衡对象是各 stage 执行时间；token 维度切前长后短、batch 维度按 token 数均摊 decode，各一句并链接前置页｜380｜paper/src-2102.07988v2/terapipe.tex L305-L305；blog/sglang-pp-blog.txt L139-L154；paper/src-2504.14775v2/chapters/4.design.tex L90-L96
