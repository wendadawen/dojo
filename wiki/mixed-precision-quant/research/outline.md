# 逐层混合精度量化

- 描述：逐层混合精度量化给同一模型的不同层分别指定不同的量化档位，对量化误差敏感的层留高比特、不敏感的层压低比特。
- 摘要：各张量对量化误差的耐受度不同：校准数据统计出的重要性越高、或写回残差流的张量，档位留得越高；占体积主体的路由专家 gate/up 压到最低档。Hy4 的 MIX-STQ1_0 里路由专家 `gate`/`up` 在 $29$ 层用 $1.3125$ bpw、另 $48$ 层用 $2.0625$ bpw，整模平均 $2.38$ bpw。
- 主题：推理系统, 内存与缓存
- 标签：量化

## 长页

- 开头｜总览：是什么、为什么、大概怎么工作；解释量化、位宽、每权重比特、量化误差、前向、激活、校准、张量族、张量角色的最小含义；一句区分单一档位量化｜580｜blog/hy4-gguf-model-card.md L8-L12；blog/hy4-gguf-model-card.md L30-L32；blog/hy4-gguf-model-card.md L105-L116；blog/llama-cpp-imatrix-README.md L92-L102
- ### 各张量族取不同档位｜路由专家（按 token 选择的专家）的 gate/up/down 投影、attention、indexer 各取哪一档，后两者作链接；分配表与体积数字折叠｜800｜blog/hy4-gguf-model-card.md L105-L116；blog/hy4-gguf-model-card.md L8-L12
- ### 敏感度怎么量：校准数据与重要性分数｜用校准文本前向、按张量统计激活统计（其中一项是激活能量）得重要性分数（imatrix）；能量高的张量通常需要更高精度；统计口径折叠｜620｜blog/llama-cpp-imatrix-README.md L92-L102；blog/hy4-gguf-model-card.md L110
- ### 一个两层两档的例子｜自行构造：两层各两档，平均每权重比特相同时比较两种分配的总误差，说明分配方向的决定性；设定与逐步手算折叠｜580｜自行构造
- ### 配方文件的写法与两处易错点｜逐层分配落地成一份「张量名模式＝档位」的列表；规则按先后优先匹配，逐层规则排在通用规则前；两处命名匹配坑与最后三层再提档，片段折叠｜620｜repo/Hy4-preview-STQ1_0.tensortypes L121-L124；repo/Hy4-preview-STQ1_0.tensortypes L138-L150；repo/Hy4-preview-STQ1_0.tensortypes L167-L171；repo/Hy4-preview-STQ1_0.tensortypes L109-L111
- ### 适用条件与边界｜分配只决定哪一层用哪一档，不改单档编码本身；STQ1_0 的编码器强制需要 imatrix（用于 scale 求解与零位置选择）；各档需要支持它的运行时｜520｜blog/hy4-gguf-model-card.md L93-L95；blog/hy4-gguf-model-card.md L122-L123；blog/hy4-gguf-model-card.md L14-L15

## 短页

- ### 同一模型里，不同层存成不同比特宽度｜是什么：量化把权重存成更低位宽，每权重比特衡量占用；逐层混合精度给不同张量族或层定不同档位，平均每权重比特是加权结果；一句区分单一档位｜330｜blog/hy4-gguf-model-card.md L8-L12；blog/hy4-gguf-model-card.md L30-L32
- ### 各张量对误差的耐受度不一样｜为什么需要：写回残差流或功能关键的张量留高档，占体积主体且不敏感的压到最低档；残差流一句带过并链接｜330｜blog/hy4-gguf-model-card.md L105-L111
- ### 先量重要性，再按张量族分档｜怎么做：一句说明校准数据是什么，再用它算出重要性分数，据此给每个张量族和层指定档位｜320｜blog/hy4-gguf-model-card.md L105-L111；blog/hy4-gguf-model-card.md L122-L123
