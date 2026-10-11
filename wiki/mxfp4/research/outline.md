# MXFP4

- 描述：MXFP4 是一种 4-bit 微缩浮点数据格式，把连续若干个权重分为一块、整块共享一个 8-bit 幂次尺度，每个数存成 4-bit 浮点元素。
- 摘要：权重从 16 位压到 4 位能省显存，但低位宽元素能覆盖的数值范围太窄。MXFP4 让每 $32$ 个数共享一个 8-bit 幂次尺度、每个数存 4-bit 浮点元素 E2M1，块内各数只需表示相对大小。
- 主题：内存与缓存, 训练与优化
- 标签：量化

## 长页

- 开头｜先说大模型权重占显存、把每个权重从 16 位压到 4 位能省显存，再说 MXFP4 是什么（块、共享尺度、元素）、为什么需要它｜300｜paper/rouhani-2023-mx.txt L20-L30；paper/rouhani-2023-mx.txt L42-L48；standard/ocp-mx-spec-v1.0.txt L266-L296；repo/kimi-k3-hf/config.json L11
- ### E2M1 元素：块内一个数怎么存｜说明一个数的 4-bit 浮点元素 E2M1 编码、能表示哪些值，按最近值舍入、超范围截断、低于最小次正规归零；数值表折叠｜300｜standard/ocp-mx-spec-v1.0.txt L423-L452；standard/ocp-mx-spec-v1.0.txt L443-L452
- ### 共享尺度与反量化｜说明 $k$ 个数共用一个 E8M0 尺度、为什么由它承担动态范围，反量化把一个数的值写成元素值乘共享尺度，以及每权重分摊的位宽；位宽算例折叠｜300｜standard/ocp-mx-spec-v1.0.txt L266-L296；standard/ocp-mx-spec-v1.0.txt L492-L506；standard/ocp-mx-spec-v1.0.txt L289-L294；paper/rouhani-2023-mx.txt L42-L48
- ### 块尺度怎么定｜说明块的尺度取成块内最大幅值向下的那个 2 的幂、再除以元素格式能表示的最大 2 的幂，按块共享为什么优于整份权重只用一个尺度；换算步骤折叠｜300｜standard/ocp-mx-spec-v1.0.txt L548-L570；paper/rouhani-2023-mx.txt L136-L160；paper/rouhani-2023-mx.txt L42-L48
- ### 量化为什么让权重失真｜仅讲量化带来的偏差：权重换成 E2M1 能表示的值后与原来不同，差值取决于离最近档位的远近（量化作为前置概念链接）｜260｜standard/ocp-mx-spec-v1.0.txt L423-L452
- ### 适用条件与边界｜块大小固定为 32（每 32 个数共享一个尺度）；在训练中把量化算进去让模型适应这一损失（见 QAT），不把精度补回量化前｜260｜standard/ocp-mx-spec-v1.0.txt L306-L324；paper/k3-tech-report.txt L922-L928；paper/rouhani-2023-mx.txt L398-L410

## 短页

- ### 一种 4-bit 浮点格式｜是什么：模型的权重每连续 $k$ 个数分一块，整块共享一个 8-bit 尺度，每个数存 4-bit 浮点元素（值最多到 6）；读时乘以尺度｜300｜standard/ocp-mx-spec-v1.0.txt L306-L324；standard/ocp-mx-spec-v1.0.txt L443-L452；paper/rouhani-2023-mx.txt L77-L100
- ### 位宽压下去，误差从哪里来｜为什么需要：权重从 16 位压到 4 位能省显存；4 位元素存进去的值与原来不同即生误差；块尺度承担块内差异、元素只表相对大小｜300｜repo/kimi-k3-hf/config.json L11；paper/k3-tech-report.txt L922-L928；paper/rouhani-2023-mx.txt L42-L48；paper/rouhani-2023-mx.txt L144-L153；standard/ocp-mx-spec-v1.0.txt L556-L558
- ### 定尺度、编码｜怎么做：块的尺度取块内最大幅值向下的 2 的幂除以元素格式的最大 2 的幂；每个数除以尺度后按最近值取元素编码｜300｜standard/ocp-mx-spec-v1.0.txt L548-L570；paper/rouhani-2023-mx.txt L136-L160
