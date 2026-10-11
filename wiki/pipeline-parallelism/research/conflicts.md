# 资料冲突

- 「两条基本路径＝张量并行/流水线并行」的概括｜megatron-src-v1/intro.tex「intra-layer model-parallelism」与「orthogonal to pipeline-based model parallelism」把两者并列，background.tex L30 又把并行分为「layer-wise pipeline parallelism」与「more general distributed tensor computation」；gpipe-src-v1/main.tex L354 只说「partitioning a network into different computational units」、未限定为两条｜页面按两条基本路径（张量并行、流水线并行）写，并在长页点明流水线并行属资料所称的一路，与张量并行并列。
- AmoebaNet 参数量｜gpipe-src-v1/main.tex L114 图注作「550M parameter AmoebaNet model」，L130、L267 作「557(-million)」｜页面不引用该数，不涉及。
- AmoebaNet 扩到 1.8B 的倍数｜gpipe-src-v1/main.tex L218 正文作「25x more」，但同文件表 L196 的数据 82M→1.8B 算出约 22 倍｜页面按表算写「约 22 倍（82M 到 1.8B）」。
- 82M/318M/6.26GB/3.46GB 一组的硬件标签｜gpipe-src-v1/main.tex L193 表头作「NVIDIA GPUs (8GB each)」，L217 正文作「Cloud TPUv2s with 8GB memory per accelerator」｜页面只写数字，不标明硬件。
- 线性扩展的适用范围｜gpipe-src-v1/main.tex L363 作无条件的一般陈述「enables the user to scale model size linearly with the number of accelerators used」，但同文 L218、L242 说 AmoebaNet 只得到次线性扩展（参数在各层分布不均），L221 又把线性扩展限定为 Transformer｜页面不引用「线性扩展」这一说法，不涉及。
