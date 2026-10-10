# 资料冲突

- 「两条基本路径＝张量并行/流水线并行」的概括｜megatron-src-v1/intro.tex「intra-layer model-parallelism」与「orthogonal to pipeline-based model parallelism」把两者并列，background.tex L30 又把并行分为「layer-wise pipeline parallelism」与「more general distributed tensor computation」；gpipe-src-v1/main.tex L354 只说「partitioning a network into different computational units」、未限定为两条｜页面按两条基本路径（张量并行、流水线并行）写，并在长页点明张量并行属资料所称的 SPMD 一路，与流水线并行并列
