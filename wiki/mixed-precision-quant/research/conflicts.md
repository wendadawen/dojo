# 冲突记录

- repo/Hy4-preview-STQ1_0.tensortypes L109-L111：注释写「all remaining routed gate/up drop to IQ1_S」，紧接的规则行写 `ffn_gate_exps=stq1_0` / `ffn_up_exps=stq1_0`；blog/hy4-gguf-model-card.md L31 也写 29 层用 STQ1_0（1.3125 bpw）。两处指同一批 29 层。页面按规则行与模型卡取 STQ1_0。
- blog/hy4-gguf-model-card.md：同一张卡里 STQ1_0 的整模 bpw 有两种说法。中文第 1 节 L147 写「约 2.44 bpw 的混合精度，使用 UD-IQ1_M 混合精度量化压缩」（把 STQ1_0 挂到 UD-IQ1_M 名下），英文第 1 节 L30 与文件表 L12 写 2.38 bpw、MIX-STQ1_0。页面按文件表与英文节取 2.38 bpw。
