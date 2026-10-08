# 资料冲突

- 门控公式｜paper/src-v2/sections/arch.tex Context-aware Gating 写 $\alpha_t=\sigma(\text{RMSNorm}(\mathbf{h}_t)^\top\text{RMSNorm}(\mathbf{k}_t)/\sqrt{d})$；repo/engram/engram_demo_v1.py L371-L373 在 sigmoid 之前多做一步 $\mathrm{sign}(x)\sqrt{|x|}$｜页面按论文公式写
