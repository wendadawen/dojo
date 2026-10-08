# 资料冲突

- 门控公式｜paper/src-v2/sections/arch.tex Context-aware Gating 写 $\alpha_t=\sigma(\text{RMSNorm}(\mathbf{h}_t)^\top\text{RMSNorm}(\mathbf{k}_t)/\sqrt{d})$；repo/engram/engram_demo_v1.py L371-L373 在 sigmoid 之前多做一步 $\mathrm{sign}(x)\sqrt{|x|}$｜页面按论文公式写
- MMLU 提升｜paper/src-v2/sections/intro.tex L12 与摘要写 +3.4；paper/src-v2/sections/large_scale_exp.tex L104 与表（57.4 → 60.4）写 +3.0｜页面不写这个数字；如需引用，按正文和表格取 +3.0
