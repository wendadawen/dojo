# 资料矛盾记录

- decode 阶段的最小部署单元：`blog/deepseek-v3-r1-inference-system-overview.md` L23 写「Decoding Phase [Routed Expert EP144, MLA/Shared Expert DP144]…Each deployment unit spans 18 nodes…each GPU manages 2 routed experts」；`paper/src-dsv3/main.tex` L615-L617 写「The minimum deployment unit of the decoding stage consists of 40 nodes with 320 GPUs…EP320…each GPU hosts only one expert」。两处对 decode 阶段 EP 规模口径不同。页面不引用这组数字，两条都不采用。
- prefill 阶段冗余专家：`blog/deepseek-v3-r1-inference-system-overview.md` L22 写「each GPU handles 9 routed experts and 1 shared expert」；`paper/src-dsv3/main.tex` L603 写「besides the original 8 experts it hosts, it will also host one additional redundant expert」（即 9 个）。两处一致，页面未引用该数字。
