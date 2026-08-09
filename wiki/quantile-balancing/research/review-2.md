# Quantile Balancing 独立审查（第二次）

- 审查者：独立上下文（AI 模拟小白读者 + 段B对照来源）
- 页面版本：index.html 688a06c9d47bbf3883892923747eb90022d57c28（commit ac5b744）
- 时间：2026-08-09 17:15 +0800
- 来源：/tmp/kimi-k3-research/k3-report.txt §2.3.3（Eq.13-14）、Appendix C（Eq.20-27 + Algorithm 1）、Appendix D；DeepSeek-V3 Technical Report arXiv:2412.19437 §3.3；HuggingFace moonshotai/Kimi-K3 config.json

## 问题

- [轻微·盲读] §"训练时的直方图估计与推理冻结"引入 $r_{i,j}=\alpha_i-s_{i,j}$（与 §QB 核心机制中 margin $=s_{i,j}-\alpha_i$ 符号相反），随后分位数由 $(1-k/n)$ 改取 $k/n$；页面仅以一句"注意 $r=\alpha-s=-\text{margin}$，所以取 $k/n$ 分位数"提示，未写出推导。源 §D 行 2907-2908 明确"the QB target $\tilde b_j$ of Eq.14 is exactly the $(k/n)$-quantile of $r_{:,j}$"，结论正确，但读者需自行由 $\text{quantile}_p(-r)=-\text{quantile}_{1-p}(r)$ 推出 $b̃=\text{quantile}_{k/n}(r)$，符号翻转较隐蔽。：在 S6 直方图机制处补一行推导 $\tilde b_j=-\text{quantile}_{1-k/n}(\text{margin})=-\text{quantile}_{1-k/n}(-r)=\text{quantile}_{k/n}(r)$，使符号翻转可验证。 ｜ 修复： ｜ 复验：
- [轻微·技术] §"手算 m=8,n=4,k=1 的 QB 例子"标"以下分数为人为构造"，但其设定 $m=8,n=4,k=1$ 与初始负载 $(4,3,1,0)$→QB 后 $(2,2,2,2)$ 来自源 Figure 5（行 530-538）的同名图示例子（源 Fig.5 caption 即"m=8 tokens, n=4 routed experts, k=1...loads (4,3,1,0)...(2,2,2,2)"），并非完全自创；具体分数矩阵为本页构造使其可手算。页面未注明该设定与源 Fig.5 的对应。：在 S4 手算例子开头或"来源与教学说明·教学示例"注明"例子设定对应源 Fig.5（m=8,n=4,k=1，负载 (4,3,1,0)→(2,2,2,2)），具体分数为本页构造使其可手算"，使出处完整。 ｜ 修复： ｜ 复验：
- [轻微·盲读] §"DeepSeek-V3 的固定步长更新"教学示例"负载 100、目标 50、γ=0.001...前者需要约 50000 步才能把 bias 调到合理水平"：50000 步的得出隐含"bias 需移动约 50（=100-50）且 1 单位 bias 改变对应 1 个 token 负载"的假设，但页面未说明 bias 变化量与负载变化量的关系，读者无法从给定信息推出 50000。该例已标"教学示例"，定性结论（sign 丢失幅度信息）正确且源 §C 行 2885-2889 支持。：在 50000 步后补一句推导，如"若需 bias 移动约 50 单位（量级估计），则 50/0.001≈50000 步"，或改为不给出具体步数、只说"量级上需要数万步"。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 段A盲读核对学习目标：5 条学习目标（QB 解决什么/sign 为何 896 失效、Eq.14 每步做什么、手算 m=8/n=4/k=1 由 (4,3,1,0) 变 (2,2,2,2)、QB vs sign 本质区别与无需超参、训练时直方图估计与推理冻结）均由正文（§aux-loss-free 路由、§DeepSeek-V3 sign 更新、§QB 核心机制、§手算例子、§对偶理论、§直方图估计与推理冻结）完整回答，无悬空。
- 段B对照来源结论：Eq.13 路由（源行 552）与 Eq.14 QB 更新（行 586-589）逐项一致；sign 更新 $b_j\leftarrow b_j+\gamma\,\text{sign}(\bar\ell-\ell_j)$（源行 557）一致；DeepSeek-V3 引用 arXiv:2412.19437 真实有效；对偶目标 Eq.23（§C 行 2823）、coordinate minimizer Eq.25-26（§C 行 2856/2863）、sign subgradient Eq.27（§C 行 2882）、Algorithm 1（§C 行 2832-2840）均与源一致；"两次更新同一分位数→得名 Quantile Balancing"（§C 行 2866）、"SignSGD 保留方向、QB 跳到精确 coordinate minimizer、无需学习率类超参、近 10³ 专家几步内收敛"（§C 行 2885-2889）逐条核实通过；直方图分箱范围 $[b_{\min}-1,b_{\max}+1]$、bin width $w=(b_{\max}-b_{\min}+2)/B$、恢复公式 $\tilde b_j=b_{\min}-1+(\beta_j+\text{clip}((q-c_j)/h_j,0,1))w$（§D 行 2913-2928）、$B=1000$ 误差"几个 10⁻³"（§D 行 2932）、通信"$nB$ 整数/layer/step、低于 1%"（§D 行 2933-2934）、EMA 平滑（§D 行 2938）、推理冻结（§C 行 2874）均与源一致；config `num_experts:896`/`num_experts_per_token:16`/`moe_router_activation_func:"sigmoid"`/`topk_method:"noaux_tc"` 核实通过；K2 路由专家 384（Table 1 行 748）核实通过。
- 可运行代码：S4 折叠块 Python 代码已实际执行（python3），输出与页面"预期输出"逐字一致——初始 loads=[4,3,1,0]、b_tilde=[-0.2,-0.1,-0.0,0.1]、mean=-0.05、b_new=[-0.15,-0.05,0.05,0.15]、QB 后 loads=[2,2,2,2]、改变 token T4(E1→E3)/T5(E1→E4)/T8(E2→E4)。手算 margins、3rd-largest 分位数、mean-centering、新路由逐项复算正确。
- 处置：进入修复（3 轻微，均非阻断、不涉及核心结论失真）。三处均为可读性/出处表述问题，修复后可发布；若选择保留为轻微遗留，需按 check.md §4 逐项写明接受理由。
