# 冲突记录

- 角频率 θ_i 的式子｜论文 L283 写 $\theta_i=10000^{-2(i-1)/d},\ i\in[1,\dots,d/2]$；L305 与 L504 写 $\theta_i=10000^{-2i/d}$（索引自 0 起）｜页面取 L283 的 $10000^{-2(i-1)/d}$（索引自 $i=1$ 起，与 $i\in[1,\dots,d/2]$ 自洽），正文、摘要、例子统一用它。
- RoPE 是绝对还是相对位置编码｜同一官方仓库的中英 README 说法相反：repo/roformer/README.md L5 写 “The RoPE is a relative position encoding method”；repo/roformer/README_zh.md L5 写 “RoPE……是目前唯一一种可以应用到线性Attention的绝对位置编码”。论文 L110 写 “RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency”｜页面按论文取「相对式」（位置只以相对位置进入注意力分数，论文 L292 亦称其为 relative position encoding）。
