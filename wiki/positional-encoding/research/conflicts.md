# 位置编码 冲突记录

- 正弦式位置编码的排布与频率口径｜论文正文写 $PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{\text{model}}})$、$PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{\text{model}}})$，按维度下标奇偶交错；官方实现 repo/tensor2tensor/tensor2tensor/layers/common_attention.py L451-L455 把前 `channels/2` 个通道全放 sin、后一半全放 cos，并自注「slightly differs from the published paper」，频率用 min_timescale=1 到 max_timescale=1e4 的等比 timescale｜页面采用论文正文的公式，正弦式一节只依据论文。
- 最长一维的波长上端｜论文正文 L153 写「The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$」；但同一节的公式里 $i$ 取 $0$ 到 $d_{\text{model}}/2-1$，最长一维（$i=d_{\text{model}}/2-1$）的波长是 $2\pi \cdot 10000^{(d_{\text{model}}-2)/d_{\text{model}}}$，不到 $10000 \cdot 2\pi$｜页面按公式写「最短的一维是 $2\pi$，最长的一维接近 $10000 \cdot 2\pi$」。
