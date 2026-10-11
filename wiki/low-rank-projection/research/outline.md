# 低秩分解

- 描述：低秩分解把一个大矩阵 $W$ 写成两个小矩阵 $A,B$ 的乘积 $W \approx AB$，内维 $r$ 远小于原矩阵的两维，用 $r(m+n)$ 个参数表达几乎相同的变换。
- 摘要：$W \approx AB$，把输入先投影到 $r$ 维再投影回原维度；$r$ 远小于原矩阵维度时，参数和存储降一个量级，损失的多少由被丢掉的奇异值决定。
- 主题：数学基础, 模型结构
- 标签：数学与数值

## 长页

- 开头｜概览：低秩分解是什么、为什么需要；末句指出它在大模型里的两处用法并链接到各自页面｜330｜blog/mml-book/mml-ch4-svd.txt L668-L706；paper/lora-2106.09685.txt L145-L147；paper/lora-2106.09685.txt L212-L213；paper/deepseek-v2-2405.04434.txt L328-L345
- ### 矩阵的秩｜秩是独立方向数；W=AB 说明秩不超过内维 r，满秩矩阵无法无损压缩；参数从 mn 降到 r(m+n) 的对比与数值代入折叠｜420｜blog/mml-book/mml-ch4-svd.txt L27-L28；blog/mml-book/mml-ch4-svd.txt L668-L698；blog/mml-book/mml-ch4-svd.txt L700-L706；blog/ee263/ee263-17_svd.txt L484-L535；paper/lora-2106.09685.txt L206-L207
- ### 最优低秩近似：截断 SVD｜本章只有标题，内容在两个小节｜0｜blog/mml-book/mml-ch4-svd.txt L628-L698
- #### 奇异值分解｜SVD 的形式与两个正交因子、奇异值（矩阵在正交主方向上的伸缩系数）按从大到小排序、秩等于非零奇异值个数、截断 SVD｜360｜blog/mml-book/mml-ch4-svd.txt L27-L100；blog/mml-book/mml-ch4-svd.txt L605-L698；blog/lapack-lug/svd.txt L9-L32
- #### 误差由被截掉的奇异值决定｜Eckart-Young 定理：截断 SVD 是最优秩-k 近似；谱范数（最大的奇异值）误差等于被截掉的第一个奇异值；最优性直觉折叠｜400｜blog/mml-book/mml-ch4-svd.txt L743-L812；blog/ee263/ee263-17_svd.txt L484-L539；blog/ee263/ee263-16_norm.txt L37-L53
- ### 一个手算的低秩近似｜3×3 对角矩阵的秩-1 近似与误差，与误差公式对照；全过程折叠｜60｜自行构造
- ### 适用条件与边界｜奇异值平坦时截断误差大；低秩近似是有损压缩；把低秩分解用在具体对象上要有收益，前提是该对象真的低秩，用 LoRA 举一个边界例子并链接｜320｜blog/mml-book/mml-ch4-svd.txt L808-L816；blog/mml-book/mml-ch4-svd.txt L754-L812；paper/lora-2106.09685.txt L206-L213；paper/lora-2106.09685.txt L600-L603

## 短页

- ### 用两个小矩阵代替一个大矩阵｜低秩分解是什么：W≈AB，r 远小于原矩阵维度，参数从 mn 降到 r(m+n)｜300｜paper/lora-2106.09685.txt L145-L147；blog/mml-book/mml-ch4-svd.txt L668-L698；paper/lora-2106.09685.txt L212-L213
- ### 无损做不到，近似可以｜为什么需要：矩阵通常满秩、无损压缩做不到；按奇异值（矩阵在正交主方向上的伸缩系数）分解、只留最大的前 r 个，得到的低秩矩阵与原矩阵最接近｜320｜blog/mml-book/mml-ch4-svd.txt L668-L698；blog/mml-book/mml-ch4-svd.txt L754-L812；paper/lora-2106.09685.txt L206-L207
- ### 先降到低维再升回去｜怎么做：把 W=AB 看成把输入先降 r 维再升回原维度；损失由内维 r 决定｜300｜paper/lora-2106.09685.txt L212-L213；blog/mml-book/mml-ch4-svd.txt L754-L812；paper/deepseek-v2-2405.04434.txt L328-L345
