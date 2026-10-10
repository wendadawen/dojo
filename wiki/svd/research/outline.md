# 奇异值分解

- 描述：奇异值分解把任意矩阵分解为两个正交矩阵夹一个对角非负矩阵 $A=U\Sigma V^{\!\top}$，对角元是奇异值，刻画矩阵在正交方向上的伸缩，并给出最优低秩近似。
- 摘要：任意矩阵都能写成一「旋转、按坐标伸缩、再旋转」的复合，伸缩系数就是奇异值。据此得到最优低秩近似、伪逆与条件数。
- 主题：数学基础
- 标签：数学与数值

## 长页

- 开头｜SVD 是什么、为什么需要、大概怎么工作，点出后续各章｜400｜blog/ee263/ee263-17_svd.txt L16-L70；blog/mml-book/mml-ch4-svd.txt L16-L25
- ### 分解式与两种写法｜左右奇异向量、三个矩阵的形状与正交性、奇异值约定；薄 SVD 与全 SVD；前 $r$ 个非零奇异值对应的奇异向量张成的子空间｜600｜blog/mml-book/mml-ch4-svd.txt L27-L100；blog/ee263/ee263-17_svd.txt L48-L93；blog/lapack-lug/svd.txt L9-L39；blog/ee263/ee263-17_svd.txt L130-L145；blog/mml-book/mml-ch4-svd.txt L347-L352
- ### 几何：单位球映成椭球｜半轴长是奇异值、方向是左奇异向量；旋转-伸缩-旋转三步；2×2 手算例子折叠｜550｜blog/ee263/ee263-17_svd.txt L16-L47；blog/ee263/ee263-17_svd.txt L146-L198；blog/ee263/ee263-17_svd.txt L386-L425；blog/ee263/ee263-15_ellipsoids.txt L18-L55
- ### 奇异值就是 $A^{\!\top}A$ 特征值的平方根｜与 $A^{\!\top}A$、$AA^{\!\top}$ 特征分解的关系；构造 SVD 的步骤与手算例子折叠｜550｜blog/mml-book/mml-ch4-svd.txt L230-L459；blog/ee263/ee263-17_svd.txt L94-L129；blog/ee263/ee263-16_norm.txt L61-L89
- ### 奇异值刻画矩阵｜秩、谱范数、Frobenius 范数、条件数；自行构造的小例子折叠｜600｜blog/ee263/ee263-17_svd.txt L94-L106；blog/ee263/ee263-16_norm.txt L42-L52；blog/ee263/ee263-17_svd.txt L426-L483；blog/ee263/ee263-16_norm.txt L162-L182；blog/lapack-lug/svd.txt L139-L140
- ### 最优低秩近似｜截断 SVD；Eckart–Young 定理；$\sigma_i$ 是到最近秩 $i-1$ 矩阵的距离；压缩解释；证明与自行构造的小例子折叠｜750｜blog/mml-book/mml-ch4-svd.txt L628-L817；blog/ee263/ee263-17_svd.txt L484-L559
- ### 伪逆｜由 SVD 定义 $A^{+}=V\Sigma^{-1}U^{\!\top}$；长、宽矩阵两种情形；与 $A^{\!\top}A$ 求逆的关系折叠｜400｜blog/ee263/ee263-17_svd.txt L245-L312
- ### 适用条件与边界｜SVD 对任意矩阵存在而特征分解只对方阵、且要有一组特征向量；$\Sigma$ 唯一；由 $A^{\!\top}A$ 直接算的数值稳定性；误差用哪个范数｜350｜blog/mml-book/mml-ch4-svd.txt L16-L25；blog/mml-book/mml-ch4-svd.txt L46-L51；blog/mml-book/mml-ch4-svd.txt L460-L531；blog/mml-book/mml-ch4-svd.txt L454-L456；blog/mml-book/mml-ch4-svd.txt L743-L774

## 短页

- ### 两组正交方向夹一组伸缩｜SVD 是什么：任意矩阵写成 $A=U\Sigma V^{\!\top}$；解释正交矩阵、奇异值｜300｜blog/mml-book/mml-ch4-svd.txt L16-L100；blog/lapack-lug/svd.txt L9-L39
- ### 特征分解的局限｜为什么需要：特征分解只对方阵且有条件；SVD 对任意矩阵都存在｜320｜blog/mml-book/mml-ch4-svd.txt L16-L25；blog/mml-book/mml-ch4-svd.txt L460-L531
- ### 旋转、伸缩、再旋转｜怎么做：$y=Ax$ 拆成按 $V^{\!\top}$ 旋转、按 $\Sigma$ 伸缩、按 $U$ 旋转三步｜320｜blog/ee263/ee263-17_svd.txt L146-L198
