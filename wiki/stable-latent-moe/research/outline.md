# Stable LatentMoE

- 描述：Stable LatentMoE 是一种混合专家（MoE）层结构，把路由专家放进比模型全宽更窄的隐空间里计算，再用归一化、有界激活和分位数负载均衡三处改动，压住极端稀疏规模下路由分支的激活爆炸与近千专家的负载失衡。
- 摘要：把路由专家从全宽 $d$ 搬进宽度为 $\ell$ 的隐空间，换来 896 个路由专家、每 token 激活 16 个、稀疏度 56；三处稳定化分别管住路由分支的激活爆炸与专家负载失衡。
- 主题：模型结构, 训练与优化
- 标签：MoE

## 长页

- 开头｜讲清前馈层、MoE 一层与共享/路由组织、隐空间路由、无辅助损失路由各一两句，并定义稀疏度；再说是什么、为何需要、三处稳定化简说；K3 取值表折叠｜380｜paper/k3-src/2-model-architecture.tex L211-L221；paper/k3-src/3-pre-training.tex L57-L90
- ### 一层怎么算｜共享专家直接吃输入；路由分支投进隐空间、分发给 top-$k$ 专家、聚合、归一化（RMSNorm）后经上投影投回全宽、两支相加；符号说明与小算例折叠｜500｜paper/k3-src/2-model-architecture.tex L223-L234
- ### 极端稀疏放大出的两个失败模式｜稀疏度 56 下路由分支近四连矩阵乘加 2.8T 规模致激活爆炸；近千专家超出无辅助损失更新范围｜340｜paper/k3-src/2-model-architecture.tex L215-L220
- ### 聚合之后、上投影之前加归一化｜RMSNorm 是前置概念；$u$ 的尺度随选中专家与路由权重变化，插 RMSNorm 降敏感度、改善验证损失（验证集上的损失）；对照折叠｜400｜paper/k3-src/2-model-architecture.tex L236-L240
- ### 两个乘性因子各加上界｜SwiGLU 是前置概念；两因子无界，坐标同时很大时可能出外点、低精度下抬高溢出风险；softcap、SiTU-GLU 定义与上界 100 折叠｜440｜paper/k3-src/2-model-architecture.tex L243-L274
- ### 用分位数给专家定 bias｜由路由分数的 $(1-k/n)$ 分位数设专家 bias 使负载为 $q=mk/n$；bias 只进 top-$k$ 选择；与固定步长更新之别、直方图估计折叠｜420｜paper/k3-src/2-model-architecture.tex L289-L338
- ### 适用条件与边界｜结论依赖的规模与配置；三处稳定化针对极端稀疏的两个失败模式，资料未给出其他规模是否需要；与 LatentMoE 之别一两句｜380｜paper/k3-src/2-model-architecture.tex L214-L221；paper/k3-src/2-model-architecture.tex L257-L259

## 短页

- ### 隐空间里算专家，三处加稳定｜是什么：讲清 MoE、路由专家、共享专家、隐空间，再说把它搬进隐空间并加三处稳定化；不展开推导｜330｜paper/k3-src/2-model-architecture.tex L211-L221
- ### 全宽路由撑不起近千专家｜为什么需要：常规 MoE 每个选中专家都收全宽表示，扩大池与激活数抬高流量；到 896 专家、16 激活后路由分支激活爆炸与专家负载失衡压不住｜320｜paper/k3-src/2-model-architecture.tex L214-L221
- ### 压进隐空间，再各修一处｜怎么做：共享专家走全宽，路由分支投影、分发、聚合、归一化（RMSNorm）后投回、相加；再点出三处各修什么｜290｜paper/k3-src/2-model-architecture.tex L223-L240；paper/k3-src/2-model-architecture.tex L253-L274；paper/k3-src/2-model-architecture.tex L289-L338
