# 资料冲突

- ViT-L/16（JFT-300M 预训练）与 BiT-L 的比较｜paper/src-v2/04_experiments.tex L122 正文写 "outperforms BiT-L ... on all tasks"；同一文件表 tbl:best_results（L90 ImageNet ReaL、L95 VTAB）显示 ViT-L/16 的 ImageNet ReaL 为 90.54、与 BiT-L 的 90.54 持平，VTAB 为 76.28、低于 BiT-L 的 76.29｜页面按表格写「在多数任务上超过 BiT-L」
- 视觉里注意力与卷积网络的关系｜paper/src-v2/main.tex L61-L62 摘要写 attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure；paper/src-v2/01_introduction.tex L11 引言写 multiple works ... some replacing the convolutions entirely，且 L13 称大规模图像识别上仍是 ResNet 这类架构领先｜页面按引言写「多与卷积网络结合、或替换其部分结构」
