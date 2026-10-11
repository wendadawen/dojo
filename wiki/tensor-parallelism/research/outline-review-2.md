# 大纲审查 第 2 轮

## 读者

- 顺序错乱｜MLP 的列切与行切（长页）｜「GeLU」首次出现，全页没讲过｜读到这里不知道 GeLU 是什么｜首次出现处一句说明 GeLU 是逐元素非线性函数
- 顺序错乱｜切完每块卡的激活显存怎么变（长页）｜「层归一化、dropout、残差连接」三个词首次出现、没讲｜无法确认它们为什么一起处理｜各补一句括注或链接
- 顺序错乱｜按列或按行切（短页）｜「注意力」「head」短页内没讲也没链｜短页不自足｜短页内一句并链接 attention

## 对照资料

- 资料里没有｜切完每块卡的激活显存怎么变｜「逐元素的算子（层归一化、dropout、残差连接）」在依据范围内无对应，layer normalization 本身并非逐元素算子｜原文「Rather than having one GPU compute part of the dropout, layer normalization, or residual connections and broadcast the results to other GPUs, we choose to duplicate the computation across GPUs」（modelpar.tex L68）只说「重复算一份」｜去掉「逐元素」这一定语，改写为「dropout、层归一化、残差连接不切分、各卡各自重复算一份」

## 篇幅与冗余

- 重复｜MLP 的列切与行切／注意力按 head 切／一层两次 all-reduce｜「按列切让非线性本地施加、按行切第二段吃本地结果、输出处一次 all-reduce」这条通用机制被三节复述｜MLP 节「列切让 GeLU 本地施加…输出处一次 all-reduce」；注意力节「输出投影按行切直接吃本地输出，再 all-reduce 合并」｜通用取舍只在 MLP 节讲透，注意力节只留「Q/K/V 按列切使 head 整落一卡、注意力本身不通信」+ 输出投影共用同一处 all-reduce 一句带过；all-reduce 次数节不再展开复述

## 处理

- 读者·GeLU：采纳。MLP 节首次出现处括注「逐元素非线性函数」。
- 读者·层归一化/dropout/残差连接：采纳。显存章各半句解释，残差连接链 wiki/residual-connection；并去掉「逐元素」定语。
- 读者·短页注意力/head：采纳。短页第三节括注 head 并把注意力链到 wiki/standard-attention。
- 对照资料·去掉「逐元素」：采纳，同上。
- 篇幅·重复：采纳。注意力节删去取舍复述，只留 head 归属与共用 all-reduce 一句；预算仍 320（该节还需讲 Q/K/V 三处切分与 head 归属）。

改后：长页预算合计 2200，短页 940，机械检查通过。

## 最后一批改动核对

无改动（第 2 轮后还有第 3 轮，核对在最后一轮进行）。
