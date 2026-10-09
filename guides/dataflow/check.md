# 核查页面

页面生成后核查，三步都要过。

## 1. 源码位置

    python3 .dojo/scripts/dataflow/check_source.py wiki/<name>/research/dataflow.md --source-root <源码目录>

回查「源码」列指的行不是空行或注释、且含节点名。改过计划后要重跑。

## 2. 节点名与形状

    python3 .dojo/scripts/dataflow/check_source.py wiki/<name>/index.html \
        --source <源码文件> --source <源码文件> ... \
        --shapes wiki/<name>/research/sources/shapes.json

把「资料」一节列出的每个源码文件各传一次（`--source` 可重复）。只传其中一份，别的图上的名字会被判「源码里找不到」。

源码拆在多个文件、且不同视图该对不同的文件时，写成 `--source 标签=路径`：标签是视图标题的子串，命中哪个标签就用那几份源码；没命中任何标签的视图拿全部源码比对。形状表漏了某个键记提示，不报错；页面写的形状和材料对不上、或一个键都没比对上，报错。

这一步**对示范页也要通过**——跑 `wiki/deepseek-v4-1-dataflow/index.html` 与它那四个源码文件，应当 0 错。

## 3. 页面本身

    python3 .dojo/scripts/site/check.py wiki/<name>/index.html

## 改

- 页面内容改 `dataflow.md` 后重新生成，再按本页从头核查。
- 改了事实错误就回到第 3 步重派一轮，这一轮自己也要没有事实错误才通过。
- 整条路径重画时回到第 2 步重写计划。
- 交付前按 `plan.md` 的「怎么写」核对每张图的节点、连线、折叠和面板。
