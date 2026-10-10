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

源码拆在多个文件、且不同视图该对不同的文件时，写成 `--source 标签=路径`：标签是视图标题的子串，命中哪个标签的视图就用那几份源码；**没有标签命中的视图，用不带标签的那些源码**（全部源码都带了标签时，才退回去用全部源码，免得变成一份都不核）。页面写的形状和材料对不上、一个键都没比对上、或写了「（检查点键 X）」而 X 不在形状表里（拼错或材料缺这个键），都报错——这几种都等于那个形状没被核过。标签要写清是哪张视图的（别用一个所有标题都含的字），免得「限定」变成对每张图都生效。

这一步**对示范页也要通过**——跑 `wiki/deepseek-v4-1-dataflow/index.html`，`--source` 传 `sources/repo/` 下这一页实际用到的源码文件，`--shapes` 传 `sources/shapes.json`，应当 0 错。

## 3. 页面本身

    python3 .dojo/scripts/site/check.py wiki/<name>/index.html

第 1、2 步要靠外部源码和形状表，CI 跑不了（它只跑 `site/check.py` 和两个脚本的 `--selftest`）。**CI 绿不等于这三步过了**——交付前必须自己在本地把三步都跑过，第 1、2 步尤其不能省。

## 改

- 页面内容改 `dataflow.md` 后重新生成，再按本页从头核查。
- 改了事实错误就回到第 3 步重派一轮，这一轮自己也要没有事实错误才通过。
- 整条路径重画时回到第 2 步重写计划。
- 交付前按 `plan.md` 的「怎么写」核对每张图的节点、连线、折叠和面板。
