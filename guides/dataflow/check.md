# 核查页面

页面生成后核查。机械检查对源码、量几何、截图，三重都要过。

## 1. 源码位置

    python3 .dojo/scripts/dataflow/check_source.py wiki/<name>/research/dataflow.md --source-root <源码目录>

回查「源码」列指的行不是空行或注释、且含节点名。这一步在第 3 步之前就做过，改过计划后要重跑。

## 2. 节点名与形状

    python3 .dojo/scripts/dataflow/check_source.py wiki/<name>/index.html \
        --source <源码文件> --shapes wiki/<name>/research/sources/shapes.json

- 节点名里出现的标识符必须在源码里出现，防止凭空捏造。源码拆在多个文件时写成 `--source 标签=路径`，标签只在视图标题含该标签时生效。
- 形状表漏了某个键时记提示，不报错；页面写了形状而和材料对不上时报错。一个键都没比对上也算错。

## 3. 页面本身

    python3 .dojo/scripts/site/check.py wiki/<name>/index.html

跑页面的公共检查：本地引用存在、id 不重复、公式用 `$...$` 包起来、SVG 文字里不放公式、`<pre>` 里不画框线图。

## 4. 几何

    python3 .dojo/scripts/dataflow/check_source.py wiki/<name>/index.html --geometry

用无头 Chrome 逐视图量连线：有没有穿过别的节点框，有没有两条线并排重合。判据来自渲染后的像素，不能靠几何推算。需要本机装了 Chrome 或 Chromium。

## 5. 截图

    python3 .dojo/scripts/dataflow/check_source.py wiki/<name>/index.html --shots <目录>

每张图截三张：画布、说明面板、折叠。看折叠里的公式能不能正常显示、面板能不能打开。人眼过一遍。

## 改

- 页面内容改 `dataflow.md` 后重新生成，再按本页从头核查。
- 改了事实错误就回到第 3 步重派一轮，这一轮自己也要没有事实错误才通过。
- 整条路径重画时回到第 2 步重写计划。
- 交付前按 `plan.md` 的「怎么写」核对每张图的节点、连线、折叠和面板。
