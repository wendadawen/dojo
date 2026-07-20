#!/usr/bin/env python3
"""把图片转成 base64 data URI 字符串，粘进 HTML 里 <img src="..."> 的引号部分。

用法：
    python img_to_b64.py <图片路径>

输出示例：
    data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...

在 HTML 里这样用：
    <img src="data:image/png;base64,iVBORw0KGgo..." alt="图 1：模型架构（论文 Figure 1）"
         style="width: 100%; max-width: 700px; display: block; margin: 1em auto;">

注意：脚本只输出 data URI 字符串本身，不包含 <img> 标签外壳。
这样占位符替换是纯字符串替换，不会产生嵌套标签。
"""
import base64
import sys


def main():
    if len(sys.argv) < 2:
        print("用法: python img_to_b64.py <图片路径>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]

    ext = path.rsplit(".", 1)[-1].lower()
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
    }
    if ext not in mime_map:
        print(f"不支持的图片格式: .{ext}", file=sys.stderr)
        sys.exit(1)

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    print(f"data:{mime_map[ext]};base64,{b64}")


if __name__ == "__main__":
    main()
