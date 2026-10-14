"""Render the self-contained four-coordinate example as a bitmap."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
IMAGE = Image.new("RGB", (1120, 640), "#ffffff")
DRAW = ImageDraw.Draw(IMAGE)
INK = "#25282a"
MUTED = "#61676d"
GRID = "#dde1e4"


def text(position, value, size=24, color=INK, anchor=None):
    DRAW.text(position, value, font=ImageFont.truetype(FONT, size), fill=color, anchor=anchor)


text((48, 32), "Hadamard changes coordinates, not squared length", 32)
text((48, 82), "Self-constructed example: z = H4 x; no sign flips or quantization yet", 22, MUTED)

for left, title, values, color in [
    (90, "Input x", [7, 1, 0, 0], "#ae3d44"),
    (650, "Transformed z", [4, 3, 4, 3], "#167b72"),
]:
    text((left + 155, 138), title, 28, anchor="mm")
    baseline = 455
    for tick in range(8):
        y = baseline - tick * 34
        DRAW.line((left - 5, y, left + 365, y), fill=GRID, width=1)
        text((left - 17, y), str(tick), 18, MUTED, "rm")
    text((left - 35, 177), "Value", 20, MUTED)
    for index, value in enumerate(values):
        center = left + 30 + index * 94
        top = baseline - value * 34
        if value:
            DRAW.rectangle((center - 25, top, center + 25, baseline - 1), fill=color)
        else:
            DRAW.line((center - 25, baseline, center + 25, baseline), fill=color, width=3)
        text((center, top - 18), str(value), 24, anchor="mm")
        text((center, baseline + 23), str(index + 1), 22, MUTED, "mm")
    text((left + 172, baseline + 63), "Coordinate index (1-4)", 22, MUTED, "mm")
    text((left, 563), f"Max absolute value: {max(values)}", 24)
    text((left, 600), f"Sum of squares: {sum(value * value for value in values)}", 24)

(ROOT / "assets").mkdir(exist_ok=True)
IMAGE.save(ROOT / "assets" / "coordinates.png")
