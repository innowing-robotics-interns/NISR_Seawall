#!/usr/bin/env python3
"""Compose checkerboard view crops across clipped-loss runs."""
from __future__ import annotations
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

QUADRANTS = [(0, 0), (1, 0), (0, 1), (1, 1)]
ROW_NAMES = ['view 0', 'view 5', 'view 10', 'view 15']


def font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', action='append', default=[])
    ap.add_argument('--image', action='append', default=[])
    ap.add_argument('--notes', action='append', default=[])
    a = ap.parse_args()
    assert len(a.label) == len(a.image) == len(a.notes)

    crops = []
    for path in a.image:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        cw, ch = w // 2, h // 2
        crops.append([im.crop((qx * cw, qy * ch, (qx + 1) * cw, (qy + 1) * ch))
                      for qx, qy in QUADRANTS])

    cell_w = 512
    cell_h = 512
    left = 96
    top = 86
    bottom = 18
    header = 60
    ncol = len(crops)
    W = left + ncol * cell_w
    H = header + top + 4 * cell_h + bottom
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    f_title = font(26)
    f_col = font(20)
    f_note = font(18)
    f_row = font(22)
    for r in range(4):
        draw.text((10, header + top + r * cell_h + 8), ROW_NAMES[r],
                  fill='black', font=f_row)
    for c, (label, note) in enumerate(zip(a.label, a.notes)):
        x = left + c * cell_w
        draw.text((x + 6, 6), label, fill='black', font=f_title)
        draw.text((x + 6, 36), note, fill=(60, 60, 60), font=f_note)
        for r in range(4):
            canvas.paste(crops[c][r].resize((cell_w, cell_h),
                                            Image.LANCZOS),
                         (x, header + top + r * cell_h))
    canvas.save(a.out)
    print(a.out, canvas.size)


if __name__ == '__main__':
    main()
