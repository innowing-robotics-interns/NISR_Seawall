#!/usr/bin/env python
"""Render dancing_children100K with its visible openings highlighted."""

from __future__ import annotations

import numpy as np
import trimesh
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from pathlib import Path

SRC = "data/mpz14/inputmodels/dancing_children100K.obj"
OUT = Path("experiments/rosa_holes/dancing_children_visuals")
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "raw"
RAW.mkdir(exist_ok=True)

VIEWS = {
    "front": ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    "back": ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
    "oblique": ([0.85, 0.22, 0.55], [0.0, 1.0, 0.0]),
}


def render() -> None:
    m = trimesh.load(SRC, process=False)
    m.apply_translation(-m.centroid)
    m.apply_scale(2.0 / m.scale)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(m.vertices),
        o3d.utility.Vector3iVector(m.faces),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.82, 0.80, 0.76])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=2000, height=1500)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.12, 0.16])
    opt.light_on = True
    vis.poll_events()
    vis.update_renderer()
    vc = vis.get_view_control()
    center = mesh.get_center()
    for name, (front, up) in VIEWS.items():
        vc.set_lookat(center)
        vc.set_up(up)
        vc.set_front(front)
        vc.set_zoom(0.85)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(RAW / f"{name}.png"), do_render=True)
    vis.destroy_window()


def hole_mask(path: Path, threshold: float = 45.0, min_area: int = 150):
    im = np.array(Image.open(path).convert("RGB"))
    gray = im.mean(axis=2)
    obj = gray > threshold
    filled = ndi.binary_fill_holes(obj)
    holes = filled & ~obj
    lab, n = ndi.label(holes)
    sizes = ndi.sum(holes, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
    keep.sort(key=lambda k: -sizes[k - 1])
    return im, obj, lab, keep, sizes


def annotate(path: Path, out_path: Path, color=(0, 225, 255, 255)):
    im, obj, lab, keep, sizes = hole_mask(path)
    rgba = Image.fromarray(im).convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    import cv2

    for k in keep:
        m = (lab == k).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = c[:, 0, :]
            if len(pts) < 3:
                continue
            draw.polygon([tuple(map(int, p)) for p in pts], fill=(*color[:3], 55))
            draw.line(
                [tuple(map(int, p)) for p in pts] + [tuple(map(int, pts[0]))],
                fill=color,
                width=4,
            )
    out = Image.alpha_composite(rgba, overlay).convert("RGB")
    out = crop_to_object(out, (1600, 1400), margin_frac=0.12)
    out.save(out_path, quality=95)
    return len(keep)


def crop_to_object(src, out_size: tuple[int, int], margin_frac=0.10):
    im = np.array((src if isinstance(src, Image.Image) else Image.open(src)).convert("RGB"))
    gray = im.mean(axis=2)
    obj = gray > 45
    ys, xs = np.where(obj)
    if len(xs) == 0:
        return Image.fromarray(im)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * margin_frac), int(h * margin_frac)
    x0, x1 = max(0, x0 - mx), min(im.shape[1], x1 + mx)
    y0, y1 = max(0, y0 - my), min(im.shape[0], y1 + my)
    crop = Image.fromarray(im[y0:y1, x0:x1])
    crop.thumbnail(out_size, Image.LANCZOS)
    return crop


def make_sheet() -> None:
    counts = {}
    for name in ("front", "back"):
        counts[name] = annotate(
            RAW / f"{name}.png", OUT / f"dancing_children_{name}_holes.png"
        )
    # clean oblique, cropped to the sculpture
    Image.open(RAW / "oblique.png").save(OUT / "oblique_raw.png", quality=95)
    crop_to_object(OUT / "oblique_raw.png", (1600, 1400), margin_frac=0.12).save(
        OUT / "dancing_children_oblique.png", quality=95
    )
    (OUT / "oblique_raw.png").unlink()

    W, H = 2200, 1680
    bg = (14, 18, 26)
    sheet = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(sheet)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        f_title = ImageFont.truetype(bold_path, 60)
        f_sub = ImageFont.truetype(font_path, 30)
        f_cap = ImageFont.truetype(font_path, 27)
        f_small = ImageFont.truetype(font_path, 24)
        f_info = ImageFont.truetype(font_path, 29)
        f_info_title = ImageFont.truetype(bold_path, 40)
    except OSError:
        f_title = f_sub = f_cap = f_small = f_info = f_info_title = ImageFont.load_default()

    d.text((70, 42), "Dancing Children: openings between arms, legs, and bodies",
           font=f_title, fill=(245, 245, 245))
    d.text((72, 122),
           "Closed orientable mesh  ·  V=49,986  E=150,000  F=100,000  ·  "
           "χ = −14  ⇒  genus 8",
           font=f_sub, fill=(170, 190, 205))

    cell_w, cell_h = 980, 655
    x0, y0 = 70, 190
    gap_x, gap_y = 90, 70
    positions = {
        "front": (x0, y0),
        "back": (x0 + cell_w + gap_x, y0),
        "oblique": (x0, y0 + cell_h + gap_y),
    }
    for name, (x, y) in positions.items():
        src = (
            OUT / f"dancing_children_{name}_holes.png"
            if name in counts
            else OUT / "dancing_children_oblique.png"
        )
        img = crop_to_object(src, (cell_w, cell_h))
        px = x + (cell_w - img.width) // 2
        py = y + (cell_h - img.height) // 2
        sheet.paste(img, (px, py))
        cap = {
            "front": f"Front view — {counts.get('front', 0)} enclosed background regions outlined",
            "back": f"Back view — {counts.get('back', 0)} enclosed background regions outlined",
            "oblique": "Three-quarter view — openings read as dark gaps",
        }[name]
        d.text((x, y + cell_h + 12), cap, font=f_cap, fill=(205, 215, 225))

    # Info panel bottom-right
    ix, iy = x0 + cell_w + gap_x, y0 + cell_h + gap_y
    d.rounded_rectangle((ix, iy, ix + cell_w, iy + cell_h), radius=24,
                        outline=(60, 78, 94), width=2, fill=(20, 26, 36))
    tx, ty = ix + 40, iy + 40
    d.text((tx, ty), "Euler characteristic", font=f_info_title, fill=(0, 225, 255))
    ty += 65
    lines = [
        ("χ = V − E + F = −14", (245, 245, 245)),
        ("g = (2 − χ) / 2 = 8", (245, 245, 245)),
        ("", (0, 0, 0)),
        ("Cyan outlines mark background visible through", (190, 205, 218)),
        ("the sculpture. These are silhouette openings,", (190, 205, 218)),
        ("not ROSA loop detections, and not a one-to-one", (190, 205, 218)),
        ("handle basis: handles can overlap, merge, or", (190, 205, 218)),
        ("hide behind a limb.", (190, 205, 218)),
        ("", (0, 0, 0)),
        ("Genus is robust: it stays 8 after quadric", (190, 205, 218)),
        ("simplification to 500 triangles.", (190, 205, 218)),
    ]
    for text, c in lines:
        d.text((tx, ty), text, font=f_info, fill=c)
        ty += 44
    sheet.save(OUT / "dancing_children_holes_sheet.png", quality=95)


if __name__ == "__main__":
    render()
    make_sheet()
    print("wrote", OUT)
