#!/usr/bin/env python
"""Project the tuned ROSA cycles onto dancing_children100K.

The tuned run returns 10 graph cycles. This script shows where they sit so
they can be compared with the eight actual handles: six on the base, three on
the right child's lower leg, one at the clasped hands.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/mpz14/inputmodels/dancing_children100K.obj"
HOLES = ROOT / "experiments/rosa_holes/dancing_children100K_tuned/holes.json"
OUT = ROOT / "experiments/rosa_holes/dancing_children100K_tuned"
W, H = 3000, 2250

BG = (0.10, 0.12, 0.16)
GROUPS = {
    "A": {
        "ids": [3, 5, 6, 8, 9, 10],
        "color": (0, 220, 255),
        "label": "base between the feet",
    },
    "B": {
        "ids": [1, 2, 7],
        "color": (255, 150, 60),
        "label": "right child's lower leg",
    },
    "C": {
        "ids": [4],
        "color": (120, 255, 120),
        "label": "clasped hands",
    },
}


def load_scene():
    data = json.loads(HOLES.read_text())
    loops = [np.asarray(c["points"], dtype=float) for c in data["cycles"]]
    if len(loops) != 10 or data["detected_holes"] != 10:
        raise RuntimeError(f"expected the 10-loop tuned run, got {len(loops)}")

    mesh = trimesh.load(SRC, process=False)
    centroid = mesh.centroid.copy()
    scale = mesh.scale
    mesh.apply_translation(-centroid)
    mesh.apply_scale(2.0 / scale)
    loops = [(L - centroid) * (2.0 / scale) for L in loops]

    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    geometry.compute_vertex_normals()
    geometry.paint_uniform_color([0.82, 0.80, 0.76])
    return data, geometry, loops


def render_front(geometry) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=W, height=H)
    vis.add_geometry(geometry)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(BG)
    opt.light_on = True
    vis.poll_events()
    vis.update_renderer()
    vc = vis.get_view_control()
    vc.set_lookat(geometry.get_center())
    vc.set_up([0.0, 1.0, 0.0])
    vc.set_front([0.0, 0.0, 1.0])
    vc.set_zoom(0.85)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(OUT / "loops_front_raw.png"), do_render=True)
    camera = vc.convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    K = np.asarray(camera.intrinsic.intrinsic_matrix)
    E = np.asarray(camera.extrinsic)
    image = Image.open(OUT / "loops_front_raw.png").convert("RGB")
    return K, E, image


def project(points: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    cam = (E @ np.c_[points, np.ones(len(points))].T).T
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    return np.c_[u, v]


def crop_object(image: Image.Image, margin=0.06) -> Image.Image:
    arr = np.asarray(image)
    obj = arr.mean(axis=2) > 45
    ys, xs = np.where(obj)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    mx = int((x1 - x0) * margin)
    my = int((y1 - y0) * margin)
    return image.crop(
        (max(0, x0 - mx), max(0, y0 - my),
         min(image.width, x1 + mx), min(image.height, y1 + my))
    )


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = image.copy()
    out.thumbnail(size, Image.LANCZOS)
    return out


def draw_cluster_boxes(draw: ImageDraw.ImageDraw, boxes, fonts):
    for key, (x0, y0, x1, y1) in boxes.items():
        color = GROUPS[key]["color"]
        draw.rounded_rectangle((x0, y0, x1, y1), radius=16, outline=color, width=6)
        letter = key
        lx = x0 + 8
        ly = max(10, y0 - 88)
        draw.rounded_rectangle((lx, ly, lx + 74, ly + 74), radius=12,
                               fill=(14, 18, 26), outline=color, width=5)
        draw.text((lx + 21, ly + 6), letter, font=fonts["badge"], fill=color)


def make_detail(image, loops, ids, K, E, box, size, color, letter, fonts):
    x0, y0, x1, y1 = box
    crop = image.crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(crop)
    for idx in ids:
        pts = project(loops[idx - 1], K, E) - np.array([x0, y0])
        draw.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=color, width=8)
        c = pts.mean(axis=0)
        draw.ellipse((c[0] - 10, c[1] - 10, c[0] + 10, c[1] + 10),
                     fill=(14, 18, 26), outline=color, width=4)
    crop = fit(crop, size)
    return crop


def main() -> None:
    data, geometry, loops = load_scene()
    K, E, image = render_front(geometry)

    proj = [project(L, K, E) for L in loops]
    boxes = {}
    for key, group in GROUPS.items():
        pts = np.concatenate([proj[i - 1] for i in group["ids"]])
        pad = 46 if key == "C" else 34
        boxes[key] = (
            float(pts[:, 0].min() - pad), float(pts[:, 1].min() - pad),
            float(pts[:, 0].max() + pad), float(pts[:, 1].max() + pad),
        )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        fonts = {
            "badge": ImageFont.truetype(bold_path, 54),
            "title": ImageFont.truetype(bold_path, 66),
            "sub": ImageFont.truetype(font_path, 32),
            "body": ImageFont.truetype(font_path, 29),
            "body_bold": ImageFont.truetype(bold_path, 29),
            "small": ImageFont.truetype(font_path, 26),
            "cap": ImageFont.truetype(font_path, 29),
        }
    except OSError:
        fallback = ImageFont.load_default()
        fonts = {k: fallback for k in
                 ("badge", "title", "sub", "body", "body_bold", "small", "cap")}

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for key, group in GROUPS.items():
        for idx in group["ids"]:
            pts = proj[idx - 1]
            draw.line([tuple(p) for p in pts] + [tuple(pts[0])],
                      fill=group["color"], width=8)
    draw_cluster_boxes(draw, boxes, fonts)
    overview = fit(crop_object(annotated, margin=0.04), (1290, 990))

    sheet = Image.new("RGB", (2400, 1700), (14, 18, 26))
    d = ImageDraw.Draw(sheet)
    d.text((62, 40), "Where ROSA's 10 loops are", font=fonts["title"],
           fill=(245, 245, 245))
    d.text((64, 122),
           "Tuned run on dancing_children100K  ·  20k samples  ·  "
           "10 cycles on a closed genus-8 mesh",
           font=fonts["sub"], fill=(170, 190, 205))

    ox, oy = 60, 200
    sheet.paste(overview, (ox + (1290 - overview.width) // 2, oy))
    d.text((ox, oy + 1000), "Front view. A/B/C mark the three clusters; "
           "the paths are projected through the surface.",
           font=fonts["small"], fill=(190, 205, 218))

    # Right-hand explanation
    px, py = 1410, 200
    d.rounded_rectangle((px, py, px + 930, py + 990), radius=24,
                        outline=(60, 78, 94), width=2, fill=(20, 26, 36))
    tx, ty = px + 38, py + 34
    d.text((tx, ty), "The 10 are not holes", font=fonts["body_bold"],
           fill=(255, 120, 120))
    ty += 52
    lines = [
        ("They are 10 spurious graph cycles. Six sit", fonts["body"], (205, 215, 225)),
        ("on the base between the feet, three on the", fonts["body"], (205, 215, 225)),
        ("right child's lower leg, and one at the", fonts["body"], (205, 215, 225)),
        ("clasped hands. All are 4-10 node spikes,", fonts["body"], (205, 215, 225)),
        ("29-90 units long.", fonts["body"], (205, 215, 225)),
        ("", fonts["body"], (205, 215, 225)),
        ("Axis alignment is 0.21-0.52. The true", fonts["body"], (205, 215, 225)),
        ("torus ring in the benchmark scores 0.99,", fonts["body"], (205, 215, 225)),
        ("so none of these tracks a tunnel.", fonts["body"], (205, 215, 225)),
        ("", fonts["body"], (205, 215, 225)),
        ("Loop IDs: A 3, 5, 6, 8, 9, 10", fonts["small"], (0, 220, 255)),
        ("          B 1, 2, 7     C 4", fonts["small"], (255, 150, 60)),
        ("", fonts["body"], (205, 215, 225)),
        ("The eight real handles come from", fonts["body"], (205, 215, 225)),
        ("chi = V - E + F = -14, not from this", fonts["body"], (205, 215, 225)),
        ("count. The real openings are the large", fonts["body"], (205, 215, 225)),
        ("see-through gaps between arms, legs and", fonts["body"], (205, 215, 225)),
        ("bodies in the previous figure.", fonts["body"], (205, 215, 225)),
    ]
    for text, font, color in lines:
        d.text((tx, ty), text, font=font, fill=color)
        ty += 44 if font is fonts["body"] else 36

    # Three detail cards
    cards = [
        ("A", boxes["A"], (0, 220, 255), "A — base between the feet: 6 loops"),
        ("B", boxes["B"], (255, 150, 60), "B — right child's lower leg: 3 loops"),
        ("C", boxes["C"], (120, 255, 120), "C — clasped hands: 1 loop"),
    ]
    widths = [1000, 600, 640]
    x = 60
    cy = 1248
    for (key, box, color, caption), cw in zip(cards, widths):
        d.rounded_rectangle((x, cy, x + cw, cy + 390), radius=20,
                            outline=(60, 78, 94), width=2, fill=(20, 26, 36))
        crop_box = (
            int(box[0] - 90), int(box[1] - 90),
            int(box[2] + 90), int(box[3] + 90),
        )
        detail = make_detail(
            image, loops, GROUPS[key]["ids"], K, E, crop_box,
            (cw - 36, 300), color, key, fonts,
        )
        sheet.paste(detail, (x + (cw - detail.width) // 2, cy + 12))
        d.text((x + 22, cy + 330), caption, font=fonts["cap"], fill=color)
        x += cw + 40

    d.text((64, 1652),
           "Projection is diagnostic: paths are drawn through the mesh. "
           "The loops themselves hug the surface, not the tunnels.",
           font=fonts["small"], fill=(160, 175, 190))

    sheet.save(OUT / "dancing_children100K_10_loops.png", quality=95)
    (OUT / "loops_front_raw.png").unlink()
    print("wrote", OUT / "dancing_children100K_10_loops.png")


if __name__ == "__main__":
    main()
