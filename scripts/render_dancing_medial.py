#!/usr/bin/env python
"""Render the validated medial-axis loop result for dancing_children100K.

This is the "after" figure for the detector improvement. It draws the eight
medial cycles as tubes inside a ghosted sculpture, pairs them with the solid
silhouette openings, and puts the Euler / voxel / timing evidence on one sheet.

Outputs (under experiments/rosa_holes/dancing_children100K_medial):
    medial_loops_front.png    clean ghost view with the eight cycles
    medial_loops_oblique.png  three-quarter ghost view
    dancing_medial_sheet.png  composed diagnostic sheet

Run with:
    python scripts/render_dancing_medial.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT / "data/mpz14/inputmodels/dancing_children100K.obj"
HOLES = ROOT / "experiments/rosa_holes/dancing_children100K_medial/holes.json"
ROSA_TUNED = ROOT / "experiments/rosa_holes/dancing_children100K_tuned/holes.json"
OUT = ROOT / "experiments/rosa_holes/dancing_children100K_medial"
SOLID = ROOT / "experiments/rosa_holes/dancing_children_visuals/dancing_children_front_holes.png"
BENCH_ROSA = ROOT / "experiments/rosa_holes/benchmark_rosa/benchmark.json"
BENCH_MEDIAL = ROOT / "experiments/rosa_holes/benchmark_medial/benchmark.json"

BG = (11, 14, 20)
PANEL = (15, 21, 31)
BORDER = (33, 45, 60)
WHITE = (244, 247, 250)
MUTED = (158, 176, 200)
CYAN = (0, 210, 255)
GREEN = (85, 224, 122)
RED = (255, 122, 107)
AMBER = (255, 200, 80)

# Okabe-Ito-ish, eight distinguishable hues.
LOOP_COLORS = [
    (0, 200, 255),
    (255, 140, 40),
    (90, 227, 107),
    (255, 92, 138),
    (255, 217, 61),
    (140, 125, 255),
    (45, 225, 194),
    (255, 98, 80),
]


def load_result(mesh_path: Path, holes_path: Path):
    """Load and sanity-check the medial run; return mesh + loops in display units."""
    data = json.loads(holes_path.read_text())
    if data.get("method") != "medial":
        raise SystemExit(f"{holes_path} is not a medial result (method={data.get('method')!r})")
    if data.get("validated") is not True:
        raise SystemExit(f"{holes_path} did not validate; refusing to draw it as a result")
    loops = [np.asarray(c["points"], dtype=float) for c in data["cycles"]]
    if data["detected_holes"] != len(loops):
        raise SystemExit("detected_holes does not match the stored cycles")
    if data.get("genus") is not None and len(loops) != int(data["genus"]):
        raise SystemExit(f"{len(loops)} loops but genus {data['genus']}")

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    centroid = mesh.centroid.copy()
    scale = 2.0 / mesh.scale
    loops = [(loop - centroid) * scale for loop in loops]
    mesh.apply_translation(-centroid)
    mesh.apply_scale(scale)
    return data, mesh, loops


def tube(loop: np.ndarray, radius: float) -> trimesh.Trimesh:
    sections = [
        trimesh.creation.cylinder(
            radius=radius, segment=[loop[i], loop[(i + 1) % len(loop)]], sections=12
        )
        for i in range(len(loop))
    ]
    return trimesh.util.concatenate(sections)


def look_at(eye, target, up) -> np.ndarray:
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up2
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def render_ghost(
    mesh: trimesh.Trimesh,
    loops: list[np.ndarray],
    eye,
    out_path: Path,
    width: int = 1700,
    height: int = 1275,
) -> None:
    scene = pyrender.Scene(bg_color=[c / 255 for c in BG] + [1.0],
                           ambient_light=[0.42, 0.44, 0.50])
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.82, 0.80, 0.77, 0.50],
        alphaMode="BLEND",
        metallicFactor=0.0,
        roughnessFactor=0.82,
        doubleSided=False,
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))

    for color, loop in zip(LOOP_COLORS, loops):
        body = tube(loop, radius=0.013)
        loop_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.0, 0.0, 0.0, 1.0],
            emissiveFactor=[c / 255 for c in color],
            alphaMode="OPAQUE",
            metallicFactor=0.0,
            roughnessFactor=0.45,
        )
        scene.add(pyrender.Mesh.from_trimesh(body, material=loop_mat, smooth=False))

    key = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.6, -0.8, 0.0],
         [0.0, 0.8, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    fill = np.array(
        [[-0.5, 0.0, 0.8, 0.0], [0.0, -0.6, 0.0, 0.0],
         [0.8, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.2), pose=key)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.1), pose=fill)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.2, aspectRatio=width / height)
    scene.add(camera, pose=look_at(eye, [0.0, -0.03, 0.0], [0.0, 1.0, 0.0]))
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()
    Image.fromarray(color).save(out_path)


def crop_content(image: Image.Image, margin: float = 0.055, threshold: int = 45) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    mask = arr.max(axis=2) > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image
    mx = int((xs.max() - xs.min()) * margin)
    my = int((ys.max() - ys.min()) * margin)
    return image.crop(
        (
            max(0, xs.min() - mx),
            max(0, ys.min() - my),
            min(image.width, xs.max() + mx),
            min(image.height, ys.max() + my),
        )
    )


def fit_contain(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    out = image.copy()
    out.thumbnail(box, Image.LANCZOS)
    return out


def load_fonts():
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    if not Path(regular).exists():
        regular = "/home/lyangiw/anaconda3/fonts/DejaVuSans.ttf"
        bold = regular
        mono = regular

    def font(size: int, path: str = regular):
        return ImageFont.truetype(path, size)

    return {
        "title": font(62, bold),
        "sub": font(31),
        "panel": font(36, bold),
        "body": font(27),
        "small": font(24),
        "tiny": font(21),
        "mono": font(25, mono),
        "big": font(86, bold),
        "stat": font(31, bold),
    }


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def draw_paragraph(draw, xy, text, font, fill, max_width, spacing=10):
    x, y = xy
    for line in wrap(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + spacing
    return y


def make_sheet(data, rosa, bench_medial, bench_rosa, front, oblique, solid) -> None:
    W, H = 2600, 1500
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    f = load_fonts()

    # Header
    d.text((84, 52), "dancing_children100K — 8 handles, measured and validated",
           font=f["title"], fill=WHITE)
    runtime = data["runtime"]
    topo = data["voxel_topology"]
    n_loops = len(data["cycles"])
    subtitle = (
        f"Solid medial-axis backend  ·  χ = −14 ⇒ genus 8  ·  "
        f"voxel b1 = {topo['b1']}  ·  {n_loops} loops in {runtime:.1f} s"
    )
    d.text((88, 140), subtitle, font=f["sub"], fill=MUTED)
    d.line((84, 205, W - 84, 205), fill=BORDER, width=2)

    # Hero: ghost view with the eight cycles
    hero_box = (1470, 1070)
    hero = fit_contain(crop_content(front), hero_box)
    hx = 84 + (hero_box[0] - hero.width) // 2
    hy = 236 + (hero_box[1] - hero.height) // 2
    sheet.paste(hero, (hx, hy))
    d.rounded_rectangle((84, 236, 84 + hero_box[0], 236 + hero_box[1]),
                        radius=18, outline=BORDER, width=2)
    d.text((108, 258), "8 independent medial cycles", font=f["panel"], fill=CYAN)
    cap_y = 236 + hero_box[1] + 18
    cap_y = draw_paragraph(
        d, (88, cap_y),
        "Each tube is a closed path through the material, away from the surface. "
        "Together they match the measured handle rank; a minimum cycle basis is not "
        "a one-loop-per-window labelling.",
        f["small"], MUTED, 1440, spacing=7,
    )
    d.text((88, cap_y + 4),
           f"skeleton {data['graph_nodes']} nodes / {data['graph_edges']} edges  ·  "
           f"{data['artifacts']} short voxel artifacts rejected  ·  "
           f"voxel pitch {data['voxel']:.2f}",
           font=f["tiny"], fill=(126, 142, 162))

    # Solid silhouette panel
    right_x, right_w = 1620, 900
    solid_box = (right_w, 580)
    solid_img = fit_contain(crop_content(Image.open(SOLID)), solid_box)
    sx = right_x + (right_w - solid_img.width) // 2
    sy = 300 + (solid_box[1] - solid_img.height) // 2
    sheet.paste(solid_img, (sx, sy))
    d.rounded_rectangle((right_x, 236, right_x + right_w, 236 + solid_box[1]),
                        radius=18, outline=BORDER, width=2)
    d.text((right_x + 24, 256), "Solid front", font=f["panel"], fill=WHITE)
    solid_cap_y = 300 + solid_box[1] + 14
    for line in wrap(d, "9 enclosed silhouette windows from the front (8 from the back). "
                        "Projected mouths can share one tunnel, so mouths ≠ genus.",
                     f["tiny"], right_w):
        d.text((right_x, solid_cap_y), line, font=f["tiny"], fill=MUTED)
        solid_cap_y += 28

    # Evidence card
    card_top = 986
    card_h = H - card_top - 80
    d.rounded_rectangle((right_x, card_top, right_x + right_w, card_top + card_h),
                        radius=18, fill=PANEL, outline=BORDER, width=2)
    x = right_x + 32
    y = card_top + 26
    d.text((x, y), "WHY 8", font=f["panel"], fill=CYAN)
    y += 58
    rows = [
        ("Euler", "49,986 − 150,000 + 100,000 = −14"),
        ("", "g = (2 − χ) / 2 = 8"),
        ("Voxel", f"b0 = {topo['b0']}, b2 = {topo['b2']}, χ = {topo['chi']}  →  b1 = {topo['b1']}"),
    ]
    for label, text in rows:
        if label:
            d.text((x, y), label, font=f["small"], fill=MUTED)
        d.text((x + 110, y), text, font=f["mono"], fill=WHITE)
        y += 40
    y += 8

    d.line((x, y, x + right_w - 64, y), fill=BORDER, width=2)
    y += 20
    rosa_loops = rosa["detected_holes"]
    rosa_time = rosa["runtime"]
    d.text((x, y), "BEFORE", font=f["small"], fill=RED)
    d.text((x + 125, y), f"ROSA (tuned, 20k samples)  {rosa_loops} loops · {rosa_time:.1f} s · unchecked",
           font=f["small"], fill=WHITE)
    y += 38
    d.text((x, y), "AFTER", font=f["small"], fill=GREEN)
    d.text((x + 125, y), f"medial (diag/80)  {n_loops} loops · {runtime:.1f} s · validated",
           font=f["small"], fill=WHITE)
    y += 50

    n_med = sum(1 for r in bench_medial["rows"] if r.get("match"))
    v_med = sum(1 for r in bench_medial["rows"] if r.get("validated"))
    n_rosa = sum(1 for r in bench_rosa["rows"] if r.get("match"))
    d.text((x, y), f"Benchmark  {n_med}/{len(bench_medial['rows'])} exact  ·  "
                   f"{v_med}/{len(bench_medial['rows'])} validated  ·  ROSA {n_rosa}/11",
           font=f["small"], fill=GREEN)
    y += 40
    d.text((x, y), "Watertight meshes only; coarse pitch avoids noise tunnels.",
           font=f["tiny"], fill=MUTED)

    # Small oblique view, bottom-left corner of the hero panel
    inset_w, inset_h = 380, 300
    inset = fit_contain(crop_content(oblique), (inset_w, inset_h))
    ix = 84 + hero_box[0] - inset.width - 22
    iy = 236 + hero_box[1] - inset.height - 22
    sheet.paste(inset, (ix, iy))
    d.rounded_rectangle((ix - 2, iy - 2, ix + inset.width + 2, iy + inset.height + 2),
                        radius=12, outline=(60, 80, 100), width=2)
    d.text((ix + 10, iy + 8), "3/4 view", font=f["tiny"], fill=WHITE)

    out = OUT / "dancing_medial_sheet.png"
    sheet.save(out, quality=95)
    print("wrote", out)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=MESH)
    parser.add_argument("--holes", type=Path, default=HOLES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)

    data, mesh, loops = load_result(args.mesh, args.holes)
    rosa = json.loads(ROSA_TUNED.read_text())
    bench_medial = json.loads(BENCH_MEDIAL.read_text())
    bench_rosa = json.loads(BENCH_ROSA.read_text())

    front_path = OUT / "medial_loops_front.png"
    oblique_path = OUT / "medial_loops_oblique.png"
    render_ghost(mesh, loops, [0.0, 0.0, -2.05], front_path, 1700, 1275)
    render_ghost(mesh, loops, [-0.82, -0.58, -1.98], oblique_path, 1500, 1200)
    print("wrote", front_path)
    print("wrote", oblique_path)

    make_sheet(
        data, rosa, bench_medial, bench_rosa,
        Image.open(front_path), Image.open(oblique_path), SOLID,
    )


if __name__ == "__main__":
    main()
