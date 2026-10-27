#!/usr/bin/env python3
"""Render every patch of a checkpoint in its own checkerboard color.

Each active patch gets a distinct hue.  The checker pattern is evaluated per
pixel from the interpolated UV, so the pattern stays crisp even when a patch
is sampled with a coarse mesh.  Works for fixed-grid atlases (single, two,
six-sheet) and for adaptive quad-tree checkpoints, where every leaf is one
patch.

Fixed-grid runs also write one texture per patch for ``patch_vis.py`` plus an
atlas key image.  Adaptive runs (hundreds of leaves) write a compact palette
image instead of hundreds of individual textures.

Example:
    python scripts/render_checkerboard_patches.py \
        --checkpoint logs/render_phase3/ab/g05_r1/checkpoint.pt \
        --out logs/render_phase3/checkerboard
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import nvdiffrast.torch as dr
import utils.utils as nisr_utils
from checkpoint_atlas import (active_patch_ids, build_forward_model, num_faces,
                              patch_layout_info)
from model.ngf_rendering_loss import NGFRenderer
from utils.render_mesh import vertex_normals
from utils.render_targets import make_orbit_views

BORDER = 0.004        # UV band darkened to mark patch seams
BG = np.array([0.94, 0.94, 0.95], dtype=np.float32)


def palette(count: int):
    """Golden-angle hues so neighbouring patches never share a colour."""
    colors = []
    for index in range(count):
        hue = (index * 0.61803398875) % 1.0
        colors.append(colorsys.hsv_to_rgb(hue, 0.72, 0.95))
    return np.asarray(colors, dtype=np.float32)


def save_texture(path: str, base: np.ndarray, checkers: int,
                 size: int = 512) -> None:
    light = 0.42 * np.asarray(base) + 0.58
    step = size // checkers
    image = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(image)
    for i in range(checkers):
        for j in range(checkers):
            color = tuple(int(round(c * 255)) for c in
                          (base if (i + j) % 2 == 0 else light))
            draw.rectangle([j * step, i * step, (j + 1) * step - 1,
                            (i + 1) * step - 1], fill=color)
    image.save(path, quality=95)


def draw_swatch(draw, box, base, checkers):
    """Draw a checker swatch in a PIL box ``[x0, y0, x1, y1]``."""
    light = 0.42 * np.asarray(base) + 0.58
    x0, y0, x1, y1 = box
    step = max(1, (x1 - x0) // checkers)
    for i in range(checkers):
        for j in range(checkers):
            color = tuple(int(round(c * 255)) for c in
                          (base if (i + j) % 2 == 0 else light))
            draw.rectangle([x0 + j * step, y0 + i * step,
                            x0 + (j + 1) * step - 1, y0 + (i + 1) * step - 1],
                           fill=color)


def sample_patch(F, patch_id, resolution, device):
    u = torch.linspace(0, 1, resolution, device=device)
    grid_u, grid_v = torch.meshgrid(u, u, indexing='ij')
    uv = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)
    with torch.no_grad():
        verts = F(patch_id, uv)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            i00 = i * resolution + j
            i10 = (i + 1) * resolution + j
            i01 = i * resolution + (j + 1)
            i11 = (i + 1) * resolution + (j + 1)
            faces.append([i00, i10, i11])
            faces.append([i00, i11, i01])
    return (verts.float(), uv, np.asarray(faces, dtype=np.int32))


def label(image: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), text, font=font, fill=(20, 20, 20),
              stroke_width=2, stroke_fill=(255, 255, 255))
    return np.asarray(pil)


def montage(tiles, cols):
    rows = int(math.ceil(len(tiles) / cols))
    height, width = tiles[0].shape[:2]
    canvas = np.full((rows * height, cols * width, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        canvas[row * height:(row + 1) * height,
               col * width:(col + 1) * width] = tile
    return canvas


def write_atlas(path, F, patch_info, colors, n_patches, n_faces,
                n_rows, n_cols, checkers):
    """Fixed-grid key: one labelled swatch per patch, grouped by face."""
    cell = 256
    block_cols = min(3, n_faces) if n_faces > 1 else 1
    block_rows = int(math.ceil(n_faces / block_cols))
    atlas = Image.new('RGB', (block_cols * n_cols * cell,
                              block_rows * n_rows * cell), (235, 235, 238))
    draw = ImageDraw.Draw(atlas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
    except Exception:
        font = ImageFont.load_default()
    by_id = {p['patch_id']: p for p in patch_info}
    for patch_id in range(n_patches):
        layout = patch_layout_info(F, patch_id)
        face, row, col = layout['face'], layout['row'], layout['col']
        block_row, block_col = divmod(face, block_cols)
        x0 = (block_col * n_cols + col) * cell
        y0 = (block_row * n_rows + row) * cell
        label_text = (f'patch {patch_id}' if n_faces == 1
                      else f'f{face} p{patch_id}')
        if patch_id not in by_id:
            draw.rectangle([x0 + 4, y0 + 4, x0 + cell - 4, y0 + cell - 4],
                           fill=(200, 200, 205))
            draw.text((x0 + 12, y0 + 12), f'{label_text} inactive', font=font,
                      fill=(90, 90, 90))
            continue
        base = np.asarray(by_id[patch_id]['color'], dtype=np.float32)
        draw_swatch(draw, [x0 + 4, y0 + 4, x0 + cell - 4, y0 + cell - 4],
                    base, checkers)
        draw.text((x0 + 12, y0 + 12), label_text, font=font,
                  fill=(20, 20, 20), stroke_width=2, stroke_fill=(255, 255, 255))
    atlas.save(path)


def write_palette(path, patch_info, checkers):
    """Adaptive key: compact swatch grid; leaves keep their flat patch id."""
    count = len(patch_info)
    cell = 72
    cols = min(32, max(1, int(math.ceil(math.sqrt(count)))))
    rows = int(math.ceil(count / cols))
    canvas = Image.new('RGB', (cols * cell, rows * cell), (235, 235, 238))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12)
    except Exception:
        font = ImageFont.load_default()
    for slot, info in enumerate(patch_info):
        row, col = divmod(slot, cols)
        x0 = col * cell
        y0 = row * cell
        base = np.asarray(info['color'], dtype=np.float32)
        draw_swatch(draw, [x0 + 3, y0 + 3, x0 + cell - 4, y0 + cell - 4],
                    base, checkers)
        draw.text((x0 + 6, y0 + 6), f"p{info['patch_id']}", font=font,
                  fill=(20, 20, 20), stroke_width=2,
                  stroke_fill=(255, 255, 255))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--grid-res', type=int, default=0,
                        help='UV samples per patch side; 0 = auto '
                             '(12 adaptive, 128 fixed-grid)')
    parser.add_argument('--checkers', type=int, default=0,
                        help='checker squares per UV side; 0 = auto '
                             '(4 adaptive, 8 fixed-grid)')
    parser.add_argument('--views', type=int, default=4)
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ck_args = ck['args']
    F, atlas_info = build_forward_model(ck, device)
    adaptive = bool(atlas_info.get('adaptive'))
    active_ids = active_patch_ids(ck, F)
    n_patches = int(F.n_patches)
    n_faces = num_faces(F)
    n_rows, n_cols = (None, None) if adaptive else ck['grid_dims']
    colors = palette(len(active_ids))

    grid_res = (args.grid_res if args.grid_res > 0 else
                (12 if adaptive else 128))
    checkers = (args.checkers if args.checkers > 0 else
                (4 if adaptive else 8))
    print(f"  atlas: {atlas_info['atlas_mode']}  faces {n_faces}  "
          f"patches {n_patches}  grid {grid_res}  checkers {checkers}")

    tex_dir = None
    if not adaptive:
        tex_dir = os.path.join(args.out, 'textures')
        os.makedirs(tex_dir, exist_ok=True)
        for index, color in enumerate(colors, start=1):
            save_texture(os.path.join(tex_dir, f'Slide{index}.jpg'),
                         color, checkers)

    pts3n, meta = nisr_utils.load_point_cloud(ck_args['file'])

    all_verts, all_faces, all_uv, all_slot = [], [], [], []
    offset = 0
    patch_info = []
    for slot, patch_id in enumerate(active_ids):
        verts, uv, faces = sample_patch(F, patch_id, grid_res, device)
        all_verts.append(verts)
        all_faces.append(torch.as_tensor(faces, device=device) + offset)
        all_uv.append(uv)
        all_slot.append(torch.full((uv.shape[0], 1), float(slot),
                                   device=device))
        layout = patch_layout_info(F, patch_id)
        entry = {
            'patch_id': patch_id,
            'face': layout['face'],
            'row': layout['row'],
            'col': layout['col'],
            'color': [float(c) for c in colors[slot]],
        }
        if layout['depth'] is not None:
            entry['depth'] = layout['depth']
        if tex_dir is not None:
            entry['texture'] = f'Slide{slot + 1}.jpg'
        patch_info.append(entry)
        offset += verts.shape[0]

    verts = torch.cat(all_verts)
    faces = torch.cat(all_faces).to(torch.int32)
    uv_attr = torch.cat(all_uv)
    slot_attr = torch.cat(all_slot)
    normals = vertex_normals(verts, faces)
    palette_t = torch.as_tensor(colors, device=device)

    proj = NGFRenderer.projection(45.0, 1.0, 0.1, 1000.0, device)
    pts_t = torch.as_tensor(pts3n, dtype=torch.float32, device=device)
    views, _, _, _ = make_orbit_views(pts_t, count=16, fov=45.0,
                                      device=device)
    selected = np.linspace(0, len(views) - 1, args.views).round().astype(int)
    ctx = dr.RasterizeCudaContext()
    light = torch.tensor([0.35, 0.55, 0.75], device=device)
    light = light / light.norm()

    tiles = []
    for view_index in selected:
        # Keep a minibatch axis: nvdiffrast interprets a 2-D pos array as
        # range mode and then rejects the empty ranges.
        view = views[view_index:view_index + 1].to(device)
        mvp = proj @ view
        v_hom = torch.nn.functional.pad(verts, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom.unsqueeze(0), mvp.transpose(1, 2))
        rast = dr.rasterize(ctx, v_ndc, faces,
                            (args.resolution, args.resolution))[0]
        raw_uv = dr.interpolate(uv_attr.unsqueeze(0), rast, faces)[0]
        raw_slot = dr.interpolate(slot_attr.unsqueeze(0), rast, faces)[0]
        normal = dr.interpolate(normals, rast, faces)[0]
        normal = dr.antialias(normal, rast, v_ndc, faces)
        alpha = (rast[..., 3:4] > 0).float()

        # Per-pixel checker: crisp regardless of the UV sampling density.
        slot_index = raw_slot[..., :1].round().long().clamp_(
            0, len(colors) - 1)
        base = palette_t[slot_index[..., 0]]
        tint = 0.42 * base + 0.58
        cell = (torch.floor(raw_uv[..., :1] * checkers)
                + torch.floor(raw_uv[..., 1:] * checkers)) % 2
        color = torch.where(cell < 0.5, base, tint)
        edge = ((raw_uv[..., :1] < BORDER)
                | (raw_uv[..., :1] > 1.0 - BORDER)
                | (raw_uv[..., 1:] < BORDER)
                | (raw_uv[..., 1:] > 1.0 - BORDER))
        color = torch.where(edge, color * 0.35, color)

        shade = 0.45 + 0.55 * (normal * light).sum(-1, keepdim=True).abs()
        rgb = color * shade * alpha + torch.as_tensor(
            BG, device=device) * (1.0 - alpha)
        image = (rgb[0].clamp(0, 1).flip(0).cpu().numpy() * 255
                 + 0.5).astype(np.uint8)
        imageio.imwrite(os.path.join(args.out, f'checkerboard_view_{view_index:02d}.png'),
                        image)
        tiles.append(label(image, f'view {view_index}'))

    imageio.imwrite(os.path.join(args.out, 'checkerboard_all_views.png'),
                    montage(tiles, cols=2))

    layout_name = 'checkerboard_palette.png' if adaptive else \
        'checkerboard_atlas.png'
    if adaptive:
        write_palette(os.path.join(args.out, layout_name), patch_info,
                      checkers)
    else:
        write_atlas(os.path.join(args.out, layout_name), F, patch_info,
                    colors, n_patches, n_faces, n_rows, n_cols, checkers)

    summary = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'atlas_mode': atlas_info['atlas_mode'],
        'atlas_faces': n_faces,
        'adaptive': adaptive,
        'grid': [n_rows, n_cols] if n_rows is not None else None,
        'active_patches': active_ids,
        'checker_squares_per_side': checkers,
        'uv_grid_resolution': grid_res,
        'render_resolution': args.resolution,
        'views_rendered': selected.tolist(),
        'mesh_vertices': int(verts.shape[0]),
        'mesh_faces': int(faces.shape[0]),
        'layout_image': layout_name,
        'patches': patch_info,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(f"  {len(active_ids)} patches, {verts.shape[0]} verts, "
          f"{faces.shape[0]} faces")
    if tex_dir is not None:
        print(f"  textures -> {tex_dir}")
    print(f"  palette  -> {os.path.join(args.out, layout_name)}")
    print(f"  images   -> {args.out}")


if __name__ == '__main__':
    main()
