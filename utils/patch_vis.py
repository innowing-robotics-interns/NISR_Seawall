#!/usr/bin/env python3
# patch_vis.py

import os
import argparse
import importlib.util
import glob
import colorsys
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import trimesh
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pc_presegmentation as pc_presegmentation
import utils as utils


# ─────────────────────────────────────────────────────────────────────────────
# Texture loading
# ─────────────────────────────────────────────────────────────────────────────
def load_checkerboard_textures(texture_path, pattern="Slide5.jpg", n_images=1):
    """
    Load checkerboard textures from a file or directory.

    Change pattern to "Slide{}.jpg" if you want to load multiple images from a directory.

    Args:
        texture_path: Path to one image file or a directory of images.
        pattern: Filename pattern used in directory mode.
        n_images: Number of images to load in directory mode.

    Returns:
        List of `PIL.Image` objects.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(texture_path):
        candidates = [
            texture_path,
            os.path.join(script_dir, texture_path),
            os.path.join(os.path.dirname(script_dir), texture_path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                texture_path = candidate
                break

    if os.path.isfile(texture_path):
        img = Image.open(texture_path).convert("RGB")
        print(f"  Using ONE checkerboard texture for all patches: {texture_path}")
        return [img]

    if os.path.isdir(texture_path):
        textures = []
        for i in range(1, n_images + 1):
            p = os.path.join(texture_path, pattern.format(i))
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing checkerboard texture: {p}")
            textures.append(Image.open(p).convert("RGB"))
        print(f"  Loaded {len(textures)} checkerboard texture(s) from {texture_path}")
        return textures

    raise FileNotFoundError(
        f"Checkerboard texture path not found: {texture_path}\n"
        f"  Pass either a single image file or a directory of numbered images."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-patch grid sampling
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _sample_patch_grid(F, patch_idx, resolution, device):
    """
    Sample one patch of `MultiPatchForwardMap` on a regular UV grid.

    This mirrors utils.sample_multi_patch_grid's per-patch sampling exactly:
    the same F(patch_idx, uv) call on a [0,1]^2 grid. The only addition is that
    we also return the UV grid so it can be used as texture coordinates.

    Returns:
        Tuple `(verts, uv, faces)`.
    """
    u = torch.linspace(0, 1, resolution, device=device)
    v = torch.linspace(0, 1, resolution, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
    uv = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)  # (res*res, 2)

    batch_size = 4096
    verts_list = []
    for i in range(0, uv.shape[0], batch_size):
        verts_list.append(F(patch_idx, uv[i:i + batch_size]).cpu())
    verts = torch.cat(verts_list, dim=0).numpy()

    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx00 = i * resolution + j
            idx10 = (i + 1) * resolution + j
            idx01 = i * resolution + (j + 1)
            idx11 = (i + 1) * resolution + (j + 1)
            faces.append([idx00, idx10, idx11])
            faces.append([idx00, idx11, idx01])
    faces = np.array(faces, dtype=np.int32)

    return verts.astype(np.float32), uv.cpu().numpy().astype(np.float32), faces


def _bake_vertex_colors(mesh):
    """Convert a textured mesh into a vertex-colored copy for PLY export."""
    colored = mesh.copy()
    colored.visual = mesh.visual.to_color()
    return colored


def _make_double_sided(verts, uv, faces):
    """
    Duplicate sheet geometry so it renders from both sides.

    Returns:
        Tuple `(verts2, uv2, faces2)`.
    """
    n = verts.shape[0]
    verts2 = np.concatenate([verts, verts], axis=0)
    uv2 = np.concatenate([uv, uv], axis=0)
    faces_front = faces
    faces_back = faces[:, [0, 2, 1]] + n  # reversed winding -> opposite normal
    faces2 = np.concatenate([faces_front, faces_back], axis=0).astype(np.int32)
    return verts2, uv2, faces2


def _build_patch_uv_occupancy(points, subdivision_depth: int, min_points: int = 1):
    """Build an occupancy mask over a patch-local UV domain [0,1]^2. (legacy mode)"""
    if points is None or len(points) < min_points:
        return np.zeros((1, 1), dtype=bool)

    final_res = 1 << max(0, subdivision_depth)
    occ = np.zeros((final_res, final_res), dtype=bool)

    if subdivision_depth == 0:
        occ[0, 0] = len(points) >= min_points
        return occ

    cells = [(points, 0, 0, final_res)]

    for depth in range(subdivision_depth):
        next_cells = []
        for cell_points, row0, col0, size in cells:
            if len(cell_points) < min_points:
                continue
            if size <= 1:
                next_cells.append((cell_points, row0, col0, size))
                continue
            assignments, _, _ = pc_presegmentation.pca_grid_segmentation(
                cell_points, n_patches_u=2, n_patches_v=2
            )
            half = size // 2
            for sub_id in range(4):
                sub_points = cell_points[assignments == sub_id]
                if len(sub_points) < min_points:
                    continue
                sub_r = sub_id // 2
                sub_c = sub_id % 2
                next_cells.append((sub_points, row0 + sub_r * half, col0 + sub_c * half, half))
        cells = next_cells
        if not cells:
            break

    for _, row0, col0, size in cells:
        occ[row0:row0 + size, col0:col0 + size] = True
    return occ


def _filter_patch_mesh_by_occupancy(verts, uv, faces, occ_mask):
    """Keep only vertices/faces whose UVs lie in occupied cells. (legacy mode)"""
    if occ_mask.size == 0 or not occ_mask.any():
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32))

    res_u, res_v = occ_mask.shape
    u_idx = np.clip((uv[:, 0] * res_u).astype(int), 0, res_u - 1)
    v_idx = np.clip((uv[:, 1] * res_v).astype(int), 0, res_v - 1)
    keep_v = occ_mask[u_idx, v_idx]

    kept_indices = np.flatnonzero(keep_v)
    if kept_indices.size == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32))

    index_map = -np.ones(len(verts), dtype=np.int32)
    index_map[kept_indices] = np.arange(len(kept_indices), dtype=np.int32)
    keep_f = keep_v[faces].all(axis=1)
    faces_kept = index_map[faces[keep_f]]
    return verts[kept_indices], uv[kept_indices], faces_kept


def _save_debug_uv_png(uv, tex_img, save_dir, cid):
    """Save a UV debug image with sampled points overlaid on the texture."""
    try:
        import cv2
    except ImportError:
        return

    w, h = tex_img.size
    img = np.array(tex_img.convert("RGB")).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pts_px = (uv * [w, h]).astype(np.int32)
    for px, py in pts_px[::max(1, len(pts_px) // 2000)]:
        cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)

    patch_dir = os.path.join(save_dir, str(cid))
    os.makedirs(patch_dir, exist_ok=True)
    out_path = os.path.join(patch_dir, "debug_uv.png")
    cv2.imwrite(out_path, img)


# ─────────────────────────────────────────────────────────────────────────────
# Export function — LEGACY (current behaviour, non-quadtree)
# ─────────────────────────────────────────────────────────────────────────────
def export_checkerboard_patches(F, meta, save_dir, texture_path,
                                 resolution=100, device='cuda',
                                 epoch='10k', name=None,
                                 texture_pattern="Slide5.jpg", n_images=1,
                                 unnormalize=True, debug_uv_png=True,
                                 export_ply=True, double_sided=True,
                                 active_patch_ids=None,
                                 patch_points_by_id=None,
                                 subdivision_depth=1,
                                 min_points_per_cell=1,
                                 save_each_patch=True):
    """Export checkerboard-textured patch meshes and a combined scene (legacy)."""
    os.makedirs(save_dir, exist_ok=True)
    textures = load_checkerboard_textures(texture_path, pattern=texture_pattern,
                                          n_images=n_images)
    n_textures = len(textures)

    sample_all_patches = subdivision_depth < 0

    if active_patch_ids is None or sample_all_patches:
        patch_ids = list(range(F.n_patches))
    else:
        patch_ids = [int(patch_idx) for patch_idx in active_patch_ids]

    if len(patch_ids) == 0:
        print("  No active patches found; skipping checkerboard export.")
        return []

    meshes = []
    colored_meshes = []

    use_legacy_sampling = subdivision_depth <= 0 or patch_points_by_id is None

    for export_idx, cid in enumerate(patch_ids):
        verts, uv, faces = _sample_patch_grid(F, cid, resolution, device)
        if not use_legacy_sampling:
            occ_mask = _build_patch_uv_occupancy(
                patch_points_by_id.get(int(cid)),
                subdivision_depth=subdivision_depth,
                min_points=min_points_per_cell,
            )
            verts, uv, faces = _filter_patch_mesh_by_occupancy(verts, uv, faces, occ_mask)
            if verts.shape[0] == 0 or faces.shape[0] == 0:
                print(f"    Patch {cid:02d} skipped after occupancy filtering")
                continue
        tex_img = textures[export_idx % n_textures]

        if debug_uv_png:
            _save_debug_uv_png(uv, tex_img, save_dir, cid)

        if double_sided:
            verts, uv, faces = _make_double_sided(verts, uv, faces)

        verts_out = verts * meta['scale'] + meta['center'] if unnormalize else verts

        uv_visuals = trimesh.visual.texture.TextureVisuals(uv=uv, image=tex_img)
        mesh = trimesh.Trimesh(vertices=verts_out, faces=faces,
                               visual=uv_visuals, process=False,
                               maintain_order=True)
        meshes.append(mesh)

        if save_each_patch:
            obj_path = os.path.join(save_dir, f"patch_{cid}_{epoch}.obj")
            mesh.export(obj_path)
            print(f"    Patch {cid:02d} textured OBJ → {obj_path}")

        if export_ply:
            colored_mesh = _bake_vertex_colors(mesh)
            colored_meshes.append(colored_mesh)
            if save_each_patch:
                ply_path = os.path.join(save_dir, f"patch_{cid}_{epoch}.ply")
                colored_mesh.export(ply_path)
                print(f"    Patch {cid:02d} vertex-colored PLY → {ply_path}")

    if len(meshes) == 0:
        print("  No meshes remained after occupancy filtering; skipping combined export.")
        return []

    _export_combined(meshes, colored_meshes, save_dir, name, epoch, export_ply)
    return meshes


# ─────────────────────────────────────────────────────────────────────────────
# Export function — QUADTREE (reproduces utils.sample_multi_patch_grid)
# ─────────────────────────────────────────────────────────────────────────────
def export_checkerboard_patches_quadtree(F, meta, save_dir, texture_path,
                                         resolution=100, device='cuda',
                                         epoch='10k', name=None,
                                         texture_pattern="Slide5.jpg", n_images=1,
                                         unnormalize=True, debug_uv_png=True,
                                         export_ply=True, double_sided=True,
                                         save_each_patch=True):
    """
    Quadtree export: sample EXACTLY the model's active leaves (leaf_count > 0),
    identical to utils.sample_multi_patch_grid, then attach the checkerboard
    texture per leaf. --subdivision_depth is ignored on purpose: the trained
    model already knows which leaves are active.
    """
    os.makedirs(save_dir, exist_ok=True)
    textures = load_checkerboard_textures(texture_path, pattern=texture_pattern,
                                          n_images=n_images)
    n_textures = len(textures)

    # This is the crucial line: use the model's own active leaves, exactly like
    # utils.sample_multi_patch_grid(..., active_patch_ids=F.active_patch_ids).
    patch_ids = list(getattr(F, 'active_patch_ids', range(F.n_patches)))
    if len(patch_ids) == 0:
        print("  No active leaves found in model; nothing to export.")
        return []

    print(f"  Quadtree export: {len(patch_ids)} active leaves "
          f"(of {F.n_patches} total) @ resolution {resolution}")

    meshes = []
    colored_meshes = []

    for export_idx, cid in enumerate(patch_ids):
        verts, uv, faces = _sample_patch_grid(F, cid, resolution, device)
        tex_img = textures[export_idx % n_textures]

        if debug_uv_png:
            _save_debug_uv_png(uv, tex_img, save_dir, cid)

        if double_sided:
            verts, uv, faces = _make_double_sided(verts, uv, faces)

        verts_out = verts * meta['scale'] + meta['center'] if unnormalize else verts

        uv_visuals = trimesh.visual.texture.TextureVisuals(uv=uv, image=tex_img)
        mesh = trimesh.Trimesh(vertices=verts_out, faces=faces,
                               visual=uv_visuals, process=False,
                               maintain_order=True)
        meshes.append(mesh)

        if save_each_patch:
            obj_path = os.path.join(save_dir, f"patch_{cid}_{epoch}.obj")
            mesh.export(obj_path)
            print(f"    Leaf {cid:04d} textured OBJ → {obj_path}")

        if export_ply:
            colored_mesh = _bake_vertex_colors(mesh)
            colored_meshes.append(colored_mesh)
            if save_each_patch:
                ply_path = os.path.join(save_dir, f"patch_{cid}_{epoch}.ply")
                colored_mesh.export(ply_path)
                print(f"    Leaf {cid:04d} vertex-colored PLY → {ply_path}")

    _export_combined(meshes, colored_meshes, save_dir, name, epoch, export_ply)
    return meshes


def _export_combined(meshes, colored_meshes, save_dir, name, epoch, export_ply):
    """Write the combined textured OBJ scene and merged vertex-colored PLY."""
    scene = trimesh.Scene(meshes)
    obj_scene_name = f"{name}_checkerboard_{epoch}.obj" if name else f"checkerboard_{epoch}.obj"
    obj_scene_path = os.path.join(save_dir, obj_scene_name)
    scene.export(obj_scene_path)
    print(f"    Combined checkerboard scene (OBJ) → {obj_scene_path}")

    if export_ply and colored_meshes:
        combined_colored = trimesh.util.concatenate(colored_meshes)
        ply_scene_name = f"{name}_checkerboard_{epoch}.ply" if name else f"checkerboard_{epoch}.ply"
        ply_scene_path = os.path.join(save_dir, ply_scene_name)
        combined_colored.export(ply_scene_path)
        print(f"    Combined checkerboard scene (PLY) → {ply_scene_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  QUADTREE PATCH-COLOUR MODULE   (self-contained, always runs in --quadtree)
#
#  Writes EXACTLY three extra files, never any per-patch file:
#     <name>_quadtree_colors.jpg   two panels: all leaves / inactive blank
#     <name>_patch_colors.ply      all leaves sampled, flat colour per leaf
#     <name>_patch_colors.obj      same mesh, vertex colours inline
#
#  Neighbouring leaves (edge- or corner-touching) always get different colours.
# ═════════════════════════════════════════════════════════════════════════════

_PALETTE = np.array([
    (228,  26,  28), ( 55, 126, 184), ( 77, 175,  74), (152,  78, 163),
    (255, 127,   0), (255, 214,  20), (166,  86,  40), (247, 129, 191),
    ( 26, 188, 156), (106,  61, 154), (178, 223, 138), (251, 154, 153),
    ( 31, 119, 180), (255, 187, 120), ( 44, 160,  44), (214,  39,  40),
    (148, 103, 189), (140,  86,  75), ( 23, 190, 207), (188, 189,  34),
], dtype=np.uint8)


def _color_palette(n_needed):
    """Return at least `n_needed` visually distinct RGB colours."""
    if n_needed <= len(_PALETTE):
        return _PALETTE.copy()
    extra, golden, h = [], 0.618033988749895, 0.13
    for k in range(n_needed - len(_PALETTE)):
        h = (h + golden) % 1.0
        s = 0.50 + 0.40 * ((k % 3) / 2.0)
        v = 0.95 - 0.25 * (k % 2)
        extra.append(tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, s, v)))
    return np.concatenate([_PALETTE, np.array(extra, dtype=np.uint8)], axis=0)


def _to_np(x):
    """Tensor / array / numeric list → ndarray (or None)."""
    try:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, (list, tuple)) and len(x):
            a = np.asarray(x)
            if a.dtype != object and np.issubdtype(a.dtype, np.number):
                return a
    except Exception:
        return None
    return None


def _topology_arrays(F, topology=None):
    """Collect the quadtree arrays from the checkpoint topology / model complex."""
    wanted = ('leaf_bbox', 'leaf_count', 'leaf_k', 'leaf_poly_ids',
              'leaf_poly_uv', 'vertex_uv', 'vertex_features')
    out = {}
    sources = []
    if isinstance(topology, dict):
        sources.append(topology)
    cx = getattr(F, 'complex', None)
    if cx is not None:
        sources.append(cx)
    for src in sources:
        for key in wanted:
            if key in out:
                continue
            val = src.get(key) if isinstance(src, dict) else getattr(src, key, None)
            arr = _to_np(val)
            if arr is not None and arr.size:
                out[key] = arr
    return out


# ── leaf squares in the global UV domain ─────────────────────────────────────
def _tiling_score(boxes, grid):
    """1.0 when the boxes tile [0,1]^2 exactly once; overlaps/gaps are penalised."""
    G = int(min(max(grid, 1), 256))
    cov = np.zeros((G, G), dtype=np.int32)
    for x0, y0, w, h in boxes:
        i0, j0 = int(np.floor(x0 * G + 1e-6)), int(np.floor(y0 * G + 1e-6))
        i1 = max(int(np.ceil((x0 + w) * G - 1e-6)), i0 + 1)
        j1 = max(int(np.ceil((y0 + h) * G - 1e-6)), j0 + 1)
        if i0 < 0 or j0 < 0 or i1 > G or j1 > G:
            return -1.0
        cov[i0:i1, j0:j1] += 1
    return float((cov == 1).mean()) - 1.5 * float((cov > 1).mean()) \
        - 0.5 * float((cov == 0).mean())


def _normalise_boxes(raw):
    """Scale (x0, y0, w, h) boxes into [0,1]; return (boxes, base_grid)."""
    b = np.asarray(raw, dtype=float)
    if b.ndim != 2 or b.shape[1] != 4 or not np.isfinite(b).all():
        return None, None
    if (b[:, 2] <= 0).any() or (b[:, 3] <= 0).any() or (b[:, :2] < -1e-6).any():
        return None, None
    span = max(float((b[:, 0] + b[:, 2]).max()), float((b[:, 1] + b[:, 3]).max()))
    if span <= 0:
        return None, None
    if span > 1.5:                              # integer base-grid units
        b = b / float(np.rint(span))
    if b.max() > 1.001:
        return None, None
    smallest = max(float(np.minimum(b[:, 2], b[:, 3]).min()), 1e-9)
    return b, max(int(np.rint(1.0 / smallest)), 1)


def _leaf_boxes(arrays, n_leaves, verbose=True):
    """
    Recover the leaf squares as (n_leaves, 4) = (x0, y0, w, h) in [0,1].

    Two independent encodings are tried and the one that tiles the UV domain
    best is kept: the stored `leaf_bbox` table, and the polygon corners
    (`leaf_poly_ids` indexing the global `vertex_uv`).
    """
    candidates = []

    bb = arrays.get('leaf_bbox')
    if bb is not None and bb.ndim == 2 and bb.shape[0] == n_leaves and bb.shape[1] >= 4:
        a = bb[:, :4].astype(float)
        candidates.append(('leaf_bbox as (x0,y0,x1,y1)',
                           np.stack([a[:, 0], a[:, 1],
                                     a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]], axis=1)))
        candidates.append(('leaf_bbox as (x0,y0,w,h)', a.copy()))

    ids, vuv, lk = (arrays.get('leaf_poly_ids'), arrays.get('vertex_uv'),
                    arrays.get('leaf_k'))
    if ids is not None and vuv is not None and ids.shape[0] == n_leaves:
        rows, ok = [], True
        for i in range(n_leaves):
            k = int(lk[i]) if lk is not None else ids.shape[1]
            k = int(np.clip(k, 3, ids.shape[1]))
            jj = ids[i, :k].astype(np.int64)
            jj = jj[(jj >= 0) & (jj < len(vuv))]
            if len(jj) < 3:
                ok = False
                break
            P = vuv[jj].astype(float)
            mn, mx = P.min(0), P.max(0)
            rows.append([mn[0], mn[1], mx[0] - mn[0], mx[1] - mn[1]])
        if ok:
            candidates.append(('polygon corners in vertex_uv', np.asarray(rows, float)))

    best = (None, None, -1e9, '')
    for desc, raw in candidates:
        boxes, grid = _normalise_boxes(raw)
        if boxes is None:
            continue
        score = _tiling_score(boxes, grid)
        if score > best[2]:
            best = (boxes, grid, score, desc)

    boxes, grid, score, desc = best
    if boxes is None:
        if verbose:
            print("  [color] WARNING: no usable leaf-box encoding found; "
                  "colours will fall back to leaf order.")
        return None, None
    if verbose:
        flag = '' if score >= 0.9 else '  (imperfect tiling!)'
        print(f"  [color] leaf boxes from '{desc}' "
              f"(base grid={grid}, tiling score={score:.3f}){flag}")
    return boxes, grid


# ── adjacency + greedy colouring ─────────────────────────────────────────────
def _leaf_adjacency(boxes, grid, include_diagonal=True):
    """Neighbour graph of the leaf squares (edge- and corner-touching)."""
    G = int(min(max(grid, 1), 512))
    owner = defaultdict(list)
    for lid, (x0, y0, w, h) in enumerate(boxes):
        i0, j0 = int(np.floor(x0 * G + 1e-6)), int(np.floor(y0 * G + 1e-6))
        i1 = max(int(np.ceil((x0 + w) * G - 1e-6)), i0 + 1)
        j1 = max(int(np.ceil((y0 + h) * G - 1e-6)), j0 + 1)
        for a in range(max(i0, 0), min(i1, G)):
            for b in range(max(j0, 0), min(j1, G)):
                owner[(a, b)].append(lid)

    adj = defaultdict(set)

    def link(p, q):
        if p != q:
            adj[p].add(q)
            adj[q].add(p)

    offsets = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if include_diagonal else [])
    for (a, b), here in owner.items():
        for p in here:                       # overlapping leaves also conflict
            for q in here:
                link(p, q)
        for da, db in offsets:
            for q in owner.get((a + da, b + db), ()):
                for p in here:
                    link(p, q)
    return adj


def _greedy_coloring(adjacency, n_nodes):
    """Welsh–Powell: highest degree first, smallest unused colour."""
    order = sorted(range(n_nodes), key=lambda i: (-len(adjacency.get(i, ())), i))
    colors = {}
    for node in order:
        used = {colors[nb] for nb in adjacency.get(node, ()) if nb in colors}
        c = 0
        while c in used:
            c += 1
        colors[node] = c
    return colors


def _count_conflicts(adjacency, colors):
    return sum(1 for node, nbs in adjacency.items() for nb in nbs
               if nb > node and colors.get(node) == colors.get(nb))


# ── sampling inactive leaves: guard unlock + calibrated direct sampler ───────
@contextmanager
def _unlock_all_leaves(F):
    """
    Temporarily neutralise the model's 'inactive leaf' guard.

    The checkpoint already carries reconstructed polygons for the empty leaves
    ("Augmented N leaves with reconstructed polygons"), so interpolation is well
    defined for them; only explicit count/active flags reject the call. Every
    touched attribute is restored on exit.
    """
    saved, touched = [], []

    def patch(obj, attr, new):
        try:
            old = getattr(obj, attr)
        except Exception:
            return
        try:
            setattr(obj, attr, new)
        except Exception:
            return
        saved.append((obj, attr, old))
        touched.append(f"{type(obj).__name__}.{attr}")

    n = int(getattr(F, 'n_patches', 0))
    targets = [F]
    for a in ('complex', 'topology', 'tree', 'quadtree'):
        obj = getattr(F, a, None)
        if obj is not None and not isinstance(obj, dict):
            targets.append(obj)

    count_names = ('leaf_count', 'leaf_counts', 'counts', 'count', 'leaf_n',
                   'n_points', 'point_count', 'points_per_leaf')
    mask_names = ('leaf_active', 'active_mask', 'is_active', 'active_leaf_mask',
                  'leaf_mask', 'occupied', 'is_empty', 'empty_mask', 'is_inactive')
    id_names = ('active_patch_ids', 'active_leaf_ids', 'active_ids',
                'active_leaves', 'active_set')

    for obj in targets:
        for name in count_names:
            cur = getattr(obj, name, None)
            if torch.is_tensor(cur) and cur.dtype != torch.bool:
                patch(obj, name, torch.clamp(cur, min=1))
            elif isinstance(cur, np.ndarray) and cur.dtype != bool:
                patch(obj, name, np.maximum(cur, 1))

        for name in mask_names:
            cur = getattr(obj, name, None)
            negative = ('empty' in name) or ('inactive' in name)
            if torch.is_tensor(cur):
                patch(obj, name,
                      torch.zeros_like(cur) if negative else torch.ones_like(cur))
            elif isinstance(cur, np.ndarray):
                patch(obj, name,
                      np.zeros_like(cur) if negative else np.ones_like(cur))

        if n:
            for name in id_names:
                cur = getattr(obj, name, None)
                if isinstance(cur, set):
                    patch(obj, name, set(range(n)))
                elif isinstance(cur, (list, tuple)):
                    patch(obj, name, type(cur)(range(n)))
                elif torch.is_tensor(cur) and cur.ndim == 1:
                    patch(obj, name, torch.arange(n, device=cur.device, dtype=cur.dtype))
                elif isinstance(cur, np.ndarray) and cur.ndim == 1:
                    patch(obj, name, np.arange(n, dtype=cur.dtype))

    if touched:
        print(f"  [color] inactive-leaf guard unlocked via: "
              f"{', '.join(sorted(set(touched)))}")
    try:
        yield
    finally:
        for obj, attr, old in reversed(saved):
            try:
                setattr(obj, attr, old)
            except Exception:
                pass


def _polygon_weights(P, Q, scheme, eps=1e-9):
    """
    Generalised barycentric weights of query points Q (N,2) w.r.t. polygon P (K,2).

    scheme: 'wachspress' (= bilinear on a square) or 'meanvalue' (Floater).
    Queries are nudged infinitesimally inside the polygon so that samples lying
    exactly on an edge or a corner stay numerically well behaved.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    K = len(P)
    centre = P.mean(axis=0)
    Q = centre + (Q - centre) * (1.0 - 1e-6)

    ip1 = (np.arange(K) + 1) % K
    im1 = (np.arange(K) - 1) % K

    D = Q[:, None, :] - P[None, :, :]                        # (N,K,2)
    r = np.maximum(np.linalg.norm(D, axis=2), eps)           # (N,K)
    cross = D[:, :, 0] * D[:, ip1, 1] - D[:, :, 1] * D[:, ip1, 0]
    dot = (D * D[:, ip1, :]).sum(axis=2)

    if scheme == 'wachspress':
        Pm, Pp = P[im1], P[ip1]
        C = ((P[:, 0] - Pm[:, 0]) * (Pp[:, 1] - Pm[:, 1])
             - (P[:, 1] - Pm[:, 1]) * (Pp[:, 0] - Pm[:, 0]))
        den = cross[:, im1] * cross
        den = np.where(np.abs(den) < eps, eps, den)
        W = C[None, :] / den
    else:                                                    # mean value
        t = cross / (r * r[:, ip1] + dot + eps)              # tan(alpha_i / 2)
        W = (t[:, im1] + t) / r

    s = W.sum(axis=1, keepdims=True)
    s = np.where(np.abs(s) < eps, eps, s)
    return (W / s).astype(np.float32)


class _DirectLeafSampler:
    """
    Fallback sampler that rebuilds a leaf from the checkpoint topology and the
    model's own decoder, so leaves the model refuses to interpolate can still be
    exported. It is calibrated (and only used) if it reproduces the model's own
    output on an active leaf.
    """

    def __init__(self, F, arrays, device):
        self.F = F
        self.device = device
        self.ready = False
        self.scheme = None
        self.assemble = None

        self.ids = arrays.get('leaf_poly_ids')
        self.puv = arrays.get('leaf_poly_uv')
        self.k = arrays.get('leaf_k')
        self.decoder = getattr(F, 'decoder', None)

        cx = getattr(F, 'complex', None)
        vf = getattr(cx, 'vertex_features', None) if cx is not None else None
        if vf is None:
            vf = arrays.get('vertex_features')
        if isinstance(vf, np.ndarray):
            vf = torch.as_tensor(vf)
        self.vf = vf.to(device).float() if torch.is_tensor(vf) else None

    # ---- input assembly candidates -----------------------------------------
    def _assemblers(self):
        d_feat = int(self.vf.shape[1])
        in_dim = None
        lins = getattr(self.decoder, 'lins', None)
        if lins is not None and len(lins):
            in_dim = int(lins[0].weight.shape[1])
        if in_dim is None or in_dim == d_feat:
            return [('features only', lambda feat, uv: feat)]

        pe = None
        for nm in ('pe', 'pos_enc', 'positional_encoding', 'encoding', 'embed', 'enc'):
            cand = getattr(self.F, nm, None)
            if callable(cand):
                pe = cand
                break
        if pe is None:
            return []
        return [('pe ⊕ features', lambda feat, uv: torch.cat([pe(uv), feat], dim=-1)),
                ('features ⊕ pe', lambda feat, uv: torch.cat([feat, pe(uv)], dim=-1))]

    def _features(self, cid, uv_np, scheme):
        k = int(self.k[cid]) if self.k is not None else self.ids.shape[1]
        k = int(np.clip(k, 3, self.ids.shape[1]))
        jj = self.ids[cid, :k].astype(np.int64)
        valid = (jj >= 0) & (jj < int(self.vf.shape[0]))
        jj, P = jj[valid], self.puv[cid, :k][valid]
        if len(jj) < 3:
            raise ValueError(f"leaf {cid} has no usable polygon")
        W = torch.as_tensor(_polygon_weights(P, uv_np, scheme), device=self.device)
        return W @ self.vf[torch.as_tensor(jj, device=self.device)]

    @torch.no_grad()
    def _predict(self, cid, uv_np, scheme, assemble):
        feat = self._features(cid, uv_np, scheme)
        uv = torch.as_tensor(uv_np, dtype=torch.float32, device=self.device)
        out = self.decoder(assemble(feat, uv))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out[:, :3]

    # ---- calibration --------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, reference_leaf, tol=1e-3, res=9):
        if (self.ids is None or self.puv is None or self.vf is None
                or self.decoder is None or reference_leaf is None):
            return False

        g = torch.linspace(0, 1, res, device=self.device)
        gu, gv = torch.meshgrid(g, g, indexing='ij')
        uv = torch.stack([gu.flatten(), gv.flatten()], dim=-1)
        uv_np = uv.cpu().numpy().astype(np.float32)
        try:
            target = self.F(int(reference_leaf), uv)[:, :3]
        except Exception:
            return False

        best = (None, None, np.inf)
        for scheme in ('wachspress', 'meanvalue'):
            for label, assemble in self._assemblers():
                try:
                    pred = self._predict(int(reference_leaf), uv_np, scheme, assemble)
                    err = float((pred - target).abs().max())
                except Exception:
                    continue
                if err < best[2]:
                    best = ((scheme, label, assemble), None, err)

        if best[0] is None or best[2] > tol:
            if best[0] is not None:
                print(f"  [color] direct sampler rejected "
                      f"(best mismatch {best[2]:.3e} > {tol:.1e})")
            return False

        self.scheme, label, self.assemble = best[0]
        self.ready = True
        print(f"  [color] direct sampler calibrated: {self.scheme} + {label} "
              f"(max error {best[2]:.2e} on leaf {int(reference_leaf)})")
        return True

    # ---- public sampling ----------------------------------------------------
    @torch.no_grad()
    def sample(self, cid, resolution):
        g = torch.linspace(0, 1, resolution, device=self.device)
        gu, gv = torch.meshgrid(g, g, indexing='ij')
        uv = torch.stack([gu.flatten(), gv.flatten()], dim=-1)
        uv_np = uv.cpu().numpy().astype(np.float32)

        verts = []
        for i in range(0, len(uv_np), 4096):
            chunk = uv_np[i:i + 4096]
            verts.append(self._predict(cid, chunk, self.scheme, self.assemble).cpu())
        verts = torch.cat(verts, dim=0).numpy().astype(np.float32)

        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                a = i * resolution + j
                b = (i + 1) * resolution + j
                c = i * resolution + (j + 1)
                d = (i + 1) * resolution + (j + 1)
                faces.append([a, b, d])
                faces.append([a, d, c])
        return verts, uv_np, np.asarray(faces, dtype=np.int32)


# ── writers ─────────────────────────────────────────────────────────────────
def _write_obj_vertex_colors(path, verts, faces, colors_u8):
    """OBJ with inline vertex colours ('v x y z r g b'), so no .mtl is needed."""
    with open(path, 'w') as fh:
        fh.write("# per-patch colour mesh: 'v x y z r g b' (vertex colours)\n")
        block = np.concatenate([verts.astype(np.float64),
                                colors_u8.astype(np.float64) / 255.0], axis=1)
        np.savetxt(fh, block, fmt='v %.6f %.6f %.6f %.4f %.4f %.4f')
        np.savetxt(fh, faces.astype(np.int64) + 1, fmt='f %d %d %d')


def _save_quadtree_color_jpg(save_path, boxes, leaf_rgb, active_ids,
                             n_colors, title_suffix=''):
    """Two panels: every leaf coloured, and the same colours with inactive blank."""
    n = len(leaf_rgb)
    rgb01 = leaf_rgb.astype(np.float32) / 255.0
    active = set(int(i) for i in active_ids)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def draw(ax, only_active):
        if boxes is None:
            ax.text(0.5, 0.5, 'leaf boxes unavailable', ha='center', va='center')
        else:
            smallest = float(np.minimum(boxes[:, 2], boxes[:, 3]).min())
            for lid in range(n):
                x0, y0, w, h = boxes[lid]
                blank = only_active and lid not in active
                ax.add_patch(plt.Rectangle(
                    (x0, y0), w, h,
                    facecolor='white' if blank else rgb01[lid],
                    edgecolor='black', linewidth=0.45, zorder=2))
                if not blank and min(w, h) >= max(3.0 * smallest, 1.0 / 40):
                    ax.text(x0 + w / 2, y0 + h / 2, str(lid), ha='center',
                            va='center', fontsize=5.5, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('u')
        ax.set_ylabel('v')

    draw(axes[0], only_active=False)
    axes[0].set_title(f'All quadtree leaves ({n}) — colours of the exported mesh')
    draw(axes[1], only_active=True)
    axes[1].set_title(f'Active leaves only ({len(active)}) — inactive left blank')

    fig.suptitle(f'Quadtree patch colouring | leaves={n} | active={len(active)} | '
                 f'colours={n_colors}{title_suffix}', fontsize=13)
    if not save_path.lower().endswith(('.jpg', '.jpeg')):
        save_path = os.path.splitext(save_path)[0] + '.jpg'
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pil_kwargs={'quality': 95})
    plt.close(fig)
    print(f"    Quadtree colour figure (JPG) → {save_path}")


# ── the single entry point of the colour module ──────────────────────────────
@torch.no_grad()
def export_patch_colors_quadtree(F, meta, save_dir, topology,
                                 resolution=100, device='cuda',
                                 name='checkpoint', unnormalize=True,
                                 double_sided=True, corner_adjacency=True):
    """
    Colour every quadtree leaf so that no two touching leaves share a colour,
    sample ALL leaves (active and inactive) and write exactly three files:
    <name>_patch_colors.ply, <name>_patch_colors.obj, <name>_quadtree_colors.jpg.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_leaves = int(F.n_patches)
    active_ids = [int(i) for i in getattr(F, 'active_patch_ids', range(n_leaves))]

    arrays = _topology_arrays(F, topology)

    # 1) leaf squares → adjacency → colours
    boxes, grid = _leaf_boxes(arrays, n_leaves)
    if boxes is not None:
        adjacency = _leaf_adjacency(boxes, grid, include_diagonal=corner_adjacency)
        color_index = _greedy_coloring(adjacency, n_leaves)
        n_edges = sum(len(v) for v in adjacency.values()) // 2
        n_colors = max(color_index.values()) + 1
        print(f"  [color] adjacency: {n_edges} edges → {n_colors} colours, "
              f"conflicts = {_count_conflicts(adjacency, color_index)}")
    else:
        color_index = {i: i for i in range(n_leaves)}
        n_colors = max(n_leaves, 1)

    palette = _color_palette(n_colors)
    leaf_rgb = np.array([palette[color_index.get(i, 0) % len(palette)]
                         for i in range(n_leaves)], dtype=np.uint8)

    # 2) sample every leaf, unlocking the inactive-leaf guard
    print(f"  [color] sampling all {n_leaves} leaves "
          f"({len(active_ids)} active) @ resolution {resolution}")
    direct = _DirectLeafSampler(F, arrays, device)
    calibrated = False
    verts_by_leaf, faces_ref, uv_ref, failed = {}, None, None, []

    with _unlock_all_leaves(F):
        for cid in range(n_leaves):
            try:
                v, uv, f = _sample_patch_grid(F, cid, resolution, device)
            except Exception as exc:
                if not calibrated:
                    calibrated = True
                    direct.calibrate(active_ids[0] if active_ids else None)
                if direct.ready:
                    try:
                        v, uv, f = direct.sample(cid, resolution)
                    except Exception as exc2:
                        failed.append((cid, f"{exc} / direct: {exc2}"))
                        continue
                else:
                    failed.append((cid, str(exc)))
                    continue
            if v.shape[0] == 0 or not np.isfinite(v).all():
                failed.append((cid, 'empty or non-finite samples'))
                continue
            verts_by_leaf[cid] = v * meta['scale'] + meta['center'] if unnormalize else v
            faces_ref, uv_ref = f, uv

    if failed:
        shown = [c for c, _ in failed[:12]]
        print(f"  [color] {len(failed)} leaf/leaves not sampled {shown}"
              f"{' …' if len(failed) > 12 else ''}; first reason: {failed[0][1]}")
    if not verts_by_leaf:
        print("  [color] nothing sampled; colour export aborted.")
        return None
    n_inactive_done = len(set(verts_by_leaf) - set(active_ids))
    print(f"  [color] sampled {len(verts_by_leaf)}/{n_leaves} leaves "
          f"({n_inactive_done} inactive included)")

    # 3) one combined vertex-coloured mesh
    V_all, F_all, C_all, offset = [], [], [], 0
    for cid, verts_out in verts_by_leaf.items():
        v_mesh, _, f_mesh = (_make_double_sided(verts_out, uv_ref, faces_ref)
                             if double_sided else (verts_out, uv_ref, faces_ref))
        V_all.append(v_mesh.astype(np.float32))
        F_all.append(np.asarray(f_mesh, dtype=np.int64) + offset)
        C_all.append(np.tile(np.append(leaf_rgb[cid], 255).astype(np.uint8),
                             (len(v_mesh), 1)))
        offset += len(v_mesh)

    V = np.concatenate(V_all, axis=0)
    Fc = np.concatenate(F_all, axis=0)
    C = np.concatenate(C_all, axis=0)

    stem = f"{name}_patch_colors"
    ply_path = os.path.join(save_dir, stem + '.ply')
    obj_path = os.path.join(save_dir, stem + '.obj')

    trimesh.Trimesh(vertices=V, faces=Fc, vertex_colors=C,
                    process=False, maintain_order=True).export(ply_path)
    print(f"    Per-patch COLOUR mesh (PLY) → {ply_path}  "
          f"[{len(V)} verts, {len(Fc)} faces]")
    _write_obj_vertex_colors(obj_path, V, Fc, C[:, :3])
    print(f"    Per-patch COLOUR mesh (OBJ) → {obj_path}")

    # 4) the figure
    _save_quadtree_color_jpg(
        os.path.join(save_dir, f"{name}_quadtree_colors.jpg"),
        boxes, leaf_rgb, active_ids, n_colors, title_suffix=f" | {name}")

    return {'leaf_rgb': leaf_rgb, 'boxes': boxes, 'color_index': color_index}


# ─────────────────────────────────────────────────────────────────────────────
# Model module loader
# ─────────────────────────────────────────────────────────────────────────────
def _import_model_module(model_path=None):
    """Load the model module without relying on the current working directory."""
    if model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(here)
        candidates = [
            os.path.join(repo_root, 'model', 'model.py'),
            os.path.join(repo_root, 'model'),
            os.path.join(here, 'model.py'),
            os.path.join(here, '..', 'model.py'),
            os.path.join(os.getcwd(), 'model', 'model.py'),
            os.path.join(os.getcwd(), 'model'),
            os.path.join(os.getcwd(), 'model.py'),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Could not locate the model module automatically. Pass its location "
            "explicitly with --model_path /path/to/model/model.py."
        )

    model_path = os.path.abspath(model_path)
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'model.py')
    spec = importlib.util.spec_from_file_location("_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"  Loaded model classes from: {model_path}")
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loaders
# ─────────────────────────────────────────────────────────────────────────────
def _load_model_from_checkpoint(ckpt_path, device, model_path=None):
    """LEGACY loader (old grid-based MultiPatchForwardMap signature)."""
    model_module = _import_model_module(model_path)
    MultiPatchForwardMap = getattr(model_module, 'MultiPatchForwardMap', None)
    if MultiPatchForwardMap is None:
        raise AttributeError("model module does not define MultiPatchForwardMap")

    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get('mode') not in ('multi_patch', 'multi_patch_pretrain_flat_sheet'):
        raise ValueError(
            f"Checkpoint mode is '{ckpt.get('mode')}', expected 'multi_patch' "
            f"or 'multi_patch_pretrain_flat_sheet'."
        )

    args = ckpt['args']
    n_rows, n_cols = ckpt['grid_dims']

    F = MultiPatchForwardMap(
        n_rows=n_rows, n_cols=n_cols,
        d_features=args['d_features'],
        L=args['L'], W=args['W'], D=args['D'], beta=args['beta'],
    ).to(device)
    F.load_state_dict(ckpt['F_state'])
    F.eval()

    normalization = ckpt.get('normalization')
    if normalization is None:
        meta = {'center': np.zeros(3, dtype=np.float32), 'scale': 1.0}
    else:
        meta = {
            'center': np.array(normalization['center'], dtype=np.float32),
            'scale': float(normalization['scale']),
        }
    active_patch_ids = ckpt.get('active_patch_ids')
    return F, meta, args, active_patch_ids, ckpt.get('grid_dims')


def _infer_forward_map_hparams(F_state):
    """
    Recover (d_features, L, W, D) from a MultiPatchForwardMap state dict so we
    can rebuild it without needing the full training args.

    Layout (from model.py):
        complex.vertex_features : (n_vertices, d_features)
        decoder.lins.{i}.weight : first layer in-dim = d_features + 4*L
        decoder head             : SkipMLP.head
    """
    d_features = int(F_state['complex.vertex_features'].shape[1])

    lin_keys = sorted(
        [k for k in F_state if k.startswith('decoder.lins.') and k.endswith('.weight')],
        key=lambda k: int(k.split('.')[2]),
    )
    if not lin_keys:
        raise ValueError("Could not find decoder.lins.* in checkpoint state dict.")

    D = len(lin_keys)
    W = int(F_state[lin_keys[0]].shape[0])
    first_in = int(F_state[lin_keys[0]].shape[1])
    d_pe = first_in - d_features           # = 2 * 2 * L  (PositionalEncoding(2, L))
    L = max(0, d_pe // 4)
    return d_features, L, W, D


def _load_quadtree_model_from_checkpoint(ckpt_path, device, model_path=None,
                                         beta_cli=100.0):
    """
    NEW loader: rebuild the quadtree MultiPatchForwardMap from `topology`,
    exactly the model main.py trained/saved.
    """
    model_module = _import_model_module(model_path)
    MultiPatchForwardMap = getattr(model_module, 'MultiPatchForwardMap', None)
    if MultiPatchForwardMap is None:
        raise AttributeError("model module does not define MultiPatchForwardMap")

    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get('mode') not in ('multi_patch', 'multi_patch_pretrain_flat_sheet'):
        raise ValueError(
            f"Checkpoint mode is '{ckpt.get('mode')}', expected 'multi_patch' "
            f"or 'multi_patch_pretrain_flat_sheet'."
        )

    topology = ckpt.get('topology')
    if topology is None:
        raise ValueError(
            "Quadtree mode requires a checkpoint saved with 'topology' "
            "(F.topology_state()). This checkpoint has none."
        )

    F_state = ckpt['F_state']
    d_features, L, W, D = _infer_forward_map_hparams(F_state)

    ckpt_args = ckpt.get('args', {}) or {}
    # beta is a Softplus hyperparameter (not stored in weights); take it from
    # the checkpoint args if present, else from the CLI/default.
    beta = float(ckpt_args.get('beta', beta_cli))

    print(f"  Rebuilding quadtree model: d_features={d_features}, L={L}, "
          f"W={W}, D={D}, beta={beta}")

    F = MultiPatchForwardMap(topology, d_features, L=L, W=W, D=D, beta=beta).to(device)
    F.load_state_dict(F_state)
    F.eval()

    normalization = ckpt.get('normalization')
    if normalization is None:
        meta = {'center': np.zeros(3, dtype=np.float32), 'scale': 1.0}
    else:
        meta = {
            'center': np.array(normalization['center'], dtype=np.float32),
            'scale': float(normalization['scale']),
        }

    return F, meta, ckpt_args, ckpt.get('active_patch_ids'), topology


def _resolve_checkpoint_paths(ckpt_path):
    """Return a sorted list of checkpoint files from a file or directory."""
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isdir(ckpt_path):
        candidates = glob.glob(os.path.join(ckpt_path, 'checkpoint*.pt'))

        def _sort_key(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            suffix = stem.replace('checkpoint_', '')
            digits = ''.join(ch for ch in suffix if ch.isdigit())
            return (0, int(digits)) if digits else (1, stem)

        return sorted(candidates, key=_sort_key)
    return [ckpt_path]


def _resegment_current_input_points(input_points, n_rows, n_cols, active_patch_ids=None):
    """Legacy helper for occupancy masks (non-quadtree mode)."""
    assignments, _, _ = pc_presegmentation.pca_grid_segmentation(
        input_points, n_patches_u=n_rows, n_patches_v=n_cols
    )
    if active_patch_ids is None:
        active_patch_ids = sorted(np.unique(assignments).tolist())
    patch_points = {}
    for cid in active_patch_ids:
        patch_points[int(cid)] = input_points[assignments == int(cid)]
    return patch_points


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Export checkerboard-textured per-patch meshes from a '
                    'trained multi-patch checkpoint.')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint.pt, or a directory of checkpoint*.pt')
    parser.add_argument('--texture_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'texture'),
                        help='Single checkerboard image, OR a directory of Slide1..SlideN images')
    parser.add_argument('--out_dir', type=str, default='checkerboard_export')
    parser.add_argument('--resolution', type=int, default=100,
                        help='Per-patch UV grid resolution')
    parser.add_argument('--n_images', type=int, default=1)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_unnormalize', action='store_true',
                        help='Keep vertices in normalized [-1,1]^3 space')
    parser.add_argument('--no_ply', action='store_true',
                        help='Skip vertex-colored .ply export (OBJ only)')
    parser.add_argument('--single_sided', action='store_true',
                        help='Disable double-sided geometry')
    parser.add_argument('--model_path', '--main_path', dest='model_path', type=str, default=None,
                        help='Explicit path to model/model.py or the model directory')

    # ── QUADTREE MODE ────────────────────────────────────────────────────────
    parser.add_argument('--quadtree', action='store_true',
                        help='Use the quadtree code path: rebuild the model from '
                             'the checkpoint topology and sample ONLY the model\'s '
                             'active leaves (exactly like main.py). '
                             '--subdivision_depth / --input_file are ignored.')
    parser.add_argument('--beta', type=float, default=100.0,
                        help='[Quadtree] Softplus beta used to rebuild the model if '
                             'the checkpoint does not store it in args (default 100).')

    # ── SAVE CONTROL (both modes) ────────────────────────────────────────────
    parser.add_argument('--no_save_each_patches', action='store_true',
                        help='Do NOT write per-patch OBJ/PLY files; only save the '
                             'final combined mesh.')

    # ── LEGACY-ONLY ARGS (ignored when --quadtree) ───────────────────────────
    parser.add_argument('--input_file', type=str, default=None,
                        help='[Legacy] Point cloud used to build occupancy masks')
    parser.add_argument('--subdivision_depth', type=int, default=1,
                        help='[Legacy] UV occupancy subdivision depth (ignored in --quadtree)')
    parser.add_argument('--min_points_per_cell', type=int, default=10,
                        help='[Legacy] Min points for a subdivided UV cell to be kept')
    args = parser.parse_args()

    ckpt_paths = _resolve_checkpoint_paths(args.ckpt)
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found at: {args.ckpt}")
    if len(ckpt_paths) > 1:
        print(f"  Found {len(ckpt_paths)} checkpoint files in {os.path.abspath(args.ckpt)}")

    save_each_patch = not args.no_save_each_patches

    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        export_dir = os.path.join(args.out_dir, ckpt_name)

        if args.quadtree:
            # ── quadtree path ────────────────────────────────────────────────
            F, meta, ckpt_args, _, topology = _load_quadtree_model_from_checkpoint(
                ckpt_path, args.device, args.model_path, beta_cli=args.beta
            )
            print(f"  Loaded quadtree model: {F.n_patches} leaves total, "
                  f"{len(F.active_patch_ids)} active")
            print(f"  Exporting checkpoint {ckpt_name} → {export_dir}")

            export_checkerboard_patches_quadtree(
                F, meta,
                save_dir=export_dir,
                texture_path=args.texture_path,
                resolution=args.resolution,
                device=args.device,
                epoch=ckpt_name,
                name=ckpt_name,
                n_images=args.n_images,
                unnormalize=not args.no_unnormalize,
                export_ply=not args.no_ply,
                double_sided=not args.single_sided,
                save_each_patch=save_each_patch,
            )

            # ── extra colour module: always exactly 3 more files ─────────────
            export_patch_colors_quadtree(
                F, meta,
                save_dir=export_dir,
                topology=topology,
                resolution=args.resolution,
                device=args.device,
                name=ckpt_name,
                unnormalize=not args.no_unnormalize,
                double_sided=not args.single_sided,
            )
        else:
            # ── CURRENT (legacy) path, unchanged behaviour ───────────────────
            F, meta, ckpt_args, active_patch_ids, grid_dims = _load_model_from_checkpoint(
                ckpt_path, args.device, args.model_path
            )
            patch_points_by_id = None
            if active_patch_ids is not None and args.subdivision_depth > 0:
                input_file = args.input_file or ckpt_args.get('file')
                if input_file is None:
                    raise ValueError('Occupancy-aware export requires --input_file or checkpoint args[file].')
                input_points, _ = utils.load_point_cloud(input_file)
                patch_points_by_id = _resegment_current_input_points(
                    input_points, F.n_rows, F.n_cols, active_patch_ids=active_patch_ids
                )
            print(f"  Loaded model: {F.n_rows}x{F.n_cols} = {F.n_patches} patches")
            if active_patch_ids is not None:
                print(f"  Active patches in checkpoint: {len(active_patch_ids)}")
            print(f"  Exporting checkpoint {ckpt_name} → {export_dir}")

            export_checkerboard_patches(
                F, meta,
                save_dir=export_dir,
                texture_path=args.texture_path,
                resolution=args.resolution,
                device=args.device,
                epoch=ckpt_name,
                name=ckpt_name,
                n_images=args.n_images,
                unnormalize=not args.no_unnormalize,
                export_ply=not args.no_ply,
                double_sided=not args.single_sided,
                active_patch_ids=active_patch_ids,
                patch_points_by_id=patch_points_by_id,
                subdivision_depth=args.subdivision_depth,
                min_points_per_cell=args.min_points_per_cell,
                save_each_patch=save_each_patch,
            )

    print(f"\n  Done. Textured patches → {args.out_dir}")


if __name__ == '__main__':
    main()