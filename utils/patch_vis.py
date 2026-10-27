#!/usr/bin/env python3
# patch_vis.py

"""
Checkerboard export for BOTH atlas families:

* fixed-grid checkpoints written by `main.py` (single_sheet / two_sheet /
  six_sheet), which additionally support point-cloud occupancy filtering via
  --subdivision_depth, and
* adaptive quadtree checkpoints written by `main.py --adaptive`
  (mode='adaptive_cube'), where each leaf is exported with its quadtree depth
  in the filename.

Which path runs is decided by --adaptive (default 'auto' = read the
checkpoint's own 'mode' field).

Notes:
- To change the pattern to several images, edit the argument `pattern` in `load_checkerboard_textures()` and `export_checkerboard_patches` to "Slide{}.jpg".
- To change the number of images, edit the argument --n_images in the Command line

"""
import os
import argparse
import importlib.util
import glob
import json
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

import pc_presegmentation as pc_presegmentation
import utils as utils

_HERE = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_HERE), _HERE, os.getcwd()):
    if os.path.isdir(os.path.join(_root, 'model')) and _root not in sys.path:
        sys.path.insert(0, _root)

from model.model import AdaptiveCubeForwardMap  # noqa: E402
from model.losses import surface_jacobian, tangent_loss_from_jac  # noqa: E402

# Load the texture
def load_checkerboard_textures(texture_path, pattern="Slide{}.jpg", n_images=4):
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
            print(f"  Loaded checkerboard texture: {p}")
        print(f"  Loaded {len(textures)} checkerboard texture(s) from {texture_path}")
        return textures

    raise FileNotFoundError(
        f"Checkerboard texture path not found: {texture_path}\n"
        f"  Pass either a single image file or a directory of numbered images."
    )



# Per-patch grid sampling.
@torch.no_grad()
def _sample_patch_grid(F, patch_idx, resolution, device):
    """
    Sample one patch of `MultiPatchForwardMap` on a regular UV grid.

    Args:
        F: Forward map in eval mode.
        patch_idx: Patch index.
        resolution: Grid resolution per side.
        device: Torch device.

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
    """
    Convert a textured mesh into a vertex-colored copy for PLY export.
    """
    colored = mesh.copy()
    colored.visual = mesh.visual.to_color()
    return colored


def _make_double_sided(verts, uv, faces):
    """
    Duplicate sheet geometry so it renders from both sides.

    Args:
        verts: Vertex positions.
        uv: UV coordinates matching `verts`.
        faces: Triangle indices.

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
    """Build an occupancy mask over a patch-local UV domain [0,1]^2."""
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
    """Keep only vertices/faces whose UVs lie in occupied cells."""
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


# Singular-value (SVD loss) diagnostics
def svd_loss_per_sample(t_u, t_v, mode, target, eps, chunk=65536):
    """
    Per-sample value of the SVD term, computed by the training function
    `tangent_loss_from_jac` itself, vmapped so each sample is its own batch of
    one. It therefore follows whatever equation losses.py holds for `mode`, and
    its mean equals what `tangent_loss_from_jac` returns for the whole batch
    (checked by `check_svd_decomposition`).

    `t_u`, `t_v` must already be in the wanted units (leaf-local or face-UV).
    """
    def one(tu, tv):
        return tangent_loss_from_jac(t_u=tu[None], t_v=tv[None], mode=mode,
                                     target=target, eps=eps)
    f = torch.func.vmap(one)
    return torch.cat([f(t_u[i:i + chunk], t_v[i:i + chunk])
                      for i in range(0, t_u.shape[0], chunk)])


def check_svd_decomposition(t_u, t_v, mode, target, eps, n=4096):
    """
    Raise if the per-sample values do not average to the batch loss, i.e. the
    mode couples samples (e.g. a batch-wide normalization) and cannot be split
    into a per-vertex field.
    """
    t_u, t_v = t_u[:n], t_v[:n]
    batch = float(tangent_loss_from_jac(t_u=t_u, t_v=t_v, mode=mode, target=target, eps=eps))
    per = float(svd_loss_per_sample(t_u, t_v, mode, target, eps).mean())
    if not np.isclose(per, batch, rtol=1e-4, atol=1e-7):
        raise RuntimeError(
            f"svd_mode '{mode}': mean of per-sample values {per:.6g} != batch loss "
            f"{batch:.6g}. This mode couples samples, so it has no per-vertex value.")


def _patch_jacobian(F, patch_idx, uv, device, normalize_patch_scale=False,
                    batch_size=4096):
    """
    Tangents t_u, t_v of F at the given UV samples, detached.

    Evaluated at the export mesh's own UVs, so row k belongs to vertex k. Units
    match `tangent_loss_from_jac`: the leaf's local UV, or face-UV when
    `normalize_patch_scale` (adaptive atlas only, divides by the leaf size, as
    the function's own normalize_patch_scale does).

    Returns:
        Tuple `(t_u, t_v)`, each (n, 3).
    """
    scale = 1.0
    if normalize_patch_scale and hasattr(F.complex, 'leaf_rect'):
        F.complex._sync_device()
        scale = float(F.complex.leaf_rect[patch_idx, 2])

    # Exactly on a leaf corner / edge, mvc_weights_torch switches to its one-hot
    # / linear branch, whose uv-gradient is zero / wrong across the edge. Pull
    # the grid's boundary samples a hair inside to get the limit value instead.
    uv_all = torch.as_tensor(uv, dtype=torch.float32, device=device).clamp(1e-4, 1.0 - 1e-4)
    tu_out, tv_out = [], []
    for i in range(0, uv_all.shape[0], batch_size):
        with torch.enable_grad():
            uv_b = uv_all[i:i + batch_size].clone().requires_grad_(True)
            Q = F(patch_idx, uv_b)
            # create_graph=True keeps the graph alive across the three
            # per-coordinate grads inside surface_jacobian.
            t_u, t_v = surface_jacobian(Q, uv_b, create_graph=True)
        tu_out.append(t_u.detach() / scale)
        tv_out.append(t_v.detach() / scale)
    return torch.cat(tu_out, dim=0), torch.cat(tv_out, dim=0)


def _patch_svd(F, patch_idx, uv, device, svd_cfg, normalize_patch_scale=False):
    """
    Singular values and per-sample SVD loss of one patch at the given UVs.

    Returns:
        Tuple `(sigma, loss)`: (n, 2) float32 with sigma1 >= sigma2, and (n,)
        float32 from `svd_loss_per_sample`.
    """
    t_u, t_v = _patch_jacobian(F, patch_idx, uv, device, normalize_patch_scale)
    with torch.no_grad():
        sigma = torch.linalg.svdvals(torch.stack([t_u, t_v], dim=2))
        loss = svd_loss_per_sample(t_u, t_v, **svd_cfg)
    return (sigma.cpu().numpy().astype(np.float32),
            loss.cpu().numpy().astype(np.float32))


def _svd_stats(sigma, loss):
    """Summary of one set of singular values and their SVD loss."""
    s1 = sigma[:, 0]
    return {
        'n_samples': int(s1.size),
        'loss_mean': float(loss.mean()),
        'frac_penalized': float((loss > 0).mean()),
        'sigma1_mean': float(s1.mean()),
        'sigma1_p50': float(np.percentile(s1, 50)),
        'sigma1_p90': float(np.percentile(s1, 90)),
        'sigma1_p99': float(np.percentile(s1, 99)),
        'sigma1_max': float(s1.max()),
        'sigma2_mean': float(sigma[:, 1].mean()),
        'area_mean': float((s1 * sigma[:, 1]).mean()),
    }


def _write_svd_ply(path, verts, faces, sigma, loss, patch, svd_cfg, color_by, color_max):
    """
    Binary PLY of the surface with per-vertex singular-value fields.

    The float properties svd_loss, sigma1, sigma2, area (= sigma1 * sigma2)
    and patch can be picked as scalar fields in CloudCompare / MeshLab. The
    colours show `color_by` ('loss' or 'sigma1') on the FIXED range
    [0, color_max] (blue = 0, white = color_max / 2, red = color_max or more),
    so exports of different checkpoints compare by eye as long as they share
    color_max.
    """
    import matplotlib
    values = loss if color_by == 'loss' else sigma[:, 0]
    ratio = np.clip(values / color_max, 0.0, 1.0)
    rgb = (matplotlib.colormaps['coolwarm'](ratio)[:, :3] * 255).round().astype(np.uint8)

    vdtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                       ('svd_loss', '<f4'), ('sigma1', '<f4'), ('sigma2', '<f4'),
                       ('area', '<f4'), ('patch', '<f4')])
    varr = np.empty(verts.shape[0], dtype=vdtype)
    varr['x'], varr['y'], varr['z'] = verts[:, 0], verts[:, 1], verts[:, 2]
    varr['red'], varr['green'], varr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    varr['svd_loss'] = loss
    varr['sigma1'], varr['sigma2'] = sigma[:, 0], sigma[:, 1]
    varr['area'] = sigma[:, 0] * sigma[:, 1]
    varr['patch'] = patch

    fdtype = np.dtype([('n', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')])
    farr = np.empty(faces.shape[0], dtype=fdtype)
    farr['n'] = 3
    farr['v0'], farr['v1'], farr['v2'] = faces[:, 0], faces[:, 1], faces[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment svd_mode {svd_cfg['mode']} svd_target {svd_cfg['target']} "
        f"color_by {color_by} color_max {color_max}\n"
        f"element vertex {verts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float svd_loss\nproperty float sigma1\nproperty float sigma2\n"
        "property float area\nproperty float patch\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(varr.tobytes())
        f.write(farr.tobytes())


def export_svd_diagnostics(parts, meta, save_dir, name, svd_cfg, units,
                           unnormalize=True, leaf_patches=None,
                           color_by='loss', color_max=None):
    """
    Write the singular-value PLY and svd_stats.json for one checkpoint.

    Args:
        parts: List of `(patch_id, verts, faces, sigma, loss)`, verts in the
            normalized frame, one entry per exported patch.
        svd_cfg: dict(mode, target, eps) the loss was computed with.
        color_by: 'loss' (svd_loss) or 'sigma1', the field the colours show.
        color_max: Top of the colour range (None = this checkpoint's p99 of
            the `color_by` field).

    Returns:
        The stats dict written to svd_stats.json.
    """
    per_patch = []
    for cid, _, _, sigma, loss in parts:
        row = {'patch': int(cid), **_svd_stats(sigma, loss)}
        if leaf_patches is not None:
            row.update({'face': int(leaf_patches[cid].face),
                        'depth': int(leaf_patches[cid].depth)})
        per_patch.append(row)

    offsets = np.cumsum([0] + [p[1].shape[0] for p in parts[:-1]])
    verts = np.concatenate([p[1] for p in parts], axis=0)
    faces = np.concatenate([p[2] + o for p, o in zip(parts, offsets)], axis=0)
    sigma = np.concatenate([p[3] for p in parts], axis=0)
    loss = np.concatenate([p[4] for p in parts], axis=0)
    patch = np.concatenate([np.full(p[1].shape[0], p[0], np.float32) for p in parts])
    if unnormalize:
        verts = verts * meta['scale'] + meta['center']
    if color_max is None:
        values = loss if color_by == 'loss' else sigma[:, 0]
        color_max = float(np.percentile(values, 99))
        if color_max <= 0:
            # A one-sided loss can be zero on 99% of the surface.
            color_max = float(values.max()) if values.max() > 0 else 1.0

    ply_path = os.path.join(save_dir, f"{name}_svd.ply")
    _write_svd_ply(ply_path, verts, faces, sigma, loss, patch, svd_cfg,
                   color_by, color_max)

    stats = {
        'svd_mode': svd_cfg['mode'],
        'svd_target': float(svd_cfg['target']),
        'svd_eps': float(svd_cfg['eps']),
        'units': units,
        'color_by': color_by,
        'color_max': float(color_max),
        'note': ('svd_loss comes from model.losses.tangent_loss_from_jac with the '
                 'mode/target/eps above. sigma is in the model\'s normalized frame '
                 '(not rescaled by unnormalize). loss_mean is the unweighted mean '
                 'over all UV samples, every patch sampled equally, the same '
                 'estimator as the SVD term in the training log.'),
        'overall': _svd_stats(sigma, loss),
        'per_patch': sorted(per_patch, key=lambda r: -r['loss_mean']),
    }
    with open(os.path.join(save_dir, 'svd_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    o = stats['overall']
    print(f"    SVD ({svd_cfg['mode']}, target={svd_cfg['target']:g}, {units} units): "
          f"loss mean={o['loss_mean']:.6f}  penalized on "
          f"{100 * o['frac_penalized']:.2f}% of samples  sigma1 p50/p90/max="
          f"{o['sigma1_p50']:.4f}/{o['sigma1_p90']:.4f}/{o['sigma1_max']:.4f}")
    print(f"    Singular-value PLY → {ply_path}")
    return stats


# Export functions
def export_checkerboard_patches(F, meta, save_dir, texture_path,
                                 resolution=100, device='cuda',
                                 epoch='10k', name=None,
                                 texture_pattern="Slide{}.jpg", n_images=4,
                                 unnormalize=True, debug_uv_png=True,
                                 export_obj=True, export_ply=True, double_sided=True,
                                 active_patch_ids=None,
                                 patch_points_by_id=None,
                                 subdivision_depth=1,
                                 min_points_per_cell=1,
                                 svd_cfg=None,
                                 svd_normalize=False,
                                 svd_color_by='loss',
                                 svd_color_max=None):
    """
    Export checkerboard-textured patch meshes and a combined scene.

    With `svd_cfg` = dict(mode, target, eps) set, also writes `<name>_svd.ply`
    (per-vertex singular values and SVD loss from `tangent_loss_from_jac`) and
    `svd_stats.json`.
    """
    os.makedirs(save_dir, exist_ok=True)
    textures = load_checkerboard_textures(texture_path, pattern=texture_pattern,
                                          n_images=n_images)
    n_textures = len(textures)

    # Adaptive (quadtree) atlases expose their leaf records; fixed grids don't.
    # When present, each patch's face + quadtree depth goes into its filename.
    leaf_patches = getattr(F.complex, 'leaf_patches', None)

    sample_all_patches = subdivision_depth < 0

    if active_patch_ids is None or sample_all_patches:
        patch_ids = list(range(F.n_patches))
    else:
        patch_ids = [int(patch_idx) for patch_idx in active_patch_ids]

    if len(patch_ids) == 0:
        print("  No active patches found; skipping checkerboard export.")
        return []

    meshes = []          # textured (OBJ) versions
    colored_meshes = []  # vertex-colored (PLY) versions
    svd_parts = []       # (patch_id, verts, faces, sigma, loss), single-sided
    n_exported = 0

    use_legacy_sampling = subdivision_depth <= 0 or patch_points_by_id is None

    if svd_cfg is not None:
        # The per-vertex field is only meaningful if the mode splits into
        # per-sample terms; verify on random samples of one patch.
        rand_uv = np.random.default_rng(0).random((4096, 2), dtype=np.float32)
        t_u, t_v = _patch_jacobian(F, patch_ids[0], rand_uv, device, svd_normalize)
        check_svd_decomposition(t_u, t_v, **svd_cfg)

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

        if svd_cfg is not None:
            # Before double-siding, so each vertex carries one value.
            sigma, loss = _patch_svd(F, cid, uv, device, svd_cfg, svd_normalize)
            svd_parts.append((cid, verts, faces, sigma, loss))

        if debug_uv_png:
            # Use the original UV layout before optional geometry duplication.
            _save_debug_uv_png(uv, tex_img, save_dir, cid)

        if double_sided:
            verts, uv, faces = _make_double_sided(verts, uv, faces)

        if unnormalize:
            verts_out = verts * meta['scale'] + meta['center']
        else:
            verts_out = verts

        uv_visuals = trimesh.visual.texture.TextureVisuals(uv=uv, image=tex_img)
        mesh = trimesh.Trimesh(vertices=verts_out, faces=faces,
                               visual=uv_visuals, process=False,
                               maintain_order=True)

        if leaf_patches is not None:
            leaf = leaf_patches[cid]
            tag = f"patch_{cid:03d}_d{leaf.depth}_{epoch}"
            label = f"Patch {cid:03d} (face={leaf.face}, depth={leaf.depth})"
        else:
            tag = f"patch_{cid}_{epoch}"
            label = f"Patch {cid:02d}"

        if export_obj:
            meshes.append(mesh)
            obj_path = os.path.join(save_dir, tag + ".obj")
            mesh.export(obj_path)
            print(f"    {label} textured OBJ → {obj_path}")
            n_exported += 1

        if export_ply:
            colored_mesh = _bake_vertex_colors(mesh)
            colored_meshes.append(colored_mesh)
            ply_path = os.path.join(save_dir, tag + ".ply")
            colored_mesh.export(ply_path)
            print(f"    {label} vertex-colored PLY → {ply_path}")
            n_exported += 1

        if debug_uv_png:
            _save_debug_uv_png(uv, tex_img, save_dir, cid)

    if n_exported == 0:
        print("  No meshes remained after occupancy filtering; skipping combined export.")
        return []

    # Combined OBJ scene with per-patch materials.
    if export_obj:
        scene = trimesh.Scene(meshes)
        obj_scene_name = f"{name}_checkerboard_{epoch}.obj" if name else f"checkerboard_{epoch}.obj"
        obj_scene_path = os.path.join(save_dir, obj_scene_name)
        scene.export(obj_scene_path)
        print(f"    Combined checkerboard scene (OBJ) → {obj_scene_path}")

    # Combined PLY as one merged vertex-colored mesh.
    if export_ply:
        combined_colored = trimesh.util.concatenate(colored_meshes)
        ply_scene_name = f"{name}_checkerboard_{epoch}.ply" if name else f"checkerboard_{epoch}.ply"
        ply_scene_path = os.path.join(save_dir, ply_scene_name)
        combined_colored.export(ply_scene_path)
        print(f"    Combined checkerboard scene (PLY) → {ply_scene_path}")

    if svd_parts:
        export_svd_diagnostics(
            svd_parts, meta, save_dir, name or 'checkerboard', svd_cfg,
            units='face-UV' if svd_normalize else 'leaf-local UV',
            unnormalize=unnormalize, leaf_patches=leaf_patches,
            color_by=svd_color_by, color_max=svd_color_max)

    return meshes


def _save_debug_uv_png(uv, tex_img, save_dir, cid):
    """
    Save a UV debug image with sampled points overlaid on the texture.

    Requires `cv2`. If unavailable, this function returns silently.
    """
    try:
        import cv2
    except ImportError:
        return

    w, h = tex_img.size
    img = np.array(tex_img.convert("RGB")).copy()  # (h, w, 3), RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pts_px = (uv * [w, h]).astype(np.int32)
    for px, py in pts_px[::max(1, len(pts_px) // 2000)]:
        cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)

    patch_dir = os.path.join(save_dir, str(cid))
    os.makedirs(patch_dir, exist_ok=True)
    out_path = os.path.join(patch_dir, "debug_uv.png")
    cv2.imwrite(out_path, img)


# Run against a saved checkpoint
def _import_model_module(model_path=None):
    """
    Load the model module without relying on the current working directory.

    Args:
        model_path: Optional explicit path to `model/model.py` or the model package directory.

    Returns:
        Loaded module object.
    """
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
            "Could not locate the model module automatically (checked the "
            "repo's model/ package, next to patch_vis.py, its parent folder, "
            "and the current working directory). Pass its location explicitly "
            "with --model_path /path/to/model/model.py (or model_path=... if "
            "calling from Python)."
        )

    model_path = os.path.abspath(model_path)
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'model.py')
    spec = importlib.util.spec_from_file_location("_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"  Loaded model classes from: {model_path}")
    return module


def _normalization_meta(ckpt):
    """Recover the {center, scale} export transform from a checkpoint."""
    normalization = ckpt.get('normalization')
    if normalization is None:
        # Pretrain-only checkpoints may omit normalization metadata.
        return {'center': np.zeros(3, dtype=np.float32), 'scale': 1.0}
    return {
        'center': np.array(normalization['center'], dtype=np.float32),
        'scale': float(normalization['scale']),
    }


def _load_adaptive_from_checkpoint(ckpt_path, device):
    """
    Rebuild an `AdaptiveCubeForwardMap` (quadtree cube atlas) from a checkpoint
    written by `main.py --adaptive`.

    Returns the same tuple shape as `_load_model_from_checkpoint`; the adaptive
    atlas has no fixed patch grid, so `active_patch_ids` and `grid_dims` are
    always None.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get('mode') != 'adaptive_cube':
        raise ValueError(
            f"Checkpoint mode is '{ckpt.get('mode')}', expected 'adaptive_cube'. "
            f"Pass --adaptive no (or leave it at 'auto') for fixed-grid checkpoints."
        )
    F = AdaptiveCubeForwardMap.from_checkpoint(ckpt, device=device)
    return F, _normalization_meta(ckpt), ckpt.get('args') or {}, None, None


def _load_model_from_checkpoint(ckpt_path, device, model_path=None):
    """
    Rebuild `MultiPatchForwardMap` from a saved checkpoint.

    Args:
        ckpt_path: Path to checkpoint file.
        device: Torch device string.
        model_path: Optional explicit path to `model.py`.
    """
    model_module = _import_model_module(model_path)
    MultiPatchForwardMap = getattr(model_module, 'MultiPatchForwardMap', None)
    TwoSheetForwardMap = getattr(model_module, 'TwoSheetForwardMap', None)
    SixSheetForwardMap = getattr(model_module, 'SixSheetForwardMap', None)
    if MultiPatchForwardMap is None and model_path is not None and os.path.basename(model_path) == 'main.py':
        sibling_model_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'model', 'model.py')
        if os.path.exists(sibling_model_path):
            model_module = _import_model_module(sibling_model_path)
            MultiPatchForwardMap = model_module.MultiPatchForwardMap
    if MultiPatchForwardMap is None and TwoSheetForwardMap is None and SixSheetForwardMap is None:
        raise AttributeError("model module does not define MultiPatchForwardMap, TwoSheetForwardMap, or SixSheetForwardMap")

    ckpt = torch.load(ckpt_path, map_location=device)
    supported_modes = {
        'multi_patch',
        'multi_patch_pretrain_flat_sheet',
        'multi_patch_pretrain_closed_shape',
    }
    if ckpt.get('mode') not in supported_modes:
        raise ValueError(
            f"Checkpoint mode is '{ckpt.get('mode')}', expected one of {sorted(supported_modes)}. "
            f"Checkerboard export only supports multi-patch checkpoints."
        )

    args = ckpt['args']
    n_rows, n_cols = ckpt['grid_dims']
    atlas_mode = ckpt.get('atlas_mode', args.get('atlas_mode', 'single_sheet'))

    if atlas_mode == 'two_sheet':
        if TwoSheetForwardMap is None:
            raise AttributeError("model module does not define TwoSheetForwardMap")
        F = TwoSheetForwardMap(
            n_rows=n_rows,
            n_cols=n_cols,
            d_features=args['d_features'],
            L=args['L'], W=args['W'], D=args['D'], beta=args['beta'],
        ).to(device)
    elif atlas_mode == 'six_sheet':
        if SixSheetForwardMap is None:
            raise AttributeError("model module does not define SixSheetForwardMap")
        F = SixSheetForwardMap(
            n_rows=n_rows,
            n_cols=n_cols,
            d_features=args['d_features'],
            L=args['L'], W=args['W'], D=args['D'], beta=args['beta'],
        ).to(device)
    else:
        if MultiPatchForwardMap is None:
            raise AttributeError("model module does not define MultiPatchForwardMap")
        F = MultiPatchForwardMap(
            n_rows=n_rows, n_cols=n_cols,
            d_features=args['d_features'],
            L=args['L'], W=args['W'], D=args['D'], beta=args['beta'],
        ).to(device)
    F.load_state_dict(ckpt['F_state'])
    F.eval()

    meta = _normalization_meta(ckpt)
    active_patch_ids = ckpt.get('active_patch_ids')
    return F, meta, args, active_patch_ids, ckpt.get('grid_dims')


def _is_adaptive_checkpoint(ckpt_path):
    """Peek at a checkpoint's 'mode' field to pick the export path."""
    return torch.load(ckpt_path, map_location='cpu').get('mode') == 'adaptive_cube'


def _resolve_checkpoint_paths(ckpt_path):
    """Return a sorted list of checkpoint files from a file or directory."""
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isdir(ckpt_path):
        # '*checkpoint*.pt' also catches 'pretrain_checkpoint.pt'.
        candidates = glob.glob(os.path.join(ckpt_path, '*checkpoint*.pt'))

        def _sort_key(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            suffix = stem.replace('checkpoint_', '')
            digits = ''.join(ch for ch in suffix if ch.isdigit())
            return (0, int(digits)) if digits else (1, stem)

        return sorted(candidates, key=_sort_key)
    return [ckpt_path]


def _build_patch_points_by_id(input_points: np.ndarray, assignments, active_patch_ids):
    assignments = np.asarray(assignments)
    patch_points = {}
    for cid in active_patch_ids:
        patch_points[int(cid)] = input_points[assignments == int(cid)]
    return patch_points


def _resegment_current_input_points(input_points: np.ndarray, n_rows: int, n_cols: int,
                                    active_patch_ids=None):
    """Use the current normalized input cloud directly to recover top-level patch groups."""
    assignments, _, _ = pc_presegmentation.pca_grid_segmentation(
        input_points, n_patches_u=n_rows, n_patches_v=n_cols
    )
    if active_patch_ids is None:
        active_patch_ids = sorted(np.unique(assignments).tolist())
    patch_points = {}
    for cid in active_patch_ids:
        patch_points[int(cid)] = input_points[assignments == int(cid)]
    return patch_points


def main():
    parser = argparse.ArgumentParser(
        description='Export checkerboard-textured per-patch meshes from a '
                   'trained multi-patch or adaptive-quadtree checkpoint.')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint.pt saved by main.py, or a directory containing checkpoint*.pt files')
    parser.add_argument('--adaptive', type=str, default='auto',
                        choices=['auto', 'yes', 'no'],
                        help="Which atlas the checkpoint holds: 'yes' = adaptive "
                             "quadtree cube atlas (main.py --adaptive), 'no' = "
                             "fixed patch grid, 'auto' (default) = decide per "
                             "checkpoint from its stored 'mode' field")
    parser.add_argument('--texture_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'texture'),
                        help='Path to a SINGLE checkerboard image (used for every '
                             'patch), OR a directory of Slide1.jpg...SlideN.jpg '
                             'images for a distinct texture per patch')
    parser.add_argument('--out_dir', type=str, default='checkerboard_export',
                        help='Output directory for textured OBJ/PLY files')
    parser.add_argument('--resolution', type=int, default=100,
                        help='Per-patch UV grid resolution')
    parser.add_argument('--n_images', type=int, default=4,
                        help='Number of checkerboard texture images to load '
                             '(only used if --texture_path is a directory)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_unnormalize', action='store_true',
                        help='Keep vertices in normalized [-1,1]^3 space '
                             '(default: export in original coordinates)')
    parser.add_argument('--no_obj', action='store_true',
                        help='Skip textured .obj export (PLY only)')
    parser.add_argument('--no_ply', action='store_true',
                        help='Skip vertex-colored .ply export (OBJ only)')
    parser.add_argument('--single_sided', action='store_true',
                        help='Disable double-sided geometry (default: patches '
                             'are duplicated so they render from both sides). '
                             'Pass this to keep the original single-layer sheet '
                             'and rely on viewer-side backface settings instead.')
    parser.add_argument('--model_path', '--main_path', dest='model_path', type=str, default=None,
                        help='Explicit path to model/model.py or the model directory (alias: --main_path). '
                            'Needed if patch_vis.py is not next to the model package and '
                            'not run from its folder. '
                            'e.g. /media/.../NISR_Seawall/model/model.py')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input point cloud used to build occupancy masks inside active patches')
    parser.add_argument('--subdivision_depth', type=int, default=1,
                        help='Recursive subdivision depth inside each active patch for UV occupancy filtering. '
                            'Use 0 to keep only active patches without subdivision, and -1 to sample all UVs '
                            'from every patch including empty/inactive ones.')
    parser.add_argument('--min_points_per_cell', type=int, default=10,
                        help='Minimum number of input points required for a subdivided UV cell to be kept')
    parser.add_argument('--svd_mode', type=str, default=None,
                        help="Mode passed to model.losses.tangent_loss_from_jac for the "
                             "svd_loss field (default: the checkpoint's own --svd_mode)")
    parser.add_argument('--svd_target', type=float, default=None,
                        help="'target' passed to tangent_loss_from_jac (default: the "
                             "checkpoint's own --svd_target)")
    parser.add_argument('--svd_color_by', type=str, default='loss',
                        choices=['loss', 'sigma1'],
                        help="Field the _svd.ply colours show: 'loss' = svd_loss from "
                             "tangent_loss_from_jac (default), 'sigma1' = largest "
                             "singular value. Every field stays in the PLY either way")
    parser.add_argument('--svd_color_max', type=float, default=None,
                        help='Value shown as full red in the _svd.ply. Default: the '
                             'FIRST checkpoint\'s p99 of the --svd_color_by field; every '
                             'later checkpoint reuses it so colours compare')
    parser.add_argument('--svd_units', type=str, default='local',
                        choices=['local', 'face'],
                        help="'local' = each leaf's own [0,1]^2 UV, as main.py currently "
                             "computes the SVD loss (normalize_patch_scale=False); "
                             "'face' = divided by leaf size, depth-independent "
                             "(adaptive only)")
    parser.add_argument('--no_svd', action='store_true',
                        help='Skip the singular-value PLY and svd_stats.json')
    args = parser.parse_args()

    ckpt_paths = _resolve_checkpoint_paths(args.ckpt)
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found at: {args.ckpt}")

    if len(ckpt_paths) > 1:
        print(f"  Found {len(ckpt_paths)} checkpoint files in {os.path.abspath(args.ckpt)}")

    svd_rows = []        # one summary row per checkpoint, for svd_summary.csv
    svd_color_max = args.svd_color_max
    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        export_dir = os.path.join(args.out_dir, ckpt_name)

        if args.adaptive == 'auto':
            is_adaptive = _is_adaptive_checkpoint(ckpt_path)
        else:
            is_adaptive = args.adaptive == 'yes'

        if is_adaptive:
            F, meta, ckpt_args, active_patch_ids, _ = _load_adaptive_from_checkpoint(
                ckpt_path, args.device
            )
            stats = F.complex.stats()
            print(f"  Loaded adaptive atlas: {F.n_patches} leaves, "
                  f"{stats['n_vertices']} vertices, depth hist {stats['depth_hist']}")
        else:
            F, meta, ckpt_args, active_patch_ids, _ = _load_model_from_checkpoint(
                ckpt_path, args.device, args.model_path
            )
            print(f"  Loaded model: {F.n_rows}x{F.n_cols} = {F.n_patches} patches")

        # Occupancy filtering re-segments the input cloud over the FIXED patch
        # grid, so it only applies to the non-adaptive atlas.
        patch_points_by_id = None
        if not is_adaptive and active_patch_ids is not None and args.subdivision_depth > 0:
            input_file = args.input_file or ckpt_args.get('file')
            if input_file is None:
                raise ValueError('Occupancy-aware export requires --input_file or checkpoint args[file].')
            input_points, _ = utils.load_point_cloud(input_file)
            patch_points_by_id = _resegment_current_input_points(
                input_points, F.n_rows, F.n_cols, active_patch_ids=active_patch_ids
            )
        if active_patch_ids is not None:
            print(f"  Active patches in checkpoint: {len(active_patch_ids)}")
        print(f"  Exporting checkpoint {ckpt_name} → {export_dir}")

        svd_cfg = None
        if not args.no_svd:
            mode = args.svd_mode or ckpt_args.get('svd_mode')
            if mode is None:
                print("  [warn] The checkpoint records no svd_mode (not an adaptive "
                      "run?) and --svd_mode was not given; skipping the SVD export.")
            else:
                target = (args.svd_target if args.svd_target is not None
                          else ckpt_args.get('svd_target', 1.0))
                svd_cfg = {'mode': mode, 'target': float(target),
                           'eps': float(ckpt_args.get('svd_eps', 1e-4))}
                if not ckpt_args.get('lam_svd'):
                    print(f"  [note] lam_svd was 0 in this run: svd_loss shows what "
                          f"'{mode}' WOULD be, the term was not trained.")
        stats_path = os.path.join(export_dir, 'svd_stats.json')
        if os.path.exists(stats_path):
            os.remove(stats_path)    # never summarize a stale file from an earlier run

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
            debug_uv_png=not is_adaptive,
            export_obj=not args.no_obj,
            export_ply=not args.no_ply,
            double_sided=not args.single_sided,
            active_patch_ids=active_patch_ids,
            patch_points_by_id=patch_points_by_id,
            subdivision_depth=args.subdivision_depth,
            min_points_per_cell=args.min_points_per_cell,
            svd_cfg=svd_cfg,
            svd_normalize=args.svd_units == 'face',
            svd_color_by=args.svd_color_by,
            svd_color_max=svd_color_max,
        )

        if svd_cfg is not None and os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            svd_color_max = stats['color_max']     # fixed for later checkpoints
            svd_rows.append((ckpt_name, F.n_patches, stats['overall']))

    # Same mode, target and units for every checkpoint of a run, so the rows
    # compare directly.
    if len(svd_rows) > 1:
        keys = ('loss_mean', 'frac_penalized', 'sigma1_p50', 'sigma1_p90',
                'sigma1_p99', 'sigma1_max', 'area_mean')
        csv_path = os.path.join(args.out_dir, 'svd_summary.csv')
        with open(csv_path, 'w') as f:
            f.write('checkpoint,n_patches,' + ','.join(keys) + '\n')
            for name, n, o in svd_rows:
                f.write(f"{name},{n}," + ','.join(f"{o[k]:.6g}" for k in keys) + '\n')
        print(f"\n  SVD loss per checkpoint:")
        print(f"    {'checkpoint':<24} {'patches':>7} {'loss mean':>11} "
              f"{'% penal.':>9} {'s1 p90':>8}")
        for name, n, o in svd_rows:
            print(f"    {name:<24} {n:>7} {o['loss_mean']:>11.6f} "
                  f"{100 * o['frac_penalized']:>8.2f}% {o['sigma1_p90']:>8.4f}")
        print(f"    → {csv_path}")

    print(f"\n  Done. Textured patches → {args.out_dir}")


if __name__ == '__main__':
    main()