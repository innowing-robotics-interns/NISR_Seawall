#!/usr/bin/env python3
# utils/extension_mask.py

"""
In-training detection of extensions on the adaptive cube atlas.

Runs the uv_hole_mask -> opening_labels -> pair_opening chain on the live
model, in memory, and selects the openings that are UNPAIRED (no stitch can
close them) and SMALL (3D area below a fraction of the whole surface). Those
are taken to be extensions, e.g. a membrane bridging the gap between two ears,
and `main.py --ext_mask_epoch` treats them per `--ext_mode`: no regularizer
('no_reg') or no fitting loss ('no_fit').

The selection is rasterized per cube face and looked up in face-UV
coordinates, so it stays valid when later subdivision renumbers the leaves.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# The chain's scripts import each other as top-level modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from uv_hole_mask import (adaptive_leaf_rects, composite_face_canvases,  # noqa: E402
                          median_nn_spacing, write_colored_ply)
from opening_labels import label_openings, weld_boundary_vertices  # noqa: E402
from pair_opening import OUTCOMES, pair_openings, ray_mesh_triangles, sample_atlas  # noqa: E402

from model.model import FACE_NAMES  # noqa: E402


# Sample categories, also the colour index of the atlas figure.
CAT_SURFACE, CAT_PAIRED, CAT_KEPT, CAT_SELECTED = 0, 1, 2, 3
CAT_COLORS = np.array([(215, 215, 215),     # white: has a correspondence
                       (40, 140, 230),      # paired opening (a handle cap)
                       (240, 150, 30),      # unpaired, above the area threshold
                       (230, 30, 30)],      # unpaired and small: masked
                      dtype=np.uint8)
CAT_NAMES = ('surface', 'paired opening', 'unpaired, too large (kept)',
             'extension (masked)')


class ExtensionMask:
    """Per-cube-face raster of the UV area treated as an extension."""

    def __init__(self, canvas: torch.Tensor, report: dict):
        self.canvas = canvas                  # (6, res, res) bool, [face, u_index, v_index]
        self.res = int(canvas.shape[1])
        self.report = report

    @torch.no_grad()
    def excluded(self, model, pids: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """(B,) bool: True where a training sample lies in a selected opening."""
        cx = model.complex
        cx._sync_device()
        face = cx.leaf_face[pids]
        fuv = cx.face_uv(pids, uv.detach())
        idx = (fuv * self.res).long().clamp_(0, self.res - 1)
        return self.canvas[face, idx[:, 0], idx[:, 1]]


def _triangle_areas(xyz, tris, chunk=2_000_000):
    out = np.empty(tris.shape[0], np.float64)
    for i in range(0, tris.shape[0], chunk):
        t = tris[i:i + chunk]
        a = xyz[t[:, 0]].astype(np.float64)
        out[i:i + chunk] = 0.5 * np.linalg.norm(
            np.cross(xyz[t[:, 1]] - a, xyz[t[:, 2]] - a), axis=1)
    return out


def _save_atlas(path, cat_canvases, faces, title):
    """Six cube faces, every UV sample coloured by its category."""
    cmap = ListedColormap(CAT_COLORS.astype(np.float64) / 255.0)
    cmap.set_bad(color='0.95')
    # Explicit spacing: equal-aspect panels defeat tight_layout's row gap.
    fig, axes = plt.subplots(2, 3, figsize=(15, 12.5), squeeze=False,
                             gridspec_kw={'hspace': 0.12, 'wspace': 0.05})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
    for f in range(len(FACE_NAMES)):
        ax = axes[f // 3][f % 3]
        ax.imshow(np.ma.masked_invalid(cat_canvases[f]), cmap=cmap, vmin=-0.5,
                  vmax=len(CAT_COLORS) - 0.5, interpolation='nearest',
                  extent=[0, 1, 1, 0])
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'face {FACE_NAMES[f]}   ({int((faces == f).sum())} leaves)',
                     fontsize=10)
    fig.legend(handles=[Patch(facecolor=CAT_COLORS[i] / 255.0, label=n)
                        for i, n in enumerate(CAT_NAMES)],
               loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f'{title}   (u down, v right)', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def find_extension_openings(model, target_pts, device, resolution=64, nn_scale=10.0,
                            weld_tol=1e-5, min_opening_vertices=10,
                            max_area_frac=0.05, face_res=1024, out_dir=None,
                            batch_size=8192, pair_kwargs=None, verbose=True):
    """
    Detect the extensions of an adaptive atlas against its target cloud.

    Args:
        model: `AdaptiveCubeForwardMap`, in the frame of `target_pts`.
        target_pts: (N, 3) target cloud in the model's normalized frame.
        resolution: UV grid per leaf per side (uv_hole_mask --resolution).
        nn_scale: tau = nn_scale x the cloud's median NN spacing.
        weld_tol, min_opening_vertices: opening_labels.py settings.
        max_area_frac: an unpaired opening is selected only when its 3D area is
            at most this fraction of the whole surface's area.
        face_res: resolution per cube face of the returned raster.
        out_dir: if set, writes extension_atlas.png, extension_samples.ply,
            extension.json and extension_mask.npz there.
        pair_kwargs: overrides for `pair_openings` (pair_opening.py settings).

    Returns:
        `ExtensionMask`.
    """
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    cx = model.complex
    cx._sync_device()
    patch_ids = list(range(model.n_patches))
    P, R = len(patch_ids), resolution
    n_per = R * R

    # uv_hole_mask: distance from every UV sample to the target cloud
    xyz, nrm, _ = sample_atlas(model, patch_ids, R, device, batch_size)
    tree = cKDTree(target_pts)
    d_nn = median_nn_spacing(target_pts, tree)
    tau = nn_scale * d_nn
    dist, _ = tree.query(xyz, k=1, workers=-1)
    white_flat = dist <= tau

    # opening_labels: connected black components across the welded atlas
    xyz_fields = {p: xyz[i * n_per:(i + 1) * n_per] for i, p in enumerate(patch_ids)}
    masks = {p: white_flat[i * n_per:(i + 1) * n_per].reshape(R, R)
             for i, p in enumerate(patch_ids)}
    weld_pairs, _ = weld_boundary_vertices(xyz_fields, patch_ids, R, weld_tol)
    opening_id, n_openings, _ = label_openings(
        masks, xyz_fields, patch_ids, R, weld_pairs, min_opening_vertices)
    opening_flat = opening_id.reshape(-1).astype(np.int64)

    # pair_opening: which openings face a partner across a handle
    partner, top_outcome, faces_overlap = {}, {}, set()
    if n_openings >= 2:
        res = pair_openings(xyz, nrm, opening_flat, white_flat, n_openings, P, R,
                            device, verbose=False, **(pair_kwargs or {}))
        for p in res['pairs']:
            partner[p['a']], partner[p['b']] = p['b'], p['a']
        faces_overlap = res['overlap_front']
        counts = res['outcome_counts']
        for o in range(n_openings):
            top_outcome[o] = OUTCOMES[int(np.argmax(counts[o]))] if counts[o].sum() else None

    # Areas: 3D from the sample mesh (a triangle counts for an opening when
    # all three corners are in it), UV from each leaf's footprint on its face.
    tris = ray_mesh_triangles(P, R, 1)
    tri_area = _triangle_areas(xyz, tris)
    total_area = float(tri_area.sum())
    t_open = opening_flat[tris]
    inside = (t_open[:, 0] >= 0) & (t_open[:, 0] == t_open[:, 1]) & (t_open[:, 1] == t_open[:, 2])
    area = np.bincount(t_open[inside, 0], weights=tri_area[inside], minlength=n_openings)

    rects, faces, _ = adaptive_leaf_rects(model)
    sample_uv_area = np.repeat(rects[:, 2] ** 2 / n_per, n_per)        # per UV sample
    is_black = opening_flat >= 0
    uv_area = np.bincount(opening_flat[is_black], weights=sample_uv_area[is_black],
                          minlength=n_openings)
    n_samples = np.bincount(opening_flat[is_black], minlength=n_openings)

    openings, selected = [], []
    for o in range(n_openings):
        area_frac = float(area[o] / max(total_area, 1e-12))
        if o in partner:
            decision = f'paired with {partner[o]}'
        elif area_frac > max_area_frac:
            decision = f'unpaired, kept: area {area_frac:.4f} > {max_area_frac:g}'
        else:
            decision = 'EXTENSION: masked'
            selected.append(o)
        pts = xyz[opening_flat == o]
        openings.append({
            'id': o, 'n_samples': int(n_samples[o]),
            'area_3d': float(area[o]), 'area_frac': area_frac,
            'uv_area_frac': float(uv_area[o] / 6.0),
            'partner': partner.get(o), 'selected': o in selected,
            'top_ray_outcome': top_outcome.get(o),
            'faces_overlap': o in faces_overlap,
            'centroid': pts.mean(axis=0).tolist(),
            'extent': (pts.max(axis=0) - pts.min(axis=0)).tolist(),
            'decision': decision,
        })

    # Rasterize per cube face: the training lookup reads `selected`, the
    # figure reads all four categories.
    cat_flat = np.full(P * n_per, CAT_SURFACE, np.int64)
    cat_flat[is_black] = CAT_KEPT
    cat_flat[np.isin(opening_flat, list(partner))] = CAT_PAIRED
    cat_flat[np.isin(opening_flat, selected)] = CAT_SELECTED
    cat = cat_flat.reshape(P, R, R).astype(np.float32)
    cat_canvases = composite_face_canvases({p: cat[i] for i, p in enumerate(patch_ids)},
                                           patch_ids, rects, faces, face_res)
    canvas = np.stack([cat_canvases[f] == CAT_SELECTED for f in range(len(FACE_NAMES))])

    report = {
        'n_leaves': P, 'resolution': R, 'face_res': face_res,
        'tau': float(tau), 'median_nn': float(d_nn), 'nn_scale': nn_scale,
        'max_area_frac': max_area_frac, 'total_area_3d': total_area,
        'black_fraction': float(1.0 - white_flat.mean()),
        'n_openings': n_openings, 'n_pairs': len(partner) // 2,
        'selected': selected,
        'masked_uv_fraction': float(canvas.mean()),
        'openings': openings,
        'seconds': time.time() - t0,
    }

    log(f"  Extension detection: {n_openings} opening(s), {len(partner) // 2} pair(s), "
        f"{len(selected)} extension(s) selected  (tau={tau:.5f} = {nn_scale:g} x median NN, "
        f"area threshold {max_area_frac:g} of {total_area:.4f})  [{report['seconds']:.1f}s]")
    for o in openings:
        log(f"    opening {o['id']:>3}  {o['n_samples']:>7} samples  "
            f"area {o['area_frac']:.4f}  {o['decision']}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        _save_atlas(os.path.join(out_dir, 'extension_atlas.png'), cat_canvases, faces,
                    f"Extensions: {len(selected)} of {n_openings} opening(s) masked")
        black = np.flatnonzero(is_black)
        if black.size:
            write_colored_ply(os.path.join(out_dir, 'extension_samples.ply'),
                              xyz[black], CAT_COLORS[cat_flat[black]])
        np.savez_compressed(os.path.join(out_dir, 'extension_mask.npz'),
                            canvas=canvas, opening_id=opening_id,
                            white=white_flat.reshape(P, R, R),
                            selected=np.asarray(selected, np.int32),
                            leaf_rect=rects.astype(np.float32),
                            leaf_face=faces.astype(np.int32), tau=np.float32(tau))
        with open(os.path.join(out_dir, 'extension.json'), 'w') as f:
            json.dump(report, f, indent=2)
        log(f"    → {out_dir}")

    return ExtensionMask(torch.as_tensor(canvas, device=device), report)
