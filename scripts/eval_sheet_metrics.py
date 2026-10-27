#!/usr/bin/env python3
"""Geometry diagnostics for a trained checkpoint.

Chamfer distance is *not* used in the NGF-objective runs, so it makes a clean
held-out metric there; runs that train on Chamfer are flagged by their stored
``lambda_chamfer``.  The script also reports cloud coverage, normal agreement
and per-patch UV stretch from the learned forward map.  The atlas layout
(fixed or adaptive quad-tree) is read from the checkpoint.

Example:
    python scripts/eval_sheet_metrics.py \
        --checkpoint logs/render_phase3/ab/g05_r2_ngf/checkpoint.pt \
        --out logs/render_phase3/metrics/g05_r2_ngf
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import utils.utils as nisr_utils
from checkpoint_atlas import (active_patch_ids, build_forward_model,
                              patch_layout_info)
from utils.render_mesh import sample_patch_mesh, vertex_normals


def _sample_uv_patches(F, patch_ids, cells, device):
    """Evaluate F on a (cells+1)^2 grid per patch; return (P, cells+1, cells+1, 3)."""
    u = torch.linspace(0.0, 1.0, cells + 1, device=device)
    gu, gv = torch.meshgrid(u, u, indexing='ij')
    uv = torch.stack([gu.reshape(-1), gv.reshape(-1)], dim=-1)
    out = []
    with torch.no_grad():
        for patch_id in patch_ids:
            pidx = torch.full((uv.shape[0],), int(patch_id), dtype=torch.long,
                              device=device)
            pts = F(pidx, uv)
            out.append(pts.reshape(cells + 1, cells + 1, 3))
    return torch.stack(out).cpu().numpy()


def _stretch(pts, cells):
    """Per-cell checker edge lengths, area scale and anisotropy.

    A cell is one square of an ``cells x cells`` checker laid on the patch's
    UV square.  Edge lengths are 3-D lengths at the cell's lower corner.
    """
    du = pts[:, 1:, :-1] - pts[:, :-1, :-1]          # (P, cells, cells, 3)
    dv = pts[:, :-1, 1:] - pts[:, :-1, :-1]
    E = np.sum(du * du, axis=-1)
    G = np.sum(dv * dv, axis=-1)
    Fdot = np.sum(du * dv, axis=-1)
    area = np.linalg.norm(np.cross(du, dv), axis=-1)
    # Singular values of [du dv] from the first fundamental form.
    tr = E + G
    disc = np.sqrt(np.maximum((E - G) ** 2 + 4.0 * Fdot ** 2, 0.0))
    s1 = np.sqrt(np.maximum(0.5 * (tr + disc), 0.0))
    s2 = np.sqrt(np.maximum(0.5 * (tr - disc), 0.0))
    aspect = s1 / np.maximum(s2, 1e-12)
    long_edge = np.maximum(np.sqrt(E), np.sqrt(G))
    return long_edge, area, aspect


def _best_layer_normals(verts, vert_normals, cloud, cloud_normals,
                        k=12, radius=0.03):
    """Normal agreement allowing for a nearby opposite layer.

    A closed six-face atlas wraps the scan, so the nearest mesh vertex to a
    cloud point can belong to the hidden back side at thin features (ears).
    Taking the best-matching vertex within ``radius`` measures the surface
    layer that is actually visible.
    """
    tree = cKDTree(verts)
    k = min(k, len(verts))
    dist, idx = tree.query(cloud, k=k)
    best = np.full(len(cloud), -2.0)
    best_d = np.full(len(cloud), np.inf)
    for column in range(k):
        d = dist[:, column]
        nearest = idx[:, column]
        ok = np.isfinite(d) & (d <= radius) & (nearest < len(verts))
        if not ok.any():
            continue
        dot = np.sum(cloud_normals[ok] * vert_normals[nearest[ok]], axis=1)
        take = dot > best[ok]
        rows = np.flatnonzero(ok)[take]
        best[rows] = dot[take]
        best_d[rows] = d[ok][take]

    found = np.isfinite(best_d)
    if not found.any():
        return {'found_fraction': 0.0}
    angles = np.degrees(np.arccos(np.abs(best[found]).clip(0.0, 1.0)))
    return {
        'found_fraction': float(found.mean()),
        'normal_angle_median_deg': float(np.median(angles)),
        'normal_angle_p95_deg': float(np.percentile(angles, 95)),
        'normal_signed_negative_fraction': float((best[found] < 0).mean()),
        'distance_median': float(np.median(best_d[found])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mesh-res', type=int, default=0,
                        help='UV samples per patch side for the Chamfer metric '
                             '(0 = auto: 96 fixed-grid, 16 adaptive)')
    parser.add_argument('--cells', type=int, default=8,
                        help='Checker cells per patch side for stretch stats')
    parser.add_argument('--near', type=float, default=0.02,
                        help='Cloud-to-sheet distance counted as covered')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ck_args = ck['args']

    pts3n, meta = nisr_utils.load_point_cloud(ck_args['file'])
    center = np.asarray(ck['normalization']['center'], dtype=np.float64)
    scale = float(ck['normalization']['scale'])
    print(f"  normalization check: center drift "
          f"{np.abs(meta['center'] - center).max():.2e}, "
          f"scale ratio {meta['scale'] / scale:.6f}")

    F, atlas_info = build_forward_model(ck, device)
    active_ids = active_patch_ids(ck, F)
    mesh_res = (args.mesh_res if args.mesh_res > 0 else
                (16 if atlas_info.get('adaptive') else 96))
    print(f"  atlas: {atlas_info['atlas_mode']}  faces {atlas_info['n_sides']}  "
          f"patches {atlas_info['actual_n_patches']}")

    # --- Chamfer / coverage at a resolution finer than the cloud spacing ---
    verts, faces = sample_patch_mesh(F, mesh_res, patch_ids=active_ids)
    verts_np = verts.detach().cpu().numpy()
    nrm_np = vertex_normals(verts, faces).detach().cpu().numpy()
    print(f"  metric mesh: {verts_np.shape[0]} verts / {faces.shape[0]} faces "
          f"(R={mesh_res})")

    cloud = np.asarray(pts3n, dtype=np.float64)
    cloud_nrm = np.asarray(meta['normals'], dtype=np.float64)
    sheet_tree = cKDTree(verts_np)
    cloud_tree = cKDTree(cloud)

    d_sheet_to_cloud, _ = cloud_tree.query(verts_np)
    d_cloud_to_sheet, nn_sheet = sheet_tree.query(cloud)
    chamfer = float(d_sheet_to_cloud.mean() + d_cloud_to_sheet.mean())

    covered = d_cloud_to_sheet < args.near
    cos = np.abs(np.sum(cloud_nrm * nrm_np[nn_sheet], axis=-1)).clip(0.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    signed = np.sum(cloud_nrm * nrm_np[nn_sheet], axis=-1)
    best_layer = _best_layer_normals(
        verts_np, nrm_np, cloud, cloud_nrm,
        radius=max(args.near, 0.03))

    # --- per-patch UV stretch on a cells x cells checker ---
    pts = _sample_uv_patches(F, active_ids, args.cells, device)
    long_edge, area, aspect = _stretch(pts, args.cells)
    patch_stats = {}
    for k, patch_id in enumerate(active_ids):
        layout = patch_layout_info(F, patch_id)
        entry = {
            'face': layout['face'],
            'row': layout['row'],
            'col': layout['col'],
            'long_edge_p50': float(np.percentile(long_edge[k], 50)),
            'long_edge_p95': float(np.percentile(long_edge[k], 95)),
            'long_edge_max': float(long_edge[k].max()),
            'aspect_p50': float(np.percentile(aspect[k], 50)),
            'aspect_p95': float(np.percentile(aspect[k], 95)),
            'aspect_max': float(aspect[k].max()),
            'area_scale_p50': float(np.percentile(area[k], 50)),
            'area_scale_min': float(area[k].min()),
            'area_scale_max': float(area[k].max()),
        }
        if layout['depth'] is not None:
            entry['depth'] = layout['depth']
        patch_stats[str(patch_id)] = entry

    summary = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'cloud': os.path.abspath(ck_args['file']),
        'atlas_mode': atlas_info['atlas_mode'],
        'atlas_faces': int(atlas_info['n_sides']),
        'points': int(cloud.shape[0]),
        'mesh_vertices': int(verts_np.shape[0]),
        'mesh_faces': int(faces.shape[0]),
        'mesh_resolution': int(mesh_res),
        'mesh_resolution_auto': bool(args.mesh_res <= 0),
        'active_patches': len(active_ids),
        'chamfer_symmetric': chamfer,
        'chamfer_sheet_to_cloud': float(d_sheet_to_cloud.mean()),
        'chamfer_cloud_to_sheet': float(d_cloud_to_sheet.mean()),
        'sheet_to_cloud_p95': float(np.percentile(d_sheet_to_cloud, 95)),
        'sheet_to_cloud_max': float(d_sheet_to_cloud.max()),
        'cloud_to_sheet_p95': float(np.percentile(d_cloud_to_sheet, 95)),
        'cloud_to_sheet_max': float(d_cloud_to_sheet.max()),
        'covered_fraction_near': float(covered.mean()),
        'near_threshold': float(args.near),
        'normal_angle_median_deg': float(np.median(angles)),
        'normal_angle_p95_deg': float(np.percentile(angles, 95)),
        'normal_signed_negative_fraction': float((signed < 0).mean()),
        'visible_layer_normal': best_layer,
        'long_edge_global_p95': float(np.percentile(long_edge, 95)),
        'long_edge_global_max': float(long_edge.max()),
        'aspect_global_p50': float(np.percentile(aspect, 50)),
        'aspect_global_p95': float(np.percentile(aspect, 95)),
        'aspect_global_max': float(aspect.max()),
        'per_patch': patch_stats,
        'args': {k: ck_args.get(k) for k in
                 ('lambda_chamfer', 'lambda_render', 'render_start_epoch',
                  'render_warmup_epochs', 'render_laplacian_weight', 'mu',
                  'gamma', 'lam_svd', 'lambda_outer_boundary', 'schedule')},
    }
    with open(os.path.join(args.out, 'metrics.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)

    print(f"  Chamfer sym {chamfer:.5f}  "
          f"(s->c {summary['chamfer_sheet_to_cloud']:.5f}, "
          f"c->s {summary['chamfer_cloud_to_sheet']:.5f})")
    print(f"  coverage <{args.near}: {covered.mean():.3f}  "
          f"normal angle median {np.median(angles):.2f} deg, "
          f"p95 {np.percentile(angles, 95):.2f}  "
          f"inverted {summary['normal_signed_negative_fraction']:.3f}")
    if 'normal_angle_median_deg' in best_layer:
        print(f"  visible layer: {best_layer['found_fraction']:.3f} of points "
              f"matched, normal angle median "
              f"{best_layer['normal_angle_median_deg']:.2f} deg, "
              f"p95 {best_layer['normal_angle_p95_deg']:.2f}  "
              f"inverted {best_layer['normal_signed_negative_fraction']:.3f}")
    print(f"  stretch p95/max {summary['long_edge_global_p95']:.3f}/"
          f"{summary['long_edge_global_max']:.3f}  "
          f"aspect p95/max {summary['aspect_global_p95']:.2f}/"
          f"{summary['aspect_global_max']:.2f}")
    print(f"  metrics -> {args.out}")


if __name__ == '__main__':
    main()
