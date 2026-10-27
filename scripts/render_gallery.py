#!/usr/bin/env python3
"""Render every orbit view of a trained checkpoint.

Builds the same point-cloud targets used for training, samples the learned
sheet mesh, renders it with ``NGFRenderer`` and writes target/prediction
contact sheets.  Read-only: the checkpoint is not modified.  The atlas layout
(fixed single/two/six-sheet or adaptive quad-tree) is read from the checkpoint.

Example:
    python scripts/render_gallery.py \
        --checkpoint logs/render_phase3/ab/g05_r1/checkpoint.pt \
        --out logs/render_phase3/gallery
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import utils.utils as nisr_utils
from checkpoint_atlas import active_patch_ids, build_forward_model
from model.ngf_rendering_loss import NGFRenderer
from utils.render_mesh import (auto_mesh_resolution, sample_patch_mesh,
                               vertex_normals)
from utils.render_targets import (build_render_targets, _label_uint8,
                                  _montage, _normal_shading)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--mesh-res', type=int, default=0,
                        help='UV samples per patch side (0 = auto)')
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

    F, info = build_forward_model(ck, device)
    active_ids = active_patch_ids(ck, F)
    print(f"  atlas: {info['atlas_mode']}  faces {info['n_sides']}  "
          f"patches {info['actual_n_patches']}")

    targets = build_render_targets(
        pts3n, meta['normals'], num_views=16, resolution=args.resolution,
        device=device, min_coverage=0.1, preview_dir=None, verbose=False)
    print(f"  targets: {len(targets)} views at {args.resolution}px, "
          f"coverage {float(targets.coverage.mean()):.3f}")

    mesh_res = (args.mesh_res if args.mesh_res > 0 else
                auto_mesh_resolution(
                    len(active_ids),
                    max_vertices=(120000 if info.get('adaptive') else 30000)))
    vertices, faces = sample_patch_mesh(F, mesh_res, patch_ids=active_ids)
    normals = vertex_normals(vertices, faces)
    print(f"  mesh: {vertices.shape[0]} verts / {faces.shape[0]} faces "
          f"(R={mesh_res})")

    renderer = NGFRenderer(width=args.resolution, height=args.resolution,
                           fov=targets.fov, device=device)
    with torch.no_grad():
        prediction = renderer.render(vertices, normals, faces,
                                     targets.views.to(device))
    prediction = prediction.detach().cpu()
    target_geometry = targets.geometry.detach().cpu()
    target_alpha = target_geometry[..., 6:7]
    pred_alpha = (prediction[..., :3].norm(dim=-1, keepdim=True) > 1e-6).float()

    # Normal agreement on pixels where both the target and the render are
    # covered: this is the angle a viewer sees, independent of which back
    # layer happens to be nearest in 3-D.
    pred_unit = prediction[..., :3] / prediction[..., :3].norm(
        dim=-1, keepdim=True).clamp_min(1e-8)
    target_unit = target_geometry[..., :3] / target_geometry[..., :3].norm(
        dim=-1, keepdim=True).clamp_min(1e-8)
    normal_cos = (pred_unit * target_unit).sum(-1)
    normal_mask = (target_alpha[..., 0] > 0.5) & (pred_alpha[..., 0] > 0.5)
    rendered_normal = {}
    if normal_mask.any():
        angles = torch.rad2deg(
            torch.arccos(normal_cos.abs().clamp(0.0, 1.0)))[normal_mask]
        rendered_normal = {
            'median_deg': float(angles.median()),
            'p95_deg': float(torch.quantile(angles, 0.95)),
            'signed_negative_fraction': float(
                (normal_cos[normal_mask] < 0).float().mean()),
            'pixels': int(normal_mask.sum()),
            'pixel_fraction': float(normal_mask.float().mean()),
        }

    # Per-view six-channel L1 over the full frame (matches the training loss).
    per_view = (prediction - target_geometry[..., :6]).abs().mean(dim=(1, 2, 3))
    channel_l1 = (prediction - target_geometry[..., :6]).abs().mean(dim=(0, 1, 2))

    view_dir = os.path.join(args.out, 'views')
    os.makedirs(view_dir, exist_ok=True)
    target_tiles, pred_tiles, pair_tiles = [], [], []
    for index in range(prediction.shape[0]):
        target_tile = _label_uint8(
            _normal_shading(target_geometry[index, ..., :3],
                            target_alpha[index]), f'v{index} target')
        pred_tile = _label_uint8(
            _normal_shading(prediction[index, ..., :3], pred_alpha[index]),
            f'v{index} pred  L1 {float(per_view[index]):.3f}')
        imageio.imwrite(os.path.join(view_dir, f'target_{index:04d}.png'),
                        target_tile)
        imageio.imwrite(os.path.join(view_dir, f'pred_{index:04d}.png'),
                        pred_tile)
        target_tiles.append(target_tile)
        pred_tiles.append(pred_tile)
        pair_tiles.extend([target_tile, pred_tile])

    # 8 columns x 4 rows: one target/prediction pair per view, left to right.
    imageio.imwrite(os.path.join(args.out, 'target_vs_pred_16.png'),
                    _montage(pair_tiles, cols=8))
    imageio.imwrite(os.path.join(args.out, 'target_16.png'),
                    _montage(target_tiles, cols=4))
    imageio.imwrite(os.path.join(args.out, 'prediction_16.png'),
                    _montage(pred_tiles, cols=4))

    summary = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'cloud': os.path.abspath(ck_args['file']),
        'atlas_mode': info['atlas_mode'],
        'atlas_faces': int(info['n_sides']),
        'points': int(pts3n.shape[0]),
        'views': int(prediction.shape[0]),
        'resolution': int(args.resolution),
        'mesh_vertices': int(vertices.shape[0]),
        'mesh_faces': int(faces.shape[0]),
        'mesh_resolution': int(mesh_res),
        'active_patches': len(active_ids),
        'per_view_full_l1': [float(v) for v in per_view],
        'mean_full_l1': float(per_view.mean()),
        'l1_normal': float(channel_l1[:3].mean()),
        'l1_position': float(channel_l1[3:6].mean()),
        'per_channel_l1': [float(v) for v in channel_l1],
        'rendered_normal': rendered_normal,
        'target_coverage': float(target_alpha.mean()),
        'prediction_coverage': float(pred_alpha.mean()),
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(f"  mean full L1 {summary['mean_full_l1']:.4f}  "
          f"min {float(per_view.min()):.4f}  max {float(per_view.max()):.4f}")
    if rendered_normal:
        print(f"  rendered normal angle median "
              f"{rendered_normal['median_deg']:.2f} deg, p95 "
              f"{rendered_normal['p95_deg']:.2f} deg  "
              f"inverted {rendered_normal['signed_negative_fraction']:.3f}")
    print(f"  gallery -> {args.out}")


if __name__ == '__main__':
    main()
