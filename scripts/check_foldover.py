#!/usr/bin/env python3
"""Quick foldover check for trained atlases.

A foldover means two UV-distant parts of a patch sheet land on top of each
other in 3D.  This script samples every active patch on a coarse UV grid,
finds nearest neighbours in 3D, and reports the fraction of samples that have
a neighbour closer than ``delta`` while being farther than ``uv_separation``
away in UV space.  It needs only the checkpoint, no meshing and no
self-intersection tests, and runs in a few seconds for a few hundred patches.

Example:
    python scripts/check_foldover.py logs/render_phase10_ghpr_nosvd/armadillo_ghpr_nosvd
    python scripts/check_foldover.py run_a run_b --json foldover.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for path in (REPO_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from checkpoint_atlas import active_patch_ids, build_forward_model  # noqa: E402


def checkpoint_path(target: str) -> str:
    if target.endswith('.pt'):
        return target
    return os.path.join(target, 'checkpoint.pt')


def sample_patches(ckpt: str, grid: int, device: str):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    forward, info = build_forward_model(ck, device)
    ids = active_patch_ids(ck, forward)
    axis = torch.linspace(0.0, 1.0, grid, device=device)
    u, v = torch.meshgrid(axis, axis, indexing='ij')
    uv = torch.stack([u.flatten(), v.flatten()], dim=-1)
    points, patch, uvs = [], [], []
    with torch.no_grad():
        for slot, patch_id in enumerate(ids):
            xyz = forward(patch_id, uv).float().cpu().numpy()
            points.append(xyz)
            patch.append(np.full(len(xyz), slot, dtype=np.int32))
            uvs.append(uv.cpu().numpy())
    return (np.vstack(points), np.concatenate(patch), np.vstack(uvs),
            info, [int(i) for i in ids])


def foldover(target: str, grid: int, uv_separation: float,
             deltas, device: str):
    ckpt = checkpoint_path(target)
    t0 = time.time()
    P, patch, UV, info, ids = sample_patches(ckpt, grid, device)
    neighbours = min(25, len(P))
    dist, index = cKDTree(P).query(P, k=neighbours, workers=-1)
    same_patch = patch[index] == patch[:, None]
    duv = np.linalg.norm(UV[index] - UV[:, None, :], axis=-1)
    # True foldover: same leaf, far apart in UV, close in 3D.
    intra = same_patch & (duv > uv_separation)
    # Cross-leaf proximity is reported separately; it also fires on thin parts.
    inter = ~same_patch

    fractions = {}
    flags = {}
    for delta in deltas:
        near = dist < delta
        hit = (near & intra).any(axis=1)
        flags[delta] = hit
        fractions[str(delta)] = {
            'intra': float(hit.mean()),
            'inter': float((near & inter).any(axis=1).mean()),
        }

    # Scale-free view: how close is the nearest UV-distant sheet?
    nearest = np.where(intra, dist, np.inf).min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    percentiles = (
        {str(p): float(np.percentile(nearest, p)) for p in (1, 5, 25, 50)}
        if len(nearest) else {})

    per_patch = []
    for slot, patch_id in enumerate(ids):
        row = {'patch_id': patch_id}
        for delta in deltas:
            row[str(delta)] = float(flags[delta][patch == slot].mean())
        per_patch.append(row)
    per_patch.sort(key=lambda r: -r[str(deltas[0])])

    result = {
        'checkpoint': os.path.abspath(ckpt),
        'atlas_mode': info.get('atlas_mode'),
        'patches': len(ids),
        'grid': grid,
        'uv_separation': uv_separation,
        'deltas': list(deltas),
        'fraction': fractions,
        'nearest_intra_distance': percentiles,
        'worst_patches': per_patch[:10],
        'seconds': round(time.time() - t0, 2),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+',
                        help='run directory or checkpoint.pt')
    parser.add_argument('--grid', type=int, default=24,
                        help='UV samples per side (default 24)')
    parser.add_argument('--uv-separation', type=float, default=0.15,
                        help='minimum UV distance to count as folded (default 0.15)')
    parser.add_argument('--deltas', type=float, nargs='+',
                        default=[0.002, 0.004, 0.008],
                        help='3D proximity thresholds in normalized units')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--json', default=None)
    parser.add_argument('--fail-over', type=float, default=None,
                        help='exit 1 if the smallest-delta intra fraction '
                             'exceeds this value')
    args = parser.parse_args()

    results = []
    for target in args.runs:
        result = foldover(target, args.grid, args.uv_separation,
                          args.deltas, args.device)
        results.append(result)
        name = os.path.basename(os.path.normpath(target))
        frac = result['fraction']
        parts = '  '.join(
            f"d={d}: {frac[str(d)]['intra'] * 100:5.2f}%"
            for d in result['deltas'])
        pct = result['nearest_intra_distance']
        tail = (f"  nearest UV-distant p1/p5/p25={pct['1']:.4f}/"
                f"{pct['5']:.4f}/{pct['25']:.4f}" if pct else '')
        print(f"{name:36s} patches={result['patches']:4d}  "
              f"inter(d={result['deltas'][0]}): "
              f"{frac[str(result['deltas'][0])]['inter'] * 100:5.1f}%  "
              f"intra {parts}  [{result['seconds']}s]{tail}")

    if args.json:
        with open(args.json, 'w') as handle:
            json.dump(results, handle, indent=1)
        print(f"wrote {args.json}")

    if args.fail_over is not None:
        worst = max(r['fraction'][str(r['deltas'][0])]['intra']
                    for r in results)
        sys.exit(1 if worst > args.fail_over else 0)


if __name__ == '__main__':
    main()
