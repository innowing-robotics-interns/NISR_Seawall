#!/usr/bin/env python3
# utils/ghpr.py

"""
Generalized Hidden Point Removal (GHPR) for raw point clouds (need normals).

Katz & Tal, "On visibility and empty-region graphs", 2017.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
for _root in (_HERE, os.path.dirname(_HERE)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

GAMMA_SWEEP = (-1.0, -0.5, -0.2, -0.1, -0.05, -0.01, -0.001)


def _utils():
    """utils/utils.py, whether this module was imported as `ghpr` or `utils.ghpr`."""
    import importlib
    m = importlib.import_module('utils')
    return m if hasattr(m, 'load_point_cloud') else importlib.import_module('utils.utils')

def ghpr_transform(points, viewpoint, gamma):
    """Invert the cloud about `viewpoint`: direction kept, radius -> radius**gamma.

    Distances are divided by their median first. That only rescales the cloud
    about the viewpoint, which cannot change which points reach the hull, so
    gamma stays the operator's single scale-free parameter.
    """
    q = np.asarray(points, dtype=np.float64) - np.asarray(viewpoint, dtype=np.float64)
    d = np.linalg.norm(q, axis=1)

    out = np.zeros_like(q)
    nz = d > 0.0
    d_scaled = d[nz] / np.median(d[nz])
    out[nz] = q[nz] * (d_scaled ** (gamma - 1.0))[:, None]
    return out


def ghpr_visible(points, viewpoint, gamma=-0.01):
    """Boolean mask of the points visible from `viewpoint`."""
    points = np.asarray(points, dtype=np.float64)
    p_hat = ghpr_transform(points, viewpoint, gamma)

    # A point is visible iff its image lands on the hull of {p_hat} U {viewpoint}.
    hull = ConvexHull(np.vstack([p_hat, np.zeros(3)]))

    visible = np.zeros(len(points), dtype=bool)
    visible[np.linalg.norm(points - viewpoint, axis=1) == 0.0] = True
    on_hull = hull.vertices
    visible[on_hull[on_hull < len(points)]] = True
    return visible


# ── mesh-free viewpoint selection ───────────────────────────────────────────
def orient_normals_outward(points, normals):
    """Flip the whole normal field if it points inward on average."""
    radial = points - points.mean(axis=0)
    return -normals if float((radial * normals).sum()) < 0.0 else normals


def inside_fraction(p, tree, points, normals, k=16):
    """Fraction of the k nearest surface points that put `p` on their inner side.

    Mesh-free stand-in for a watertight inside/outside test: if `p` is inside,
    the vector from a surface point to `p` opposes that point's outward normal.
    """
    _, idx = tree.query(p, k=k)
    to_p = p - points[idx]
    return float((np.einsum('ij,ij->i', to_p, normals[idx]) < 0.0).mean())


def pick_viewpoint(points, normals=None, k=16, shift=0.02, max_steps=10):
    """Interior viewpoint from the cloud alone.

    Starts at the centroid. If the cloud carries normals and the centroid tests
    as outside — the usual case for a torus-like shape, whose centre of mass
    sits in the hole — steps inward from the nearest surface point until it is.

    Returns:
        Tuple `(viewpoint, inside_score, how)`; `inside_score` is None when the
        cloud has no normals and the choice could not be verified.
    """
    centroid = points.mean(axis=0)
    if normals is None:
        return centroid, None, 'centroid (unverified: cloud has no normals)'

    tree = cKDTree(points)
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    score = inside_fraction(centroid, tree, points, normals, k)
    if score > 0.5:
        return centroid, score, 'centroid'

    _, nearest = tree.query(centroid, k=1)
    for step in range(1, max_steps + 1):
        cand = points[nearest] - normals[nearest] * (shift * step * diag)
        cand_score = inside_fraction(cand, tree, points, normals, k)
        if cand_score > 0.5:
            return cand, cand_score, f'stepped inward from surface (depth {step})'
    return centroid, score, 'centroid (fallback: no interior point found)'


def inscribed_radius(points, viewpoint):
    """Distance from `viewpoint` to its nearest point — the largest box that fits inside."""
    return float(np.linalg.norm(points - np.asarray(viewpoint), axis=1).min())


def ghpr_from_points(points, normals=None, gamma=-0.01, k=16, viewpoint=None):
    """GHPR on an already-loaded cloud, no file I/O and no re-normalization.

    Entry point for the in-run warm start: `main.py` calls this on the same
    normalized cloud it trains on, so both phases share one coordinate frame.

    Returns:
        dict with 'visible' (bool mask), 'viewpoint', 'radius', 'normals'
        (outward-oriented), 'inside_score', 'source', 'visible_fraction'.
    """
    points = np.asarray(points, np.float64)
    if normals is not None:
        normals = orient_normals_outward(points, np.asarray(normals, np.float64))

    if viewpoint is not None:
        vp = np.asarray(viewpoint, np.float64).reshape(3)
        score, how = None, 'given by the caller'
    else:
        vp, score, how = pick_viewpoint(points, normals, k=k)

    visible = ghpr_visible(points, vp, gamma)
    return {
        'visible': visible,
        'viewpoint': vp,
        'radius': inscribed_radius(points[visible], vp),
        'normals': normals,
        'inside_score': score,
        'source': how,
        'visible_fraction': float(visible.mean()),
    }


# ── connectivity filter ─────────────────────────────────────────────────────
def median_nn_spacing(points):
    """Median distance from each point to its nearest neighbour."""
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


def _groups(points, mask, radius):
    """Connected groups of the masked points, linking pairs closer than `radius`.

    Returns:
        Tuple `(idx, labels, n_groups)`: `idx` are the masked point indices and
        `labels[i]` is the group of point `idx[i]`.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx, np.zeros(0, np.int64), 0
    pairs = cKDTree(points[idx]).query_pairs(radius, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])),
                       shape=(idx.size, idx.size))
    n_groups, labels = connected_components(graph, directed=False)
    return idx, labels, n_groups


def keep_connected(points, masks, radius):
    """Reduce each visible mask of a GHPR stage sequence to one connected group.

    GHPR visibility is not connected: parts of the surface seen through a gap
    come back as islands away from the main visible region. Chamfer then drags
    the surface across the empty gap toward them. Keeping one group per stage
    avoids that; the dropped points return in a later stage or the full cloud.

    Stage 1 keeps its largest group. A later stage keeps the group holding the
    previous stage's kept points, so the targets still only grow. A group that
    was connected stays connected when points are added, so that group always
    exists when the raw masks are nested; otherwise the most-overlapping group
    is kept and `contains_previous` reports the shortfall.

    Returns:
        Tuple `(kept_masks, stats)`, one entry per input mask.
    """
    points = np.asarray(points, np.float64)
    kept, stats, prev = [], [], None
    for mask in masks:
        mask = np.asarray(mask, dtype=bool)
        idx, labels, n_groups = _groups(points, mask, radius)
        keep = np.zeros(len(points), dtype=bool)
        contains_prev = None
        if n_groups:
            sizes = np.bincount(labels, minlength=n_groups)
            pick = int(np.argmax(sizes))
            if prev is not None and prev.any():
                overlap = np.bincount(labels[prev[idx]], minlength=n_groups)
                if overlap.max() > 0:
                    pick = int(np.argmax(overlap))
                contains_prev = float(overlap[pick] / prev.sum())
            keep[idx[labels == pick]] = True
        kept.append(keep)
        stats.append({
            'n_visible_raw': int(mask.sum()),
            'n_groups': int(n_groups),
            'n_kept': int(keep.sum()),
            'n_dropped': int(mask.sum() - keep.sum()),
            'kept_fraction_of_visible': float(keep.sum() / max(mask.sum(), 1)),
            'contains_previous': contains_prev,
        })
        prev = keep
    return kept, stats


# ── output ──────────────────────────────────────────────────────────────────
def write_classified_ply(path, points, visible, viewpoint, dropped=None):
    """Binary PLY: visible red, hidden grey, viewpoint black as the last vertex.

    `dropped` optionally marks visible points removed by `keep_connected`;
    they are drawn orange.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    xyz = np.vstack([np.asarray(points, np.float32),
                     np.asarray(viewpoint, np.float32).reshape(1, 3)])
    rgb = np.empty((xyz.shape[0], 3), np.uint8)
    rgb[:-1] = np.where(np.asarray(visible)[:, None],
                        np.array([220, 50, 47], np.uint8),
                        np.array([170, 170, 170], np.uint8))
    if dropped is not None:
        rgb[:-1][np.asarray(dropped, dtype=bool)] = (240, 150, 30)
    rgb[-1] = (0, 0, 0)

    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr = np.empty(xyz.shape[0], dtype=dtype)
    arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment viewpoint {viewpoint[0]:.8f} {viewpoint[1]:.8f} {viewpoint[2]:.8f}\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(arr.tobytes())
    print(f"    Classified PLY → {path}")


# ── driver ──────────────────────────────────────────────────────────────────
def run(file, gamma=-0.01, out_dir=None, downsample=None, normalized=False,
        k=16, viewpoint=None, sweep=True):
    """Load a cloud, pick a viewpoint, keep the visible points, write them out."""
    utils = _utils()
    print(f"\n  Loading: {file}")
    pts, meta = utils.load_point_cloud(file, downsample_n=downsample)
    pts = pts.astype(np.float64)
    center = np.asarray(meta['center'], np.float64)
    scale = float(meta['scale'])

    # Everything below lives in the normalized frame, like `pts`.
    res = ghpr_from_points(pts, meta.get('normals'), gamma=gamma, k=k,
                           viewpoint=viewpoint)
    normals, vp, score, how = (res['normals'], res['viewpoint'],
                               res['inside_score'], res['source'])

    stem = os.path.splitext(os.path.basename(file))[0]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(file)), 'ghpr', stem)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─' * 64}")
    print(f"  Viewpoint (normalized): {np.round(vp, 5)}   [{how}]")
    print(f"  Viewpoint (original)  : {np.round(vp * scale + center, 5)}")
    if score is not None:
        print(f"  Interior score        : {score:.2f}  "
              f"({'inside' if score > 0.5 else 'OUTSIDE — results unreliable'})")
    else:
        print(f"  Interior score        : n/a (no normals to test against)")
    print(f"{'─' * 64}")

    sweep_fracs = None
    if sweep:
        sweep_fracs = {str(g): float(ghpr_visible(pts, vp, g).mean()) for g in GAMMA_SWEEP}
        print("  Visible fraction vs gamma (pick a value on a plateau):")
        for g in GAMMA_SWEEP:
            mark = '  <- selected' if abs(g - gamma) < 1e-12 else ''
            print(f"    gamma = {g:>7}   {100.0 * sweep_fracs[str(g)]:6.2f}% visible{mark}")

    visible = res['visible']
    n_vis = int(visible.sum())
    print(f"\n  gamma = {gamma}:  {n_vis:,} of {len(pts):,} points visible "
          f"({100.0 * visible.mean():.2f}%)")
    print(f"  Inscribed box radius  : {res['radius']:.5f} (normalized units)")

    # Default to the input file's own coordinates so the visible cloud overlays
    # the source and re-normalizes the same way when main.py loads it.
    frame = 'normalized' if normalized else 'original'
    out_pts = pts[visible] if normalized else pts[visible] * scale + center
    # Normals are invariant under the uniform scale + translation.
    out_nrm = normals[visible] if normals is not None else None

    visible_path = os.path.join(out_dir, f'{stem}_visible.ply')
    utils.export_point_cloud_ply(out_pts, visible_path, normals=out_nrm)
    write_classified_ply(os.path.join(out_dir, 'ghpr_classified.ply'),
                         pts if normalized else pts * scale + center,
                         visible, vp if normalized else vp * scale + center)

    summary = {
        'input_file': os.path.abspath(file),
        'gamma': gamma,
        'n_input': int(len(pts)),
        'n_visible': n_vis,
        'visible_fraction': float(visible.mean()),
        'viewpoint_normalized': vp.tolist(),
        'viewpoint_original': (vp * scale + center).tolist(),
        'viewpoint_source': how,
        'interior_score': score,
        'box_radius_normalized': res['radius'],
        'has_normals': normals is not None,
        'export_frame': frame,
        'normalization': {'center': center.tolist(), 'scale': scale},
        'gamma_sweep': sweep_fracs,
    }
    with open(os.path.join(out_dir, 'ghpr.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if visible.mean() > 0.98:
        print("\n  [warn] Almost everything is visible — gamma is too loose to "
              "remove anything. Lower it (more negative).")
    elif visible.mean() < 0.15:
        print("\n  [warn] Very little is visible. Either gamma is too strict, or "
              "the viewpoint is not actually inside the shape.")

    print(f"\n  Outputs → {out_dir}")
    print(f"    {stem}_visible.ply    — visible points ({frame} coords), "
          f"feed to main.py --file")
    print(f"    ghpr_classified.ply   — red = visible, grey = hidden (START HERE)")
    print(f"    ghpr.json             — viewpoint, gamma, counts")
    print(f"{'─' * 64}\n")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description='GHPR visible-region extraction for raw point clouds.')
    ap.add_argument('--file', type=str, required=True,
                    help='Input point cloud (.ply/.xyz/.txt/.npy)')
    ap.add_argument('--gamma', type=float, default=-0.01,
                    help='GHPR kernel exponent, f(d) = d**gamma. Nearer 0 keeps '
                         'more points; -1.0 is the canonical strict value.')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='Default: <input dir>/ghpr/<input stem>')
    ap.add_argument('--N', type=int, default=-1,
                    help='Downsample to this many points (-1 keeps all)')
    ap.add_argument('--viewpoint', type=float, nargs=3, default=None,
                    help='Explicit viewpoint in the normalized frame, overriding '
                         'the automatic pick')
    ap.add_argument('--k', type=int, default=15,
                    help='Neighbours used by the interior test')
    ap.add_argument('--normalized', action='store_true',
                    help="Write outputs in the model's normalized [-1,1] frame "
                         "instead of the input file's own coordinates")
    ap.add_argument('--no_sweep', action='store_true',
                    help='Skip the gamma sweep table')
    args = ap.parse_args()

    run(args.file, gamma=args.gamma, out_dir=args.out_dir,
        downsample=None if args.N < 0 else args.N,
        normalized=args.normalized, k=args.k,
        viewpoint=args.viewpoint, sweep=not args.no_sweep)


if __name__ == '__main__':
    main()
