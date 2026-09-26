#!/usr/bin/env python3
# model/stitching_hole.py
"""
Hole stitching: glue a tube between the two boundary loops left by
model/cutting_hole.py, which raises the genus of the surface by one.

The tube is an N x R grid of cells, periodic around the loops:

    v = 1  ──  loop B vertices (existing ids, hanging on the top edge)
               ring R-1 ─┐
                 ...     ├─ new vertices, N per ring
               ring 1  ──┘
    v = 0  ──  loop A vertices (existing ids, hanging on the bottom edge)

Vertex matrix (vertex_features, one row per vertex id):
  * Loop vertices are NOT duplicated, averaged or moved. The tube cells list
    the same ids the cut surface already uses, so the tube and the surface
    read the same rows and the seam is exactly C0.
  * Loops of different length (vertices on A != on B) need no new loop
    vertices: each end cell carries a whole chain of loop vertices on its
    bottom/top edge (like hanging vertices on a quadtree edge), so a column can
    hold 3 A-edges and 1 B-edge.
  * New rows: (R - 1) x N, appended after the existing ones. Ring r of column
    k is initialised to  (1 - r/R) * z[A corner k] + (r/R) * z[B corner k],
    a straight blend of the two matched loop corners.

Cells are _TubePatch objects: they decode through the same MVC path as
quadtree leaves, are always frozen, and are serialized in the topology.

See docs/documentation.md.
"""

import numpy as np
import torch


def _arc_params(xyz):
    """Normalized arc-length position of each vertex of a closed polyline."""
    seg = np.linalg.norm(np.roll(xyz, -1, axis=0) - xyz, axis=1)
    total = float(seg.sum())
    if total <= 0:
        return np.arange(len(xyz)) / len(xyz)
    return np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / total


def _chain_u(xyz):
    """u in [0, 1] along an open chain, by arc length."""
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return np.linspace(0.0, 1.0, len(xyz))
    u = np.concatenate([[0.0], np.cumsum(seg)]) / total
    u[-1] = 1.0                 # exact: corners are looked up by uv
    return u


def _circular_mean(d):
    z = np.exp(2j * np.pi * d).mean()
    return (np.angle(z) / (2 * np.pi)) % 1.0, float(np.abs(z))


def align_loops(xa, xb, anchors):
    """
    Phase offset between the two loops. `xa` is loop A in TUBE order (reversed
    kept direction), `xb` loop B in kept direction; both walk the hole the
    same way round. Each anchor point is mapped to its nearest vertex on each
    loop; the offset is the circular mean of the arc-length differences.

    Returns (offset, concentration, concentration_if_reversed). Concentration
    is |mean phasor| in [0, 1]: near 1 = anchors agree. A reversed-direction
    fit that is clearly better means the loops run opposite ways and the tube
    would twist (should not happen on a consistently oriented surface).
    """
    sa, sb = _arc_params(xa), _arc_params(xb)
    ia = np.argmin(np.linalg.norm(anchors[:, None] - xa[None], axis=2), axis=1)
    ib = np.argmin(np.linalg.norm(anchors[:, None] - xb[None], axis=2), axis=1)
    offset, conc = _circular_mean(sb[ib] - sa[ia])
    _, conc_rev = _circular_mean(-sb[ib] - sa[ia])
    return offset, conc, conc_rev


def _monotone(idx, n):
    """Force strictly increasing indices in [0, n-1] (needs len(idx) <= n)."""
    idx = np.clip(np.asarray(idx, np.int64), 0, n - 1)
    for k in range(1, len(idx)):
        idx[k] = max(idx[k], idx[k - 1] + 1)
    idx[-1] = min(idx[-1], n - 1)
    for k in range(len(idx) - 2, -1, -1):
        idx[k] = min(idx[k], idx[k + 1] - 1)
    if idx[0] < 0 or np.any(np.diff(idx) <= 0):
        raise RuntimeError("could not place tube corners on the loop")
    return idx


def build_tube(record, vertex_features, n_columns, n_rows, first_vid):
    """
    Tube cells and new vertex rows for one cut record (pure, no model change).

    Returns (polys, new_keys, new_features, info); polys are
    (ids, uvs, col, row) as AdaptiveCubeComplex.add_tube expects.
    """
    # Every edge must be walked in opposite directions by its two cells, and
    # the kept leaves walk the loops in record order: so the tube's bottom
    # (left -> right) walks loop A reversed, and its top, which a CCW cell
    # walks right -> left, walks loop B in record order.
    A = np.asarray(record['loop_a'][::-1], np.int64)
    xa = np.asarray(record['loop_a_xyz'], np.float64)[::-1]
    B = np.asarray(record['loop_b'], np.int64)
    xb = np.asarray(record['loop_b_xyz'], np.float64)
    nA, nB = len(A), len(B)

    # Anchors for the phase: every loop vertex (mapped to its nearest vertex on
    # the other loop), plus the crossing points when the loops hug C.
    anchors = [xa, xb]
    if record.get('cut_mode') == 'crossing' and record.get('crossing_xyz'):
        anchors.append(np.asarray(record['crossing_xyz'], np.float64).reshape(-1, 3))
    anchors = np.concatenate(anchors)
    offset, conc, conc_rev = align_loops(xa, xb, anchors)

    # Rotate B so its phase-corrected arc length starts near 0 and increases.
    sb = (_arc_params(xb) - offset) % 1.0
    start = int(np.argmin(sb))
    B, xb, sb = np.roll(B, -start), np.roll(xb, -start, axis=0), np.roll(sb, -start)
    sa = _arc_params(xa)

    N = int(n_columns) if n_columns and n_columns > 0 else min(nA, nB, 64)
    N = max(3, min(N, nA, nB))
    R = max(1, int(n_rows))
    cA = _monotone(np.round(np.arange(N) * nA / N), nA)
    cB = _monotone([int(np.argmin(np.abs(sb - t))) for t in sa[cA]], nB)

    def chain(ids, xyz, lo, hi, n):
        """Loop ids from index lo to hi inclusive, wrapping past n."""
        j = np.arange(lo, hi + 1 if hi >= lo else hi + n + 1) % n
        return ids[j].tolist(), _chain_u(xyz[j])

    new_keys, rows = [], []
    t_idx = record.get('tube_index', 0)
    zA = vertex_features[A[cA]]
    zB = vertex_features[B[cB]]
    for r in range(1, R):
        t = r / R
        for k in range(N):
            new_keys.append(('tube', t_idx, r, k))
            rows.append((1.0 - t) * zA[k] + t * zB[k])
    new_features = (torch.stack(rows) if rows
                    else vertex_features.new_zeros(0, vertex_features.shape[1]))

    def corner(r, k):
        k %= N
        if r == 0:
            return int(A[cA[k]])
        if r == R:
            return int(B[cB[k]])
        return first_vid + (r - 1) * N + k

    polys = []
    for r in range(R):
        for k in range(N):
            k1 = (k + 1) % N
            if r == 0:
                bot, bu = chain(A, xa, cA[k], cA[k1], nA)
            else:
                bot, bu = [corner(r, k), corner(r, k1)], np.array([0.0, 1.0])
            if r + 1 == R:
                top, tu = chain(B, xb, cB[k], cB[k1], nB)
            else:
                top, tu = [corner(r + 1, k), corner(r + 1, k1)], np.array([0.0, 1.0])
            # CCW: bottom left->right, then top right->left.
            ids = bot + top[::-1]
            uvs = [(float(u), 0.0) for u in bu] + [(float(u), 1.0) for u in tu[::-1]]
            polys.append((ids, uvs, k, r))

    info = {'n_columns': N, 'n_rows': R, 'loop_a_len': nA, 'loop_b_len': nB,
            'n_new_vertices': len(new_keys), 'phase_offset': float(offset),
            'phase_concentration': conc, 'phase_concentration_reversed': conc_rev,
            'max_cell_vertices': max(len(p[0]) for p in polys)}
    return polys, new_keys, new_features, info


def stitch_hole(model, record, cfg, verbose=True):
    """
    Add the tube for a cut record to the model, in place. Replaces
    vertex_features (the optimizer must be rebuilt). Returns the stitch info
    and records it in `record['stitch']`.
    """
    cx = model.complex
    record['tube_index'] = len(cx.tube_log)
    polys, keys, feats, info = build_tube(
        record, cx.vertex_features.detach(), cfg.tube_columns, cfg.tube_rows,
        first_vid=cx.n_vertices)
    cx.add_tube(polys, keys, feats)
    if info['phase_concentration_reversed'] > info['phase_concentration'] + 0.2:
        print("    [warn] the loops fit better running opposite ways; the tube "
              "may be twisted")
    if verbose:
        print(f"    stitch: tube {info['n_columns']} x {info['n_rows']} cells, "
              f"{info['n_new_vertices']} new vertices (loops {info['loop_a_len']} / "
              f"{info['loop_b_len']}), phase fit {info['phase_concentration']:.2f}")
    before, after = prefit_tube(model, record['tube_index'],
                                steps=getattr(cfg, 'tube_prefit_steps', 0),
                                lr=getattr(cfg, 'tube_prefit_lr', 5e-3),
                                verbose=verbose)
    info['prefit_error'] = {'before': before, 'after': after}
    record['stitch'] = info
    return info


def prefit_tube(model, tube_index, steps=500, lr=5e-3, grid=12, verbose=True):
    """
    Fit the tube's NEW vertex rows so the tube decodes to the ruled surface
    between its two rims, before any training on the point cloud.

    Blending features (build_tube) does not blend positions: the decoder is
    nonlinear, so the freshly stitched tube can bulge or fold anywhere. Here
    every tube sample (column k, row r, local u, v) gets the target

        (1 - w) * A_k(u) + w * B_k(u),   w = (r + v) / R

    where A_k(u) / B_k(u) are the decoded rim points under column k (the
    bottom edge of row 0 and the top edge of row R-1). Those edges only read
    loop vertices, so the targets are fixed. Only the new rows are updated;
    the decoder and every other vertex row are held, so the rest of the
    surface and the seams do not move.

    Returns (mean distance to target before, after).
    """
    cx = model.complex
    first = cx.tube_log[tube_index]['first_vid']
    n_new = len(cx.tube_log[tube_index]['keys'])
    base = cx.n_quad_leaves
    cells = {(p.col, p.row): base + i for i, p in enumerate(cx.tube_patches)
             if p.tube == tube_index}
    N = 1 + max(k for k, _ in cells)
    R = 1 + max(r for _, r in cells)
    if n_new == 0 or steps <= 0:
        return None, None
    device = cx.vertex_features.device

    t = torch.linspace(0.0, 1.0, grid, device=device)
    gu, gv = torch.meshgrid(t, t, indexing='ij')
    uv = torch.stack([gu.flatten(), gv.flatten()], -1)            # (G, 2)
    pids, targets = [], []
    with torch.no_grad():
        for k in range(N):
            rimA = model(torch.full((grid,), cells[(k, 0)], device=device),
                         torch.stack([t, torch.zeros_like(t)], -1))
            rimB = model(torch.full((grid,), cells[(k, R - 1)], device=device),
                         torch.stack([t, torch.ones_like(t)], -1))
            for r in range(R):
                w = ((r + uv[:, 1]) / R).unsqueeze(-1)
                iu = torch.round(uv[:, 0] * (grid - 1)).long()
                targets.append((1 - w) * rimA[iu] + w * rimB[iu])
                pids.append(torch.full((uv.shape[0],), cells[(k, r)], device=device))
    pids = torch.cat(pids)
    targets = torch.cat(targets)
    uv_all = uv.repeat(N * R, 1)

    vf = cx.vertex_features
    mask = torch.zeros(vf.shape[0], 1, device=device)
    mask[first:first + n_new] = 1.0
    frozen = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in frozen:
        p.requires_grad_(p is vf)
    hook = vf.register_hook(lambda g: g * mask)
    opt = torch.optim.Adam([vf], lr=lr)

    def err():
        with torch.no_grad():
            return float((model(pids, uv_all) - targets).norm(dim=1).mean())

    before = err()
    for _ in range(steps):
        loss = ((model(pids, uv_all) - targets) ** 2).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    hook.remove()
    for p, flag in frozen:
        p.requires_grad_(flag)
    after = err()
    if verbose:
        print(f"    tube prefit: mean distance to the rim-to-rim surface "
              f"{before:.4f} -> {after:.4f} ({steps} steps)")
    return before, after


@torch.no_grad()
def seam_gaps(model, only_tube=True, samples_per_edge=5, batch=16384):
    """
    Max/mean 3D gap across shared edges. For every directed edge (a, b) of a
    cell whose reverse (b, a) belongs to another cell, decode points along the
    edge from both cells and compare. Also counts boundary edges (edges with
    no partner); a closed surface has none.

    only_tube restricts the check to edges of tube cells (the new seams).
    """
    cx = model.complex
    device = next(model.parameters()).device
    owner = {}
    polys = []
    for i, p in enumerate(cx.leaf_patches):
        ids, uvs = cx._polygon(p)
        polys.append({v: uv for v, uv in zip(ids, uvs)})
        for k in range(len(ids)):
            owner[(ids[k], ids[(k + 1) % len(ids)])] = i
    n_quad = cx.n_quad_leaves
    t = (np.arange(samples_per_edge) + 0.5) / samples_per_edge
    pi, ui, pj, uj = [], [], [], []
    n_boundary = 0
    for (a, b), i in owner.items():
        j = owner.get((b, a))
        if j is None:
            n_boundary += 1
            continue
        if only_tube and i < n_quad and j < n_quad:
            continue
        if i > j:
            continue            # each undirected edge once
        a_i, b_i = np.asarray(polys[i][a]), np.asarray(polys[i][b])
        a_j, b_j = np.asarray(polys[j][a]), np.asarray(polys[j][b])
        for s in t:
            pi.append(i); ui.append((1 - s) * a_i + s * b_i)
            pj.append(j); uj.append((1 - s) * a_j + s * b_j)
    if not pi:
        return {'max_gap': 0.0, 'mean_gap': 0.0, 'n_edges': 0,
                'n_boundary_edges': n_boundary}
    P = lambda x: torch.tensor(np.asarray(x), device=device)
    pi, pj = P(pi).long(), P(pj).long()
    ui, uj = P(ui).float(), P(uj).float()
    gaps = []
    for s in range(0, pi.shape[0], batch):
        xi = model(pi[s:s + batch], ui[s:s + batch])
        xj = model(pj[s:s + batch], uj[s:s + batch])
        gaps.append((xi - xj).norm(dim=1).cpu())
    g = torch.cat(gaps)
    return {'max_gap': float(g.max()), 'mean_gap': float(g.mean()),
            'n_edges': int(g.numel() // samples_per_edge),
            'n_boundary_edges': n_boundary}
