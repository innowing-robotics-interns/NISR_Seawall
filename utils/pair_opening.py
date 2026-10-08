#!/usr/bin/env python3
# utils/pair_opening.py

"""
Pair the openings found by opening_labels.py without requiring the two
membranes to intersect.

Usage:

    python utils/pair_opening.py --ckpt logs/.../checkpoint.pt

Note: 
    Requires uv_hole_mask.py and opening_labels.py to have been run first.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import open3d as o3d
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

import patch_vis
from opening_labels import opening_colors
from patch_intersection import vertex_normals
from uv_hole_mask import (adaptive_leaf_rects, composite_face_canvases, grid_faces,
                          sample_patch_uv_grid, sheet_layout, write_colored_ply)

from model.model import FACE_NAMES  # noqa: E402


# Ray outcomes, in the order they are checked. Only 'vote' pairs anything.
OUTCOMES = ('vote', 'miss', 'frontface', 'surface', 'self', 'too_far', 'solid')
COLOR_UNPAIRED = (90, 90, 90)


# sampling 
def sample_atlas(F, patch_ids, resolution, device, batch_size):
    """
    Every patch on its RxR grid, with outward unit vertex normals.

    Returns:
        Tuple `(xyz, normals, uv)`: xyz and normals are (n_patches * R*R, 3)
        float32 in patch_ids order, uv is the (R*R, 2) grid shared by all
        patches, flattened `[u_index, v_index]` like the masks.
    """
    R = resolution
    _, faces = grid_faces(R, 1)
    xyz = np.empty((len(patch_ids), R * R, 3), np.float32)
    nrm = np.empty_like(xyz)
    uv = None
    for pos, patch_id in enumerate(patch_ids):
        x, uv = sample_patch_uv_grid(F, patch_id, R, device, batch_size)
        x = x.astype(np.float64)
        xyz[pos] = x
        nrm[pos] = vertex_normals(x, faces)
    return xyz.reshape(-1, 3), nrm.reshape(-1, 3), uv


def ray_mesh_triangles(n_patches, resolution, stride):
    """
    Triangles of the whole atlas as GLOBAL sample indices (pos * R*R + flat).

    `stride > 1` decimates the casting mesh only. Every corner is still a real
    grid sample, so a hit maps straight back to masks and opening IDs.
    """
    sel, local = grid_faces(resolution, stride)
    corners = sel[local]                                           # (T, 3)
    offsets = np.arange(n_patches, dtype=np.int64) * resolution * resolution
    return (corners[None, :, :] + offsets[:, None, None]).reshape(-1, 3)


def signed_volume(verts, tris, chunk=2_000_000):
    """Volume enclosed by the mesh. Positive when the normals point outward."""
    vol = 0.0
    for i in range(0, tris.shape[0], chunk):
        t = tris[i:i + chunk]
        v0 = verts[t[:, 0]].astype(np.float64)
        v1 = verts[t[:, 1]].astype(np.float64)
        v2 = verts[t[:, 2]].astype(np.float64)
        vol += float(np.einsum('ij,ij->', v0, np.cross(v1, v2))) / 6.0
    return vol


def median_edge(verts, tris, max_tris=500_000, seed=0):
    """Median edge length of the casting mesh, the length unit for t_min."""
    rng = np.random.default_rng(seed)
    pick = tris if tris.shape[0] <= max_tris else \
        tris[rng.choice(tris.shape[0], max_tris, replace=False)]
    return float(np.median(np.linalg.norm(verts[pick[:, 1]] - verts[pick[:, 0]], axis=1)))


# ray sources
def pick_ray_sources(opening_flat, n_openings, resolution, n_patches, stride,
                     max_rays, min_rays, rng):
    """
    Black samples to cast from, per opening.

    Samples on every `stride`-th grid line keep the rays spread evenly over the
    cap. An opening too small to collect `min_rays` that way casts from all of
    its samples instead, so small caps are not silently left out.

    Returns:
        dict `opening_id -> (n,) global sample indices`.
    """
    R = resolution
    i, j = np.meshgrid(np.arange(R), np.arange(R), indexing='ij')
    on_grid = np.tile(((i % stride == 0) & (j % stride == 0)).ravel(), n_patches)

    black = np.flatnonzero(opening_flat >= 0)
    order = np.argsort(opening_flat[black], kind='stable')
    black = black[order]
    bounds = np.searchsorted(opening_flat[black], np.arange(n_openings + 1))

    sources = {}
    for o in range(n_openings):
        members = black[bounds[o]:bounds[o + 1]]
        idx = members[on_grid[members]]
        if idx.size < min_rays:
            idx = members
        if idx.size > max_rays:
            idx = np.sort(rng.choice(idx, max_rays, replace=False))
        sources[o] = idx
    return sources


# ray casting
def build_scene(verts, tris):
    """Embree BVH over the whole atlas (open3d)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(verts, np.float32)),
                        o3d.core.Tensor(np.ascontiguousarray(tris, np.uint32)))
    return scene


def cast_first_hit(scene, orig, dirs, chunk=1_000_000):
    """
    First hit of every ray.

    Returns:
        Tuple `(t, prim, bary)`: t is inf on a miss, prim the hit triangle, bary
        the (u, v) barycentrics of the hit, p = (1-u-v) v0 + u v1 + v v2.
    """
    t_out, p_out, b_out = [], [], []
    for i in range(0, orig.shape[0], chunk):
        rays = np.concatenate([orig[i:i + chunk], dirs[i:i + chunk]], axis=1)
        ans = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        t_out.append(ans['t_hit'].numpy())
        p_out.append(ans['primitive_ids'].numpy().astype(np.int64))
        b_out.append(ans['primitive_uvs'].numpy())
    return np.concatenate(t_out), np.concatenate(p_out), np.concatenate(b_out)


def cast_past_own_sheet(scene, src, xyz, dirs, tris, resolution, edge,
                        local_steps, max_recasts=4):
    """
    First hit of every ray that is NOT the source's own sheet.

    Facing caps can sit closer together than one mesh edge, so a ray cannot
    simply start a fixed distance inside the surface: it would start beyond the
    partner. Rays instead start a hair off the surface, and a hit that is only
    the source's own neighbourhood (a triangle of the same patch within
    `local_steps` grid steps, or anything closer than 1% of an edge) is stepped
    past and re-cast.

    Returns:
        Tuple `(t, prim, bary)` like `cast_first_hit`, with t measured from the
        source sample itself.
    """
    R = resolution
    eps = 1e-3 * edge
    src_pos, src_flat = src // (R * R), src % (R * R)
    src_i, src_j = src_flat // R, src_flat % R

    start = np.full(src.size, eps)
    t = np.full(src.size, np.inf)
    prim = np.full(src.size, -1, np.int64)
    bary = np.zeros((src.size, 2), np.float32)
    todo = np.arange(src.size)
    for _ in range(max_recasts + 1):
        orig = xyz[src[todo]].astype(np.float64) + start[todo, None] * dirs[todo]
        t_i, p_i, b_i = cast_first_hit(scene, orig, dirs[todo])
        hit = np.isfinite(t_i)

        local = np.zeros(todo.size, bool)
        if hit.any():
            c = tris[p_i[hit]]                                      # (h, 3)
            c_pos, c_flat = c // (R * R), c % (R * R)
            same = (c_pos == src_pos[todo][hit, None]).all(axis=1)
            near = ((np.abs(c_flat // R - src_i[todo][hit, None]) <= local_steps)
                    & (np.abs(c_flat % R - src_j[todo][hit, None]) <= local_steps)).all(axis=1)
            local[hit] = (same & near) | (t_i[hit] < 0.01 * edge)

        done = ~local
        t[todo[done]] = t_i[done] + start[todo[done]]
        prim[todo[done]] = p_i[done]
        bary[todo[done]] = b_i[done]
        start[todo[local]] += t_i[local] + eps
        todo = todo[local]
        if todo.size == 0:
            break
    return t, prim, bary


def pick_opening(corner_ids):
    """
    Opening a hit landed in, read off the hit triangle's three corners.

    The first black corner wins, unless two corners agree (majority). -1 when
    no corner is black, i.e. the ray hit ordinary surface.
    """
    a, b, c = corner_ids[:, 0], corner_ids[:, 1], corner_ids[:, 2]
    out = np.full(corner_ids.shape[0], -1, dtype=np.int64)
    out = np.where(c >= 0, c, out)
    out = np.where(b >= 0, b, out)
    out = np.where(a >= 0, a, out)
    return np.where((b >= 0) & (b == c), b, out)


# empty space test
@torch.no_grad()
def winding_numbers(q, v0, v1, v2, device, q_chunk=256, t_chunk=32768):
    """
    Generalized winding number of a triangle soup at query points q.

    Sum of signed solid angles (Van Oosterom & Strackee) over 4 pi: ~1 inside a
    closed outward-oriented surface, ~0 outside, and it degrades gracefully
    where the surface has holes.
    """
    q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
    tri = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (v0, v1, v2)]
    out = torch.empty(q_t.shape[0], dtype=torch.float64, device=device)
    for i in range(0, q_t.shape[0], q_chunk):
        qc = q_t[i:i + q_chunk, None, :]
        acc = torch.zeros(qc.shape[0], dtype=torch.float64, device=device)
        for j in range(0, tri[0].shape[0], t_chunk):
            a, b, c = (t[None, j:j + t_chunk] - qc for t in tri)
            la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
            det = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
            den = (la * lb * lc + (a * b).sum(-1) * lc
                   + (b * c).sum(-1) * la + (c * a).sum(-1) * lb)
            acc += (2.0 * torch.atan2(det, den)).sum(1).double()
        out[i:i + q_chunk] = acc / (4.0 * np.pi)
    return out.cpu().numpy()


def segment_in_air(p0, p1, white_tris, verts, device, n_samples, air_min):
    """
    Does the segment p0 -> p1 run through air in the target?

    The white region is the fitted surface minus the caps, i.e. the target
    surface itself, so its winding number is ~0 in air (a handle's tunnel
    included) and ~1 inside solid material. A point counts as air below 0.5;
    the segment counts as air when at least `air_min` of its `n_samples`
    interior points do.

    Returns:
        Tuple `(is_air, air_fraction)`, both (n,).
    """
    fracs = (np.arange(n_samples) + 1.0) / (n_samples + 1.0)
    q = (p0[:, None, :] + (p1 - p0)[:, None, :] * fracs[None, :, None]).reshape(-1, 3)
    w = winding_numbers(q, verts[white_tris[:, 0]], verts[white_tris[:, 1]],
                        verts[white_tris[:, 2]], device)
    air_frac = (w < 0.5).reshape(-1, n_samples).mean(axis=1)
    return air_frac >= air_min, air_frac


def cap_sides(sources, xyz, nrm, tris, offset, device, n_probe, rng):
    """
    Winding number of the WHOLE fitted surface just behind and just in front of
    each opening, at up to `n_probe` of its ray sources.

    A cap closing a tunnel reads ~1 behind (inside the fitted volume) and ~0 in
    front. A region the fit folded inside-out reads ~0 behind and ~-1 in front:
    there is nothing to pair across, the fit itself is broken there.

    Returns:
        dict `opening_id -> (median_behind, median_front)`.
    """
    probe = {o: (idx if idx.size <= n_probe else rng.choice(idx, n_probe, replace=False))
             for o, idx in sources.items()}
    flat = np.concatenate([probe[o] for o in sorted(probe)])
    if flat.size == 0:
        return {o: (np.nan, np.nan) for o in sources}
    p, n = xyz[flat].astype(np.float64), nrm[flat].astype(np.float64)
    q = np.concatenate([p - offset * n, p + offset * n])
    w = winding_numbers(q, xyz[tris[:, 0]], xyz[tris[:, 1]], xyz[tris[:, 2]], device)
    behind, front = w[:flat.size], w[flat.size:]
    out, start = {}, 0
    for o in sorted(probe):
        k = probe[o].size
        out[o] = ((float(np.median(behind[start:start + k])),
                   float(np.median(front[start:start + k]))) if k else (np.nan, np.nan))
        start += k
    return out


# matching
def match_openings(votes, n_rays, crossings, min_frac, min_crossings):
    """
    One-to-one pairing from the vote matrix.

    `frac[a, b]` is the share of a's rays that landed in b. A candidate needs
    both directions above `min_frac` (a cap facing a much larger one still gets
    most of its own rays into it), or at least `min_crossings` crossings.
    Candidates are taken greedily by the geometric mean of the two fractions,
    so each opening joins at most one pair.

    Returns:
        Tuple `(pairs, candidates, frac, score)`.
    """
    n = votes.shape[0]
    frac = votes / np.maximum(n_rays, 1)[:, None]
    score = np.sqrt(frac * frac.T)

    candidates = []
    for a in range(n):
        for b in range(a + 1, n):
            by_rays = min(frac[a, b], frac[b, a]) >= min_frac
            by_cross = min_crossings > 0 and crossings[a, b] >= min_crossings
            if by_rays or by_cross:
                evidence = '+'.join(e for e, on in (('rays', by_rays),
                                                    ('crossings', by_cross)) if on)
                candidates.append({'a': a, 'b': b, 'score': float(score[a, b]),
                                   'frac_ab': float(frac[a, b]),
                                   'frac_ba': float(frac[b, a]),
                                   'votes_ab': int(votes[a, b]),
                                   'votes_ba': int(votes[b, a]),
                                   'crossings': int(crossings[a, b]),
                                   'evidence': evidence})
    candidates.sort(key=lambda c: (-c['score'], -c['crossings']))

    taken, pairs = set(), []
    for c in candidates:
        if c['a'] in taken or c['b'] in taken:
            continue
        taken.update((c['a'], c['b']))
        pairs.append(dict(c))

    # Strongest competing candidate for either member, to flag weak margins.
    for p in pairs:
        rivals = [c for c in candidates if c is not p
                  and {c['a'], c['b']} != {p['a'], p['b']}
                  and ({c['a'], c['b']} & {p['a'], p['b']})]
        best = max(rivals, key=lambda c: c['score'], default=None)
        p['runner_up'] = None if best is None else \
            {'openings': [best['a'], best['b']], 'score': best['score']}
    return pairs, candidates, frac, score


def load_crossings(path, n_openings):
    """Symmetric crossing counts between openings from intersections.npz."""
    C = np.zeros((n_openings, n_openings), np.int64)
    d = np.load(path)
    if 'opening_src' not in d.files:
        return C, False
    a, b = d['opening_src'].astype(np.int64), d['opening_dst'].astype(np.int64)
    keep = (a >= 0) & (b >= 0) & (a != b) & (a < n_openings) & (b < n_openings)
    np.add.at(C, (a[keep], b[keep]), 1)
    np.add.at(C, (b[keep], a[keep]), 1)
    return C, True


# figures
def save_score_matrix(path, score, crossings, pairs):
    """Openings x openings: mutual ray score, crossings marked, pairs boxed."""
    n = score.shape[0]
    side = min(12.0, 1.5 + 0.45 * n)
    fig, ax = plt.subplots(figsize=(side + 1.2, side))
    im = ax.imshow(score, cmap='magma', vmin=0.0, vmax=max(float(score.max()), 1e-6),
                   interpolation='nearest')
    if n <= 30:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        for i in range(n):
            for j in range(n):
                if crossings[i, j]:
                    ax.text(j, i, 'x', ha='center', va='center', fontsize=7,
                            color='#39c5cf')
    for p in pairs:
        for i, j in ((p['a'], p['b']), (p['b'], p['a'])):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='#56d364', linewidth=1.8))
    ax.set_xlabel('opening ID')
    ax.set_ylabel('opening ID')
    ax.set_title('Mutual ray score between openings\n'
                 '(green box = paired, x = has crossings)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='sqrt(frac_ab * frac_ba)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pair_id_map(opening_id, n_openings, pairs):
    """
    Pair index of every UV sample, shape like `opening_id` (n_patches, R, R).

    k >= 0 for both caps of pair k, -2 for an opening left unpaired, -1 for a
    sample that is not in any opening.
    """
    lut = np.full(n_openings, -2, np.int32)
    for k, p in enumerate(pairs):
        lut[p['a']] = k
        lut[p['b']] = k
    return np.where(opening_id >= 0, lut[np.clip(opening_id, 0, None)], -1).astype(np.int32)


def _opening_label_spots(opening_id, n_openings, group_of, coord_of, cell_area_of):
    """
    Where to print each opening's ID: the mean position of its samples on the
    group (cube face, or patch) holding most of them.

    `group_of(pos)` names the group of patch position `pos`; `coord_of(pos, i, j)`
    maps grid indices of that patch to the (x, y) plotting coordinates;
    `cell_area_of(pos)` is the plotted area of one of its grid samples.

    Returns:
        dict `opening_id -> (group, x, y, plotted_area_on_that_group)`.
    """
    acc = {}
    for pos in range(opening_id.shape[0]):
        i, j = np.nonzero(opening_id[pos] >= 0)
        if i.size == 0:
            continue
        ids = opening_id[pos][i, j]
        x, y = coord_of(pos, i, j)
        g = group_of(pos)
        for o in np.unique(ids):
            sel = ids == o
            s = acc.setdefault((int(o), g), [0, 0.0, 0.0, 0.0])
            s[0] += int(sel.sum())
            s[1] += float(x[sel].sum())
            s[2] += float(y[sel].sum())
            s[3] += float(sel.sum()) * cell_area_of(pos)
    spots = {}
    for (o, g), (n, sx, sy, area) in acc.items():
        if o not in spots or n > spots[o][4]:
            spots[o] = (g, sx / n, sy / n, area, n)
    return {o: v[:4] for o, v in spots.items() if o < n_openings}


def save_pairing_atlas(path, pair_map, opening_id, n_openings, patch_ids, F,
                       is_adaptive, atlas_mode, pairs, colors, face_res):
    """
    The pairing painted onto the UV domain: both caps of a pair share one
    colour, unpaired openings are dark grey, the rest of the surface is light
    grey. Each opening is labelled with its ID at its centre.
    """
    n_pairs = len(pairs)
    palette = np.vstack([colors[:n_pairs], np.array(COLOR_UNPAIRED)[None]])
    cmap = ListedColormap(palette.astype(np.float64) / 255.0)
    cmap.set_bad(color='0.92')
    # Unpaired (-2) goes to the last colour; not-a-hole (-1) becomes NaN.
    field = np.where(pair_map == -2, n_pairs, pair_map).astype(np.float32)
    field[pair_map == -1] = np.nan
    vmin, vmax = -0.5, n_pairs + 0.5
    R = pair_map.shape[1]
    color_of = {o: palette[n_pairs] / 255.0 for o in range(n_openings)}
    for k, p in enumerate(pairs):
        color_of[p['a']] = color_of[p['b']] = palette[k] / 255.0

    def label(ax, o, x, y, area, small_area):
        # An opening in deeply refined leaves can cover only a few pixels of the
        # figure; ring it in its colour so it cannot go unseen.
        if area < small_area:
            ax.scatter([x], [y], s=260, facecolors='none', edgecolors=[color_of[o]],
                       linewidths=2.0, zorder=3)
        ax.text(x, y, str(o), fontsize=7, ha='center', va='center', zorder=4,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))

    if is_adaptive:
        rects, faces, _ = adaptive_leaf_rects(F)
        fields = {p: field[i] for i, p in enumerate(patch_ids)}
        canvases = composite_face_canvases(fields, patch_ids, rects, faces, face_res)
        spots = _opening_label_spots(
            opening_id, n_openings,
            group_of=lambda pos: int(faces[patch_ids[pos]]),
            coord_of=lambda pos, i, j: (
                rects[patch_ids[pos], 1] + j / (R - 1) * rects[patch_ids[pos], 2],
                rects[patch_ids[pos], 0] + i / (R - 1) * rects[patch_ids[pos], 2]),
            cell_area_of=lambda pos: (rects[patch_ids[pos], 2] / (R - 1)) ** 2)
        small_area = 2e-4                    # of a unit face, i.e. ~1.4% of its width

        # Explicit spacing: equal-aspect panels defeat tight_layout's row gap.
        fig, axes = plt.subplots(2, 3, figsize=(15, 12.5), squeeze=False,
                                 gridspec_kw={'hspace': 0.12, 'wspace': 0.05})
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
        for f in range(len(FACE_NAMES)):
            ax = axes[f // 3][f % 3]
            ax.imshow(np.ma.masked_invalid(canvases[f]), cmap=cmap, vmin=vmin,
                      vmax=vmax, interpolation='nearest', extent=[0, 1, 1, 0])
            for p in patch_ids:
                if int(faces[p]) == f:
                    u0, v0, size = rects[p]
                    ax.add_patch(Rectangle((v0, u0), size, size, fill=False,
                                           edgecolor='#2f7fd0', linewidth=0.3))
            for o, (g, x, y, area) in spots.items():
                if g == f:
                    label(ax, o, x, y, area, small_area)
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'face {FACE_NAMES[f]}', fontsize=10)
        title = 'Opening pairs per cube face'
    else:
        names, per_sheet, n_rows, n_cols = sheet_layout(F, atlas_mode)
        pos_of = {p: i for i, p in enumerate(patch_ids)}
        spots = _opening_label_spots(
            opening_id, n_openings, group_of=lambda pos: patch_ids[pos],
            coord_of=lambda pos, i, j: (j.astype(np.float64), i.astype(np.float64)),
            cell_area_of=lambda pos: 1.0)
        small_area = 2e-4 * R * R            # of a tile, in pixels

        fig, axes = plt.subplots(n_rows, len(names) * n_cols,
                                 figsize=(1.9 * len(names) * n_cols + 1.0,
                                          2.0 * n_rows + 1.8), squeeze=False)
        for sheet in range(len(names)):
            for row in range(n_rows):
                for col in range(n_cols):
                    ax = axes[row][sheet * n_cols + col]
                    pid = sheet * per_sheet + row * n_cols + col
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if pid not in pos_of:
                        ax.set_facecolor('0.85')
                        continue
                    ax.imshow(np.ma.masked_invalid(field[pos_of[pid]]), cmap=cmap,
                              vmin=vmin, vmax=vmax, interpolation='nearest')
                    for o, (g, x, y, area) in spots.items():
                        if g == pid:
                            label(ax, o, x, y, area, small_area)
                    ax.set_title(f'p{pid}', fontsize=7, pad=2)
                    if row == 0 and col == 0:
                        ax.text(0.0, 1.35, names[sheet], transform=ax.transAxes,
                                fontsize=11, fontweight='bold', ha='left')
        title = 'Opening pairs per patch'

    handles = [Patch(facecolor=colors[k] / 255.0, label=f"pair {k}: {p['a']} ↔ {p['b']}")
               for k, p in enumerate(pairs)]
    handles.append(Patch(facecolor=np.array(COLOR_UNPAIRED) / 255.0, label='unpaired'))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 6),
               fontsize=8, frameon=False)
    fig.suptitle(f'{title}   (same colour = one pair; numbers = opening IDs; '
                 f'light grey = not a hole; u down, v right)', fontsize=12)
    if not is_adaptive:
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description='Pair openings that face each other across a handle, '
                    'whether or not their membranes intersect.')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--mask_npz', type=str, default=None,
                    help='uv_mask.npz (default: <ckpt dir>/uv_masks/<ckpt name>/uv_mask.npz)')
    ap.add_argument('--openings_npz', type=str, default=None,
                    help='openings.npz (default: <mask_npz dir>/openings/openings.npz)')
    ap.add_argument('--intersections_npz', type=str, default=None,
                    help='intersections.npz from patch_intersection.py, used as extra '
                         'evidence (default: <mask_npz dir>/intersections/'
                         'intersections.npz if it exists). Must come from the same '
                         'openings.npz.')
    ap.add_argument('--no_intersections', action='store_true',
                    help='Pair from rays alone, even if intersections.npz exists')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='Default: <mask_npz dir>/opening_pairing')
    ap.add_argument('--adaptive', type=str, default='auto',
                    choices=['auto', 'yes', 'no'])

    ap.add_argument('--mesh_stride', type=int, default=2,
                    help='Decimate the casting mesh by this factor. 1 casts against '
                         'the full mask grid (slower, more memory)')
    ap.add_argument('--ray_stride', type=int, default=2,
                    help='Cast from every Nth grid sample of each opening')
    ap.add_argument('--max_rays_per_opening', type=int, default=20000)
    ap.add_argument('--min_rays_per_opening', type=int, default=32,
                    help='An opening with fewer strided samples casts from all of them')
    ap.add_argument('--local_steps', type=int, default=None,
                    help='A hit on the source\'s own patch within this many grid '
                         'steps is the source\'s own sheet and is stepped past '
                         '(default: 2 x mesh_stride)')
    ap.add_argument('--max_gap', type=float, default=0.0,
                    help='Ignore hits farther than this, in normalized units (0 = no limit)')

    ap.add_argument('--air_samples', type=int, default=3,
                    help='Points tested along each ray segment')
    ap.add_argument('--air_min', type=float, default=0.67,
                    help='Fraction of those points that must be in air')
    ap.add_argument('--winding_stride', type=int, default=0,
                    help='Grid decimation of the white mesh used for the winding '
                         'number (0 = auto, about --winding_max_tris triangles)')
    ap.add_argument('--winding_max_tris', type=int, default=200_000)
    ap.add_argument('--no_air_test', action='store_true',
                    help='Skip the empty-space test (diagnostic only)')

    ap.add_argument('--min_frac', type=float, default=0.1,
                    help="Share of EACH opening's rays that must land in the other")
    ap.add_argument('--min_crossings', type=int, default=5,
                    help='Crossings that pair two openings on their own (0 = ignore crossings)')

    ap.add_argument('--unnormalize', action='store_true',
                    help='Write the PLYs in ORIGINAL input coordinates instead of the '
                         'normalized frame. The npz files carry both.')
    ap.add_argument('--face_resolution', type=int, default=512,
                    help='[Adaptive] Canvas resolution per cube face in pairing_atlas.png')
    ap.add_argument('--batch_size', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model_path', type=str, default=None)
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # inputs
    if args.mask_npz is None:
        stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.mask_npz = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                     'uv_masks', stem, 'uv_mask.npz')
    if not os.path.exists(args.mask_npz):
        raise FileNotFoundError(
            f"No uv_mask.npz at {args.mask_npz}. Run utils/uv_hole_mask.py first.")
    mask_dir = os.path.dirname(os.path.abspath(args.mask_npz))
    if args.openings_npz is None:
        args.openings_npz = os.path.join(mask_dir, 'openings', 'openings.npz')
    if not os.path.exists(args.openings_npz):
        raise FileNotFoundError(
            f"No openings.npz at {args.openings_npz}. Run utils/opening_labels.py first.")
    if args.out_dir is None:
        args.out_dir = os.path.join(mask_dir, 'opening_pairing')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n  Loading checkpoint: {args.ckpt}")
    if args.adaptive == 'auto':
        is_adaptive = patch_vis._is_adaptive_checkpoint(args.ckpt)
    else:
        is_adaptive = args.adaptive == 'yes'
    if is_adaptive:
        F, meta, _, _, _ = patch_vis._load_adaptive_from_checkpoint(args.ckpt, args.device)
        atlas_mode = 'adaptive_cube'
        print(f"  Atlas: adaptive quadtree cube, {F.n_patches} leaves")
    else:
        F, meta, ckpt_args, _, _ = patch_vis._load_model_from_checkpoint(
            args.ckpt, args.device, args.model_path)
        atlas_mode = ckpt_args.get('atlas_mode', 'single_sheet')
        print(f"  Atlas: {atlas_mode}, {F.n_patches} patches")

    mask_data = np.load(args.mask_npz)
    od = np.load(args.openings_npz)
    patch_ids = [int(p) for p in mask_data['patch_ids']]
    R = int(mask_data['resolution'])
    if [int(p) for p in od['patch_ids']] != patch_ids or int(od['resolution']) != R:
        raise ValueError(f"{args.openings_npz} does not match the mask: patch_ids or "
                         f"resolution differ. Re-run opening_labels.py on this mask.")
    n_openings = int(od['n_openings'])
    P = len(patch_ids)
    opening_flat = od['opening_id'].reshape(-1).astype(np.int64)
    white_flat = mask_data['masks'].reshape(-1).astype(bool)
    print(f"  Openings: {n_openings} from {args.openings_npz}")
    if n_openings < 2:
        print("\n  Fewer than two openings, nothing to pair.\n")
        return

    crossings = np.zeros((n_openings, n_openings), np.int64)
    have_crossings = False
    if not args.no_intersections:
        if args.intersections_npz is None:
            cand = os.path.join(mask_dir, 'intersections', 'intersections.npz')
            args.intersections_npz = cand if os.path.exists(cand) else None
        if args.intersections_npz is not None:
            crossings, have_crossings = load_crossings(args.intersections_npz, n_openings)
            print(f"  Crossings: {int(crossings.sum()) // 2} opening-to-opening hits "
                  f"from {args.intersections_npz}")
    if not have_crossings:
        print("  Crossings: none used (pairing from rays alone)")

    center = np.asarray(meta['center'], np.float64)
    scale = float(meta['scale'])

    def to_frame(p):
        return p * scale + center if args.unnormalize else p

    # surface
    print(f"\n  Sampling {P} patches at {R}x{R}...")
    t0 = time.time()
    xyz, nrm, grid_uv = sample_atlas(F, patch_ids, R, args.device, args.batch_size)
    tris = ray_mesh_triangles(P, R, args.mesh_stride)
    print(f"    done in {time.time() - t0:.1f}s  ({tris.shape[0]:,} casting triangles, "
          f"mesh_stride={args.mesh_stride})")

    # verify whether or not the atlas normals point outward, and flip them if not
    vol = signed_volume(xyz, tris)
    if vol < 0:
        nrm = -nrm
        print(f"  [warn] Enclosed volume is negative ({vol:.4f}): the atlas normals "
              f"point INWARD. Flipped them for casting.")
    else:
        print(f"  Enclosed volume {vol:.4f} > 0: atlas normals point outward")

    edge = median_edge(xyz, tris)
    local_steps = args.local_steps if args.local_steps is not None else 2 * args.mesh_stride
    print(f"  Median casting edge = {edge:.6f}   own-sheet window = {local_steps} grid steps")

    # rays
    rng = np.random.default_rng(args.seed)
    sources = pick_ray_sources(opening_flat, n_openings, R, P, args.ray_stride,
                               args.max_rays_per_opening, args.min_rays_per_opening, rng)
    src = np.concatenate([sources[o] for o in range(n_openings)])
    src_open = np.concatenate([np.full(sources[o].size, o, np.int64)
                               for o in range(n_openings)])
    n_rays = np.array([sources[o].size for o in range(n_openings)], np.int64)

    dirs = -nrm[src].astype(np.float64)

    print(f"  Casting {src.size:,} inward rays from {n_openings} openings...")
    t0 = time.time()
    scene = build_scene(xyz, tris)
    t_hit, prim, bary = cast_past_own_sheet(scene, src, xyz, dirs, tris, R, edge,
                                            local_steps)
    print(f"    done in {time.time() - t0:.1f}s")

    outcome = np.full(src.size, OUTCOMES.index('miss'), np.int64)
    hit = np.isfinite(t_hit)
    gap = np.where(hit, t_hit, np.inf)

    corners = tris[prim[hit]]                                       # (h, 3)
    w = np.stack([1.0 - bary[hit, 0] - bary[hit, 1],
                  bary[hit, 0], bary[hit, 1]], axis=1)              # (h, 3)
    hit_xyz = xyz[src[hit]].astype(np.float64) + t_hit[hit, None] * dirs[hit]
    # Barycentric convention check: the blended corners must land on the hit.
    recon = np.einsum('hk,hkd->hd', w, xyz[corners].astype(np.float64))
    bary_err = float(np.max(np.linalg.norm(recon - hit_xyz, axis=1))) if hit.any() else 0.0
    if bary_err > 0.5 * edge:
        print(f"  [warn] Barycentric reconstruction off by {bary_err:.2e} "
              f"(> half an edge): hit UVs below may be wrong")

    hit_nrm = np.einsum('hk,hkd->hd', w, nrm[corners].astype(np.float64))
    hit_nrm /= np.maximum(np.linalg.norm(hit_nrm, axis=1, keepdims=True), 1e-16)
    hit_open = pick_opening(opening_flat[corners])

    h_idx = np.flatnonzero(hit)
    # Going inward, the first sheet is met from behind. A front face means the
    # ray started in a fold or self-overlap, which is not a cap-to-cap gap.
    front = np.einsum('hd,hd->h', dirs[hit], hit_nrm) < 0.0
    surface = ~front & (hit_open < 0)
    own = ~front & (hit_open == src_open[hit])
    other = ~front & (hit_open >= 0) & (hit_open != src_open[hit])
    too_far = other & (args.max_gap > 0) & (gap[hit] > args.max_gap)
    cand = other & ~too_far

    outcome[h_idx[front]] = OUTCOMES.index('frontface')
    outcome[h_idx[surface]] = OUTCOMES.index('surface')
    outcome[h_idx[own]] = OUTCOMES.index('self')
    outcome[h_idx[too_far]] = OUTCOMES.index('too_far')

    # Coarse meshes for winding numbers: the whole fitted surface (cap side
    # check) and its white part, i.e. the target surface (empty space test).
    stride = args.winding_stride
    if stride <= 0:
        full = P * 2 * (R - 1) ** 2
        stride = max(1, int(np.ceil(np.sqrt(full / args.winding_max_tris))))
    all_tris = ray_mesh_triangles(P, R, stride)
    if vol < 0:
        all_tris = all_tris[:, [0, 2, 1]]    # match the flipped normals
    w_tris = all_tris[white_flat[all_tris].all(axis=1)]

    sides = cap_sides(sources, xyz, nrm, all_tris, 0.5 * edge, args.device, 256, rng)
    inverted = {o for o, (behind, front) in sides.items()
                if behind < 0.5 and front < -0.5}
    if inverted:
        print(f"  [warn] Opening(s) {sorted(inverted)} sit on a region the fit folded "
              f"INSIDE-OUT (winding ~0 behind, ~-1 in front). Not a handle cap.")

    air_frac = np.full(src.size, np.nan)
    if args.no_air_test:
        outcome[h_idx[cand]] = OUTCOMES.index('vote')
        print("  Empty-space test: SKIPPED (--no_air_test)")
    elif cand.any():
        print(f"  Empty-space test: {cand.sum():,} candidate segments, winding number "
              f"of {w_tris.shape[0]:,} white triangles (stride {stride})")
        t0 = time.time()
        is_air, frac_air = segment_in_air(
            xyz[src[h_idx[cand]]].astype(np.float64), hit_xyz[cand],
            w_tris, xyz, args.device, args.air_samples, args.air_min)
        air_frac[h_idx[cand]] = frac_air
        outcome[h_idx[cand][is_air]] = OUTCOMES.index('vote')
        outcome[h_idx[cand][~is_air]] = OUTCOMES.index('solid')
        print(f"    {is_air.sum():,} through air, {(~is_air).sum():,} through solid "
              f"[{time.time() - t0:.1f}s]")

    # Per-hit bookkeeping for the votes, indexed like the rays.
    hit_open_all = np.full(src.size, -1, np.int64)
    hit_open_all[h_idx] = hit_open
    hit_xyz_all = np.full((src.size, 3), np.nan)
    hit_xyz_all[h_idx] = hit_xyz
    hit_nrm_all = np.full((src.size, 3), np.nan)
    hit_nrm_all[h_idx] = hit_nrm
    corners_all = np.full((src.size, 3), -1, np.int64)
    corners_all[h_idx] = corners
    w_all = np.full((src.size, 3), np.nan)
    w_all[h_idx] = w

    vote = outcome == OUTCOMES.index('vote')
    votes = np.zeros((n_openings, n_openings), np.int64)
    np.add.at(votes, (src_open[vote], hit_open_all[vote]), 1)

    # pairing
    pairs, candidates, frac, score = match_openings(
        votes, n_rays, crossings if have_crossings else np.zeros_like(crossings),
        args.min_frac, args.min_crossings if have_crossings else 0)
    paired = {o for p in pairs for o in (p['a'], p['b'])}

    outcome_counts = np.zeros((n_openings, len(OUTCOMES)), np.int64)
    np.add.at(outcome_counts, (src_open, outcome), 1)

    def unpaired_reason(o):
        """Why an opening found no partner, from where its rays went."""
        if o in inverted:
            return ('the fit folded inside-out here: winding behind '
                    f'{sides[o][0]:+.2f}, in front {sides[o][1]:+.2f} (fix the fit, '
                    'there is nothing to stitch)')
        if any(o in (c['a'], c['b']) for c in candidates):
            return 'its candidate partner joined a stronger pair'
        cnt = dict(zip(OUTCOMES, outcome_counts[o].tolist()))
        cnt.pop('vote')
        top = max(cnt, key=cnt.get)
        return {
            'solid': 'rays reach another opening only through solid material '
                     '(two unrelated holes, or a boundary)',
            'surface': 'rays hit ordinary surface: no facing cap (likely a boundary '
                       'opening or a fitting gap)',
            'self': 'rays land back in the same opening (a fold)',
            'miss': 'rays escape the surface (orientation or open sheet)',
            'frontface': 'rays start inside a self-overlap',
            'too_far': 'facing cap is beyond --max_gap',
        }[top] + f" [{top}: {cnt[top]}/{int(n_rays[o])}]"

    # Correspondences of one pair, from both directions, canonical a < b.
    origin_norm = xyz[src].astype(np.float64)
    src_uv = grid_uv[src % (R * R)].astype(np.float64)
    src_patch = np.asarray(patch_ids, np.int64)[src // (R * R)]

    def pair_rows(a, b):
        rows = {k: [] for k in ('patch_a', 'uv_a', 'xyz_a', 'patch_b', 'uv_b',
                                'xyz_b', 'gap', 'normal_dot', 'air_fraction',
                                'src_is_a')}
        for s, t, src_is_a in ((a, b, True), (b, a, False)):
            sel = np.flatnonzero(vote & (src_open == s) & (hit_open_all == t))
            if sel.size == 0:
                continue
            c = corners_all[sel]
            uv_hit = np.einsum('nk,nkd->nd', w_all[sel],
                               grid_uv[c % (R * R)].astype(np.float64))
            patch_hit = np.asarray(patch_ids, np.int64)[c[:, 0] // (R * R)]
            ndot = np.einsum('nd,nd->n', nrm[src[sel]].astype(np.float64),
                             hit_nrm_all[sel])
            side_src = (src_patch[sel], src_uv[sel], origin_norm[sel])
            side_hit = (patch_hit, uv_hit, hit_xyz_all[sel])
            first, second = (side_src, side_hit) if src_is_a else (side_hit, side_src)
            for key, val in zip(('patch_a', 'uv_a', 'xyz_a'), first):
                rows[key].append(val)
            for key, val in zip(('patch_b', 'uv_b', 'xyz_b'), second):
                rows[key].append(val)
            rows['gap'].append(gap[sel])
            rows['normal_dot'].append(ndot)
            rows['air_fraction'].append(air_frac[sel])
            rows['src_is_a'].append(np.full(sel.size, src_is_a))
        return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}

    
    print(f"\n{'─' * 72}")
    print(f"  {len(pairs)} pair(s) from {n_openings} openings "
          f"(min_frac={args.min_frac}"
          + (f", min_crossings={args.min_crossings}" if have_crossings else "") + ")")
    print(f"\n    {'pair':>9}  {'score':>6}  {'a→b':>6}  {'b→a':>6}  {'cross':>5}  "
          f"{'gap med':>8}  {'n·n med':>7}  evidence")
    pair_dir = os.path.join(args.out_dir, 'pairs')
    os.makedirs(pair_dir, exist_ok=True)
    # Pair numbering changes between runs, so files from an earlier run would
    # sit next to this run's and be read as current pairs.
    for stale in glob.glob(os.path.join(pair_dir, 'pair*_o*_o*.npz')):
        os.remove(stale)
    for k, p in enumerate(pairs):
        a, b = p['a'], p['b']
        rows = pair_rows(a, b)
        p['n_correspondences'] = int(rows['gap'].size)
        p['gap'] = ({'median': float(np.median(rows['gap'])),
                     'min': float(rows['gap'].min()),
                     'max': float(rows['gap'].max())} if rows['gap'].size else None)
        p['normal_dot_median'] = (float(np.median(rows['normal_dot']))
                                  if rows['normal_dot'].size else None)
        gap_s = f"{p['gap']['median']:.4f}" if p['gap'] else '--'
        nd_s = f"{p['normal_dot_median']:+.3f}" if p['normal_dot_median'] is not None else '--'
        print(f"    {a:>3} ↔ {b:<3}  {p['score']:6.3f}  {p['frac_ab']:6.3f}  "
              f"{p['frac_ba']:6.3f}  {p['crossings']:5d}  {gap_s:>8}  {nd_s:>7}  "
              f"{p['evidence']}")
        if p['runner_up'] and p['runner_up']['score'] > 0.5 * p['score'] > 0:
            ru = p['runner_up']
            print(f"              [warn] close rival {ru['openings']} "
                  f"(score {ru['score']:.3f})")

        stem = f"pair{k:02d}_o{a}_o{b}"
        p['file'] = os.path.join('pairs', stem + '.npz')
        np.savez_compressed(
            os.path.join(pair_dir, stem + '.npz'),
            opening_a=np.int32(a), opening_b=np.int32(b),
            score=np.float64(p['score']), crossings=np.int64(p['crossings']),
            evidence=p['evidence'],
            patch_a=rows['patch_a'].astype(np.int32), uv_a=rows['uv_a'].reshape(-1, 2),
            xyz_a=rows['xyz_a'].reshape(-1, 3),
            xyz_a_original=rows['xyz_a'].reshape(-1, 3) * scale + center,
            patch_b=rows['patch_b'].astype(np.int32), uv_b=rows['uv_b'].reshape(-1, 2),
            xyz_b=rows['xyz_b'].reshape(-1, 3),
            xyz_b_original=rows['xyz_b'].reshape(-1, 3) * scale + center,
            gap=rows['gap'], normal_dot=rows['normal_dot'],
            air_fraction=rows['air_fraction'], src_is_a=rows['src_is_a'].astype(bool))

    unpaired = [{'opening': o, 'n_rays': int(n_rays[o]), 'reason': unpaired_reason(o)}
                for o in range(n_openings) if o not in paired]
    if unpaired:
        print(f"\n  Unpaired ({len(unpaired)}):")
        for u in unpaired:
            print(f"    opening {u['opening']:>3}  ({u['n_rays']:>5} rays)  {u['reason']}")
    print(f"\n  An unpaired opening is not necessarily a mistake: a real boundary "
          f"(an open edge of the target) has no partner by design.")

    # outputs
    colors = opening_colors(max(len(pairs), 1))
    pair_of = {}
    for k, p in enumerate(pairs):
        pair_of[p['a']] = k
        pair_of[p['b']] = k
    black = np.flatnonzero(opening_flat >= 0)
    rgb = np.tile(np.array(COLOR_UNPAIRED, np.uint8), (black.size, 1))
    for o, k in pair_of.items():
        rgb[opening_flat[black] == o] = colors[k]
    write_colored_ply(os.path.join(args.out_dir, 'paired_openings.ply'),
                      to_frame(xyz[black].astype(np.float64)), rgb)

    # Cap-to-cap segments as two endpoint clouds in the pair's colour, so the
    # gap each pair bridges can be seen next to paired_openings.ply.
    seg_pts, seg_rgb = [], []
    for k, p in enumerate(pairs):
        for s, t in ((p['a'], p['b']), (p['b'], p['a'])):
            sel = np.flatnonzero(vote & (src_open == s) & (hit_open_all == t))
            seg_pts += [origin_norm[sel], hit_xyz_all[sel]]
            seg_rgb += [np.tile(colors[k], (2 * sel.size, 1))]
    if seg_pts:
        write_colored_ply(os.path.join(args.out_dir, 'pair_ray_endpoints.ply'),
                          to_frame(np.concatenate(seg_pts)), np.concatenate(seg_rgb))

    save_score_matrix(os.path.join(args.out_dir, 'pairing_matrix.png'),
                      score, crossings, pairs)

    pair_map = pair_id_map(od['opening_id'], n_openings, pairs)
    save_pairing_atlas(os.path.join(args.out_dir, 'pairing_atlas.png'), pair_map,
                       od['opening_id'], n_openings, patch_ids, F, is_adaptive,
                       atlas_mode, pairs, colors, args.face_resolution)

    np.savez_compressed(
        os.path.join(args.out_dir, 'pairing.npz'),
        votes=votes, n_rays=n_rays, frac=frac, score=score, crossings=crossings,
        pairs=np.array([[p['a'], p['b']] for p in pairs], np.int32).reshape(-1, 2),
        # UV view, indexed like openings.npz: pair_id[patch_position, u_index, v_index]
        patch_ids=np.asarray(patch_ids, np.int32), resolution=np.int32(R),
        pair_id=pair_map,
        outcome_names=np.array(OUTCOMES), outcome_counts=outcome_counts,
        normalization_center=center, normalization_scale=np.float64(scale))

    summary = {
        'checkpoint': os.path.abspath(args.ckpt),
        'mask_npz': os.path.abspath(args.mask_npz),
        'openings_npz': os.path.abspath(args.openings_npz),
        'intersections_npz': (os.path.abspath(args.intersections_npz)
                              if have_crossings else None),
        'adaptive': bool(is_adaptive),
        'resolution': R,
        'n_openings': n_openings,
        'settings': {k: getattr(args, k) for k in (
            'mesh_stride', 'ray_stride', 'max_rays_per_opening', 'local_steps', 'max_gap',
            'air_samples', 'air_min', 'winding_stride', 'no_air_test',
            'min_frac', 'min_crossings', 'seed')},
        'enclosed_volume': vol,
        'median_casting_edge': edge,
        'barycentric_check_max_error': bary_err,
        'pairs': pairs,
        'unpaired': unpaired,
        'per_opening': [{'opening': o, 'n_rays': int(n_rays[o]),
                         'outcomes': dict(zip(OUTCOMES, outcome_counts[o].tolist())),
                         'winding_behind': sides[o][0], 'winding_front': sides[o][1],
                         'inside_out': o in inverted}
                        for o in range(n_openings)],
        'convention': 'pairs/pairNN_oA_oB.npz row i: (patch_a[i], uv_a[i]) on opening a '
                      'faces (patch_b[i], uv_b[i]) on opening b across gap[i]; '
                      'src_is_a says which side cast the ray',
        'pair_id_convention': 'pairing.npz pair_id[patch_position, u_index, v_index]: '
                              'k = pair k (index into pairs), -2 = unpaired opening, '
                              '-1 = not a hole; patch_position indexes patch_ids',
    }
    with open(os.path.join(args.out_dir, 'pairing.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    frame = 'original' if args.unnormalize else 'normalized'
    print(f"\n  Outputs → {args.out_dir}")
    print(f"    pairing.json               — pairs, scores, unpaired reasons (START HERE)")
    print(f"    pairing_atlas.png          — the pairing in UV: one colour per pair")
    print(f"    pairing_matrix.png         — mutual ray score, pairs boxed")
    print(f"    pairs/pairNN_oA_oB.npz     — ONE PAIR per file: facing points with")
    print(f"                                 patch + uv on each side, ready to stitch")
    print(f"    paired_openings.ply        — openings coloured by pair ({frame})")
    print(f"    pair_ray_endpoints.ply     — both ends of every voting ray ({frame})")
    print(f"    pairing.npz                — vote matrix, per-opening ray outcomes, and")
    print(f"                                 pair_id per UV sample")
    print(f"{'─' * 72}\n")


if __name__ == '__main__':
    main()
