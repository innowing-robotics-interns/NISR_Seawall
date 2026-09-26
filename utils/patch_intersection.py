#!/usr/bin/env python3
# utils/patch_intersection.py

import argparse
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

try:
    from . import patch_vis
    from .uv_hole_mask import sample_patch_uv_grid, write_colored_ply
except ImportError:
    import patch_vis
    from uv_hole_mask import sample_patch_uv_grid, write_colored_ply

from model.model import FACE_NAMES  # noqa: E402


COLOR_INTERSECTION = (235, 40, 40)


# ── per-patch meshes ────────────────────────────────────────────────────────
def grid_triangles(resolution: int) -> np.ndarray:
    """
    Triangulate a `resolution x resolution` UV grid flattened as `[u_index, v_index]`.

    Matches the ordering `sample_patch_uv_grid` produces, so vertex k of the
    returned faces indexes both the xyz array and the flattened mask.
    """
    flat = np.arange(resolution * resolution).reshape(resolution, resolution)
    a = flat[:-1, :-1].ravel()      # (u  , v  )
    b = flat[1:, :-1].ravel()       # (u+1, v  )
    c = flat[1:, 1:].ravel()        # (u+1, v+1)
    d = flat[:-1, 1:].ravel()       # (u  , v+1)
    return np.concatenate([np.stack([a, b, c], axis=1),
                           np.stack([a, c, d], axis=1)], axis=0).astype(np.int32)


def vertex_normals(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals (area weighting falls out of not
    normalizing the face cross-products before accumulating)."""
    fn = np.cross(xyz[faces[:, 1]] - xyz[faces[:, 0]],
                  xyz[faces[:, 2]] - xyz[faces[:, 0]])
    n = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-16)


def build_patch_meshes(F, patch_ids, resolution, device, batch_size=8192):
    """
    Sample every patch on the same RxR grid used for the masks.

    Returns:
        dict `patch_id -> {'xyz', 'uv', 'faces', 'normals', 'edge_len'}`.
    """
    faces = grid_triangles(resolution)
    meshes = {}
    for patch_id in patch_ids:
        xyz, uv = sample_patch_uv_grid(F, patch_id, resolution, device, batch_size)
        xyz = xyz.astype(np.float64)
        e = np.linalg.norm(xyz[faces[:, 1]] - xyz[faces[:, 0]], axis=1)
        meshes[int(patch_id)] = {
            'xyz': xyz,
            'uv': uv.astype(np.float64),
            'faces': faces,
            'normals': vertex_normals(xyz, faces),
            'edge_len': float(np.median(e)) if e.size else 0.0,
        }
    return meshes


def barycentric_uv(mesh, tri_idx, loc):
    """
    Exact UV of a point lying on a known triangle.

    The source side of a crossing gets its UV for free (the hit is at a known
    fraction along an edge whose endpoints have known UV). The TARGET side needs
    this: solve the hit's barycentric weights within the triangle it landed on,
    then blend that triangle's three UV corners with the same weights. Replaces
    the earlier triangle-centroid approximation, which was only accurate to one
    grid cell -- too coarse to drive a stitch.
    """
    f = mesh['faces'][tri_idx]
    v, uv = mesh['xyz'], mesh['uv']
    a = v[f[:, 0]]
    e0 = v[f[:, 1]] - a
    e1 = v[f[:, 2]] - a
    rel = loc - a

    d00 = np.einsum('ij,ij->i', e0, e0)
    d01 = np.einsum('ij,ij->i', e0, e1)
    d11 = np.einsum('ij,ij->i', e1, e1)
    d20 = np.einsum('ij,ij->i', rel, e0)
    d21 = np.einsum('ij,ij->i', rel, e1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-20, 1.0, denom)   # degenerate triangle
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2

    return (uv[f[:, 0]] * w0[:, None]
            + uv[f[:, 1]] * w1[:, None]
            + uv[f[:, 2]] * w2[:, None])


def black_triangle_ids(mask_flat: np.ndarray, faces: np.ndarray, rule: str) -> np.ndarray:
    """
    Triangles overlapping the hole region. `mask_flat` is True where a
    correspondence EXISTS, so a hole vertex is `~mask_flat`.

    'any'  — the triangle touches the hole (default; keeps the rim, which is
             where the two sheets actually cross).
    'all'  — the triangle lies entirely inside the hole.
    """
    hole = ~mask_flat
    hit = hole[faces]
    return np.flatnonzero(hit.all(axis=1) if rule == 'all' else hit.any(axis=1))


def unique_edges(faces: np.ndarray) -> np.ndarray:
    """
    Undirected edges of a triangle set, deduplicated.

    Interior edges are shared by two triangles; casting each once halves the
    ray count without changing the result.
    """
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


# ── seam network (false-positive guard) ─────────────────────────────────────
def build_seam_tree(meshes, patch_ids, resolution):
    """
    KD-tree over every patch's four boundary curves.

    Adjacent patches share these curves exactly, so any "crossing" sitting on
    one is C0 continuity, not a self-intersection.
    """
    flat = np.arange(resolution * resolution).reshape(resolution, resolution)
    border = np.unique(np.concatenate([flat[0, :], flat[-1, :],
                                       flat[:, 0], flat[:, -1]]))
    pts = [meshes[int(p)]['xyz'][border] for p in patch_ids]
    return cKDTree(np.concatenate(pts, axis=0))


# ── ray casting ─────────────────────────────────────────────────────────────
def build_accel(verts, faces):
    """
    Broad-phase accelerator for one patch mesh: triangle edge vectors plus a
    cKDTree over centroids. Built once per patch and reused across every pair,
    since rebuilding it per pair dominates the runtime otherwise.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    centroid = (v0 + v1 + v2) / 3.0
    tri_radius = float(np.max(np.linalg.norm(
        np.stack([v0, v1, v2], axis=1) - centroid[:, None, :], axis=2))) \
        if faces.shape[0] else 0.0
    return {'v0': v0, 'e1': v1 - v0, 'e2': v2 - v0,
            'tree': cKDTree(centroid), 'tri_radius': tri_radius}


def _dedup_hits(ray_idx, tri_idx, t, tol=1e-9):
    """
    Collapse hits that are the same crossing seen twice.

    The barycentric test `u >= 0 and v >= 0 and u + v <= 1` is boundary-inclusive,
    so a hit landing exactly ON an edge shared by two triangles is claimed by BOTH
    of them. Embree's watertight traversal reports such a hit once; this makes the
    numpy path agree. Sorting by (ray, t) puts the duplicates adjacent, so one
    pass over neighbours is enough.
    """
    if ray_idx.size == 0:
        return ray_idx, tri_idx, t
    order = np.lexsort((tri_idx, t, ray_idx))
    ray_idx, tri_idx, t = ray_idx[order], tri_idx[order], t[order]
    first = np.ones(ray_idx.size, dtype=bool)
    first[1:] = (ray_idx[1:] != ray_idx[:-1]) | (np.abs(t[1:] - t[:-1]) > tol)
    return ray_idx[first], tri_idx[first], t[first]


def _segment_hits_numpy(orig, dirv, accel, eps_t, backface=True):
    """
    Vectorized Moller-Trumbore, restricted to the SEGMENT [0, 1] of each ray.

    The cKDTree over triangle centroids provides the broad phase, so only the
    handful of triangles near each short edge is actually tested. Returns flat
    arrays over surviving (ray, triangle) pairs.

    Returns:
        Tuple `(ray_idx, tri_idx, t, locations)` where `t` is the normalized
        parameter along the edge, already clamped to `(eps_t, 1 - eps_t)`.
    """
    v0, e1, e2 = accel['v0'], accel['e1'], accel['e2']
    mid = orig + 0.5 * dirv
    half = 0.5 * np.linalg.norm(dirv, axis=1)
    cand = accel['tree'].query_ball_point(mid, half + accel['tri_radius'])

    counts = np.fromiter((len(c) for c in cand), dtype=np.int64, count=len(cand))
    if counts.sum() == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0), np.zeros((0, 3))

    ray_idx = np.repeat(np.arange(len(cand), dtype=np.int64), counts)
    tri_idx = np.fromiter((t for c in cand for t in c), dtype=np.int64,
                          count=int(counts.sum()))

    O = orig[ray_idx]
    D = dirv[ray_idx]
    V0, E1, E2 = v0[tri_idx], e1[tri_idx], e2[tri_idx]

    P = np.cross(D, E2)
    det = np.einsum('ij,ij->i', E1, P)
    # Near-zero determinant = ray parallel to the triangle plane. Those are the
    # coincident-geometry cases the seam guard also targets; drop them here so
    # the divide stays finite.
    live = np.abs(det) > 1e-14
    if not backface:
        live &= det > 0
    if not live.any():
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0), np.zeros((0, 3))

    ray_idx, tri_idx = ray_idx[live], tri_idx[live]
    O, D, V0, E1, E2 = O[live], D[live], V0[live], E1[live], E2[live]
    P, det = P[live], det[live]

    inv = 1.0 / det
    T = O - V0
    u = np.einsum('ij,ij->i', T, P) * inv
    Q = np.cross(T, E1)
    v = np.einsum('ij,ij->i', D, Q) * inv
    t = np.einsum('ij,ij->i', E2, Q) * inv

    # Barycentric inside-test, plus the segment clamp that makes this an EDGE
    # test rather than an infinite-ray test.
    keep = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > eps_t) & (t < 1.0 - eps_t)
    ray_idx, tri_idx, t = _dedup_hits(ray_idx[keep], tri_idx[keep], t[keep])
    loc = orig[ray_idx] + t[:, None] * dirv[ray_idx]
    return ray_idx, tri_idx, t, loc


def _segment_hits_trimesh(orig, dirv, mesh, eps_t):
    """
    Same test through `trimesh`'s `intersects_id`, when an accelerated ray
    backend is available. `intersects_id` casts INFINITE rays, so the returned
    locations are re-projected onto the edge and clamped to (eps_t, 1 - eps_t)
    exactly as in the numpy path.
    """
    tri_idx, ray_idx, loc = mesh.ray.intersects_id(
        ray_origins=orig, ray_directions=dirv,
        return_locations=True, multiple_hits=True)
    if len(ray_idx) == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0), np.zeros((0, 3))

    d = dirv[ray_idx]
    t = np.einsum('ij,ij->i', loc - orig[ray_idx], d) / np.einsum('ij,ij->i', d, d)
    keep = (t > eps_t) & (t < 1.0 - eps_t)
    return ray_idx[keep], tri_idx[keep], t[keep], loc[keep]


def make_ray_backend(name: str):
    """Resolve --backend, falling back to numpy when trimesh has no ray accel."""
    if name == 'numpy':
        return 'numpy', None
    try:
        import trimesh
        # Capability probe: trimesh picks its ray backend lazily and a missing
        # dependency (rtree / embreex) only raises when a ray is actually cast,
        # never at import. So fire one — with the SAME keywords the real call
        # uses, so the probe exercises the same code path and the same 3-tuple
        # return shape that _segment_hits_trimesh unpacks.
        probe = trimesh.creation.box()
        got = probe.ray.intersects_id(
            ray_origins=np.array([[-3.0, 0.0, 0.0]]),
            ray_directions=np.array([[1.0, 0.0, 0.0]]),
            return_locations=True, multiple_hits=True)
        if len(got) != 3:
            raise RuntimeError(
                f"intersects_id(return_locations=True) returned {len(got)} values, "
                f"expected 3 (tri_idx, ray_idx, locations)")
        return 'trimesh', trimesh
    except Exception as exc:
        if name == 'trimesh':
            raise RuntimeError(
                f"--backend trimesh requested but trimesh ray casting is unavailable "
                f"({type(exc).__name__}: {exc}). Install a ray backend "
                f"('pip install embreex', or 'pip install rtree' for the slow "
                f"pure-python one), or use --backend numpy.")
        print(f"  [info] trimesh ray backend unavailable ({type(exc).__name__}); "
              f"using the built-in numpy Moller-Trumbore path.")
        return 'numpy', None


# ── main detection ──────────────────────────────────────────────────────────
def detect_intersections(meshes, black_tris, patch_ids, seam_tree, seam_eps,
                         eps_t, backend, trimesh_mod, include_self, verbose=True):
    """
    Ordered-pair sweep: edges of each patch's hole triangles against every other
    patch's full mesh.

    Returns:
        List of per-hit dicts.
    """
    # Patch AABBs, padded, so obviously-disjoint pairs are skipped outright.
    boxes = {}
    for p in patch_ids:
        xyz = meshes[p]['xyz']
        boxes[p] = (xyz.min(axis=0) - seam_eps, xyz.max(axis=0) + seam_eps)

    # Built once per patch, reused for every pair it takes part in.
    accels, tri_meshes = {}, {}
    for p in patch_ids:
        if backend == 'trimesh':
            tri_meshes[p] = trimesh_mod.Trimesh(vertices=meshes[p]['xyz'],
                                                faces=meshes[p]['faces'],
                                                process=False)
        else:
            accels[p] = build_accel(meshes[p]['xyz'], meshes[p]['faces'])

    hits = []
    n_pairs = 0
    for src in patch_ids:
        tri_ids = black_tris[src]
        if tri_ids.size == 0:
            continue
        edges = unique_edges(meshes[src]['faces'][tri_ids])
        xyz_src = meshes[src]['xyz']
        orig = xyz_src[edges[:, 0]]
        dirv = xyz_src[edges[:, 1]] - orig
        alive = np.linalg.norm(dirv, axis=1) > 1e-12
        edges, orig, dirv = edges[alive], orig[alive], dirv[alive]
        if edges.shape[0] == 0:
            continue

        ends = orig + dirv
        lo_s = np.minimum(orig.min(axis=0), ends.min(axis=0))
        hi_s = np.maximum(orig.max(axis=0), ends.max(axis=0))

        for dst in patch_ids:
            if dst == src and not include_self:
                continue
            lo_d, hi_d = boxes[dst]
            if np.any(hi_s < lo_d) or np.any(hi_d < lo_s):
                continue
            n_pairs += 1

            if backend == 'trimesh':
                r_i, t_i, t_par, loc = _segment_hits_trimesh(
                    orig, dirv, tri_meshes[dst], eps_t)
            else:
                r_i, t_i, t_par, loc = _segment_hits_numpy(
                    orig, dirv, accels[dst], eps_t)
            if r_i.size == 0:
                continue

            # Reject crossings sitting on the shared C0 seam network.
            seam_d, _ = seam_tree.query(loc, k=1, workers=-1)
            ok = seam_d > seam_eps
            if not ok.any():
                continue
            r_i, t_i, t_par, loc, seam_d = r_i[ok], t_i[ok], t_par[ok], loc[ok], seam_d[ok]

            # UV of the crossing: linear along the source edge, since the edge
            # endpoints are grid samples with known uv.
            uv_src = (meshes[src]['uv'][edges[r_i, 0]] * (1.0 - t_par)[:, None]
                      + meshes[src]['uv'][edges[r_i, 1]] * t_par[:, None])
            # UV on the target: exact, via the hit's barycentric weights inside
            # the triangle it landed on.
            uv_dst = barycentric_uv(meshes[dst], t_i, loc)

            # Diagnostic only: two sheets of a membrane face opposite ways, so a
            # genuine crossing has a strongly NEGATIVE dot. Recorded, not filtered.
            n_src = (meshes[src]['normals'][edges[r_i, 0]]
                     + meshes[src]['normals'][edges[r_i, 1]])
            n_src /= np.maximum(np.linalg.norm(n_src, axis=1, keepdims=True), 1e-16)
            n_dst = meshes[dst]['normals'][meshes[dst]['faces'][t_i]].mean(axis=1)
            n_dst /= np.maximum(np.linalg.norm(n_dst, axis=1, keepdims=True), 1e-16)
            dot_n = np.einsum('ij,ij->i', n_src, n_dst)

            for k in range(r_i.size):
                hits.append({
                    'src_patch': int(src), 'dst_patch': int(dst),
                    'dst_triangle': int(t_i[k]),
                    # Grid-vertex endpoints of the source edge. Kept so an
                    # opening ID can be read off the vertices the crossing
                    # actually sits on, rather than by rounding a continuous UV
                    # (which can land in a white cell at the rim).
                    'src_edge': [int(edges[r_i[k], 0]), int(edges[r_i[k], 1])],
                    't': float(t_par[k]),
                    'location': loc[k].tolist(),
                    'uv_src': uv_src[k].tolist(),
                    'uv_dst': uv_dst[k].tolist(),
                    'seam_distance': float(seam_d[k]),
                    'normal_dot': float(dot_n[k]),
                })
            if verbose:
                print(f"    patch {src:3d} -> {dst:3d}   {r_i.size:6d} crossings   "
                      f"median normal dot = {np.median(dot_n):+.3f}")

    return hits, n_pairs


# ── grouping by patch pair ──────────────────────────────────────────────────
def group_by_pair(hits):
    """
    Reorganize the flat hit list into one group per PAIR of patches.

    Each crossing is really a CORRESPONDENCE: a point in one patch's UV domain
    and the matching point in the other's, both naming the same 3D location.
    That is exactly the input a stitch needs, so it is worth storing as such.

    Two bookkeeping details:

    * The sweep tests ordered pairs, so a crossing between patches 7 and 40 can
      arrive as (src=7, dst=40) or (src=40, dst=7). Both are folded onto the
      canonical unordered pair (7, 40) with `a < b`, and the UVs are swapped to
      match, so `uv_a` is always in patch `a`.
    * `src_is_a` records which side was the ray's source. Both UVs are exact,
      but the source side comes from a 1D interpolation along an edge and the
      target side from a 2D barycentric solve inside a triangle -- worth knowing
      if you ever need to weight one over the other.

    Returns:
        List of dicts sorted by decreasing crossing count, one per pair.
    """
    buckets = {}
    for h in hits:
        a, b = h['src_patch'], h['dst_patch']
        src_is_a = a <= b
        key = (a, b) if src_is_a else (b, a)
        rec = buckets.setdefault(key, {'uv_a': [], 'uv_b': [], 'xyz': [],
                                       'normal_dot': [], 'seam_distance': [],
                                       't': [], 'src_is_a': []})
        rec['uv_a'].append(h['uv_src'] if src_is_a else h['uv_dst'])
        rec['uv_b'].append(h['uv_dst'] if src_is_a else h['uv_src'])
        rec['xyz'].append(h['location'])
        rec['normal_dot'].append(h['normal_dot'])
        rec['seam_distance'].append(h['seam_distance'])
        rec['t'].append(h['t'])
        rec['src_is_a'].append(bool(src_is_a))

    groups = []
    for (a, b), rec in buckets.items():
        g = {'patch_a': int(a), 'patch_b': int(b),
             'n_crossings': len(rec['xyz'])}
        for k, dt in (('uv_a', np.float64), ('uv_b', np.float64),
                      ('xyz', np.float64), ('normal_dot', np.float64),
                      ('seam_distance', np.float64), ('t', np.float64),
                      ('src_is_a', bool)):
            g[k] = np.asarray(rec[k], dtype=dt)
        groups.append(g)

    groups.sort(key=lambda g: -g['n_crossings'])
    return groups


def curve_order(xyz):
    """
    Order points along the crossing curve.

    A crossing is a curve, not a blob, so its points are nearly 1D. Projecting
    onto their own principal direction gives a monotone coordinate along it.
    Used to colour both halves of a pair figure consistently, so you can SEE
    which end of patch a's curve corresponds to which end of patch b's -- the
    thing you need to get right when stitching.

    Returns:
        (n,) float in [0, 1], position along the curve.
    """
    if xyz.shape[0] < 2:
        return np.zeros(xyz.shape[0])
    centred = xyz - xyz.mean(axis=0)
    # Principal direction = first right-singular vector.
    direction = np.linalg.svd(centred, full_matrices=False)[2][0]
    s = centred @ direction
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


# ── grouping by OPENING pair ────────────────────────────────────────────────
def assign_openings(hits, opening_flat, faces):
    """
    Attach `opening_src` / `opening_dst` to every hit.

    The ID is read off the GRID VERTICES the crossing sits on -- the two
    endpoints of the source edge, the three corners of the target triangle --
    not by rounding the continuous UV to a cell. Rounding can land on a white
    vertex at the rim and lose the ID; the incident vertices never do, because a
    crossing inside the hole region always touches at least one black vertex.
    When several black vertices disagree (only possible exactly on a boundary
    between two openings) the majority wins. -1 means no incident black vertex,
    i.e. that side of the crossing is ordinary surface, not a membrane.
    """
    def pick(ids):
        ids = ids[ids >= 0]
        if ids.size == 0:
            return -1
        vals, counts = np.unique(ids, return_counts=True)
        return int(vals[np.argmax(counts)])

    for h in hits:
        e = np.asarray(h['src_edge'], dtype=np.int64)
        h['opening_src'] = pick(opening_flat[h['src_patch']][e])
        h['opening_dst'] = pick(opening_flat[h['dst_patch']][faces[h['dst_triangle']]])


def group_by_opening(hits):
    """
    Reorganize hits into one group per unordered pair of OPENING IDs.

    This is the grouping stitching actually wants. A patch pair is an artefact
    of where the atlas happened to cut the surface; an opening is a whole hole,
    however many patches it straddles. Two openings that cross each other are
    the two ends of one handle, and every crossing between them -- from
    whichever patches -- belongs to the same stitch.

    Because an opening spans several patches, each correspondence row carries
    its own patch on both sides: row i says "(patch_a[i], uv_a[i]) in opening a
    is the same 3D point as (patch_b[i], uv_b[i]) in opening b".

    Groups are classified:
      'pair'        a != b, both >= 0   -> a stitch candidate
      'self'        a == b              -> an opening folding through itself
      'unassigned'  a or b is -1        -> a membrane crossing ordinary surface
    Only 'pair' groups are stitch candidates; the other two are reported as
    diagnostics so nothing is silently dropped.
    """
    buckets = {}
    for h in hits:
        oa, ob = h['opening_src'], h['opening_dst']
        src_first = oa <= ob
        key = (oa, ob) if src_first else (ob, oa)
        rec = buckets.setdefault(key, {
            'patch_a': [], 'patch_b': [], 'uv_a': [], 'uv_b': [], 'xyz': [],
            'normal_dot': [], 'seam_distance': [], 't': [], 'src_is_a': []})
        rec['patch_a'].append(h['src_patch'] if src_first else h['dst_patch'])
        rec['patch_b'].append(h['dst_patch'] if src_first else h['src_patch'])
        rec['uv_a'].append(h['uv_src'] if src_first else h['uv_dst'])
        rec['uv_b'].append(h['uv_dst'] if src_first else h['uv_src'])
        rec['xyz'].append(h['location'])
        rec['normal_dot'].append(h['normal_dot'])
        rec['seam_distance'].append(h['seam_distance'])
        rec['t'].append(h['t'])
        rec['src_is_a'].append(bool(src_first))

    groups = []
    for (a, b), rec in buckets.items():
        if a < 0 or b < 0:
            kind = 'unassigned'
        elif a == b:
            kind = 'self'
        else:
            kind = 'pair'
        g = {'opening_a': int(a), 'opening_b': int(b), 'kind': kind,
             'n_crossings': len(rec['xyz'])}
        for k, dt in (('patch_a', np.int32), ('patch_b', np.int32),
                      ('uv_a', np.float64), ('uv_b', np.float64),
                      ('xyz', np.float64), ('normal_dot', np.float64),
                      ('seam_distance', np.float64), ('t', np.float64),
                      ('src_is_a', bool)):
            g[k] = np.asarray(rec[k], dtype=dt)
        # Which patch pairs fed this opening pair -- ties the two views together.
        pp = np.stack([np.minimum(g['patch_a'], g['patch_b']),
                       np.maximum(g['patch_a'], g['patch_b'])], axis=1)
        g['patch_pairs'] = np.unique(pp, axis=0)
        groups.append(g)

    order = {'pair': 0, 'self': 1, 'unassigned': 2}
    groups.sort(key=lambda g: (order[g['kind']], -g['n_crossings']))
    return groups


def opening_pairing(groups, n_openings):
    """
    Is the crossing structure a clean one-to-one pairing of openings?

    Raising the genus by g needs 2g openings matched into g disjoint pairs. So
    the ideal is: every opening crosses exactly ONE other opening. Anything else
    is a warning sign worth seeing before a stitch is attempted:
      unpaired   an opening that crosses nothing (spurious hole, or the crossing
                 was lost to the seam guard / a too-tight mask)
      ambiguous  an opening that crosses two or more others (two holes merged
                 into one ID, or a fold)

    Returns:
        dict with 'matched' (list of [a, b, n]), 'unpaired', 'ambiguous'.
    """
    partners = {o: {} for o in range(n_openings)}
    for g in groups:
        if g['kind'] != 'pair':
            continue
        a, b = g['opening_a'], g['opening_b']
        partners[a][b] = g['n_crossings']
        partners[b][a] = g['n_crossings']

    matched, unpaired, ambiguous, seen = [], [], [], set()
    for o in range(n_openings):
        p = partners[o]
        if not p:
            unpaired.append(o)
        elif len(p) > 1:
            ambiguous.append({'opening': o,
                              'crosses': sorted(p.items(), key=lambda kv: -kv[1])})
        else:
            (q, n), = p.items()
            if len(partners[q]) == 1 and (q, o) not in seen:
                matched.append([o, q, int(n)])
                seen.add((o, q))
    return {'matched': matched, 'unpaired': unpaired, 'ambiguous': ambiguous}


def save_opening_matrix(path, groups, n_openings):
    """Openings x openings: how many crossings each pair shares."""
    if n_openings == 0:
        return False
    M = np.zeros((n_openings, n_openings), np.int64)
    for g in groups:
        if g['kind'] == 'unassigned':
            continue
        a, b = g['opening_a'], g['opening_b']
        M[a, b] += g['n_crossings']
        if a != b:
            M[b, a] += g['n_crossings']

    side = min(12.0, 1.5 + 0.45 * n_openings)
    fig, ax = plt.subplots(figsize=(side + 1.2, side))
    im = ax.imshow(np.log1p(M), cmap='magma', interpolation='nearest')
    ax.set_xlabel('opening ID')
    ax.set_ylabel('opening ID')
    ax.set_title('Crossings between openings   (log scale; diagonal = self-fold)')
    if n_openings <= 30:
        ax.set_xticks(range(n_openings))
        ax.set_yticks(range(n_openings))
        for i in range(n_openings):
            for j in range(n_openings):
                if M[i, j]:
                    ax.text(j, i, str(M[i, j]), ha='center', va='center',
                            fontsize=7, color='white' if np.log1p(M[i, j]) < 0.6 * np.log1p(M).max() else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8, label='log(1 + crossings)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def opening_group_summary(g, tag):
    """Per-opening-pair statistics for intersections.json."""
    return {
        'openings': [g['opening_a'], g['opening_b']],
        'kind': g['kind'],
        'n_crossings': int(g['n_crossings']),
        'patch_pairs': [[int(a), int(b)] for a, b in g['patch_pairs']],
        'patch_pair_labels': [[tag(int(a)), tag(int(b))] for a, b in g['patch_pairs']],
        'normal_dot': {'median': float(np.median(g['normal_dot'])),
                       'min': float(g['normal_dot'].min()),
                       'max': float(g['normal_dot'].max())},
        'xyz_centroid': g['xyz'].mean(axis=0).tolist(),
        'xyz_extent': (g['xyz'].max(axis=0) - g['xyz'].min(axis=0)).tolist(),
    }


def pair_summary(g, tag):
    """Per-pair statistics for intersections.json."""
    def bbox(uv):
        return {'u_min': float(uv[:, 0].min()), 'u_max': float(uv[:, 0].max()),
                'v_min': float(uv[:, 1].min()), 'v_max': float(uv[:, 1].max())}
    return {
        'patches': [g['patch_a'], g['patch_b']],
        'labels': [tag(g['patch_a']), tag(g['patch_b'])],
        'n_crossings': int(g['n_crossings']),
        'normal_dot': {'median': float(np.median(g['normal_dot'])),
                       'min': float(g['normal_dot'].min()),
                       'max': float(g['normal_dot'].max())},
        'seam_distance_min': float(g['seam_distance'].min()),
        'uv_a_bbox': bbox(g['uv_a']),
        'uv_b_bbox': bbox(g['uv_b']),
        'uv_a_centroid': g['uv_a'].mean(axis=0).tolist(),
        'uv_b_centroid': g['uv_b'].mean(axis=0).tolist(),
        'xyz_centroid': g['xyz'].mean(axis=0).tolist(),
        'xyz_extent': (g['xyz'].max(axis=0) - g['xyz'].min(axis=0)).tolist(),
    }


def save_pair_figures(path, groups, tag, max_pairs=12):
    """
    One row per patch pair: patch a's UV curve beside patch b's.

    Both panels are coloured by position along the SAME 3D curve, so matching
    colours are matching points. That correspondence is what a stitch consumes.
    """
    shown = groups[:max_pairs]
    if not shown:
        return False
    fig, axes = plt.subplots(len(shown), 2,
                             figsize=(7.6, 3.1 * len(shown)), squeeze=False)
    for r, g in enumerate(shown):
        s = curve_order(g['xyz'])
        for c, (uv, pid) in enumerate(((g['uv_a'], g['patch_a']),
                                       (g['uv_b'], g['patch_b']))):
            ax = axes[r][c]
            ax.scatter(uv[:, 1], uv[:, 0], c=s, cmap='viridis', s=6,
                       linewidths=0, vmin=0.0, vmax=1.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{tag(pid)}   ({g["n_crossings"]} pts)', fontsize=8)
        axes[r][0].set_ylabel(f'pair {r}', fontsize=8)
    fig.suptitle('Crossing correspondence, per patch pair\n'
                 'same colour = same 3D point   (u down, v right)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.6)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ── reporting ───────────────────────────────────────────────────────────────
def save_uv_scatter(path, hits, tag):
    """Where the crossings land in each involved patch's UV domain."""
    by_patch = {}
    for h in hits:
        by_patch.setdefault(h['src_patch'], []).append(h['uv_src'])
        by_patch.setdefault(h['dst_patch'], []).append(h['uv_dst'])
    if not by_patch:
        return False

    ids = sorted(by_patch)
    n = len(ids)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows),
                             squeeze=False)
    for k, pid in enumerate(ids):
        ax = axes[k // cols][k % cols]
        pts = np.asarray(by_patch[pid])
        ax.scatter(pts[:, 1], pts[:, 0], s=2, c='#e02828', linewidths=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{tag(pid)}  ({len(pts)})', fontsize=8)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle('Self-intersection crossings in UV (u down, v right)\n'
                 'these are the curves to cut along', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=2.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(
        description='Detect self-intersections between patches inside the hole '
                    'regions found by uv_hole_mask.py, and report them in UV.')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--mask_npz', type=str, default=None,
                    help='uv_mask.npz from uv_hole_mask.py (default: '
                         '<ckpt dir>/uv_masks/<ckpt name>/uv_mask.npz)')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='Default: <mask_npz dir>/intersections')
    ap.add_argument('--openings_npz', type=str, default=None,
                    help='openings.npz from opening_labels.py. When present, '
                         'crossings are ALSO grouped by OPENING pair, which is '
                         'the unit stitching needs. Default: '
                         '<mask_npz dir>/openings/openings.npz if it exists.')
    ap.add_argument('--no_openings', action='store_true',
                    help='Skip the opening-level grouping even if openings.npz exists')
    ap.add_argument('--adaptive', type=str, default='auto',
                    choices=['auto', 'yes', 'no'])
    ap.add_argument('--resolution', type=int, default=None,
                    help='Mesh resolution per patch. Defaults to the mask '
                         'resolution so mask and mesh vertices line up.')
    ap.add_argument('--triangle_rule', type=str, default='any',
                    choices=['any', 'all'],
                    help="'any' (default) = triangle touches the hole, keeping "
                         "the rim where the sheets actually cross; 'all' = "
                         "triangle lies wholly inside the hole")
    ap.add_argument('--all_triangles', action='store_true',
                    help='Ignore the mask and test every triangle (slow; for '
                         'checking whether the mask missed the crossing)')
    ap.add_argument('--eps_t', type=float, default=1e-4,
                    help='Edge-parameter guard band. A hit counts only when '
                         'eps_t < t < 1-eps_t, which drops the shared-vertex '
                         'hits at the ends of every edge.')
    ap.add_argument('--seam_eps_scale', type=float, default=1.5,
                    help='Reject crossings within this many median triangle edge '
                         'lengths of the shared C0 patch seams')
    ap.add_argument('--include_self', action='store_true',
                    help='Also test a patch against itself (true self-overlap '
                         'within one patch)')
    ap.add_argument('--unnormalize', action='store_true',
                    help='Also write the crossings in ORIGINAL input coordinates '
                         '(p_orig = p_norm * scale + center, using the checkpoint\'s '
                         'stored normalization), so they overlay the raw input file '
                         'and mask.sh\'s --unnormalize exports in a viewer. The '
                         'normalized PLY is written either way.')
    ap.add_argument('--backend', type=str, default='auto',
                    choices=['auto', 'trimesh', 'numpy'],
                    help="'trimesh' uses intersects_id (needs embreex or rtree); "
                         "'numpy' uses the built-in vectorized Moller-Trumbore; "
                         "'auto' (default) prefers trimesh and falls back")
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--model_path', type=str, default=None)
    args = ap.parse_args()

    # ── inputs ──────────────────────────────────────────────────────────────
    if args.mask_npz is None:
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.mask_npz = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                     'uv_masks', ckpt_name, 'uv_mask.npz')
    if not os.path.exists(args.mask_npz):
        raise FileNotFoundError(
            f"No uv_mask.npz at {args.mask_npz}. Run utils/uv_hole_mask.py first.")
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(args.mask_npz)),
                                    'intersections')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n  Loading checkpoint: {args.ckpt}")
    if args.adaptive == 'auto':
        is_adaptive = patch_vis._is_adaptive_checkpoint(args.ckpt)
    else:
        is_adaptive = args.adaptive == 'yes'
    if is_adaptive:
        F, meta, ckpt_args, _, _ = patch_vis._load_adaptive_from_checkpoint(
            args.ckpt, args.device)
        print(f"  Atlas: adaptive quadtree cube, {F.n_patches} leaves")
    else:
        F, meta, ckpt_args, _, _ = patch_vis._load_model_from_checkpoint(
            args.ckpt, args.device, args.model_path)
        print(f"  Atlas: {ckpt_args.get('atlas_mode', 'single_sheet')}, "
              f"{F.n_patches} patches")

    mask_data = np.load(args.mask_npz)
    patch_ids = [int(p) for p in mask_data['patch_ids']]
    masks = mask_data['masks']
    R = int(args.resolution or int(mask_data['resolution']))
    if R != int(mask_data['resolution']):
        raise ValueError(
            f"--resolution {R} != mask resolution {int(mask_data['resolution'])}. "
            f"The mask is per-vertex, so the meshes must use the same grid.")
    # Everything in the npz is stacked in patch_ids ORDER, while the rest of this
    # script addresses things by patch ID. Keep the translation in one place.
    mask_of = {p: masks[i].ravel() for i, p in enumerate(patch_ids)}
    if is_adaptive:
        face_of = {p: int(mask_data['leaf_face'][i]) for i, p in enumerate(patch_ids)}
        depth_of = {p: int(mask_data['leaf_depth'][i]) for i, p in enumerate(patch_ids)}
    else:
        face_of = depth_of = {}

    def tag(pid):
        """Short human label for a patch in printouts."""
        if not is_adaptive:
            return f'p{pid}'
        return f'p{pid} {FACE_NAMES[face_of[pid]]} d{depth_of[pid]}'

    # ── opening IDs (optional) ──────────────────────────────────────────────
    opening_flat, n_openings = None, 0
    if not args.no_openings:
        if args.openings_npz is None:
            cand = os.path.join(os.path.dirname(os.path.abspath(args.mask_npz)),
                                'openings', 'openings.npz')
            args.openings_npz = cand if os.path.exists(cand) else None
        if args.openings_npz is not None:
            od = np.load(args.openings_npz)
            o_ids = [int(p) for p in od['patch_ids']]
            if o_ids != patch_ids or int(od['resolution']) != R:
                raise ValueError(
                    f"{args.openings_npz} does not match the mask: patch_ids or "
                    f"resolution differ. Re-run opening_labels.py on this mask.")
            n_openings = int(od['n_openings'])
            opening_flat = {p: od['opening_id'][i].ravel()
                            for i, p in enumerate(patch_ids)}
            print(f"  Openings: {n_openings} from {args.openings_npz}")
        else:
            print("  Openings: none (run opening_labels.py to also group crossings "
                  "by opening pair)")

    # ── meshes ──────────────────────────────────────────────────────────────
    print(f"\n  Sampling {len(patch_ids)} patches at {R}x{R} "
          f"({len(patch_ids) * 2 * (R - 1) ** 2:,} triangles total)...")
    t0 = time.time()
    meshes = build_patch_meshes(F, patch_ids, R, args.device)
    faces = meshes[patch_ids[0]]['faces']
    print(f"    done in {time.time() - t0:.1f}s")

    black_tris = {}
    for p in patch_ids:
        black_tris[p] = (np.arange(faces.shape[0]) if args.all_triangles
                         else black_triangle_ids(mask_of[p], faces, args.triangle_rule))
    involved = [p for p in patch_ids if black_tris[p].size]
    n_black = sum(int(black_tris[p].size) for p in patch_ids)
    print(f"  Hole triangles ({args.triangle_rule}): {n_black:,} across "
          f"{len(involved)} patches -> {involved if len(involved) <= 20 else '...'}")
    if n_black == 0:
        print("\n  Nothing segmented as a hole; no crossings to look for. "
              "Loosen uv_hole_mask.py's --nn_scale, or pass --all_triangles.")
        return

    med_edge = float(np.median([meshes[p]['edge_len'] for p in patch_ids]))
    seam_eps = args.seam_eps_scale * med_edge
    print(f"  Median triangle edge = {med_edge:.6f}   "
          f"seam guard = {seam_eps:.6f} ({args.seam_eps_scale} x)")

    seam_tree = build_seam_tree(meshes, patch_ids, R)
    backend, trimesh_mod = make_ray_backend(args.backend)
    print(f"  Ray backend: {backend}")

    # ── detect ──────────────────────────────────────────────────────────────
    print(f"\n  Casting edge rays (ordered pairs, both directions)...")
    t0 = time.time()
    hits, n_pairs = detect_intersections(
        meshes, black_tris, patch_ids, seam_tree, seam_eps, args.eps_t,
        backend, trimesh_mod, args.include_self)
    print(f"  {len(hits):,} crossings from {n_pairs} patch pairs tested "
          f"[{time.time() - t0:.1f}s]")

    # ── report ──────────────────────────────────────────────────────────────
    groups = group_by_pair(hits)

    print(f"\n{'─' * 64}")
    if not hits:
        print("  No self-intersection found in the segmented region.")
        print("  If you expect one: widen the region (--triangle_rule any, or a")
        print("  looser --nn_scale in uv_hole_mask.py), lower --seam_eps_scale,")
        print("  or run --all_triangles to check whether the mask missed it.")
    else:
        print(f"  {len(groups)} patch pair(s) cross — each is a stitch candidate.")
        print(f"  'uv extent' is the box the crossing curve spans in that "
              f"patch's own UV square.\n")
        for i, g in enumerate(groups):
            a, b = g['patch_a'], g['patch_b']
            ua, ub = g['uv_a'], g['uv_b']
            print(f"    [pair {i}]  {tag(a)}  <->  {tag(b)}")
            print(f"              {g['n_crossings']:6d} crossings   "
                  f"normal dot median {np.median(g['normal_dot']):+.3f} "
                  f"(min {g['normal_dot'].min():+.3f})")
            print(f"              uv extent in {tag(a):>14s}: "
                  f"u [{ua[:, 0].min():.3f}, {ua[:, 0].max():.3f}]  "
                  f"v [{ua[:, 1].min():.3f}, {ua[:, 1].max():.3f}]")
            print(f"              uv extent in {tag(b):>14s}: "
                  f"u [{ub[:, 0].min():.3f}, {ub[:, 0].max():.3f}]  "
                  f"v [{ub[:, 1].min():.3f}, {ub[:, 1].max():.3f}]")

    # ── opening-level view ──────────────────────────────────────────────────
    ogroups, pairing = [], None
    if opening_flat is not None and hits:
        assign_openings(hits, opening_flat, faces)
        ogroups = group_by_opening(hits)
        pairing = opening_pairing(ogroups, n_openings)

        n_pair = sum(1 for g in ogroups if g['kind'] == 'pair')
        n_self = sum(1 for g in ogroups if g['kind'] == 'self')
        n_unas = sum(1 for g in ogroups if g['kind'] == 'unassigned')
        print(f"\n{'─' * 64}")
        print(f"  BY OPENING: {n_pair} opening pair(s) cross"
              + (f", {n_self} self-fold(s)" if n_self else "")
              + (f", {n_unas} membrane-vs-surface group(s)" if n_unas else ""))
        for g in ogroups:
            a, b = g['opening_a'], g['opening_b']
            if g['kind'] == 'pair':
                head = f"    opening {a} <-> opening {b}"
            elif g['kind'] == 'self':
                head = f"    opening {a} through ITSELF"
            else:
                o = b if a < 0 else a
                head = f"    opening {o} <-> plain surface (no opening)"
            pp = ', '.join(f"{tag(int(x))}-{tag(int(y))}" for x, y in g['patch_pairs'][:6])
            if g['patch_pairs'].shape[0] > 6:
                pp += f", +{g['patch_pairs'].shape[0] - 6} more"
            print(f"{head:<48s} {g['n_crossings']:6d} crossings   "
                  f"normal dot {np.median(g['normal_dot']):+.3f}")
            print(f"        via patch pairs: {pp}")

        print(f"\n  Pairing check ({n_openings} openings):")
        if pairing['matched']:
            print(f"    matched 1-to-1: "
                  + ", ".join(f"({a},{b})" for a, b, _ in pairing['matched']))
        if pairing['unpaired']:
            print(f"    UNPAIRED (cross nothing): {pairing['unpaired']}")
        if pairing['ambiguous']:
            for amb in pairing['ambiguous']:
                print(f"    AMBIGUOUS: opening {amb['opening']} crosses "
                      + ", ".join(f"{q} ({n})" for q, n in amb['crosses']))
        if (pairing['matched'] and not pairing['unpaired']
                and not pairing['ambiguous']):
            print(f"    -> clean: every opening pairs with exactly one other. "
                  f"{len(pairing['matched'])} handle(s) to stitch.")

    loc = np.array([h['location'] for h in hits], dtype=np.float64) if hits \
        else np.zeros((0, 3))

    # The model works in the normalized frame; the raw input file does not.
    # p_orig = p_norm * scale + center recovers the input's own coordinates.
    center = np.asarray(meta['center'], dtype=np.float64)
    scale = float(meta['scale'])
    loc_orig = loc * scale + center
    if args.unnormalize and scale == 1.0 and not np.any(center):
        print("  [warn] --unnormalize requested but the checkpoint stores an "
              "identity normalization (center=0, scale=1); the two PLYs will be "
              "identical.")

    if hits:
        rgb = np.tile(np.array(COLOR_INTERSECTION, np.uint8), (loc.shape[0], 1))
        write_colored_ply(
            os.path.join(args.out_dir, 'intersection_points.ply'), loc, rgb)
        if args.unnormalize:
            write_colored_ply(
                os.path.join(args.out_dir, 'intersection_points_original.ply'),
                loc_orig, rgb)
        save_uv_scatter(os.path.join(args.out_dir, 'intersection_uv.png'),
                        hits, tag)
        save_pair_figures(os.path.join(args.out_dir, 'pair_correspondence.png'),
                          groups, tag)

        # One PLY per pair, so a single crossing curve can be inspected on its
        # own instead of picking it out of the combined cloud.
        pair_dir = os.path.join(args.out_dir, 'pairs')
        os.makedirs(pair_dir, exist_ok=True)
        for i, g in enumerate(groups):
            stem = f"pair{i:02d}_p{g['patch_a']}_p{g['patch_b']}"
            xyz_pair = g['xyz'] * scale + center if args.unnormalize else g['xyz']
            write_colored_ply(
                os.path.join(pair_dir, stem + '.ply'), xyz_pair,
                np.tile(np.array(COLOR_INTERSECTION, np.uint8),
                        (g['n_crossings'], 1)))
            # Everything needed to stitch THIS pair, on its own.
            np.savez_compressed(
                os.path.join(pair_dir, stem + '.npz'),
                patch_a=np.int32(g['patch_a']), patch_b=np.int32(g['patch_b']),
                uv_a=g['uv_a'], uv_b=g['uv_b'],
                xyz=g['xyz'], xyz_original=g['xyz'] * scale + center,
                curve_order=curve_order(g['xyz']),
                normal_dot=g['normal_dot'], seam_distance=g['seam_distance'],
                src_is_a=g['src_is_a'])

    # One file per OPENING pair: every correspondence between the two openings,
    # from all contributing patch pairs at once. This is the stitch unit.
    if ogroups:
        from opening_labels import opening_colors
        odir = os.path.join(args.out_dir, 'opening_pairs')
        os.makedirs(odir, exist_ok=True)
        pair_cols = opening_colors(len(ogroups))
        all_xyz, all_rgb = [], []
        for i, g in enumerate(ogroups):
            a, b = g['opening_a'], g['opening_b']
            stem = f"opening{i:02d}_o{a}_o{b}_{g['kind']}"
            xyz_g = g['xyz'] * scale + center if args.unnormalize else g['xyz']
            rgb = np.tile(pair_cols[i], (g['n_crossings'], 1))
            write_colored_ply(os.path.join(odir, stem + '.ply'), xyz_g, rgb)
            all_xyz.append(xyz_g)
            all_rgb.append(rgb)
            np.savez_compressed(
                os.path.join(odir, stem + '.npz'),
                opening_a=np.int32(a), opening_b=np.int32(b), kind=g['kind'],
                # per-row patch on each side: an opening spans several patches
                patch_a=g['patch_a'], patch_b=g['patch_b'],
                uv_a=g['uv_a'], uv_b=g['uv_b'],
                xyz=g['xyz'], xyz_original=g['xyz'] * scale + center,
                curve_order=curve_order(g['xyz']),
                normal_dot=g['normal_dot'], seam_distance=g['seam_distance'],
                src_is_a=g['src_is_a'], patch_pairs=g['patch_pairs'])
        # All crossings in one cloud, coloured by opening pair, so the pairing
        # can be read off in a viewer directly.
        write_colored_ply(os.path.join(args.out_dir, 'crossings_by_opening_pair.ply'),
                          np.concatenate(all_xyz), np.concatenate(all_rgb))
        save_opening_matrix(os.path.join(args.out_dir, 'opening_matrix.png'),
                            ogroups, n_openings)

    # Flat arrays, but ORDERED BY PAIR and carrying an index, so a consumer can
    # either iterate everything or slice one pair out with
    #     s = pair_start[k]; n = pair_count[k]
    #     uv_a[s:s+n], uv_b[s:s+n]
    # without re-deriving the grouping.
    pair_patches = np.array([[g['patch_a'], g['patch_b']] for g in groups],
                            np.int32).reshape(-1, 2)
    pair_count = np.array([g['n_crossings'] for g in groups], np.int64)
    pair_start = np.concatenate([[0], np.cumsum(pair_count)[:-1]]).astype(np.int64) \
        if groups else np.zeros(0, np.int64)

    def cat(key, dtype):
        if not groups:
            return np.zeros((0, 2) if key.startswith('uv') else 0, dtype)
        return np.concatenate([g[key] for g in groups]).astype(dtype)

    xyz_grouped = cat('xyz', np.float64).reshape(-1, 3) if groups \
        else np.zeros((0, 3))
    np.savez_compressed(
        os.path.join(args.out_dir, 'intersections.npz'),
        # --- grouped view: pair k occupies rows pair_start[k] : +pair_count[k]
        pair_patches=pair_patches,
        pair_start=pair_start,
        pair_count=pair_count,
        pair_index=np.repeat(np.arange(len(groups), dtype=np.int32), pair_count)
        if groups else np.zeros(0, np.int32),
        uv_a=cat('uv_a', np.float64).reshape(-1, 2),
        uv_b=cat('uv_b', np.float64).reshape(-1, 2),
        xyz=xyz_grouped,
        xyz_original=xyz_grouped * scale + center,
        src_is_a=cat('src_is_a', bool),
        normal_dot=cat('normal_dot', np.float64),
        seam_distance=cat('seam_distance', np.float64),
        t=cat('t', np.float64),
        # --- normalization, so a consumer never has to re-derive the frame
        normalization_center=center,
        normalization_scale=np.float64(scale),
        # --- raw ungrouped view, kept for continuity with earlier runs
        src_patch=np.array([h['src_patch'] for h in hits], np.int32),
        dst_patch=np.array([h['dst_patch'] for h in hits], np.int32),
        dst_triangle=np.array([h['dst_triangle'] for h in hits], np.int32),
        # opening on each side of every raw hit (-1 = not a hole; all -1 when
        # no openings.npz was supplied)
        opening_src=np.array([h.get('opening_src', -1) for h in hits], np.int32),
        opening_dst=np.array([h.get('opening_dst', -1) for h in hits], np.int32),
        location=loc,
        location_original=loc_orig,
        uv_src=np.array([h['uv_src'] for h in hits], np.float64).reshape(-1, 2),
        uv_dst=np.array([h['uv_dst'] for h in hits], np.float64).reshape(-1, 2),
    )

    summary = {
        'checkpoint': os.path.abspath(args.ckpt),
        'mask_npz': os.path.abspath(args.mask_npz),
        'adaptive': bool(is_adaptive),
        'resolution': R,
        'triangle_rule': 'all_triangles' if args.all_triangles else args.triangle_rule,
        'n_hole_triangles': int(n_black),
        'patches_with_hole': [int(p) for p in involved],
        'eps_t': args.eps_t,
        'seam_eps': seam_eps,
        'median_triangle_edge': med_edge,
        'ray_backend': backend,
        'normalization': {'center': center.tolist(), 'scale': scale},
        'unnormalized_export': bool(args.unnormalize),
        'n_pairs_tested': int(n_pairs),
        'n_crossings': len(hits),
        'n_pairs_crossing': len(groups),
        'uv_index_convention': 'uv = [u, v] in [0,1]^2 of that patch; '
                               'pairs are canonical with patch_a < patch_b',
        'pairs': [pair_summary(g, tag) for g in groups],
        # Opening-level view (empty when no openings.npz was available).
        'openings_npz': os.path.abspath(args.openings_npz) if opening_flat is not None else None,
        'n_openings': int(n_openings),
        'opening_pairs': [opening_group_summary(g, tag) for g in ogroups],
        'opening_pairing': pairing,
    }
    with open(os.path.join(args.out_dir, 'intersections.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Outputs → {args.out_dir}")
    print(f"    intersections.json        — per-pair stats (START HERE)")
    print(f"    intersections.npz         — grouped arrays; slice pair k with")
    print(f"                                s=pair_start[k]; n=pair_count[k];")
    print(f"                                uv_a[s:s+n] <-> uv_b[s:s+n]")
    if hits:
        print(f"    pairs/pairNN_pA_pB.npz    — ONE PAIR per file, ready to stitch")
        print(f"    pairs/pairNN_pA_pB.ply    — that pair's crossing curve alone")
        print(f"    pair_correspondence.png   — patch a UV beside patch b UV, matching")
        print(f"                                colours = matching points")
        print(f"    intersection_points.ply   — all crossings in 3D (normalized frame)")
        if args.unnormalize:
            print(f"    intersection_points_original.ply"
                  f"\n                              — same, in ORIGINAL input coordinates")
        print(f"    intersection_uv.png       — all crossings, per patch")
    if ogroups:
        print(f"    opening_matrix.png        — openings x openings crossing counts")
        print(f"    crossings_by_opening_pair.ply")
        print(f"                              — all crossings, one colour per opening pair")
        print(f"    opening_pairs/openingNN_oA_oB_<kind>.npz")
        print(f"                              — ONE OPENING PAIR per file: every")
        print(f"                                correspondence between opening A and B,")
        print(f"                                with the patch on each side per row")
    print(f"{'─' * 64}\n")


if __name__ == '__main__':
    main()
