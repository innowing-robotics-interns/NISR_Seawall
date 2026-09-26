#!/usr/bin/env python3
# model/cutting_hole.py
"""
Hole cutting on the adaptive cube complex — the first half of turning the
genus-0 cube atlas into a surface with a handle. Stitching is in
model/stitching_hole.py; the driver is utils/cutting_hole.py.

The cube complex is a sphere, so the fitted surface is genus 0. Where the
target has a handle, training pulls two membranes (openings A and B) through
each other along a closed crossing curve C = A ∩ B. C splits each opening into
an inner disk (past the other membrane) and an outer annulus (attached to real
surface). Cutting removes the two inner disks, snapped to leaf boundaries, and
leaves two boundary loops of vertex ids for the stitched tube to share.

detect_crossing() finds A, B and C in-process with the functions of
utils/uv_hole_mask.py, utils/opening_labels.py and utils/patch_intersection.py.
cut_hole() does the cut.

Everything here addresses the surface in FACE coordinates, (face, fu, fv) with
fu, fv in [0, 1], instead of leaf indices: leaf indices are renumbered by every
subdivision, face coordinates are not.

See docs/documentation.md.
"""

from collections import deque
from dataclasses import dataclass, fields

import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass
class HoleConfig:
    """Settings for detect_crossing / cut_hole / stitch_hole. main.py exposes
    each field as --hole_<name>."""
    # detection (same meaning as the uv_hole_mask / opening_labels /
    # patch_intersection command-line options)
    resolution: int = 128          # per-leaf UV grid for mask + meshes
    nn_scale: float = 5.0          # black = farther than nn_scale x median NN spacing
    weld_tol: float = 1e-5
    min_opening_vertices: int = 10
    triangle_rule: str = 'any'
    seam_eps_scale: float = 1.5
    eps_t: float = 1e-4
    backend: str = 'auto'
    openings: str = ''             # force the pair to cut, e.g. '0,3' ('' = auto)
    min_closedness: float = 0.75   # [crossing] warn when C is less loop-like than this
    # cutting
    cut_mode: str = 'opening'      # 'opening': remove both openings whole and stitch
                                   # rim to rim; 'crossing': cut along the crossing
                                   # curve C (needs C to be a closed loop)
    min_disk_leaves: int = 16      # [opening] refine until each opening covers this many leaves
    min_loop_leaves: int = 24      # [crossing] refine until C spans this many leaves
    max_refine_rounds: int = 8
    max_depth: int = 12
    min_leaf_cells: float = 8.0    # [crossing] leaf size floor, in mask-grid cells (> seam-guard gaps)
    opening_min_leaf_cells: float = 2.0  # [opening] leaf size floor, in mask-grid cells
    cut_dilate: int = 0
    allow_non_disk: bool = False
    # stitching
    tube_columns: int = 0          # 0 = min(len(loop A), len(loop B), 64)
    tube_rows: int = 4
    tube_prefit_steps: int = 500   # fit the new tube rows to the ruled surface rim A -> rim B
    tube_prefit_lr: float = 5e-3


def add_hole_args(parser, prefix='hole_'):
    """Expose every HoleConfig field as --<prefix><name>."""
    for f in fields(HoleConfig):
        name = f'--{prefix}{f.name}'
        if f.type is bool or f.type == 'bool':
            parser.add_argument(name, action='store_true', default=f.default)
        else:
            parser.add_argument(name, type=type(f.default), default=f.default)


def hole_config_from_args(args, prefix='hole_'):
    return HoleConfig(**{f.name: getattr(args, prefix + f.name) for f in fields(HoleConfig)})


class NoCrossingFound(RuntimeError):
    """The hole detection found no pair of crossing openings."""


def loop_closedness(xyz, bins=24):
    """
    How much the crossing points look like a closed loop, in [0, 1]: the
    fraction of angular bins around their centroid, in their best-fit plane,
    that hold at least one point. A closed loop around a membrane scores ~1;
    an arc or a line through a fold scores low. Only a closed loop can be cut
    (it must enclose a disk).
    """
    xyz = np.asarray(xyz, np.float64)
    if xyz.shape[0] < 3:
        return 0.0
    c = xyz - xyz.mean(axis=0)
    plane = np.linalg.svd(c, full_matrices=False)[2][:2]
    q = c @ plane.T
    ang = np.arctan2(q[:, 1], q[:, 0])
    occ = np.unique(np.floor((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins)
    return occ.size / bins


# ── face coordinates ────────────────────────────────────────────────────────
def leaf_geometry(cx):
    """Snapshot of the current leaves: (rects (n,3) [u0, v0, size], faces (n,))
    as numpy, in [0, 1] face units."""
    return (cx.leaf_rect.detach().cpu().numpy().astype(np.float64),
            cx.leaf_face.detach().cpu().numpy().astype(np.int64))


def to_face_coords(rects, faces, leaf_idx, uv):
    """Leaf-local (leaf_idx, uv) -> (face, fuv) using a leaf_geometry snapshot."""
    leaf_idx = np.asarray(leaf_idx, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    r = rects[leaf_idx]
    return faces[leaf_idx], r[:, :2] + uv * r[:, 2:3]


def locate(rects, faces, face, fuv, chunk=4096):
    """
    Index of the leaf containing each face point, -1 if none (e.g. the point
    lies in a removed leaf). Leaves on one face tile it, so a point on a shared
    edge may resolve to either neighbour; both touch it, which is all callers
    need.
    """
    face = np.asarray(face, dtype=np.int64)
    fuv = np.clip(np.asarray(fuv, dtype=np.float64).reshape(-1, 2), 0.0, 1.0)
    out = np.full(face.shape[0], -1, np.int64)
    tol = 1e-12
    for f in np.unique(face):
        q_idx = np.flatnonzero(face == f)
        l_idx = np.flatnonzero(faces == f)
        if l_idx.size == 0:
            continue
        r = rects[l_idx]
        for s in range(0, q_idx.size, chunk):
            qi = q_idx[s:s + chunk]
            q = fuv[qi]
            inside = ((q[:, None, 0] >= r[None, :, 0] - tol)
                      & (q[:, None, 0] <= r[None, :, 0] + r[None, :, 2] + tol)
                      & (q[:, None, 1] >= r[None, :, 1] - tol)
                      & (q[:, None, 1] <= r[None, :, 1] + r[None, :, 2] + tol))
            hit = inside.any(axis=1)
            out[qi[hit]] = l_idx[inside[hit].argmax(axis=1)]
    return out


class OpeningLookup:
    """
    Opening ID at any face point, read off the openings.npz grids of the atlas
    the labels were computed on (the ORIGINAL leaves, before any refinement).

    opening_id[i] is (R, R) indexed [u_index, v_index], as written by
    utils/opening_labels.py; -1 means not a hole.
    """

    def __init__(self, rects, faces, patch_ids, opening_id):
        self.rects, self.faces = rects, faces
        self.opening_id = np.asarray(opening_id)
        self.R = int(self.opening_id.shape[-1])
        # Patches absent from the npz get no labels (-1 everywhere).
        self.patch_ids = np.asarray(patch_ids, dtype=np.int64)
        self.row_of = np.full(rects.shape[0], -1, np.int64)
        self.row_of[self.patch_ids] = np.arange(len(patch_ids))

    def opening_points(self, opening):
        """Every detection-grid vertex labelled `opening`, as face coordinates,
        plus the size of the (detection-time) leaf it belongs to."""
        rows, i, j = np.nonzero(self.opening_id == opening)
        leaf = self.patch_ids[rows]
        r = self.rects[leaf]
        ij = np.stack([i, j], axis=1).astype(np.float64) / (self.R - 1)
        return self.faces[leaf], r[:, :2] + ij * r[:, 2:3], r[:, 2]

    def __call__(self, face, fuv):
        leaf = locate(self.rects, self.faces, face, fuv)
        out = np.full(leaf.shape[0], -1, np.int32)
        ok = leaf >= 0
        ok[ok] &= self.row_of[leaf[ok]] >= 0
        if not ok.any():
            return out
        r = self.rects[leaf[ok]]
        uv = (np.asarray(fuv, np.float64).reshape(-1, 2)[ok] - r[:, :2]) / r[:, 2:3]
        ij = np.clip(np.rint(uv * (self.R - 1)).astype(np.int64), 0, self.R - 1)
        out[ok] = self.opening_id[self.row_of[leaf[ok]], ij[:, 0], ij[:, 1]]
        return out


# ── leaf graph ──────────────────────────────────────────────────────────────
def directed_edges(cx, patches=None):
    """
    Map directed sub-edge (a, b) -> owning leaf position in `patches`
    (default: cx.leaf_patches).

    `_polygon` includes hanging vertices, so two neighbours of different depth
    list the same sub-edges. The cube complex is consistently oriented: every
    interior sub-edge (a, b) of one leaf appears as (b, a) in exactly one other.
    """
    patches = cx.leaf_patches if patches is None else patches
    owner = {}
    for i, p in enumerate(patches):
        ids, _ = cx._polygon(p)
        n = len(ids)
        for k in range(n):
            owner[(ids[k], ids[(k + 1) % n])] = i
    return owner


def leaf_adjacency(cx):
    """Edge adjacency between current leaves, as a list of neighbour sets."""
    owner = directed_edges(cx)
    nbr = [set() for _ in cx.leaf_patches]
    for (a, b), i in owner.items():
        j = owner.get((b, a))
        if j is not None and j != i:
            nbr[i].add(j)
    return nbr


def check_closed_orientable(cx):
    """Every directed sub-edge must have exactly one reversed partner. True on
    the untouched cube complex; a failure means the cutting input is already
    inconsistent."""
    owner = directed_edges(cx)
    return [e for e in owner if (e[1], e[0]) not in owner]


def euler_characteristic(cx, patches):
    """V - E + F of a set of leaves (as polygons with hanging vertices)."""
    verts, edges = set(), set()
    for p in patches:
        ids, _ = cx._polygon(p)
        verts.update(ids)
        n = len(ids)
        for k in range(n):
            a, b = ids[k], ids[(k + 1) % n]
            edges.add((min(a, b), max(a, b)))
    return len(verts) - len(edges) + len(patches)


# ── refinement ──────────────────────────────────────────────────────────────
def refine_along_curves(cx, curves, min_loop_leaves, max_rounds, max_depth,
                        min_size, verbose=True):
    """
    Subdivide the leaves the crossing curves pass through until every curve
    touches at least `min_loop_leaves` leaves, so the cut loop can follow C
    closely. Subdivision does not move the surface.

    curves    list of (face, fuv) arrays, one per opening side.
    min_size  per-curve-point lower bound on leaf size (face units). Keeps
              leaves larger than the gaps patch_intersection.py's seam guard
              leaves in the curve, so the cut ring stays closed.
    """
    for rnd in range(max_rounds):
        rects, faces = leaf_geometry(cx)
        counts, to_split = [], set()
        for (face, fuv), lo in zip(curves, min_size):
            leaf = locate(rects, faces, face, fuv)
            ok = leaf >= 0
            touched = np.unique(leaf[ok])
            counts.append(int(touched.size))
            if touched.size >= min_loop_leaves:
                continue
            # Refine a leaf only while it is above the size floor of every
            # curve point it holds.
            floor = np.zeros(rects.shape[0])
            np.maximum.at(floor, leaf[ok], np.asarray(lo)[ok])
            for i in touched:
                p = cx.leaf_patches[i]
                if (p.depth < max_depth and p.size >= 2 and not p.frozen
                        and rects[i, 2] * 0.5 >= floor[i]):
                    to_split.add(i)
        if verbose:
            print(f"    refine round {rnd}: leaves on each curve = {counts}  "
                  f"splitting {len(to_split)}")
        if not to_split:
            return counts
        cx.subdivide_patches([cx.leaf_patches[i] for i in sorted(to_split)])
    rects, faces = leaf_geometry(cx)
    return [curve_leaf_count(rects, faces, f, q) for f, q in curves]


def curve_leaf_count(rects, faces, face, fuv):
    leaf = locate(rects, faces, face, fuv)
    return int(np.unique(leaf[leaf >= 0]).size)


# ── choosing what to remove ─────────────────────────────────────────────────
def leaf_sample_points(rects, faces, n=3):
    """n x n interior sample points per leaf, in face coords. Returns
    (leaf_of_sample, face, fuv)."""
    t = (np.arange(n) + 0.5) / n
    g = np.stack(np.meshgrid(t, t, indexing='ij'), -1).reshape(-1, 2)   # (n^2, 2)
    L = rects.shape[0]
    leaf = np.repeat(np.arange(L), g.shape[0])
    fuv = rects[leaf, :2] + np.tile(g, (L, 1)) * rects[leaf, 2:3]
    return leaf, faces[leaf], fuv


def select_inner_disk(cx, crossing_face, crossing_fuv, opening, lookup,
                      dilate=0):
    """
    Leaves to remove for ONE opening: the cut ring (leaves the crossing curve
    passes through) plus everything the ring encloses on the membrane side.

    The enclosed side is found by flood fill over edge-adjacent leaves, seeded
    from every leaf with no sample labelled `opening` (plain surface or another
    opening) and never entering the ring. What the fill cannot reach is the
    inner disk. Opening labels only choose the seeds, so a slightly loose or
    tight hole mask does not move the cut.

    Returns (remove_idx, info). Raises RuntimeError when the ring does not
    close (the fill reaches the whole opening).
    """
    rects, faces = leaf_geometry(cx)
    L = rects.shape[0]
    nbr = leaf_adjacency(cx)

    ring_leaf = locate(rects, faces, crossing_face, crossing_fuv)
    ring = set(int(i) for i in ring_leaf[ring_leaf >= 0])
    for _ in range(max(0, dilate)):
        ring |= {j for i in ring for j in nbr[i]}

    s_leaf, s_face, s_fuv = leaf_sample_points(rects, faces)
    lab = lookup(s_face, s_fuv)
    in_x = np.zeros(L, bool)
    np.logical_or.at(in_x, s_leaf, lab == opening)

    seeds = [i for i in range(L) if not in_x[i] and i not in ring]
    reached = np.zeros(L, bool)
    q = deque(seeds)
    reached[seeds] = True
    while q:
        i = q.popleft()
        for j in nbr[i]:
            if not reached[j] and j not in ring:
                reached[j] = True
                q.append(j)

    disk = [i for i in range(L) if not reached[i] and i not in ring]
    n_x = int(in_x.sum())
    if not disk:
        raise RuntimeError(
            f"opening {opening}: the crossing ring does not enclose any leaf "
            f"({len(ring)} ring leaves, {n_x} leaves in the opening). The ring "
            f"is open (seam-guard gaps, sparse crossings) or too coarse. Try "
            f"hole_cut_dilate=1, fewer refine rounds, or a lower "
            f"hole_seam_eps_scale.")
    remove = sorted(ring | set(disk))
    info = {
        'opening': int(opening),
        'n_ring_leaves': len(ring),
        'n_disk_leaves': len(disk),
        'n_removed': len(remove),
        'n_leaves_in_opening': n_x,
        # Removed leaves with no sample in this opening: real surface being
        # cut. A large fraction means the cut is in the wrong place.
        'n_removed_outside_opening': int((~in_x[remove]).sum()),
    }
    return remove, info


def opening_fraction(cx, opening, lookup):
    """
    Per current leaf: fraction of its area covered by `opening`.

    Built from the opening's own detection-grid vertices: each one is located
    in the current leaf that contains it and stands for one grid cell of area.
    Unlike sampling each leaf, this cannot miss an opening that is small
    compared with the leaf it sits in.
    """
    rects, faces = leaf_geometry(cx)
    face, fuv, osize = lookup.opening_points(opening)
    leaf = locate(rects, faces, face, fuv)
    ok = leaf >= 0
    cell = (osize[ok] / (lookup.R - 1)) ** 2
    covered = np.bincount(leaf[ok], weights=cell, minlength=rects.shape[0])
    return np.minimum(covered / rects[:, 2] ** 2, 1.0)


def refine_openings(cx, openings, lookup, min_disk_leaves, max_rounds, max_depth,
                    min_leaf_cells, verbose=True):
    """
    [opening mode] Subdivide the leaves an opening touches until at least
    `min_disk_leaves` leaves lie mostly inside it, so the removed region (and
    the rim loop) follows the opening's shape. A leaf is never refined below
    `min_leaf_cells` cells of the detection grid of the leaf it came from, the
    resolution the opening labels have.
    """
    counts = []
    for rnd in range(max_rounds):
        rects, faces = leaf_geometry(cx)
        centers = rects[:, :2] + 0.5 * rects[:, 2:3]
        orig = locate(lookup.rects, lookup.faces, faces, centers)
        floor = lookup.rects[orig, 2] * min_leaf_cells / (lookup.R - 1)
        counts, to_split = [], set()
        for o in openings:
            frac = opening_fraction(cx, o, lookup)
            counts.append(int((frac >= 0.5).sum()))
            if counts[-1] >= min_disk_leaves:
                continue
            for i in np.flatnonzero(frac > 0):
                p = cx.leaf_patches[i]
                if (p.depth < max_depth and p.size >= 2 and not p.frozen
                        and rects[i, 2] * 0.5 >= floor[i]):
                    to_split.add(int(i))
        if verbose:
            print(f"    refine round {rnd}: leaves inside each opening = {counts}  "
                  f"splitting {len(to_split)}")
        if not to_split:
            return counts
        cx.subdivide_patches([cx.leaf_patches[i] for i in sorted(to_split)])
    return [int((opening_fraction(cx, o, lookup) >= 0.5).sum()) for o in openings]


def _components(nodes, nbr):
    """Connected components of a node subset under the adjacency lists."""
    nodes, seen, comps = set(nodes), set(), []
    for s in nodes:
        if s in seen:
            continue
        comp, q = [], deque([s])
        seen.add(s)
        while q:
            i = q.popleft()
            comp.append(i)
            for j in nbr[i]:
                if j in nodes and j not in seen:
                    seen.add(j)
                    q.append(j)
        comps.append(comp)
    return sorted(comps, key=len, reverse=True)


def select_opening_disk(cx, opening, lookup):
    """
    [opening mode] Leaves to remove for ONE opening: the leaves lying mostly
    inside it (largest connected piece), with any enclosed islands of other
    leaves filled in so the removed region has a single boundary loop.
    """
    L = cx.n_leaves
    nbr = leaf_adjacency(cx)
    frac = opening_fraction(cx, opening, lookup)
    core = np.flatnonzero(frac >= 0.5)
    if core.size == 0:
        raise RuntimeError(
            f"opening {opening}: no leaf lies mostly inside it (the opening is "
            f"only a few mask cells wide). Try a higher hole_resolution, a lower "
            f"hole_opening_min_leaf_cells, or more hole_max_refine_rounds.")
    comps = _components(core.tolist(), nbr)
    disk = set(comps[0])
    rest = _components([i for i in range(L) if i not in disk], nbr)
    filled = [i for comp in rest[1:] for i in comp]      # islands inside the disk
    disk |= set(filled)
    remove = sorted(disk)
    return remove, {
        'opening': int(opening),
        'n_removed': len(remove),
        'n_leaves_in_opening': int(core.size),
        'n_pieces_dropped': len(comps) - 1,
        'n_islands_filled': len(filled),
        'n_removed_outside_opening': int((frac[remove] < 0.5).sum()),
        'mean_opening_fraction': float(frac[remove].mean()),
    }


# ── loops ───────────────────────────────────────────────────────────────────
def boundary_loops(cx):
    """
    Closed boundary loops of the (cut) surface.

    A boundary sub-edge is a directed edge of a kept leaf whose reverse no kept
    leaf owns. Each loop is returned in the kept leaves' own direction (CCW in
    their UV), which is the orientation the stitching tube must reverse.

    Returns a list of dicts: 'vids' (ordered vertex ids, not repeated at the
    end), 'leaf' and 'uv' (a kept leaf and local uv at which each vertex can be
    decoded), 'edge_leaf' (kept leaf owning each edge vids[k] -> vids[k+1]).
    Raises on a non-manifold boundary (a vertex with two outgoing boundary
    edges, i.e. the removed region pinches).
    """
    owner, uv_at = {}, {}
    for i, p in enumerate(cx.leaf_patches):
        ids, uvs = cx._polygon(p)
        n = len(ids)
        for k in range(n):
            owner[(ids[k], ids[(k + 1) % n])] = i
            uv_at[(i, ids[k])] = uvs[k]
    nxt = {}
    for (a, b), i in owner.items():
        if (b, a) in owner:
            continue
        if a in nxt:
            raise RuntimeError(f"non-manifold cut: vertex {a} has two outgoing "
                               f"boundary edges (removed region pinches there)")
        nxt[a] = (b, i)

    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        vids, leaf, uv, eleaf = [], [], [], []
        v = start
        while v not in seen:
            seen.add(v)
            b, i = nxt[v]
            vids.append(v)
            leaf.append(i)
            uv.append(uv_at[(i, v)])
            eleaf.append(i)
            v = b
        if v != start:
            raise RuntimeError("boundary walk did not close; cut is inconsistent")
        loops.append({'vids': vids, 'leaf': leaf, 'uv': uv, 'edge_leaf': eleaf})
    return loops


@torch.no_grad()
def decode_loop(model, loop, samples_per_edge=1):
    """3D points along a loop. samples_per_edge=1 gives the vertices; more
    samples follow each sub-edge through its owning leaf (the edge is linear in
    feature space, not in 3D)."""
    cx = model.complex
    device = next(model.parameters()).device
    n = len(loop['vids'])
    pids, uvs = [], []
    for k in range(n):
        i = loop['edge_leaf'][k]
        ids, puv = cx._polygon(cx.leaf_patches[i])
        a = np.asarray(puv[ids.index(loop['vids'][k])])
        b = np.asarray(puv[ids.index(loop['vids'][(k + 1) % n])])
        for t in np.arange(samples_per_edge) / samples_per_edge:
            pids.append(i)
            uvs.append((1 - t) * a + t * b)
    pids = torch.tensor(pids, dtype=torch.long, device=device)
    uvs = torch.tensor(np.asarray(uvs), dtype=torch.float32, device=device)
    return model(pids, uvs).cpu().numpy().astype(np.float64)


def loops_touching(cx, loops, removed_patches):
    """Index of the loop bordering each removed set (None if none / several)."""
    out = []
    for patches in removed_patches:
        edges = directed_edges(cx, patches)
        hits = [k for k, lp in enumerate(loops)
                if any((lp['vids'][(j + 1) % len(lp['vids'])], lp['vids'][j]) in edges
                       for j in range(len(lp['vids'])))]
        out.append(hits)
    return out


# ── detection (in-process uv_hole_mask -> opening_labels -> patch_intersection) ──
def _detection_modules():
    """The detection scripts, imported as utils.* from main.py or bare when
    running from utils/ as a script."""
    try:
        from utils import opening_labels, patch_intersection, uv_hole_mask
    except ImportError:
        import opening_labels
        import patch_intersection
        import uv_hole_mask
    return uv_hole_mask, opening_labels, patch_intersection


def detect_crossing(model, pts, cfg: HoleConfig, device, verbose=True):
    """
    Run the hole-detection pipeline on the current model against the target
    cloud `pts` (normalized frame) and return the crossing to cut.

    1. uv_hole_mask:       sample every leaf on an R x R grid; a vertex is
                           black (hole) when its nearest target point is
                           farther than nn_scale x the cloud's median spacing.
    2. opening_labels:     weld patch boundaries, connected components of the
                           black vertices -> opening IDs.
    3. patch_intersection: cast the edges of black triangles against every
                           other leaf, attach opening IDs, group by opening pair.

    Assumes one handle. Crossings also appear where the surface folds through
    itself away from the hole, and those folds make openings too. The hole's
    two membranes are the largest no-data regions, so the pair whose SMALLER
    opening has the largest 3D area is taken — or cfg.openings when set.
    Raises NoCrossingFound when there is no pair.

    Returns a dict with the two opening IDs, the crossing curve on each side in
    face coordinates, the 3D crossing points, per-point leaf-size floors for
    refinement, and an OpeningLookup over the detection-time atlas.
    """
    um, ol, pi = _detection_modules()
    cx = model.complex
    if cx.tube_patches:
        raise RuntimeError("detect_crossing() on a model that already has a "
                           "stitched tube is not supported (one handle only)")
    was_training = model.training
    model.eval()
    R = int(cfg.resolution)
    patch_ids = list(range(cx.n_quad_leaves))

    meshes = pi.build_patch_meshes(model, patch_ids, R, device)
    xyz_fields = {p: meshes[p]['xyz'].astype(np.float32) for p in patch_ids}

    pts = np.asarray(pts, dtype=np.float32)
    tree = cKDTree(pts)
    d_nn = um.median_nn_spacing(pts, tree)
    tau = cfg.nn_scale * d_nn
    masks = {}
    for p in patch_ids:
        d, _ = tree.query(xyz_fields[p], k=1, workers=-1)
        masks[p] = (d <= tau).reshape(R, R)          # True = has correspondence
    n_black = int(sum((~m).sum() for m in masks.values()))

    weld_pairs, _ = ol.weld_boundary_vertices(xyz_fields, patch_ids, R, cfg.weld_tol)
    opening_id, n_openings, _ = ol.label_openings(
        masks, xyz_fields, patch_ids, R, weld_pairs, cfg.min_opening_vertices)
    if verbose:
        print(f"    mask: tau={tau:.6f} ({cfg.nn_scale} x median NN {d_nn:.6f}), "
              f"{n_black:,} black vertices, {n_openings} openings")
    if n_openings < 2:
        model.train(was_training)
        raise NoCrossingFound(f"only {n_openings} opening(s) found; a handle needs 2")

    faces = meshes[patch_ids[0]]['faces']
    black = {p: pi.black_triangle_ids(masks[p].ravel(), faces, cfg.triangle_rule)
             for p in patch_ids}
    med_edge = float(np.median([meshes[p]['edge_len'] for p in patch_ids]))
    seam_eps = cfg.seam_eps_scale * med_edge
    seam_tree = pi.build_seam_tree(meshes, patch_ids, R)
    backend, trimesh_mod = pi.make_ray_backend(cfg.backend)
    hits, _ = pi.detect_intersections(meshes, black, patch_ids, seam_tree, seam_eps,
                                      cfg.eps_t, backend, trimesh_mod,
                                      include_self=False, verbose=False)
    opening_flat = {p: opening_id[i].ravel() for i, p in enumerate(patch_ids)}
    if hits:
        pi.assign_openings(hits, opening_flat, faces)
    groups = [g for g in pi.group_by_opening(hits) if g['kind'] == 'pair'] if hits else []
    model.train(was_training)
    # 3D area of each opening: triangles whose three vertices all carry its ID.
    area = np.zeros(n_openings)
    for i, p in enumerate(patch_ids):
        lab = opening_id[i].ravel()[faces]
        full = (lab[:, 0] >= 0) & (lab[:, 0] == lab[:, 1]) & (lab[:, 0] == lab[:, 2])
        if full.any():
            v = meshes[p]['xyz'][faces[full]]
            a = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
            np.add.at(area, lab[full, 0], a)
    for g in groups:
        g['closedness'] = loop_closedness(g['xyz'])
        g['area'] = float(min(area[g['opening_a']], area[g['opening_b']]))

    # Every opening, for checking the pair choice by eye (openings.ply).
    o_xyz, o_id = [], []
    for i, p in enumerate(patch_ids):
        lab = opening_id[i].ravel()
        keep = lab >= 0
        o_xyz.append(xyz_fields[p][keep])
        o_id.append(lab[keep])
    o_xyz, o_id = np.concatenate(o_xyz), np.concatenate(o_id)
    partners = {o: [] for o in range(n_openings)}
    for g in groups:
        partners[g['opening_a']].append(g['opening_b'])
        partners[g['opening_b']].append(g['opening_a'])
    openings = [{'id': o, 'area': float(area[o]), 'n_vertices': int((o_id == o).sum()),
                 'centroid': o_xyz[o_id == o].mean(axis=0).round(4).tolist(),
                 'crosses': partners[o]} for o in range(n_openings)]
    if verbose:
        print("    openings (id: 3D area, centroid, crosses):")
        for o in openings:
            print(f"      {o['id']:3d}: area {o['area']:.4f}  centroid {o['centroid']}  "
                  f"crosses {o['crosses'] or '-'}")
    if verbose:
        print(f"    crossings: {len(hits):,} hits, {len(groups)} opening pair(s)"
              + (": " + ", ".join(f"({g['opening_a']},{g['opening_b']}) x{g['n_crossings']} "
                                  f"area {g['area']:.3f} loop {g['closedness']:.2f}"
                                  for g in groups[:8])
                 if groups else ""))
    if not groups:
        raise NoCrossingFound("no two openings cross each other")

    if cfg.openings:
        want = tuple(sorted(int(x) for x in cfg.openings.split(',')))
        chosen = [g for g in groups if (g['opening_a'], g['opening_b']) == want]
        if not chosen:
            raise NoCrossingFound(f"openings {want} do not cross")
        g = chosen[0]
    else:
        g = max(groups, key=lambda x: (x['area'], x['n_crossings']))
    if cfg.cut_mode == 'crossing' and g['closedness'] < cfg.min_closedness:
        print(f"    [warn] the chosen crossing curve is not loop-like "
              f"(closedness {g['closedness']:.2f}); the cut will likely fail")
    rects, lfaces = leaf_geometry(cx)
    face_a, fuv_a = to_face_coords(rects, lfaces, g['patch_a'], g['uv_a'])
    face_b, fuv_b = to_face_coords(rects, lfaces, g['patch_b'], g['uv_b'])
    return {
        'opening_a': int(g['opening_a']), 'opening_b': int(g['opening_b']),
        'face_a': face_a, 'fuv_a': fuv_a, 'face_b': face_b, 'fuv_b': fuv_b,
        'xyz': np.asarray(g['xyz'], np.float64),
        # Refinement floor per crossing point: min_leaf_cells grid cells of the
        # leaf it was detected on.
        'min_size_a': rects[g['patch_a'], 2] * cfg.min_leaf_cells / (R - 1),
        'min_size_b': rects[g['patch_b'], 2] * cfg.min_leaf_cells / (R - 1),
        'lookup': OpeningLookup(rects, lfaces, patch_ids, opening_id),
        'opening_xyz': o_xyz, 'opening_ids': o_id,
        'stats': {'tau': float(tau), 'median_nn': float(d_nn),
                  'n_black_vertices': n_black, 'n_openings': int(n_openings),
                  'n_hits': len(hits), 'n_pairs': len(groups),
                  'pairs': [[int(x['opening_a']), int(x['opening_b']),
                             int(x['n_crossings']), float(x['area']),
                             float(x['closedness'])] for x in groups],
                  'opening_area': area.tolist(),
                  'openings': openings,
                  'chosen': [int(g['opening_a']), int(g['opening_b'])],
                  'closedness': float(g['closedness'])},
    }


# ── cutting ─────────────────────────────────────────────────────────────────
def cut_hole(model, crossing, cfg: HoleConfig, verbose=True):
    """
    Cut openings A and B out of the model, in place.

    cfg.cut_mode = 'opening' (default): refine the leaves each opening
    touches, remove the leaves lying mostly inside it. The loops are the
    openings' rims. Works whether or not the crossing curve C closes.
    cfg.cut_mode = 'crossing': refine along C, remove the ring of leaves C
    passes through plus what it encloses. Needs C to be a closed loop.

    Then: validate (disjoint, disks), remove, extract the two boundary loops,
    freeze the leaves on the loops.

    Returns the cut record: the two loops as ordered vertex ids (in the kept
    leaves' direction) with their 3D positions, the crossing curve, the removed
    rect keys, diagnostics and the Euler characteristic before/after.
    """
    cx = model.complex
    bad = check_closed_orientable(cx)
    if bad:
        raise RuntimeError(f"surface already has {len(bad)} boundary edges; "
                           f"cut_hole() expects the closed cube complex")
    chi0 = euler_characteristic(cx, cx.leaf_patches)
    lookup = crossing['lookup']
    sides = [
        {'opening': crossing['opening_a'], 'face': crossing['face_a'],
         'fuv': crossing['fuv_a'], 'min_size': crossing['min_size_a']},
        {'opening': crossing['opening_b'], 'face': crossing['face_b'],
         'fuv': crossing['fuv_b'], 'min_size': crossing['min_size_b']},
    ]

    if cfg.cut_mode == 'opening':
        if verbose:
            print("    refining the openings")
        counts = refine_openings(cx, [s['opening'] for s in sides], lookup,
                                 cfg.min_disk_leaves, cfg.max_refine_rounds,
                                 cfg.max_depth, cfg.opening_min_leaf_cells,
                                 verbose=verbose)
    elif cfg.cut_mode == 'crossing':
        if verbose:
            print("    refining along the crossing curve")
        counts = refine_along_curves(cx, [(s['face'], s['fuv']) for s in sides],
                                     cfg.min_loop_leaves, cfg.max_refine_rounds,
                                     cfg.max_depth, [s['min_size'] for s in sides],
                                     verbose=verbose)
    else:
        raise ValueError(f"unknown cut_mode {cfg.cut_mode!r}")

    removed_idx, infos = [], []
    for s in sides:
        if cfg.cut_mode == 'opening':
            idx, info = select_opening_disk(cx, s['opening'], lookup)
        else:
            idx, info = select_inner_disk(cx, s['face'], s['fuv'], s['opening'],
                                          lookup, dilate=cfg.cut_dilate)
        removed_idx.append(idx)
        infos.append(info)
        if verbose:
            print(f"    opening {s['opening']}: {info['n_removed']} leaves to remove "
                  f"({info['n_removed_outside_opening']} mostly outside the opening)")

    if set(removed_idx[0]) & set(removed_idx[1]):
        raise RuntimeError("the two removed regions overlap")
    vsets = [set(v for i in idx for v in cx._polygon(cx.leaf_patches[i])[0])
             for idx in removed_idx]
    if vsets[0] & vsets[1]:
        raise RuntimeError("the two removed regions share vertices; their loops "
                           "would touch")
    removed = [[cx.leaf_patches[i] for i in idx] for idx in removed_idx]
    for s, info, patches in zip(sides, infos, removed):
        info['euler_characteristic'] = euler_characteristic(cx, patches)
        if info['euler_characteristic'] != 1:
            msg = (f"opening {s['opening']}: removed region has Euler "
                   f"characteristic {info['euler_characteristic']} (a disk has 1)")
            if not cfg.allow_non_disk:
                raise RuntimeError(msg)
            print(f"    [warn] {msg}")

    cx.remove_patches(removed[0] + removed[1])
    loops = boundary_loops(cx)
    touch = loops_touching(cx, loops, removed)
    for s, t in zip(sides, touch):
        if len(t) != 1:
            raise RuntimeError(f"opening {s['opening']}: removed region is bounded "
                               f"by {len(t)} loops, expected 1")
    if len(loops) != 2:
        raise RuntimeError(f"{len(loops)} boundary loops after the cut, expected 2")
    loop_a, loop_b = loops[touch[0][0]], loops[touch[1][0]]

    frozen = sorted({i for lp in (loop_a, loop_b) for i in lp['edge_leaf']})
    cx.freeze_patches([cx.leaf_patches[i] for i in frozen])
    chi1 = euler_characteristic(cx, cx.leaf_patches)
    if verbose:
        print(f"    cut: {len(removed[0]) + len(removed[1])} leaves removed, loops of "
              f"{len(loop_a['vids'])} / {len(loop_b['vids'])} vertices, "
              f"{len(frozen)} loop leaves frozen, Euler characteristic {chi0} -> {chi1}")

    return {
        'cut_mode': cfg.cut_mode,
        'openings': [crossing['opening_a'], crossing['opening_b']],
        'loop_a': [int(v) for v in loop_a['vids']],
        'loop_b': [int(v) for v in loop_b['vids']],
        'loop_a_xyz': decode_loop(model, loop_a).tolist(),
        'loop_b_xyz': decode_loop(model, loop_b).tolist(),
        'crossing_a': {'face': np.asarray(crossing['face_a']).tolist(),
                       'fuv': np.asarray(crossing['fuv_a']).tolist()},
        'crossing_b': {'face': np.asarray(crossing['face_b']).tolist(),
                       'fuv': np.asarray(crossing['fuv_b']).tolist()},
        'crossing_xyz': np.asarray(crossing['xyz']).tolist(),
        'removed_a': [list(p.rect_key()) for p in removed[0]],
        'removed_b': [list(p.rect_key()) for p in removed[1]],
        'sides': infos,
        'leaves_after_refine': counts,
        'n_frozen': len(frozen),
        'euler_characteristic': {'before': chi0, 'after_cut': chi1},
    }



# ── subdivision after the hole ──────────────────────────────────────────────
def local_subdiv_predicate(cx, rings=2):
    """
    `allow(patch)` for subdivide_by_distortion that only lets quadtree leaves
    near the cut split: the leaves within `rings` edge-rings of the frozen
    loop leaves (the loop leaves themselves and tube cells never split). The
    region is fixed now, as rects in face coordinates, so it stays valid while
    leaves are renumbered; a later leaf is allowed when it lies inside one of
    these rects (i.e. descends from a leaf of the region).
    """
    nbr = leaf_adjacency(cx)
    n_quad = cx.n_quad_leaves
    leaves = cx.leaf_patches
    region = {i for i in range(n_quad) if leaves[i].frozen}
    frontier = set(region)
    for _ in range(max(0, rings)):
        frontier = {j for i in frontier for j in nbr[i] if j < n_quad} - region
        region |= frontier
    rects = {}
    for i in region:
        p = leaves[i]
        if not p.frozen:
            rects.setdefault(p.face, []).append((p.u0, p.v0, p.size))
    rects = {f: np.asarray(r, np.int64) for f, r in rects.items()}
    n_allowed = sum(len(r) for r in rects.values())

    def allow(p):
        r = rects.get(p.face)
        if r is None:
            return False
        return bool(np.any((p.u0 >= r[:, 0]) & (p.u0 + p.size <= r[:, 0] + r[:, 2])
                           & (p.v0 >= r[:, 1]) & (p.v0 + p.size <= r[:, 1] + r[:, 2])))
    return allow, n_allowed
