#!/usr/bin/env python3
# model/subdivision.py
"""Distortion measurement, distortion-driven subdivision, seam diagnostics."""

import torch

from . import losses

def compute_patch_distortion(model, samples_per_patch: int = 128,
                             mode: str = 'area', chunk_leaves: int = 128):
    """
    Per-leaf distortion score, shape (n_leaves,) on CPU.

    Modes (from the first fundamental form E, F, G of F(uv)):
      'area'      mean sqrt(EG - F^2): 3D area covered by the leaf's unit UV
                  square. Big patches covering lots of geometry score high.
      'dirichlet' mean 0.5 (E + G): stretch energy.
      'conformal' mean ((E-G)^2 + 4F^2) / (E+G)^2: scale-invariant
                  anisotropy in [0,1].
    """
    device = next(model.parameters()).device
    n = model.n_patches
    out = torch.zeros(n)
    for start in range(0, n, chunk_leaves):
        stop = min(start + chunk_leaves, n)
        idx = torch.arange(start, stop, device=device)
        k = idx.numel()
        pids = idx.repeat_interleave(samples_per_patch)
        with torch.enable_grad():
            uv = torch.rand(k * samples_per_patch, 2, device=device,
                            requires_grad=True)
            Q = model(pids, uv)
            t_u, t_v = losses.surface_jacobian(Q, uv)
        E = (t_u * t_u).sum(-1)
        G = (t_v * t_v).sum(-1)
        Fd = (t_u * t_v).sum(-1)

        tr_sigma  = E + G                                          # s1^2 + s2^2
        det_sigma = torch.linalg.cross(t_u, t_v, dim=-1).pow(2).sum(-1)
        if mode == 'area':
            e = torch.sqrt(torch.clamp(E * G - Fd ** 2, min=0.0))
        elif mode == 'dirichlet':
            e = 0.5 * (E + G)
        elif mode == 'symmetric_dirichlet':
            # -log(det(J^T J)) + tr(J^T J), per singular value
            e = tr_sigma - torch.log(det_sigma.clamp_min(1e-12))
        elif mode == 'conformal':
            e = ((E - G) ** 2 + 4.0 * Fd ** 2) / ((E + G) ** 2 + 1e-12)
        else:
            raise ValueError(f"unknown distortion mode: {mode}")
        out[start:stop] = e.detach().reshape(k, samples_per_patch).mean(dim=1).cpu()
    return out


def subdivide_by_distortion(model, threshold: float, max_depth: int,
                            samples_per_patch: int = 128, mode: str = 'area',
                            max_splits_per_round: int = 0, distortion=None):
    """
    Subdivide every leaf whose distortion exceeds `threshold`, unless it is
    already at `max_depth` (absolute quadtree depth; base_subdivisions count
    toward it). Returns a summary dict. The caller MUST rebuild its optimizer
    if n_subdivided > 0 (vertex_features is a new Parameter afterwards).
    """
    if distortion is None:
        distortion = compute_patch_distortion(model, samples_per_patch, mode)
    leaves = list(model.complex.leaf_patches)
    cand = [(float(distortion[i]), i, p) for i, p in enumerate(leaves)
            if float(distortion[i]) > threshold and p.depth < max_depth and p.size >= 2]
    cand.sort(key=lambda t: -t[0])
    if max_splits_per_round and max_splits_per_round > 0:
        cand = cand[:max_splits_per_round]
    if cand:
        model.complex.subdivide_patches([p for _, _, p in cand])
    return {
        'n_subdivided': len(cand),
        'n_leaves': model.n_patches,
        'n_vertices': model.complex.n_vertices,
        'distortion': distortion,
        'max_distortion': float(distortion.max()) if len(leaves) else 0.0,
        'mean_distortion': float(distortion.mean()) if len(leaves) else 0.0,
        'subdivided_depths': sorted(p.depth for _, _, p in cand),
    }


@torch.no_grad()
def compute_patch_cd(model, target, grid_uv, ext_mask=None,
                     chunk_size: int = 2048, target_chunk_size: int = 4096):
    """Per-leaf Chamfer error on a fixed UV grid, shape (n_leaves,).

    Unlike :func:`compute_patch_distortion`, this measures fitting error, not
    geometric stretch.  Every leaf is evaluated on the same ``grid_uv`` (an
    ``(R*R, 2)`` grid such as :func:`utils.render_mesh.uv_grid`), so the score
    changes only when the surface or the leaf layout does.

    Two terms are returned, both in the units of the global Chamfer monitor:

    ``fwd``  surface -> cloud: mean distance from the leaf's grid points to
             the nearest target point.  High where the sheet misses the cloud.
    ``bwd``  cloud -> surface: each target point votes for the leaf owning its
             nearest grid point; the leaf's term is the mean distance of its
             votes.  High where the leaf is too coarse to reach target
             detail, i.e. where the leaf is under-resolved.  This is the term
             that actually grows with leaf size; ``fwd`` alone is nearly
             insensitive to under-resolution.

    ``score = fwd + bwd`` matches the symmetric Chamfer used in training.
    Leaves with no grid samples inside a detected extension get ``fwd`` from
    their remaining samples only; ``valid_frac`` reports the usable fraction
    so the caller can skip leaves that are mostly opening.

    Args:
        model: adaptive forward map.
        target: ``(N, 3)`` cloud in the model's normalized frame, on the
            model's device.
        grid_uv: ``(R*R, 2)`` grid on the same device.
        ext_mask: optional :class:`utils.extension_mask.ExtensionMask`; its
            samples are excluded from ``fwd``.

    Returns:
        dict with ``score``, ``fwd``, ``bwd``, ``count`` (target votes per
        leaf) and ``valid_frac``, all on the model's device.
    """
    device = next(model.parameters()).device
    if target.device != device:
        target = target.to(device)
    n_leaves = model.n_patches
    grid_size = int(grid_uv.shape[0])

    pids = torch.arange(n_leaves, device=device).repeat_interleave(grid_size)
    uv = grid_uv.repeat(n_leaves, 1)
    Q = model(pids, uv)

    nearest = []
    for start in range(0, Q.shape[0], chunk_size):
        d = torch.cdist(Q[start:start + chunk_size], target)
        nearest.append(d.min(dim=1).values)
    nearest = torch.cat(nearest).reshape(n_leaves, grid_size)

    valid = torch.ones((n_leaves, grid_size), dtype=torch.bool, device=device)
    if ext_mask is not None:
        valid = ~ext_mask.excluded(model, pids, uv).reshape(n_leaves, grid_size)
    n_valid = valid.sum(dim=1)
    fwd = (nearest * valid).sum(dim=1) / n_valid.clamp_min(1).to(nearest.dtype)
    fwd = torch.where(n_valid > 0, fwd, torch.zeros_like(fwd))

    bwd_sum = torch.zeros(n_leaves, device=device, dtype=Q.dtype)
    bwd_cnt = torch.zeros(n_leaves, device=device, dtype=Q.dtype)
    for start in range(0, target.shape[0], target_chunk_size):
        d = torch.cdist(target[start:start + target_chunk_size], Q)
        vals, idx = d.min(dim=1)
        leaf = torch.div(idx, grid_size, rounding_mode='floor')
        bwd_sum.index_add_(0, leaf, vals)
        bwd_cnt.index_add_(0, leaf, torch.ones_like(vals))
    bwd = bwd_sum / bwd_cnt.clamp_min(1.0)

    return {
        'score': fwd + bwd,
        'fwd': fwd,
        'bwd': bwd,
        'count': bwd_cnt,
        'valid_frac': n_valid.to(fwd.dtype) / float(grid_size),
    }


def subdivide_by_cd(model, target, grid_uv, threshold: float, max_depth: int,
                    max_splits_per_round: int = 0, ext_mask=None,
                    min_valid_frac: float = 0.25, **cd_kwargs):
    """Subdivide leaves whose grid Chamfer error exceeds ``threshold``.

    Same contract as :func:`subdivide_by_distortion`: the caller must rebuild
    its optimizer when ``n_subdivided > 0``.  Leaves deeper than ``max_depth``,
    too small to split, or mostly inside a detected extension
    (``valid_frac < min_valid_frac``) are never selected.  Selection is by
    descending ``score`` and optionally capped per round.

    Returns a summary dict; the per-leaf ``score``/``fwd``/``bwd`` vectors and
    their max/mean/p90 are included for logging.
    """
    stats = compute_patch_cd(model, target, grid_uv, ext_mask=ext_mask,
                             **cd_kwargs)
    score = stats['score']
    leaves = list(model.complex.leaf_patches)
    cand = []
    for i, p in enumerate(leaves):
        if p.depth >= max_depth or p.size < 2:
            continue
        if float(stats['valid_frac'][i]) < min_valid_frac:
            continue
        if float(score[i]) > threshold:
            cand.append((float(score[i]), i, p))
    cand.sort(key=lambda t: -t[0])
    if max_splits_per_round and max_splits_per_round > 0:
        cand = cand[:max_splits_per_round]
    if cand:
        model.complex.subdivide_patches([p for _, _, p in cand])

    n = len(leaves)
    quantile = (float(torch.quantile(score, 0.9)) if n else 0.0)
    return {
        'n_subdivided': len(cand),
        'n_leaves': model.n_patches,
        'n_vertices': model.complex.n_vertices,
        'cd_score': score,
        'cd_fwd': stats['fwd'],
        'cd_bwd': stats['bwd'],
        'cd_max': float(score.max()) if n else 0.0,
        'cd_mean': float(score.mean()) if n else 0.0,
        'cd_p90': quantile,
        'n_eligible': sum(1 for i, p in enumerate(leaves)
                          if p.depth < max_depth and p.size >= 2
                          and float(stats['valid_frac'][i]) >= min_valid_frac),
        'ext_skipped': sum(1 for i in range(n)
                           if float(stats['valid_frac'][i]) < min_valid_frac),
        'subdivided_depths': sorted(p.depth for _, _, p in cand),
    }


def _leaf_edge_queries(model, samples_per_edge, device):
    """(pids, uv) covering all 4 boundary edges of every leaf."""
    n = model.n_patches
    t = torch.linspace(0.0, 1.0, samples_per_edge, device=device).unsqueeze(1)
    z, o = torch.zeros_like(t), torch.ones_like(t)
    edges = torch.cat([torch.cat([t, z], 1), torch.cat([o, t], 1),
                       torch.cat([t, o], 1), torch.cat([z, t], 1)], dim=0)  # (4S,2)
    uv = edges.repeat(n, 1)
    pids = torch.arange(n, device=device).repeat_interleave(4 * samples_per_edge)
    return pids, uv


@torch.no_grad()
def check_seam_continuity(model, samples_per_edge: int = 9,
                          key_scale: int = 1 << 14, batch: int = 8192):
    """
    Sample every leaf's boundary edges, bucket samples by their (quantized)
    reference-cube position — points shared by 2+ patches land in the same
    bucket — and report the max/mean 3D gap between decoded positions.
    A working data structure gives gaps at float-precision level (~1e-6).
    Use samples_per_edge = 2^k + 1 so parameters are dyadic and exact.
    """
    device = next(model.parameters()).device
    pids, uv = _leaf_edge_queries(model, samples_per_edge, device)
    keys = torch.round(model.cube_xyz(pids, uv).double() * key_scale).long().cpu()
    out = []
    for i in range(0, pids.shape[0], batch):
        out.append(model(pids[i:i + batch], uv[i:i + batch]).cpu())
    xyz = torch.cat(out, dim=0)

    buckets = {}
    for i in range(keys.shape[0]):
        buckets.setdefault(tuple(keys[i].tolist()), []).append(i)
    gaps = []
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        pts = xyz[idxs]
        gaps.append(float((pts.max(0).values - pts.min(0).values).norm()))
    gaps = torch.tensor(gaps) if gaps else torch.zeros(1)
    return {'max_gap': float(gaps.max()), 'mean_gap': float(gaps.mean()),
            'n_shared_samples': int((gaps > -1).sum())}