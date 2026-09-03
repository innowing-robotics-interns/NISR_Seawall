#!/usr/bin/env python3
# model/subdivision.py
"""Distortion measurement, distortion-driven subdivision, seam diagnostics."""

import torch

from .losses import surface_jacobian


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
            t_u, t_v = surface_jacobian(Q, uv)
        E = (t_u * t_u).sum(-1)
        G = (t_v * t_v).sum(-1)
        Fd = (t_u * t_v).sum(-1)
        if mode == 'area':
            e = torch.sqrt(torch.clamp(E * G - Fd ** 2, min=0.0))
        elif mode == 'dirichlet':
            e = 0.5 * (E + G)
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