#!/usr/bin/env python3
# correspondence.py

"""
Build a fixed point -> (patch_id, u, v) correspondence table for a trained
multi-patch forward map.

Used to switch training from Chamfer distance (soft, per-step nearest-neighbor
matching) to direct pointwise MSE regression against a frozen correspondence,
once the model's geometry is already close enough to the target for nearest-
neighbor matching to be meaningful.
"""

import torch


@torch.no_grad()
def _build_candidate_pool(F, search_resolution: int, device: str, eval_chunk: int = 4096):
    """
    Densely sample every patch on a uniform UV grid and evaluate F, building a
    pool of (xyz, patch_id, uv) candidates for the nearest-neighbor search.
    """
    u = torch.linspace(0.0, 1.0, search_resolution, device=device)
    v = torch.linspace(0.0, 1.0, search_resolution, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
    uv_grid = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)  # (R*R, 2)
    n_per_patch = uv_grid.shape[0]

    pool_xyz = []
    pool_patch_ids = []
    pool_uv = []

    for patch_id in range(F.n_patches):
        patch_idx = torch.full((n_per_patch,), patch_id, dtype=torch.long, device=device)
        xyz_chunks = []
        for i in range(0, n_per_patch, eval_chunk):
            xyz_chunks.append(F(patch_idx[i:i + eval_chunk], uv_grid[i:i + eval_chunk]))
        pool_xyz.append(torch.cat(xyz_chunks, dim=0))
        pool_patch_ids.append(patch_idx)
        pool_uv.append(uv_grid)

    return (torch.cat(pool_xyz, dim=0),
            torch.cat(pool_patch_ids, dim=0),
            torch.cat(pool_uv, dim=0))


@torch.no_grad()
def _nearest_neighbor_assignment(points, pool_xyz, pool_patch_ids, pool_uv, point_chunk: int = 1000):
    """Chunked nearest-neighbor search assigning each point the closest candidate's (patch_id, u, v)."""
    n_points = points.shape[0]
    patch_ids = torch.empty(n_points, dtype=torch.long, device=points.device)
    uv = torch.empty(n_points, 2, dtype=points.dtype, device=points.device)

    for i in range(0, n_points, point_chunk):
        chunk = points[i:i + point_chunk]
        dists = torch.cdist(chunk, pool_xyz)          # (chunk, n_pool)
        nn_idx = dists.argmin(dim=1)
        patch_ids[i:i + point_chunk] = pool_patch_ids[nn_idx]
        uv[i:i + point_chunk] = pool_uv[nn_idx]

    return patch_ids, uv


def build_hard_correspondence(F, points: torch.Tensor,
                              search_resolution: int = 64,
                              refine_steps: int = 200,
                              refine_lr: float = 1e-2,
                              device: str = 'cuda'):
    """
    Assign each point in `points` a fixed (patch_id, u, v) correspondence.

    Stage 1 (hard patch assignment): densely sample every patch and pick each
    point's nearest candidate via chunked nearest-neighbor search. The patch id
    from this stage is never revisited.
    Stage 2 (continuous refinement): with `patch_id` fixed and F's weights
    frozen, gradient-descend each point's own (u, v) to minimize its distance
    to the target, past the resolution limit of the search grid.

    Returns:
        patch_ids: (N,) long tensor, fixed patch ownership per point.
        uv: (N, 2) float tensor, refined local UV coordinates, detached.
    """
    points = points.to(device)
    pool_xyz, pool_patch_ids, pool_uv = _build_candidate_pool(F, search_resolution, device)
    patch_ids, uv0 = _nearest_neighbor_assignment(points, pool_xyz, pool_patch_ids, pool_uv)

    was_training = F.training
    F.eval()
    param_grad_state = [(p, p.requires_grad) for p in F.parameters()]
    for p in F.parameters():
        p.requires_grad_(False)

    uv_param = uv0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([uv_param], lr=refine_lr)

    for _ in range(refine_steps):
        opt.zero_grad()
        pred = F(patch_ids, uv_param)
        loss = torch.nn.functional.mse_loss(pred, points)
        loss.backward()
        opt.step()
        with torch.no_grad():
            uv_param.clamp_(0.0, 1.0)

    for p, requires_grad in param_grad_state:
        p.requires_grad_(requires_grad)
    if was_training:
        F.train()

    return patch_ids.detach(), uv_param.detach()
