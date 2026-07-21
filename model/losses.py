#!/usr/bin/env python3
# losses.py

import math
import torch


def mu_warmup_schedule(epoch: int, warmup_epochs: int, mu_target: float,
                       schedule: str = 'cosine',
                       delay_epochs: int = 300) -> float:
    """
    Compute the effective mu value based on a warmup schedule with an optional
    initial delay phase where μ stays at exactly 0.
    """
    if mu_target <= 0:
        return mu_target

    # Phase 1: delay — mu stays at zero
    if epoch <= delay_epochs:
        return 0.0

    # Shift epoch so the ramp phase starts at epoch 1 relative to itself
    ramp_epoch = epoch - delay_epochs

    # Phase 2: warmup ramp
    if warmup_epochs <= 0:
        return mu_target

    if ramp_epoch >= warmup_epochs:
        return mu_target

    t = ramp_epoch / warmup_epochs  # t ∈ (0, 1)

    if schedule == 'linear':
        factor = t
    elif schedule == 'cosine':
        factor = 0.5 * (1.0 - math.cos(math.pi * t))
    elif schedule == 'exponential':
        # 1 - exp(-k*t); k=5 means ~99.3% at t=1
        factor = 1.0 - math.exp(-5.0 * t)
    elif schedule == 'sigmoid':
        # Shifted sigmoid: smooth S-curve from ~0 to ~1 over t∈[0,1]
        x = 12.0 * (t - 0.5)  # map t∈[0,1] → x∈[-6,6]
        factor = 1.0 / (1.0 + math.exp(-x))
    else:
        raise ValueError(f"Unknown warmup schedule: {schedule}. "
                         f"Use 'linear', 'cosine', 'exponential', or 'sigmoid'.")

    return mu_target * factor


def chamfer_distance(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Compute the symmetric Chamfer distance between two point sets."""
    D = torch.cdist(P, Q)
    loss_P = D.min(dim=1).values.mean()
    loss_Q = D.min(dim=0).values.mean()
    return loss_P + loss_Q


def chamfer_distance_chunked(P: torch.Tensor, Q: torch.Tensor,
                             chunk_size: int = 2048) -> torch.Tensor:
    """Compute Chamfer distance in chunks to reduce memory use."""
    N, M = P.shape[0], Q.shape[0]

    min_p2q = []
    for i in range(0, N, chunk_size):
        d = torch.cdist(P[i:i + chunk_size], Q)
        min_p2q.append(d.min(dim=1).values)
    loss_P = torch.cat(min_p2q).mean()

    min_q2p = []
    for i in range(0, M, chunk_size):
        d = torch.cdist(Q[i:i + chunk_size], P)
        min_q2p.append(d.min(dim=1).values)
    loss_Q = torch.cat(min_q2p).mean()

    return loss_P + loss_Q


def surface_jacobian(Q, uv):
    """
    Compute tangent vectors with autograd.

    Args:
        Q: Surface points from `F(uv)`.
        uv: UV inputs with gradients enabled.
    Returns:
        Tuple `(t_u, t_v)`.
    """
    ones = torch.ones_like(Q[:, 0])
    gx = torch.autograd.grad(Q[:, 0], uv, ones, create_graph=True)[0]
    gy = torch.autograd.grad(Q[:, 1], uv, ones, create_graph=True)[0]
    gz = torch.autograd.grad(Q[:, 2], uv, ones, create_graph=True)[0]
    t_u = torch.stack([gx[:, 0], gy[:, 0], gz[:, 0]], dim=-1)
    t_v = torch.stack([gx[:, 1], gy[:, 1], gz[:, 1]], dim=-1)
    return t_u, t_v


def tangent_loss_from_jac(t_u, t_v, mode='arap', eps=1e-4):
    """
    Compute Jacobian-based tangent regularization.
    """
    J = torch.stack([t_u, t_v], dim=2)

    if mode == 'conformal_fff':
        # First fundamental form entries avoid SVD entirely.
        E = (t_u * t_u).sum(dim=-1)
        G = (t_v * t_v).sum(dim=-1)
        Fd = (t_u * t_v).sum(dim=-1)
        energy = ((E - G) ** 2 + 4.0 * Fd ** 2).mean()
        # Collapse guard from the local area term.
        area2 = torch.clamp(E * G - Fd ** 2, min=0.0)
        collapse = torch.relu(eps ** 2 - area2).mean()
        return energy + collapse

    # Use singular values only. Singular vectors are not needed.
    S = torch.linalg.svdvals(J)

    collapse = torch.relu(eps - S).pow(2).sum(dim=-1).mean()

    if mode == 'arap':
        energy = ((S - 0.25) ** 2).sum(dim=-1).mean()
    elif mode == 'arap_si':
        s_mean = S.mean(dim=-1, keepdim=True).detach()
        energy = ((S - s_mean) ** 2).sum(dim=-1).mean()
    elif mode == 'conformal':
        energy = (S[:, 0] - S[:, 1]).pow(2).mean()
    elif mode == 'collapse':
        energy = torch.zeros((), device=J.device, dtype=J.dtype)
    else:
        raise ValueError(f"unknown tangent mode: {mode}")

    return energy + collapse


def tangent_fold_loss(Q, uv):
    """Wrapper that computes its own Jacobian."""
    t_u, t_v = surface_jacobian(Q, uv)
    return tangent_loss_from_jac(t_u, t_v)


def normal_consistency_loss(Q, uv, P_data, N_data):
    """
    Compute single-patch normal consistency loss.
    """
    t_u, t_v = surface_jacobian(Q, uv)
    n_surf = torch.cross(t_u, t_v, dim=-1)
    n_surf = n_surf / (n_surf.norm(dim=-1, keepdim=True) + 1e-8)

    D = torch.cdist(Q, P_data)
    nn_idx = D.argmin(dim=1)
    n_target = N_data[nn_idx]

    cos = torch.sum(n_surf * n_target, dim=-1)
    return (1.0 - cos).mean()


def chamfer_1d(pts_a, pts_b):
    """Compute Chamfer distance between two boundary point sets."""
    diff_ab = pts_a.unsqueeze(1) - pts_b.unsqueeze(0)
    dist_ab = (diff_ab ** 2).sum(dim=2)
    min_ab = dist_ab.min(dim=1)[0].mean()
    min_ba = dist_ab.min(dim=0)[0].mean()
    return min_ab + min_ba


def boundary_chamfer_loss(F_model, grid_topology, n_boundary_samples=50, device='cuda'):
    """
    Compute boundary Chamfer distance between adjacent patches.
    """
    n_rows, n_cols = grid_topology.shape
    t = torch.linspace(0, 1, n_boundary_samples, device=device).unsqueeze(1)

    total_loss = torch.tensor(0.0, device=device)
    n_edges = 0

    for r in range(n_rows):
        for c in range(n_cols):
            patch_id = int(grid_topology[r, c])

            if c + 1 < n_cols:
                neighbor_id = int(grid_topology[r, c + 1])
                uv_i = torch.cat([t, torch.ones_like(t)], dim=1)
                uv_j = torch.cat([t, torch.zeros_like(t)], dim=1)
                pts_i = F_model(patch_id, uv_i)
                pts_j = F_model(neighbor_id, uv_j)
                total_loss += chamfer_1d(pts_i, pts_j)
                n_edges += 1

            if r + 1 < n_rows:
                neighbor_id = int(grid_topology[r + 1, c])
                uv_i = torch.cat([torch.ones_like(t), t], dim=1)
                uv_j = torch.cat([torch.zeros_like(t), t], dim=1)
                pts_i = F_model(patch_id, uv_i)
                pts_j = F_model(neighbor_id, uv_j)
                total_loss += chamfer_1d(pts_i, pts_j)
                n_edges += 1

    if n_edges > 0:
        total_loss /= n_edges
    return total_loss