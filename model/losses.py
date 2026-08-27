#!/usr/bin/env python3
# losses.py

import math
import torch


def _sample_rectangle_patch_edge(n_rows: int,
                                 n_cols: int,
                                 row: int,
                                 col: int,
                                 edge_name: str,
                                 t: torch.Tensor) -> torch.Tensor:
    """
    Sample the rectangle edge segment assigned to one specific outer patch.

    The rectangle is partitioned exactly like the pretrained flat sheet:
    each outer patch owns only its corresponding sub-segment of the global
    rectangle perimeter, not the full side.
    """
    row_f = torch.full_like(t, float(row))
    col_f = torch.full_like(t, float(col))

    if edge_name == 'top':
        global_u = (row_f + 0.0) / n_rows
        global_v = (col_f + t) / n_cols
    elif edge_name == 'bottom':
        global_u = (row_f + 1.0) / n_rows
        global_v = (col_f + t) / n_cols
    elif edge_name == 'left':
        global_u = (row_f + t) / n_rows
        global_v = (col_f + 0.0) / n_cols
    elif edge_name == 'right':
        global_u = (row_f + t) / n_rows
        global_v = (col_f + 1.0) / n_cols
    else:
        raise ValueError(f"Unknown rectangle edge: {edge_name}")

    x = 2.0 * global_u - 1.0
    y = 2.0 * global_v - 1.0
    z = torch.zeros_like(x)
    return torch.cat([x, y, z], dim=1)


def outer_boundary_rectangle_loss(F_model,
                                  n_boundary_samples: int = 64,
                                  loss_type: str = 'l1',
                                  device: str = 'cuda'):
    """
    Match the true global outer border of the multi-patch sheet to the
    rectangle boundary [-1,1]^2 x {0} using patch-aware correspondence.

    Only patch edges on the outer grid boundary are constrained. Internal
    shared edges are excluded. Each outer patch edge is matched only to its
    own rectangle sub-segment, consistent with the flat-sheet pretraining.
    """
    if n_boundary_samples <= 0:
        return torch.tensor(0.0, device=device)

    t = torch.linspace(0.0, 1.0, n_boundary_samples, device=device).unsqueeze(1)

    if loss_type == 'l1':
        point_loss = torch.nn.L1Loss(reduction='mean')
    elif loss_type == 'mse':
        point_loss = torch.nn.MSELoss(reduction='mean')
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'l1' or 'mse'.")

    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    for row in range(F_model.n_rows):
        for col in range(F_model.n_cols):
            patch_id = row * F_model.n_cols + col

            if row == 0:
                uv = torch.cat([torch.zeros_like(t), t], dim=1)
                pred = F_model(patch_id, uv)
                target = _sample_rectangle_patch_edge(
                    F_model.n_rows, F_model.n_cols, row, col, 'top', t)
                total_loss = total_loss + point_loss(pred, target)
                n_terms += 1

            if row == F_model.n_rows - 1:
                uv = torch.cat([torch.ones_like(t), t], dim=1)
                pred = F_model(patch_id, uv)
                target = _sample_rectangle_patch_edge(
                    F_model.n_rows, F_model.n_cols, row, col, 'bottom', t)
                total_loss = total_loss + point_loss(pred, target)
                n_terms += 1

            if col == 0:
                uv = torch.cat([t, torch.zeros_like(t)], dim=1)
                pred = F_model(patch_id, uv)
                target = _sample_rectangle_patch_edge(
                    F_model.n_rows, F_model.n_cols, row, col, 'left', t)
                total_loss = total_loss + point_loss(pred, target)
                n_terms += 1

            if col == F_model.n_cols - 1:
                uv = torch.cat([t, torch.ones_like(t)], dim=1)
                pred = F_model(patch_id, uv)
                target = _sample_rectangle_patch_edge(
                    F_model.n_rows, F_model.n_cols, row, col, 'right', t)
                total_loss = total_loss + point_loss(pred, target)
                n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / n_terms


def sample_outer_boundary_correspondence(F_model,
                                         n_boundary_samples: int = 64,
                                         device: str = 'cuda'):
    """
    Sample model/target point pairs used by the outer-boundary rectangle loss.

    Returns:
        dict with batched model points, target points, patch ids, and edge labels.
    """
    if n_boundary_samples <= 0:
        empty_pts = torch.empty(0, 0, 3, device=device)
        empty_ids = torch.empty(0, dtype=torch.long, device=device)
        return {
            'model_points': empty_pts,
            'target_points': empty_pts,
            'patch_ids': empty_ids,
            'edge_names': [],
        }

    t = torch.linspace(0.0, 1.0, n_boundary_samples, device=device).unsqueeze(1)
    model_batches = []
    target_batches = []
    patch_ids = []
    edge_names = []

    for row in range(F_model.n_rows):
        for col in range(F_model.n_cols):
            patch_id = row * F_model.n_cols + col

            if row == 0:
                uv = torch.cat([torch.zeros_like(t), t], dim=1)
                model_batches.append(F_model(patch_id, uv))
                target_batches.append(_sample_rectangle_patch_edge(
                    F_model.n_rows, F_model.n_cols, row, col, 'top', t))
                patch_ids.append(patch_id)
                edge_names.append('top')

            if row == F_model.n_rows - 1:
                uv = torch.cat([torch.ones_like(t), t], dim=1)
                model_batches.append(F_model(patch_id, uv))
                target_batches.append(_sample_rectangle_patch_edge(
                    F_model.n_rows, F_model.n_cols, row, col, 'bottom', t))
                patch_ids.append(patch_id)
                edge_names.append('bottom')

            if col == 0:
                uv = torch.cat([t, torch.zeros_like(t)], dim=1)
                model_batches.append(F_model(patch_id, uv))
                target_batches.append(_sample_rectangle_patch_edge(
                    F_model.n_rows, F_model.n_cols, row, col, 'left', t))
                patch_ids.append(patch_id)
                edge_names.append('left')

            if col == F_model.n_cols - 1:
                uv = torch.cat([t, torch.ones_like(t)], dim=1)
                model_batches.append(F_model(patch_id, uv))
                target_batches.append(_sample_rectangle_patch_edge(
                    F_model.n_rows, F_model.n_cols, row, col, 'right', t))
                patch_ids.append(patch_id)
                edge_names.append('right')

    return {
        'model_points': torch.stack(model_batches, dim=0),
        'target_points': torch.stack(target_batches, dim=0),
        'patch_ids': torch.tensor(patch_ids, dtype=torch.long, device=device),
        'edge_names': edge_names,
    }


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


def _ddf_closest_point(query: torch.Tensor, candidates: torch.Tensor,
                       k: int = 5, chunk_size: int = 2048,
                       eps: float = 1e-8) -> torch.Tensor:
    """
    Approximate each query point's closest point on a point-cloud surface as
    an inverse-squared-distance weighted average of its k nearest neighbors. 
    Differentiable w.r.t. "candidates".
    """
    k = min(k, candidates.shape[0])
    closest_chunks = []
    for i in range(0, query.shape[0], chunk_size):
        q = query[i:i + chunk_size]
        d = torch.cdist(q, candidates)
        knn_dist, knn_idx = torch.topk(d, k, dim=1, largest=False)
        knn_pts = candidates[knn_idx]
        w = 1.0 / (knn_dist ** 2 + eps)
        w = w / w.sum(dim=1, keepdim=True)
        closest_chunks.append((w.unsqueeze(-1) * knn_pts).sum(dim=1))
    return torch.cat(closest_chunks, dim=0)


def directional_distance_field(query: torch.Tensor, candidates: torch.Tensor,
                               k: int = 5, chunk_size: int = 2048) -> torch.Tensor:
    """
    Directional Distance Field
    for each query (reference) point, the unsigned distance to a point-cloud surface
    concatenated with the direction toward its closest point.

    Returns:
        (M, 4) tensor: [distance, direction_x, direction_y, direction_z].
    """
    closest = _ddf_closest_point(query, candidates, k=k, chunk_size=chunk_size)
    direction = closest - query
    dist = direction.norm(dim=-1, keepdim=True)
    return torch.cat([dist, direction], dim=-1)


def sample_ddf_reference_points(points: torch.Tensor, sigma: float = 0.05,
                                n_points: int = None) -> torch.Tensor:
    """
    Generate DDM reference points near a surface by adding Gaussian noise to
    a (sub)sample of its points.
    """
    if n_points is None or n_points >= points.shape[0]:
        base = points
    else:
        idx = torch.randint(0, points.shape[0], (n_points,), device=points.device)
        base = points[idx]
    return base + sigma * torch.randn_like(base)


def directional_distance_loss(pred_points: torch.Tensor, ref_points: torch.Tensor,
                              ref_ddf_gt: torch.Tensor, k: int = 5,
                              beta: float = 20.0, chunk_size: int = 2048) -> torch.Tensor:
    """
    DDM surface-fitting loss: compares the DDF of
    the current predicted point cloud against a precomputed ground-truth DDF,
    both evaluated at the same fixed `ref_points`.

    `ref_ddf_gt` is expected to be precomputed once (target is static) via
    `directional_distance_field(ref_points, target_points, ...)`.

    The confidence weight `s(q) = exp(-beta * d(q))` is detached from the
    graph before weighting, so the model can't reduce the loss by driving a
    reference point's distance past 1/beta (where s(q)*d(q) would otherwise
    start decreasing with larger d).
    """
    ddf_pred = directional_distance_field(ref_points, pred_points, k=k, chunk_size=chunk_size)
    d = (ddf_pred - ref_ddf_gt).abs().sum(dim=-1)
    if beta > 0:
        s = torch.exp(-beta * d.detach())
        return (s * d).sum() / s.sum().clamp_min(1e-8)
    return d.mean()


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


def tangent_loss_from_jac(t_u, t_v, mode='dirichlet', eps=1e-4, scale_invariant=True):
    """
    Compute Jacobian-based tangent regularization with optional scale normalization.
    """
    # # 1. OPTIONAL: Normalize the scale of the Jacobian vectors per-sample/patch
    # if scale_invariant:
    #     # Calculate the local patch scale (Frobenius norm of the Jacobian)
    #     # This represents the average "stretch" factor of this specific point
    #     local_scale = torch.sqrt((t_u ** 2).sum(dim=-1) + (t_v ** 2).sum(dim=-1) + 1e-8)
        
    #     # Keep dimensions aligned for broadcasting [batch, 1]
    #     local_scale = local_scale.unsqueeze(-1) 
        
    #     # Normalize vectors so the local metric scale is 1.0
    #     t_u = t_u / local_scale
    #     t_v = t_v / local_scale

    J = torch.stack([t_u, t_v], dim=2)

    ## dirichlet
    ### \int_S(||df/du||^2 + ||df/dv||^2)

    e_dirichlet = 1.0*torch.mean(0.5*torch.sum(J ** 2, dim=1))

    if mode == 'conformal_fff':
        E = (t_u * t_u).sum(dim=-1)
        G = (t_v * t_v).sum(dim=-1)
        Fd = (t_u * t_v).sum(dim=-1)
        energy = ((E - G) ** 2 + 4.0 * Fd ** 2).mean()
        area2 = torch.clamp(E * G - Fd ** 2, min=0.0)
        collapse = torch.relu(eps ** 2 - area2).mean()
        return energy + collapse

    S = torch.linalg.svdvals(J)

    # If scaled to 1, a hardcoded eps (like 1e-4) is now safe and universally meaningful
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
    elif mode == 'dirichlet':
        return e_dirichlet + collapse
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