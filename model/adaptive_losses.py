#!/usr/bin/env python3
# model/adaptive_losses.py
"""
SVD-based tangent/Jacobian regularization for the adaptive cube atlas.

Same energy families as losses.tangent_loss_from_jac (arap / arap_si /
conformal + anti-collapse), but adapted to the quadtree structure:

Because leaves have different parametric sizes, the Jacobian w.r.t. LOCAL
uv scales with leaf size (a depth-d leaf spans 1/2^d of its face, so its
raw t_u, t_v are ~2^d times smaller for the same geometry). Before taking
singular values we rescale the tangents to the FACE-UV frame:

    face_uv = rect_origin + size * local_uv
    dQ/d(face_uv) = dQ/d(local_uv) / size

so every leaf's singular values live on the same scale regardless of depth,
and a single target / threshold is meaningful across the whole atlas.
"""

import torch

from .losses import tangent_loss


def svd_tangent_loss(t_u: torch.Tensor,
                     t_v: torch.Tensor,
                     patch_sizes: torch.Tensor = None,
                     mode: str = 'arap_si',
                     target: float = 1.0,
                     eps: float = 1e-4) -> torch.Tensor:
    """
    Args
    ----
    t_u, t_v    : (B, 3) tangents from surface_jacobian(Q, uv) — derivatives
                  w.r.t. the leaf's LOCAL uv in [0,1]^2.
    patch_sizes : (B,) per-sample leaf size FRACTION (leaf_rect[:, 2]).
                  If given, tangents are rescaled to the face-UV frame so
                  singular values are depth-independent. Pass None only for
                  a uniform-depth atlas.
    mode        : 'arap'      — (S - target)^2: near-isometry to a fixed scale.
                  'arap_si'   — (S - mean(S))^2 with detached mean:
                                scale-invariant near-isometry (recommended
                                default: no target needed, plays well with
                                ongoing subdivision).
                  'conformal' — (S1 - S2)^2: angles preserved, scale free.
    target      : target singular value for mode='arap' (in face-UV units;
                  a cube face is a 2x2 square, so an exactly isometric box
                  fit has S = 2.0 in this frame).
    eps         : anti-collapse floor on singular values.

    Returns scalar: energy + collapse penalty (same structure as losses.py).
    """
    if patch_sizes is not None:
        s = patch_sizes.clamp_min(1e-12).unsqueeze(-1)      # (B, 1)
        t_u = t_u / s
        t_v = t_v / s

    J = torch.stack([t_u, t_v], dim=2)                      # (B, 3, 2)
    S = torch.linalg.svdvals(J)                             # (B, 2)

    # Anti-fold/collapse: keep both singular values above eps.
    collapse = torch.relu(eps - S).pow(2).sum(dim=-1).mean()

    if mode == 'arap':
        energy = ((S - target) ** 2).sum(dim=-1).mean()
    elif mode == 'arap_si':
        s_mean = S.mean(dim=-1, keepdim=True).detach()
        energy = ((S - s_mean) ** 2).sum(dim=-1).mean()
    elif mode == 'conformal':
        energy = (S[:, 0] - S[:, 1]).pow(2).mean()
    elif mode == 'collapse':
        energy = torch.zeros((), device=J.device, dtype=J.dtype)
    else:
        raise ValueError(f"unknown svd loss mode: {mode}")

    return energy + collapse


def adaptive_svd_loss(model, pids: torch.Tensor,
                      t_u: torch.Tensor, t_v: torch.Tensor,
                      mode: str = 'arap', target: float = 0.25,
                      eps: float = 1e-4) -> torch.Tensor:
    """
    Convenience wrapper: fetches per-sample leaf sizes from the complex's
    flat tensors and applies svd_tangent_loss with depth normalization.
    Reuses tangents already computed by surface_jacobian so the loss shares
    one autograd graph with the Dirichlet tangent term when both are on.
    """
    model.complex._sync_device()
    sizes = model.complex.leaf_rect[pids, 2]                # (B,) size fraction
    return tangent_loss(
        model=model,
        pids=pids,
        t_u=t_u,
        t_v=t_v,
        mode=mode,
        target=target,
        eps=eps,
        patch_sizes=sizes,
        normalize_patch_scale=True,
    )