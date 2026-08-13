#!/usr/bin/env python3
# model.py

import math

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
#  Mean Value Coordinates (MVC) — vectorized, differentiable PyTorch port
#  of the demo's mvc.mvc_weights(). This is the core of the bilinear→MVC switch.
# ─────────────────────────────────────────────────────────────────────────────
def mvc_weights_torch(points: torch.Tensor,
                      polys: torch.Tensor,
                      eps: float = 1e-7,
                      edge_eps: float = 1e-6) -> torch.Tensor:
    """
    Batched Mean Value Coordinates (Floater 2003).

    Args
    ----
    points : (B, 2)      query points, one per row.
    polys  : (B, K, 2)   the K polygon vertices for EACH query point, given in a
                         consistent CCW order. In Phase 1 (regular grid) K == 4
                         and the polygon is always the local-UV unit square, so
                         the same square is broadcast to every row. In Phase 2
                         (quadtree) K becomes the leaf's n-gon (corners + hanging
                         nodes) and can differ per query — the math below is
                         already written for the general K case.

    Returns
    -------
    (B, K) weights, non-negative for convex polygons, summing to 1 per row.

    Formula (tangent-of-half-angle):
        tan(alpha_i / 2) = (r_i * r_{i+1} - <s_i, s_{i+1}>) / cross(s_i, s_{i+1})
        w_i = ( tan(alpha_{i-1}/2) + tan(alpha_i/2) ) / r_i
    where s_i = v_i - p and r_i = |s_i|.

    Two degenerate cases are handled exactly like the demo:
      * Case A — query coincides with a vertex  -> one-hot on that vertex.
      * Case B — query lies ON an edge          -> LINEAR interpolation between
        that edge's two endpoints. This is the "on the edge we only use linear
        interpolation" rule you asked for, and it is what keeps neighbouring
        patches continuous when hanging nodes appear in Phase 2. Note it is not
        a hack: as p approaches an edge, MVC naturally converges to linear
        interpolation along it; Case B just evaluates that limit robustly.
    """
    B, K, _ = polys.shape

    s = polys - points.unsqueeze(1)                    # (B, K, 2) vectors p->v_i
    r = torch.linalg.norm(s, dim=-1)                   # (B, K) distances

    # "next" vertex (i -> i+1), wrapping around the polygon.
    s_nxt = torch.roll(s, shifts=-1, dims=1)           # (B, K, 2)
    r_nxt = torch.roll(r, shifts=-1, dims=1)           # (B, K)

    # cross / dot per edge (i, i+1); indexed by the edge's start vertex i.
    cross = s[..., 0] * s_nxt[..., 1] - s[..., 1] * s_nxt[..., 0]   # (B, K)
    dot = (s * s_nxt).sum(dim=-1)                                   # (B, K)

    # tan(alpha_i / 2). Clamp the denominator so degenerate rows stay FINITE
    # (never inf/nan): those rows are overwritten below via Case A / Case B, but
    # keeping them finite avoids the classic torch.where 0*inf=nan grad trap.
    safe_cross = torch.where(cross.abs() < eps,
                             torch.full_like(cross, eps), cross)
    tan_half = (r * r_nxt - dot) / safe_cross          # (B, K), edge i
    tan_half_prev = torch.roll(tan_half, shifts=1, dims=1)  # tan(alpha_{i-1}/2)

    r_safe = torch.where(r < eps, torch.full_like(r, eps), r)
    w = (tan_half_prev + tan_half) / r_safe            # (B, K) unnormalized

    # --- interior (generic) case: normalize ---------------------------------
    w_sum = w.sum(dim=1, keepdim=True)
    w_sum = torch.where(w_sum.abs() < eps, torch.ones_like(w_sum), w_sum)
    w_interior = w / w_sum

    # --- Case B: query on an edge -> linear interpolation on that edge -------
    # An edge (i, i+1) contains p iff p is collinear (cross≈0) AND between the
    # endpoints (dot<0, i.e. the vectors to the two endpoints point opposite).
    on_edge = (cross.abs() < edge_eps) & (dot < 0.0)   # (B, K) flagged at edge i
    any_edge = on_edge.any(dim=1)                      # (B,)

    ei = on_edge.to(w.dtype)
    denom_edge = torch.where((r + r_nxt) < eps,
                             torch.ones_like(r), r + r_nxt)
    w_i = (r_nxt / denom_edge) * ei                    # weight to vertex i
    w_ip1 = (r / denom_edge) * ei                      # weight to vertex i+1
    w_ip1_at_next = torch.roll(w_ip1, shifts=1, dims=1)  # place at position i+1
    w_edge = w_i + w_ip1_at_next
    w_edge_sum = w_edge.sum(dim=1, keepdim=True)       # guards double-flag corners
    w_edge_sum = torch.where(w_edge_sum.abs() < eps,
                             torch.ones_like(w_edge_sum), w_edge_sum)
    w_edge = w_edge / w_edge_sum

    # --- Case A: query on a vertex -> one-hot -------------------------------
    on_vertex = r < eps                                # (B, K)
    any_vertex = on_vertex.any(dim=1)                  # (B,)
    w_vert = on_vertex.to(w.dtype)
    w_vert_sum = w_vert.sum(dim=1, keepdim=True)
    w_vert_sum = torch.where(w_vert_sum.abs() < eps,
                             torch.ones_like(w_vert_sum), w_vert_sum)
    w_vert = w_vert / w_vert_sum

    # priority: vertex  >  edge  >  interior
    out = torch.where(any_edge.unsqueeze(1), w_edge, w_interior)
    out = torch.where(any_vertex.unsqueeze(1), w_vert, out)
    return out


# Neural network architecture
class PositionalEncoding(nn.Module):
    """Apply sin/cos Fourier features to the input coordinates."""
    def __init__(self, d_in: int, L: int = 6):
        super().__init__()
        self.d_in = d_in
        self.L = L
        self.register_buffer('freq', 2.0 ** torch.arange(L).float() * np.pi)

    @property
    def d_out(self):
        return self.d_in * 2 * self.L

    def forward(self, x):
        parts = []
        for i in range(self.d_in):
            v = x[:, i:i + 1] * self.freq
            parts += [v.sin(), v.cos()]
        return torch.cat(parts, dim=-1)


class SkipMLP(nn.Module):
    """MLP with a midpoint skip connection and Softplus activations."""
    def __init__(self, d_in: int, d_out: int,
                 W: int = 256, D: int = 6, out_act: str = None, beta: float = 1.0):
        super().__init__()
        self.skip = D // 2
        self.lins = nn.ModuleList()
        for i in range(D):
            fan_in = d_in if i == 0 else (W + d_in if i == self.skip else W)
            self.lins.append(nn.Linear(fan_in, W))
        self.head = nn.Linear(W, d_out)
        self.act = nn.Softplus(beta=beta)
        self.out_act = out_act

    def forward(self, x0):
        h = x0
        for i, lin in enumerate(self.lins):
            if i == self.skip:
                h = torch.cat([h, x0], dim=-1)
            h = self.act(lin(h))
        h = self.head(h)
        if self.out_act == 'sigmoid':
            h = 0.5 + torch.atan(h / 3.0) / math.pi  # smooth sigmoid in [0,1]
        return h


class ForwardMap(nn.Module):
    """Map local UV coordinates to 3D points."""
    def __init__(self, L: int = 6, W: int = 256, D: int = 6, beta: float = 1.0):
        super().__init__()
        self.L = L
        if L > 0:
            self.pe = PositionalEncoding(2, L)
            d_in = self.pe.d_out
        else:
            self.pe = None
            d_in = 2
        self.net = SkipMLP(d_in, 3, W, D, beta=beta)

    def forward(self, uv):
        """Evaluate the forward map on UV samples."""
        x = self.pe(uv) if self.pe is not None else uv
        return self.net(x)


class InverseMap(nn.Module):
    """Map 3D points back to UV coordinates."""
    def __init__(self, L: int = 6, W: int = 256, D: int = 6, beta: float = 1.0):
        super().__init__()
        self.L = L
        if L > 0:
            self.pe = PositionalEncoding(3, L)
            d_in = self.pe.d_out
        else:
            self.pe = None
            d_in = 3
        self.net = SkipMLP(d_in, 2, W, D, out_act='sigmoid', beta=beta)

    def forward(self, xyz):
        """Evaluate the inverse map on 3D samples."""
        x = self.pe(xyz) if self.pe is not None else xyz
        return self.net(x)


# Feature complex and multi-patch maps.
class FeatureComplex(nn.Module):
    """Shared-vertex MVC feature complex for an adaptive quadtree."""
    def __init__(self, topology: dict, d_features: int = 64):
        super().__init__()
        def tensor(value, dtype):
            return value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=dtype)

        self.register_buffer('vertex_uv', tensor(topology['vertex_uv'], torch.float32))
        self.register_buffer('leaf_bbox', tensor(topology['leaf_bbox'], torch.float32))
        self.register_buffer('leaf_poly_ids', tensor(topology['leaf_poly_ids'], torch.long))
        self.register_buffer('leaf_poly_uv', tensor(topology['leaf_poly_uv'], torch.float32))
        self.register_buffer('leaf_k', tensor(topology['leaf_k'], torch.long))
        self.register_buffer('leaf_count', tensor(topology['leaf_count'], torch.long))
        self.d_features = d_features
        self.n_vertices = int(self.vertex_uv.shape[0])
        self.n_leaves = int(self.leaf_bbox.shape[0])
        self.vertex_features = nn.Parameter(torch.randn(self.n_vertices, d_features) * 0.1)

    def topology_state(self):
        return {
            'vertex_uv': self.vertex_uv.detach().cpu(),
            'leaf_bbox': self.leaf_bbox.detach().cpu(),
            'leaf_poly_ids': self.leaf_poly_ids.detach().cpu(),
            'leaf_poly_uv': self.leaf_poly_uv.detach().cpu(),
            'leaf_k': self.leaf_k.detach().cpu(),
            'leaf_count': self.leaf_count.detach().cpu(),
        }

    def interpolate(self, leaf_id, uv):
        k = int(self.leaf_k[leaf_id].item())
        if k == 0:
            raise ValueError(f'Cannot interpolate inactive leaf {leaf_id}')
        ids = self.leaf_poly_ids[leaf_id, :k]
        poly = self.leaf_poly_uv[leaf_id, :k]
        polys = poly.unsqueeze(0).expand(uv.shape[0], -1, -1)
        weights = mvc_weights_torch(uv, polys)
        return weights @ self.vertex_features[ids]


class MultiPatchForwardMap(nn.Module):
    """Vectorized quadtree forward map using shared MVC vertex features."""
    def __init__(self, topology: dict, d_features: int = 64,
                 L: int = 8, W: int = 256, D: int = 6, beta: float = 5.0):
        super().__init__()
        self.complex = FeatureComplex(topology, d_features)
        self.n_leaves = self.complex.n_leaves
        self.n_patches = self.n_leaves
        self.d_features = d_features
        self.L = L

        if L > 0:
            self.pe = PositionalEncoding(2, L)
            d_pe = self.pe.d_out
        else:
            self.pe = None
            d_pe = 0

        self.decoder = SkipMLP(d_features + d_pe, 3, W, D, beta=beta)

    @property
    def active_patch_ids(self):
        return torch.nonzero(self.complex.leaf_count > 0,
                              as_tuple=False).flatten().tolist()

    def topology_state(self):
        return self.complex.topology_state()

        # Optional flat-plane initialization.
        # nn.init.zeros_(self.decoder.head.weight)
        # nn.init.zeros_(self.decoder.head.bias)

    def forward(self, patch_idx, uv: torch.Tensor):
        """
        Args:
            patch_idx: Patch index or batch of patch indices.
            uv: Local UV coordinates.
        Returns:
            Predicted 3D points.
        """
        if not torch.is_tensor(patch_idx):
            patch_idx = int(patch_idx)
            features = self.complex.interpolate(patch_idx, uv)
            bbox = self.complex.leaf_bbox[patch_idx]
            global_u = bbox[0] + uv[:, 0:1] * (bbox[2] - bbox[0])
            global_v = bbox[1] + uv[:, 1:2] * (bbox[3] - bbox[1])
        else:
            patch_idx = patch_idx.to(uv.device)
            features = uv.new_empty(uv.shape[0], self.d_features)
            global_u = uv.new_empty(uv.shape[0], 1)
            global_v = uv.new_empty(uv.shape[0], 1)
            for leaf_id in torch.unique(patch_idx).tolist():
                mask = patch_idx == leaf_id
                features[mask] = self.complex.interpolate(int(leaf_id), uv[mask])
                bbox = self.complex.leaf_bbox[int(leaf_id)]
                global_u[mask] = bbox[0] + uv[mask, 0:1] * (bbox[2] - bbox[0])
                global_v[mask] = bbox[1] + uv[mask, 1:2] * (bbox[3] - bbox[1])

        if self.pe is not None:
            pe = self.pe(torch.cat([global_u, global_v], dim=1))
            dec_in = torch.cat([features, pe], dim=1)
        else:
            dec_in = features

        # Optional explicit flat-plane embedding.
        # flat = torch.cat([
        #     2 * global_u - 1,
        #     2 * global_v - 1,
        #     torch.zeros_like(global_u)
        # ], dim=1)

        correction = self.decoder(dec_in)

        return correction


class MultiPatchInverseMap(nn.Module):
    """
    Multi-patch inverse map from 3D points to local UV coordinates.

    NOTE (Phase 1): `de_interpolate` analytically inverts BILINEAR interpolation
    and therefore no longer matches the forward map now that F uses MVC. Per the
    project decision this inverse map is a deprecated experiment and is NOT used
    by the training loop, so it is left untouched. Do not rely on cycle
    consistency through G while MVC is active.
    """
    def __init__(self, feature_complex: FeatureComplex, d_features: int = 64,
                 L: int = 0, W: int = 256, D: int = 6, beta: float = 5.0):
        super().__init__()
        self.d_features = d_features
        self.L = L

        # Keep a plain reference to avoid duplicate optimizer parameters.
        self._fc_ref = [feature_complex]

        if L > 0:
            self.pe = PositionalEncoding(3, L)
            d_in = self.pe.d_out
        else:
            self.pe = None
            d_in = 3

        self.encoder = SkipMLP(d_in, d_features, W, D, beta=beta)

    @property
    def feature_complex(self) -> FeatureComplex:
        return self._fc_ref[0]

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        """Encode 3D points into feature space."""
        x = self.pe(xyz) if self.pe is not None else xyz
        return self.encoder(x)

    def de_interpolate(self, patch_idx, z_pred: torch.Tensor) -> torch.Tensor:
        """
        Recover UV coordinates from predicted features (bilinear inverse).

        DEPRECATED under MVC — see class docstring.
        """
        B = z_pred.shape[0]
        if not torch.is_tensor(patch_idx):
            patch_idx = torch.full((B,), int(patch_idx),
                                   dtype=torch.long, device=z_pred.device)

        fc = self.feature_complex
        row = patch_idx // fc.n_cols
        col = patch_idx % fc.n_cols
        i00, i01, i10, i11 = fc._corner_indices(row, col)

        vf = fc.vertex_features
        z00 = vf[i00]
        z01 = vf[i01]
        z10 = vf[i10]
        z11 = vf[i11]

        A = z01 - z00
        Bd = z10 - z00
        C = z11 - z10 - z01 + z00
        R = z_pred - z00

        # Per-sample basis matrix M = [B | A].
        M_mat = torch.stack([Bd, A], dim=2)
        Mt = M_mat.transpose(1, 2)
        MtM = Mt @ M_mat
        reg = 1e-5 * torch.eye(2, device=MtM.device, dtype=MtM.dtype)
        MtM = MtM + reg

        # Initial solve without the bilinear cross term.
        rhs = Mt @ R.unsqueeze(-1)
        params = torch.linalg.solve(MtM, rhs)
        u_est = params[:, 0]
        v_est = params[:, 1]

        # One refinement step with the estimated cross term.
        R_ref = R - (u_est * v_est) * C
        rhs2 = Mt @ R_ref.unsqueeze(-1)
        params2 = torch.linalg.solve(MtM, rhs2)
        u = params2[:, 0]
        v = params2[:, 1]

        return torch.cat([u, v], dim=1)

    def forward(self, patch_idx, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_idx: Patch index or batch of patch indices.
            xyz: 3D points.
        Returns:
            UV coordinates.
        """
        z_pred = self.encode(xyz)
        return self.de_interpolate(patch_idx, z_pred)