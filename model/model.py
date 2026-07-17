#!/usr/bin/env python3
# model.py

import math

import numpy as np
import torch
import torch.nn as nn


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
    """
    Grid-based feature complex with shared vertex features.

    Adjacent patches share corner features, which enforces C0 continuity.
    """
    def __init__(self, n_rows: int, n_cols: int, d_features: int = 64):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.d_features = d_features

        n_vertices = (n_rows + 1) * (n_cols + 1)
        self.vertex_features = nn.Parameter(
            torch.randn(n_vertices, d_features) * 0.1
        )

    def _corner_indices(self, row, col):
        """
        Return vectorized indices of the four patch corners.

        Args:
            row, col: Patch grid positions.
        Returns:
            Tuple `(i00, i01, i10, i11)`.
        """
        ncp = self.n_cols + 1
        i00 = row * ncp + col
        i01 = row * ncp + (col + 1)
        i10 = (row + 1) * ncp + col
        i11 = (row + 1) * ncp + (col + 1)
        return i00, i01, i10, i11

    def interpolate(self, row, col, uv):
        """
        Bilinearly interpolate vertex features.

        Args:
            row, col: Patch grid positions per sample.
            uv: UV coordinates in `[0, 1]`.
        Returns:
            Interpolated feature tensor.
        """
        i00, i01, i10, i11 = self._corner_indices(row, col)
        vf = self.vertex_features
        z00 = vf[i00]
        z01 = vf[i01]
        z10 = vf[i10]
        z11 = vf[i11]

        u = uv[:, 0:1]
        v = uv[:, 1:2]

        features = ((1 - u) * (1 - v) * z00 +
                    (1 - u) * v * z01 +
                    u * (1 - v) * z10 +
                    u * v * z11)
        return features


class MultiPatchForwardMap(nn.Module):
    """
        Vectorized multi-patch forward map.

        Shared features and global UV coordinates keep neighboring patches
        continuous across boundaries.
    """
    def __init__(self, n_rows: int, n_cols: int, d_features: int = 64,
                 L: int = 8, W: int = 256, D: int = 6, beta: float = 5.0):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_patches = n_rows * n_cols
        self.d_features = d_features
        self.L = L

        self.complex = FeatureComplex(n_rows, n_cols, d_features)

        if L > 0:
            self.pe = PositionalEncoding(2, L)
            d_pe = self.pe.d_out
        else:
            self.pe = None
            d_pe = 0

        self.decoder = SkipMLP(d_features + d_pe, 3, W, D, beta=beta)

        # Optional flat-plane initialization.
        # nn.init.zeros_(self.decoder.head.weight)
        # nn.init.zeros_(self.decoder.head.bias)

    def patch_idx_to_rowcol(self, patch_idx):
        """Convert patch indices to row and column indices."""
        row = patch_idx // self.n_cols
        col = patch_idx % self.n_cols
        return row, col

    def forward(self, patch_idx, uv: torch.Tensor):
        """
        Args:
            patch_idx: Patch index or batch of patch indices.
            uv: Local UV coordinates.
        Returns:
            Predicted 3D points.
        """
        B = uv.shape[0]
        if not torch.is_tensor(patch_idx):
            patch_idx = torch.full((B,), int(patch_idx),
                                   dtype=torch.long, device=uv.device)

        row = patch_idx // self.n_cols
        col = patch_idx % self.n_cols

        features = self.complex.interpolate(row, col, uv)

        u = uv[:, 0:1]
        v = uv[:, 1:2]
        # Convert local UV to global UV over the full patch grid.
        global_u = (row.unsqueeze(1).float() + u) / self.n_rows
        global_v = (col.unsqueeze(1).float() + v) / self.n_cols

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

    The model encodes points into feature space, then recovers UV values by
    de-interpolating the shared corner features.
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
        Recover UV coordinates from predicted features.

        Args:
            patch_idx: Patch index or batch of patch indices.
            z_pred: Predicted feature vectors.
        Returns:
            UV coordinates.
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