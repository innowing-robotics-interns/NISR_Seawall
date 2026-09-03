#!/usr/bin/env python3
# model/adaptive_model.py
"""Forward map on the adaptive cube complex: F(leaf_id, uv) -> xyz."""

import torch
import torch.nn as nn

from .model import PositionalEncoding, SkipMLP
from .adaptive_complex import AdaptiveCubeComplex


class AdaptiveCubeForwardMap(nn.Module):
    """
    features = MVC(vertex features over the leaf's polygon)
    xyz      = decoder(features [+ PE(cube_xyz)])

    PE (L > 0) is applied to the REFERENCE-CUBE position of the query, which
    is continuous across faces and subdivision levels, so it never breaks
    seam continuity. L=0 keeps the map purely feature-driven.
    """

    def __init__(self, d_features: int = 88, base_subdivisions: int = 0,
                 L: int = 0, W: int = 512, D: int = 6, beta: float = 100.0):
        super().__init__()
        self.d_features, self.L, self.W, self.D, self.beta = d_features, L, W, D, beta
        self.base_subdivisions = base_subdivisions
        self.complex = AdaptiveCubeComplex(d_features, base_subdivisions)

        if L > 0:
            self.pe = PositionalEncoding(3, L)
            d_pe = self.pe.d_out
        else:
            self.pe = None
            d_pe = 0
        self.decoder = SkipMLP(d_features + d_pe, 3, W, D, beta=beta)

    @property
    def n_patches(self):
        return self.complex.n_leaves

    def _pids(self, patch_idx, uv):
        if not torch.is_tensor(patch_idx):
            return torch.full((uv.shape[0],), int(patch_idx),
                              dtype=torch.long, device=uv.device)
        return patch_idx.to(device=uv.device, dtype=torch.long)

    def forward(self, patch_idx, uv: torch.Tensor):
        pids = self._pids(patch_idx, uv)
        feats = self.complex.interpolate(pids, uv)
        if self.pe is not None:
            dec_in = torch.cat([feats, self.pe(self.complex.cube_xyz(pids, uv))], dim=1)
        else:
            dec_in = feats
        return self.decoder(dec_in)

    def cube_xyz(self, patch_idx, uv):
        return self.complex.cube_xyz(self._pids(patch_idx, uv), uv)

    # ── checkpointing ───────────────────────────────────────────────────
    def config(self):
        return {'d_features': self.d_features, 'L': self.L, 'W': self.W,
                'D': self.D, 'beta': self.beta,
                'base_subdivisions': self.base_subdivisions}

    def checkpoint_payload(self, extra: dict = None):
        payload = {
            'mode': 'adaptive_cube',
            'F_state': self.state_dict(),
            'config': self.config(),
            'topology': self.complex.serialize(),
        }
        if extra:
            payload.update(extra)
        return payload

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str = 'cpu'):
        cfg = ckpt['config']
        model = cls(d_features=cfg['d_features'], base_subdivisions=0,
                    L=cfg['L'], W=cfg['W'], D=cfg['D'], beta=cfg['beta'])
        model.complex.replay(ckpt['topology'])      # grows features to size
        model.load_state_dict(ckpt['F_state'])
        model.to(device)
        model.complex.rebuild_flat()
        model.eval()
        return model