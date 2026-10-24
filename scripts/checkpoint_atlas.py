#!/usr/bin/env python3
"""Rebuild the forward map stored in a checkpoint.

Fixed-grid checkpoints (single/two/six-sheet) are rebuilt from their stored
args.  Adaptive quadtree checkpoints (``mode='adaptive_cube'``) are rebuilt
from their stored topology and expose one patch per leaf, so the inspection
scripts can render them without knowing the atlas family in advance.
"""

from __future__ import annotations

import torch

import main as nisr_main


def is_adaptive(ck) -> bool:
    return ck.get('mode') == 'adaptive_cube'


def is_adaptive_forward(F) -> bool:
    return getattr(getattr(F, 'complex', None), 'leaf_patches', None) is not None


def build_forward_model(ck, device='cpu'):
    """Create and load the checkpoint's forward map; return ``(F, atlas_info)``."""
    if is_adaptive(ck):
        from model.model import AdaptiveCubeForwardMap
        F = AdaptiveCubeForwardMap.from_checkpoint(ck, device=device)
        stats = F.complex.stats()
        info = {
            'atlas_mode': 'adaptive_cube',
            'n_sides': 6,
            'actual_n_patches': int(F.n_patches),
            'patches_per_side': None,
            'adaptive': True,
            'leaf_depths': stats['depth_hist'],
        }
        return F, info

    a = ck['args']
    atlas_mode = a.get('atlas_mode', 'single_sheet')
    F, atlas_info = nisr_main._build_forward_model(
        atlas_mode,
        int(a['n_patches']),
        int(a['d_features']),
        L=int(a['L']),
        W=int(a['W']),
        D=int(a['D']),
        beta=float(a['beta']),
        device=device,
        two_sheet_side_rows=int(a.get('two_sheet_side_rows', 2)),
        two_sheet_side_cols=int(a.get('two_sheet_side_cols', 2)),
        six_sheet_face_rows=int(a.get('six_sheet_face_rows', 2)),
        six_sheet_face_cols=int(a.get('six_sheet_face_cols', 2)),
    )
    F.load_state_dict(ck['F_state'])
    F.eval()
    return F, atlas_info


def active_patch_ids(ck, F=None):
    """Patch ids the render loss should use.

    Adaptive atlases have no active-patch mask: every leaf is fitted.  Pass
    ``F`` for those checkpoints so the leaf count comes from the topology.
    """
    if is_adaptive(ck):
        if F is None:
            raise ValueError('adaptive checkpoints need F to enumerate leaves')
        return list(range(int(F.n_patches)))
    return [int(patch_id) for patch_id in ck['active_patch_ids']]


def patch_face_row_col(F, patch_id):
    """Return ``(face, row, col)`` for a flat patch id (face is 0 for a sheet)."""
    leaves = getattr(getattr(F, 'complex', None), 'leaf_patches', None)
    if leaves is not None:
        leaf = leaves[int(patch_id)]
        return int(leaf.face), int(leaf.depth), 0
    if hasattr(F, 'patch_idx_to_face_rowcol'):
        face, row, col = F.patch_idx_to_face_rowcol(int(patch_id))
        return int(face), int(row), int(col)
    return 0, int(patch_id) // F.n_cols, int(patch_id) % F.n_cols


def patch_layout_info(F, patch_id) -> dict:
    """Face/depth for adaptive leaves, face/row/col for fixed grids."""
    leaves = getattr(getattr(F, 'complex', None), 'leaf_patches', None)
    if leaves is not None:
        leaf = leaves[int(patch_id)]
        return {'face': int(leaf.face), 'row': None, 'col': None,
                'depth': int(leaf.depth)}
    face, row, col = patch_face_row_col(F, patch_id)
    return {'face': face, 'row': row, 'col': col, 'depth': None}


def num_faces(F):
    if is_adaptive_forward(F):
        return 6
    return int(getattr(F, 'n_faces', getattr(F, 'n_sides', 1)))
