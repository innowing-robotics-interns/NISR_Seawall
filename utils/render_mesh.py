#!/usr/bin/env python3
"""Differentiable patch-grid mesh sampling for the NGF rendering loss.

The export path in :mod:`utils.utils` is intentionally detached: it evaluates
each patch, copies to CPU, and calls ``.numpy()`` for writing a PLY.  A
rendering loss needs the opposite properties -- one GPU tensor with a live
autograd graph, an indexed triangle mesh, and vertex normals.

This module adds that path.  It does not replace the exporter.

Typical use inside a training step::

    layout = get_patch_mesh_layout(active_ids, resolution, device)
    vertices, faces = sample_patch_mesh(F, resolution, patch_ids=active_ids)
    normals = vertex_normals(vertices, faces)

Face winding matches NISR's ``n = t_u x t_v`` convention (``model/losses.py``):
for a grid with ``u`` along the first axis and ``v`` along the second, each
cell is split as ``[00, 10, 11]`` and ``[00, 11, 01]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch


__all__ = [
    'PatchMeshLayout',
    'auto_mesh_resolution',
    'get_patch_mesh_layout',
    'sample_patch_mesh',
    'uv_grid',
    'vertex_normals',
]


# Layouts are small (a UV grid plus faces) and reused for the whole run.
_LAYOUT_CACHE = {}
_LAYOUT_CACHE_LIMIT = 32

# Fixed per-patch UV grids used by deterministic Chamfer sampling.
_UV_GRID_CACHE = {}


@dataclass(frozen=True)
class PatchMeshLayout:
    """Cached UV grid and triangle topology for a set of patches."""

    resolution: int
    patch_ids: Tuple[int, ...]
    uv: torch.Tensor                    # (R * R, 2)
    faces: torch.Tensor                 # (P * 2 * (R - 1)^2, 3) int32
    vertex_offsets: torch.Tensor        # (P,) int64
    device: torch.device

    @property
    def num_patches(self) -> int:
        return len(self.patch_ids)

    @property
    def vertices_per_patch(self) -> int:
        return self.resolution * self.resolution

    @property
    def num_vertices(self) -> int:
        return self.num_patches * self.vertices_per_patch

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]


def _normalize_patch_ids(patch_ids: Optional[Iterable[int]],
                         n_patches: int) -> Tuple[int, ...]:
    if patch_ids is None:
        return tuple(range(int(n_patches)))

    ids = tuple(int(patch_id) for patch_id in patch_ids)
    if not ids:
        raise ValueError('patch_ids is empty; at least one patch is required')

    for patch_id in ids:
        if patch_id < 0 or patch_id >= int(n_patches):
            raise ValueError(
                f'patch id {patch_id} is outside [0, {int(n_patches) - 1}]')

    return ids


def _device_cache_key(device: torch.device):
    if device.type == 'cuda':
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return ('cuda', int(index))
    return (device.type, device.index)


def _build_layout(patch_ids: Tuple[int, ...],
                  resolution: int,
                  device: torch.device,
                  dtype: torch.dtype) -> PatchMeshLayout:
    if resolution < 2:
        raise ValueError(f'resolution must be >= 2, got {resolution}')

    # Same ordering as utils.sample_multi_patch_grid: first axis is u, second
    # is v, and the flattened index is i * resolution + j.
    u = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    grid_u, grid_v = torch.meshgrid(u, u, indexing='ij')
    uv = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=-1)
    uv = uv.contiguous()

    cells = torch.arange(resolution * resolution,
                         device=device, dtype=torch.long).reshape(
                             resolution, resolution)
    i00 = cells[:-1, :-1].reshape(-1)
    i10 = cells[1:, :-1].reshape(-1)
    i01 = cells[:-1, 1:].reshape(-1)
    i11 = cells[1:, 1:].reshape(-1)

    # Triangulate each cell along the 00->11 diagonal.  The two triangles have
    # the same winding, so their geometric normals agree with t_u x t_v.
    triangle_a = torch.stack([i00, i10, i11], dim=-1)
    triangle_b = torch.stack([i00, i11, i01], dim=-1)
    patch_faces = torch.stack([triangle_a, triangle_b], dim=1).reshape(-1, 3)

    vertex_offsets = (torch.arange(len(patch_ids), device=device,
                                   dtype=torch.long) * (resolution * resolution))
    faces = (patch_faces.unsqueeze(0) + vertex_offsets.view(-1, 1, 1))
    faces = faces.reshape(-1, 3).to(torch.int32).contiguous()

    return PatchMeshLayout(
        resolution=int(resolution),
        patch_ids=patch_ids,
        uv=uv,
        faces=faces,
        vertex_offsets=vertex_offsets,
        device=device,
    )


def uv_grid(resolution: int, device, dtype: torch.dtype = torch.float32):
    """Cached ``(R*R, 2)`` grid of UV coordinates over one patch square.

    Ordering matches :func:`sample_patch_mesh` (u-major, both endpoints
    included), so a caller can expand it over ``P`` patches with
    ``grid.repeat(P, 1)`` and address patch ``p`` at ``p * R*R + i``.  The
    tensor is read-only and shared; do not modify it in place.
    """
    if resolution < 2:
        raise ValueError(f'resolution must be >= 2, got {resolution}')

    device = torch.device(device)
    key = (int(resolution), _device_cache_key(device), str(dtype))
    grid = _UV_GRID_CACHE.get(key)
    if grid is None:
        u = torch.linspace(0.0, 1.0, int(resolution), device=device,
                           dtype=dtype)
        grid_u, grid_v = torch.meshgrid(u, u, indexing='ij')
        grid = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)],
                           dim=-1).contiguous()
        _UV_GRID_CACHE[key] = grid
    return grid


def get_patch_mesh_layout(patch_ids: Iterable[int],
                          resolution: int,
                          device,
                          dtype: torch.dtype = torch.float32) -> PatchMeshLayout:
    """Return a cached layout for ``patch_ids`` at ``resolution``.

    The cache key includes the device and dtype.  Build the layout once per
    training run, not per optimization step.
    """
    device = torch.device(device)
    ids = tuple(int(patch_id) for patch_id in patch_ids)
    key = (ids, int(resolution), _device_cache_key(device), str(dtype))

    layout = _LAYOUT_CACHE.get(key)
    if layout is not None:
        return layout

    if len(_LAYOUT_CACHE) >= _LAYOUT_CACHE_LIMIT:
        # Layouts in active use stay alive through the caller's reference.
        _LAYOUT_CACHE.clear()

    layout = _build_layout(ids, int(resolution), device, dtype)
    _LAYOUT_CACHE[key] = layout
    return layout


def auto_mesh_resolution(num_patches: int,
                         max_vertices: int = 30_000,
                         min_resolution: int = 8,
                         max_resolution: int = 32) -> int:
    """Pick a per-patch grid resolution under a vertex budget.

    The renderer cost is roughly linear in the number of faces, while the
    supervision detail is set by ``resolution``.  For the small patch grids
    used by NISR the cap is usually reached before the budget.
    """
    num_patches = max(1, int(num_patches))
    budget_resolution = int(math.sqrt(max_vertices / num_patches))
    return max(int(min_resolution),
               min(int(max_resolution), budget_resolution))


def _model_device_dtype(F) -> Tuple[torch.device, torch.dtype]:
    try:
        reference = next(F.parameters())
        return reference.device, reference.dtype
    except StopIteration:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, torch.float32


def sample_patch_mesh(F,
                      resolution: int,
                      patch_ids: Optional[Iterable[int]] = None,
                      chunk_size: int = 4096):
    """Evaluate a fixed-grid forward map on a triangle mesh.

    Args:
        F: module with ``forward(patch_idx, uv)`` and an ``n_patches``
            attribute, typically ``MultiPatchForwardMap``.
        resolution: vertices per patch edge.
        patch_ids: patches to include.  ``None`` means all patches.  Pass the
            training loop's ``active_ids`` so inactive patches receive no
            render gradient.
        chunk_size: points per forward call.  Chunks may span patch
            boundaries; the forward map handles mixed patch indices.

    Returns:
        ``(vertices, faces)`` with shapes ``(P*R^2, 3)`` and
        ``(P*2*(R-1)^2, 3)``.  ``faces`` is int32 on the vertices' device.
        Neither tensor is detached; gradients flow to every parameter used by
        ``F``.
    """
    if chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size}')

    device, dtype = _model_device_dtype(F)
    ids = _normalize_patch_ids(patch_ids, F.n_patches)
    layout = get_patch_mesh_layout(ids, resolution, device, dtype)

    # Evaluate every patch in as few forward calls as the point budget allows.
    # Calling F once per leaf serializes hundreds of tiny launches and forces a
    # GPU sync inside feature interpolation.
    patch_index = torch.as_tensor(ids, device=device, dtype=torch.long)
    patch_index = patch_index.repeat_interleave(layout.uv.shape[0])
    uv = layout.uv.repeat(len(ids), 1)

    vertices = []
    for start in range(0, uv.shape[0], chunk_size):
        vertices.append(F(patch_index[start:start + chunk_size],
                          uv[start:start + chunk_size]))

    if not vertices:
        raise RuntimeError('forward map produced no vertices')

    return torch.cat(vertices, dim=0), layout.faces


def vertex_normals(vertices: torch.Tensor,
                   faces: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """Area-weighted vertex normals from a triangle mesh.

    The face normal is ``cross(v1 - v0, v2 - v0)``, whose length is twice the
    triangle area.  Accumulating those vectors and normalizing therefore
    weights each face by its area.  The operation is differentiable and costs
    one pass over the faces.

    At duplicated patch seams each copy sees only its own side.  The positions
    coincide and the tangent planes are continuous, so the one-sided normals
    agree up to discretization error.
    """
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(
            f'vertices must be (N, 3), got {tuple(vertices.shape)}')
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f'faces must be (F, 3), got {tuple(faces.shape)}')
    if faces.device != vertices.device:
        raise ValueError(
            f'faces are on {faces.device} but vertices are on {vertices.device}')

    if faces.dtype != torch.long:
        faces = faces.long()

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    normals = torch.zeros_like(vertices)
    normals = normals.index_add(0, faces[:, 0], face_normals)
    normals = normals.index_add(0, faces[:, 1], face_normals)
    normals = normals.index_add(0, faces[:, 2], face_normals)

    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(eps)
