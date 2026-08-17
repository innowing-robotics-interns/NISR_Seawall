# pc_presegmentation.py

"""
Pre-segmentation utilities for point clouds.

Each method returns patch assignments, grid topology, and per-point
parameter coordinates.
"""

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import KDTree
import open3d as o3d

import json
import math
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

def poisson_spectral_segmentation(points, normals, n_patches_u=4, n_patches_v=4,
                                   poisson_depth=8, trim_quantile=0.1,
                                   export_mesh_path=None):
    """
    Segment points with Poisson reconstruction and spectral parameterization.

    Args:
        points: Point positions.
        normals: Point normals.
        n_patches_u: Patch count along the first parameter axis.
        n_patches_v: Patch count along the second parameter axis.
        poisson_depth: Octree depth for Poisson reconstruction.
        trim_quantile: Fraction of low-density vertices to remove.
        export_mesh_path: Optional output path for the reconstructed mesh.

    Returns:
        Tuple `(patch_assignments, grid_topology, patch_params)`.
    """
    # Step 1: Poisson surface reconstruction.
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(points)

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # Extract normals as numpy array
    normals = np.asarray(pcd.normals)  # shape (N, 3)

    print(f"normals shape: {normals.shape}")  # should print (N, 3)

    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, linear_fit=False
    )

    # Trim low-density regions to reduce reconstruction artifacts.
    densities = np.asarray(densities)
    if trim_quantile > 0:
        threshold = np.quantile(densities, trim_quantile)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()

    if export_mesh_path is not None:
        export_dir = os.path.dirname(export_mesh_path)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        success = o3d.io.write_triangle_mesh(export_mesh_path, mesh)
        if success:
            print(f"[Poisson] Exported reconstructed mesh to: {export_mesh_path}")
        else:
            print(f"[Poisson] Warning: failed to export reconstructed mesh to: {export_mesh_path}")

    # Spectral parameterization on the reconstructed mesh.
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n_verts = len(vertices)

    print(f"[Poisson] Reconstructed mesh: {n_verts} vertices, {len(triangles)} triangles")

    # Build a cotangent Laplacian for smoother parameterization.
    L, M = _build_cotangent_laplacian(vertices, triangles)

    # Solve the generalized eigenvalue problem and keep the first non-trivial modes.
    n_eigvecs = min(10, n_verts - 2)  # compute a few extra for robustness
    try:
        eigenvalues, eigenvectors = eigsh(L, k=n_eigvecs, M=M, which='SM', sigma=0)
    except Exception:
        # Fall back to a uniform Laplacian if the cotangent version fails.
        print("[Poisson] Cotangent Laplacian failed, falling back to uniform Laplacian")
        L_uniform = _build_uniform_laplacian(vertices, triangles)
        eigenvalues, eigenvectors = eigsh(L_uniform, k=n_eigvecs, which='SM')

    # Sort by eigenvalue and skip the trivial constant mode.
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Find the first non-trivial eigenvector.
    nontrivial_start = 0
    for i, ev in enumerate(eigenvalues):
        if ev > 1e-7:
            nontrivial_start = i
            break

    u_mesh = eigenvectors[:, nontrivial_start]
    v_mesh = eigenvectors[:, nontrivial_start + 1]

    # Normalize to [0, 1].
    u_mesh = _normalize_01(u_mesh)
    v_mesh = _normalize_01(v_mesh)

    # Step 3: grid partition.
    u_bins_mesh = np.clip((u_mesh * n_patches_u).astype(int), 0, n_patches_u - 1)
    v_bins_mesh = np.clip((v_mesh * n_patches_v).astype(int), 0, n_patches_v - 1)
    mesh_patch_ids = u_bins_mesh * n_patches_v + v_bins_mesh

    # Step 4: transfer assignments back to the original points.
    mesh_tree = KDTree(vertices)
    _, nearest_verts = mesh_tree.query(points)

    patch_assignments = mesh_patch_ids[nearest_verts]
    patch_params = np.stack([u_mesh[nearest_verts], v_mesh[nearest_verts]], axis=1)

    grid_topology = np.arange(n_patches_u * n_patches_v).reshape(n_patches_u, n_patches_v)

    # Report empty patches if any appear.
    patch_assignments, grid_topology = _handle_empty_patches(
        patch_assignments, grid_topology, patch_params, n_patches_u, n_patches_v
    )

    return patch_assignments, grid_topology, patch_params

def spectral_direct_segmentation(points, n_patches_u=4, n_patches_v=4,
                                  k_neighbors=30, subsample_for_eigen=5000):
    """
    Segment points directly with a spectral embedding of the point cloud.

    Args:
        points: Point positions.
        n_patches_u: Patch count along the first parameter axis.
        n_patches_v: Patch count along the second parameter axis.
        k_neighbors: Neighbor count for the K-NN graph.
        subsample_for_eigen: Subsample size used for eigen computation.

    Returns:
        Tuple `(patch_assignments, grid_topology, patch_params)`.
    """
    N = len(points)

    # Subsample large point clouds before eigen decomposition.
    if N > subsample_for_eigen:
        print(f"[Spectral] Subsampling {subsample_for_eigen}/{N} points for eigen computation")
        subsample_idx = _farthest_point_sampling(points, subsample_for_eigen)
        points_sub = points[subsample_idx]
    else:
        subsample_idx = None
        points_sub = points

    N_sub = len(points_sub)

    # 1. Build the K-NN graph.
    tree = KDTree(points_sub)
    distances, indices = tree.query(points_sub, k=k_neighbors + 1)

    # Gaussian kernel bandwidth.
    sigma = np.median(distances[:, 1:].flatten())

    # Build the sparse adjacency matrix.
    rows = []
    cols = []
    weights = []
    for i in range(N_sub):
        for j_idx in range(1, k_neighbors + 1):
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            w = np.exp(-d**2 / (2 * sigma**2))
            rows.append(i)
            cols.append(j)
            weights.append(w)

    adj = csr_matrix((weights, (rows, cols)), shape=(N_sub, N_sub))
    adj = (adj + adj.T) / 2  # symmetrize

    # Graph Laplacian.
    degree = np.array(adj.sum(axis=1)).flatten()
    D = diags(degree)
    L = D - adj

    # Use the normalized random-walk Laplacian for non-uniform sampling.
    D_inv = diags(1.0 / (degree + 1e-10))
    L_rw = D_inv @ L

    # Smallest non-trivial eigenvectors.
    n_eigvecs = min(10, N_sub - 2)
    eigenvalues, eigenvectors = eigsh(L_rw, k=n_eigvecs, which='SM')

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip trivial eigenvector
    nontrivial_start = 0
    for i, ev in enumerate(eigenvalues):
        if ev > 1e-7:
            nontrivial_start = i
            break

    u_sub = eigenvectors[:, nontrivial_start]
    v_sub = eigenvectors[:, nontrivial_start + 1]

    # If we subsampled, interpolate back to full point cloud
    if subsample_idx is not None:
        full_tree = KDTree(points_sub)
        dists_full, idx_full = full_tree.query(points, k=min(5, N_sub))

        # Inverse distance weighted interpolation
        weights_full = 1.0 / (dists_full + 1e-10)
        weights_full = weights_full / weights_full.sum(axis=1, keepdims=True)

        u_full = (weights_full * u_sub[idx_full]).sum(axis=1)
        v_full = (weights_full * v_sub[idx_full]).sum(axis=1)
    else:
        u_full = u_sub
        v_full = v_sub

    # Normalize to [0, 1]
    u_full = _normalize_01(u_full)
    v_full = _normalize_01(v_full)

    # Grid partition
    u_bins = np.clip((u_full * n_patches_u).astype(int), 0, n_patches_u - 1)
    v_bins = np.clip((v_full * n_patches_v).astype(int), 0, n_patches_v - 1)
    patch_assignments = u_bins * n_patches_v + v_bins
    patch_params = np.stack([u_full, v_full], axis=1)

    grid_topology = np.arange(n_patches_u * n_patches_v).reshape(n_patches_u, n_patches_v)

    patch_assignments, grid_topology = _handle_empty_patches(
        patch_assignments, grid_topology, patch_params, n_patches_u, n_patches_v
    )

    return patch_assignments, grid_topology, patch_params

def pca_grid_segmentation(points, n_patches_u=4, n_patches_v=4):
    """
    Simple PCA projection + uniform grid. Only works for open, non-folding surfaces.

    Args:
        points: (N, 3)
        n_patches_u, n_patches_v: grid dimensions

    Returns:
        patch_assignments, grid_topology, patch_params
    """
    # PCA
    centroid = points.mean(axis=0)
    pts_centered = points - centroid
    cov = pts_centered.T @ pts_centered / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Project onto two largest principal components
    idx = np.argsort(eigenvalues)[::-1]
    pc1 = eigenvectors[:, idx[0]]
    pc2 = eigenvectors[:, idx[1]]

    u = pts_centered @ pc1
    v = pts_centered @ pc2

    u = _normalize_01(u)
    v = _normalize_01(v)

    u_bins = np.clip((u * n_patches_u).astype(int), 0, n_patches_u - 1)
    v_bins = np.clip((v * n_patches_v).astype(int), 0, n_patches_v - 1)
    patch_assignments = u_bins * n_patches_v + v_bins
    patch_params = np.stack([u, v], axis=1)

    grid_topology = np.arange(n_patches_u * n_patches_v).reshape(n_patches_u, n_patches_v)

    return patch_assignments, grid_topology, patch_params

def axis_aligned_grid_segmentation(points, n_patches_u=4, n_patches_v=4,
                                   axes=(0, 1)):
    """
    Naive segmentation by projecting points onto two coordinate axes and
    dividing that 2D bounding box into a uniform rectangular grid.

    This is the simplest possible segmentation baseline. It does not try to
    respect curvature, topology, or intrinsic geometry.

    Args:
        points: (N, 3) point positions
        n_patches_u: number of bins along the first selected axis
        n_patches_v: number of bins along the second selected axis
        axes: tuple of two coordinate indices, e.g. (0, 1), (0, 2), or (1, 2)

    Returns:
        patch_assignments, grid_topology, patch_params
    """
    if len(axes) != 2:
        raise ValueError(f"axes must contain exactly two entries, got {axes}")
    if axes[0] == axes[1]:
        raise ValueError(f"axes must be different, got {axes}")
    if any(ax not in (0, 1, 2) for ax in axes):
        raise ValueError(f"axes must be chosen from (0, 1, 2), got {axes}")

    u = points[:, axes[0]]
    v = points[:, axes[1]]

    u = _normalize_01(u)
    v = _normalize_01(v)

    u_bins = np.clip((u * n_patches_u).astype(int), 0, n_patches_u - 1)
    v_bins = np.clip((v * n_patches_v).astype(int), 0, n_patches_v - 1)
    patch_assignments = u_bins * n_patches_v + v_bins
    patch_params = np.stack([u, v], axis=1)

    grid_topology = np.arange(n_patches_u * n_patches_v).reshape(n_patches_u, n_patches_v)

    patch_assignments, grid_topology = _handle_empty_patches(
        patch_assignments, grid_topology, patch_params, n_patches_u, n_patches_v
    )

    return patch_assignments, grid_topology, patch_params


def two_sheet_axis_aligned_segmentation(points,
                                        n_patches_u=2,
                                        n_patches_v=2,
                                        split_axis=2,
                                        side_axes=(0, 1),
                                        split_value=None):
    """
    Segment a point cloud into two axis-aligned sheets, each with its own grid.

    The cloud is first split into two sides using one coordinate axis, then each
    side is segmented independently with axis-aligned 2D binning. Patch ids are
    flattened globally across both sides.

    Args:
        points: (N, 3) point positions.
        n_patches_u: grid rows per side.
        n_patches_v: grid cols per side.
        split_axis: coordinate axis used to split the cloud into two sides.
        side_axes: coordinate axes used for per-side 2D axis-aligned segmentation.
        split_value: optional threshold on `split_axis`. If None, uses the median.

    Returns:
        assignments: (N,) global patch ids in [0, 2*n_patches_u*n_patches_v).
        grid_topology: (2, n_patches_u, n_patches_v) patch ids per side.
        patch_params: (N, 2) local per-side normalized UV coordinates.
        side_assignments: (N,) side id in {0, 1}.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if split_axis not in (0, 1, 2):
        raise ValueError(f"split_axis must be one of (0, 1, 2), got {split_axis}")
    if len(side_axes) != 2:
        raise ValueError(f"side_axes must contain exactly two entries, got {side_axes}")
    if side_axes[0] == side_axes[1]:
        raise ValueError(f"side_axes must be different, got {side_axes}")
    if any(ax not in (0, 1, 2) for ax in side_axes):
        raise ValueError(f"side_axes must be chosen from (0, 1, 2), got {side_axes}")

    if split_value is None:
        split_value = float(np.median(points[:, split_axis]))

    side_assignments = np.zeros(points.shape[0], dtype=np.int32)
    side_assignments[points[:, split_axis] < split_value] = 1

    patches_per_side = n_patches_u * n_patches_v
    assignments = np.full(points.shape[0], -1, dtype=np.int32)
    patch_params = np.zeros((points.shape[0], 2), dtype=np.float32)
    grid_topology = np.stack([
        np.arange(patches_per_side, dtype=np.int32).reshape(n_patches_u, n_patches_v),
        (np.arange(patches_per_side, dtype=np.int32) + patches_per_side).reshape(n_patches_u, n_patches_v),
    ], axis=0)

    for side in range(2):
        mask = side_assignments == side
        pts_side = points[mask]
        if pts_side.shape[0] == 0:
            continue

        u = _normalize_01(pts_side[:, side_axes[0]])
        v = _normalize_01(pts_side[:, side_axes[1]])

        u_bins = np.clip((u * n_patches_u).astype(int), 0, n_patches_u - 1)
        v_bins = np.clip((v * n_patches_v).astype(int), 0, n_patches_v - 1)
        local_patch_ids = u_bins * n_patches_v + v_bins
        global_patch_ids = local_patch_ids + side * patches_per_side

        assignments[mask] = global_patch_ids.astype(np.int32)
        patch_params[mask] = np.stack([u, v], axis=1).astype(np.float32)

    return assignments, grid_topology, patch_params, side_assignments


def _infer_box_face_assignments(points):
    """Infer cube-face ids from the dominant absolute coordinate per point."""
    points = np.asarray(points)
    abs_points = np.abs(points)
    dominant_axis = np.argmax(abs_points, axis=1)

    face_assignments = np.empty(points.shape[0], dtype=np.int32)

    x_mask = dominant_axis == 0
    y_mask = dominant_axis == 1
    z_mask = dominant_axis == 2

    face_assignments[x_mask] = np.where(points[x_mask, 0] >= 0.0, 0, 1)
    face_assignments[y_mask] = np.where(points[y_mask, 1] >= 0.0, 2, 3)
    face_assignments[z_mask] = np.where(points[z_mask, 2] >= 0.0, 4, 5)

    return face_assignments


def _box_face_local_uv(points, face_assignments):
    """Compute face-local UV coordinates consistent with cube atlas embeddings."""
    points = np.asarray(points)
    face_assignments = np.asarray(face_assignments)

    patch_params = np.zeros((points.shape[0], 2), dtype=np.float32)

    for face_id in range(6):
        mask = face_assignments == face_id
        if not np.any(mask):
            continue

        pts_face = points[mask]
        x = pts_face[:, 0]
        y = pts_face[:, 1]
        z = pts_face[:, 2]

        if face_id == 0:      # +X
            u = 0.5 * (z + 1.0)
            v = 0.5 * (1.0 - y)
        elif face_id == 1:    # -X
            u = 0.5 * (z + 1.0)
            v = 0.5 * (y + 1.0)
        elif face_id == 2:    # +Y
            u = 0.5 * (x + 1.0)
            v = 0.5 * (z + 1.0)
        elif face_id == 3:    # -Y
            u = 0.5 * (x + 1.0)
            v = 0.5 * (1.0 - z)
        elif face_id == 4:    # +Z
            u = 0.5 * (x + 1.0)
            v = 0.5 * (1.0 - y)
        elif face_id == 5:    # -Z
            u = 0.5 * (x + 1.0)
            v = 0.5 * (y + 1.0)
        else:
            raise ValueError(f"Unknown face_id: {face_id}")

        patch_params[mask, 0] = np.clip(u, 0.0, 1.0).astype(np.float32)
        patch_params[mask, 1] = np.clip(v, 0.0, 1.0).astype(np.float32)

    return patch_params


def six_sheet_axis_aligned_segmentation(points,
                                        n_patches_u=2,
                                        n_patches_v=2,
                                        face_assignments=None,
                                        assignment_mode='argmax_abs'):
    """
    Segment a point cloud into six cube-like atlas faces, each with its own
    axis-aligned patch grid.

    Args:
        points: (N, 3) point positions.
        n_patches_u: grid rows per face.
        n_patches_v: grid cols per face.
        face_assignments: optional (N,) face ids in {0, ..., 5}. If provided,
            these are used directly.
        assignment_mode: method used to infer face ids when face_assignments is
            None. Currently supports 'argmax_abs'.

    Returns:
        assignments: (N,) global patch ids in [0, 6*n_patches_u*n_patches_v).
        grid_topology: (6, n_patches_u, n_patches_v) patch ids per face.
        patch_params: (N, 2) local UV coordinates within each face.
        face_assignments: (N,) face ids in {0, ..., 5}.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if n_patches_u < 1 or n_patches_v < 1:
        raise ValueError(
            f"n_patches_u and n_patches_v must both be >= 1, got {(n_patches_u, n_patches_v)}"
        )

    if face_assignments is None:
        if assignment_mode != 'argmax_abs':
            raise ValueError(
                f"Unknown assignment_mode: {assignment_mode}. Use 'argmax_abs'."
            )
        face_assignments = _infer_box_face_assignments(points)
    else:
        face_assignments = np.asarray(face_assignments, dtype=np.int32)
        if face_assignments.shape != (points.shape[0],):
            raise ValueError(
                f"face_assignments must have shape ({points.shape[0]},), got {face_assignments.shape}"
            )
        if np.any((face_assignments < 0) | (face_assignments > 5)):
            raise ValueError("face_assignments values must lie in {0, 1, 2, 3, 4, 5}")

    patches_per_face = n_patches_u * n_patches_v
    assignments = np.full(points.shape[0], -1, dtype=np.int32)
    patch_params = _box_face_local_uv(points, face_assignments)
    grid_topology = np.stack([
        np.arange(face_id * patches_per_face,
                  (face_id + 1) * patches_per_face,
                  dtype=np.int32).reshape(n_patches_u, n_patches_v)
        for face_id in range(6)
    ], axis=0)

    for face_id in range(6):
        mask = face_assignments == face_id
        if not np.any(mask):
            continue

        uv_face = patch_params[mask]
        u = uv_face[:, 0]
        v = uv_face[:, 1]

        u_bins = np.clip((u * n_patches_u).astype(int), 0, n_patches_u - 1)
        v_bins = np.clip((v * n_patches_v).astype(int), 0, n_patches_v - 1)
        local_patch_ids = u_bins * n_patches_v + v_bins
        assignments[mask] = (local_patch_ids + face_id * patches_per_face).astype(np.int32)

    counts = np.bincount(assignments, minlength=6 * patches_per_face)
    empty_patches = np.where(counts == 0)[0]
    if len(empty_patches) > 0:
        print(
            f"[Warning] {len(empty_patches)} empty six-sheet patches detected. "
            f"Consider reducing patch count or using a different face assignment strategy."
        )

    return assignments, grid_topology, patch_params, face_assignments



# Helper functions
def _normalize_01(x):
    """Normalize array to [0, 1] range."""
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-10:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _build_cotangent_laplacian(vertices, triangles):
    """
    Build the cotangent Laplacian matrix and mass matrix for a triangle mesh.
    This gives better parameterization than uniform Laplacian.

    Returns:
        L: (n_verts, n_verts) sparse cotangent Laplacian
        M: (n_verts, n_verts) sparse lumped mass matrix
    """
    n_verts = len(vertices)
    rows = []
    cols = []
    vals = []
    areas = np.zeros(n_verts)

    for tri in triangles:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]

        # Edge vectors
        eij = vj - vi
        eik = vk - vi
        ejk = vk - vj

        # Triangle area
        area = 0.5 * np.linalg.norm(np.cross(eij, eik))
        if area < 1e-12:
            continue

        # Cotangent weights for each edge
        # cot(angle at vertex i) = dot(eij, eik) / (2 * area)
        cot_i = np.dot(eij, eik) / (2 * area)
        cot_j = np.dot(-eij, ejk) / (2 * area)
        cot_k = np.dot(-eik, -ejk) / (2 * area)

        # Clamp to avoid negative weights (for non-Delaunay meshes)
        cot_i = max(cot_i, 0.01)
        cot_j = max(cot_j, 0.01)
        cot_k = max(cot_k, 0.01)

        # Edge (j, k) opposite vertex i: weight = cot_i / 2
        w_jk = cot_i / 2
        w_ik = cot_j / 2
        w_ij = cot_k / 2

        # Add to Laplacian (off-diagonal: negative weight, diagonal: sum of weights)
        for (a, b, w) in [(i, j, w_ij), (j, k, w_jk), (i, k, w_ik)]:
            rows.extend([a, b, a, b])
            cols.extend([b, a, a, b])
            vals.extend([-w, -w, w, w])

        # Lumped mass: area / 3 to each vertex
        areas[i] += area / 3
        areas[j] += area / 3
        areas[k] += area / 3

    L = csr_matrix((vals, (rows, cols)), shape=(n_verts, n_verts))
    M = diags(areas + 1e-10)  # avoid zero mass

    return L, M


def _build_uniform_laplacian(vertices, triangles):
    """Build uniform (combinatorial) graph Laplacian from mesh."""
    n_verts = len(vertices)
    edges = set()
    for tri in triangles:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]:
            edges.add((a, b))
            edges.add((b, a))

    rows, cols = zip(*edges)
    data = np.ones(len(rows))
    adj = csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))

    degree = np.array(adj.sum(axis=1)).flatten()
    D = diags(degree)
    L = D - adj
    return L


def _farthest_point_sampling(points, n_samples):
    """
    Farthest point sampling for representative subsampling.
    Gives better coverage than random sampling.
    """
    N = len(points)
    if n_samples >= N:
        return np.arange(N)

    selected = [np.random.randint(N)]
    min_dists = np.full(N, np.inf)

    for _ in range(n_samples - 1):
        last = points[selected[-1]]
        dists = np.linalg.norm(points - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

    return np.array(selected)


def _handle_empty_patches(patch_assignments, grid_topology, patch_params,
                          n_patches_u, n_patches_v):
    """
    Handle empty patches by merging them with neighbors or reassigning.
    Empty patches can occur when the parameterization is non-uniform.
    """
    n_patches = n_patches_u * n_patches_v
    counts = np.bincount(patch_assignments, minlength=n_patches)

    empty_patches = np.where(counts == 0)[0]
    if len(empty_patches) == 0:
        return patch_assignments, grid_topology

    print(f"[Warning] {len(empty_patches)} empty patches detected. "
          f"Consider reducing patch count or using adaptive gridding.")

    # For now, just report. The training loop should handle empty patches gracefully
    # (skip them or assign minimum points from neighbors)
    return patch_assignments, grid_topology


def compute_grid_dims(n_patches: int):
    """
    Find the most square-like (n_rows, n_cols) factorization for n_patches.
    
    Tries all factor pairs and picks the one closest to a square.
    If n_patches is prime, returns (1, n_patches).
    
    Args:
        n_patches: desired number of patches (will be exactly n_rows * n_cols)
    
    Returns:
        (n_rows, n_cols) tuple where n_rows <= n_cols and n_rows * n_cols == n_patches
    """
    best = (1, n_patches)
    for r in range(1, int(math.sqrt(n_patches)) + 1):
        if n_patches % r == 0:
            c = n_patches // r
            best = (r, c)
    return best  # (n_rows, n_cols) with n_rows <= n_cols



# Unified interface for pre-segmentation methods
def presegment(points, normals=None, method='poisson_spectral',
               n_patches_u=4, n_patches_v=4, **kwargs):
    """
    Unified pre-segmentation interface.

    Args:
        points: (N, 3) numpy array
        normals: (N, 3) numpy array or None
        method: one of 'poisson_spectral', 'spectral_direct', 'pca_grid',
            'axis_aligned_grid'
        n_patches_u, n_patches_v: grid dimensions
        **kwargs: additional arguments passed to the specific method

    Returns:
        dict with keys:
            'assignments': (N,) int array of patch indices
            'grid': (n_u, n_v) int array of grid topology
            'params': (N, 2) float array of (u,v) parameters
            'method': string name of method used
    """
    if method == 'poisson_spectral':
        if normals is None:
            raise ValueError("Poisson method requires normals. "
                           "Use method='spectral_direct' if normals unavailable.")
        assignments, grid, params = poisson_spectral_segmentation(
            points, normals, n_patches_u, n_patches_v, **kwargs
        )
    elif method == 'spectral_direct':
        assignments, grid, params = spectral_direct_segmentation(
            points, n_patches_u, n_patches_v, **kwargs
        )
    elif method == 'pca_grid':
        assignments, grid, params = pca_grid_segmentation(
            points, n_patches_u, n_patches_v
        )
    elif method == 'axis_aligned_grid':
        assignments, grid, params = axis_aligned_grid_segmentation(
            points, n_patches_u, n_patches_v, **kwargs
        )
    elif method == 'two_sheet_axis_aligned':
        assignments, grid, params, side_assignments = two_sheet_axis_aligned_segmentation(
            points, n_patches_u, n_patches_v, **kwargs
        )
    elif method == 'six_sheet_axis_aligned':
        assignments, grid, params, face_assignments = six_sheet_axis_aligned_segmentation(
            points, n_patches_u, n_patches_v, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Choose from: poisson_spectral, spectral_direct, pca_grid, axis_aligned_grid, two_sheet_axis_aligned, six_sheet_axis_aligned")

    # Print statistics
    n_patches = int(assignments.max()) + 1 if assignments.size > 0 else n_patches_u * n_patches_v
    counts = np.bincount(assignments, minlength=n_patches)
    print(f"[{method}] Patch statistics:")
    print(f"  Total points: {len(points)}")
    if method == 'two_sheet_axis_aligned':
        print(f"  Two sheets: 2 × ({n_patches_u} × {n_patches_v}) = {n_patches} patches")
        side_counts = np.bincount(side_assignments, minlength=2)
        print(f"  Side split counts: side0={side_counts[0]}, side1={side_counts[1]}")
    elif method == 'six_sheet_axis_aligned':
        print(f"  Six sheets: 6 × ({n_patches_u} × {n_patches_v}) = {n_patches} patches")
        face_counts = np.bincount(face_assignments, minlength=6)
        print("  Face split counts: "
              + ", ".join([
                  f"face{face_id}={face_counts[face_id]}" for face_id in range(6)
              ]))
    else:
        print(f"  Grid: {n_patches_u} × {n_patches_v} = {n_patches} patches")
    print(f"  Points per patch: min={counts[counts>0].min()}, "
          f"max={counts.max()}, mean={counts[counts>0].mean():.0f}")
    print(f"  Empty patches: {(counts == 0).sum()}")

    result = {
        'assignments': assignments,
        'grid': grid,
        'params': params,
        'method': method,
    }
    if method == 'two_sheet_axis_aligned':
        result['side_assignments'] = side_assignments
    if method == 'six_sheet_axis_aligned':
        result['face_assignments'] = face_assignments
    return result



# Visualization usiung matplotlib
def visualize_segmentation(points, result, save_path=None):
    """Visualize the segmentation result in 3D and parameter space."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return

    assignments = result['assignments']
    params = result['params']

    fig = plt.figure(figsize=(14, 5))

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=assignments, cmap='tab20', s=1, alpha=0.6)
    ax1.set_title('3D Segmentation')

    # Parameter space
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(params[:, 0], params[:, 1],
                           c=assignments, cmap='tab20', s=1, alpha=0.6)
    ax2.set_xlabel('u')
    ax2.set_ylabel('v')
    ax2.set_title('Parameter Space')
    ax2.set_aspect('equal')

    # Grid lines in parameter space
    n_u, n_v = result['grid'].shape
    for i in range(1, n_u):
        ax2.axhline(y=i/n_u, color='gray', linewidth=0.5, alpha=0.5)
    for j in range(1, n_v):
        ax2.axvline(x=j/n_v, color='gray', linewidth=0.5, alpha=0.5)

    # Histogram of patch sizes
    ax3 = fig.add_subplot(133)
    n_patches = n_u * n_v
    counts = np.bincount(assignments, minlength=n_patches)
    ax3.bar(range(n_patches), counts)
    ax3.set_xlabel('Patch ID')
    ax3.set_ylabel('Point Count')
    ax3.set_title('Points per Patch')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    plt.show()