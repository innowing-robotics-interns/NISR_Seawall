# utils.py

from email import utils
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
from sklearn.neighbors import NearestNeighbors

try:
    import open3d as o3d
except ImportError:
    o3d = None


# Point cloud loader
def _load_point_file(path):
    """
    Load a point cloud file.

    Supported formats include `.ply`, `.xyz`, `.txt`, `.csv`, `.pts`, and
    `.npy`.
    
    Returns:
        Tuple `(pts, normals)`.
    """
    ext = os.path.splitext(path)[1].lower()

    # NPY files.
    if ext == '.npy':
        data = np.load(path).astype(np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        pts = data[:, :3]
        normals = data[:, 3:6] if data.shape[1] >= 6 else None
        return pts, normals

    # PLY files.
    if ext == '.ply':
        # Try Open3D first because it handles binary PLY and normals.
        if o3d is not None:
            try:
                pcd = o3d.io.read_point_cloud(path)
                pts = np.asarray(pcd.points)
                if pts.shape[0] > 0:
                    normals_arr = np.asarray(pcd.normals)
                    if normals_arr.shape[0] == pts.shape[0] and normals_arr.shape[1] == 3:
                        # Ignore empty normal arrays returned as zeros.
                        if np.any(np.abs(normals_arr) > 1e-10):
                            return pts, normals_arr
                    return pts, None
            except Exception:
                pass

        # Fall back to manual ASCII PLY parsing.
        try:
            with open(path, 'rb') as f:
                # Read the header.
                header_lines = []
                while True:
                    line = f.readline()
                    if not line:
                        raise ValueError('Unexpected end of PLY header')
                    decoded = line.decode('ascii', errors='ignore').strip()
                    header_lines.append(decoded)
                    if decoded == 'end_header':
                        break

                # Parse header fields.
                vertex_count = None
                is_binary = False
                has_normals = False
                property_names = []
                for h in header_lines:
                    if h.startswith('element vertex'):
                        vertex_count = int(h.split()[-1])
                    if 'binary' in h:
                        is_binary = True
                    if h.startswith('property') and 'nx' in h:
                        has_normals = True
                    if h.startswith('property'):
                        parts = h.split()
                        if len(parts) >= 3:
                            property_names.append(parts[-1])

                if vertex_count is None:
                    raise ValueError('PLY header missing vertex count')

                if is_binary:
                    if o3d is None:
                        raise RuntimeError(
                            "Binary PLY detected but open3d is not installed.\n"
                            "  Install with: pip install open3d\n"
                            "  Or convert your PLY to ASCII format."
                        )
                    else:
                        raise RuntimeError("open3d failed to read this binary PLY.")

                # Determine column indices for coordinates and normals.
                try:
                    xi = property_names.index('x')
                    yi = property_names.index('y')
                    zi = property_names.index('z')
                except ValueError:
                    xi, yi, zi = 0, 1, 2

                nxi, nyi, nzi = None, None, None
                if has_normals:
                    try:
                        nxi = property_names.index('nx')
                        nyi = property_names.index('ny')
                        nzi = property_names.index('nz')
                    except ValueError:
                        has_normals = False

                # Read ASCII vertex data.
                pts = []
                normals_list = []
                for _ in range(vertex_count):
                    line = f.readline().decode('ascii', errors='ignore')
                    if not line:
                        break
                    parts = line.split()
                    if len(parts) >= 3:
                        pts.append([float(parts[xi]), float(parts[yi]), float(parts[zi])])
                        if has_normals and len(parts) > max(nxi, nyi, nzi):
                            normals_list.append([float(parts[nxi]), float(parts[nyi]), float(parts[nzi])])

                pts_arr = np.asarray(pts, dtype=np.float64)
                normals_arr = None
                if has_normals and len(normals_list) == len(pts):
                    normals_arr = np.asarray(normals_list, dtype=np.float64)

                return pts_arr, normals_arr
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load PLY file: {path}\n  Error: {e}")

    # XYZ / TXT / CSV / PTS files.
    if ext in ('.xyz', '.txt', '.csv', '.pts'):
        try:
            # Try comma first, then whitespace
            try:
                data = np.loadtxt(path, delimiter=',')
            except ValueError:
                data = np.loadtxt(path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 6:
                return data[:, :3].astype(np.float64), data[:, 3:6].astype(np.float64)
            elif data.shape[1] >= 3:
                return data[:, :3].astype(np.float64), None
            else:
                raise ValueError(f"Expected at least 3 columns, got {data.shape[1]}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {ext} file: {path}\n  Error: {e}")

    raise RuntimeError(f"Unsupported file extension: {ext}\n"
                       f"  Supported: .ply, .xyz, .txt, .csv, .pts, .npy")


def load_point_cloud(filepath, downsample_n=None):
    """
    Load and normalize a 3D point cloud.

    The returned metadata stores the transform needed to recover original
    coordinates. Normals, when present, are re-normalized to unit length.
    
    Returns:
        Tuple `(pts_norm, meta)`.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    pts, normals = _load_point_file(filepath)
    print(f"  Loaded {pts.shape[0]} points from {filepath}")
    if normals is not None:
        print(f"  Normals found: {normals.shape}")
    else:
        print(f"  Normals: NOT found in file")

    # Deduplicate points while keeping normals aligned.
    pts_rounded = pts.round(decimals=8)
    _, unique_idx = np.unique(pts_rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve original order
    pts = pts[unique_idx]
    if normals is not None:
        normals = normals[unique_idx]

    # Optional downsampling.
    if downsample_n is not None and pts.shape[0] > downsample_n:
        idx = np.random.choice(pts.shape[0], downsample_n, replace=False)
        pts = pts[idx]
        if normals is not None:
            normals = normals[idx]

    # Normalize positions to roughly [-1, 1].
    center = pts.mean(axis=0)
    pts_centered = pts - center
    scale = np.abs(pts_centered).max()
    if scale < 1e-8:
        scale = 1.0
    pts_norm = pts_centered / scale

    # Re-normalize normals to unit length.
    if normals is not None:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals = (normals / norms).astype(np.float32)

    meta = {
        'center': center,
        'scale': scale,
        'n_raw': pts.shape[0],
        'normals': normals,
    }
    print(f"  After dedup/downsample: {pts_norm.shape[0]} points")
    print(f"  Normalization: center={center}, scale={scale:.6f}")
    print(f"  To recover original coords: p_orig = p_norm * {scale:.6f} + center")
    return pts_norm.astype(np.float32), meta


def estimate_normals(pts, k=30):
    """Estimate per-point normals with local PCA."""

    nbrs = NearestNeighbors(n_neighbors=k).fit(pts)
    _, idx = nbrs.kneighbors(pts)
    normals = np.zeros_like(pts)
    for i in range(pts.shape[0]):
        nb = pts[idx[i]] - pts[idx[i]].mean(axis=0)
        cov = nb.T @ nb
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest-eigenvalue direction
    return normals.astype(np.float32)

# Make synthetic_surface if no file is given
def make_synthetic_surface(shape='saddle', n=20000, noise=0.02, seed=42):
    """
    Generate a synthetic 3D point cloud on a known surface.
    Returns (pts_normalized, meta) where meta contains identity transform.
    """
    rng = np.random.default_rng(seed)

    if shape == 'saddle':
        u = rng.uniform(-1, 1, n)
        v = rng.uniform(-1, 1, n)
        x = u
        y = v
        z = u**2 - v**2
    elif shape == 'hemisphere':
        theta = rng.uniform(0, np.pi / 2, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
    elif shape == 'torus_patch':
        R, r = 1.0, 0.4
        u = rng.uniform(0, np.pi, n)
        v = rng.uniform(0, 2 * np.pi, n)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
    elif shape == 'wavy':
        u = rng.uniform(-1, 1, n)
        v = rng.uniform(-1, 1, n)
        x = u
        y = v
        z = 0.3 * np.sin(2 * np.pi * u) * np.cos(2 * np.pi * v)
    elif shape == 'sphere':
        theta = rng.uniform(0, np.pi, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
    elif shape == 'flat_sheet':
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.ones(n, dtype=np.float32)
    else:
        raise ValueError(f"Unknown surface type: {shape}")

    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    pts += rng.normal(0, noise, pts.shape).astype(np.float32)

    # Normalize to [-1, 1]
    center = pts.mean(axis=0)
    pts -= center
    scale = float(np.abs(pts).max())
    if scale > 1e-8:
        pts /= scale

    meta = {
        'center': center,
        'scale': scale,
        'n_raw': n,
    }

    ply_path = os.path.abspath(f"synthetic_{shape}_n{n}_seed{seed}.ply")
    export_point_cloud_ply(pts, ply_path)
    meta['synthetic_ply_path'] = ply_path

    print(f"  Generated '{shape}' surface: {n} points, noise={noise}")
    print(f"  Normalization: center={center}, scale={scale:.6f}")
    print(f"  Synthetic point cloud saved to: {meta['synthetic_ply_path']}")
    return pts, meta


def export_point_cloud_ply(points, path, normals=None):
    """Export a point cloud as an ASCII PLY file, optionally with normals."""
    points = np.asarray(points)
    normals = None if normals is None else np.asarray(normals)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if normals is not None and normals.shape != points.shape:
        raise ValueError(
            f"normals must have shape {points.shape}, got {normals.shape}"
        )

    with open(path, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("end_header\n")

        if normals is None:
            for p in points:
                f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        else:
            for p, n in zip(points, normals):
                f.write(
                    f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                    f"{n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n"
                )

    print(f"    Point-cloud PLY → {path}")





# Mesh sampling and export
@torch.no_grad()
def sample_surface_grid(F, resolution=1000, device='cuda'):
    """Sample single-patch F on a regular UV grid → vertices + triangle faces."""
    u = torch.linspace(0, 1, resolution, device=device)
    v = torch.linspace(0, 1, resolution, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
    uv = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)

    # Batch evaluation to avoid OOM
    batch_size = 4096
    verts_list = []
    for i in range(0, uv.shape[0], batch_size):
        verts_list.append(F(uv[i:i+batch_size]).cpu())
    verts = torch.cat(verts_list, dim=0).numpy()  # (res*res, 3)

    # Build triangle faces from grid connectivity
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx00 = i * resolution + j
            idx10 = (i + 1) * resolution + j
            idx01 = i * resolution + (j + 1)
            idx11 = (i + 1) * resolution + (j + 1)
            faces.append([idx00, idx10, idx11])
            faces.append([idx00, idx11, idx01])
    faces = np.array(faces, dtype=np.int32)

    return verts, faces


@torch.no_grad()
def sample_multi_patch_grid(F, resolution=100, device='cuda', active_patch_ids=None):
    """
    Sample ALL patches of a MultiPatchForwardMap on regular UV grids.
    
    Each patch produces a resolution×resolution vertex grid with 2*(res-1)² triangles.
    All patch meshes are concatenated into one combined mesh.
    
    At shared boundaries between adjacent patches, the vertices will be at
    identical 3D positions (due to shared vertex features), giving a seamless mesh.
    Duplicate vertices at boundaries are kept (could be merged in post-processing).
    
    Args:
        F: MultiPatchForwardMap model (eval mode)
        resolution: per-patch UV grid resolution
        device: computation device
        active_patch_ids: optional iterable of patch indices to sample. If None,
            all patches are sampled.
    
    Returns:
        all_verts: (total_verts, 3) combined vertex array
        all_faces: (total_faces, 3) combined face index array
    """
    if active_patch_ids is None:
        patch_ids = list(range(F.n_patches))
    else:
        patch_ids = [int(patch_idx) for patch_idx in active_patch_ids]

    if len(patch_ids) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    all_verts = []
    all_faces = []
    vertex_offset = 0

    for patch_idx in patch_ids:
        # Generate UV grid for this patch
        u = torch.linspace(0, 1, resolution, device=device)
        v = torch.linspace(0, 1, resolution, device=device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        uv = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)

        # Batch evaluation to avoid OOM
        batch_size = 4096
        verts_list = []
        for i in range(0, uv.shape[0], batch_size):
            verts_list.append(F(patch_idx, uv[i:i+batch_size]).cpu())
        verts = torch.cat(verts_list, dim=0).numpy()  # (res*res, 3)

        # Build triangle faces for this patch (with global vertex offset)
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                idx00 = i * resolution + j + vertex_offset
                idx10 = (i + 1) * resolution + j + vertex_offset
                idx01 = i * resolution + (j + 1) + vertex_offset
                idx11 = (i + 1) * resolution + (j + 1) + vertex_offset
                faces.append([idx00, idx10, idx11])
                faces.append([idx00, idx11, idx01])

        all_verts.append(verts)
        all_faces.extend(faces)
        vertex_offset += verts.shape[0]

    all_verts = np.vstack(all_verts)
    all_faces = np.array(all_faces, dtype=np.int32)

    return all_verts, all_faces


def unnormalize_vertices(verts, meta):
    """
    Transform vertices from normalized [-1, 1]³ space back to the original
    coordinate system of the input point cloud.

    Inverse of:  pts_norm = (pts - center) / scale
    Therefore:   pts_orig = pts_norm * scale + center
    """
    center = meta['center']  # shape (3,)
    scale = meta['scale']    # scalar
    verts_original = verts * scale + center
    return verts_original


def export_obj(verts, faces, path):
    """Export mesh as OBJ file."""
    with open(path, 'w') as f:
        f.write("# Learned sheet mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"    OBJ → {path}")


def export_ply(verts, faces, path):
    """Export mesh as PLY file."""
    nv, nf = verts.shape[0], faces.shape[0]
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    print(f"    PLY → {path}")



# Visualization
@torch.no_grad()
def visualise(F, G, pts3n, history, out_path, resolution=80, device='cuda'):
    """
    6-panel figure for single-patch mode:
      [0] Input point cloud (3D)
      [1] Learned sheet F(u,v) overlaid on input (view 1)
      [2] Learned sheet (view 2 — rotated)
      [3] Wireframe view
      [4] Inverse map G(p) — UV color on input points
      [5] Loss curves
    """
    device_model = next(F.parameters()).device
    P_np = pts3n

    # ── sample mesh ───────────────────────────────────────────────────────────
    verts, faces = sample_surface_grid(F, resolution, device_model)
    res = resolution
    X = verts[:, 0].reshape(res, res)
    Y = verts[:, 1].reshape(res, res)
    Z = verts[:, 2].reshape(res, res)

    # ── compute G(P) for coloring ─────────────────────────────────────────────
    P_t = torch.tensor(P_np, dtype=torch.float32, device=device_model)
    uv_inv = G(P_t).cpu().numpy()  # (N, 2)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.35, wspace=0.30,
                           left=0.04, right=0.97,
                           top=0.92, bottom=0.06)

    stride = max(1, res // 40)

    # ── Panel 0: Input point cloud ────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0], projection='3d')
    ax0.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                s=3, c='#3fb950', alpha=0.6, linewidths=0)
    ax0.set_title("① Input Point Cloud", fontsize=10, pad=8)
    ax0.set_xlabel("x", fontsize=7)
    ax0.set_ylabel("y", fontsize=7)
    ax0.set_zlabel("z", fontsize=7)

    # ── Panel 1: Sheet + points (view 1) ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1], projection='3d')
    ax1.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis',
                     edgecolor='none', rstride=stride, cstride=stride,
                     antialiased=True)
    ax1.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                s=3, c='red', alpha=0.5, linewidths=0)
    ax1.set_title("② Learned Sheet F(u,v) — View 1", fontsize=10, pad=8)
    ax1.set_xlabel("x", fontsize=7)
    ax1.set_ylabel("y", fontsize=7)
    ax1.set_zlabel("z", fontsize=7)

    # ── Panel 2: Sheet + points (view 2) ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2], projection='3d')
    ax2.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis',
                     edgecolor='none', rstride=stride, cstride=stride,
                     antialiased=True)
    ax2.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                s=3, c='red', alpha=0.5, linewidths=0)
    ax2.view_init(elev=15, azim=60)
    ax2.set_title("③ View 2 (rotated)", fontsize=10, pad=8)
    ax2.set_xlabel("x", fontsize=7)
    ax2.set_ylabel("y", fontsize=7)
    ax2.set_zlabel("z", fontsize=7)

    # ── Panel 3: Wireframe ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    wire_stride = max(1, res // 20)
    ax3.plot_wireframe(X, Y, Z, alpha=0.35, color='steelblue', linewidth=0.4,
                       rstride=wire_stride, cstride=wire_stride)
    ax3.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                s=3, c='red', alpha=0.5, linewidths=0)
    ax3.set_title("④ Wireframe", fontsize=10, pad=8)
    ax3.set_xlabel("x", fontsize=7)
    ax3.set_ylabel("y", fontsize=7)
    ax3.set_zlabel("z", fontsize=7)

    # ── Panel 4: Inverse map G(p) — color by UV ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    u_color = uv_inv[:, 0]
    sc = ax4.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                     s=6, c=u_color, cmap='plasma', alpha=0.7, linewidths=0,
                     vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax4, fraction=0.03, pad=0.08, label="G(p)_u")
    ax4.set_title("⑤ Inverse Map G(p) — u component", fontsize=10, pad=8)
    ax4.set_xlabel("x", fontsize=7)
    ax4.set_ylabel("y", fontsize=7)
    ax4.set_zlabel("z", fontsize=7)

    # ── Panel 5: Loss curves ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ep = history['epoch']
    ax5.plot(ep, history['cd'], color='#58a6ff', lw=1.8, label='Chamfer (CD)')
    ax5.plot(ep, history['cycle'], color='#d2a8ff', lw=1.8, label='Cycle (λ₁)')
    ax5.plot(ep, history['param'], color='#f0883e', lw=1.8, label='Param (λ₂)')
    ax5.plot(ep, history['total'], color='#f78166', lw=2.2, label='Total', ls='--')
    ax5.set_yscale('log')
    ax5.set_xlabel("Epoch", fontsize=8)
    ax5.set_ylabel("Loss (log)", fontsize=8)
    ax5.legend(fontsize=7)
    ax5.set_title("⑥ Training Loss", fontsize=10, pad=8)
    ax5.grid(True, alpha=0.3)

    fig.suptitle("Chamfer Distance Sheet Fitting  ·  F: [0,1]² → R³  &  G: R³ → [0,1]²",
                 fontsize=14, fontweight='bold', y=0.97)
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"    PNG → {out_path}")

    return verts, faces


@torch.no_grad()
def visualise_multi_patch(F, pts3n, assignments, history, out_path,
                          resolution=80, device='cuda', active_patch_ids=None):
    """
    6-panel figure for multi-patch feature complex mode:
      [0] Input point cloud colored by patch assignment
      [1] Multi-patch surface + input points (view 1)
      [2] Multi-patch surface + input points (view 2, rotated)
      [3] Per-patch surfaces shown with distinct colors
      [4] Patch assignment map (top-down view of segmentation)
      [5] Loss curves
    
    Args:
        F: trained MultiPatchForwardMap
        pts3n: (N, 3) normalized point cloud
        assignments: (N,) patch index for each point
        history: dict with 'epoch', 'cd', 'tangent', 'total' lists
        out_path: path to save the figure
        resolution: per-patch UV grid resolution for visualization
        device: computation device
        active_patch_ids: optional iterable of patch indices to visualize and export.
    """
    device_model = next(F.parameters()).device
    P_np = pts3n
    if active_patch_ids is None:
        patch_ids = list(range(F.n_patches))
    else:
        patch_ids = [int(patch_idx) for patch_idx in active_patch_ids]

    n_patches = max(len(patch_ids), 1)
    n_rows = F.n_rows
    n_cols = F.n_cols

    # ── Sample each patch on its UV grid ──────────────────────────────────────
    # Store per-patch surface grids for individual plotting
    patch_surfaces = []  # list of (X, Y, Z) arrays, each (res, res)
    for patch_idx in patch_ids:
        u = torch.linspace(0, 1, resolution, device=device_model)
        v = torch.linspace(0, 1, resolution, device=device_model)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        uv = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)

        batch_size = 4096
        verts_list = []
        for i in range(0, uv.shape[0], batch_size):
            verts_list.append(F(patch_idx, uv[i:i+batch_size]).cpu())
        verts = torch.cat(verts_list, dim=0).numpy()

        X = verts[:, 0].reshape(resolution, resolution)
        Y = verts[:, 1].reshape(resolution, resolution)
        Z = verts[:, 2].reshape(resolution, resolution)
        patch_surfaces.append((X, Y, Z))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.35, wspace=0.30,
                           left=0.04, right=0.97,
                           top=0.92, bottom=0.06)

    stride = max(1, resolution // 40)

    # Color map for patches
    patch_cmap = plt.get_cmap('tab10', n_patches)

    # ── Panel 0: Input cloud colored by patch assignment ──────────────────────
    ax0 = fig.add_subplot(gs[0, 0], projection='3d')
    ax0.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                s=3, c=assignments, cmap='tab10', alpha=0.6, linewidths=0,
                vmin=0, vmax=max(n_patches - 1, 1))
    ax0.set_title("① Input Cloud (patch assignment)", fontsize=10, pad=8)
    ax0.set_xlabel("x", fontsize=7)
    ax0.set_ylabel("y", fontsize=7)
    ax0.set_zlabel("z", fontsize=7)

    # ── Panel 1: All patches surface + points (view 1) ───────────────────────
    ax1 = fig.add_subplot(gs[0, 1], projection='3d')
    for k, (X, Y, Z) in enumerate(patch_surfaces):
        color = patch_cmap(k / max(n_patches - 1, 1))
        ax1.plot_surface(X, Y, Z, alpha=0.4, color=color,
                         edgecolor='none', rstride=stride, cstride=stride,
                         antialiased=True)
    ax1.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                s=2, c='red', alpha=0.4, linewidths=0)
    ax1.set_title("② Multi-Patch Surface — View 1", fontsize=10, pad=8)
    ax1.set_xlabel("x", fontsize=7)
    ax1.set_ylabel("y", fontsize=7)
    ax1.set_zlabel("z", fontsize=7)

    # ── Panel 2: All patches surface + points (view 2, rotated) ───────────────
    ax2 = fig.add_subplot(gs[0, 2], projection='3d')
    for k, (X, Y, Z) in enumerate(patch_surfaces):
        color = patch_cmap(k / max(n_patches - 1, 1))
        ax2.plot_surface(X, Y, Z, alpha=0.4, color=color,
                         edgecolor='none', rstride=stride, cstride=stride,
                         antialiased=True)
    ax2.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                s=2, c='red', alpha=0.4, linewidths=0)
    ax2.view_init(elev=15, azim=60)
    ax2.set_title("③ View 2 (rotated)", fontsize=10, pad=8)
    ax2.set_xlabel("x", fontsize=7)
    ax2.set_ylabel("y", fontsize=7)
    ax2.set_zlabel("z", fontsize=7)

    # ── Panel 3: Wireframe per patch ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    wire_stride = max(1, resolution // 20)
    for k, (X, Y, Z) in enumerate(patch_surfaces):
        color = patch_cmap(k / max(n_patches - 1, 1))
        ax3.plot_wireframe(X, Y, Z, alpha=0.3, color=color, linewidth=0.4,
                           rstride=wire_stride, cstride=wire_stride)
    ax3.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                s=2, c='red', alpha=0.4, linewidths=0)
    ax3.set_title("④ Wireframe (per patch)", fontsize=10, pad=8)
    ax3.set_xlabel("x", fontsize=7)
    ax3.set_ylabel("y", fontsize=7)
    ax3.set_zlabel("z", fontsize=7)

    # ── Panel 4: Patch boundary view (surface only, no points) ────────────────
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    for k, (X, Y, Z) in enumerate(patch_surfaces):
        color = patch_cmap(k / max(n_patches - 1, 1))
        ax4.plot_surface(X, Y, Z, alpha=0.6, color=color,
                         edgecolor='gray', rstride=stride, cstride=stride,
                         linewidth=0.1, antialiased=True)
    ax4.set_title("⑤ Patch Layout (surface only)", fontsize=10, pad=8)
    ax4.set_xlabel("x", fontsize=7)
    ax4.set_ylabel("y", fontsize=7)
    ax4.set_zlabel("z", fontsize=7)

    # ── Panel 5: Loss curves ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ep = history['epoch']
    ax5.plot(ep, history['cd'], color='#58a6ff', lw=1.8, label='Chamfer (CD)')
    ax5.plot(ep, history['tangent'], color='#d2a8ff', lw=1.8, label='Tangent (μ)')
    ax5.plot(ep, history['total'], color='#f78166', lw=2.2, label='Total', ls='--')
    ax5.set_yscale('log')
    ax5.set_xlabel("Epoch", fontsize=8)
    ax5.set_ylabel("Loss (log)", fontsize=8)
    ax5.legend(fontsize=7)
    ax5.set_title("⑥ Training Loss", fontsize=10, pad=8)
    ax5.grid(True, alpha=0.3)

    fig.suptitle(f"Feature Complex Sheet Fitting  ·  {n_rows}×{n_cols} grid  ·  "
                 f"active patches: {len(patch_ids)}  ·  "
                 f"shared vertices guarantee C0 continuity",
                 fontsize=13, fontweight='bold', y=0.97)
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"    PNG → {out_path}")

    # ── Also produce the combined mesh for export ─────────────────────────────
    verts, faces = sample_multi_patch_grid(
        F, resolution, device_model, active_patch_ids=patch_ids
    )
    return verts, faces


def _visualize_patch_assignments(pts3n, assignments, grid_topology, patch_params,
                                  n_rows, n_cols, save_path='patch_assignments.png'):
    """
    Visualize patch assignments in 2D parameter space and 3D.
    
    Args:
        pts3n: (N, 3) normalized point cloud
        assignments: (N,) patch indices
        grid_topology: (n_rows, n_cols) grid mapping
        patch_params: (N, 2) UV parameters
        n_rows, n_cols: grid dimensions
        save_path: output file path
    """
    n_patches = n_rows * n_cols
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color map for patches
    cmap = plt.get_cmap('tab20', n_patches)
    
    # ── Panel 0: 3D view with patch colors ────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0], projection='3d')
    scatter0 = ax0.scatter(pts3n[:, 0], pts3n[:, 1], pts3n[:, 2],
                           c=assignments, cmap='tab20', s=2, alpha=0.7,
                           vmin=0, vmax=max(n_patches - 1, 1))
    ax0.set_title(f'3D Point Cloud Colored by Patch (Grid: {n_rows}×{n_cols})', fontsize=10)
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    
    # ── Panel 1: Parameter space (UV) ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    scatter1 = ax1.scatter(patch_params[:, 0], patch_params[:, 1],
                           c=assignments, cmap='tab20', s=3, alpha=0.7,
                           vmin=0, vmax=max(n_patches - 1, 1))
    ax1.set_title('Parameter Space (UV) with Grid', fontsize=10)
    ax1.set_xlabel('u')
    ax1.set_ylabel('v')
    ax1.set_aspect('equal')
    
    # Draw grid lines
    for i in range(1, n_rows):
        ax1.axhline(y=i/n_rows, color='black', linewidth=0.5, alpha=0.5)
    for j in range(1, n_cols):
        ax1.axvline(x=j/n_cols, color='black', linewidth=0.5, alpha=0.5)
    
    # ── Panel 2: Patch sizes histogram ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    counts = np.bincount(assignments, minlength=n_patches)
    bars = ax2.bar(range(n_patches), counts, color=[cmap(i) for i in range(n_patches)])
    ax2.set_title('Points per Patch', fontsize=10)
    ax2.set_xlabel('Patch ID')
    ax2.set_ylabel('Point Count')
    ax2.axhline(y=np.mean(counts[counts > 0]), color='red', linestyle='--', 
                label=f'Mean: {counts[counts > 0].mean():.0f}')
    ax2.legend()
    
    # ── Panel 3: Grid topology visualization ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    # Create a colormap for the grid
    grid_colors = np.arange(n_patches).reshape(n_rows, n_cols)
    im = ax3.imshow(grid_colors, cmap='tab20', aspect='equal', 
                    vmin=0, vmax=max(n_patches - 1, 1))
    ax3.set_title('Grid Topology (Patch IDs)', fontsize=10)
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    # Add text labels
    for i in range(n_rows):
        for j in range(n_cols):
            patch_id = grid_topology[i, j]
            color = 'white' if patch_id < n_patches // 2 else 'black'
            ax3.text(j, i, str(patch_id), ha='center', va='center', color=color, fontsize=8)
    
    # ── Panel 4: Individual patches in 3D (grid layout) ──────────────────────
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    # Show each patch with different colors
    for k in range(min(n_patches, 20)):  # Limit to 20 patches for clarity
        mask = assignments == k
        if mask.sum() > 0:
            ax4.scatter(pts3n[mask, 0], pts3n[mask, 1], pts3n[mask, 2],
                       c=[cmap(k)], s=2, alpha=0.6, label=f'Patch {k}')
    ax4.set_title(f'Individual Patches (showing up to 20)', fontsize=10)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.legend(loc='upper right', fontsize=6, ncol=2)
    
    # ── Panel 5: Patch boundary visualization ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    # Show the grid topology as a colored grid with patch sizes
    patch_sizes = counts.reshape(n_rows, n_cols)
    im2 = ax5.imshow(patch_sizes, cmap='viridis', aspect='equal')
    ax5.set_title('Patch Sizes (Grid Layout)', fontsize=10)
    ax5.set_xlabel('Column')
    ax5.set_ylabel('Row')
    # Add text labels with sizes
    for i in range(n_rows):
        for j in range(n_cols):
            ax5.text(j, i, f'{patch_sizes[i, j]}', ha='center', va='center', 
                    color='white', fontsize=8)
    plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04, label='Point Count')
    
    plt.suptitle(f'Patch Assignment Visualization (Grid: {n_rows}×{n_cols} = {n_patches} patches)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Patch visualization saved to: {save_path}")


def _visualize_patch_assignments_3d(pts3n, assignments, grid_topology,
                                     n_rows, n_cols, save_path='patch_assignments_3d.png'):
    """
    Create a 3D visualization showing patch boundaries with different views.
    """
    n_patches = n_rows * n_cols
    cmap = plt.get_cmap('tab20', n_patches)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Three different views
    views = [
        (30, 45, 'View 1 (default)'),
        (10, 90, 'View 2 (side)'),
        (80, 45, 'View 3 (top)'),
    ]
    
    for idx, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Plot each patch
        for k in range(n_patches):
            mask = assignments == k
            if mask.sum() > 0:
                ax.scatter(pts3n[mask, 0], pts3n[mask, 1], pts3n[mask, 2],
                          c=[cmap(k)], s=2, alpha=0.6)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'{title}\nElev={elev}°, Azim={azim}°', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # Set equal aspect ratio
        max_range = np.array([pts3n[:, 0].max() - pts3n[:, 0].min(),
                             pts3n[:, 1].max() - pts3n[:, 1].min(),
                             pts3n[:, 2].max() - pts3n[:, 2].min()]).max() / 2.0
        mid_x = (pts3n[:, 0].max() + pts3n[:, 0].min()) * 0.5
        mid_y = (pts3n[:, 1].max() + pts3n[:, 1].min()) * 0.5
        mid_z = (pts3n[:, 2].max() + pts3n[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(f'3D Patch Visualization (Grid: {n_rows}×{n_cols} = {n_patches} patches)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  3D patch visualization saved to: {save_path}")


def _save_patch_statistics(pts3n, assignments, n_rows, n_cols, save_path='patch_statistics.txt'):
    """
    Save detailed patch statistics to a text file.
    """
    n_patches = n_rows * n_cols
    counts = np.bincount(assignments, minlength=n_patches)
    
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PATCH STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total points: {len(pts3n):,}\n")
        f.write(f"Grid: {n_rows} rows × {n_cols} cols = {n_patches} patches\n\n")
        
        f.write("Patch Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Patch ID':>8} {'Points':>12} {'Percentage':>12} {'Status':>10}\n")
        f.write("-" * 60 + "\n")
        
        for k in range(n_patches):
            count = counts[k]
            percentage = (count / len(pts3n)) * 100
            status = "ACTIVE" if count > 0 else "EMPTY"
            f.write(f"{k:>8} {count:>12,} {percentage:>11.2f}% {status:>10}\n")
        
        f.write("-" * 60 + "\n\n")
        
        active_counts = counts[counts > 0]
        if len(active_counts) > 0:
            f.write(f"Active patches: {len(active_counts)} / {n_patches}\n")
            f.write(f"Empty patches: {n_patches - len(active_counts)}\n")
            f.write(f"Min points in active patch: {active_counts.min():,}\n")
            f.write(f"Max points in active patch: {active_counts.max():,}\n")
            f.write(f"Mean points per active patch: {active_counts.mean():.1f}\n")
            f.write(f"Std deviation: {active_counts.std():.1f}\n")
    
    print(f"  Patch statistics saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: UNIQUE FOLDER & METADATA
# ─────────────────────────────────────────────────────────────────────────────

def _get_unique_folder(base_dir: str) -> str:
    """Create a unique folder name. If base_dir exists, append _1, _2, etc."""
    if not os.path.exists(base_dir):
        return base_dir
    i = 1
    while os.path.exists(f"{base_dir}_{i}"):
        i += 1
    return f"{base_dir}_{i}"


def _save_run_metadata(log_path: str, args, input_file: str, result_dir: str,
                       result_png: str, history: dict, n_points: int, meta: dict):
    """Save comprehensive run metadata including hyperparameters and loss history."""

    def _safe_getattr(obj, name, default=None):
        try:
            return getattr(obj, name)
        except AttributeError as exc:
            print(f"    Warning: missing attribute '{name}' on {type(obj).__name__}; using None ({exc})")
            return default

    def _safe_getitem(mapping, key, default=None):
        try:
            return mapping[key]
        except (KeyError, TypeError) as exc:
            print(f"    Warning: missing key '{key}' on {type(mapping).__name__}; using None ({exc})")
            return default

    multi_patch = _safe_getattr(args, 'multi_patch', False)

    # Build hyperparameters dict based on mode
    hyperparams = {
        'epochs': _safe_getattr(args, 'epochs', None),
        'M_surface_samples': _safe_getattr(args, 'M', None),
        'W_mlp_width': _safe_getattr(args, 'W', None),
        'D_mlp_depth': _safe_getattr(args, 'D', None),
        'learning_rate': _safe_getattr(args, 'lr', None),
        'mu_tangent_weight': _safe_getattr(args, 'mu', None),
        'gamma_normal_weight': _safe_getattr(args, 'gamma', None),
        'Softplus_beta': _safe_getattr(args, 'beta', None),
        'mesh_resolution': _safe_getattr(args, 'mesh_res', None),
        'device': _safe_getattr(args, 'device', None),
        'multi_patch': multi_patch,
    }

    if multi_patch:
        hyperparams.update({
            'n_patches': _safe_getattr(args, 'n_patches', None),
            'd_features': _safe_getattr(args, 'd_features', None),
            'L_fourier_frequencies': _safe_getattr(args, 'L', None),
            'L_inverse_map_frequencies': _safe_getattr(args, 'L_inv', None),
            'lambda1_cycle_weight': _safe_getattr(args, 'lam', None),
            'lambda2_param_weight': _safe_getattr(args, 'lam2', None),
        })
    else:
        hyperparams.update({
            'L_fourier_frequencies': _safe_getattr(args, 'L', None),
            'lambda1_cycle_weight': _safe_getattr(args, 'lam', None),
            'lambda2_param_weight': _safe_getattr(args, 'lam2', None),
        })

    # Build loss history dict based on what's available
    loss_hist = {
        'epoch': _safe_getitem(history, 'epoch', None),
        'chamfer_distance_loss': _safe_getitem(history, 'cd', None),
        'total_loss': _safe_getitem(history, 'total', None),
    }
    if _safe_getitem(history, 'cycle', None) is not None:
        loss_hist['cycle_consistency_loss'] = _safe_getitem(history, 'cycle', None)
    if _safe_getitem(history, 'param', None) is not None:
        loss_hist['parameter_distribution_loss'] = _safe_getitem(history, 'param', None)
    if _safe_getitem(history, 'tangent', None) is not None:
        loss_hist['tangent_loss'] = _safe_getitem(history, 'tangent', None)
    if _safe_getitem(history, 'normal', None) is not None:
        loss_hist['normal_loss'] = _safe_getitem(history, 'normal', None)


    center = _safe_getitem(meta, 'center', None)
    if center is not None and hasattr(center, 'tolist'):
        center_list = center.tolist()
    elif center is not None:
        center_list = list(center)
    else:
        center_list = None

    scale = _safe_getitem(meta, 'scale', None)

    payload = {
        'metadata': {
            'result_dir': result_dir,
            'result_png': result_png,
            'input_file': input_file,
            'n_input_points': n_points,
            'task': 'sheet_fitting_3D',
            'map': 'F: [0,1]^2 -> R^3 (multi-patch)' if multi_patch
                   else 'F: [0,1]^2 -> R^3 (single-patch)',
        },
        'normalization': {
            'description': 'p_original = p_normalized * scale + center',
            'center': center_list,
            'scale': float(scale) if scale is not None else None,
        },
        'hyperparameters': hyperparams,
        'loss_history': loss_hist,
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    print(f"    Metadata → {log_path}")
