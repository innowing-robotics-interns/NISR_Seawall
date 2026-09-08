#!/usr/bin/env python3
# uv_hole_mask.py

"""
Usage: 

    python utils/uv_hole_mask.py \
        --ckpt logs/.../checkpoint_5000.pt \
        --out_dir logs/.../uv_masks \
        --resolution 256

The input cloud defaults to the one recorded in the checkpoint's args.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import patch_vis
import utils as utils


# Colors used for the 3D validation PLY. No correspondence points are drawn red
COLOR_HAS_CORRESPONDENCE = (220, 220, 220)
COLOR_NO_CORRESPONDENCE = (230, 30, 30)



# sampling
@torch.no_grad()
def sample_patch_uv_grid(F, patch_id: int, resolution: int, device: str,
                         batch_size: int = 8192):
    """
    Evaluate one patch on a regular `resolution x resolution` UV grid.

    Returns:
        xyz: (resolution**2, 3) predicted points, flattened in `[u_index, v_index]`
             row-major order so `.reshape(resolution, resolution, 3)` is indexed
             as `[u_index, v_index]`.
        uv:  (resolution**2, 2) the query coordinates, same ordering.
    """
    u = torch.linspace(0.0, 1.0, resolution, device=device)
    v = torch.linspace(0.0, 1.0, resolution, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
    uv = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)

    chunks = []
    for i in range(0, uv.shape[0], batch_size):
        uv_chunk = uv[i:i + batch_size]
        pid = torch.full((uv_chunk.shape[0],), int(patch_id),
                         dtype=torch.long, device=device)
        chunks.append(F(pid, uv_chunk).cpu())

    xyz = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    return xyz, uv.cpu().numpy().astype(np.float32)


# Threshold selection
def median_nn_spacing(points: np.ndarray, tree: cKDTree,
                      max_samples: int = 200000, seed: int = 0) -> float:
    """
    Median nearest-neighbor distance within the target cloud.

    This is the natural length unit for the threshold: any residual below a few
    multiples of it is indistinguishable from the cloud's own sampling density.
    """
    n = points.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_samples, replace=False)
        probe = points[idx]
    else:
        probe = points

    # k=2 because the first neighbor of a cloud point is itself.
    dists, _ = tree.query(probe, k=2, workers=-1)
    return float(np.median(dists[:, 1]))

# Not used. Use auto instead
def otsu_threshold(values: np.ndarray, n_bins: int = 512) -> float:
    """
    Otsu's between-class-variance threshold, for splitting a bimodal histogram.

    Applied to log10(distance) by the caller: the raw distance distribution is
    far too skewed (a huge spike near zero) for Otsu to split sensibly, while in
    log space the "on surface" and "in empty space" modes are comparable in width.
    """
    hist, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    total = hist.sum()
    if total == 0:
        return float(centers[-1])

    p = hist.astype(np.float64) / total
    omega = np.cumsum(p)                      # class-0 weight
    mu = np.cumsum(p * centers)               # class-0 cumulative mean
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b = (mu_total * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0

    return float(centers[int(np.argmax(sigma_b))])


def resolve_threshold(mode: str, distances: np.ndarray, d_nn: float,
                      nn_scale: float) -> tuple:
    """
    Turn the --threshold argument into a concrete value.

    Returns:
        Tuple `(tau, description)`.
    """
    if mode == 'auto':
        tau = nn_scale * d_nn
        return tau, f"auto = {nn_scale} x median_nn({d_nn:.6f})"

    if mode == 'otsu':
        safe = np.maximum(distances, 1e-9)
        tau = float(10.0 ** otsu_threshold(np.log10(safe)))
        return tau, f"otsu on log10(d) = {tau / d_nn:.2f} x median_nn"

    try:
        tau = float(mode)
    except ValueError:
        raise ValueError(
            f"--threshold must be 'auto', 'otsu', or a float in normalized units, got {mode!r}")
    if tau <= 0:
        raise ValueError(f"--threshold must be positive, got {tau}")
    return tau, f"explicit = {tau / d_nn:.2f} x median_nn"


# Output
def write_colored_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary little-endian colored point-cloud PLY."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr = np.empty(points.shape[0], dtype=dtype)
    arr['x'], arr['y'], arr['z'] = points[:, 0], points[:, 1], points[:, 2]
    arr['red'], arr['green'], arr['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(arr.tobytes())


def grid_faces(resolution: int, stride: int = 1):
    """
    Triangulate a `resolution x resolution` UV grid flattened as `[u_index, v_index]`.

    Returns:
        Tuple `(vertex_selection, faces)` where `vertex_selection` indexes into the
        flat `resolution**2` sample array and `faces` indexes into that selection.
        With `stride > 1` the grid is decimated, which is the cheap way to keep
        exported meshes manageable at high `--resolution`.
    """
    flat = np.arange(resolution * resolution).reshape(resolution, resolution)
    sel = flat[::stride, ::stride]
    h, w = sel.shape

    local = np.arange(h * w).reshape(h, w)
    a = local[:-1, :-1].ravel()
    b = local[1:, :-1].ravel()
    c = local[1:, 1:].ravel()
    d = local[:-1, 1:].ravel()

    faces = np.concatenate([np.stack([a, b, c], axis=1),
                            np.stack([a, c, d], axis=1)], axis=0)
    return sel.ravel(), faces.astype(np.int32)


def write_colored_mesh_ply(path: str, verts: np.ndarray, colors: np.ndarray,
                           faces: np.ndarray) -> None:
    """Write a binary little-endian vertex-colored triangle-mesh PLY."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    verts = np.asarray(verts, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    faces = np.asarray(faces, dtype=np.int32)

    vdtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    varr = np.empty(verts.shape[0], dtype=vdtype)
    varr['x'], varr['y'], varr['z'] = verts[:, 0], verts[:, 1], verts[:, 2]
    varr['red'], varr['green'], varr['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]

    # Every face is a triangle, so the list property has a fixed 3-element body
    # and can be written as a plain structured array.
    fdtype = np.dtype([('n', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')])
    farr = np.empty(faces.shape[0], dtype=fdtype)
    farr['n'] = 3
    farr['v0'], farr['v1'], farr['v2'] = faces[:, 0], faces[:, 1], faces[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {verts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(varr.tobytes())
        f.write(farr.tobytes())


def build_mask_meshes(xyz_fields: dict, masks: dict, patch_ids: list,
                      resolution: int, stride: int):
    """
    Assemble the per-patch UV grids into two meshes.

    Returns:
        Tuple `(verts, colors, faces_all, faces_white)`:
          * `faces_all`   — every triangle, vertex-colored by the mask. The
            segmentation painted onto the surface.
          * `faces_white` — only triangles whose three corners all have a
            correspondence, i.e. the surface with the hole regions removed.
    """
    sel, local_faces = grid_faces(resolution, stride)
    n_local = sel.shape[0]

    verts_parts, mask_parts, face_parts = [], [], []
    for offset_index, patch_id in enumerate(patch_ids):
        verts_parts.append(xyz_fields[patch_id][sel])
        mask_parts.append(masks[patch_id].ravel()[sel])
        face_parts.append(local_faces + offset_index * n_local)

    verts = np.concatenate(verts_parts, axis=0)
    vmask = np.concatenate(mask_parts, axis=0)
    faces_all = np.concatenate(face_parts, axis=0)

    colors = np.where(vmask[:, None],
                      np.array(COLOR_HAS_CORRESPONDENCE, dtype=np.uint8),
                      np.array(COLOR_NO_CORRESPONDENCE, dtype=np.uint8))

    # A triangle survives only if none of its corners is a hole sample.
    faces_white = faces_all[vmask[faces_all].all(axis=1)]

    return verts, colors, faces_all, faces_white


def compact_mesh(verts: np.ndarray, colors: np.ndarray, faces: np.ndarray):
    """Drop vertices no surviving face references, and reindex."""
    if faces.shape[0] == 0:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8),
                np.zeros((0, 3), np.int32))

    used = np.unique(faces)
    remap = np.full(verts.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    return verts[used], colors[used], remap[faces]


def sheet_layout(F, atlas_mode: str):
    """
    Group patch ids into per-sheet blocks for the contact-sheet figures.

    Returns:
        Tuple `(sheet_names, patches_per_sheet, n_rows, n_cols)`.
    """
    n_rows, n_cols = F.n_rows, F.n_cols
    patches_per_sheet = getattr(F, 'patches_per_side', F.n_patches)
    n_sheets = max(1, F.n_patches // patches_per_sheet)

    if atlas_mode == 'six_sheet':
        # Read the face names off the model itself rather than importing the
        # model package, which may not be on sys.path when run as a script.
        topology = getattr(getattr(F, 'complex', None), 'topology', None)
        face_names = getattr(topology, 'FACE_NAMES', ('+X', '-X', '+Y', '-Y', '+Z', '-Z'))
        names = list(face_names)[:n_sheets]
    elif atlas_mode == 'two_sheet':
        names = [f'side {i}' for i in range(n_sheets)]
    else:
        names = ['sheet']

    return names, patches_per_sheet, n_rows, n_cols


def save_contact_sheet(path: str, per_patch: dict, F, atlas_mode: str,
                       title: str, binary: bool, vmax: float = None) -> None:
    """Lay every patch out as one figure, grouped into per-sheet blocks."""
    names, patches_per_sheet, n_rows, n_cols = sheet_layout(F, atlas_mode)
    n_sheets = len(names)

    fig, axes = plt.subplots(
        n_rows, n_sheets * n_cols,
        figsize=(1.9 * n_sheets * n_cols + 1.0, 2.0 * n_rows + 1.0),
        squeeze=False,
    )

    for sheet in range(n_sheets):
        for row in range(n_rows):
            for col in range(n_cols):
                ax = axes[row][sheet * n_cols + col]
                patch_id = sheet * patches_per_sheet + row * n_cols + col
                field = per_patch.get(patch_id)

                if field is None:
                    ax.set_facecolor('0.85')
                else:
                    if binary:
                        ax.imshow(field.astype(np.float32), cmap='gray',
                                  vmin=0.0, vmax=1.0, interpolation='nearest')
                    else:
                        ax.imshow(field, cmap='inferno', vmin=0.0, vmax=vmax,
                                  interpolation='nearest')

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'p{patch_id}', fontsize=7, pad=2)

                if row == 0 and col == 0:
                    ax.text(0.0, 1.35, names[sheet], transform=ax.transAxes,
                            fontsize=11, fontweight='bold', ha='left')

    fig.suptitle(f'{title}   (each tile: u down, v right)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_diagnostics(path: str, distances: np.ndarray, tau: float,
                     d_nn: float, nn_scale: float) -> None:
    """
    Histogram of d(u,v) plus the black-fraction sweep.

    The histogram is the go/no-go gate: if it is not visibly bimodal, the fit is
    not accurate enough (or the hole is too small) for any threshold to separate
    membrane from surface.
    """
    safe = np.maximum(distances, 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.hist(safe, bins=400, color='#3b6ea5')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.axvline(tau, color='crimson', lw=2,
                label=f'tau = {tau:.5f}')
    ax1.axvline(d_nn, color='seagreen', lw=1.5, ls='--',
                label=f'median NN spacing = {d_nn:.5f}')
    ax1.set_xlabel('distance from F(u,v) to nearest target point')
    ax1.set_ylabel('# UV samples')
    ax1.set_title('Distance histogram (want two modes)')
    ax1.legend(fontsize=8)

    sweep = np.linspace(0.5, 20.0, 120) * d_nn
    frac = [(safe > t).mean() for t in sweep]
    ax2.plot(sweep / d_nn, np.array(frac) * 100.0, color='#3b6ea5')
    ax2.axvline(tau / d_nn, color='crimson', lw=2,
                label=f'tau = {tau / d_nn:.2f} x median_nn')
    ax2.set_xlabel('threshold (multiples of median NN spacing)')
    ax2.set_ylabel('% of UV area marked black')
    ax2.set_title('Black fraction vs threshold\n(a plateau here = a robust choice)')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# main
def main():
    parser = argparse.ArgumentParser(
        description='Segment each patch\'s UV domain into white (has a '
                    'correspondence in the target cloud) and black (no '
                    'correspondence -> assume hole).')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Trained multi-patch checkpoint (checkpoint_N.pt)')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Target point cloud. Defaults to the file recorded '
                             'in the checkpoint args.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: <ckpt dir>/uv_masks/<ckpt name>)')
    parser.add_argument('--resolution', type=int, default=256,
                        help='UV grid resolution per patch (per side)')
    parser.add_argument('--threshold', type=str, default='auto',
                        help="'auto' (nn_scale x median NN spacing), 'otsu', or "
                             'an explicit float in NORMALIZED units')
    parser.add_argument('--nn_scale', type=float, default=5.0,
                        help="Multiplier on the cloud's median NN spacing for --threshold auto")
    parser.add_argument('--target_max_points', type=int, default=-1,
                        help='Downsample the target cloud to this many points '
                             '(-1 keeps all; more points = more accurate distances)')
    parser.add_argument('--patch_ids', type=int, nargs='*', default=None,
                        help='Only process these patch ids (default: all)')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='UV samples per forward pass')
    parser.add_argument('--ply_max_points', type=int, default=400000,
                        help='Cap on points written to the 3D validation point cloud')
    parser.add_argument('--no_mesh', action='store_true',
                        help='Skip the mesh exports (point cloud only)')
    parser.add_argument('--mesh_stride', type=int, default=1,
                        help='Decimate the UV grid by this factor when building the '
                             'meshes. 1 keeps full resolution; 2 quarters the triangle '
                             'count. Raise it if the PLYs get unwieldy at high --resolution.')
    parser.add_argument('--unnormalize', action='store_true',
                        help='Write all 3D outputs in ORIGINAL input coordinates instead '
                             'of the normalized frame, so they overlay the raw input file.')
    parser.add_argument('--no_per_patch_png', action='store_true',
                        help='Skip the individual per-patch PNGs (keep only the contact sheets)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Explicit path to model/model.py if auto-discovery fails')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Model and checkpoint load
    print(f"\n  Loading checkpoint: {args.ckpt}")
    F, meta, ckpt_args, active_patch_ids, _ = patch_vis._load_model_from_checkpoint(
        args.ckpt, args.device, args.model_path
    )
    atlas_mode = ckpt_args.get('atlas_mode', 'single_sheet')
    print(f"  Atlas: {atlas_mode}  grid={F.n_rows}x{F.n_cols}  patches={F.n_patches}")

    center = np.asarray(meta['center'], dtype=np.float64)
    scale = float(meta['scale'])
    if scale == 1.0 and not np.any(center):
        print("  [warn] The checkpoint carries an identity normalization "
              "(center=0, scale=1). If it was trained on a normalized cloud, the "
              "target below will land in a different coordinate frame and "
              "EVERYTHING will come out black.")

    # Target point cloud in the model's normalized frame
    input_file = args.input_file or ckpt_args.get('file')
    if input_file is None:
        raise ValueError('No target cloud: pass --input_file (the checkpoint records none).')
    if not os.path.exists(input_file):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(repo_root, input_file)
        if os.path.exists(candidate):
            input_file = candidate

    print(f"\n  Loading target cloud: {input_file}")
    downsample_n = None if args.target_max_points < 0 else args.target_max_points
    # Reuse the checkpoint's center/scale so the cloud sits in the SAME frame the
    # model was trained in. Recomputing them per-file would shift it.
    target_pts, _ = utils.load_point_cloud(
        input_file, downsample_n=downsample_n, center=center, scale=scale
    )
    target_pts = target_pts.astype(np.float32)

    tree = cKDTree(target_pts)
    d_nn = median_nn_spacing(target_pts, tree)
    print(f"  Target median NN spacing: {d_nn:.6f} (normalized units)")

    # output layout
    if args.out_dir is None:
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                    'uv_masks', ckpt_name)
    mask_dir = os.path.join(args.out_dir, 'masks')
    dist_dir = os.path.join(args.out_dir, 'distance')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    # distance field per patch
    patch_ids = args.patch_ids if args.patch_ids else list(range(F.n_patches))
    R = args.resolution
    print(f"\n  Evaluating {len(patch_ids)} patches on a {R}x{R} UV grid "
          f"({len(patch_ids) * R * R:,} samples)...")

    dist_fields = {}
    xyz_fields = {}
    for patch_id in patch_ids:
        xyz, _uv = sample_patch_uv_grid(F, patch_id, R, args.device, args.batch_size)
        d, _ = tree.query(xyz, k=1, workers=-1)
        dist_fields[patch_id] = d.astype(np.float32).reshape(R, R)   # [u_idx, v_idx]
        xyz_fields[patch_id] = xyz
        print(f"    Patch {patch_id:02d}  d: mean={d.mean():.6f}  "
              f"median={np.median(d):.6f}  p99={np.quantile(d, 0.99):.6f}  max={d.max():.6f}")

    all_d = np.concatenate([f.ravel() for f in dist_fields.values()])

    # threshold
    tau, tau_desc = resolve_threshold(args.threshold, all_d, d_nn, args.nn_scale)

    print(f"\n{'─' * 64}")
    print(f"  Threshold tau = {tau:.6f}   [{tau_desc}]")
    print(f"{'─' * 64}")
    print("  Sweep (pick a value on a plateau):")
    print(f"    {'x median_nn':>12} {'tau':>12} {'% black':>10}")
    for mult in (1, 2, 3, 4, 5, 7, 10, 15, 20):
        t = mult * d_nn
        print(f"    {mult:>12} {t:>12.6f} {100.0 * (all_d > t).mean():>9.2f}%")

    # masks
    masks = {}
    per_patch_stats = {}
    for patch_id in patch_ids:
        # White (True) = has a correspondence. Black (False) = none -> hole.
        mask = dist_fields[patch_id] <= tau
        masks[patch_id] = mask

        black_frac = float(1.0 - mask.mean())
        per_patch_stats[int(patch_id)] = {
            'black_fraction': black_frac,
            'mean_distance': float(dist_fields[patch_id].mean()),
            'max_distance': float(dist_fields[patch_id].max()),
        }

        if not args.no_per_patch_png:
            Image.fromarray((mask.astype(np.uint8) * 255), mode='L').save(
                os.path.join(mask_dir, f'mask_patch_{patch_id:02d}.png'))
            plt.imsave(os.path.join(dist_dir, f'dist_patch_{patch_id:02d}.png'),
                       dist_fields[patch_id], cmap='inferno',
                       vmin=0.0, vmax=float(np.quantile(all_d, 0.99)))

    total_black = float((all_d > tau).mean())

    # Figures
    save_contact_sheet(os.path.join(args.out_dir, 'mask_atlas.png'), masks, F,
                       atlas_mode, 'White = has correspondence   |   Black = no correspondence',
                       binary=True)
    save_contact_sheet(os.path.join(args.out_dir, 'distance_atlas.png'), dist_fields, F,
                       atlas_mode, 'Distance from F(u,v) to nearest target point',
                       binary=False, vmax=float(np.quantile(all_d, 0.99)))
    save_diagnostics(os.path.join(args.out_dir, 'distance_histogram.png'),
                     all_d, tau, d_nn, args.nn_scale)

    # 3D outputs
    def to_output_frame(points):
        """Normalized -> original coordinates, when --unnormalize is set."""
        return points * scale + center if args.unnormalize else points

    frame_note = 'original input coordinates' if args.unnormalize else 'normalized [-1,1] frame'
    print(f"\n  Writing 3D outputs in the {frame_note}.")

    all_xyz = np.concatenate([xyz_fields[p] for p in patch_ids], axis=0)
    all_mask = np.concatenate([masks[p].ravel() for p in patch_ids], axis=0)

    if all_xyz.shape[0] > args.ply_max_points:
        rng = np.random.default_rng(0)
        sub = rng.choice(all_xyz.shape[0], args.ply_max_points, replace=False)
        ply_xyz, ply_mask = all_xyz[sub], all_mask[sub]
    else:
        ply_xyz, ply_mask = all_xyz, all_mask

    colors = np.where(ply_mask[:, None],
                      np.array(COLOR_HAS_CORRESPONDENCE, dtype=np.uint8),
                      np.array(COLOR_NO_CORRESPONDENCE, dtype=np.uint8))
    write_colored_ply(os.path.join(args.out_dir, 'uv_samples_classified.ply'),
                      to_output_frame(ply_xyz), colors)

    mesh_stats = None
    if not args.no_mesh:
        if args.mesh_stride < 1:
            raise ValueError(f"--mesh_stride must be >= 1, got {args.mesh_stride}")

        mverts, mcolors, faces_all, faces_white = build_mask_meshes(
            xyz_fields, masks, patch_ids, R, args.mesh_stride)

        write_colored_mesh_ply(os.path.join(args.out_dir, 'uv_mask_mesh.ply'),
                               to_output_frame(mverts), mcolors, faces_all)

        # The surface with the no-correspondence regions removed. This is a
        # preview of the trimming step, not the finished article: the opening
        # follows UV grid cells, so its rim is a staircase at --resolution. A
        # clean rim needs marching squares on the distance field.
        tverts, tcolors, tfaces = compact_mesh(mverts, mcolors, faces_white)
        write_colored_mesh_ply(os.path.join(args.out_dir, 'uv_mask_trimmed_mesh.ply'),
                               to_output_frame(tverts), tcolors, tfaces)

        dropped = faces_all.shape[0] - faces_white.shape[0]
        mesh_stats = {
            'mesh_stride': int(args.mesh_stride),
            'vertices': int(mverts.shape[0]),
            'faces_total': int(faces_all.shape[0]),
            'faces_kept': int(faces_white.shape[0]),
            'faces_dropped': int(dropped),
        }
        print(f"    Mesh: {mverts.shape[0]:,} verts, {faces_all.shape[0]:,} tris "
              f"(stride={args.mesh_stride}); trimmed drops {dropped:,} tris "
              f"({100.0 * dropped / max(faces_all.shape[0], 1):.2f}%)")

    # output NPZ for the trimming step
    npz_path = os.path.join(args.out_dir, 'uv_mask.npz')
    np.savez_compressed(
        npz_path,
        patch_ids=np.asarray(patch_ids, dtype=np.int32),
        masks=np.stack([masks[p] for p in patch_ids], axis=0),
        distances=np.stack([dist_fields[p] for p in patch_ids], axis=0),
        threshold=np.float32(tau),
        resolution=np.int32(R),
        median_nn=np.float32(d_nn),
    )

    summary = {
        'checkpoint': os.path.abspath(args.ckpt),
        'input_file': os.path.abspath(input_file),
        'atlas_mode': atlas_mode,
        'n_patches': int(F.n_patches),
        'grid_dims': [int(F.n_rows), int(F.n_cols)],
        'resolution': int(R),
        'normalization': {'center': center.tolist(), 'scale': scale},
        'median_nn_spacing': d_nn,
        'threshold': float(tau),
        'threshold_mode': args.threshold,
        'threshold_description': tau_desc,
        'total_black_fraction': total_black,
        'mask_index_convention': 'mask[u_index, v_index]; True = white = has correspondence',
        'export_frame': 'original' if args.unnormalize else 'normalized',
        'mesh': mesh_stats,
        'per_patch': per_patch_stats,
    }
    summary_path = os.path.join(args.out_dir, 'mask_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


    print(f"\n{'─' * 64}")
    print(f"  Black (no correspondence) overall: {100.0 * total_black:.2f}% of UV area")
    fully_white = [p for p in patch_ids if per_patch_stats[p]['black_fraction'] < 1e-6]
    fully_black = [p for p in patch_ids if per_patch_stats[p]['black_fraction'] > 1.0 - 1e-6]
    print(f"  Patches fully white: {len(fully_white)}/{len(patch_ids)}")
    print(f"  Patches fully black: {len(fully_black)}/{len(patch_ids)}"
          + (f"  -> {fully_black}" if fully_black else ""))
    print("  Per-patch black fraction:")
    for patch_id in patch_ids:
        frac = per_patch_stats[patch_id]['black_fraction']
        bar = '#' * int(round(frac * 40))
        print(f"    p{patch_id:02d}  {100.0 * frac:6.2f}%  {bar}")

    if total_black > 0.60:
        print("\n  [warn] Over 60% of the UV area is black. That is almost never a real")
        print("         hole — check that --input_file matches the training cloud and")
        print("         that the checkpoint's normalization is the one it was trained with.")
    elif total_black < 1e-6:
        print("\n  [warn] Nothing was marked black. Either the target really is genus 0,")
        print("         or tau is too loose — lower --nn_scale and re-run.")

    print(f"\n  Outputs → {args.out_dir}")
    print(f"    mask_atlas.png            — all patch masks at a glance (START HERE)")
    print(f"    distance_atlas.png        — the underlying distance field")
    print(f"    distance_histogram.png    — bimodality check + threshold sweep")
    print(f"    uv_samples_classified.ply — 3D points: grey = correspondence, RED = none")
    if mesh_stats is not None:
        print(f"    uv_mask_mesh.ply          — same classification as a MESH")
        print(f"    uv_mask_trimmed_mesh.ply  — mesh with the black regions removed (preview)")
    print(f"    masks/, distance/         — per-patch PNGs")
    print(f"    uv_mask.npz               — masks + distances for the trimming step")
    print(f"    mask_summary.json         — threshold and per-patch stats")
    print(f"{'─' * 64}\n")


if __name__ == '__main__':
    main()
