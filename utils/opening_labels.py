#!/usr/bin/env python3
# utils/opening_labels.py

import argparse
import colorsys
import json
import os

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

try:
    from . import patch_vis
    from .uv_hole_mask import (adaptive_leaf_rects, composite_face_canvases,
                               grid_faces, sample_patch_uv_grid, sheet_layout,
                               write_colored_mesh_ply)
except ImportError:
    import patch_vis
    from uv_hole_mask import (adaptive_leaf_rects, composite_face_canvases,
                              grid_faces, sample_patch_uv_grid, sheet_layout,
                              write_colored_mesh_ply)

from model.model import FACE_NAMES  # noqa: E402


COLOR_NO_OPENING = (200, 200, 200)   # surface that is not a hole


# ── colours ─────────────────────────────────────────────────────────────────
def opening_colors(n: int) -> np.ndarray:
    """
    `n` visually distinct RGB colours.

    Hues advance by the golden angle so neighbouring IDs never look alike, and
    saturation/value cycle so the sequence stays separable well past the ~20
    colours a categorical colormap can offer.
    """
    cols = []
    for k in range(max(n, 1)):
        h = (k * 0.6180339887498949) % 1.0
        s = 0.55 + 0.30 * ((k % 3) / 2.0)
        v = 0.95 - 0.22 * (k % 2)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.asarray(cols, dtype=np.uint8)


# ── welding ─────────────────────────────────────────────────────────────────
def boundary_indices(resolution: int) -> np.ndarray:
    """Flat indices of a patch grid's four boundary curves."""
    flat = np.arange(resolution * resolution).reshape(resolution, resolution)
    return np.unique(np.concatenate([flat[0, :], flat[-1, :],
                                     flat[:, 0], flat[:, -1]]))


def weld_boundary_vertices(xyz_fields: dict, patch_ids: list, resolution: int,
                           tol: float):
    """
    Union coincident boundary vertices across patches.

    Only boundary vertices are considered. Interior vertices of different
    patches CAN come arbitrarily close where a membrane pinches, and welding
    those would fuse two genuinely separate openings into one ID.

    Returns:
        Tuple `(pairs, stats)` where `pairs` is (m, 2) of GLOBAL vertex indices
        (patch_position * resolution**2 + flat_index) to be unioned.
    """
    border = boundary_indices(resolution)
    n_per = resolution * resolution

    pts, gidx = [], []
    for pos, patch_id in enumerate(patch_ids):
        pts.append(xyz_fields[patch_id][border])
        gidx.append(pos * n_per + border)
    pts = np.concatenate(pts, axis=0).astype(np.float64)
    gidx = np.concatenate(gidx, axis=0)
    owner = np.repeat(np.arange(len(patch_ids)), border.shape[0])

    tree = cKDTree(pts)
    raw = tree.query_pairs(tol, output_type='ndarray')
    # Same-patch pairs are the grid's own corners meeting; they carry no
    # cross-patch information and are already connected by grid adjacency.
    if raw.shape[0]:
        raw = raw[owner[raw[:, 0]] != owner[raw[:, 1]]]

    pairs = gidx[raw] if raw.shape[0] else np.zeros((0, 2), np.int64)

    # Per-patch-edge coverage. A seam with no welded vertices is a seam the
    # connectivity below cannot cross, so an opening spanning it would be split
    # into two IDs. Worth reporting rather than discovering later.
    flat = np.arange(n_per).reshape(resolution, resolution)
    named = {'top': flat[0, :], 'bottom': flat[-1, :],
             'left': flat[:, 0], 'right': flat[:, -1]}
    weak = []
    for pos, patch_id in enumerate(patch_ids):
        for edge_name, idx in named.items():
            q = xyz_fields[patch_id][idx].astype(np.float64)
            near = tree.query_ball_point(q, tol, workers=-1)
            frac = float(np.mean([any(owner[j] != pos for j in c) for c in near]))
            if frac < 0.5:
                weak.append((int(patch_id), edge_name, frac))

    return pairs, {'n_pairs': int(pairs.shape[0]),
                   'n_boundary_vertices': int(pts.shape[0]),
                   'weak_edges': weak}


# ── connectivity ────────────────────────────────────────────────────────────
def grid_neighbour_pairs(resolution: int) -> np.ndarray:
    """
    Within-patch vertex adjacency, matching the triangulation.

    `grid_faces` emits triangles (a,b,c) and (a,c,d) over the cell corners
    a=(u,v) b=(u+1,v) c=(u+1,v+1) d=(u,v+1), so the mesh actually connects
    vertical, horizontal AND the (u+1, v+1) diagonal steps.
    """
    flat = np.arange(resolution * resolution).reshape(resolution, resolution)
    vertical = np.stack([flat[:-1, :].ravel(), flat[1:, :].ravel()], axis=1)
    horizontal = np.stack([flat[:, :-1].ravel(), flat[:, 1:].ravel()], axis=1)
    diagonal = np.stack([flat[:-1, :-1].ravel(), flat[1:, 1:].ravel()], axis=1)
    return np.concatenate([vertical, horizontal, diagonal], axis=0)


def label_openings(masks: dict, xyz_fields: dict, patch_ids: list,
                   resolution: int, weld_pairs: np.ndarray,
                   min_vertices: int):
    """
    Connected components of the black region across the whole welded atlas.

    Only BLACK vertices become graph nodes, so a white rim vertex can never
    bridge two openings that happen to pass within one triangle of each other.

    Returns:
        Tuple `(opening_id, n_openings, dropped)` where `opening_id` is
        (n_patches, resolution, resolution) int32, -1 outside any opening, and
        IDs are assigned largest-opening-first.
    """
    n_per = resolution * resolution
    n_total = len(patch_ids) * n_per

    black = np.concatenate([~masks[p].ravel() for p in patch_ids])   # (n_total,)

    # Within-patch edges, offset into the global index space.
    local = grid_neighbour_pairs(resolution)
    per_patch = [local + pos * n_per for pos in range(len(patch_ids))]
    edges = np.concatenate(per_patch + [weld_pairs], axis=0) if weld_pairs.shape[0] \
        else np.concatenate(per_patch, axis=0)

    # Keep only edges whose both endpoints are black.
    edges = edges[black[edges[:, 0]] & black[edges[:, 1]]]

    graph = coo_matrix(
        (np.ones(edges.shape[0], np.int8), (edges[:, 0], edges[:, 1])),
        shape=(n_total, n_total))
    n_comp, comp = connected_components(graph, directed=False)

    # connected_components labels every node, including all the white ones that
    # are isolated singletons. Only black nodes count.
    sizes = np.bincount(comp[black], minlength=n_comp)
    keep = np.flatnonzero(sizes >= min_vertices)
    dropped = int((sizes >= 1).sum() - keep.size)

    # Largest opening becomes ID 0, so IDs are stable and meaningful to read.
    keep = keep[np.argsort(-sizes[keep])]
    remap = np.full(n_comp, -1, np.int32)
    remap[keep] = np.arange(keep.size, dtype=np.int32)

    flat_id = np.where(black, remap[comp], -1).astype(np.int32)
    return (flat_id.reshape(len(patch_ids), resolution, resolution),
            int(keep.size), dropped)


# figures
def label_cmap(colors: np.ndarray):
    """Discrete colormap over opening IDs; NaN (not a hole) renders light grey."""
    cmap = ListedColormap(colors.astype(np.float64) / 255.0)
    cmap.set_bad(color='0.90')
    return cmap


def save_opening_atlas_fixed(path, opening_id, patch_ids, F, atlas_mode, colors):
    """Contact sheet for a fixed patch grid, one tile per patch."""
    names, patches_per_sheet, n_rows, n_cols = sheet_layout(F, atlas_mode)
    n_sheets = len(names)
    pos_of = {p: i for i, p in enumerate(patch_ids)}
    cmap = label_cmap(colors)
    n = max(len(colors), 1)

    fig, axes = plt.subplots(n_rows, n_sheets * n_cols,
                             figsize=(1.9 * n_sheets * n_cols + 1.0,
                                      2.0 * n_rows + 1.0), squeeze=False)
    for sheet in range(n_sheets):
        for row in range(n_rows):
            for col in range(n_cols):
                ax = axes[row][sheet * n_cols + col]
                pid = sheet * patches_per_sheet + row * n_cols + col
                ax.set_xticks([]); ax.set_yticks([])
                if pid not in pos_of:
                    ax.set_facecolor('0.85')
                    continue
                field = opening_id[pos_of[pid]].astype(np.float64)
                ax.imshow(np.ma.masked_less(field, 0), cmap=cmap,
                          vmin=-0.5, vmax=n - 0.5, interpolation='nearest')
                # Cap the ID list: a noisy mask can put dozens of openings in
                # one patch, and the full list overruns the neighbouring tiles.
                present = sorted(set(np.unique(field[field >= 0]).astype(int)))
                if not present:
                    label = f'p{pid}'
                elif len(present) <= 4:
                    label = f'p{pid} ' + ','.join(str(o) for o in present)
                else:
                    label = (f'p{pid} ' + ','.join(str(o) for o in present[:4])
                             + f' +{len(present) - 4}')
                ax.set_title(label, fontsize=6, pad=2)
                if row == 0 and col == 0:
                    ax.text(0.0, 1.35, names[sheet], transform=ax.transAxes,
                            fontsize=11, fontweight='bold', ha='left')

    fig.suptitle('Opening ID per patch   (grey = not a hole; u down, v right)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_opening_atlas_adaptive(path, opening_id, patch_ids, F, colors, face_res):
    """Per-face canvas for the adaptive atlas, leaves in their quadtree rects."""
    rects, faces, _ = adaptive_leaf_rects(F)
    fields = {p: np.where(opening_id[i] < 0, np.nan,
                          opening_id[i].astype(np.float32))
              for i, p in enumerate(patch_ids)}
    canvases = composite_face_canvases(fields, patch_ids, rects, faces, face_res)

    cmap = label_cmap(colors)
    n = max(len(colors), 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), squeeze=False)
    for f in range(len(FACE_NAMES)):
        ax = axes[f // 3][f % 3]
        ax.imshow(np.ma.masked_invalid(canvases[f]), cmap=cmap,
                  vmin=-0.5, vmax=n - 0.5, interpolation='nearest',
                  extent=[0, 1, 1, 0])
        for i, p in enumerate(patch_ids):
            if int(faces[p]) != f:
                continue
            u0, v0, size = rects[p]
            ax.add_patch(Rectangle((v0, u0), size, size, fill=False,
                                   edgecolor='#2f7fd0', linewidth=0.4))
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'face {FACE_NAMES[f]}', fontsize=10)

    fig.suptitle('Opening ID per cube face   (grey = not a hole; u down, v right)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)

# meshes
def build_opening_meshes(xyz_fields, opening_id, patch_ids, resolution, stride,
                         colors, to_frame):
    """
    Two vertex-coloured meshes.

    Returns:
        Tuple `(hole_mesh, full_mesh)`, each `(verts, colors, faces)`. A triangle
        joins an opening when ANY corner carries that ID, so the coloured region
        includes the rim and the two meshes line up exactly.
    """
    sel, local_faces = grid_faces(resolution, stride)
    n_local = sel.shape[0]

    verts, ids, faces = [], [], []
    for pos, patch_id in enumerate(patch_ids):
        verts.append(xyz_fields[patch_id][sel])
        ids.append(opening_id[pos].ravel()[sel])
        faces.append(local_faces + pos * n_local)

    verts = np.concatenate(verts, axis=0)
    vids = np.concatenate(ids, axis=0)
    faces = np.concatenate(faces, axis=0)

    vcol = np.where(vids[:, None] >= 0,
                    colors[np.clip(vids, 0, max(len(colors) - 1, 0))],
                    np.array(COLOR_NO_OPENING, np.uint8))

    face_ids = vids[faces]
    hole_faces = faces[(face_ids >= 0).any(axis=1)]

    used = np.unique(hole_faces) if hole_faces.size else np.zeros(0, np.int64)
    remap = np.full(verts.shape[0], -1, np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)

    hole_mesh = (to_frame(verts[used]), vcol[used],
                 remap[hole_faces] if hole_faces.size else np.zeros((0, 3), np.int32))
    full_mesh = (to_frame(verts), vcol, faces)
    return hole_mesh, full_mesh


def main():
    ap = argparse.ArgumentParser(
        description='Group uv_hole_mask.py\'s black region into separate '
                    'openings, each with its own ID and colour.')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--mask_npz', type=str, default=None,
                    help='uv_mask.npz (default: <ckpt dir>/uv_masks/<ckpt name>/uv_mask.npz)')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='Default: <mask_npz dir>/openings')
    ap.add_argument('--adaptive', type=str, default='auto',
                    choices=['auto', 'yes', 'no'])
    ap.add_argument('--weld_tol', type=float, default=1e-5,
                    help='Boundary vertices closer than this are the same point. '
                         'Must stay far below the gap between the two sheets of a '
                         'membrane, or two openings get fused into one ID.')
    ap.add_argument('--min_opening_vertices', type=int, default=10,
                    help='Drop components smaller than this many black vertices '
                         '(mask speckle). Set 1 to keep everything.')
    ap.add_argument('--mesh_stride', type=int, default=1,
                    help='Decimate the UV grid for the exported meshes only')
    ap.add_argument('--no_mesh', action='store_true')
    ap.add_argument('--normalized', action='store_true',
                    help='Write meshes in the model\'s normalized [-1,1] frame '
                         'instead of the default ORIGINAL input coordinates. Use '
                         'this to overlay them on patch_intersection.py\'s default '
                         '(normalized) output.')
    ap.add_argument('--face_resolution', type=int, default=512,
                    help='[Adaptive] canvas resolution per cube face in the figure')
    ap.add_argument('--batch_size', type=int, default=8192)
    ap.add_argument('--model_path', type=str, default=None)
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # ── inputs ──────────────────────────────────────────────────────────────
    if args.mask_npz is None:
        stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.mask_npz = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                     'uv_masks', stem, 'uv_mask.npz')
    if not os.path.exists(args.mask_npz):
        raise FileNotFoundError(
            f"No uv_mask.npz at {args.mask_npz}. Run utils/uv_hole_mask.py first.")
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(args.mask_npz)),
                                    'openings')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n  Loading checkpoint: {args.ckpt}")
    if args.adaptive == 'auto':
        is_adaptive = patch_vis._is_adaptive_checkpoint(args.ckpt)
    else:
        is_adaptive = args.adaptive == 'yes'

    if is_adaptive:
        F, meta, ckpt_args, _, _ = patch_vis._load_adaptive_from_checkpoint(
            args.ckpt, args.device)
        atlas_mode = 'adaptive_cube'
        print(f"  Atlas: adaptive quadtree cube, {F.n_patches} leaves")
    else:
        F, meta, ckpt_args, _, _ = patch_vis._load_model_from_checkpoint(
            args.ckpt, args.device, args.model_path)
        atlas_mode = ckpt_args.get('atlas_mode', 'single_sheet')
        print(f"  Atlas: {atlas_mode}, {F.n_patches} patches")

    mask_data = np.load(args.mask_npz)
    patch_ids = [int(p) for p in mask_data['patch_ids']]
    R = int(mask_data['resolution'])
    masks = {p: mask_data['masks'][i] for i, p in enumerate(patch_ids)}
    n_black_total = int(sum((~masks[p]).sum() for p in patch_ids))
    print(f"  Mask: {len(patch_ids)} patches at {R}x{R}, "
          f"tau={float(mask_data['threshold']):.6f}, "
          f"{n_black_total:,} black vertices")

    if n_black_total == 0:
        print("\n  Nothing is segmented as a hole, so there are no openings to "
              "label. Loosen uv_hole_mask.py's --nn_scale and re-run.\n")
        return

    center = np.asarray(meta['center'], np.float64)
    scale = float(meta['scale'])

    # sample
    print(f"\n  Sampling {len(patch_ids)} patches at {R}x{R}...")
    xyz_fields = {p: sample_patch_uv_grid(F, p, R, args.device, args.batch_size)[0]
                  for p in patch_ids}

    # weld + label
    print(f"  Welding patch boundaries (tol={args.weld_tol:g})...")
    weld_pairs, weld_stats = weld_boundary_vertices(
        xyz_fields, patch_ids, R, args.weld_tol)
    print(f"    {weld_stats['n_pairs']:,} cross-patch vertex pairs welded "
          f"from {weld_stats['n_boundary_vertices']:,} boundary vertices")
    if weld_stats['weak_edges']:
        print(f"    [warn] {len(weld_stats['weak_edges'])} patch edge(s) weld "
              f"under 50%. An opening crossing those seams will be SPLIT into "
              f"separate IDs. Raise --weld_tol if these are T-junctions:")
        for pid, edge_name, frac in weld_stats['weak_edges'][:12]:
            print(f"        p{pid:03d} {edge_name:<7} {100 * frac:5.1f}%")
    else:
        print(f"    all patch edges weld cleanly")

    opening_id, n_openings, dropped = label_openings(
        masks, xyz_fields, patch_ids, R, weld_pairs, args.min_opening_vertices)
    colors = opening_colors(n_openings)

    # per opening summary
    openings = []
    for oid in range(n_openings):
        pts, patches = [], []
        for pos, patch_id in enumerate(patch_ids):
            hit = opening_id[pos] == oid
            if hit.any():
                patches.append(int(patch_id))
                pts.append(xyz_fields[patch_id].reshape(R, R, 3)[hit])
        pts = np.concatenate(pts, axis=0)
        openings.append({
            'id': oid,
            'n_vertices': int(pts.shape[0]),
            'n_patches': len(patches),
            'patches': patches,
            'color': [int(c) for c in colors[oid]],
            'centroid': pts.mean(axis=0).tolist(),
            'bbox_min': pts.min(axis=0).tolist(),
            'bbox_max': pts.max(axis=0).tolist(),
            'extent': (pts.max(axis=0) - pts.min(axis=0)).tolist(),
        })

    # figures in UV 
    atlas_path = os.path.join(args.out_dir, 'opening_atlas.png')
    if is_adaptive:
        save_opening_atlas_adaptive(atlas_path, opening_id, patch_ids, F,
                                    colors, args.face_resolution)
    else:
        save_opening_atlas_fixed(atlas_path, opening_id, patch_ids, F,
                                 atlas_mode, colors)

    # meshes
    # The model works in a normalized [-1,1] frame; p_orig = p_norm * scale + center
    # recovers the input file's own coordinates, which is the default here so the
    # meshes overlay the raw input cloud directly in a viewer.
    export_frame = 'normalized' if args.normalized else 'original'
    if not args.normalized and scale == 1.0 and not np.any(center):
        print("  [warn] The checkpoint stores an identity normalization "
              "(center=0, scale=1), so 'original' and 'normalized' coordinates "
              "are the same here.")

    def to_frame(p):
        return p if args.normalized else p * scale + center

    mesh_stats = None
    if not args.no_mesh:
        (hv, hc, hf), (fv, fc, ff) = build_opening_meshes(
            xyz_fields, opening_id, patch_ids, R, args.mesh_stride, colors, to_frame)
        write_colored_mesh_ply(os.path.join(args.out_dir, 'openings_mesh.ply'),
                               hv, hc, hf)
        write_colored_mesh_ply(
            os.path.join(args.out_dir, 'surface_with_openings.ply'), fv, fc, ff)
        mesh_stats = {'hole_vertices': int(hv.shape[0]),
                      'hole_faces': int(hf.shape[0]),
                      'full_faces': int(ff.shape[0]),
                      'mesh_stride': int(args.mesh_stride),
                      'frame': export_frame}
        print(f"\n  Opening mesh: {hv.shape[0]:,} verts / {hf.shape[0]:,} tris "
              f"({export_frame} coordinates)")

    # save
    np.savez_compressed(
        os.path.join(args.out_dir, 'openings.npz'),
        patch_ids=np.asarray(patch_ids, np.int32),
        opening_id=opening_id,
        n_openings=np.int32(n_openings),
        resolution=np.int32(R),
        colors=colors,
        opening_sizes=np.array([o['n_vertices'] for o in openings], np.int64),
        threshold=mask_data['threshold'],
        normalization_center=center,
        normalization_scale=np.float64(scale),
        adaptive=np.bool_(is_adaptive),
    )

    summary = {
        'checkpoint': os.path.abspath(args.ckpt),
        'mask_npz': os.path.abspath(args.mask_npz),
        'adaptive': bool(is_adaptive),
        'resolution': R,
        'weld_tol': args.weld_tol,
        'weld_pairs': weld_stats['n_pairs'],
        'weak_seam_edges': [{'patch': p, 'edge': e, 'welded_fraction': f}
                            for p, e, f in weld_stats['weak_edges']],
        'min_opening_vertices': args.min_opening_vertices,
        'n_openings': n_openings,
        'n_dropped_components': dropped,
        'opening_id_convention':
            'opening_id[patch_position, u_index, v_index]; -1 = not a hole; '
            'patch_position indexes patch_ids, IDs sorted largest first',
        'export_frame': export_frame,
        'mesh': mesh_stats,
        'openings': openings,
    }
    with open(os.path.join(args.out_dir, 'openings.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # report
    print(f"\n{'─' * 68}")
    print(f"  {n_openings} opening(s) found"
          + (f"   ({dropped} smaller component(s) dropped below "
             f"{args.min_opening_vertices} vertices)" if dropped else ""))
    if n_openings:
        print(f"\n    {'ID':>3}  {'vertices':>9}  {'patches':>7}  "
              f"{'extent (x,y,z)':>26}   patches")
        for o in openings:
            ext = '  '.join(f'{e:.3f}' for e in o['extent'])
            plist = str(o['patches'][:10]) + ('...' if o['n_patches'] > 10 else '')
            print(f"    {o['id']:>3}  {o['n_vertices']:>9,}  {o['n_patches']:>7}  "
                  f"{ext:>26}   {plist}")

    # An opening is one of two kinds, and only patch_intersection.py can tell
    # them apart:
    #   handle   - the atlas had to push a membrane THROUGH a hole (torus);
    #              two funnels meet and CROSS. Handles come in crossing pairs,
    #              2 per unit of genus, and are what stitching joins.
    #   boundary - a real edge of an open surface (a dress's neck, sleeves,
    #              hem). One membrane caps it; nothing crosses. Not stitched.
    if n_openings:
        print(f"\n  [note] {n_openings} opening(s). Which are HANDLES (come in "
              f"crossing pairs, 2 per unit of genus) and which are BOUNDARY "
              f"openings (a real edge, cross nothing, not stitched) is decided by "
              f"patch_intersection.py's pairing check, not by the count alone.")

    print(f"\n  Outputs → {args.out_dir}")
    print(f"    opening_atlas.png          — opening IDs in UV (START HERE)")
    if mesh_stats:
        print(f"    openings_mesh.ply          — the openings alone, one colour each")
        print(f"    surface_with_openings.ply  — whole surface, openings coloured")
    print(f"    openings.npz               — opening_id per UV sample")
    print(f"    openings.json              — per-opening stats")
    print(f"{'─' * 68}\n")


if __name__ == '__main__':
    main()
