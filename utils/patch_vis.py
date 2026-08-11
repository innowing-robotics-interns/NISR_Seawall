#!/usr/bin/env python3
# patch_vis.py

import os
import argparse
import importlib.util
import glob
import numpy as np
import torch
import trimesh
from PIL import Image

import pc_presegmentation as pc_presegmentation
import utils as utils

# Load the texture
def load_checkerboard_textures(texture_path, pattern="Slide5.jpg", n_images=1):
    """
    Load checkerboard textures from a file or directory.

    Change pattern to "Slide{}.jpg" if you want to load multiple images from a directory.   

    Args:
        texture_path: Path to one image file or a directory of images.
        pattern: Filename pattern used in directory mode.
        n_images: Number of images to load in directory mode.

    Returns:
        List of `PIL.Image` objects.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(texture_path):
        candidates = [
            texture_path,
            os.path.join(script_dir, texture_path),
            os.path.join(os.path.dirname(script_dir), texture_path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                texture_path = candidate
                break

    if os.path.isfile(texture_path):
        img = Image.open(texture_path).convert("RGB")
        print(f"  Using ONE checkerboard texture for all patches: {texture_path}")
        return [img]

    if os.path.isdir(texture_path):
        textures = []
        for i in range(1, n_images + 1):
            p = os.path.join(texture_path, pattern.format(i))
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing checkerboard texture: {p}")
            textures.append(Image.open(p).convert("RGB"))
        print(f"  Loaded {len(textures)} checkerboard texture(s) from {texture_path}")
        return textures

    raise FileNotFoundError(
        f"Checkerboard texture path not found: {texture_path}\n"
        f"  Pass either a single image file or a directory of numbered images."
    )



# Per-patch grid sampling.
@torch.no_grad()
def _sample_patch_grid(F, patch_idx, resolution, device):
    """
    Sample one patch of `MultiPatchForwardMap` on a regular UV grid.

    Args:
        F: Forward map in eval mode.
        patch_idx: Patch index.
        resolution: Grid resolution per side.
        device: Torch device.

    Returns:
        Tuple `(verts, uv, faces)`.
    """
    u = torch.linspace(0, 1, resolution, device=device)
    v = torch.linspace(0, 1, resolution, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
    uv = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)  # (res*res, 2)

    batch_size = 4096
    verts_list = []
    for i in range(0, uv.shape[0], batch_size):
        verts_list.append(F(patch_idx, uv[i:i + batch_size]).cpu())
    verts = torch.cat(verts_list, dim=0).numpy()

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

    return verts.astype(np.float32), uv.cpu().numpy().astype(np.float32), faces


def _bake_vertex_colors(mesh):
    """
    Convert a textured mesh into a vertex-colored copy for PLY export.
    """
    colored = mesh.copy()
    colored.visual = mesh.visual.to_color()
    return colored


def _make_double_sided(verts, uv, faces):
    """
    Duplicate sheet geometry so it renders from both sides.

    Args:
        verts: Vertex positions.
        uv: UV coordinates matching `verts`.
        faces: Triangle indices.

    Returns:
        Tuple `(verts2, uv2, faces2)`.
    """
    n = verts.shape[0]
    verts2 = np.concatenate([verts, verts], axis=0)
    uv2 = np.concatenate([uv, uv], axis=0)
    faces_front = faces
    faces_back = faces[:, [0, 2, 1]] + n  # reversed winding -> opposite normal
    faces2 = np.concatenate([faces_front, faces_back], axis=0).astype(np.int32)
    return verts2, uv2, faces2


def _build_patch_uv_occupancy(points, subdivision_depth: int, min_points: int = 1):
    """Build an occupancy mask over a patch-local UV domain [0,1]^2."""
    if points is None or len(points) < min_points:
        return np.zeros((1, 1), dtype=bool)

    final_res = 1 << max(0, subdivision_depth)
    occ = np.zeros((final_res, final_res), dtype=bool)

    if subdivision_depth == 0:
        occ[0, 0] = len(points) >= min_points
        return occ

    cells = [(points, 0, 0, final_res)]

    for depth in range(subdivision_depth):
        next_cells = []
        for cell_points, row0, col0, size in cells:
            if len(cell_points) < min_points:
                continue
            if size <= 1:
                next_cells.append((cell_points, row0, col0, size))
                continue
            assignments, _, _ = pc_presegmentation.pca_grid_segmentation(
                cell_points, n_patches_u=2, n_patches_v=2
            )
            half = size // 2
            for sub_id in range(4):
                sub_points = cell_points[assignments == sub_id]
                if len(sub_points) < min_points:
                    continue
                sub_r = sub_id // 2
                sub_c = sub_id % 2
                next_cells.append((sub_points, row0 + sub_r * half, col0 + sub_c * half, half))
        cells = next_cells
        if not cells:
            break

    for _, row0, col0, size in cells:
        occ[row0:row0 + size, col0:col0 + size] = True
    return occ


def _filter_patch_mesh_by_occupancy(verts, uv, faces, occ_mask):
    """Keep only vertices/faces whose UVs lie in occupied cells."""
    if occ_mask.size == 0 or not occ_mask.any():
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32))

    res_u, res_v = occ_mask.shape
    u_idx = np.clip((uv[:, 0] * res_u).astype(int), 0, res_u - 1)
    v_idx = np.clip((uv[:, 1] * res_v).astype(int), 0, res_v - 1)
    keep_v = occ_mask[u_idx, v_idx]

    kept_indices = np.flatnonzero(keep_v)
    if kept_indices.size == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32))

    index_map = -np.ones(len(verts), dtype=np.int32)
    index_map[kept_indices] = np.arange(len(kept_indices), dtype=np.int32)
    keep_f = keep_v[faces].all(axis=1)
    faces_kept = index_map[faces[keep_f]]
    return verts[kept_indices], uv[kept_indices], faces_kept



# Export functions
def export_checkerboard_patches(F, meta, save_dir, texture_path,
                                 resolution=100, device='cuda',
                                 epoch='10k', name=None,
                                 texture_pattern="Slide5.jpg", n_images=1,
                                 unnormalize=True, debug_uv_png=True,
                                 export_ply=True, double_sided=True,
                                 active_patch_ids=None,
                                 patch_points_by_id=None,
                                 subdivision_depth=1,
                                 min_points_per_cell=1):
    """
    Export checkerboard-textured patch meshes and a combined scene.
    """
    os.makedirs(save_dir, exist_ok=True)
    textures = load_checkerboard_textures(texture_path, pattern=texture_pattern,
                                          n_images=n_images)
    n_textures = len(textures)

    sample_all_patches = subdivision_depth < 0

    if active_patch_ids is None or sample_all_patches:
        patch_ids = list(range(F.n_patches))
    else:
        patch_ids = [int(patch_idx) for patch_idx in active_patch_ids]

    if len(patch_ids) == 0:
        print("  No active patches found; skipping checkerboard export.")
        return []

    meshes = []          # textured (OBJ) versions
    colored_meshes = []  # vertex-colored (PLY) versions

    use_legacy_sampling = subdivision_depth <= 0 or patch_points_by_id is None

    for export_idx, cid in enumerate(patch_ids):
        verts, uv, faces = _sample_patch_grid(F, cid, resolution, device)
        if not use_legacy_sampling:
            occ_mask = _build_patch_uv_occupancy(
                patch_points_by_id.get(int(cid)),
                subdivision_depth=subdivision_depth,
                min_points=min_points_per_cell,
            )
            verts, uv, faces = _filter_patch_mesh_by_occupancy(verts, uv, faces, occ_mask)
            if verts.shape[0] == 0 or faces.shape[0] == 0:
                print(f"    Patch {cid:02d} skipped after occupancy filtering")
                continue
        tex_img = textures[export_idx % n_textures]

        if debug_uv_png:
            # Use the original UV layout before optional geometry duplication.
            _save_debug_uv_png(uv, tex_img, save_dir, cid)

        if double_sided:
            verts, uv, faces = _make_double_sided(verts, uv, faces)

        if unnormalize:
            verts_out = verts * meta['scale'] + meta['center']
        else:
            verts_out = verts

        uv_visuals = trimesh.visual.texture.TextureVisuals(uv=uv, image=tex_img)
        mesh = trimesh.Trimesh(vertices=verts_out, faces=faces,
                               visual=uv_visuals, process=False,
                               maintain_order=True)
        meshes.append(mesh)

        obj_path = os.path.join(save_dir, f"patch_{cid}_{epoch}.obj")
        mesh.export(obj_path)
        print(f"    Patch {cid:02d} textured OBJ → {obj_path}")

        if export_ply:
            colored_mesh = _bake_vertex_colors(mesh)
            colored_meshes.append(colored_mesh)
            ply_path = os.path.join(save_dir, f"patch_{cid}_{epoch}.ply")
            colored_mesh.export(ply_path)
            print(f"    Patch {cid:02d} vertex-colored PLY → {ply_path}")

        if debug_uv_png:
            _save_debug_uv_png(uv, tex_img, save_dir, cid)

    # Combined OBJ scene with per-patch materials.
    if len(meshes) == 0:
        print("  No meshes remained after occupancy filtering; skipping combined export.")
        return []

    scene = trimesh.Scene(meshes)
    obj_scene_name = f"{name}_checkerboard_{epoch}.obj" if name else f"checkerboard_{epoch}.obj"
    obj_scene_path = os.path.join(save_dir, obj_scene_name)
    scene.export(obj_scene_path)
    print(f"    Combined checkerboard scene (OBJ) → {obj_scene_path}")

    # Combined PLY as one merged vertex-colored mesh.
    if export_ply:
        combined_colored = trimesh.util.concatenate(colored_meshes)
        ply_scene_name = f"{name}_checkerboard_{epoch}.ply" if name else f"checkerboard_{epoch}.ply"
        ply_scene_path = os.path.join(save_dir, ply_scene_name)
        combined_colored.export(ply_scene_path)
        print(f"    Combined checkerboard scene (PLY) → {ply_scene_path}")

    return meshes


def _save_debug_uv_png(uv, tex_img, save_dir, cid):
    """
    Save a UV debug image with sampled points overlaid on the texture.

    Requires `cv2`. If unavailable, this function returns silently.
    """
    try:
        import cv2
    except ImportError:
        return

    w, h = tex_img.size
    img = np.array(tex_img.convert("RGB")).copy()  # (h, w, 3), RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pts_px = (uv * [w, h]).astype(np.int32)
    for px, py in pts_px[::max(1, len(pts_px) // 2000)]:
        cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)

    patch_dir = os.path.join(save_dir, str(cid))
    os.makedirs(patch_dir, exist_ok=True)
    out_path = os.path.join(patch_dir, "debug_uv.png")
    cv2.imwrite(out_path, img)


# Run against a saved checkpoint
def _import_model_module(model_path=None):
    """
    Load the model module without relying on the current working directory.

    Args:
        model_path: Optional explicit path to `model/model.py` or the model package directory.

    Returns:
        Loaded module object.
    """
    if model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(here)
        candidates = [
            os.path.join(repo_root, 'model', 'model.py'),
            os.path.join(repo_root, 'model'),
            os.path.join(here, 'model.py'),
            os.path.join(here, '..', 'model.py'),
            os.path.join(os.getcwd(), 'model', 'model.py'),
            os.path.join(os.getcwd(), 'model'),
            os.path.join(os.getcwd(), 'model.py'),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Could not locate the model module automatically (checked the "
            "repo's model/ package, next to patch_vis.py, its parent folder, "
            "and the current working directory). Pass its location explicitly "
            "with --model_path /path/to/model/model.py (or model_path=... if "
            "calling from Python)."
        )

    model_path = os.path.abspath(model_path)
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'model.py')
    spec = importlib.util.spec_from_file_location("_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"  Loaded model classes from: {model_path}")
    return module


def _load_model_from_checkpoint(ckpt_path, device, model_path=None):
    """
    Rebuild `MultiPatchForwardMap` from a saved checkpoint.

    Args:
        ckpt_path: Path to checkpoint file.
        device: Torch device string.
        model_path: Optional explicit path to `model.py`.
    """
    model_module = _import_model_module(model_path)
    MultiPatchForwardMap = getattr(model_module, 'MultiPatchForwardMap', None)
    TwoSheetForwardMap = getattr(model_module, 'TwoSheetForwardMap', None)
    if MultiPatchForwardMap is None and model_path is not None and os.path.basename(model_path) == 'main.py':
        sibling_model_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'model', 'model.py')
        if os.path.exists(sibling_model_path):
            model_module = _import_model_module(sibling_model_path)
            MultiPatchForwardMap = model_module.MultiPatchForwardMap
    if MultiPatchForwardMap is None and TwoSheetForwardMap is None:
        raise AttributeError("model module does not define MultiPatchForwardMap or TwoSheetForwardMap")

    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get('mode') != 'multi_patch' and ckpt.get('mode') != 'multi_patch_pretrain_flat_sheet':
        raise ValueError(
            f"Checkpoint mode is '{ckpt.get('mode')}', expected 'multi_patch' or 'multi_patch_pretrain_flat_sheet'. "
            f"Checkerboard export only supports multi-patch checkpoints."
        )

    args = ckpt['args']
    n_rows, n_cols = ckpt['grid_dims']
    atlas_mode = ckpt.get('atlas_mode', args.get('atlas_mode', 'single_sheet'))

    if atlas_mode == 'two_sheet':
        if TwoSheetForwardMap is None:
            raise AttributeError("model module does not define TwoSheetForwardMap")
        F = TwoSheetForwardMap(
            n_rows=n_rows,
            n_cols=n_cols,
            d_features=args['d_features'],
            L=args['L'], W=args['W'], D=args['D'], beta=args['beta'],
        ).to(device)
    else:
        if MultiPatchForwardMap is None:
            raise AttributeError("model module does not define MultiPatchForwardMap")
        F = MultiPatchForwardMap(
            n_rows=n_rows, n_cols=n_cols,
            d_features=args['d_features'],
            L=args['L'], W=args['W'], D=args['D'], beta=args['beta'],
        ).to(device)
    F.load_state_dict(ckpt['F_state'])
    F.eval()

    normalization = ckpt.get('normalization')
    if normalization is None:
        # Pretrain-only checkpoints may omit normalization metadata.
        meta = {
            'center': np.zeros(3, dtype=np.float32),
            'scale': 1.0,
        }
    else:
        meta = {
            'center': np.array(normalization['center'], dtype=np.float32),
            'scale': float(normalization['scale']),
        }
    active_patch_ids = ckpt.get('active_patch_ids')
    return F, meta, args, active_patch_ids, ckpt.get('grid_dims')


def _resolve_checkpoint_paths(ckpt_path):
    """Return a sorted list of checkpoint files from a file or directory."""
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isdir(ckpt_path):
        candidates = glob.glob(os.path.join(ckpt_path, 'checkpoint*.pt'))

        def _sort_key(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            suffix = stem.replace('checkpoint_', '')
            digits = ''.join(ch for ch in suffix if ch.isdigit())
            return (0, int(digits)) if digits else (1, stem)

        return sorted(candidates, key=_sort_key)
    return [ckpt_path]


def _build_patch_points_by_id(input_points: np.ndarray, assignments, active_patch_ids):
    assignments = np.asarray(assignments)
    patch_points = {}
    for cid in active_patch_ids:
        patch_points[int(cid)] = input_points[assignments == int(cid)]
    return patch_points


def _resegment_current_input_points(input_points: np.ndarray, n_rows: int, n_cols: int,
                                    active_patch_ids=None):
    """Use the current normalized input cloud directly to recover top-level patch groups."""
    assignments, _, _ = pc_presegmentation.pca_grid_segmentation(
        input_points, n_patches_u=n_rows, n_patches_v=n_cols
    )
    if active_patch_ids is None:
        active_patch_ids = sorted(np.unique(assignments).tolist())
    patch_points = {}
    for cid in active_patch_ids:
        patch_points[int(cid)] = input_points[assignments == int(cid)]
    return patch_points


def main():
    parser = argparse.ArgumentParser(
        description='Export checkerboard-textured per-patch meshes from a '
                   'trained multi-patch checkpoint.')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint.pt saved by main.py, or a directory containing checkpoint*.pt files')
    parser.add_argument('--texture_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'texture'),
                        help='Path to a SINGLE checkerboard image (used for every '
                             'patch), OR a directory of Slide1.jpg...SlideN.jpg '
                             'images for a distinct texture per patch')
    parser.add_argument('--out_dir', type=str, default='checkerboard_export',
                        help='Output directory for textured OBJ/PLY files')
    parser.add_argument('--resolution', type=int, default=500,
                        help='Per-patch UV grid resolution')
    parser.add_argument('--n_images', type=int, default=1,
                        help='Number of checkerboard texture images to load '
                             '(only used if --texture_path is a directory)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_unnormalize', action='store_true',
                        help='Keep vertices in normalized [-1,1]^3 space '
                             '(default: export in original coordinates)')
    parser.add_argument('--no_ply', action='store_true',
                        help='Skip vertex-colored .ply export (OBJ only)')
    parser.add_argument('--single_sided', action='store_true',
                        help='Disable double-sided geometry (default: patches '
                             'are duplicated so they render from both sides). '
                             'Pass this to keep the original single-layer sheet '
                             'and rely on viewer-side backface settings instead.')
    parser.add_argument('--model_path', '--main_path', dest='model_path', type=str, default=None,
                        help='Explicit path to model/model.py or the model directory (alias: --main_path). '
                            'Needed if patch_vis.py is not next to the model package and '
                            'not run from its folder. '
                            'e.g. /media/.../NISR_Seawall/model/model.py')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input point cloud used to build occupancy masks inside active patches')
    parser.add_argument('--subdivision_depth', type=int, default=1,
                        help='Recursive subdivision depth inside each active patch for UV occupancy filtering. '
                            'Use 0 to keep only active patches without subdivision, and -1 to sample all UVs '
                            'from every patch including empty/inactive ones.')
    parser.add_argument('--min_points_per_cell', type=int, default=10,
                        help='Minimum number of input points required for a subdivided UV cell to be kept')
    args = parser.parse_args()

    ckpt_paths = _resolve_checkpoint_paths(args.ckpt)
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found at: {args.ckpt}")

    if len(ckpt_paths) > 1:
        print(f"  Found {len(ckpt_paths)} checkpoint files in {os.path.abspath(args.ckpt)}")

    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        export_dir = os.path.join(args.out_dir, ckpt_name)

        F, meta, ckpt_args, active_patch_ids, _ = _load_model_from_checkpoint(
            ckpt_path, args.device, args.model_path
        )
        patch_points_by_id = None
        if active_patch_ids is not None and args.subdivision_depth > 0:
            input_file = args.input_file or ckpt_args.get('file')
            if input_file is None:
                raise ValueError('Occupancy-aware export requires --input_file or checkpoint args[file].')
            input_points, _ = utils.load_point_cloud(input_file)
            patch_points_by_id = _resegment_current_input_points(
                input_points, F.n_rows, F.n_cols, active_patch_ids=active_patch_ids
            )
        print(f"  Loaded model: {F.n_rows}x{F.n_cols} = {F.n_patches} patches")
        if active_patch_ids is not None:
            print(f"  Active patches in checkpoint: {len(active_patch_ids)}")
        print(f"  Exporting checkpoint {ckpt_name} → {export_dir}")

        export_checkerboard_patches(
            F, meta,
            save_dir=export_dir,
            texture_path=args.texture_path,
            resolution=args.resolution,
            device=args.device,
            epoch=ckpt_name,
            name=ckpt_name,
            n_images=args.n_images,
            unnormalize=not args.no_unnormalize,
            export_ply=not args.no_ply,
            double_sided=not args.single_sided,
            active_patch_ids=active_patch_ids,
            patch_points_by_id=patch_points_by_id,
            subdivision_depth=args.subdivision_depth,
            min_points_per_cell=args.min_points_per_cell,
        )

    print(f"\n  Done. Textured patches → {args.out_dir}")


if __name__ == '__main__':
    main()