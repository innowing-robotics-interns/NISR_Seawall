#!/usr/bin/env python3
# utils/cutting_hole.py
"""
Detect -> cut -> stitch one handle on an adaptive model (docs/documentation.md).

Used two ways:

  * from main.py (--cut_hole): cut_and_stitch_hole() runs between the two
    training phases, on the live model;
  * standalone, on a saved adaptive checkpoint:

        python utils/cutting_hole.py --ckpt logs/.../checkpoint.pt
        python main.py --adaptive --load_ckpt <out_dir>/checkpoint_hole.pt ...

Outputs (out_dir):
    crossing.ply          the crossing curve C found by patch_intersection
    loops.ply             the two cut loops (red = A, blue = B)
    cut_surface.ply       surface after the cut, before stitching
    stitched_surface.ply  surface with the tube after the tube prefit (tube in orange)
    openings.ply          every opening found (one colour per ID; IDs and
                          areas are printed and in hole_summary.json)
    openings_chosen.ply   the two openings that were cut (red = A, blue = B)
    hole_summary.json     detection, cut and stitch diagnostics
    checkpoint_hole.pt    (standalone only) the stitched model
"""

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch

try:
    from . import patch_vis
    from .opening_labels import opening_colors
    from .uv_hole_mask import grid_faces, write_colored_mesh_ply, write_colored_ply
except ImportError:
    import patch_vis
    from opening_labels import opening_colors
    from uv_hole_mask import grid_faces, write_colored_mesh_ply, write_colored_ply

from model import cutting_hole, stitching_hole  # noqa: E402  (patch_vis sets sys.path)


COLOR_SURFACE = np.array((200, 200, 200), np.uint8)
COLOR_FROZEN = np.array((240, 200, 40), np.uint8)
COLOR_TUBE = np.array((245, 130, 40), np.uint8)
COLOR_A = np.array((230, 40, 40), np.uint8)
COLOR_B = np.array((40, 90, 230), np.uint8)


@torch.no_grad()
def sample_leaves(model, res, device, batch=16384):
    """res x res grid on every leaf (quad leaves and tube cells).
    Returns (verts, faces, leaf_of_vert)."""
    n = model.n_patches
    t = torch.linspace(0.0, 1.0, res, device=device)
    gu, gv = torch.meshgrid(t, t, indexing='ij')
    grid = torch.stack([gu.flatten(), gv.flatten()], -1)
    _, tri = grid_faces(res)
    pids = torch.arange(n, device=device).repeat_interleave(grid.shape[0])
    uv = grid.repeat(n, 1)
    out = [model(pids[i:i + batch], uv[i:i + batch]).cpu()
           for i in range(0, pids.shape[0], batch)]
    faces = (tri[None] + (np.arange(n) * res * res)[:, None, None]).reshape(-1, 3)
    return torch.cat(out).numpy(), faces, np.repeat(np.arange(n), res * res)


def export_surface(model, path, res, device):
    cx = model.complex
    verts, faces, leaf = sample_leaves(model, res, device)
    role = np.zeros(model.n_patches, np.int64)
    role[[i for i, p in enumerate(cx.leaf_patches) if p.frozen]] = 1
    role[cx.n_quad_leaves:] = 2
    colors = np.stack([COLOR_SURFACE, COLOR_FROZEN, COLOR_TUBE])[role[leaf]]
    write_colored_mesh_ply(path, verts, colors, faces)


def cut_and_stitch_hole(model, pts, cfg, device, out_dir=None, stitch=True,
                        mesh_res=12, verbose=True):
    """
    Find the handle, cut it, and (by default) stitch the tube — in place.

    Raises cutting_hole.NoCrossingFound when the detection finds no crossing
    openings; the model is untouched in that case.

    Returns the JSON-serializable record (also written to hole_summary.json).
    After this call vertex_features is a new Parameter: rebuild the optimizer.
    """
    was_training = model.training
    model.eval()
    if verbose:
        print(f"\n{'─' * 60}\n  Hole cutting: detection "
              f"(uv_hole_mask -> opening_labels -> patch_intersection)")
    crossing = cutting_hole.detect_crossing(model, pts, cfg, device, verbose)
    if verbose:
        print(f"  Cutting openings {crossing['opening_a']} <-> {crossing['opening_b']} "
              f"({len(crossing['xyz'])} crossing points)")
    record = cutting_hole.cut_hole(model, crossing, cfg, verbose)
    record['detection'] = crossing['stats']
    record['config'] = asdict(cfg)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ids = crossing['opening_ids']
        write_colored_ply(os.path.join(out_dir, 'openings.ply'), crossing['opening_xyz'],
                          opening_colors(int(ids.max()) + 1)[ids])
        chosen = np.isin(ids, record['openings'])
        write_colored_ply(os.path.join(out_dir, 'openings_chosen.ply'),
                          crossing['opening_xyz'][chosen],
                          np.where((ids[chosen] == record['openings'][0])[:, None],
                                   COLOR_A, COLOR_B))
        xyz = np.asarray(record['crossing_xyz'])
        write_colored_ply(os.path.join(out_dir, 'crossing.ply'), xyz,
                          np.tile(COLOR_TUBE, (len(xyz), 1)))
        la, lb = np.asarray(record['loop_a_xyz']), np.asarray(record['loop_b_xyz'])
        write_colored_ply(os.path.join(out_dir, 'loops.ply'), np.concatenate([la, lb]),
                          np.concatenate([np.tile(COLOR_A, (len(la), 1)),
                                          np.tile(COLOR_B, (len(lb), 1))]))
        export_surface(model, os.path.join(out_dir, 'cut_surface.ply'), mesh_res, device)

    if stitch:
        if verbose:
            print("  Stitching")
        stitching_hole.stitch_hole(model, record, cfg, verbose)
        cx = model.complex
        seams = stitching_hole.seam_gaps(model, only_tube=True)
        open_edges = cutting_hole.check_closed_orientable(cx)
        chi = cutting_hole.euler_characteristic(cx, cx.leaf_patches)
        record['seams'] = seams
        record['euler_characteristic']['after_stitch'] = chi
        record['genus'] = (2 - chi) / 2
        if verbose:
            print(f"    tube seams: max gap {seams['max_gap']:.2e}, mean "
                  f"{seams['mean_gap']:.2e} over {seams['n_edges']} edges; "
                  f"open edges {len(open_edges)}; Euler characteristic {chi} "
                  f"-> genus {record['genus']:g}")
        if open_edges:
            raise RuntimeError(f"stitched surface still has {len(open_edges)} "
                               f"open edges")
        if out_dir:
            export_surface(model, os.path.join(out_dir, 'stitched_surface.ply'),
                           mesh_res, device)

    if out_dir:
        with open(os.path.join(out_dir, 'hole_summary.json'), 'w') as f:
            json.dump(record, f, indent=2)
        if verbose:
            print(f"  Hole outputs → {out_dir}")
    if verbose:
        print(f"{'─' * 60}")
    model.train(was_training)
    return record


def main():
    ap = argparse.ArgumentParser(
        description='Detect, cut and stitch one handle on an adaptive checkpoint.')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--input_file', type=str, default=None,
                    help='Target cloud (default: the file recorded in the checkpoint)')
    ap.add_argument('--N', type=int, default=-1,
                    help='Downsample the target cloud (-1 keeps all)')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='Default: <ckpt dir>/cutting_hole/<ckpt name>')
    ap.add_argument('--no_stitch', action='store_true',
                    help='Only cut (the checkpoint then has two open loops)')
    ap.add_argument('--mesh_res', type=int, default=12)
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    cutting_hole.add_hole_args(ap, prefix='hole_')
    args = ap.parse_args()
    cfg = cutting_hole.hole_config_from_args(args, prefix='hole_')

    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    args.out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), 'cutting_hole', stem)

    print(f"\n  Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    if ckpt.get('mode') != 'adaptive_cube':
        raise ValueError("cutting_hole.py only supports adaptive (--adaptive) checkpoints")
    F, meta, ckpt_args, _, _ = patch_vis._load_adaptive_from_checkpoint(args.ckpt, args.device)

    input_file = args.input_file or (ckpt_args or {}).get('file') or ckpt.get('input_file')
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError(f"target cloud not found ({input_file}); pass --input_file")
    # Normalize with the CHECKPOINT's center/scale, so the cloud sits in the
    # frame the model was trained in. (utils.load_point_cloud currently ignores
    # its center/scale arguments, so it is not used here.)
    raw, _ = patch_vis.utils._load_point_file(input_file)
    if 0 < args.N < raw.shape[0]:
        raw = raw[np.random.default_rng(0).choice(raw.shape[0], args.N, replace=False)]
    pts = ((raw - np.asarray(meta['center'], np.float64)) / float(meta['scale'])
           ).astype(np.float32)
    print(f"  Target cloud: {pts.shape[0]} points from {input_file}")

    record = cut_and_stitch_hole(F, pts, cfg, args.device, out_dir=args.out_dir,
                                 stitch=not args.no_stitch, mesh_res=args.mesh_res)

    extra = {k: v for k, v in ckpt.items()
             if k not in ('mode', 'F_state', 'config', 'topology')}
    extra['hole'] = record
    extra['hole_source'] = os.path.abspath(args.ckpt)
    out = os.path.join(args.out_dir, 'checkpoint_hole.pt')
    torch.save(F.checkpoint_payload(extra), out)
    print(f"  Checkpoint → {out}\n  Continue training with:\n"
          f"    python main.py --adaptive --load_ckpt {out} ...\n")


if __name__ == '__main__':
    main()
