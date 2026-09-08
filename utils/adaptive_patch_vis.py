#!/usr/bin/env python3
# patch_vis.py — checkerboard export for ADAPTIVE cube-atlas checkpoints.

import argparse
import glob
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_HERE), _HERE, os.getcwd()):
    if os.path.isdir(os.path.join(_root, 'model')) and _root not in sys.path:
        sys.path.insert(0, _root)

from model.adaptive_model import AdaptiveCubeForwardMap  # noqa: E402


def load_checkerboard_textures(texture_path, pattern="Slide5.jpg", n_images=1):
    if not os.path.isabs(texture_path):
        for cand in (texture_path, os.path.join(_HERE, texture_path),
                     os.path.join(os.path.dirname(_HERE), texture_path)):
            if os.path.exists(cand):
                texture_path = cand
                break
    if os.path.isfile(texture_path):
        print(f"  Using ONE checkerboard texture for all patches: {texture_path}")
        return [Image.open(texture_path).convert("RGB")]
    if os.path.isdir(texture_path):
        textures = []
        for i in range(1, n_images + 1):
            p = os.path.join(texture_path, pattern.format(i))
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing checkerboard texture: {p}")
            textures.append(Image.open(p).convert("RGB"))
        return textures
    raise FileNotFoundError(f"Checkerboard texture path not found: {texture_path}")


@torch.no_grad()
def _sample_patch_grid(F, patch_idx, resolution, device):
    u = torch.linspace(0, 1, resolution, device=device)
    gu, gv = torch.meshgrid(u, u, indexing='ij')
    uv = torch.stack([gu.flatten(), gv.flatten()], dim=-1)
    verts = []
    for i in range(0, uv.shape[0], 4096):
        verts.append(F(patch_idx, uv[i:i + 4096]).cpu())
    verts = torch.cat(verts, 0).numpy().astype(np.float32)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = i * resolution + j
            b = (i + 1) * resolution + j
            c = i * resolution + (j + 1)
            d = (i + 1) * resolution + (j + 1)
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, uv.cpu().numpy().astype(np.float32), np.array(faces, np.int32)


def _make_double_sided(verts, uv, faces):
    n = verts.shape[0]
    return (np.concatenate([verts, verts], 0),
            np.concatenate([uv, uv], 0),
            np.concatenate([faces, faces[:, [0, 2, 1]] + n], 0).astype(np.int32))


def export_checkerboard_patches(F, meta, save_dir, texture_path, resolution=100,
                                device='cuda', epoch='final', name=None,
                                n_images=1, unnormalize=True,
                                export_ply=True, double_sided=True):
    os.makedirs(save_dir, exist_ok=True)
    textures = load_checkerboard_textures(texture_path, n_images=n_images)
    meshes, colored = [], []

    for cid in range(F.n_patches):
        p = F.complex.leaf_patches[cid]
        verts, uv, faces = _sample_patch_grid(F, cid, resolution, device)
        tex = textures[cid % len(textures)]
        if double_sided:
            verts, uv, faces = _make_double_sided(verts, uv, faces)
        if unnormalize:
            verts = verts * meta['scale'] + meta['center']
        vis = trimesh.visual.texture.TextureVisuals(uv=uv, image=tex)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=vis,
                               process=False, maintain_order=True)
        meshes.append(mesh)
        tag = f"patch_{cid:03d}_d{p.depth}_{epoch}"
        mesh.export(os.path.join(save_dir, tag + ".obj"))
        if export_ply:
            cm = mesh.copy()
            cm.visual = mesh.visual.to_color()
            colored.append(cm)
            cm.export(os.path.join(save_dir, tag + ".ply"))
        print(f"    Patch {cid:03d} (face={p.face}, depth={p.depth}) exported")

    scene_name = f"{name}_checkerboard_{epoch}" if name else f"checkerboard_{epoch}"
    trimesh.Scene(meshes).export(os.path.join(save_dir, scene_name + ".obj"))
    if export_ply and colored:
        trimesh.util.concatenate(colored).export(
            os.path.join(save_dir, scene_name + ".ply"))
    print(f"    Combined scene → {os.path.join(save_dir, scene_name)}.obj/.ply")
    return meshes


def _resolve_checkpoint_paths(ckpt_path):
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isdir(ckpt_path):
        cands = glob.glob(os.path.join(ckpt_path, '*checkpoint*.pt'))
        def _key(p):
            digits = ''.join(ch for ch in os.path.basename(p) if ch.isdigit())
            return (0, int(digits)) if digits else (1, p)
        return sorted(cands, key=_key)
    return [ckpt_path]


def main():
    ap = argparse.ArgumentParser(description='Checkerboard export for adaptive '
                                             'cube-atlas checkpoints')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--texture_path', type=str,
                    default=os.path.join(_HERE, 'texture'))
    ap.add_argument('--out_dir', type=str, default='checkerboard_export')
    ap.add_argument('--resolution', type=int, default=200)
    ap.add_argument('--n_images', type=int, default=1)
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no_unnormalize', action='store_true')
    ap.add_argument('--no_ply', action='store_true')
    ap.add_argument('--single_sided', action='store_true')
    args = ap.parse_args()

    for ckpt_path in _resolve_checkpoint_paths(args.ckpt):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if ckpt.get('mode') != 'adaptive_cube':
            print(f"  Skipping {ckpt_path}: mode={ckpt.get('mode')} "
                  f"(expected 'adaptive_cube')")
            continue
        F = AdaptiveCubeForwardMap.from_checkpoint(ckpt, device=args.device)
        norm = ckpt.get('normalization')
        meta = ({'center': np.array(norm['center'], np.float32),
                 'scale': float(norm['scale'])} if norm else
                {'center': np.zeros(3, np.float32), 'scale': 1.0})
        name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"  {name}: {F.n_patches} leaves, "
              f"{F.complex.n_vertices} vertices, depth hist "
              f"{F.complex.stats()['depth_hist']}")
        export_checkerboard_patches(
            F, meta, save_dir=os.path.join(args.out_dir, name),
            texture_path=args.texture_path, resolution=args.resolution,
            device=args.device, epoch=name, name=name, n_images=args.n_images,
            unnormalize=not args.no_unnormalize, export_ply=not args.no_ply,
            double_sided=not args.single_sided)
    print(f"\n  Done → {args.out_dir}")


if __name__ == '__main__':
    main()