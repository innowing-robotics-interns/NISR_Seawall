#!/usr/bin/env python3
"""Phase 1-2 verification for the NGF rendering-loss integration.

Checks the pieces that do not touch the training loop yet:

1. Camera / image-order alignment on a synthetic sphere: the point-cloud
   reference and the triangle mesh are generated from the same surface, so a
   correct renderer must line them up.
2. Real point cloud: build the cached orbit targets, save target previews.
3. Optional quick fit of a single-sheet fixed-grid model to that cloud, then
   sample it with ``utils.render_mesh.sample_patch_mesh``, render it with
   ``model.ngf_rendering_loss.NGFRenderer`` and save target/prediction
   comparison montages.
4. Gradient check: one render loss backward must reach the atlas vertex
   features and the decoder.

Nothing here modifies ``main.py`` or starts a full training run.

Example:
    python scripts/verify_render_integration.py \
        --ply normalized_shapes/120477.ply \
        --out /tmp/render_phase2_check \
        --fit-epochs 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import main as nisr_main
import utils.utils as nisr_utils
from model.ngf_rendering_loss import NGFRenderer, UniformLaplacian, ngf_rendering_loss
from utils.render_mesh import auto_mesh_resolution, sample_patch_mesh, vertex_normals
from utils.render_targets import (build_render_targets, make_orbit_views,
                                  save_comparison)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compare_renders(prediction: torch.Tensor,
                    target: torch.Tensor,
                    target_alpha: torch.Tensor) -> dict:
    """L1 errors for renders shaped ``(V, H, W, 6)``.

    ``prediction`` comes from NGFRenderer.  ``target`` is the packed
    normal/position map; ``target_alpha`` is the splat coverage.
    """
    pred = prediction.detach().float()
    tgt = target.detach().float().to(pred.device)[..., :6]
    alpha = target_alpha.detach().float().to(pred.device)

    pred_norm = pred[..., :3]
    target_mask = alpha.squeeze(-1) > 0.5
    pred_mask = pred_norm.norm(dim=-1) > 1e-6
    shared = target_mask & pred_mask

    error = (pred - tgt).abs()
    total_pixels = error.shape[0] * error.shape[1] * error.shape[2]

    full_l1 = float(error.mean())
    per_view_full = error.mean(dim=(1, 2, 3))

    shared_pixels = float(shared.sum().clamp_min(1))
    normal_l1 = float((error[..., :3].sum(-1) * shared).sum()
                      / (shared_pixels * 3))
    position_l1 = float((error[..., 3:].sum(-1) * shared).sum()
                        / (shared_pixels * 3))
    shared_l1 = float((error.sum(-1) * shared).sum()
                      / (shared_pixels * 6))
    normal_cosine = float(
        ((pred_norm * tgt[..., :3]).sum(-1) * shared).sum()
        / shared_pixels)

    union = target_mask | pred_mask
    intersection = target_mask & pred_mask
    iou = float(intersection.sum().clamp_min(1)
                / union.sum().clamp_min(1))

    return {
        'views': int(pred.shape[0]),
        'full_l1': full_l1,
        'full_l1_per_view': [float(v) for v in per_view_full],
        'shared_l1': shared_l1,
        'shared_normal_l1': normal_l1,
        'shared_position_l1': position_l1,
        'shared_normal_cosine': normal_cosine,
        'shared_fraction': float(shared.sum() / total_pixels),
        'silhouette_iou': iou,
        'target_coverage': float(target_mask.float().mean()),
        'prediction_coverage': float(pred_mask.float().mean()),
    }


def _splat_target_from_mesh(vertices, faces, views, resolution, fov):
    """Reference maps for a synthetic surface, using the same splat path.

    The mesh is randomly surface-sampled and splatted.  This is a target built
    from geometry, not from the renderer, so it independently exercises the
    camera convention.
    """
    device = vertices.device
    tri = vertices[faces.long()]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * edge1.cross(edge2, dim=-1).norm(dim=-1)
    counts = torch.multinomial(areas, num_samples=30000, replacement=True)
    r1 = torch.rand(counts.shape[0], 1, device=device)
    r2 = torch.rand(counts.shape[0], 1, device=device)
    flip = r1 + r2 > 1.0
    r1 = torch.where(flip, 1.0 - r1, r1)
    r2 = torch.where(flip, 1.0 - r2, r2)
    picked = tri[counts]
    points = (picked[:, 0] + r1 * edge1[counts] + r2 * edge2[counts])

    # Area-weighted vertex normals interpolate smoothly; reuse them as ground
    # truth normals for the sampled points.
    normals = vertex_normals(vertices, faces)
    normals = (normals[faces[counts, 0].long()] * (1.0 - r1 - r2)
               + normals[faces[counts, 1].long()] * r1
               + normals[faces[counts, 2].long()] * r2)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    targets = build_render_targets(
        points, normals, num_views=int(views.shape[0]),
        resolution=resolution, fov=fov, min_coverage=0.05, device=device,
        preview_dir=None, verbose=False)
    return targets


def sphere_alignment_check(out_dir: str, resolution: int = 128,
                           num_views: int = 8, fov: float = 45.0) -> dict:
    """Synthetic sphere: point-cloud target vs the mesh it came from."""
    import trimesh

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.8)
    vertices = torch.as_tensor(sphere.vertices, dtype=torch.float32,
                               device=device)
    faces = torch.as_tensor(sphere.faces, dtype=torch.int32, device=device)
    normals = vertex_normals(vertices, faces)

    views, _, _, _ = make_orbit_views(vertices, count=num_views, fov=fov,
                                      device=device)
    targets = _splat_target_from_mesh(vertices, faces, views, resolution, fov)

    renderer = NGFRenderer(width=resolution, height=resolution,
                           fov=fov, device=str(device))
    with torch.no_grad():
        prediction = renderer.render(vertices, normals, faces,
                                     targets.views.to(device))
    metrics = compare_renders(prediction, targets.geometry,
                              targets.geometry[..., 6:7])
    save_comparison(targets.geometry, prediction, out_dir, count=4,
                    prefix='sphere')
    metrics['target_config'] = targets.summary()
    return metrics


# ---------------------------------------------------------------------------
# Model quick fit
# ---------------------------------------------------------------------------


def quick_fit(pts3n: np.ndarray,
              normals: np.ndarray,
              out_dir: str,
              pretrain_epochs: int = 200,
              fit_epochs: int = 400,
              n_patches: int = 16,
              d_features: int = 64,
              mu: float = 0.5,
              gamma: float = 0.5,
              device: str = 'cuda') -> dict:
    """Warm-start a fixed-grid atlas, fit it to the cloud, return the model."""
    os.makedirs(out_dir, exist_ok=True)
    timings = {}

    t0 = time.time()
    F0, _ = nisr_main.pretrain_multi_patch_flat_sheet(
        n_patches=n_patches, d_features=d_features, epochs=pretrain_epochs,
        M_per_patch=4096, W=256, D=6, L=8, lr=1e-3, beta=5.0,
        device=device, log_every=max(50, pretrain_epochs // 4))
    timings['pretrain_s'] = time.time() - t0

    t0 = time.time()
    F, _, history, assignments, active_ids = nisr_main.train_multi_patch(
        pts3n,
        n_patches=n_patches, d_features=d_features, epochs=fit_epochs,
        M=2048, M_per_patch=512, W=256, D=6, L=8,
        lr=1e-3, mu=mu, gamma=gamma, lam=0.0, lam2=0.0, beta=5.0,
        device=device, log_every=max(25, fit_epochs // 8),
        save_patch_vis=True, vis_dir=out_dir,
        checkpoint_every=10 ** 9, normals=normals,
        pretrained_F_state=F0.state_dict(),
        lambda_outer_boundary=0.0, surface_loss_type='chamfer')
    timings['fit_s'] = time.time() - t0

    mesh_resolution = auto_mesh_resolution(len(active_ids), max_vertices=30000)
    t0 = time.time()
    vertices, faces = sample_patch_mesh(F, mesh_resolution,
                                        patch_ids=active_ids)
    normals_mesh = vertex_normals(vertices, faces)
    timings['mesh_sample_s'] = time.time() - t0

    return {
        'F': F,
        'active_ids': active_ids,
        'assignments': assignments,
        'history': history,
        'timings': timings,
        'vertices': vertices,
        'faces': faces,
        'vertex_normals': normals_mesh,
        'mesh_resolution': mesh_resolution,
    }


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------


def gradient_check(F, vertices, normals, faces, targets, renderer,
                   num_views: int = 2) -> dict:
    """One render-loss backward; confirm gradients reach both parameter groups."""
    device = vertices.device
    count = min(int(num_views), len(targets))
    views = targets.views[:count].to(device)
    reference = targets.geometry[:count].to(device)

    F.zero_grad(set_to_none=True)
    loss, parts = ngf_rendering_loss(
        vertices, normals, faces, views, reference,
        renderer=renderer, laplacian_weight=0.0, return_parts=True)
    loss.backward()

    vertex_features = F.complex.vertex_features
    decoder_grads = [p.grad for name, p in F.named_parameters()
                     if not name.startswith('complex.vertex_features')
                     and p.grad is not None]
    return {
        'loss': float(loss.detach()),
        'render': float(parts['render']),
        'views_used': count,
        'vertex_features_grad_norm': float(
            vertex_features.grad.norm()) if vertex_features.grad is not None else 0.0,
        'decoder_grad_norm': float(torch.stack(
            [g.norm() for g in decoder_grads]).sum()) if decoder_grads else 0.0,
        'decoder_params_with_grad': len(decoder_grads),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply', default=os.path.join(
        REPO_ROOT, 'normalized_shapes', '120477.ply'))
    parser.add_argument('--out', default='/tmp/render_phase2_check')
    parser.add_argument('--views', type=int, default=12)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--points', type=int, default=25000,
                        help='downsample the cloud to this many points')
    parser.add_argument('--fit-epochs', type=int, default=400)
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--n-patches', type=int, default=16)
    parser.add_argument('--mu', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--d-features', type=int, default=64)
    parser.add_argument('--skip-fit', action='store_true')
    parser.add_argument('--skip-sphere', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    summary = {
        'ply': os.path.abspath(args.ply),
        'config': {
            'views': args.views,
            'resolution': args.resolution,
            'min_coverage': 0.08,
            'points': args.points,
            'fit_epochs': args.fit_epochs,
            'pretrain_epochs': args.pretrain_epochs,
            'n_patches': args.n_patches,
            'd_features': args.d_features,
            'mu': args.mu,
            'gamma': args.gamma,
            'device': args.device,
        },
    }

    if not args.skip_sphere:
        print('\n=== 1. Synthetic sphere alignment ===')
        sphere = sphere_alignment_check(
            os.path.join(args.out, 'sphere'),
            resolution=args.resolution, num_views=8)
        summary['sphere_alignment'] = sphere
        print(f"  full L1 {sphere['full_l1']:.5f}  "
              f"shared L1 {sphere['shared_l1']:.5f}  "
              f"silhouette IoU {sphere['silhouette_iou']:.3f}")

    print('\n=== 2. Real point-cloud targets ===')
    t0 = time.time()
    pts3n, meta = nisr_utils.load_point_cloud(
        args.ply, downsample_n=args.points)
    normals = meta['normals']
    target_time = time.time() - t0

    t0 = time.time()
    targets = build_render_targets(
        pts3n, normals, num_views=args.views, resolution=args.resolution,
        device=args.device, min_coverage=0.08,
        preview_dir=os.path.join(args.out, 'targets'), preview_count=8)
    target_build_time = time.time() - t0

    summary['pointcloud'] = {
        'points': int(pts3n.shape[0]),
        'has_normals': normals is not None,
        'load_s': target_time,
        'target_build_s': target_build_time,
        'targets': targets.summary(),
    }
    print(f"  kept {len(targets)}/{args.views} views  "
          f"coverage {float(targets.coverage.mean()):.3f}  "
          f"built in {target_build_time:.1f}s")

    if args.skip_fit:
        with open(os.path.join(args.out, 'summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nSummary -> {os.path.join(args.out, 'summary.json')}")
        return

    print('\n=== 3. Quick fit + differentiable mesh render ===')
    fit = quick_fit(
        pts3n, normals, os.path.join(args.out, 'fit'),
        pretrain_epochs=args.pretrain_epochs, fit_epochs=args.fit_epochs,
        n_patches=args.n_patches, d_features=args.d_features,
        mu=args.mu, gamma=args.gamma, device=args.device)
    F = fit['F']
    vertices = fit['vertices']
    faces = fit['faces']
    mesh_normals = fit['vertex_normals']

    renderer = NGFRenderer(width=args.resolution, height=args.resolution,
                           fov=targets.fov, device=str(vertices.device))
    t0 = time.time()
    with torch.no_grad():
        prediction = renderer.render(vertices, mesh_normals, faces,
                                     targets.views.to(vertices.device))
    render_time = time.time() - t0

    metrics = compare_renders(prediction, targets.geometry,
                              targets.geometry[..., 6:7])
    save_comparison(targets.geometry, prediction,
                    os.path.join(args.out, 'compare'), count=4,
                    prefix='fit')
    print(f"  mesh {vertices.shape[0]} verts / {faces.shape[0]} faces  "
          f"render {render_time:.2f}s")
    print(f"  full L1 {metrics['full_l1']:.5f}  "
          f"shared L1 {metrics['shared_l1']:.5f}  "
          f"silhouette IoU {metrics['silhouette_iou']:.3f}")

    print('\n=== 4. Gradient check ===')
    grads = gradient_check(F, vertices, mesh_normals, faces, targets, renderer)
    print(f"  loss {grads['loss']:.5f}  "
          f"vertex-feature grad {grads['vertex_features_grad_norm']:.3e}  "
          f"decoder grad {grads['decoder_grad_norm']:.3e}")

    summary['fit'] = {
        'active_patches': len(fit['active_ids']),
        'mesh_resolution': fit['mesh_resolution'],
        'vertices': int(vertices.shape[0]),
        'faces': int(faces.shape[0]),
        'timings': fit['timings'],
        'render_s': render_time,
        'final_cd': float(fit['history']['cd'][-1])
        if fit['history'].get('cd') else None,
        'final_total': float(fit['history']['total'][-1])
        if fit['history'].get('total') else None,
    }
    summary['render_comparison'] = metrics
    summary['gradients'] = grads

    path = os.path.join(args.out, 'summary.json')
    with open(path, 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSummary -> {path}")


if __name__ == '__main__':
    main()
