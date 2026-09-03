#!/usr/bin/env python3
# main_adaptive.py
"""
Adaptive cube-atlas training.

Phase 1 (pretrain): fit the atlas DIRECTLY to the reference box via
pointwise MSE — F(leaf, uv) ≈ cube_xyz(leaf, uv). No subdivision. This
validates the data structure: the seam check must report ~1e-6 gaps and
the exported mesh must be a clean closed box.

Phase 2 (training): fit the target point cloud with global Chamfer
(+ optional Dirichlet tangent and/or SVD regularization). Every
--subdiv_every epochs the per-patch distortion is measured; leaves above
--subdiv_threshold are subdivided until --subdiv_max_depth.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import utils.utils as utils
from utils.adaptive_vis import visualize_patch_configuration, plot_history
from model.adaptive_model import AdaptiveCubeForwardMap
from model.subdivision import (compute_patch_distortion, subdivide_by_distortion,
                               check_seam_continuity)
from model.losses import (chamfer_distance_chunked, surface_jacobian,
                          tangent_loss_from_jac, mu_warmup_schedule)
from model.adaptive_losses import adaptive_svd_loss


def _make_optim(model, lr, epochs_left):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs_left, 1), eta_min=1e-6)
    return opt, sched


def _save_ckpt(model, path, extra=None):
    torch.save(model.checkpoint_payload(extra), path)
    print(f"    Checkpoint → {path}")


def pretrain_box(model, epochs, M_per_patch, lr, device, log_every,
                 loss_type='mse'):
    """Fit every leaf patch to its own piece of the reference cube."""
    loss_fn = nn.MSELoss() if loss_type == 'mse' else nn.L1Loss()
    opt, sched = _make_optim(model, lr, epochs)
    history = {'epoch': [], 'loss': []}

    print(f"\n{'─' * 60}")
    print("  Phase 1 — box pretraining (structure validation, no subdivision)")
    print(f"  leaves={model.n_patches}  vertices={model.complex.n_vertices}  "
          f"K_max={model.complex.stats()['k_max']}")
    print(f"  epochs={epochs}  M_per_patch={M_per_patch}  lr={lr}  loss={loss_type}")
    print(f"{'─' * 60}")
    t0 = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        K = model.n_patches
        pids = torch.arange(K, device=device).repeat_interleave(M_per_patch)
        uv = torch.rand(K * M_per_patch, 2, device=device)
        with torch.no_grad():
            target = model.cube_xyz(pids, uv)      # exact box [-1,1]^3
        pred = model(pids, uv)
        loss = loss_fn(pred, target)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        opt.step()
        sched.step()

        if epoch % log_every == 0 or epoch == 1:
            history['epoch'].append(epoch)
            history['loss'].append(float(loss))
            print(f"  Epoch {epoch:5d}/{epochs}  |  Box={float(loss):.6f}  "
                  f"[{time.time() - t0:.1f}s]")

    seam = check_seam_continuity(model)
    print(f"  Seam check: max gap={seam['max_gap']:.3e}  "
          f"mean gap={seam['mean_gap']:.3e}  (should be ~1e-6 or smaller)")
    print(f"{'─' * 60}\n")
    return history


def train_adaptive(model, pts3n, epochs, M_per_patch, lr, mu,
                   mu_warmup_epochs, mu_warmup_delay, schedule,
                   lam_svd, svd_mode, svd_target, svd_eps,
                   svd_warmup_epochs, svd_warmup_delay,
                   subdiv_threshold, subdiv_max_depth, subdiv_every,
                   subdiv_start, subdiv_stop, distortion_mode,
                   distortion_samples, max_splits_per_round,
                   device, log_every, vis_dir, checkpoint_every,
                   checkpoint_extra=None):
    pts_dev = torch.tensor(pts3n, dtype=torch.float32, device=device)
    opt, sched = _make_optim(model, lr, epochs)
    history = {'epoch': [], 'cd': [], 'tangent': [], 'svd': [],
               'total': [], 'n_leaves': []}
    events = []
    zero = torch.tensor(0.0, device=device)
    need_jac = (mu > 0) or (lam_svd > 0)

    print(f"\n{'─' * 60}")
    print("  Phase 2 — adaptive training with distortion-driven subdivision")
    print(f"  leaves={model.n_patches}  target points={pts3n.shape[0]}")
    print(f"  subdiv: threshold={subdiv_threshold}  max_depth={subdiv_max_depth}  "
          f"every={subdiv_every}  start={subdiv_start}  mode={distortion_mode}")
    print(f"  epochs={epochs}  M_per_patch={M_per_patch}  lr={lr}  μ={mu}")
    if lam_svd > 0:
        print(f"  SVD loss: λ_svd={lam_svd}  mode={svd_mode}"
              + (f"  target={svd_target}" if svd_mode == 'arap' else "")
              + f"  eps={svd_eps}  (depth-normalized singular values)")
    print(f"{'─' * 60}")
    t0 = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        # ── subdivision check ────────────────────────────────────────────
        do_check = (subdiv_threshold > 0 and subdiv_every > 0
                    and epoch >= subdiv_start
                    and (subdiv_stop <= 0 or epoch <= subdiv_stop)
                    and epoch % subdiv_every == 0)
        if do_check:
            rep = subdivide_by_distortion(
                model, subdiv_threshold, subdiv_max_depth,
                samples_per_patch=distortion_samples, mode=distortion_mode,
                max_splits_per_round=max_splits_per_round)
            print(f"  [subdiv @ {epoch}] distortion max={rep['max_distortion']:.4f} "
                  f"mean={rep['mean_distortion']:.4f}  split={rep['n_subdivided']} "
                  f"→ leaves={rep['n_leaves']} vertices={rep['n_vertices']}")
            if rep['n_subdivided'] > 0:
                # vertex_features was replaced → fresh optimizer/scheduler.
                cur_lr = opt.param_groups[0]['lr']
                opt, sched = _make_optim(model, cur_lr, epochs - epoch + 1)
                events.append({'epoch': epoch,
                               'n_subdivided': rep['n_subdivided'],
                               'n_leaves': rep['n_leaves']})
            visualize_patch_configuration(
                model, os.path.join(vis_dir, f'patch_config_{epoch:05d}.png'),
                pts=pts3n)

        # ── training step ────────────────────────────────────────────────
        K = model.n_patches
        pids = torch.arange(K, device=device).repeat_interleave(M_per_patch)
        uv = torch.rand(K * M_per_patch, 2, device=device,
                        requires_grad=need_jac)
        Q = model(pids, uv)

        B = min(K * M_per_patch, pts_dev.shape[0])
        ridx = torch.randint(0, pts_dev.shape[0], (B,), device=device)
        cd_loss = chamfer_distance_chunked(Q, pts_dev[ridx], chunk_size=2048)

        mu_eff = mu_warmup_schedule(epoch, mu_warmup_epochs, mu,
                                    schedule=schedule,
                                    delay_epochs=mu_warmup_delay) if mu > 0 else 0.0
        svd_eff = mu_warmup_schedule(epoch, svd_warmup_epochs, lam_svd,
                                     schedule=schedule,
                                     delay_epochs=svd_warmup_delay) if lam_svd > 0 else 0.0

        # One Jacobian, shared by both regularizers.
        if need_jac and (mu_eff > 0 or svd_eff > 0):
            t_u, t_v = surface_jacobian(Q, uv)
        else:
            t_u = t_v = None

        tangent = (tangent_loss_from_jac(t_u, t_v)
                   if (mu_eff > 0 and t_u is not None) else zero)
        svd_loss = (adaptive_svd_loss(model, pids, t_u, t_v,
                                      mode=svd_mode, target=svd_target,
                                      eps=svd_eps)
                    if (svd_eff > 0 and t_u is not None) else zero)

        loss = cd_loss + mu_eff * tangent + svd_eff * svd_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        opt.step()
        sched.step()

        if epoch % log_every == 0 or epoch == 1:
            history['epoch'].append(epoch)
            history['cd'].append(float(cd_loss))
            history['tangent'].append(float(tangent))
            history['svd'].append(float(svd_loss))
            history['total'].append(float(loss))
            history['n_leaves'].append(model.n_patches)
            print(f"  Epoch {epoch:5d}/{epochs}  |  CD={float(cd_loss):.5f}  "
                  f"Tangent={float(tangent):.5f}  SVD={float(svd_loss):.5f}  "
                  f"Total={float(loss):.5f}  μ_eff={float(mu_eff):.3f}  "
                  f"λsvd_eff={float(svd_eff):.3f}  leaves={model.n_patches}  "
                  f"[{time.time() - t0:.1f}s]")

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            extra = dict(checkpoint_extra or {})
            extra.update({'epoch': epoch, 'history': history,
                          'subdivision_events': events})
            _save_ckpt(model, os.path.join(vis_dir, f'checkpoint_{epoch}.pt'), extra)

    print(f"{'─' * 60}\n")
    return history, events


def main():
    ap = argparse.ArgumentParser(description='Adaptive cube-atlas surface fitting')
    ap.add_argument('--file', type=str, default=None)
    ap.add_argument('--shape', type=str, default='box')
    ap.add_argument('--N', type=int, default=50000)
    ap.add_argument('--result_dir', type=str, default='logs/adaptive/test_pretrain')
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--mesh_res', type=int, default=40)
    ap.add_argument('--checkpoint_every', type=int, default=1000)

    # model
    ap.add_argument('--d_features', type=int, default=88)
    ap.add_argument('--base_subdivisions', type=int, default=1,
                    help='Uniform subdivisions applied at construction '
                         '(1 → 4 patches per face = 24 leaves)')
    ap.add_argument('--L', type=int, default=0)
    ap.add_argument('--W', type=int, default=512)
    ap.add_argument('--D', type=int, default=6)
    ap.add_argument('--beta', type=float, default=100.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--M_per_patch', type=int, default=256)

    # pretrain
    ap.add_argument('--pretrain_epochs', type=int, default=2000)
    ap.add_argument('--pretrain_loss', type=str, default='mse', choices=['mse', 'l1'])
    ap.add_argument('--pretrain_only', action='store_true')
    ap.add_argument('--skip_pretrain', action='store_true')
    ap.add_argument('--load_ckpt', type=str, default=None,
                    help='Resume topology+weights from an adaptive checkpoint '
                         '(skips pretraining)')

    # training + regularization
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--mu', type=float, default=1.0,
                    help='Dirichlet tangent regularization weight')
    ap.add_argument('--mu_warmup_epochs', type=int, default=0)
    ap.add_argument('--mu_warmup_delay', type=int, default=0)
    ap.add_argument('--schedule', type=str, default='cosine')

    # SVD (singular-value) Jacobian regularization
    svd_group = ap.add_argument_group('SVD tangent regularization')
    svd_group.add_argument('--lam_svd', type=float, default=0.0,
                           help='Weight of the SVD Jacobian loss (0 disables)')
    svd_group.add_argument('--svd_mode', type=str, default='arap_si',
                           choices=['arap', 'arap_si', 'conformal', 'collapse'],
                           help='arap: (S-target)^2 · arap_si: scale-invariant '
                                'near-isometry · conformal: (S1-S2)^2 · '
                                'collapse: anti-collapse penalty only')
    svd_group.add_argument('--svd_target', type=float, default=1.0,
                           help='Target singular value for --svd_mode arap, in '
                                'FACE-UV units (isometric box fit → 2.0)')
    svd_group.add_argument('--svd_eps', type=float, default=1e-4,
                           help='Anti-collapse floor on singular values')
    svd_group.add_argument('--svd_warmup_epochs', type=int, default=0)
    svd_group.add_argument('--svd_warmup_delay', type=int, default=0)

    # subdivision policy
    ap.add_argument('--subdiv_threshold', type=float, default=0.3,
                    help='Distortion above which a leaf is subdivided (0 disables)')
    ap.add_argument('--subdiv_max_depth', type=int, default=5,
                    help='Absolute quadtree depth cap (base_subdivisions count)')
    ap.add_argument('--subdiv_every', type=int, default=500)
    ap.add_argument('--subdiv_start', type=int, default=1000)
    ap.add_argument('--subdiv_stop', type=int, default=0, help='0 = never stop')
    ap.add_argument('--distortion_mode', type=str, default='area',
                    choices=['area', 'dirichlet', 'conformal'])
    ap.add_argument('--distortion_samples', type=int, default=128)
    ap.add_argument('--max_splits_per_round', type=int, default=0,
                    help='Cap leaves split per check (0 = unlimited)')
    args = ap.parse_args()

    result_dir = utils._get_unique_folder(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f"  Output directory: {result_dir}")

    # ── model ────────────────────────────────────────────────────────────
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location='cpu')
        model = AdaptiveCubeForwardMap.from_checkpoint(ckpt, device=args.device)
        model.train()
        print(f"  Resumed adaptive model from {args.load_ckpt}: "
              f"{model.n_patches} leaves, {model.complex.n_vertices} vertices")
    else:
        model = AdaptiveCubeForwardMap(
            d_features=args.d_features, base_subdivisions=args.base_subdivisions,
            L=args.L, W=args.W, D=args.D, beta=args.beta).to(args.device)
        model.complex.rebuild_flat()

    # ── phase 1: box pretraining ─────────────────────────────────────────
    pretrain_history = None
    if not args.skip_pretrain and not args.load_ckpt:
        pretrain_history = pretrain_box(
            model, epochs=args.pretrain_epochs, M_per_patch=args.M_per_patch,
            lr=args.lr, device=args.device, log_every=args.log_every,
            loss_type=args.pretrain_loss)
        visualize_patch_configuration(
            model, os.path.join(result_dir, 'patch_config_pretrain.png'))
        verts, faces = utils.sample_multi_patch_grid(
            model, resolution=args.mesh_res, device=args.device)
        utils.export_ply(verts, faces,
                         os.path.join(result_dir, 'pretrain_box.ply'))
        _save_ckpt(model, os.path.join(result_dir, 'pretrain_checkpoint.pt'),
                   {'phase': 'pretrain', 'history': pretrain_history,
                    'args': vars(args)})
        if args.pretrain_only:
            plot_history(pretrain_history,
                         os.path.join(result_dir, 'pretrain_history.png'))
            print("  Pretraining complete (pretrain_only). Inspect the seam "
                  "check, pretrain_box.ply and patch_config_pretrain.png.")
            return

    # ── data ─────────────────────────────────────────────────────────────
    downsample_n = None if args.N is not None and args.N < 0 else args.N
    if args.file:
        pts3n, meta = utils.load_point_cloud(args.file, downsample_n=downsample_n)
        input_name = args.file
    else:
        pts3n, meta = utils.make_synthetic_surface(args.shape, n=args.N, noise=0)
        input_name = f'synthetic_{args.shape}'

    checkpoint_extra = {
        'args': vars(args),
        'input_file': input_name,
        'normalization': {
            'center': meta['center'].tolist() if hasattr(meta['center'], 'tolist')
                      else list(meta['center']),
            'scale': float(meta['scale']),
        },
    }

    # ── phase 2: adaptive training ───────────────────────────────────────
    history, events = train_adaptive(
        model, pts3n,
        epochs=args.epochs, M_per_patch=args.M_per_patch, lr=args.lr,
        mu=args.mu, mu_warmup_epochs=args.mu_warmup_epochs,
        mu_warmup_delay=args.mu_warmup_delay, schedule=args.schedule,
        lam_svd=args.lam_svd, svd_mode=args.svd_mode,
        svd_target=args.svd_target, svd_eps=args.svd_eps,
        svd_warmup_epochs=args.svd_warmup_epochs,
        svd_warmup_delay=args.svd_warmup_delay,
        subdiv_threshold=args.subdiv_threshold,
        subdiv_max_depth=args.subdiv_max_depth,
        subdiv_every=args.subdiv_every, subdiv_start=args.subdiv_start,
        subdiv_stop=args.subdiv_stop, distortion_mode=args.distortion_mode,
        distortion_samples=args.distortion_samples,
        max_splits_per_round=args.max_splits_per_round,
        device=args.device, log_every=args.log_every, vis_dir=result_dir,
        checkpoint_every=args.checkpoint_every,
        checkpoint_extra=checkpoint_extra)

    # ── final outputs ────────────────────────────────────────────────────
    model.eval()
    seam = check_seam_continuity(model)
    print(f"  Final seam check: max gap={seam['max_gap']:.3e}  "
          f"mean gap={seam['mean_gap']:.3e}")

    visualize_patch_configuration(
        model, os.path.join(result_dir, 'patch_config_final.png'), pts=pts3n)
    plot_history(history, os.path.join(result_dir, 'history.png'))

    verts, faces = utils.sample_multi_patch_grid(
        model, resolution=args.mesh_res, device=args.device)
    utils.export_ply(verts, faces,
                     os.path.join(result_dir, 'learned_sheet_normalized.ply'))
    verts_orig = utils.unnormalize_vertices(verts, meta)
    utils.export_ply(verts_orig, faces,
                     os.path.join(result_dir, 'learned_sheet.ply'))
    utils.export_obj(verts_orig, faces,
                     os.path.join(result_dir, 'learned_sheet.obj'))

    extra = dict(checkpoint_extra)
    extra.update({'history': history, 'pretrain_history': pretrain_history,
                  'subdivision_events': events, 'seam_check': seam})
    _save_ckpt(model, os.path.join(result_dir, 'checkpoint.pt'), extra)

    with open(os.path.join(result_dir, 'metadata.json'), 'w') as f:
        json.dump({'args': vars(args), 'subdivision_events': events,
                   'final_stats': model.complex.stats(),
                   'seam_check': seam}, f, indent=2)
    print(f"  Run complete → {result_dir}")


if __name__ == '__main__':
    main()