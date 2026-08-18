#!/usr/bin/env python3
# main.py
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

import utils.pc_presegmentation as pc_presegmentation
import utils.correspondence_vis as correspondence_vis
import utils.utils as utils
from model.losses import (boundary_chamfer_loss, chamfer_1d,
                                       chamfer_distance,
                                       chamfer_distance_chunked,
                                       mu_warmup_schedule,
                                       normal_consistency_loss,
                               outer_boundary_rectangle_loss,
                   sample_outer_boundary_correspondence,
                                             surface_jacobian,
                                             tangent_fold_loss,
                                             tangent_loss_from_jac)
from model.model import (FeatureComplex, ForwardMap, InverseMap, MultiPatchForwardMap,
           MultiPatchInverseMap, PositionalEncoding, SkipMLP,
       TwoSheetForwardMap, SixSheetForwardMap)
try:
    import open3d as o3d
except ImportError:
    o3d = None

# Multi-patch training with a single or two-sheet atlas.
def _build_forward_model(atlas_mode: str,
                         n_patches: int,
                         d_features: int,
                         L: int,
                         W: int,
                         D: int,
                         beta: float,
                         device: str,
                         two_sheet_side_rows: int = 2,
                         two_sheet_side_cols: int = 2,
                         six_sheet_face_rows: int = 2,
                         six_sheet_face_cols: int = 2):
    """Create the forward model for the selected atlas configuration."""
    if atlas_mode == 'single_sheet':
        n_rows, n_cols = pc_presegmentation.compute_grid_dims(n_patches)
        F = MultiPatchForwardMap(n_rows, n_cols, d_features,
                                 L=L, W=W, D=D, beta=beta).to(device)
        atlas_info = {
            'atlas_mode': atlas_mode,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'actual_n_patches': n_rows * n_cols,
            'n_sides': 1,
            'patches_per_side': n_rows * n_cols,
        }
        return F, atlas_info

    if atlas_mode == 'two_sheet':
        n_rows = two_sheet_side_rows
        n_cols = two_sheet_side_cols
        F = TwoSheetForwardMap(n_rows=n_rows, n_cols=n_cols, d_features=d_features,
                               L=L, W=W, D=D, beta=beta).to(device)
        atlas_info = {
            'atlas_mode': atlas_mode,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'actual_n_patches': F.n_patches,
            'n_sides': F.n_sides,
            'patches_per_side': F.patches_per_side,
        }
        return F, atlas_info

    if atlas_mode == 'six_sheet':
        n_rows = six_sheet_face_rows
        n_cols = six_sheet_face_cols
        F = SixSheetForwardMap(n_rows=n_rows, n_cols=n_cols, d_features=d_features,
                               L=L, W=W, D=D, beta=beta).to(device)
        atlas_info = {
            'atlas_mode': atlas_mode,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'actual_n_patches': F.n_patches,
            'n_sides': F.n_sides,
            'patches_per_side': F.patches_per_side,
        }
        return F, atlas_info

    raise ValueError(f"Unknown atlas_mode: {atlas_mode}")


def _run_presegmentation(pts3n: np.ndarray,
                         atlas_mode: str,
                         n_rows: int,
                         n_cols: int,
                         two_sheet_split_axis: int = 2,
                         two_sheet_side_axes=(0, 1)):
    """Run the segmentation strategy matching the selected atlas configuration."""
    if atlas_mode == 'single_sheet':
        assignments, grid_topology, patch_params = pc_presegmentation.axis_aligned_grid_segmentation(
            pts3n, n_rows, n_cols
        )
        return assignments, grid_topology, patch_params, None

    if atlas_mode == 'two_sheet':
        assignments, grid_topology, patch_params, side_assignments = (
            pc_presegmentation.two_sheet_axis_aligned_segmentation(
                pts3n,
                n_patches_u=n_rows,
                n_patches_v=n_cols,
                split_axis=two_sheet_split_axis,
                side_axes=two_sheet_side_axes,
            )
        )
        return assignments, grid_topology, patch_params, side_assignments

    raise ValueError(f"Unknown atlas_mode: {atlas_mode}")


def train_multi_patch(pts3n: np.ndarray,
                      n_patches: int = 4,
                      d_features: int = 64,
                      epochs: int = 5000,
                      M: int = 4096,
                      M_per_patch: int = 512,
                      W: int = 256,
                      D: int = 6,
                      L: int = 8,
                      L_inv: int = 4,
                      lr: float = 1e-3,
                      mu: float = 0.5,
                      mu_warmup_epochs: int = 0,
                      mu_warmup_delay: int = 0,
                      schedule: str = 'cosine',
                      gamma: float = 1.0,
                      lam: float = 1.0,
                      lam2: float = 1.0,
                      lambda_bcd: float = 0.1,
                      lambda_outer_boundary: float = 1.0,
                      outer_boundary_samples: int = 64,
                      outer_boundary_loss_type: str = 'l1',
                      beta: float = 5.0,
                      device: str = 'cuda',
                      log_every: int = 200,
                      save_patch_vis: bool = True,
                      vis_dir: str = None,
                      checkpoint_every: int = 1000,
                      checkpoint_payload: dict = None,
                      output_psr_mesh_path: str = None,
                      normals: np.ndarray = None,
                      reg_every: int = 1,
                      pretrained_F_state: dict = None,
                      pretrained_ckpt_path: str = None,
                      correspondence_dir: str = None,
                      save_correspondence_every: int = 0,
                      save_boundary_debug_every: int = 0,
                      correspondence_max_lines: int = 300,
                      correspondence_line_segment: str = 't_to_q',
                      atlas_mode: str = 'single_sheet',
                      no_presplit: bool = False,
                      two_sheet_side_rows: int = 2,
                      two_sheet_side_cols: int = 2,
                      two_sheet_split_axis: int = 2,
                      two_sheet_side_axes=(0, 1),
                      six_sheet_face_rows: int = 2,
                      six_sheet_face_cols: int = 2):
    """
    Train the multi-patch model and inverse map.

    Expensive regularizers can be evaluated every `reg_every` steps instead of
    every iteration.
    """
    F, atlas_info = _build_forward_model(
        atlas_mode=atlas_mode,
        n_patches=n_patches,
        d_features=d_features,
        L=L,
        W=W,
        D=D,
        beta=beta,
        device=device,
        two_sheet_side_rows=two_sheet_side_rows,
        two_sheet_side_cols=two_sheet_side_cols,
        six_sheet_face_rows=six_sheet_face_rows,
        six_sheet_face_cols=six_sheet_face_cols,
    )
    n_rows = atlas_info['n_rows']
    n_cols = atlas_info['n_cols']
    actual_n_patches = atlas_info['actual_n_patches']

    if atlas_mode == 'single_sheet':
        print(f"  Grid layout: {n_rows} rows × {n_cols} cols = {actual_n_patches} patches")
    elif atlas_mode == 'two_sheet':
        print(f"  Two-sheet layout: {atlas_info['n_sides']} sides × ({n_rows} rows × {n_cols} cols) = {actual_n_patches} patches")
    else:
        print(f"  Six-sheet layout: {atlas_info['n_sides']} faces × ({n_rows} rows × {n_cols} cols) = {actual_n_patches} patches")

    if gamma > 0:
        assert normals is not None, "normals must be provided when gamma > 0"
        assert normals.shape == pts3n.shape, \
            f"Normals shape {normals.shape} != points shape {pts3n.shape}"
        print(f"  Normal constraint active (γ={gamma}), normals shape: {normals.shape}")

    if no_presplit and atlas_mode not in ('two_sheet', 'six_sheet'):
        raise ValueError("--no_presplit is currently supported only with atlas_mode='two_sheet' or 'six_sheet'")

    if no_presplit:
        assignments = np.arange(pts3n.shape[0], dtype=np.int32) % actual_n_patches
        grid_topology = np.stack([
            np.arange(atlas_info['patches_per_side'], dtype=np.int32).reshape(n_rows, n_cols),
            (np.arange(atlas_info['patches_per_side'], dtype=np.int32) + atlas_info['patches_per_side']).reshape(n_rows, n_cols),
        ], axis=0)
        patch_params = np.zeros((pts3n.shape[0], 2), dtype=np.float32)
        side_assignments = np.full(pts3n.shape[0], -1, dtype=np.int32)
    else:
        assignments, grid_topology, patch_params, side_assignments = _run_presegmentation(
            pts3n=pts3n,
            atlas_mode=atlas_mode,
            n_rows=n_rows,
            n_cols=n_cols,
            two_sheet_split_axis=two_sheet_split_axis,
            two_sheet_side_axes=two_sheet_side_axes,
        )

    # Patch visualization.
    if save_patch_vis:
        if vis_dir is None:
            vis_dir = os.getcwd()
        os.makedirs(vis_dir, exist_ok=True)

        vis_cap = 200_000
        if pts3n.shape[0] > vis_cap:
            vsub = np.random.choice(pts3n.shape[0], vis_cap, replace=False)
            pts_vis = pts3n[vsub]
            asg_vis = assignments[vsub]
            if (hasattr(patch_params, 'shape')
                    and patch_params.shape[0] == pts3n.shape[0]):
                pp_vis = patch_params[vsub]
            else:
                pp_vis = patch_params
        else:
            pts_vis = pts3n
            asg_vis = assignments
            pp_vis = patch_params

        try:
            if atlas_mode == 'single_sheet':
                utils._visualize_patch_assignments(
                    pts_vis, asg_vis, grid_topology, pp_vis,
                    n_rows, n_cols,
                    save_path=os.path.join(vis_dir, 'patch_assignments.png')
                )
                utils._visualize_patch_assignments_3d(
                    pts_vis, asg_vis, grid_topology, n_rows, n_cols,
                    save_path=os.path.join(vis_dir, 'patch_assignments_3d.png')
                )
            else:
                side_vis = side_assignments[vsub] if pts3n.shape[0] > vis_cap else side_assignments
                utils._visualize_two_sheet_patch_assignments(
                    pts_vis, asg_vis, side_vis, grid_topology, pp_vis,
                    n_rows, n_cols,
                    save_path=os.path.join(vis_dir, 'patch_assignments_two_sheet.png')
                )
                utils._visualize_two_sheet_patch_assignments_3d(
                    pts_vis, asg_vis, side_vis, n_rows, n_cols,
                    save_path=os.path.join(vis_dir, 'patch_assignments_two_sheet_3d.png')
                )
        except Exception as e:
            print(f"  [warn] patch visualization skipped ({type(e).__name__}: {e})")

    # Keep only active patches with enough assigned points, unless using global no-presplit training.
    if no_presplit:
        active_ids = list(range(actual_n_patches))
        active_pts = None
        active_nrm = None
        K = len(active_ids)
        active_idx_dev = torch.tensor(active_ids, dtype=torch.long, device=device)
        pidx_flat = active_idx_dev.repeat_interleave(M_per_patch)
        lengths = None
    else:
        active_ids = []
        active_pts = []   # list of (N_k, 3) CPU tensors
        active_nrm = []   # list of (N_k, 3) CPU tensors or None
        for k in range(actual_n_patches):
            mask = assignments == k
            pts_k = pts3n[mask]
            if pts_k.shape[0] >= 10:
                active_ids.append(k)
                active_pts.append(torch.tensor(pts_k, dtype=torch.float32))
                if gamma > 0:
                    active_nrm.append(torch.tensor(normals[mask], dtype=torch.float32))
                else:
                    active_nrm.append(None)

        K = len(active_ids)
        if K == 0:
            raise RuntimeError("No active patches (all have < 10 points). "
                               "Reduce --n_patches or add more points.")

        active_idx_dev = torch.tensor(active_ids, dtype=torch.long, device=device)
        pidx_flat = active_idx_dev.repeat_interleave(M_per_patch)
        lengths = [p.shape[0] for p in active_pts]

    # Models.
    use_inverse_map = (lam > 0) or (lam2 > 0)
    G = None
    if use_inverse_map:
        if atlas_mode != 'single_sheet':
            raise NotImplementedError(
                "Inverse-map training is not yet implemented for atlas_mode != 'single_sheet'. "
                "Use --lam 0 --lam2 0 for multi-sheet experiments."
            )
        G = MultiPatchInverseMap(F.complex, d_features=d_features,
                                 L=L_inv, W=W, D=D, beta=beta).to(device)

    if pretrained_F_state is not None:
        missing, unexpected = F.load_state_dict(pretrained_F_state, strict=False)
        print("  Loaded pretrained F weights into multi-patch training")
        if pretrained_ckpt_path is not None:
            print(f"    Source checkpoint: {pretrained_ckpt_path}")
        if missing:
            print(f"    Missing keys: {len(missing)}")
        if unexpected:
            print(f"    Unexpected keys: {len(unexpected)}")

    print(f"  Model device: {next(F.parameters()).device}")
    n_params_F = sum(p.numel() for p in F.parameters())
    n_params_G = sum(p.numel() for p in G.parameters()) if G is not None else 0
    n_vertex = F.complex.vertex_features.numel()
    print(f"  F total params: {n_params_F:,}")
    if atlas_mode == 'single_sheet':
        print(f"    Vertex features (shared): {n_vertex:,} "
              f"({(n_rows+1)*(n_cols+1)} vertices × {d_features}d)")
    else:
        print(f"    Vertex features (two-sheet shared-boundary layout): {n_vertex:,}")
    print(f"    Shared decoder (+ global-UV PE, L={L}): {n_params_F - n_vertex:,}")
    if G is not None:
        print(f"  G encoder params (L_inv={L_inv}): {n_params_G:,}")
    else:
        print("  G encoder params: skipped (λ₁=0 and λ₂=0)")
    print(f"  Total unique params: {n_params_F + n_params_G:,}")
    print("  F parameter breakdown:")
    for name, param in F.named_parameters():
        print(f"    {name:<40} shape={tuple(param.shape)} "
              f"requires_grad={param.requires_grad}")

    opt_params = list(F.parameters()) + (list(G.parameters()) if G is not None else [])
    opt = torch.optim.Adam(opt_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = {'cd': [], 'cycle': [], 'param': [], 'tangent': [], 'normal': [],
               'outer_boundary': [],
               'mu_eff': [], 'total': [], 'epoch': []}
    

    print(f"\n{'─'*60}")
    print(f"  Multi-Patch Feature Complex Training (VECTORIZED)")
    print(f"  Grid={n_rows}×{n_cols}  active_patches={K}/{actual_n_patches}")
    if no_presplit:
        print("  Training mode: no-presplit global Chamfer over full target cloud")
    print(f"  d_features={d_features}  W={W}  D={D}  L(fwd/global-UV)={L}  L_inv={L_inv}  β={beta}")
    print(f"  M_per_patch={M_per_patch}  batch/step={K*M_per_patch}  reg_every={reg_every}")
    print(f"  μ={mu}  γ={gamma}  λ₁={lam}  λ₂={lam2}  λ_outer={lambda_outer_boundary}")
    if mu_warmup_epochs > 0 and mu > 0:
        print(f"  μ warmup: {mu_warmup_schedule} ramp over {mu_warmup_epochs} epochs "
              f"(0.0 → {mu})")
    print(f"  Epochs={epochs}  lr={lr}  device={device}")
    print(f"{'─'*60}")
    t0 = time.time()

    checkpoint_dir = vis_dir if vis_dir is not None else os.getcwd()
    os.makedirs(checkpoint_dir, exist_ok=True)
    if correspondence_dir is None:
        correspondence_dir = os.path.join(checkpoint_dir, 'correspondences')
    if save_correspondence_every > 0:
        os.makedirs(correspondence_dir, exist_ok=True)
    boundary_debug_dir = os.path.join(checkpoint_dir, 'boundary_debug')
    if save_boundary_debug_every > 0:
        os.makedirs(boundary_debug_dir, exist_ok=True)

    def _save_epoch_checkpoint(epoch: int):
        if checkpoint_every <= 0 or epoch % checkpoint_every != 0:
            return

        epoch_tag = f'{epoch}'
        epoch_ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch_tag}.pt')
        payload = {
            'mode': 'multi_patch',
            'epoch': epoch,
            'F_state': F.state_dict(),
            'G_state': G.state_dict() if G is not None else None,
            'args': (checkpoint_payload or {}).get('args', {
                'n_patches': n_patches,
                'd_features': d_features,
                'epochs': epochs,
                'M': M,
                'M_per_patch': M_per_patch,
                'W': W,
                'D': D,
                'L': L,
                'L_inv': L_inv,
                'lr': lr,
                'mu': mu,
                'gamma': gamma,
                'lam': lam,
                'lam2': lam2,
                'lambda_bcd': lambda_bcd,
                'lambda_outer_boundary': lambda_outer_boundary,
                'outer_boundary_samples': outer_boundary_samples,
                'outer_boundary_loss_type': outer_boundary_loss_type,
                'beta': beta,
                'device': device,
                'log_every': log_every,
                'save_patch_vis': save_patch_vis,
                'reg_every': reg_every,
                'atlas_mode': atlas_mode,
                'two_sheet_side_rows': two_sheet_side_rows,
                'two_sheet_side_cols': two_sheet_side_cols,
                'two_sheet_split_axis': two_sheet_split_axis,
                'two_sheet_side_axes': list(two_sheet_side_axes),
                'six_sheet_face_rows': six_sheet_face_rows,
                'six_sheet_face_cols': six_sheet_face_cols,
            }),
            'grid_dims': (n_rows, n_cols),
            'atlas_info': atlas_info,
            'history': history,
        }
        if checkpoint_payload is not None:
            for key in ('normalization', 'input_file', 'result_dir'):
                if key in checkpoint_payload:
                    payload[key] = checkpoint_payload[key]

        torch.save(payload, epoch_ckpt_path)
        print(f"    Checkpoint → {epoch_ckpt_path}")

    def _save_correspondence_snapshot(epoch: int,
                                      q_batch: torch.Tensor,
                                      tgt_batch: torch.Tensor,
                                      dist_batch: torch.Tensor):
        if save_correspondence_every <= 0 or epoch % save_correspondence_every != 0:
            return

        epoch_dir = os.path.join(correspondence_dir, f'epoch_{epoch:05d}')
        q_np = q_batch.detach().cpu().numpy()
        tgt_np = tgt_batch.detach().cpu().numpy()

        if no_presplit:
            correspondence_vis.export_global_correspondence_ply(
                q_points=q_np.reshape(-1, 3),
                t_points=tgt_np.reshape(-1, 3),
                output_ply_path=os.path.join(epoch_dir, 'global_correspondence.ply'),
                plot_direction=correspondence_line_segment,
            )
            return

        dist_np = dist_batch.detach().cpu().numpy()
        correspondence_vis.export_patchwise_chamfer_correspondences(
            q_batches=q_np,
            t_batches=tgt_np,
            distance_batches=dist_np,
            output_dir=epoch_dir,
            patch_ids=np.asarray(active_ids, dtype=np.int32),
            max_lines=correspondence_max_lines,
            plot_direction=correspondence_line_segment,
        )
        if epoch == epochs:
            correspondence_vis.export_combined_correspondence_ply(
                q_batches=q_np,
                t_batches=tgt_np,
                distance_batches=dist_np,
                output_ply_path=os.path.join(epoch_dir, 'all_patches_combined.ply'),
                plot_direction=correspondence_line_segment,
            )

    def _save_boundary_debug_snapshot(epoch: int):
        if save_boundary_debug_every <= 0 or epoch % save_boundary_debug_every != 0:
            return

        epoch_dir = os.path.join(boundary_debug_dir, f'epoch_{epoch:05d}')
        boundary_data = sample_outer_boundary_correspondence(
            F_model=F,
            n_boundary_samples=outer_boundary_samples,
            device=device,
        )
        correspondence_vis.export_boundary_correspondence_debug(
            model_batches=boundary_data['model_points'].detach().cpu().numpy(),
            target_batches=boundary_data['target_points'].detach().cpu().numpy(),
            patch_ids=boundary_data['patch_ids'].detach().cpu().numpy(),
            edge_names=boundary_data['edge_names'],
            output_dir=epoch_dir,
            max_lines=min(correspondence_max_lines, outer_boundary_samples),
        )

    zero = torch.tensor(0.0, device=device)
    vertex_features_init = F.complex.vertex_features.detach().clone()


    # Optimize the forward and inverse maps.
    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        # Build the target batch on CPU, then transfer once to the device.
        if no_presplit:
            target_batch_size = K * M_per_patch
            ridx = torch.randint(0, pts3n.shape[0], (target_batch_size,))
            tgt_flat = torch.tensor(pts3n[ridx], dtype=torch.float32, device=device)
            tgt = tgt_flat.reshape(1, target_batch_size, 3)
            if gamma > 0:
                tgt_nrm = torch.tensor(normals[ridx], dtype=torch.float32, device=device).reshape(1, target_batch_size, 3)
        else:
            pts_batch = torch.empty(K, M_per_patch, 3)
            nrm_batch = torch.empty(K, M_per_patch, 3) if gamma > 0 else None
            for i in range(K):
                ridx = torch.randint(0, lengths[i], (M_per_patch,))
                pts_batch[i] = active_pts[i][ridx]
                if gamma > 0:
                    nrm_batch[i] = active_nrm[i][ridx]

            tgt = pts_batch.to(device)
            tgt_flat = tgt.reshape(-1, 3)
            if gamma > 0:
                tgt_nrm = nrm_batch.to(device)

        # Single batched forward pass.
        uv_flat = torch.rand(K * M_per_patch, 2, device=device, requires_grad=True)
        Q_flat = F(pidx_flat, uv_flat)
        Q = Q_flat.reshape(K, M_per_patch, 3)

        # Loss 1: Chamfer distance.
        if no_presplit:
            cd_loss = chamfer_distance_chunked(Q_flat, tgt_flat, chunk_size=min(2048, K * M_per_patch))
            D = None
        else:
            D = torch.cdist(tgt, Q)
            cd_loss = D.min(dim=2).values.mean() + D.min(dim=1).values.mean()
        # cd_loss = D.min(dim=2).values.mean() # Map only xyz ->  uv


        # Loss 2: cycle consistency.
        if lam > 0:
            if no_presplit:
                raise NotImplementedError("Cycle consistency is not supported with --no_presplit")
            uv_inv = G(pidx_flat, tgt_flat)
            P_recon = F(pidx_flat, uv_inv)
            P_recon = P_recon.reshape(K, M_per_patch, 3)
            cycle_loss = torch.stack([
                chamfer_distance_chunked(P_recon[k], tgt[k], chunk_size=min(1024, M_per_patch))
                for k in range(K)
            ]).mean()
        else:
            cycle_loss = zero

        # Loss 3: inverse cycle consistency.
        if lam2 > 0:
            if no_presplit:
                raise NotImplementedError("Inverse cycle consistency is not supported with --no_presplit")
            uv_recon = G(pidx_flat, Q_flat)
            uv_recon = uv_recon.reshape(K, M_per_patch, 2)
            uv_grid = uv_flat.reshape(K, M_per_patch, 2)
            param_loss = torch.stack([
                chamfer_distance_chunked(uv_recon[k], uv_grid[k], chunk_size=min(1024, M_per_patch))
                for k in range(K)
            ]).mean()
        else:
            param_loss = zero

        # Loss 4 and 5: tangent and normal regularization.
        do_reg = (mu > 0 or gamma > 0) and (epoch % reg_every == 0)
        mu_eff = mu_warmup_schedule(epoch, mu_warmup_epochs, mu,
                                    schedule=schedule, delay_epochs=mu_warmup_delay) if mu > 0 else 0.0
        if do_reg:
            t_u, t_v = surface_jacobian(Q_flat, uv_flat)

            if mu_eff > 0:
                tangent_loss = tangent_loss_from_jac(t_u, t_v)
            else:
                tangent_loss = torch.zeros((), device=Q.device, dtype=Q.dtype)

            if gamma > 0:
                if no_presplit:
                    raise NotImplementedError("Normal consistency is not supported with --no_presplit")
                n_surf = torch.cross(t_u, t_v, dim=-1)
                n_surf = n_surf / (n_surf.norm(dim=-1, keepdim=True) + 1e-8)
                n_surf = n_surf.reshape(K, M_per_patch, 3)
                # Match each generated point to its nearest target in the patch.
                nn_idx = D.argmin(dim=1)
                n_target = torch.gather(
                    tgt_nrm, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 3))
                cos = torch.sum(n_surf * n_target, dim=-1)
                normal_loss = (1.0 - cos).mean()
            else:
                normal_loss = zero
        else:
            tangent_loss = zero
            normal_loss = zero

        if lambda_outer_boundary > 0:
            outer_boundary_loss = outer_boundary_rectangle_loss(
                F_model=F,
                n_boundary_samples=outer_boundary_samples,
                loss_type=outer_boundary_loss_type,
                device=device,
            )
        else:
            outer_boundary_loss = zero

        # Total loss.
        loss = (cd_loss
                + lam * cycle_loss
                + lam2 * param_loss
                + mu_eff * tangent_loss
                + gamma * normal_loss
                + lambda_outer_boundary * outer_boundary_loss)
        loss.backward()

        nn.utils.clip_grad_norm_(opt_params, 1.0)
        opt.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            history['epoch'].append(epoch)
            history['cd'].append(float(cd_loss))
            history['cycle'].append(float(cycle_loss))
            history['param'].append(float(param_loss))
            history['tangent'].append(float(tangent_loss))
            history['normal'].append(float(normal_loss))
            history['outer_boundary'].append(float(outer_boundary_loss))
            # history['mu_eff'].append(float(mu_eff))
            history['total'].append(float(loss))

            elapsed = time.time() - t0
            # mu_str = f"  μ_eff={float(mu_eff):.4f}" if (mu_warmup_epochs > 0 and mu > 0) else ""
            print(f"  Epoch {epoch:5d}/{epochs}  |  "
                  f"CD={float(cd_loss):.5f}  "
                  f"Cycle={float(cycle_loss):.5f}  "
                  f"Param={float(param_loss):.5f}  "
                  f"Tangent={float(tangent_loss):.5f}  "
                  f"Normal={float(normal_loss):.5f}  "
                  f"Outer={float(outer_boundary_loss):.5f}  "
                  f"Total={float(loss):.5f}"
                  f"μ_eff={float(mu_eff):.4f}  "
                  f"[{elapsed:.1f}s]")
            # vf = F.complex.vertex_features
            # vf_delta = (vf.detach() - vertex_features_init).norm().item()
            # vf_grad = (vf.grad.norm().item() if vf.grad is not None else 0.0)
            # vf_first = vf.detach()[:, 0].cpu().tolist()
            # print(f"    vertex_features: mean={vf.detach().mean().item():.6f} "
            #     f"std={vf.detach().std().item():.6f} "
            #     f"grad_norm={vf_grad:.6e} "
            #     f"delta_from_init={vf_delta:.6e}")
            # print("    vertex_features[:, 0]="
            #     + ", ".join(f"v{i}={val:.6f}" for i, val in enumerate(vf_first)))

            _save_correspondence_snapshot(epoch, Q, tgt, D)
            _save_boundary_debug_snapshot(epoch)
            _save_epoch_checkpoint(epoch)

    print(f"{'─'*60}\n")
    return F, G, history, assignments, active_ids


def pretrain_multi_patch_flat_sheet(n_patches: int = 4,
                                    d_features: int = 88,
                                    epochs: int = 2000,
                                    M_per_patch: int = 4096,
                                    W: int = 512,
                                    D: int = 6,
                                    L: int = 0,
                                    lr: float = 1e-3,
                                    beta: float = 100.0,
                                    device: str = 'cuda',
                                    log_every: int = 200,
                                    noise: float = 0.0,
                                    lam_jac: float = 0.001,
                                    loss_type: str = 'mse',
                                    atlas_mode: str = 'single_sheet',
                                    two_sheet_side_rows: int = 2,
                                    two_sheet_side_cols: int = 2):
    """
    Pretrain only the multi-patch forward map F. 

    The target is a real sampled 3D point cloud lying on z=0 in normalized
    coordinates. Each patch learns its corresponding region of the plane using
    direct pointwise supervision.

    Mapping F to plane (sample (u, v) from plane and train with MSE/L1 F(z(u, v)) = (x, y, z) = (u, v, 0))
    """
    F, atlas_info = _build_forward_model(
        atlas_mode=atlas_mode,
        n_patches=n_patches,
        d_features=d_features,
        L=L,
        W=W,
        D=D,
        beta=beta,
        device=device,
        two_sheet_side_rows=two_sheet_side_rows,
        two_sheet_side_cols=two_sheet_side_cols,
    )
    n_rows = atlas_info['n_rows']
    n_cols = atlas_info['n_cols']
    actual_n_patches = atlas_info['actual_n_patches']

    opt = torch.optim.Adam(F.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = {'plane': [], 'total': [], 'epoch': []}

    print(f"\n{'─'*60}")
    print("  Multi-Patch Flat-Sheet Pretraining")
    if atlas_mode == 'single_sheet':
        print(f"  Grid={n_rows}×{n_cols}  patches={actual_n_patches}")
    else:
        print(f"  Two-sheet grid={atlas_info['n_sides']} × ({n_rows}×{n_cols})  patches={actual_n_patches}")
    print(f"  d_features={d_features}  W={W}  D={D}  L={L}  β={beta}")
    print(f"  M_per_patch={M_per_patch}  batch/step={actual_n_patches*M_per_patch}")
    print(f"  Epochs={epochs}  lr={lr}  device={device}  loss={loss_type}")
    print(f"{'─'*60}")
    t0 = time.time()

    if loss_type == 'l1':
        point_loss_fn = nn.L1Loss()
    elif loss_type == 'mse':
        point_loss_fn = nn.MSELoss()
    elif loss_type == 'cd':
        point_loss_fn = None
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'mse', 'l1', or 'cd'.")

    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        # Sample each local uv uniformly in [0, 1]² for each patch
        uv_local = torch.rand(actual_n_patches, M_per_patch, 2, device=device, requires_grad=True)
        # create a flat array of patch IDs repeated for each sample in the patch
        patch_ids = torch.arange(actual_n_patches, device=device, dtype=torch.long)
        patch_ids_flat = patch_ids.repeat_interleave(M_per_patch)
        uv_flat = uv_local.reshape(-1, 2)

        row = patch_ids_flat // n_cols
        col = patch_ids_flat % n_cols

        u_local = uv_flat[:, 0:1]
        v_local = uv_flat[:, 1:2]
        global_u = (row.unsqueeze(1).float() + u_local) / n_rows
        global_v = (col.unsqueeze(1).float() + v_local) / n_cols

        target = torch.cat([
            2.0 * global_u - 1.0,
            2.0 * global_v - 1.0,
            torch.zeros_like(global_u)
        ], dim=1)

        if noise > 0:
            target = target + noise * torch.randn_like(target)
            target[:, 2] = 0.0

        pred = F(patch_ids_flat, uv_flat)
        if loss_type == 'cd':
            pred_patch = pred.reshape(actual_n_patches, M_per_patch, 3)
            target_patch = target.reshape(actual_n_patches, M_per_patch, 3)
            plane_loss = torch.stack([
                chamfer_distance(pred_patch[k], target_patch[k])
                for k in range(actual_n_patches)
            ]).mean()

            if lam_jac > 0:
                t_u, t_v = surface_jacobian(pred, uv_flat, "arap")
                jac_loss = tangent_loss_from_jac(t_u, t_v)
                plane_loss = plane_loss + lam_jac * jac_loss
        else:
            plane_loss = point_loss_fn(pred, target)
        plane_loss.backward()

        nn.utils.clip_grad_norm_(list(F.parameters()), 1.0)
        opt.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            history['epoch'].append(epoch)
            history['plane'].append(float(plane_loss))
            history['total'].append(float(plane_loss))

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:5d}/{epochs}  |  "
                  f"Plane={float(plane_loss):.6f}  "
                  f"[{elapsed:.1f}s]")

    print(f"{'─'*60}\n")
    return F, history


def pretrain_multi_patch_closed_shape(shape: str = 'sphere',
                                      n_patches: int = 4,
                                      d_features: int = 88,
                                      epochs: int = 2000,
                                      M_per_patch: int = 4096,
                                      W: int = 512,
                                      D: int = 6,
                                      L: int = 0,
                                      lr: float = 1e-3,
                                      beta: float = 100.0,
                                      device: str = 'cuda',
                                      log_every: int = 200,
                                      mu: float = 0.0,
                                      mu_warmup_epochs: int = 0,
                                      mu_warmup_delay: int = 0,
                                      schedule: str = 'cosine',
                                      beta_shape_samples: int = 200000,
                                      atlas_mode: str = 'two_sheet',
                                      two_sheet_side_rows: int = 2,
                                      two_sheet_side_cols: int = 2,
                                      two_sheet_split_axis: int = 2,
                                      two_sheet_side_axes=(0, 1),
                                      no_presplit: bool = False,
                                      six_sheet_face_rows: int = 2,
                                      six_sheet_face_cols: int = 2,
                                      face_aware_box_supervision: bool = False):
    """
    Pretrain the multi-patch forward map on a synthetic closed shape.

    This is intended for closed-surface atlas initialization, especially the
    two-sheet setup used for sphere-like reconstruction. The model is first
    trained to fit a synthetic closed shape before being fine-tuned on the real
    point cloud.
    """
    if atlas_mode not in ('two_sheet', 'six_sheet'):
        raise ValueError(
            "pretrain_multi_patch_closed_shape currently supports atlas_mode='two_sheet' or 'six_sheet' only"
        )

    if atlas_mode == 'six_sheet' and shape != 'box' and face_aware_box_supervision:
        raise ValueError("face-aware box supervision is only supported for atlas_mode='six_sheet' with shape='box'")

    pts3n, synthetic_meta = utils.make_synthetic_surface(shape=shape, n=beta_shape_samples, noise=0)

    F, atlas_info = _build_forward_model(
        atlas_mode=atlas_mode,
        n_patches=n_patches,
        d_features=d_features,
        L=L,
        W=W,
        D=D,
        beta=beta,
        device=device,
        two_sheet_side_rows=two_sheet_side_rows,
        two_sheet_side_cols=two_sheet_side_cols,
        six_sheet_face_rows=six_sheet_face_rows,
        six_sheet_face_cols=six_sheet_face_cols,
    )
    n_rows = atlas_info['n_rows']
    n_cols = atlas_info['n_cols']
    actual_n_patches = atlas_info['actual_n_patches']

    if atlas_mode == 'six_sheet' and face_aware_box_supervision:
        face_points_np = synthetic_meta.get('face_points')
        if face_points_np is None:
            raise ValueError("Synthetic box metadata is missing face_points required for face-aware supervision")
        patch_face_points_np = utils._box_face_points_to_patch_targets(
            face_points=face_points_np,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        active_ids = list(range(actual_n_patches))
        K = len(active_ids)
        active_idx_dev = torch.tensor(active_ids, dtype=torch.long, device=device)
        pidx_flat = active_idx_dev.repeat_interleave(M_per_patch)
        lengths = None
        active_pts = None
    elif no_presplit:
        active_ids = list(range(actual_n_patches))
        active_pts = None
        K = len(active_ids)
        active_idx_dev = torch.tensor(active_ids, dtype=torch.long, device=device)
        pidx_flat = active_idx_dev.repeat_interleave(M_per_patch)
        lengths = None
    else:
        assignments, _, _, _ = _run_presegmentation(
            pts3n=pts3n,
            atlas_mode=atlas_mode,
            n_rows=n_rows,
            n_cols=n_cols,
            two_sheet_split_axis=two_sheet_split_axis,
            two_sheet_side_axes=two_sheet_side_axes,
        )

        active_ids = []
        active_pts = []
        for k in range(actual_n_patches):
            mask = assignments == k
            pts_k = pts3n[mask]
            if pts_k.shape[0] >= 10:
                active_ids.append(k)
                active_pts.append(torch.tensor(pts_k, dtype=torch.float32))

        K = len(active_ids)
        if K == 0:
            raise RuntimeError("Closed-shape pretraining found no active patches")

        active_idx_dev = torch.tensor(active_ids, dtype=torch.long, device=device)
        pidx_flat = active_idx_dev.repeat_interleave(M_per_patch)
        lengths = [p.shape[0] for p in active_pts]

    opt = torch.optim.Adam(F.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = {'cd': [], 'tangent': [], 'mu_eff': [], 'total': [], 'epoch': []}

    print(f"\n{'─'*60}")
    print(f"  Multi-Patch Closed-Shape Pretraining")
    print(f"  Shape={shape}  atlas={atlas_mode}")
    if atlas_mode == 'two_sheet':
        print(f"  Two-sheet grid={atlas_info['n_sides']} × ({n_rows}×{n_cols})  patches={actual_n_patches}")
    else:
        print(f"  Six-sheet grid={atlas_info['n_sides']} × ({n_rows}×{n_cols})  patches={actual_n_patches}")
    print(f"  d_features={d_features}  W={W}  D={D}  L={L}  β={beta}")
    print(f"  M_per_patch={M_per_patch}  active_patches={K}/{actual_n_patches}")
    if atlas_mode == 'six_sheet' and face_aware_box_supervision:
        print("  Pretraining mode: direct face-aware, patch-wise supervision on cube faces")
    elif no_presplit:
        print("  Pretraining mode: no-presplit global Chamfer over full synthetic cloud")
    print(f"  μ={mu}  Epochs={epochs}  lr={lr}  device={device}")
    print(f"{'─'*60}")
    t0 = time.time()

    zero = torch.tensor(0.0, device=device)

    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        if atlas_mode == 'six_sheet' and face_aware_box_supervision:
            tgt_batches = []
            for patch_id in active_ids:
                pts_patch = patch_face_points_np[int(patch_id)]
                ridx = np.random.randint(0, pts_patch.shape[0], size=M_per_patch)
                tgt_batches.append(torch.tensor(pts_patch[ridx], dtype=torch.float32))
            tgt = torch.stack(tgt_batches, dim=0).to(device)
            tgt_flat = tgt.reshape(-1, 3)
        elif no_presplit:
            target_batch_size = K * M_per_patch
            ridx = torch.randint(0, pts3n.shape[0], (target_batch_size,))
            tgt_flat = torch.tensor(pts3n[ridx], dtype=torch.float32, device=device)
            tgt = tgt_flat.reshape(1, target_batch_size, 3)
        else:
            pts_batch = torch.empty(K, M_per_patch, 3)
            for i in range(K):
                ridx = torch.randint(0, lengths[i], (M_per_patch,))
                pts_batch[i] = active_pts[i][ridx]

            tgt = pts_batch.to(device)
            tgt_flat = tgt.reshape(-1, 3)

        uv_flat = torch.rand(K * M_per_patch, 2, device=device, requires_grad=True)
        Q_flat = F(pidx_flat, uv_flat)
        Q = Q_flat.reshape(K, M_per_patch, 3)

        if atlas_mode == 'six_sheet' and face_aware_box_supervision:
            cd_loss = torch.cdist(tgt, Q).min(dim=2).values.mean() + torch.cdist(tgt, Q).min(dim=1).values.mean()
        elif no_presplit:
            cd_loss = chamfer_distance_chunked(
                Q_flat, tgt_flat, chunk_size=min(2048, K * M_per_patch)
            )
        else:
            Dmat = torch.cdist(tgt, Q)
            cd_loss = Dmat.min(dim=2).values.mean() + Dmat.min(dim=1).values.mean()

        mu_eff = mu_warmup_schedule(
            epoch, mu_warmup_epochs, mu,
            schedule=schedule, delay_epochs=mu_warmup_delay
        ) if mu > 0 else 0.0

        if mu_eff > 0:
            t_u, t_v = surface_jacobian(Q_flat, uv_flat)
            tangent_loss = tangent_loss_from_jac(t_u, t_v)
        else:
            tangent_loss = zero

        loss = cd_loss + mu_eff * tangent_loss
        loss.backward()

        nn.utils.clip_grad_norm_(list(F.parameters()), 1.0)
        opt.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            history['epoch'].append(epoch)
            history['cd'].append(float(cd_loss))
            history['tangent'].append(float(tangent_loss))
            history['mu_eff'].append(float(mu_eff))
            history['total'].append(float(loss))

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:5d}/{epochs}  |  "
                  f"CD={float(cd_loss):.6f}  "
                  f"Tangent={float(tangent_loss):.6f}  "
                  f"Total={float(loss):.6f}  "
                  f"μ_eff={float(mu_eff):.4f}  "
                  f"[{elapsed:.1f}s]")

    print(f"{'─'*60}\n")
    return F, history





def main():
    parser = argparse.ArgumentParser(
        description='Chamfer Distance Sheet Fitting — Single-Patch or Multi-Patch Feature Complex')

    # Input/output
    parser.add_argument('--file', type=str, default=None,
                        help='Point cloud file (.ply/.xyz/.txt/.npy). Omit for synthetic demo')
    parser.add_argument('--shape', type=str, default='flat_sheet',
                        choices=['saddle', 'hemisphere','box', 'torus_patch', 'wavy', 'sphere', 'flat_sheet', 'stepped_sheet'],
                        help='Synthetic surface type for demo mode')
    parser.add_argument('--result_dir', type=str, default='results_sheet',
                        help='Output directory (auto-increments if exists)')
    parser.add_argument('--N', type=int, default=5000,
                        help='# points to use (downsample if larger). -1 keeps all.')
    parser.add_argument('--mesh_res', type=int, default=100,
                        help='Mesh grid resolution PER PATCH for export')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--multi_patch', action='store_true', default=True,
                        help='Use multi-patch feature complex mode')
    parser.add_argument('--atlas_mode', type=str, default='single_sheet',
                        choices=['single_sheet', 'two_sheet', 'six_sheet'],
                        help='Atlas configuration: one connected sheet or two-sheet shared-boundary atlas')

    # Shared hparam
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--M', type=int, default=4096,
                        help='[Single-patch] # UV surface samples per training step')
    parser.add_argument('--W', type=int, default=256, help='MLP hidden width')
    parser.add_argument('--D', type=int, default=6, help='MLP depth')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=100, help='Softplus beta')
    parser.add_argument('--mu', type=float, default=0.5, help='Tangent loss weight')
    parser.add_argument('--mu_warmup_epochs', type=int, default=0,
                        help='Epochs over which to ramp up μ from 0 to its target '
                             'value (0 = no warmup, use μ immediately)')
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=['linear', 'cosine', 'exponential', 'sigmoid'],
                        help='Warmup ramp shape for μ (default: cosine)')
    parser.add_argument('--gamma', type=float, default=0, help='Normal loss weight')
    parser.add_argument('--lam', type=float, default=1.0, help='Cycle-consistency weight λ₁')
    parser.add_argument('--lam2', type=float, default=1.0, help='Inverse-cycle weight λ₂')

    parser.add_argument('--L', type=int, default=0,
                        help='Positional-encoding frequencies. '
                             'Multi-patch: PE(GLOBAL uv) in the forward decoder '
                             'the main detail knob (raise to ~10 for fine detail).')

    # Multi patch specific args
    parser.add_argument('--n_patches', type=int, default=4,
                        help='[Multi-patch] Number of patches (factored into grid)')
    parser.add_argument('--d_features', type=int, default=88,
                        help='[Multi-patch] Vertex feature dimension')
    parser.add_argument('--M_per_patch', type=int, default=4096,
                        help='[Multi-patch] FIXED # UV samples per patch per step')
    parser.add_argument('--lambda_bcd', type=float, default=0,
                        help='[Multi-patch] Boundary Chamfer Distance weight (optional)')
    
    # Global Boundary Constraint
    parser.add_argument('--lambda_outer_boundary', type=float, default=1.0,
                        help='[Multi-patch] Weight for matching the global outer border to a fixed rectangle')
    parser.add_argument('--outer_boundary_samples', type=int, default=512,
                        help='[Multi-patch] Samples per constrained outer patch edge')
    parser.add_argument('--outer_boundary_loss_type', type=str, default='l1', choices=['l1', 'mse'],
                        help='[Multi-patch] Pointwise loss used for the rectangle outer-boundary constraint')
    
    parser.add_argument('--save_patch_vis', action='store_true', default=True,
                        help='Save patch assignment visualizations')
    parser.add_argument('--L_inv', type=int, default=0,
                        help='[Multi-patch] PE frequencies for inverse encoder')
    parser.add_argument('--reg_every', type=int, default=1,
                        help='[Multi-patch] Compute tangent+normal losses every N '
                             'epochs (2-5 speeds training with little quality loss)')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                        help='[Multi-patch] Save an intermediate checkpoint every N epochs')
    parser.add_argument('--save_correspondence_every', type=int, default=5000,
                        help='[Multi-patch] Save Chamfer correspondence CSV/PNG every N epochs (0 disables)')
    parser.add_argument('--save_boundary_debug_every', type=int, default=5000,
                        help='[Multi-patch] Save outer-boundary model/rectangle correspondence debug every N epochs (0 disables)')
    parser.add_argument('--correspondence_max_lines', type=int, default=300,
                        help='[Multi-patch] Max correspondence lines drawn per saved PNG')
    parser.add_argument('--correspondence_line_segment', type=str, default='t_to_q', choices=['t_to_q', 'both', 'q_to_t'],
                        help='Direction of correspondence lines')
    parser.add_argument('--pretrain_init', action='store_true', default=False,
                        help='Run multi-patch flat-sheet initialization pretraining only')
    parser.add_argument('--pretrain_then_train', action='store_true', default=False,
                        help='Run flat-sheet pretraining first, then continue with multi-patch training in one command')
    parser.add_argument('--pretrain_epochs', type=int, default=2000,
                        help='Epochs for flat-sheet initialization pretraining')
    parser.add_argument('--pretrain_loss', type=str, default='l1', choices=['mse','cd','l1'],
                        help='Pointwise loss for flat-sheet initialization pretraining')
    parser.add_argument('--pretrain_mode', type=str, default='auto',
                        choices=['auto', 'flat', 'closed_shape'],
                        help='Initialization pretraining mode: flat sheet, closed shape, or auto-select based on atlas')
    parser.add_argument('--pretrain_shape', type=str, default='sphere',
                        choices=['saddle', 'hemisphere', 'box', 'torus_patch', 'wavy', 'sphere', 'flat_sheet', 'stepped_sheet'],
                        help='Synthetic shape used for closed-shape pretraining')
    parser.add_argument('--pretrain_shape_samples', type=int, default=200000,
                        help='Number of synthetic samples used for closed-shape pretraining')
    parser.add_argument('--mu_warmup_delay', type=int, default=0,
                        help='Epochs to delay μ warmup (μ=0) before ramping up')

    # Two sheet specific args
    two_sheet_group = parser.add_argument_group('two-sheet configuration')
    two_sheet_group.add_argument('--two_sheet_side_rows', type=int, default=2,
                                 help='Rows of patches per side for atlas_mode=two_sheet')
    two_sheet_group.add_argument('--two_sheet_side_cols', type=int, default=2,
                                 help='Cols of patches per side for atlas_mode=two_sheet')
    two_sheet_group.add_argument('--two_sheet_split_axis', type=int, default=2, choices=[0, 1, 2],
                                 help='Axis used to split the point cloud into two sides')
    two_sheet_group.add_argument('--two_sheet_side_axes', type=int, nargs=2, default=(0, 1),
                                 help='Two axes used for axis-aligned segmentation inside each side')
    two_sheet_group.add_argument('--no_presplit', action='store_true', default=False,
                                 help='Train two-sheet atlas against the full target cloud without pre-splitting points by side')
    two_sheet_group.add_argument('--no_presplit_pretrain', action='store_true', default=False,
                                 help='Disable pre-splitting of points during pretraining for two-sheet mode')
    six_sheet_group = parser.add_argument_group('six-sheet configuration')
    six_sheet_group.add_argument('--six_sheet_face_rows', type=int, default=2,
                                 help='Rows of patches per face for atlas_mode=six_sheet')
    six_sheet_group.add_argument('--six_sheet_face_cols', type=int, default=2,
                                 help='Cols of patches per face for atlas_mode=six_sheet')
    six_sheet_group.add_argument('--face_aware_box_supervision', action='store_true', default=False,
                                 help='Use direct face-aware supervision during six-sheet box pretraining')
    args = parser.parse_args()

    # Load the data
    input_file_name = None
    downsample_n = None if args.N is not None and args.N < 0 else args.N

    if args.file:
        print(f"\n  Loading point cloud from: {args.file}")
        input_file_name = args.file
        pts3n, meta = utils.load_point_cloud(args.file, downsample_n=downsample_n)
    else:
        print(f"\n  No file given → generating synthetic '{args.shape}' surface (N={args.N})")
        input_file_name = f'synthetic_{args.shape}'
        pts3n, meta = utils.make_synthetic_surface(args.shape, n=args.N, noise=0)

    # Validate normals
    normals = meta.get('normals', None)

    if args.gamma > 0:
        if normals is None:
            print(f"\n  ╔══════════════════════════════════════════════════════════╗")
            print(f"  ║  ERROR: Normal consistency loss (gamma={args.gamma}) is    ║")
            print(f"  ║  enabled but the input file has NO NORMALS.              ║")
            print(f"  ║                                                          ║")
            print(f"  ║  Options:                                                ║")
            print(f"  ║    1. Pre-compute normals and save to file:              ║")
            print(f"  ║       python estimate_normals.py --input your_file.ply   ║")
            print(f"  ║    2. Disable normal loss:  --gamma 0                    ║")
            print(f"  ╚══════════════════════════════════════════════════════════╝")
            sys.exit(1)
        else:
            assert normals.shape[0] == pts3n.shape[0], \
                f"Normals count ({normals.shape[0]}) != points count ({pts3n.shape[0]})"
            print(f"  ✓ Normals validated: {normals.shape[0]} normals, unit-length")

    print(f"  Final point count: {pts3n.shape[0]}")
    if args.multi_patch:
        mode_str = "MULTI-PATCH FEATURE COMPLEX"
        if args.atlas_mode == 'two_sheet':
            mode_str += " (TWO-SHEET)"
    else:
        mode_str = "SINGLE-PATCH"
    print(f"  Mode: {mode_str}")

    # Output directory setup
    result_dir = utils._get_unique_folder(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    result_png = os.path.join(result_dir, 'result.png')
    result_log = os.path.join(result_dir, 'metadata.json')
    obj_path = os.path.join(result_dir, 'learned_sheet.obj')
    ply_path = os.path.join(result_dir, 'learned_sheet.ply')
    obj_norm_path = os.path.join(result_dir, 'learned_sheet_normalized.obj')
    ply_norm_path = os.path.join(result_dir, 'learned_sheet_normalized.ply')
    init_ply_path = os.path.join(result_dir, 'learned_sheet_initial.ply')
    psr_ply_path = os.path.join(result_dir, 'psr_reconstruction.ply')
    ckpt_path = os.path.join(result_dir, 'checkpoint.pt')
    pretrain_ckpt_path = os.path.join(result_dir, 'pretrain_checkpoint.pt')

    print(f"  Output directory: {result_dir}")

    # Training
    print(f"\n{'='*60}")
    print(f"  Starting training ({mode_str})...")
    print(f"{'='*60}")

    pretrained_F_state = None
    pretrain_history = None

    if args.pretrain_init or args.pretrain_then_train:
        if not args.multi_patch:
            raise ValueError("--pretrain_init and --pretrain_then_train currently support only --multi_patch mode")

        pretrain_mode = args.pretrain_mode
        if pretrain_mode == 'auto':
            pretrain_mode = 'closed_shape' if args.atlas_mode == 'two_sheet' else 'flat'

        if pretrain_mode == 'closed_shape':
            F_model, pretrain_history = pretrain_multi_patch_closed_shape(
                shape=args.pretrain_shape,
                n_patches=args.n_patches,
                d_features=args.d_features,
                epochs=args.pretrain_epochs,
                M_per_patch=args.M_per_patch,
                W=args.W,
                D=args.D,
                L=args.L,
                lr=args.lr,
                beta=args.beta,
                device=args.device,
                log_every=args.log_every,
                mu=args.mu,
                mu_warmup_epochs=args.mu_warmup_epochs,
                mu_warmup_delay=args.mu_warmup_delay,
                schedule=args.schedule,
                beta_shape_samples=args.pretrain_shape_samples,
                atlas_mode=args.atlas_mode,
                two_sheet_side_rows=args.two_sheet_side_rows,
                two_sheet_side_cols=args.two_sheet_side_cols,
                two_sheet_split_axis=args.two_sheet_split_axis,
                two_sheet_side_axes=tuple(args.two_sheet_side_axes),
                no_presplit=args.no_presplit_pretrain,
                six_sheet_face_rows=args.six_sheet_face_rows,
                six_sheet_face_cols=args.six_sheet_face_cols,
                face_aware_box_supervision=args.face_aware_box_supervision,
            )
        else:
            F_model, pretrain_history = pretrain_multi_patch_flat_sheet(
                n_patches=args.n_patches,
                d_features=args.d_features,
                epochs=args.pretrain_epochs,
                M_per_patch=args.M_per_patch,
                W=args.W,
                D=args.D,
                L=args.L,
                lr=args.lr,
                beta=args.beta,
                device=args.device,
                log_every=args.log_every,
                loss_type=args.pretrain_loss,
                atlas_mode=args.atlas_mode,
                two_sheet_side_rows=args.two_sheet_side_rows,
                two_sheet_side_cols=args.two_sheet_side_cols,
                six_sheet_face_rows=args.six_sheet_face_rows,
                six_sheet_face_cols=args.six_sheet_face_cols,
            )
        F_model.eval()
        pretrained_F_state = {
            k: v.detach().cpu().clone()
            for k, v in F_model.state_dict().items()
        }

        verts, faces = utils.sample_multi_patch_grid(
            F_model,
            resolution=args.mesh_res,
            device=args.device,
            active_patch_ids=list(range(F_model.n_patches)),
        )
        utils.export_ply(verts, faces, init_ply_path)

        torch.save({
            'mode': 'multi_patch_pretrain_flat_sheet' if pretrain_mode == 'flat' else 'multi_patch_pretrain_closed_shape',
            'F_state': F_model.state_dict(),
            'args': vars(args),
            'history': pretrain_history,
            'grid_dims': (F_model.n_rows, F_model.n_cols),
            'atlas_mode': args.atlas_mode,
        }, pretrain_ckpt_path)

        print(f"    Pretrain checkpoint → {pretrain_ckpt_path}")

        if args.pretrain_init and not args.pretrain_then_train:
            print(f"\n{'='*60}")
            print("  Initialization pretraining complete!")
            print("  Outputs:")
            print(f"    {init_ply_path}     — initialized flat-sheet mesh")
            print(f"    {pretrain_ckpt_path}         — pretrained forward-map weights")
            print(f"{'='*60}\n")
            return

        print(f"\n{'='*60}")
        print("  Initialization pretraining complete!")
        print("  Continuing directly into multi-patch training...")
        print(f"{'='*60}")

    if args.multi_patch:
        F_model, G_model, history, assignments, active_ids = train_multi_patch(
            pts3n,
            n_patches=args.n_patches,
            d_features=args.d_features,
            epochs=args.epochs,
            M=args.M,
            M_per_patch=args.M_per_patch,
            W=args.W,
            D=args.D,
            L=args.L,
            L_inv=args.L_inv,
            lr=args.lr,
            mu=args.mu,
            mu_warmup_epochs=args.mu_warmup_epochs,
            mu_warmup_delay=args.mu_warmup_delay,
            schedule=args.schedule,
            gamma=args.gamma,
            lam=args.lam,
            lam2=args.lam2,
            lambda_bcd=args.lambda_bcd,
            lambda_outer_boundary=args.lambda_outer_boundary,
            outer_boundary_samples=args.outer_boundary_samples,
            outer_boundary_loss_type=args.outer_boundary_loss_type,
            beta=args.beta,
            device=args.device,
            log_every=args.log_every,
            save_patch_vis=args.save_patch_vis,
            vis_dir=result_dir,
            checkpoint_every=args.checkpoint_every,
            checkpoint_payload={
                'args': vars(args),
                'normalization': {
                    'center': meta['center'].tolist() if hasattr(meta['center'], 'tolist')
                              else list(meta['center']),
                    'scale': float(meta['scale']),
                },
                'input_file': input_file_name,
                'result_dir': result_dir,
            },
            output_psr_mesh_path=psr_ply_path,
            normals=normals,
            reg_every=args.reg_every,
            pretrained_F_state=pretrained_F_state,
            pretrained_ckpt_path=pretrain_ckpt_path if pretrained_F_state is not None else None,
            correspondence_dir=os.path.join(result_dir, 'correspondences'),
            save_correspondence_every=args.save_correspondence_every,
            save_boundary_debug_every=args.save_boundary_debug_every,
            correspondence_max_lines=args.correspondence_max_lines,
            correspondence_line_segment=args.correspondence_line_segment,
            atlas_mode=args.atlas_mode,
            no_presplit=args.no_presplit,
            two_sheet_side_rows=args.two_sheet_side_rows,
            two_sheet_side_cols=args.two_sheet_side_cols,
            two_sheet_split_axis=args.two_sheet_split_axis,
            two_sheet_side_axes=tuple(args.two_sheet_side_axes),
            six_sheet_face_rows=args.six_sheet_face_rows,
            six_sheet_face_cols=args.six_sheet_face_cols,
        )
        F_model.eval()
        if G_model is not None:
            G_model.eval()

        print(f"\n  Saving results to: {result_dir}")
        utils._save_run_metadata(result_log, args, input_file_name, result_dir,
                                 result_png, history, n_points=pts3n.shape[0], meta=meta)

        verts, faces = utils.visualise_multi_patch(
            F_model, pts3n, assignments, history, out_path=result_png,
            resolution=args.mesh_res, device=args.device,
            active_patch_ids=active_ids,
        )

        torch.save({
            'mode': 'multi_patch',
            'F_state': F_model.state_dict(),
            'G_state': G_model.state_dict() if G_model is not None else None,
            'args': vars(args),
            'pretrain_history': pretrain_history,
            'history': history,
            'assignments': assignments.tolist(),
            'active_patch_ids': active_ids,
            'grid_dims': (F_model.n_rows, F_model.n_cols),
            'atlas_mode': args.atlas_mode,
            'normalization': {
                'center': meta['center'].tolist() if hasattr(meta['center'], 'tolist')
                          else list(meta['center']),
                'scale': float(meta['scale']),
            },
        }, ckpt_path)
        print(f"    Checkpoint → {ckpt_path}")

    # Mesh export
    verts_original = utils.unnormalize_vertices(verts, meta)

    print(f"\n  Exporting mesh in ORIGINAL coordinates:")
    print(f"    Transform: p_orig = p_norm * {meta['scale']:.6f} + {meta['center']}")
    utils.export_obj(verts_original, faces, obj_path)
    utils.export_ply(verts_original, faces, ply_path)

    print(f"\n  Exporting mesh in normalized coordinates (for reference):")
    utils.export_obj(verts, faces, obj_norm_path)
    utils.export_ply(verts, faces, ply_norm_path)

    print(f"\n{'='*60}")
    print(f"  Run complete! Mode: {mode_str}")
    print(f"  Outputs:")
    print(f"    {result_png}        — 6-panel visualization")
    if args.multi_patch:
        print(f"    {psr_ply_path}      — Poisson reconstruction used for pre-segmentation")
    print(f"    {obj_path}          — mesh in original coords (MeshLab)")
    print(f"    {ply_path}          — mesh in original coords (CloudCompare)")
    print(f"    {obj_norm_path}     — mesh in normalized [-1,1] coords")
    print(f"    {ckpt_path}         — model weights + normalization transform")
    print(f"    {result_log}        — hyperparams + loss log")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()