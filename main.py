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
import utils.utils as utils
from model.losses import (boundary_chamfer_loss, chamfer_1d,
                                       chamfer_distance,
                                       chamfer_distance_chunked,
                                       mu_warmup_schedule,
                                       normal_consistency_loss,
                                             surface_jacobian,
                                             tangent_fold_loss,
                                             tangent_loss_from_jac)
from model.model import (FeatureComplex, ForwardMap, InverseMap, MultiPatchForwardMap,
                   MultiPatchInverseMap, PositionalEncoding, SkipMLP)
try:
    import open3d as o3d
except ImportError:
    o3d = None

# Multi-patch training.
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
                      pretrained_ckpt_path: str = None):
    """
    Train the multi-patch model and inverse map.

    Expensive regularizers can be evaluated every `reg_every` steps instead of
    every iteration.
    """
    # Grid dimensions.
    n_rows, n_cols = pc_presegmentation.compute_grid_dims(n_patches)
    actual_n_patches = n_rows * n_cols
    print(f"  Grid layout: {n_rows} rows × {n_cols} cols = {actual_n_patches} patches")

    if gamma > 0:
        assert normals is not None, "normals must be provided when gamma > 0"
        assert normals.shape == pts3n.shape, \
            f"Normals shape {normals.shape} != points shape {pts3n.shape}"
        print(f"  Normal constraint active (γ={gamma}), normals shape: {normals.shape}")

    # Default pre-segmentation.
    assignments, grid_topology, patch_params = pc_presegmentation.pca_grid_segmentation(
        pts3n, n_rows, n_cols
    )
    
    # Alternative: Poisson spectral segmentation.
    # assignments, grid_topology, patch_params = pc_presegmentation.poisson_spectral_segmentation(
    #     pts3n, normals, n_patches_u=n_rows, n_patches_v=n_cols,
    #     export_mesh_path=output_psr_mesh_path
    # )

    # Alternative: spectral segmentation.
    # assignments, grid_topology, patch_params = pc_presegmentation.spectral_direct_segmentation(
    #     pts3n, n_patches_u=n_rows, n_patches_v=n_cols
    # )

    # Alternative: axis-aligned grid segmentation.
    # assignments, grid_topology, patch_params = pc_presegmentation.axis_aligned_grid_segmentation(
    #     pts3n, n_rows, n_cols
    # )

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
            utils._visualize_patch_assignments(
                pts_vis, asg_vis, grid_topology, pp_vis,
                n_rows, n_cols,
                save_path=os.path.join(vis_dir, 'patch_assignments.png')
            )
            utils._visualize_patch_assignments_3d(
                pts_vis, asg_vis, grid_topology, n_rows, n_cols,
                save_path=os.path.join(vis_dir, 'patch_assignments_3d.png')
            )
        except Exception as e:
            print(f"  [warn] patch visualization skipped ({type(e).__name__}: {e})")

    # Keep only active patches with enough assigned points.
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
    F = MultiPatchForwardMap(n_rows, n_cols, d_features,
                             L=L, W=W, D=D, beta=beta).to(device)
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
    n_params_G = sum(p.numel() for p in G.parameters())
    n_vertex = F.complex.vertex_features.numel()
    print(f"  F total params: {n_params_F:,}")
    print(f"    Vertex features (shared): {n_vertex:,} "
          f"({(n_rows+1)*(n_cols+1)} vertices × {d_features}d)")
    print(f"    Shared decoder (+ global-UV PE, L={L}): {n_params_F - n_vertex:,}")
    print(f"  G encoder params (L_inv={L_inv}): {n_params_G:,}")
    print(f"  Total unique params: {n_params_F + n_params_G:,}")
    print("  F parameter breakdown:")
    for name, param in F.named_parameters():
        print(f"    {name:<40} shape={tuple(param.shape)} "
              f"requires_grad={param.requires_grad}")

    opt = torch.optim.Adam(list(F.parameters()) + list(G.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = {'cd': [], 'cycle': [], 'param': [], 'tangent': [], 'normal': [],
               'mu_eff': [], 'total': [], 'epoch': []}
    

    print(f"\n{'─'*60}")
    print(f"  Multi-Patch Feature Complex Training (VECTORIZED)")
    print(f"  Grid={n_rows}×{n_cols}  active_patches={K}/{actual_n_patches}")
    print(f"  d_features={d_features}  W={W}  D={D}  L(fwd/global-UV)={L}  L_inv={L_inv}  β={beta}")
    print(f"  M_per_patch={M_per_patch}  batch/step={K*M_per_patch}  reg_every={reg_every}")
    print(f"  μ={mu}  γ={gamma}  λ₁={lam}  λ₂={lam2}")
    if mu_warmup_epochs > 0 and mu > 0:
        print(f"  μ warmup: {mu_warmup_schedule} ramp over {mu_warmup_epochs} epochs "
              f"(0.0 → {mu})")
    print(f"  Epochs={epochs}  lr={lr}  device={device}")
    print(f"{'─'*60}")
    t0 = time.time()

    checkpoint_dir = vis_dir if vis_dir is not None else os.getcwd()
    os.makedirs(checkpoint_dir, exist_ok=True)

    def _save_epoch_checkpoint(epoch: int):
        if checkpoint_every <= 0 or epoch % checkpoint_every != 0:
            return

        epoch_tag = f'{epoch}'
        epoch_ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch_tag}.pt')
        payload = {
            'mode': 'multi_patch',
            'epoch': epoch,
            'F_state': F.state_dict(),
            'G_state': G.state_dict(),
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
                'beta': beta,
                'device': device,
                'log_every': log_every,
                'save_patch_vis': save_patch_vis,
                'reg_every': reg_every,
            }),
            'grid_dims': (n_rows, n_cols),
            'history': history,
        }
        if checkpoint_payload is not None:
            for key in ('normalization', 'input_file', 'result_dir'):
                if key in checkpoint_payload:
                    payload[key] = checkpoint_payload[key]

        torch.save(payload, epoch_ckpt_path)
        print(f"    Checkpoint → {epoch_ckpt_path}")

    zero = torch.tensor(0.0, device=device)
    vertex_features_init = F.complex.vertex_features.detach().clone()


    # Optimize the forward and inverse maps.
    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        # Build the target batch on CPU, then transfer once to the device.
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
        D = torch.cdist(tgt, Q)
        cd_loss = D.min(dim=2).values.mean() + D.min(dim=1).values.mean()

        # Loss 2: cycle consistency.
        if lam > 0:
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

        # Total loss.
        loss = (cd_loss
                + lam * cycle_loss
                + lam2 * param_loss
                + mu_eff * tangent_loss
                + gamma * normal_loss)
        loss.backward()

        nn.utils.clip_grad_norm_(list(F.parameters()) + list(G.parameters()), 1.0)
        opt.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            history['epoch'].append(epoch)
            history['cd'].append(float(cd_loss))
            history['cycle'].append(float(cycle_loss))
            history['param'].append(float(param_loss))
            history['tangent'].append(float(tangent_loss))
            history['normal'].append(float(normal_loss))
            history['mu_eff'].append(float(mu_eff))
            history['total'].append(float(loss))

            elapsed = time.time() - t0
            mu_str = f"  μ_eff={float(mu_eff):.4f}" if (mu_warmup_epochs > 0 and mu > 0) else ""
            print(f"  Epoch {epoch:5d}/{epochs}  |  "
                  f"CD={float(cd_loss):.5f}  "
                  f"Cycle={float(cycle_loss):.5f}  "
                  f"Param={float(param_loss):.5f}  "
                  f"Tangent={float(tangent_loss):.5f}  "
                  f"Normal={float(normal_loss):.5f}  "
                  f"Total={float(loss):.5f}"
                  f"{mu_str}  "
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
                                    loss_type: str = 'mse'):
    """
    Pretrain only the multi-patch forward map F. 

    The target is a real sampled 3D point cloud lying on z=0 in normalized
    coordinates. Each patch learns its corresponding region of the plane using
    direct pointwise supervision.

    Mapping F to plane (sample (u, v) from plane and train with MSE/L1 F(z(u, v)) = (x, y, z) = (u, v, 0))
    """
    n_rows, n_cols = pc_presegmentation.compute_grid_dims(n_patches)
    actual_n_patches = n_rows * n_cols

    F = MultiPatchForwardMap(n_rows, n_cols, d_features,
                             L=L, W=W, D=D, beta=beta).to(device)

    opt = torch.optim.Adam(F.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = {'plane': [], 'total': [], 'epoch': []}

    print(f"\n{'─'*60}")
    print("  Multi-Patch Flat-Sheet Pretraining")
    print(f"  Grid={n_rows}×{n_cols}  patches={actual_n_patches}")
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



def main():
    parser = argparse.ArgumentParser(
        description='Chamfer Distance Sheet Fitting — Single-Patch or Multi-Patch Feature Complex')

    # Input/output
    parser.add_argument('--file', type=str, default=None,
                        help='Point cloud file (.ply/.xyz/.txt/.npy). Omit for synthetic demo')
    parser.add_argument('--shape', type=str, default='flat_sheet',
                        choices=['saddle', 'hemisphere', 'torus_patch', 'wavy', 'sphere', 'flat_sheet'],
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
    parser.add_argument('--save_patch_vis', action='store_true', default=True,
                        help='Save patch assignment visualizations')
    parser.add_argument('--L_inv', type=int, default=0,
                        help='[Multi-patch] PE frequencies for inverse encoder')
    parser.add_argument('--reg_every', type=int, default=1,
                        help='[Multi-patch] Compute tangent+normal losses every N '
                             'epochs (2-5 speeds training with little quality loss)')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                        help='[Multi-patch] Save an intermediate checkpoint every N epochs')
    parser.add_argument('--pretrain_init', action='store_true', default=False,
                        help='Run multi-patch flat-sheet initialization pretraining only')
    parser.add_argument('--pretrain_then_train', action='store_true', default=False,
                        help='Run flat-sheet pretraining first, then continue with multi-patch training in one command')
    parser.add_argument('--pretrain_epochs', type=int, default=2000,
                        help='Epochs for flat-sheet initialization pretraining')
    parser.add_argument('--pretrain_loss', type=str, default='l1', choices=['mse','cd','l1'],
                        help='Pointwise loss for flat-sheet initialization pretraining')
    parser.add_argument('--mu_warmup_delay', type=int, default=0,
                        help='Epochs to delay μ warmup (μ=0) before ramping up')

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
    mode_str = "MULTI-PATCH FEATURE COMPLEX" if args.multi_patch else "SINGLE-PATCH"
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
            'mode': 'multi_patch_pretrain_flat_sheet',
            'F_state': F_model.state_dict(),
            'args': vars(args),
            'history': pretrain_history,
            'grid_dims': (F_model.n_rows, F_model.n_cols),
        }, ckpt_path)

        print(f"    Pretrain checkpoint → {ckpt_path}")

        if args.pretrain_init and not args.pretrain_then_train:
            print(f"\n{'='*60}")
            print("  Initialization pretraining complete!")
            print("  Outputs:")
            print(f"    {init_ply_path}     — initialized flat-sheet mesh")
            print(f"    {ckpt_path}         — pretrained forward-map weights")
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
            pretrained_ckpt_path=ckpt_path if pretrained_F_state is not None else None,
        )
        F_model.eval()
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
            'G_state': G_model.state_dict(),
            'args': vars(args),
            'pretrain_history': pretrain_history,
            'history': history,
            'assignments': assignments.tolist(),
            'active_patch_ids': active_ids,
            'grid_dims': (F_model.n_rows, F_model.n_cols),
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