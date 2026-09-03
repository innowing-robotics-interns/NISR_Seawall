#!/usr/bin/env python3
# utils/adaptive_vis.py
"""Patch-configuration visualization for the adaptive cube atlas."""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from model.adaptive_complex import FACE_NAMES, S_INT


@torch.no_grad()
def _decode_leaf_edges(model, edge_samples, device, batch=8192):
    """(n_leaves, 4, S, 3) decoded boundary curves of every leaf."""
    n = model.n_patches
    t = torch.linspace(0.0, 1.0, edge_samples, device=device).unsqueeze(1)
    z, o = torch.zeros_like(t), torch.ones_like(t)
    edges = torch.cat([torch.cat([t, z], 1), torch.cat([o, t], 1),
                       torch.cat([t, o], 1), torch.cat([z, t], 1)], dim=0)
    uv = edges.repeat(n, 1)
    pids = torch.arange(n, device=device).repeat_interleave(4 * edge_samples)
    out = []
    for i in range(0, uv.shape[0], batch):
        out.append(model(pids[i:i + batch], uv[i:i + batch]).cpu())
    return torch.cat(out, 0).reshape(n, 4, edge_samples, 3).numpy()


@torch.no_grad()
def _decode_leaf_grids(model, resolution, device, batch=8192):
    """List of (res, res, 3) surface grids, one per leaf."""
    n = model.n_patches
    u = torch.linspace(0, 1, resolution, device=device)
    gu, gv = torch.meshgrid(u, u, indexing='ij')
    uv1 = torch.stack([gu.flatten(), gv.flatten()], dim=-1)
    grids = []
    for li in range(n):
        out = []
        for i in range(0, uv1.shape[0], batch):
            out.append(model(li, uv1[i:i + batch]).cpu())
        grids.append(torch.cat(out, 0).reshape(resolution, resolution, 3).numpy())
    return grids


def visualize_patch_configuration(model, out_path, pts=None, resolution=10,
                                  edge_samples=25, max_pts=20000):
    """
    Snapshot of the current patch configuration:
      top: two 3D views (leaf surfaces colored, black patch boundaries,
           optional target points) + depth histogram / stats
      bottom: the 6 face quadtree layouts (rect = leaf, color = depth).
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    grids = _decode_leaf_grids(model, resolution, device)
    curves = _decode_leaf_edges(model, edge_samples, device)
    model.train(was_training)

    cx = model.complex
    leaves = cx.leaf_patches
    depths = np.array([p.depth for p in leaves])
    max_depth = int(depths.max()) if len(depths) else 0
    stats = cx.stats()

    fig = plt.figure(figsize=(22, 11), facecolor='white')
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.35,
                           left=0.03, right=0.98, top=0.90, bottom=0.05)
    cmap = plt.get_cmap('tab20')

    for col, (elev, azim) in enumerate([(25, 45), (15, 135)]):
        ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2], projection='3d')
        for li, g in enumerate(grids):
            ax.plot_surface(g[..., 0], g[..., 1], g[..., 2],
                            color=cmap(li % 20), alpha=0.75,
                            edgecolor='none', antialiased=True)
        for li in range(curves.shape[0]):
            for e in range(4):
                c = curves[li, e]
                ax.plot(c[:, 0], c[:, 1], c[:, 2], color='k', lw=0.7)
        if pts is not None:
            P = pts if pts.shape[0] <= max_pts else pts[
                np.random.choice(pts.shape[0], max_pts, replace=False)]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1, c='red', alpha=0.15,
                       linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"3D patch layout (view {col + 1})", fontsize=10)

    ax_h = fig.add_subplot(gs[0, 4:6])
    ds, cs = np.unique(depths, return_counts=True) if len(depths) else ([], [])
    ax_h.bar(ds, cs, color='steelblue')
    ax_h.set_xlabel('leaf depth')
    ax_h.set_ylabel('# leaves')
    ax_h.set_title(f"leaves={stats['n_leaves']}  vertices={stats['n_vertices']}  "
                   f"K_max={stats['k_max']}", fontsize=10)
    ax_h.grid(True, alpha=0.3)

    dnorm = plt.Normalize(0, max(max_depth, 1))
    dcmap = plt.get_cmap('viridis')
    for f in range(6):
        ax = fig.add_subplot(gs[1, f])
        for p in leaves:
            if p.face != f:
                continue
            ax.add_patch(Rectangle((p.u0 / S_INT, p.v0 / S_INT),
                                   p.size / S_INT, p.size / S_INT,
                                   facecolor=dcmap(dnorm(p.depth)),
                                   edgecolor='black', linewidth=0.6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f"face {FACE_NAMES[f]}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Adaptive cube atlas — {stats['n_leaves']} leaf patches, "
                 f"depth histogram {stats['depth_hist']}",
                 fontsize=13, fontweight='bold')
    fig.savefig(out_path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"    Patch-configuration PNG → {out_path}")


def plot_history(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ep = history['epoch']
    for key, color in (('cd', '#58a6ff'), ('tangent', '#d2a8ff'),
                       ('svd', '#e3b341'),
                       ('total', '#f78166'), ('loss', '#3fb950')):
        if key in history and len(history[key]) == len(ep):
            ax1.plot(ep, history[key], lw=1.6, color=color, label=key)
    ax1.set_yscale('log')
    ax1.set_xlabel('epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('losses')
    if 'n_leaves' in history and len(history['n_leaves']) == len(ep):
        ax2.step(ep, history['n_leaves'], where='post', color='steelblue')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('# leaf patches')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('subdivision progress')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    History PNG → {out_path}")