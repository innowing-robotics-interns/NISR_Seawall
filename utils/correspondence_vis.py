#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_POINT_COLOR = (214, 39, 40)
TARGET_POINT_COLOR = (31, 119, 180)
Q_TO_T_LINE_COLOR = (127, 127, 127)
T_TO_Q_LINE_COLOR = (44, 160, 44)


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    """Set equal scaling and consistent limits for a 3D axis."""
    if points.size == 0:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = maxs - mins
    radius = 0.5 * float(np.max(span))

    if radius < 1e-8:
        radius = 1e-3

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


CSV_HEADER = [
    "pair_id",
    "patch_id",
    "direction",
    "q_x",
    "q_y",
    "q_z",
    "t_x",
    "t_y",
    "t_z",
    "distance",
]


def export_chamfer_correspondences(
    q_points: np.ndarray,
    t_points: np.ndarray,
    distance_matrix: np.ndarray,
    output_csv_path: str,
    output_png_path: Optional[str] = None,
    patch_ids: Optional[np.ndarray] = None,
    max_lines: int = 300,
    point_size: float = 4.0,
    line_width: float = 0.9,
    plot_direction: str = "t_to_q",
) -> None:
    """
    Export nearest-neighbor correspondences used by the Chamfer loss.

    Two directed match sets are stored:
    - q_to_t: each model output point q matched to its nearest target point t
    - t_to_q: each target point t matched to its nearest model output point q

    The CSV stores both endpoints explicitly so downstream tools can distinguish
    model output points from ground-truth points.
    """
    q_points = np.asarray(q_points, dtype=np.float32)
    t_points = np.asarray(t_points, dtype=np.float32)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float32)

    if q_points.ndim != 2 or q_points.shape[1] != 3:
        raise ValueError(f"q_points must have shape (N, 3), got {q_points.shape}")
    if t_points.ndim != 2 or t_points.shape[1] != 3:
        raise ValueError(f"t_points must have shape (M, 3), got {t_points.shape}")
    if distance_matrix.shape != (t_points.shape[0], q_points.shape[0]):
        raise ValueError(
            "distance_matrix must have shape (n_target, n_query); "
            f"got {distance_matrix.shape}, expected {(t_points.shape[0], q_points.shape[0])}"
        )

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    if output_png_path is not None:
        os.makedirs(os.path.dirname(output_png_path) or ".", exist_ok=True)

    q_to_t_idx = distance_matrix.argmin(axis=0)
    t_to_q_idx = distance_matrix.argmin(axis=1)

    if patch_ids is None:
        q_patch_ids = np.full(q_points.shape[0], -1, dtype=np.int32)
        t_patch_ids = np.full(t_points.shape[0], -1, dtype=np.int32)
    else:
        patch_ids = np.asarray(patch_ids)
        if patch_ids.ndim == 1:
            if patch_ids.shape[0] != q_points.shape[0]:
                raise ValueError(
                    "1D patch_ids must align with q_points; "
                    f"got {patch_ids.shape[0]} vs {q_points.shape[0]}"
                )
            q_patch_ids = patch_ids.astype(np.int32)
            t_patch_ids = np.full(t_points.shape[0], -1, dtype=np.int32)
        elif patch_ids.ndim == 2 and patch_ids.shape[0] == 2:
            q_patch_ids = patch_ids[0].astype(np.int32)
            t_patch_ids = patch_ids[1].astype(np.int32)
        else:
            raise ValueError(
                "patch_ids must be None, shape (n_query,), or shape (2, n_points)"
            )

    rows = []
    pair_id = 0

    for q_idx, t_idx in enumerate(q_to_t_idx):
        q = q_points[q_idx]
        t = t_points[t_idx]
        rows.append([
            pair_id,
            int(q_patch_ids[q_idx]),
            "q_to_t",
            float(q[0]), float(q[1]), float(q[2]),
            float(t[0]), float(t[1]), float(t[2]),
            float(distance_matrix[t_idx, q_idx]),
        ])
        pair_id += 1

    for t_idx, q_idx in enumerate(t_to_q_idx):
        q = q_points[q_idx]
        t = t_points[t_idx]
        rows.append([
            pair_id,
            int(t_patch_ids[t_idx]),
            "t_to_q",
            float(q[0]), float(q[1]), float(q[2]),
            float(t[0]), float(t[1]), float(t[2]),
            float(distance_matrix[t_idx, q_idx]),
        ])
        pair_id += 1

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    print(f"    Correspondence CSV → {output_csv_path}")

    if output_png_path is not None:
        _plot_correspondences(
            q_points=q_points,
            t_points=t_points,
            q_to_t_idx=q_to_t_idx,
            t_to_q_idx=t_to_q_idx,
            output_png_path=output_png_path,
            max_lines=max_lines,
            point_size=point_size,
            line_width=line_width,
            plot_direction=plot_direction,
        )

    output_ply_path = os.path.splitext(output_csv_path)[0] + ".ply"
    export_correspondence_ply(
        q_points=q_points,
        t_points=t_points,
        q_to_t_idx=q_to_t_idx,
        t_to_q_idx=t_to_q_idx,
        output_ply_path=output_ply_path,
        plot_direction=plot_direction,
    )


def export_correspondence_ply(
    q_points: np.ndarray,
    t_points: np.ndarray,
    q_to_t_idx: np.ndarray,
    t_to_q_idx: np.ndarray,
    output_ply_path: str,
    plot_direction: str,
) -> None:
    """Export correspondences as an ASCII PLY with colored vertices and edges."""
    if plot_direction not in {"q_to_t", "t_to_q", "both"}:
        raise ValueError(
            f"plot_direction must be one of 'q_to_t', 't_to_q', or 'both', got {plot_direction}"
        )

    os.makedirs(os.path.dirname(output_ply_path) or ".", exist_ok=True)

    q_points = np.asarray(q_points, dtype=np.float32)
    t_points = np.asarray(t_points, dtype=np.float32)
    q_to_t_idx = np.asarray(q_to_t_idx, dtype=np.int32)
    t_to_q_idx = np.asarray(t_to_q_idx, dtype=np.int32)

    vertices = []
    for point in q_points:
        vertices.append((*point.tolist(), *MODEL_POINT_COLOR))
    q_offset = 0
    t_offset = len(vertices)
    for point in t_points:
        vertices.append((*point.tolist(), *TARGET_POINT_COLOR))

    edges = []
    if plot_direction in {"q_to_t", "both"}:
        for q_idx, t_idx in enumerate(q_to_t_idx):
            edges.append((q_offset + q_idx, t_offset + int(t_idx), *Q_TO_T_LINE_COLOR))
    if plot_direction in {"t_to_q", "both"}:
        for t_idx, q_idx in enumerate(t_to_q_idx):
            edges.append((t_offset + t_idx, q_offset + int(q_idx), *T_TO_Q_LINE_COLOR))

    with open(output_ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for x, y, z, r, g, b in vertices:
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n")

        for v1, v2, r, g, b in edges:
            f.write(f"{v1} {v2} {r} {g} {b}\n")

    print(f"    Correspondence PLY → {output_ply_path}")


def export_combined_correspondence_ply(
    q_batches: np.ndarray,
    t_batches: np.ndarray,
    distance_batches: np.ndarray,
    output_ply_path: str,
    plot_direction: str = "t_to_q",
) -> None:
    """Export all patch correspondences into one combined ASCII PLY file."""
    if plot_direction not in {"q_to_t", "t_to_q", "both"}:
        raise ValueError(
            f"plot_direction must be one of 'q_to_t', 't_to_q', or 'both', got {plot_direction}"
        )

    q_batches = np.asarray(q_batches, dtype=np.float32)
    t_batches = np.asarray(t_batches, dtype=np.float32)
    distance_batches = np.asarray(distance_batches, dtype=np.float32)

    if q_batches.ndim != 3 or q_batches.shape[-1] != 3:
        raise ValueError(f"q_batches must have shape (K, N, 3), got {q_batches.shape}")
    if t_batches.ndim != 3 or t_batches.shape[-1] != 3:
        raise ValueError(f"t_batches must have shape (K, M, 3), got {t_batches.shape}")
    if distance_batches.ndim != 3:
        raise ValueError(
            f"distance_batches must have shape (K, M, N), got {distance_batches.shape}"
        )
    if q_batches.shape[0] != t_batches.shape[0] or q_batches.shape[0] != distance_batches.shape[0]:
        raise ValueError("Batch dimension K must match across q_batches, t_batches, and distance_batches")

    os.makedirs(os.path.dirname(output_ply_path) or ".", exist_ok=True)

    vertices = []
    edges = []
    vertex_offset = 0

    for batch_idx in range(q_batches.shape[0]):
        q_points = q_batches[batch_idx]
        t_points = t_batches[batch_idx]
        distance_matrix = distance_batches[batch_idx]
        q_to_t_idx = distance_matrix.argmin(axis=0)
        t_to_q_idx = distance_matrix.argmin(axis=1)

        q_offset = vertex_offset
        for point in q_points:
            vertices.append((*point.tolist(), *MODEL_POINT_COLOR))
        vertex_offset += q_points.shape[0]

        t_offset = vertex_offset
        for point in t_points:
            vertices.append((*point.tolist(), *TARGET_POINT_COLOR))
        vertex_offset += t_points.shape[0]

        if plot_direction in {"q_to_t", "both"}:
            for q_idx, t_idx in enumerate(q_to_t_idx):
                edges.append((q_offset + q_idx, t_offset + int(t_idx), *Q_TO_T_LINE_COLOR))
        if plot_direction in {"t_to_q", "both"}:
            for t_idx, q_idx in enumerate(t_to_q_idx):
                edges.append((t_offset + t_idx, q_offset + int(q_idx), *T_TO_Q_LINE_COLOR))

    with open(output_ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for x, y, z, r, g, b in vertices:
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n")

        for v1, v2, r, g, b in edges:
            f.write(f"{v1} {v2} {r} {g} {b}\n")

    print(f"    Combined correspondence PLY → {output_ply_path}")


def export_global_correspondence_ply(
    q_points: np.ndarray,
    t_points: np.ndarray,
    output_ply_path: str,
    plot_direction: str = "both",
) -> None:
    """Export one global correspondence PLY for no-presplit training mode."""
    q_points = np.asarray(q_points, dtype=np.float32)
    t_points = np.asarray(t_points, dtype=np.float32)

    if q_points.ndim != 2 or q_points.shape[1] != 3:
        raise ValueError(f"q_points must have shape (N, 3), got {q_points.shape}")
    if t_points.ndim != 2 or t_points.shape[1] != 3:
        raise ValueError(f"t_points must have shape (M, 3), got {t_points.shape}")

    if q_points.shape[0] == 0 or t_points.shape[0] == 0:
        raise ValueError("q_points and t_points must both be non-empty")

    distance_matrix = np.linalg.norm(
        t_points[:, None, :] - q_points[None, :, :], axis=-1
    ).astype(np.float32)
    q_to_t_idx = distance_matrix.argmin(axis=0)
    t_to_q_idx = distance_matrix.argmin(axis=1)

    export_correspondence_ply(
        q_points=q_points,
        t_points=t_points,
        q_to_t_idx=q_to_t_idx,
        t_to_q_idx=t_to_q_idx,
        output_ply_path=output_ply_path,
        plot_direction=plot_direction,
    )


def _plot_correspondences(
    q_points: np.ndarray,
    t_points: np.ndarray,
    q_to_t_idx: np.ndarray,
    t_to_q_idx: np.ndarray,
    output_png_path: str,
    max_lines: int,
    point_size: float,
    line_width: float,
    plot_direction: str,
) -> None:
    """Create a 3D plot with model points, target points, and match lines."""
    if plot_direction not in {"q_to_t", "t_to_q", "both"}:
        raise ValueError(
            f"plot_direction must be one of 'q_to_t', 't_to_q', or 'both', got {plot_direction}"
        )

    n_q_pairs = q_points.shape[0]
    n_t_pairs = t_points.shape[0]
    if n_q_pairs == 0 and n_t_pairs == 0:
        return

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        q_points[:, 0], q_points[:, 1], q_points[:, 2],
        s=point_size, c="#d62728", alpha=0.75, label="Model output q"
    )
    ax.scatter(
        t_points[:, 0], t_points[:, 1], t_points[:, 2],
        s=point_size, c="#1f77b4", alpha=0.75, label="Ground truth t"
    )

    if plot_direction in {"q_to_t", "both"} and n_q_pairs > 0:
        if n_q_pairs > max_lines:
            q_sample_idx = np.linspace(0, n_q_pairs - 1, max_lines, dtype=int)
        else:
            q_sample_idx = np.arange(n_q_pairs)

        for idx in q_sample_idx:
            q = q_points[idx]
            t = t_points[q_to_t_idx[idx]]
            ax.plot(
                [q[0], t[0]], [q[1], t[1]], [q[2], t[2]],
                color="#7f7f7f", alpha=0.35, linewidth=line_width,
                label="q → t" if idx == q_sample_idx[0] else None
            )

    if plot_direction in {"t_to_q", "both"} and n_t_pairs > 0:
        if n_t_pairs > max_lines:
            t_sample_idx = np.linspace(0, n_t_pairs - 1, max_lines, dtype=int)
        else:
            t_sample_idx = np.arange(n_t_pairs)

        for idx in t_sample_idx:
            t = t_points[idx]
            q = q_points[t_to_q_idx[idx]]
            ax.plot(
                [t[0], q[0]], [t[1], q[1]], [t[2], q[2]],
                color="#2ca02c", alpha=0.35, linewidth=line_width,
                label="t → q" if idx == t_sample_idx[0] else None
            )

    ax.set_title(f"Chamfer Correspondences ({plot_direction})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    all_points = np.vstack([q_points, t_points])
    _set_equal_3d_axes(ax, all_points)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    Correspondence PNG → {output_png_path}")


def export_patchwise_chamfer_correspondences(
    q_batches: np.ndarray,
    t_batches: np.ndarray,
    distance_batches: np.ndarray,
    output_dir: str,
    patch_ids: np.ndarray,
    max_lines: int = 500,
    plot_direction: str = "t_to_q",
    point_size: float = 4.0,
    line_width: float = 0.9,
) -> None:
    """Export one CSV/PNG pair per patch for Chamfer correspondences."""
    q_batches = np.asarray(q_batches, dtype=np.float32)
    t_batches = np.asarray(t_batches, dtype=np.float32)
    distance_batches = np.asarray(distance_batches, dtype=np.float32)
    patch_ids = np.asarray(patch_ids, dtype=np.int32)

    if q_batches.ndim != 3 or q_batches.shape[-1] != 3:
        raise ValueError(f"q_batches must have shape (K, N, 3), got {q_batches.shape}")
    if t_batches.ndim != 3 or t_batches.shape[-1] != 3:
        raise ValueError(f"t_batches must have shape (K, M, 3), got {t_batches.shape}")
    if distance_batches.ndim != 3:
        raise ValueError(
            f"distance_batches must have shape (K, M, N), got {distance_batches.shape}"
        )
    if q_batches.shape[0] != t_batches.shape[0] or q_batches.shape[0] != distance_batches.shape[0]:
        raise ValueError("Batch dimension K must match across q_batches, t_batches, and distance_batches")
    if patch_ids.shape[0] != q_batches.shape[0]:
        raise ValueError(f"patch_ids must have length {q_batches.shape[0]}, got {patch_ids.shape[0]}")

    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, patch_id in enumerate(patch_ids):
        patch_prefix = f"patch_{int(patch_id):03d}"
        csv_path = os.path.join(output_dir, f"{patch_prefix}.csv")
        png_path = os.path.join(output_dir, f"{patch_prefix}.png")
        export_chamfer_correspondences(
            q_points=q_batches[batch_idx],
            t_points=t_batches[batch_idx],
            distance_matrix=distance_batches[batch_idx],
            output_csv_path=csv_path,
            output_png_path=png_path,
            patch_ids=np.full(q_batches.shape[1], int(patch_id), dtype=np.int32),
            max_lines=max_lines,
            point_size=point_size,
            line_width=line_width,
            plot_direction=plot_direction,
        )


__all__ = [
    "export_chamfer_correspondences",
    "export_patchwise_chamfer_correspondences",
    "export_combined_correspondence_ply",
    "export_global_correspondence_ply",
]


def export_boundary_correspondence_debug(
    model_batches: np.ndarray,
    target_batches: np.ndarray,
    patch_ids: np.ndarray,
    edge_names,
    output_dir: str,
    max_lines: int = 200,
) -> None:
    """Export debug visualizations for outer-boundary rectangle correspondences."""
    model_batches = np.asarray(model_batches, dtype=np.float32)
    target_batches = np.asarray(target_batches, dtype=np.float32)
    patch_ids = np.asarray(patch_ids, dtype=np.int32)

    if model_batches.ndim != 3 or model_batches.shape[-1] != 3:
        raise ValueError(f"model_batches must have shape (K, N, 3), got {model_batches.shape}")
    if target_batches.shape != model_batches.shape:
        raise ValueError(
            f"target_batches must match model_batches shape, got {target_batches.shape} vs {model_batches.shape}"
        )
    if patch_ids.shape[0] != model_batches.shape[0]:
        raise ValueError(f"patch_ids must have length {model_batches.shape[0]}, got {patch_ids.shape[0]}")
    if len(edge_names) != model_batches.shape[0]:
        raise ValueError(f"edge_names must have length {model_batches.shape[0]}, got {len(edge_names)}")

    os.makedirs(output_dir, exist_ok=True)

    for idx, (patch_id, edge_name) in enumerate(zip(patch_ids, edge_names)):
        q_points = model_batches[idx]
        t_points = target_batches[idx]
        n = q_points.shape[0]
        pair_idx = np.arange(n, dtype=np.int32)

        csv_path = os.path.join(output_dir, f"patch_{int(patch_id):03d}_{edge_name}.csv")
        png_path = os.path.join(output_dir, f"patch_{int(patch_id):03d}_{edge_name}.png")
        ply_path = os.path.join(output_dir, f"patch_{int(patch_id):03d}_{edge_name}.ply")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER + ["edge_name", "sample_index"])
            for sample_idx, (q, t) in enumerate(zip(q_points, t_points)):
                dist = float(np.linalg.norm(q - t))
                writer.writerow([
                    sample_idx,
                    int(patch_id),
                    "boundary_direct",
                    float(q[0]), float(q[1]), float(q[2]),
                    float(t[0]), float(t[1]), float(t[2]),
                    dist,
                    edge_name,
                    sample_idx,
                ])

        _plot_correspondences(
            q_points=q_points,
            t_points=t_points,
            q_to_t_idx=pair_idx,
            t_to_q_idx=pair_idx,
            output_png_path=png_path,
            max_lines=max_lines,
            point_size=8.0,
            line_width=1.2,
            plot_direction="both",
        )

        export_correspondence_ply(
            q_points=q_points,
            t_points=t_points,
            q_to_t_idx=pair_idx,
            t_to_q_idx=pair_idx,
            output_ply_path=ply_path,
            plot_direction="both",
        )

    export_combined_correspondence_ply(
        q_batches=model_batches,
        t_batches=target_batches,
        distance_batches=np.linalg.norm(
            target_batches[:, :, None, :] - model_batches[:, None, :, :], axis=-1
        ).astype(np.float32),
        output_ply_path=os.path.join(output_dir, "all_boundary_edges_combined.ply"),
        plot_direction="both",
    )


__all__.append("export_boundary_correspondence_debug")
