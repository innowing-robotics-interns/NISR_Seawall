#!/usr/bin/env python3
"""NGF-compatible point-cloud reference maps for NISR.

The rendering loss consumes a front layer of ``[normal (3), position (3)]``.
This module turns a point cloud into those layers with an object-centric
Fibonacci orbit and a z-buffered screen-space splat.  Cameras, projection,
bottom-up image order and world-space positions intentionally match
``model/ngf_rendering_loss.py`` (and therefore NGF's renderer), so a target and
a mesh render can be compared pixel by pixel.

The point cloud is assumed to already be in the same normalized frame as the
model.  In NISR's training pipeline that means using ``pts3n`` and
``meta['normals']`` from ``utils.load_point_cloud`` directly; do not apply
NGF's bbox normalizer on top.

Ported from ``ngf_yl/source/pointcloud_render.py``.  The splat and camera code
is kept in sync with that reference; only the loader and training-facing
container are adapted to NISR.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - scipy is a project dependency
    cKDTree = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - only needed for previews
    imageio = None


__all__ = [
    'PointCloud',
    'PointCloudRender',
    'RenderTargetSet',
    'build_render_targets',
    'estimate_spacing',
    'make_orbit_views',
    'prepare_point_cloud',
    'projection_matrix',
    'render_pointcloud',
    'save_comparison',
    'save_preview',
]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass
class PointCloud:
    points: torch.Tensor                 # (N, 3), normalized frame
    normals: torch.Tensor                # (N, 3), unit length
    spacing: torch.Tensor                # (N,), local nearest-neighbour distance
    center: Optional[torch.Tensor] = None
    extent: Optional[float] = None

    def to(self, device) -> 'PointCloud':
        return PointCloud(
            self.points.to(device),
            self.normals.to(device),
            self.spacing.to(device),
            None if self.center is None else self.center.to(device),
            self.extent,
        )


@dataclass
class PointCloudRender:
    normal: torch.Tensor                  # (V, H, W, 3)
    position: torch.Tensor                # (V, H, W, 3)
    alpha: torch.Tensor                   # (V, H, W, 1) in [0, 1]
    depth: torch.Tensor                   # (V, H, W, 1), view-space z
    color: Optional[torch.Tensor] = None  # (V, H, W, 3)
    views: Optional[torch.Tensor] = None  # (V, 4, 4)

    def geometry(self, alpha: bool = True) -> torch.Tensor:
        """Concatenate ``[normal, position]``; alpha appends splat coverage."""
        channels = [self.normal, self.position]
        if alpha:
            channels.append(self.alpha)
        return torch.cat(channels, dim=-1)

    def coverage(self) -> torch.Tensor:
        """Mean alpha per view, shape ``(V,)``."""
        return self.alpha.mean(dim=(1, 2, 3))


@dataclass
class RenderTargetSet:
    """Cached reference views in NISR's normalized frame.

    ``geometry`` has seven channels (normal, position, alpha).  The loss slices
    off the alpha channel; it is retained for coverage filtering and previews.
    All tensors are on CPU so they do not compete with the model for GPU
    memory.
    """

    views: torch.Tensor                   # (V, 4, 4) world-to-camera
    eyes: torch.Tensor                    # (V, 3)
    center: torch.Tensor                  # (3,) orbit centre
    radius: float                         # bounding-sphere radius
    geometry: torch.Tensor                # (V, H, W, 7) float32
    coverage: torch.Tensor                # (V,) mean alpha
    fov: float
    resolution: Tuple[int, int]           # (height, width)
    pointcloud: Optional[PointCloud] = None

    def __len__(self) -> int:
        return int(self.views.shape[0])

    def view_batches(self, batch_size: int):
        """Split views and references into matching batches.

        Returns a list of ``(views, reference)`` pairs.  References stay on CPU;
        the training loop moves each batch to the render device.
        """
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        pairs = []
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            pairs.append((self.views[start:stop], self.geometry[start:stop]))
        return pairs

    def summary(self) -> Dict[str, object]:
        return {
            'views': len(self),
            'resolution': list(self.resolution),
            'fov': float(self.fov),
            'radius': float(self.radius),
            'coverage_mean': float(self.coverage.mean()),
            'coverage_min': float(self.coverage.min()),
            'coverage_max': float(self.coverage.max()),
            'center': self.center.tolist(),
        }


# ---------------------------------------------------------------------------
# Point-cloud preparation
# ---------------------------------------------------------------------------


def estimate_spacing(points: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Median distance to the k nearest neighbours, in normalized units."""
    if cKDTree is None:
        raise ImportError('scipy is required for point-cloud spacing')
    n = int(points.shape[0])
    if n <= 1:
        return torch.full((n,), 1e-3, device=points.device, dtype=points.dtype)

    k = max(2, min(int(k), n))
    host = points.detach().cpu().numpy()
    tree = cKDTree(host)
    dist, _ = tree.query(host, k=k)
    if dist.ndim == 1:
        dist = dist[:, None]
    spacing = np.median(dist[:, 1:], axis=1)
    return torch.as_tensor(spacing, dtype=points.dtype,
                           device=points.device).clamp_min(1e-6)


def _estimate_normals(points: torch.Tensor,
                      k: int = 16,
                      orient: str = 'auto') -> torch.Tensor:
    """PCA normals with optional Open3D orientation (portable fallback)."""
    if cKDTree is None:
        raise ImportError('scipy is required for normal estimation')

    n = int(points.shape[0])
    k = max(3, min(int(k) + 1, n))
    host = points.detach().cpu().numpy()
    tree = cKDTree(host)
    _, indices = tree.query(host, k=k)

    neighbours = points.detach().cpu()[torch.as_tensor(
        indices[:, 1:], dtype=torch.long)]
    neighbours = neighbours - neighbours.mean(dim=1, keepdim=True)
    covariance = neighbours.transpose(1, 2) @ neighbours
    _, eigenvectors = torch.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]

    oriented = False
    if orient in ('auto', 'open3d'):
        try:
            import open3d as o3d

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(host)
            cloud.normals = o3d.utility.Vector3dVector(normals.numpy())
            cloud.orient_normals_consistent_tangent_plane(20)
            normals = torch.as_tensor(np.asarray(cloud.normals),
                                      dtype=points.dtype)
            oriented = True
        except Exception:
            oriented = False

    if not oriented:
        # Star-shaped fallback: point normals away from the centroid.
        direction = points.detach().cpu() - points.detach().cpu().mean(
            dim=0, keepdim=True)
        flip = (direction * normals).sum(dim=-1) < 0
        normals[flip] *= -1.0

    length = normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (normals / length).to(points.device)


def prepare_point_cloud(points,
                        normals=None,
                        device=None,
                        k_spacing: int = 7,
                        estimate_normals_if_missing: bool = False,
                        orient_normals: str = 'auto') -> PointCloud:
    """Validate and prepare NISR's normalized points and normals.

    Args:
        points: ``(N, 3)`` array or tensor in the model's normalized frame.
        normals: ``(N, 3)`` array or tensor.  Required unless
            ``estimate_normals_if_missing`` is set.
        device: target device.  Defaults to CUDA when available.
        k_spacing: neighbours used for the local splat radius.
        estimate_normals_if_missing: PCA + Open3D fallback.  Off by default so
            a missing-normal run fails visibly rather than silently changing
            the target.
        orient_normals: orientation mode for the fallback estimator.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    points = torch.as_tensor(points, dtype=torch.float32, device=device)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f'points must be (N, 3), got {tuple(points.shape)}')
    if points.shape[0] < 2:
        raise ValueError('at least two points are required')
    if not bool(torch.isfinite(points).all()):
        raise ValueError('points contain NaN or Inf')

    if normals is None:
        if not estimate_normals_if_missing:
            raise ValueError(
                'normals are required for point-cloud reference maps; pass '
                'meta["normals"] or enable estimate_normals_if_missing')
        normals = _estimate_normals(points, k=16, orient=orient_normals)
    else:
        normals = torch.as_tensor(normals, dtype=torch.float32, device=device)
        if normals.shape != points.shape:
            raise ValueError(
                f'normals must match points, got {tuple(normals.shape)} and '
                f'{tuple(points.shape)}')
        if not bool(torch.isfinite(normals).all()):
            raise ValueError('normals contain NaN or Inf')
        length = normals.norm(dim=-1, keepdim=True)
        if float(length.max()) < 1e-6:
            raise ValueError('normals are all zero')
        normals = normals / length.clamp_min(1e-8)

    vmin = points.min(dim=0).values
    vmax = points.max(dim=0).values
    center = (vmin + vmax) * 0.5
    extent = float((vmax - vmin).max().item()) if points.numel() else 0.0
    spacing = estimate_spacing(points, k=k_spacing)

    return PointCloud(
        points=points.contiguous(),
        normals=normals.contiguous(),
        spacing=spacing.contiguous(),
        center=center,
        extent=extent,
    )


# ---------------------------------------------------------------------------
# Cameras and projection (NGF convention)
# ---------------------------------------------------------------------------


def projection_matrix(fov: float,
                      aspect: float,
                      near: float = 0.1,
                      far: float = 1000.0,
                      device=None) -> torch.Tensor:
    """NGF's perspective projection, including its x-axis flip."""
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    return torch.tensor([
        [-f, 0.0, 0.0, 0.0],
        [0.0, aspect * f, 0.0, 0.0],
        [0.0, 0.0, -(near + far) / (near - far),
         2.0 * far * near / (near - far)],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float32, device=device)


def fibonacci_directions(count: int, device=None) -> torch.Tensor:
    i = torch.arange(int(count), dtype=torch.float32, device=device) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / int(count))
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    return torch.stack([
        torch.cos(theta) * torch.sin(phi),
        torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
    ], dim=-1)


def make_orbit_views(points: torch.Tensor,
                     count: int = 120,
                     fov: float = 45.0,
                     radius_factor: float = 1.08,
                     up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
                     device=None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Object-centric cameras on a bounding sphere.

    Returns world-to-camera matrices ``(V, 4, 4)`` with columns
    ``[right, up, look, eye]``, eye positions, bbox centre and bounding-sphere
    radius.  The camera convention matches the rendering-loss renderer.
    """
    if device is None:
        device = points.device
    device = torch.device(device)
    points = points.to(device)

    center = (points.min(dim=0).values + points.max(dim=0).values) * 0.5
    radius = float((points - center).norm(dim=1).max().item())
    if radius <= 1e-8:
        radius = 1.0
    distance = radius / math.sin(math.radians(fov) / 2.0) * radius_factor

    directions = fibonacci_directions(count, device=device)
    up_vec = torch.tensor(up, dtype=torch.float32, device=device)
    views, eyes = [], []

    for direction in directions:
        eye = center + distance * direction
        look = -direction
        current_up = up_vec
        if abs(torch.dot(look, current_up)) > 0.999:
            current_up = torch.tensor(
                [1.0, 0.0, 0.0], device=device) \
                if abs(look[0]) < 0.9 else torch.tensor(
                    [0.0, 0.0, 1.0], device=device)

        right = torch.cross(look, current_up, dim=-1)
        right = right / right.norm().clamp_min(1e-8)
        up2 = torch.cross(look, right, dim=-1)
        up2 = up2 / up2.norm().clamp_min(1e-8)

        camera_to_world = torch.eye(4, device=device)
        camera_to_world[:3, 0] = right
        camera_to_world[:3, 1] = up2
        camera_to_world[:3, 2] = look
        camera_to_world[:3, 3] = eye
        views.append(camera_to_world.inverse())
        eyes.append(eye)

    return torch.stack(views), torch.stack(eyes), center, radius


# ---------------------------------------------------------------------------
# Splatting
# ---------------------------------------------------------------------------


@torch.no_grad()
def _splat_view(pcd: PointCloud,
                view: torch.Tensor,
                width: int,
                height: int,
                fov: float,
                radius_scale: float,
                depth_tol: float,
                depth_pad: float,
                max_footprint: int,
                softness: float) -> Tuple[torch.Tensor, ...]:
    device = pcd.points.device
    points, normals = pcd.points, pcd.normals
    spacing = pcd.spacing
    colors = None

    projection = projection_matrix(fov, width / height, device=device)
    points_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    camera = points_h @ view.to(device).transpose(0, 1)
    w = camera[:, 2]
    in_front = w > 0.05

    clip = camera @ projection.transpose(0, 1)
    x_ndc = clip[:, 0] / clip[:, 3]
    y_ndc = clip[:, 1] / clip[:, 3]
    px = (x_ndc * 0.5 + 0.5) * width - 0.5
    py = (y_ndc * 0.5 + 0.5) * height - 0.5  # bottom-up, matching nvdiffrast

    pad = max_footprint + 2
    valid = (in_front & (px > -pad) & (px < width + pad)
             & (py > -pad) & (py < height + pad))
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)

    if idx.numel() == 0:
        zeros = torch.zeros(height, width, 3, device=device)
        return (zeros, zeros.clone(), torch.zeros(height, width, 1, device=device),
                torch.zeros(height, width, 1, device=device), None)

    point, normal, wv = points[idx], normals[idx], w[idx]
    pxv, pyv = px[idx], py[idx]
    spacing_v = spacing[idx]

    focal = abs(projection[1, 1].item()) * height / 2.0
    radius_px = (radius_scale * spacing_v * focal / wv).clamp(
        0.5, float(max_footprint * 2))

    quantile = float(torch.quantile(radius_px, 0.95).item())
    footprint = int(min(max_footprint, max(2, math.ceil(quantile * 1.15 + 0.5))))
    offsets = torch.arange(-footprint, footprint + 1, device=device,
                           dtype=torch.float32)
    oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
    ox, oy = ox.reshape(-1), oy.reshape(-1)

    center_x = torch.round(pxv).long()
    center_y = torch.round(pyv).long()
    frac_x = pxv - center_x
    frac_y = pyv - center_y
    radius_eff = radius_px * 1.15 + 0.5

    hw = height * width
    depth_buffer = torch.full((hw,), float('inf'), device=device)

    for dx, dy in zip(ox.tolist(), oy.tolist()):
        x = center_x + int(dx)
        y = center_y + int(dy)
        ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not bool(ok.any()):
            continue
        pixels = y[ok] * width + x[ok]
        depth_buffer.scatter_reduce_(0, pixels, wv[ok], reduce='amin',
                                     include_self=True)

    weight_buffer = torch.zeros(hw, device=device)
    normal_buffer = torch.zeros(hw, 3, device=device)
    position_buffer = torch.zeros(hw, 3, device=device)

    for dx, dy in zip(ox.tolist(), oy.tolist()):
        x = center_x + int(dx)
        y = center_y + int(dy)
        ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not bool(ok.any()):
            continue
        pixels = (y * width + x).clamp(0, hw - 1)

        distance = torch.sqrt((dx + frac_x).square() + (dy + frac_y).square())
        weight = ((radius_eff - distance) / softness).clamp(0.0, 1.0)
        z_min = depth_buffer[pixels]
        keep = ok & (wv <= z_min + depth_tol * z_min + depth_pad * spacing_v)
        weight = weight * keep

        weight_buffer.index_add_(0, pixels, weight)
        normal_buffer.index_add_(0, pixels, normal * weight[:, None])
        position_buffer.index_add_(0, pixels, point * weight[:, None])

    covered = weight_buffer > 1e-8
    normal_buffer[covered] /= weight_buffer[covered, None]
    length = normal_buffer[covered].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normal_buffer[covered] /= length
    position_buffer[covered] /= weight_buffer[covered, None]

    alpha = weight_buffer.clamp(0.0, 1.0)
    depth = depth_buffer
    depth[~torch.isfinite(depth)] = 0.0

    def reshape(t):
        return t.reshape(height, width, -1)

    return reshape(normal_buffer), reshape(position_buffer), reshape(alpha), \
        reshape(depth), None


@torch.no_grad()
def render_pointcloud(pcd: PointCloud,
                      views: torch.Tensor,
                      width: int = 256,
                      height: int = 256,
                      fov: float = 45.0,
                      radius_scale: float = 0.8,
                      depth_tol: float = 0.008,
                      depth_pad: float = 4.0,
                      max_footprint: int = 6,
                      softness: float = 0.8) -> PointCloudRender:
    """Splat the point cloud from every camera; maps are bottom-up."""
    pcd = pcd.to(views.device)
    normal, position, alpha, depth = [], [], [], []

    for view in views:
        n, p, a, d, _ = _splat_view(
            pcd, view, width, height, fov,
            radius_scale, depth_tol, depth_pad, max_footprint, softness,
        )
        normal.append(n)
        position.append(p)
        alpha.append(a)
        depth.append(d)

    return PointCloudRender(
        normal=torch.stack(normal),
        position=torch.stack(position),
        alpha=torch.stack(alpha),
        depth=torch.stack(depth),
        views=views,
    )


# ---------------------------------------------------------------------------
# Training-facing cache
# ---------------------------------------------------------------------------


def build_render_targets(points,
                         normals,
                         num_views: int = 16,
                         resolution: int = 256,
                         fov: float = 45.0,
                         radius_factor: float = 1.08,
                         min_coverage: float = 0.1,
                         device=None,
                         k_spacing: int = 7,
                         estimate_normals_if_missing: bool = False,
                         radius_scale: float = 0.8,
                         depth_tol: float = 0.008,
                         depth_pad: float = 4.0,
                         max_footprint: int = 6,
                         softness: float = 0.8,
                         preview_dir: Optional[str] = None,
                         preview_count: int = 4,
                         verbose: bool = True) -> RenderTargetSet:
    """Render and cache NGF-compatible reference views.

    The target cloud is expected in the model's normalized frame.  Views with
    mean alpha below ``min_coverage`` are dropped; if none survive this raises
    instead of returning an unusable cache.
    """
    if resolution < 16:
        raise ValueError(f'resolution must be >= 16, got {resolution}')
    if num_views < 1:
        raise ValueError(f'num_views must be positive, got {num_views}')

    pcd = prepare_point_cloud(
        points,
        normals,
        device=device,
        k_spacing=k_spacing,
        estimate_normals_if_missing=estimate_normals_if_missing,
    )

    views, eyes, center, radius = make_orbit_views(
        pcd.points, count=num_views, fov=fov,
        radius_factor=radius_factor, device=pcd.points.device,
    )

    if verbose:
        print(f'  [targets] {pcd.points.shape[0]} points, '
              f'spacing median {pcd.spacing.median():.6f}, '
              f'radius {radius:.4f}')
        print(f'  [targets] {num_views} orbit views at '
              f'{resolution}x{resolution}, fov {fov:g}')

    render = render_pointcloud(
        pcd, views, resolution, resolution, fov,
        radius_scale, depth_tol, depth_pad, max_footprint, softness,
    )
    coverage = render.coverage()
    visible = (render.alpha > 0.5).float().mean(dim=(1, 2, 3))

    if verbose:
        print(f'  [targets] coverage mean {coverage.mean():.3f} '
              f'(min {coverage.min():.3f}, max {coverage.max():.3f}); '
              f'pixels >0.5 mean {visible.mean():.3f}')

    if preview_dir is not None and preview_count > 0:
        save_preview(render, preview_dir, count=preview_count)
        if verbose:
            print(f'  [targets] previews -> {preview_dir}')

    keep = coverage >= float(min_coverage)
    if not bool(keep.any()):
        stats = ', '.join(f'{v:.3f}' for v in coverage.tolist())
        raise RuntimeError(
            f'every reference view is below min_coverage={min_coverage}: '
            f'[{stats}]. The cloud may be empty, partial, or in a different '
            'frame.')

    dropped = int((~keep).sum())
    if dropped and verbose:
        print(f'  [targets] dropping {dropped} view(s) below '
              f'min_coverage={min_coverage}')

    geometry = render.geometry().detach().cpu().float()
    views_cpu = views.detach().cpu().float()
    eyes_cpu = eyes.detach().cpu().float()
    coverage_cpu = coverage.detach().cpu().float()
    keep_cpu = keep.detach().cpu()

    return RenderTargetSet(
        views=views_cpu[keep_cpu],
        eyes=eyes_cpu[keep_cpu],
        center=center.detach().cpu().float(),
        radius=float(radius),
        geometry=geometry[keep_cpu],
        coverage=coverage_cpu[keep_cpu],
        fov=float(fov),
        resolution=(int(resolution), int(resolution)),
        pointcloud=pcd,
    )


# ---------------------------------------------------------------------------
# Preview / diagnostics
# ---------------------------------------------------------------------------


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert an HxWx3 float map in bottom-up order to a top-down uint8."""
    image = image.detach().cpu().float().clamp(0.0, 1.0).flip(0)
    return (image.numpy() * 255.0 + 0.5).astype(np.uint8)


def _normal_shading(normals: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    light = torch.tensor([0.3, 0.5, 0.8], device=normals.device,
                         dtype=normals.dtype)
    light = light / light.norm()
    shade = (normals * light).sum(dim=-1).abs().clamp(0.0, 1.0)
    base = (0.08 + 0.72 * shade[..., None]) * torch.tensor(
        [0.70, 0.75, 0.82], device=normals.device, dtype=normals.dtype)
    image = base * alpha + 0.95 * (1.0 - alpha)
    return _to_uint8(image)


def _alpha_image(alpha: torch.Tensor) -> np.ndarray:
    return _to_uint8(alpha.repeat(1, 1, 3))


def _label_uint8(image: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return image

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 8), text, font=font, fill=(255, 255, 255),
              stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(pil)


def _montage(tiles: Sequence[np.ndarray], cols: int = 4) -> np.ndarray:
    if not tiles:
        raise ValueError('cannot build an empty montage')
    cols = max(1, min(int(cols), len(tiles)))
    rows = int(math.ceil(len(tiles) / cols))
    height, width = tiles[0].shape[:2]
    canvas = np.full((rows * height, cols * width, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        canvas[row * height:(row + 1) * height,
               col * width:(col + 1) * width] = tile
    return canvas


def _selected_indices(total: int, count: int) -> np.ndarray:
    count = min(int(count), int(total))
    if count <= 0:
        return np.empty((0,), dtype=int)
    return np.linspace(0, total - 1, count).round().astype(int)


def save_preview(render: PointCloudRender,
                 out_dir: str,
                 count: int = 8) -> None:
    """Save normal-shaded and alpha tiles plus montages."""
    if imageio is None:
        raise ImportError('imageio is required to write previews')

    os.makedirs(out_dir, exist_ok=True)
    indices = _selected_indices(render.normal.shape[0], count)

    normal_tiles, alpha_tiles = [], []
    for index in indices:
        normal_tile = _label_uint8(
            _normal_shading(render.normal[index], render.alpha[index]),
            f'view {index}')
        alpha_tile = _label_uint8(_alpha_image(render.alpha[index]),
                                  f'alpha {index}')
        imageio.imwrite(os.path.join(out_dir, f'normal_{index:04d}.png'),
                        normal_tile)
        imageio.imwrite(os.path.join(out_dir, f'alpha_{index:04d}.png'),
                        alpha_tile)
        normal_tiles.append(normal_tile)
        alpha_tiles.append(alpha_tile)

    if normal_tiles:
        imageio.imwrite(os.path.join(out_dir, 'montage_normals.png'),
                        _montage(normal_tiles, cols=4))
        imageio.imwrite(os.path.join(out_dir, 'montage_alpha.png'),
                        _montage(alpha_tiles, cols=4))


def save_comparison(target_geometry: torch.Tensor,
                    prediction: torch.Tensor,
                    out_dir: str,
                    count: int = 4,
                    prefix: str = 'compare') -> None:
    """Save target / prediction / silhouette-overlap tiles side by side.

    Args:
        target_geometry: ``(V, H, W, 6 or 7)`` with normal, position, alpha.
        prediction: ``(V, H, W, 6)`` normal/position from the renderer.
    """
    if imageio is None:
        raise ImportError('imageio is required to write previews')

    os.makedirs(out_dir, exist_ok=True)
    total = min(target_geometry.shape[0], prediction.shape[0])
    indices = _selected_indices(total, count)

    flat_tiles = []
    for index in indices:
        target = target_geometry[index].detach().cpu()
        pred = prediction[index].detach().cpu()

        target_normal = target[..., :3]
        pred_normal = pred[..., :3]
        if target.shape[-1] >= 7:
            target_alpha = target[..., 6:7]
        else:
            target_alpha = torch.ones_like(target_normal[..., :1])
        pred_alpha = (pred_normal.norm(dim=-1, keepdim=True) > 1e-6).float()

        target_tile = _label_uint8(
            _normal_shading(target_normal, target_alpha), 'target')
        pred_tile = _label_uint8(
            _normal_shading(pred_normal, pred_alpha), 'prediction')

        tmask = (target_alpha > 0.5).squeeze(-1)
        pmask = (pred_alpha > 0.5).squeeze(-1)
        overlay = torch.full((*tmask.shape, 3), 1.0)
        overlay[tmask & ~pmask] = torch.tensor([0.20, 0.45, 0.85])
        overlay[pmask & ~tmask] = torch.tensor([0.95, 0.50, 0.15])
        overlay[tmask & pmask] = torch.tensor([0.55, 0.55, 0.55])
        overlap_tile = _label_uint8(
            _to_uint8(overlay), 'overlap: target blue / pred orange')

        view_tiles = [target_tile, pred_tile, overlap_tile]
        imageio.imwrite(os.path.join(out_dir, f'{prefix}_{index:04d}.png'),
                        _montage(view_tiles, cols=3))
        flat_tiles.extend(view_tiles)

    if flat_tiles:
        imageio.imwrite(os.path.join(out_dir, f'{prefix}_montage.png'),
                        _montage(flat_tiles, cols=3))
