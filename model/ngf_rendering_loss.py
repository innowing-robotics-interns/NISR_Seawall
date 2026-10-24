#!/usr/bin/env python3
# ngf_rendering_loss.py

"""NGF's rendering loss with a single, front-facing layer.

This is the geometry-supervision term from Neural Geometry Fields (NGF):

    render_loss    = L1(render(vertices, normals, faces, views), reference)
    laplacian_loss = L1(vertices, mean_neighbors(vertices))
    total          = render_loss + laplacian_weight * laplacian_loss

Each rendered pixel stores ``[normal (3), world position (3)]``.  The camera,
projection, bottom-up image order and antialiasing match NGF's ``render.py``,
so reference maps produced by NGF's renderer (or by a matching point-cloud
splatter) can be consumed directly.

A point cloud observes only the front surface, so this module rasterizes one
layer.  NGF's mesh trainer peels three layers; for that variant the reference
must contain 18 channels and this module does not apply.

The module is wired into ``train_multi_patch`` behind ``--lambda_render``
(fixed-grid single-sheet atlas only). ``nvdiffrast`` is imported lazily because
it is only needed by the renderer;
``UniformLaplacian`` is pure torch.  Integration gaps with NISR's atlas
representation are documented in ``ngf_rendering_loss_notes.md`` next to this
file.  The ``views`` tensor contains world-to-camera matrices ``(B, 4, 4)``
with columns ``[right, up, look, eye]``, and vertices must already be in the
normalized frame used for the reference.

Example:
    renderer = NGFRenderer(width=256, height=256, fov=45.0)
    lap = UniformLaplacian(faces, vertices.shape[0], device=vertices.device)
    for ...:
        loss = ngf_rendering_loss(
            vertices, normals, faces, views, reference,
            renderer=renderer, laplacian=lap)
        loss.backward()

    # Loss terms for logging:
    loss, parts = ngf_rendering_loss(..., return_parts=True)
"""

import math

import torch

try:
    import nvdiffrast.torch as dr
except ImportError:  # The Laplacian remains usable without nvdiffrast.
    dr = None


__all__ = ['NGFRenderer', 'UniformLaplacian', 'ngf_rendering_loss']


class NGFRenderer:
    """Rasterize vertices/normals into NGF-compatible geometry maps.

    The projection negates x, uses ``w = camera z`` and writes images in
    nvdiffrast's bottom-up order.  ``views`` are world-to-camera matrices
    ``(B, 4, 4)`` with columns ``[right, up, look, eye]``.
    """

    def __init__(self, width: int = 256, height: int = 256,
                 fov: float = 45.0, near: float = 0.1,
                 far: float = 1000.0, device: str = 'cuda') -> None:
        if dr is None:
            raise ImportError(
                "nvdiffrast is required by NGFRenderer; install it with "
                "'pip install git+https://github.com/NVlabs/nvdiffrast'.")
        self.res = (height, width)
        self.proj = self.projection(fov, width / height, near, far, device)
        self.ctx = dr.RasterizeCudaContext()

    @staticmethod
    def projection(fov: float, ar: float, near: float, far: float,
                   device: str = 'cuda') -> torch.Tensor:
        """NGF's perspective projection, including its x-axis flip."""
        fov_rad = math.radians(fov)
        tan_half = math.tan(fov_rad / 2.0)
        return torch.tensor([
            [-1.0 / tan_half, 0.0, 0.0, 0.0],
            [0.0, ar / tan_half, 0.0, 0.0],
            [0.0, 0.0, -(near + far) / (near - far),
             2.0 * far * near / (near - far)],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float32, device=device)

    def render(self, vertices: torch.Tensor, normals: torch.Tensor,
               faces: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        """Render one layer as ``(B, H, W, 6)``: normal then world position."""
        if views.ndim != 3 or views.shape[-2:] != (4, 4):
            raise ValueError(f'views must be (B, 4, 4), got {tuple(views.shape)}')
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError(
                f'vertices must be (N, 3), got {tuple(vertices.shape)}')
        if normals.shape != vertices.shape:
            raise ValueError(
                f'normals must match vertices, got {tuple(normals.shape)}')

        device = vertices.device
        vertices = vertices.contiguous()
        normals = normals.contiguous()
        views = views.to(device=device, dtype=vertices.dtype).contiguous()
        faces = faces.to(device=device, dtype=torch.int32).contiguous()

        mvps = self.proj.to(device=device, dtype=vertices.dtype) @ views
        v_hom = torch.nn.functional.pad(vertices, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))

        rast = dr.rasterize(self.ctx, v_ndc, faces, self.res)[0]

        rendered_normals = dr.interpolate(normals, rast, faces)[0]
        rendered_normals = dr.antialias(
            rendered_normals, rast, v_ndc, faces)

        rendered_positions = dr.interpolate(vertices, rast, faces)[0]
        rendered_positions = dr.antialias(
            rendered_positions, rast, v_ndc, faces)

        return torch.cat([rendered_normals, rendered_positions], dim=-1)


class UniformLaplacian:
    """Uniform umbrella smoother over mesh edges, as used by NGF.

    NGF applies this to its uniformly sampled patch vertices:
    ``|V - mean(neighbors)|``.  This class exposes the same operator over an
    arbitrary triangle mesh.  Build it once and reuse it across iterations.

    ``faces`` may live on CPU; edges are moved to ``device`` when requested.
    Vertices passed to :meth:`smooth` and :meth:`loss` must have
    ``num_vertices`` rows and may carry any feature width.
    """

    def __init__(self, faces: torch.Tensor, num_vertices: int = None,
                 device: torch.device = None) -> None:
        if device is None:
            device = faces.device
        faces = faces.to(device=device, dtype=torch.long)
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError(f'faces must be (F, 3), got {tuple(faces.shape)}')
        if num_vertices is None:
            num_vertices = int(faces.max().item()) + 1

        edges = torch.cat([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ], dim=0)
        # Each directed edge is one neighbor relation.
        edges = torch.cat([edges, edges.flip(1)], dim=0)

        flat = edges[:, 0] * num_vertices + edges[:, 1]
        flat = torch.unique(flat)
        src = torch.div(flat, num_vertices, rounding_mode='floor')
        dst = flat - src * num_vertices

        degree = torch.zeros(num_vertices, dtype=torch.float32, device=device)
        degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))

        self.num_vertices = num_vertices
        self.src = src
        self.dst = dst
        self.inv_degree = 1.0 / degree.clamp_min(1.0)
        self.device = device

    def smooth(self, vertices: torch.Tensor) -> torch.Tensor:
        """Replace each vertex by the mean of its one-ring neighbors."""
        if vertices.shape[0] != self.num_vertices:
            raise ValueError(
                f'expected {self.num_vertices} vertices, got {vertices.shape[0]}')
        accumulator = torch.zeros_like(vertices)
        accumulator.index_add_(0, self.src, vertices[self.dst])
        return accumulator * self.inv_degree.unsqueeze(-1)

    def loss(self, vertices: torch.Tensor) -> torch.Tensor:
        """NGF's regularizer: mean absolute displacement from the smooth."""
        return (vertices - self.smooth(vertices)).abs().mean()


def ngf_rendering_loss(vertices: torch.Tensor,
                       normals: torch.Tensor,
                       faces: torch.Tensor,
                       views: torch.Tensor,
                       reference: torch.Tensor,
                       renderer: NGFRenderer = None,
                       laplacian: UniformLaplacian = None,
                       laplacian_weight: float = 1.0,
                       return_parts: bool = False):
    """Compute NGF's rendering loss plus its uniform-Laplacian regularizer.

    Args:
        vertices: ``(N, 3)`` world-space vertices in the normalized frame.
        normals: ``(N, 3)`` unit vertex normals.
        faces: ``(F, 3)`` triangle indices.
        views: ``(B, 4, 4)`` world-to-camera matrices.
        reference: ``(B, H, W, 6)`` target ``[normal, position]``.  A 7-channel
            point-cloud target is also accepted; the trailing alpha is ignored.
        renderer: Reuse an :class:`NGFRenderer` across iterations.  If omitted,
            one is built at the reference resolution (expensive per call).
        laplacian: Reuse a :class:`UniformLaplacian` across iterations.  If
            omitted, it is built from ``faces`` on every call.
        laplacian_weight: Weight of the original Laplacian term. Set to 0 to
            disable it.
        return_parts: Also return detached ``render`` and ``laplacian`` terms
            for logging.

    Returns:
        The scalar loss, or ``(loss, {'render': ..., 'laplacian': ...})``.
    """
    if reference.ndim != 4 or reference.shape[-1] not in (6, 7):
        raise ValueError(
            'reference must be (B, H, W, 6) or (B, H, W, 7), got '
            f'{tuple(reference.shape)}')
    if reference.shape[0] != views.shape[0]:
        raise ValueError(
            f'reference has {reference.shape[0]} views, views has '
            f'{views.shape[0]}')

    device = vertices.device
    target = reference[..., :6].to(device=device, dtype=vertices.dtype)

    if renderer is None:
        renderer = NGFRenderer(
            width=reference.shape[2], height=reference.shape[1],
            device=str(device))
    prediction = renderer.render(vertices, normals, faces, views)
    if prediction.shape != target.shape:
        raise ValueError(
            f'renderer produced {tuple(prediction.shape)}, expected '
            f'{tuple(target.shape)}')

    render_loss = (prediction - target).abs().mean()

    if laplacian_weight == 0.0:
        laplacian_loss = torch.zeros((), device=device, dtype=vertices.dtype)
    else:
        if laplacian is None:
            laplacian = UniformLaplacian(
                faces, vertices.shape[0], device=device)
        laplacian_loss = laplacian.loss(vertices)

    total = render_loss + laplacian_weight * laplacian_loss
    if return_parts:
        return total, {
            'render': render_loss.detach(),
            'laplacian': laplacian_loss.detach(),
        }
    return total
