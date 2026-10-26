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

A point cloud observes only the front surface, so the default path rasterizes
one layer.  ``NGFRenderer.render_layers`` and the ``layers=`` argument of
``ngf_rendering_loss`` add NGF's depth-peeling variant: hit triangles are
removed after each pass, later passes supervise the surfaces behind them, and
the reference must carry ``6 * layers`` channels (the point-cloud target
builds them with ``build_render_targets(..., num_layers=...)``).

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
        return self.render_layers(vertices, normals, faces, views, layers=1)[0]

    def render_layers(self, vertices: torch.Tensor, normals: torch.Tensor,
                      faces: torch.Tensor, views: torch.Tensor,
                      layers: int = 1) -> list:
        """Render ``layers`` depth-peeled surfaces, nearest first.

        This is the depth-peeling operation of NGF's mesh trainer: after a pass,
        the triangles that won the z-test in that pass are removed from the
        face list and the remaining faces are rasterized again.  Visibility and
        face removal are discrete (as in NGF); gradients flow into the positions
        and normals of the triangles selected by each pass.

        Peeling is done per view, not across the batch: a triangle that is the
        front surface in view A can be a legitimate back surface in view B, so
        a batch-wide removal would silently drop it from B's second layer.
        Removal is still per triangle, not per pixel, which is the standard
        peeling approximation and is why the second layer must be validated
        against the point-cloud target before it is trusted.
        """
        if views.ndim != 3 or views.shape[-2:] != (4, 4):
            raise ValueError(f'views must be (B, 4, 4), got {tuple(views.shape)}')
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError(
                f'vertices must be (N, 3), got {tuple(vertices.shape)}')
        if normals.shape != vertices.shape:
            raise ValueError(
                f'normals must match vertices, got {tuple(normals.shape)}')
        if layers < 1:
            raise ValueError(f'layers must be >= 1, got {layers}')

        device = vertices.device
        vertices = vertices.contiguous()
        normals = normals.contiguous()
        views = views.to(device=device, dtype=vertices.dtype).contiguous()
        faces = faces.to(device=device, dtype=torch.int32).contiguous()

        mvps = self.proj.to(device=device, dtype=vertices.dtype) @ views
        v_hom = torch.nn.functional.pad(vertices, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))

        if layers == 1:
            # Keep the original batched path bit-for-bit.
            rast = dr.rasterize(self.ctx, v_ndc, faces, self.res)[0]
            rendered_normals = dr.antialias(
                dr.interpolate(normals, rast, faces)[0], rast, v_ndc, faces)
            rendered_positions = dr.antialias(
                dr.interpolate(vertices, rast, faces)[0], rast, v_ndc, faces)
            return [torch.cat([rendered_normals, rendered_positions], dim=-1)]

        per_view = [[] for _ in range(layers)]
        for batch_index in range(views.shape[0]):
            view_ndc = v_ndc[batch_index:batch_index + 1]
            remaining = faces
            for layer in range(layers):
                if remaining.shape[0] == 0:
                    per_view[layer].append(torch.zeros(
                        (1, *self.res, 6), device=device,
                        dtype=vertices.dtype))
                    continue
                rast = dr.rasterize(self.ctx, view_ndc, remaining,
                                    self.res)[0]
                rendered_normals = dr.antialias(
                    dr.interpolate(normals, rast, remaining)[0],
                    rast, view_ndc, remaining)
                rendered_positions = dr.antialias(
                    dr.interpolate(vertices, rast, remaining)[0],
                    rast, view_ndc, remaining)
                per_view[layer].append(torch.cat(
                    [rendered_normals, rendered_positions], dim=-1))

                ids = rast[..., 3].long()
                hit = torch.unique(ids[ids > 0]) - 1
                if hit.numel() == 0:
                    remaining = remaining[:0]
                    continue
                keep_face = torch.ones(remaining.shape[0], dtype=torch.bool,
                                       device=device)
                keep_face[hit] = False
                remaining = remaining[keep_face]

        return [torch.cat(view_maps, dim=0) for view_maps in per_view]


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
                       view_clip: float = 0.0,
                       pixel_clip: float = 0.0,
                       layers: int = 1,
                       layer_weights=None,
                       return_parts: bool = False):
    """Compute NGF's rendering loss plus its uniform-Laplacian regularizer.

    Args:
        vertices: ``(N, 3)`` world-space vertices in the normalized frame.
        normals: ``(N, 3)`` unit vertex normals.
        faces: ``(F, 3)`` triangle indices.
        views: ``(B, 4, 4)`` world-to-camera matrices.
        reference: ``(B, H, W, 6)`` target ``[normal, position]``.  A 7-channel
            point-cloud target is also accepted; the trailing alpha is ignored.
            With ``layers > 1`` this must be ``(B, H, W, 7 * layers)`` (one
            peeled ``[normal, position, alpha]`` block per layer) or
            ``6 * layers`` without alpha; far layers are then matched on pixels
            where the target alpha exceeds 0.05 and the model rendered
            something.
        renderer: Reuse an :class:`NGFRenderer` across iterations.  If omitted,
            one is built at the reference resolution (expensive per call).
        laplacian: Reuse a :class:`UniformLaplacian` across iterations.  If
            omitted, it is built from ``faces`` on every call.
        laplacian_weight: Weight of the original Laplacian term. Set to 0 to
            disable it.
        view_clip: Robustness factor for the render term. When ``> 0``, each
            view's mean absolute error is capped at ``view_clip`` times the
            batch median view error before averaging, so a few views with
            abnormally large errors cannot dominate the update. ``0`` keeps
            the original unclipped objective.  Only available with
            ``layers == 1``.
        pixel_clip: Robustness factor at pixel level. When ``> 0``, the mean
            absolute error over the six channels at each pixel is capped at
            ``pixel_clip`` times the median over covered pixels (target or
            prediction non-empty). Background pixels stay zero. ``0`` keeps
            the original per-element L1.  Only available with ``layers == 1``.
        layers: Number of depth-peeled surfaces rendered and supervised.  Each
            layer is compared with NGF's full-image L1 over its own six
            channels; the layers are summed with ``layer_weights``.
        layer_weights: Per-layer weights, length ``layers``.  Defaults to
            ``[1.0] + [0.5] * (layers - 1)``.  Layer 0 keeps NGF's original
            weight.
        return_parts: Also return detached ``render`` and ``laplacian`` terms
            for logging.

    Returns:
        The scalar loss, or ``(loss, {'render': ..., 'laplacian': ...})``.
    """
    if reference.ndim != 4:
        raise ValueError(f'reference must be (B, H, W, C), got '
                         f'{tuple(reference.shape)}')
    if not 1 <= layers <= 4:
        raise ValueError(f'layers must be in [1, 4], got {layers}')
    if layers == 1:
        if reference.shape[-1] not in (6, 7):
            raise ValueError(
                'reference must be (B, H, W, 6) or (B, H, W, 7), got '
                f'{tuple(reference.shape)}')
    elif reference.shape[-1] not in (6 * layers, 7 * layers):
        raise ValueError(
            f'with layers={layers}, reference must have {6 * layers} or '
            f'{7 * layers} channels, got {reference.shape[-1]}')
    if reference.shape[0] != views.shape[0]:
        raise ValueError(
            f'reference has {reference.shape[0]} views, views has '
            f'{views.shape[0]}')

    device = vertices.device

    if renderer is None:
        renderer = NGFRenderer(
            width=reference.shape[2], height=reference.shape[1],
            device=str(device))

    if layers > 1:
        if view_clip > 0.0 or pixel_clip > 0.0:
            raise NotImplementedError(
                'view_clip/pixel_clip are only implemented for layers=1')
        if layer_weights is None:
            layer_weights = [1.0] + [0.5] * (layers - 1)
        if len(layer_weights) != layers:
            raise ValueError(
                f'layer_weights has {len(layer_weights)} entries, expected '
                f'{layers}')
        references = reference.to(device=device, dtype=vertices.dtype)
        has_alpha = reference.shape[-1] == 7 * layers
        stride = 7 if has_alpha else 6
        targets = [references[..., stride * layer:stride * layer + 6]
                   for layer in range(layers)]
        predictions = renderer.render_layers(
            vertices, normals, faces, views, layers=layers)
        layer_errors = [(pred - tgt).abs()
                        for pred, tgt in zip(predictions, targets)]
        # Layer 0 keeps NGF's full-frame reduction exactly.  Deeper layers are
        # matched-pixel terms: only pixels where the target has a surface AND
        # the model actually produced one are compared.  The sum is still
        # divided by B*H*W (not by the number of matches), so the far term
        # carries the same units as the front term and cannot dominate simply
        # because far surfaces cover few pixels.  Pixels the model leaves empty
        # are not rewarded either way -- the concern there is that a full-frame
        # far-layer L1 would push triangles to grow into peeling holes.
        layer_terms = [layer_errors[0].mean()]
        layer_matches = [torch.ones((), device=device, dtype=vertices.dtype)]
        for layer in range(1, layers):
            pixel_error = layer_errors[layer].mean(dim=-1)
            if has_alpha:
                target_alpha = references[..., stride * layer + 6]
            else:
                target_alpha = (
                    targets[layer][..., 3:].norm(dim=-1) > 1e-6).to(
                        pixel_error.dtype)
            pred_alpha = predictions[layer][..., 3:].norm(dim=-1)
            mask = (target_alpha > 0.05) & (pred_alpha > 1e-6)
            layer_matches.append(mask.float().mean().detach())
            layer_terms.append(
                (pixel_error * mask).sum() / mask.numel())
        render_loss = sum(weight * term for weight, term
                          in zip(layer_weights, layer_terms))
        # view-level diagnostics below use the front layer only
        error = layer_errors[0]
        per_view = error.reshape(error.shape[0], -1).mean(dim=1)
        pixel_cap = torch.zeros((), device=device, dtype=vertices.dtype)
        pixel_clip_fraction = torch.zeros((), device=device,
                                          dtype=vertices.dtype)
        cap = torch.zeros((), device=device, dtype=vertices.dtype)
        clip_fraction = torch.zeros((), device=device,
                                    dtype=vertices.dtype)
    else:
        layer_terms = None
        layer_matches = None
        target = reference[..., :6].to(device=device, dtype=vertices.dtype)
        prediction = renderer.render(vertices, normals, faces, views)
        if prediction.shape != target.shape:
            raise ValueError(
                f'renderer produced {tuple(prediction.shape)}, expected '
                f'{tuple(target.shape)}')

        error = (prediction - target).abs()
        if pixel_clip > 0.0:
            pixel_error = error.mean(dim=-1)              # (B, H, W)
            target_pos = target[..., 3:].norm(dim=-1)
            pred_pos = prediction[..., 3:].norm(dim=-1)
            covered = (target_pos > 1e-6) | (pred_pos > 1e-6)
            if covered.any():
                pixel_median = pixel_error[covered].median().detach()
            else:
                pixel_median = pixel_error.median().detach()
            pixel_cap = (pixel_clip * pixel_median).clamp_min(1e-8)
            pixel_used = torch.clamp(pixel_error, max=pixel_cap)
            pixel_clip_fraction = (
                (pixel_error.detach()[covered] > pixel_cap).float().mean()
                if covered.any() else torch.zeros((), device=device,
                                                  dtype=vertices.dtype))
            # A pixel's channel-mean already has the scale of the original L1
            # over all 6*B*H*W elements, so averaging those means is exact.
            per_view = pixel_used.reshape(pixel_used.shape[0], -1).mean(dim=1)
        else:
            per_view = error.reshape(error.shape[0], -1).mean(dim=1)
            pixel_cap = torch.zeros((), device=device, dtype=vertices.dtype)
            pixel_clip_fraction = torch.zeros((), device=device,
                                              dtype=vertices.dtype)
        if view_clip > 0.0:
            # Per-view robust cap relative to the batch's own error scale.  The
            # median keeps the threshold meaningful early in training and for
            # shapes whose reference error is intrinsically higher; only views
            # far above their peers are saturated.
            cap = (view_clip * per_view.median().detach()).clamp_min(1e-8)
            render_loss = torch.clamp(per_view, max=cap).mean()
            clip_fraction = (per_view.detach() > cap).float().mean()
        else:
            render_loss = per_view.mean()
            cap = torch.zeros((), device=device, dtype=vertices.dtype)
            clip_fraction = torch.zeros((), device=device, dtype=vertices.dtype)

    if laplacian_weight == 0.0:
        laplacian_loss = torch.zeros((), device=device, dtype=vertices.dtype)
    else:
        if laplacian is None:
            laplacian = UniformLaplacian(
                faces, vertices.shape[0], device=device)
        laplacian_loss = laplacian.loss(vertices)

    total = render_loss + laplacian_weight * laplacian_loss
    if return_parts:
        parts = {
            'render': render_loss.detach(),
            'render_raw': error.mean().detach(),
            'render_view_max': per_view.max().detach(),
            'render_clip': cap.detach(),
            'render_clip_frac': clip_fraction.detach(),
            'render_pixel_clip': pixel_cap.detach(),
            'render_pixel_clip_frac': pixel_clip_fraction.detach(),
            'laplacian': laplacian_loss.detach(),
        }
        if layer_terms is not None:
            for index, term in enumerate(layer_terms):
                parts[f'render_layer_{index + 1}'] = term.detach()
            for index, match in enumerate(layer_matches[1:], start=2):
                parts[f'render_layer_{index}_match'] = match.detach()
        return total, parts
    return total
