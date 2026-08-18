#!/usr/bin/env python3
# model.py

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
#  Mean Value Coordinates (MVC) — vectorized, differentiable PyTorch port
#  of the demo's mvc.mvc_weights(). This is the core of the bilinear→MVC switch.
# ─────────────────────────────────────────────────────────────────────────────
def mvc_weights_torch(points: torch.Tensor,
                      polys: torch.Tensor,
                      eps: float = 1e-7,
                      edge_eps: float = 1e-6) -> torch.Tensor:
    """
    Batched Mean Value Coordinates (Floater 2003).

    Args
    ----
    points : (B, 2)      query points, one per row.
    polys  : (B, K, 2)   the K polygon vertices for EACH query point, given in a
                         consistent CCW order. In Phase 1 (regular grid) K == 4
                         and the polygon is always the local-UV unit square, so
                         the same square is broadcast to every row. In Phase 2
                         (quadtree) K becomes the leaf's n-gon (corners + hanging
                         nodes) and can differ per query — the math below is
                         already written for the general K case.

    Returns
    -------
    (B, K) weights, non-negative for convex polygons, summing to 1 per row.

    Formula (tangent-of-half-angle):
        tan(alpha_i / 2) = (r_i * r_{i+1} - <s_i, s_{i+1}>) / cross(s_i, s_{i+1})
        w_i = ( tan(alpha_{i-1}/2) + tan(alpha_i/2) ) / r_i
    where s_i = v_i - p and r_i = |s_i|.

    Two degenerate cases are handled exactly like the demo:
      * Case A — query coincides with a vertex  -> one-hot on that vertex.
      * Case B — query lies ON an edge          -> LINEAR interpolation between
        that edge's two endpoints. This is the "on the edge we only use linear
        interpolation" rule you asked for, and it is what keeps neighbouring
        patches continuous when hanging nodes appear in Phase 2. Note it is not
        a hack: as p approaches an edge, MVC naturally converges to linear
        interpolation along it; Case B just evaluates that limit robustly.
    """
    B, K, _ = polys.shape

    s = polys - points.unsqueeze(1)                    # (B, K, 2) vectors p->v_i
    r = torch.linalg.norm(s, dim=-1)                   # (B, K) distances

    # "next" vertex (i -> i+1), wrapping around the polygon.
    s_nxt = torch.roll(s, shifts=-1, dims=1)           # (B, K, 2)
    r_nxt = torch.roll(r, shifts=-1, dims=1)           # (B, K)

    # cross / dot per edge (i, i+1); indexed by the edge's start vertex i.
    cross = s[..., 0] * s_nxt[..., 1] - s[..., 1] * s_nxt[..., 0]   # (B, K)
    dot = (s * s_nxt).sum(dim=-1)                                   # (B, K)

    # tan(alpha_i / 2). Clamp the denominator so degenerate rows stay FINITE
    # (never inf/nan): those rows are overwritten below via Case A / Case B, but
    # keeping them finite avoids the classic torch.where 0*inf=nan grad trap.
    safe_cross = torch.where(cross.abs() < eps,
                             torch.full_like(cross, eps), cross)
    tan_half = (r * r_nxt - dot) / safe_cross          # (B, K), edge i
    tan_half_prev = torch.roll(tan_half, shifts=1, dims=1)  # tan(alpha_{i-1}/2)

    r_safe = torch.where(r < eps, torch.full_like(r, eps), r)
    w = (tan_half_prev + tan_half) / r_safe            # (B, K) unnormalized

    # --- interior (generic) case: normalize ---------------------------------
    w_sum = w.sum(dim=1, keepdim=True)
    w_sum = torch.where(w_sum.abs() < eps, torch.ones_like(w_sum), w_sum)
    w_interior = w / w_sum

    # --- Case B: query on an edge -> linear interpolation on that edge -------
    # An edge (i, i+1) contains p iff p is collinear (cross≈0) AND between the
    # endpoints (dot<0, i.e. the vectors to the two endpoints point opposite).
    on_edge = (cross.abs() < edge_eps) & (dot < 0.0)   # (B, K) flagged at edge i
    any_edge = on_edge.any(dim=1)                      # (B,)

    ei = on_edge.to(w.dtype)
    denom_edge = torch.where((r + r_nxt) < eps,
                             torch.ones_like(r), r + r_nxt)
    w_i = (r_nxt / denom_edge) * ei                    # weight to vertex i
    w_ip1 = (r / denom_edge) * ei                      # weight to vertex i+1
    w_ip1_at_next = torch.roll(w_ip1, shifts=1, dims=1)  # place at position i+1
    w_edge = w_i + w_ip1_at_next
    w_edge_sum = w_edge.sum(dim=1, keepdim=True)       # guards double-flag corners
    w_edge_sum = torch.where(w_edge_sum.abs() < eps,
                             torch.ones_like(w_edge_sum), w_edge_sum)
    w_edge = w_edge / w_edge_sum

    # --- Case A: query on a vertex -> one-hot -------------------------------
    on_vertex = r < eps                                # (B, K)
    any_vertex = on_vertex.any(dim=1)                  # (B,)
    w_vert = on_vertex.to(w.dtype)
    w_vert_sum = w_vert.sum(dim=1, keepdim=True)
    w_vert_sum = torch.where(w_vert_sum.abs() < eps,
                             torch.ones_like(w_vert_sum), w_vert_sum)
    w_vert = w_vert / w_vert_sum

    # priority: vertex  >  edge  >  interior
    out = torch.where(any_edge.unsqueeze(1), w_edge, w_interior)
    out = torch.where(any_vertex.unsqueeze(1), w_vert, out)
    return out


# Neural network architecture
class PositionalEncoding(nn.Module):
    """Apply sin/cos Fourier features to the input coordinates."""
    def __init__(self, d_in: int, L: int = 6):
        super().__init__()
        self.d_in = d_in
        self.L = L
        self.register_buffer('freq', 2.0 ** torch.arange(L).float() * np.pi)

    @property
    def d_out(self):
        return self.d_in * 2 * self.L

    def forward(self, x):
        parts = []
        for i in range(self.d_in):
            v = x[:, i:i + 1] * self.freq
            parts += [v.sin(), v.cos()]
        return torch.cat(parts, dim=-1)


class SkipMLP(nn.Module):
    """MLP with a midpoint skip connection and Softplus activations."""
    def __init__(self, d_in: int, d_out: int,
                 W: int = 256, D: int = 6, out_act: str = None, beta: float = 1.0):
        super().__init__()
        self.skip = D // 2
        self.lins = nn.ModuleList()
        for i in range(D):
            fan_in = d_in if i == 0 else (W + d_in if i == self.skip else W)
            self.lins.append(nn.Linear(fan_in, W))
        self.head = nn.Linear(W, d_out)
        self.act = nn.Softplus(beta=beta)
        self.out_act = out_act

    def forward(self, x0):
        h = x0
        for i, lin in enumerate(self.lins):
            if i == self.skip:
                h = torch.cat([h, x0], dim=-1)
            h = self.act(lin(h))
        h = self.head(h)
        if self.out_act == 'sigmoid':
            h = 0.5 + torch.atan(h / 3.0) / math.pi  # smooth sigmoid in [0,1]
        return h


class ForwardMap(nn.Module):
    """Map local UV coordinates to 3D points."""
    def __init__(self, L: int = 6, W: int = 256, D: int = 6, beta: float = 1.0):
        super().__init__()
        self.L = L
        if L > 0:
            self.pe = PositionalEncoding(2, L)
            d_in = self.pe.d_out
        else:
            self.pe = None
            d_in = 2
        self.net = SkipMLP(d_in, 3, W, D, beta=beta)

    def forward(self, uv):
        """Evaluate the forward map on UV samples."""
        x = self.pe(uv) if self.pe is not None else uv
        return self.net(x)


class InverseMap(nn.Module):
    """Map 3D points back to UV coordinates."""
    def __init__(self, L: int = 6, W: int = 256, D: int = 6, beta: float = 1.0):
        super().__init__()
        self.L = L
        if L > 0:
            self.pe = PositionalEncoding(3, L)
            d_in = self.pe.d_out
        else:
            self.pe = None
            d_in = 3
        self.net = SkipMLP(d_in, 2, W, D, out_act='sigmoid', beta=beta)

    def forward(self, xyz):
        """Evaluate the inverse map on 3D samples."""
        x = self.pe(xyz) if self.pe is not None else xyz
        return self.net(x)


# Feature complex and multi-patch maps.
class FeatureComplex(nn.Module):
    """
    Grid-based feature complex with shared vertex features.

    Adjacent patches share corner features, which enforces C0 continuity.

    PHASE 1 CHANGE (bilinear -> MVC)
    --------------------------------
    `interpolate` previously blended the 4 corner features BILINEARLY. It now
    blends them with MEAN VALUE COORDINATES. On the regular grid each patch is
    the local-UV unit square, so the MVC polygon is just the 4 corners in CCW
    order. MVC is a strict generalization of the bilinear-on-a-square idea that
    (a) stays continuous across shared edges, and (b) extends unchanged to the
    n-gon leaf polygons the quadtree will produce in Phase 2. Along any patch
    edge MVC reduces to LINEAR interpolation between that edge's two endpoints,
    which is exactly the property that makes hanging nodes seamless later.
    """
    def __init__(self, n_rows: int, n_cols: int, d_features: int = 64):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.d_features = d_features

        n_vertices = (n_rows + 1) * (n_cols + 1)
        self.vertex_features = nn.Parameter(
            torch.randn(n_vertices, d_features) * 0.1
        )

        # Local-UV coordinates of the 4 patch corners, in CCW order.
        # This order MUST match the feature-gather order used in interpolate():
        #   (0,0) -> z00 , (1,0) -> z10 , (1,1) -> z11 , (0,1) -> z01
        corner_uv = torch.tensor(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]], dtype=torch.float32)
        self.register_buffer('corner_uv', corner_uv)  # (4, 2)

    def _corner_indices(self, row, col):
        """
        Return vectorized indices of the four patch corners.

        Args:
            row, col: Patch grid positions.
        Returns:
            Tuple `(i00, i01, i10, i11)`.
        """
        ncp = self.n_cols + 1
        i00 = row * ncp + col
        i01 = row * ncp + (col + 1)
        i10 = (row + 1) * ncp + col
        i11 = (row + 1) * ncp + (col + 1)
        return i00, i01, i10, i11

    def interpolate(self, row, col, uv):
        """
        MVC-interpolate the shared vertex features across a patch.

        Args:
            row, col: Patch grid positions per sample (each shape (B,)).
            uv: Local UV coordinates in `[0, 1]`, shape (B, 2).
        Returns:
            Interpolated feature tensor, shape (B, d_features).
        """
        i00, i01, i10, i11 = self._corner_indices(row, col)
        vf = self.vertex_features

        # Gather corner features in the SAME CCW order as self.corner_uv:
        #   [z00, z10, z11, z01]
        z = torch.stack([vf[i00], vf[i10], vf[i11], vf[i01]], dim=1)  # (B, 4, d)

        # Broadcast the local-UV unit square to every query point.
        polys = self.corner_uv.unsqueeze(0).expand(uv.shape[0], -1, -1)  # (B,4,2)

        # Mean value coordinates of uv w.r.t. the 4 corners.
        w = mvc_weights_torch(uv, polys)               # (B, 4)

        # Weighted combination of corner features.
        features = torch.einsum('bk,bkd->bd', w, z)    # (B, d)
        return features


class TwoSheetFeatureComplex(nn.Module):
    """
    Two-sided feature complex for sphere-like reconstruction.

    Each side is a separate patch grid, but the boundary vertex features are
    shared across both sides while interior vertices remain side-specific.
    """
    def __init__(self, n_rows: int, n_cols: int, d_features: int = 64, n_sides: int = 2):
        super().__init__()
        if n_sides != 2:
            raise ValueError(f"TwoSheetFeatureComplex currently supports n_sides=2, got {n_sides}")
        if n_rows < 2 or n_cols < 2:
            raise ValueError("TwoSheetFeatureComplex expects at least a 2x2 patch grid per side")

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.d_features = d_features
        self.n_sides = n_sides

        n_boundary_vertices = 2 * (n_rows + n_cols)
        n_interior_vertices_per_side = max((n_rows - 1) * (n_cols - 1), 0)
        self.n_boundary_vertices = n_boundary_vertices
        self.n_interior_vertices_per_side = n_interior_vertices_per_side
        self.n_vertices = n_boundary_vertices + n_sides * n_interior_vertices_per_side

        self.vertex_features = nn.Parameter(
            torch.randn(self.n_vertices, d_features) * 0.1
        )

        corner_uv = torch.tensor(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]], dtype=torch.float32)
        self.register_buffer('corner_uv', corner_uv)

    def _boundary_index(self, row, col):
        """Return the shared boundary-vertex index for logical grid coordinates."""
        if row == 0:
            return col
        if col == self.n_cols:
            return self.n_cols + row
        if row == self.n_rows:
            return self.n_cols + self.n_rows + (self.n_cols - col)
        if col == 0:
            return self.n_cols + self.n_rows + self.n_cols + (self.n_rows - row)
        raise ValueError(f"Vertex ({row}, {col}) is not on the boundary")

    def _interior_index(self, side, row, col):
        """Return the side-specific interior-vertex index."""
        if not (0 <= side < self.n_sides):
            raise ValueError(f"side must be in [0, {self.n_sides - 1}], got {side}")
        if not (0 < row < self.n_rows and 0 < col < self.n_cols):
            raise ValueError(f"Vertex ({row}, {col}) is not an interior vertex")

        local_row = row - 1
        local_col = col - 1
        local_idx = local_row * (self.n_cols - 1) + local_col
        return self.n_boundary_vertices + side * self.n_interior_vertices_per_side + local_idx

    def _vertex_index(self, side, row, col):
        """Map logical side/grid coordinates to the shared learnable feature index."""
        if torch.is_tensor(side) or torch.is_tensor(row) or torch.is_tensor(col):
            side_t = torch.as_tensor(side, dtype=torch.long, device=self.vertex_features.device)
            row_t = torch.as_tensor(row, dtype=torch.long, device=self.vertex_features.device)
            col_t = torch.as_tensor(col, dtype=torch.long, device=self.vertex_features.device)

            out = torch.empty_like(row_t)
            boundary_mask = ((row_t == 0) | (row_t == self.n_rows) |
                             (col_t == 0) | (col_t == self.n_cols))

            if boundary_mask.any():
                rb = row_t[boundary_mask]
                cb = col_t[boundary_mask]
                boundary_idx = torch.empty_like(rb)

                top = rb == 0
                right = (~top) & (cb == self.n_cols)
                bottom = (~top) & (~right) & (rb == self.n_rows)
                left = (~top) & (~right) & (~bottom) & (cb == 0)

                boundary_idx[top] = cb[top]
                boundary_idx[right] = self.n_cols + rb[right]
                boundary_idx[bottom] = self.n_cols + self.n_rows + (self.n_cols - cb[bottom])
                boundary_idx[left] = self.n_cols + self.n_rows + self.n_cols + (self.n_rows - rb[left])
                out[boundary_mask] = boundary_idx

            if (~boundary_mask).any():
                ri = row_t[~boundary_mask]
                ci = col_t[~boundary_mask]
                si = side_t[~boundary_mask]
                local_idx = (ri - 1) * (self.n_cols - 1) + (ci - 1)
                out[~boundary_mask] = (
                    self.n_boundary_vertices
                    + si * self.n_interior_vertices_per_side
                    + local_idx
                )
            return out

        if row == 0 or row == self.n_rows or col == 0 or col == self.n_cols:
            return self._boundary_index(row, col)
        return self._interior_index(side, row, col)

    def _corner_indices(self, side, row, col):
        """Return the four logical corner indices for a patch on a given side."""
        i00 = self._vertex_index(side, row, col)
        i01 = self._vertex_index(side, row, col + 1)
        i10 = self._vertex_index(side, row + 1, col)
        i11 = self._vertex_index(side, row + 1, col + 1)
        return i00, i01, i10, i11

    def interpolate(self, side, row, col, uv):
        """MVC-interpolate features inside a patch on one of the two sheets."""
        i00, i01, i10, i11 = self._corner_indices(side, row, col)
        vf = self.vertex_features
        z = torch.stack([vf[i00], vf[i10], vf[i11], vf[i01]], dim=1)
        polys = self.corner_uv.unsqueeze(0).expand(uv.shape[0], -1, -1)
        w = mvc_weights_torch(uv, polys)
        return torch.einsum('bk,bkd->bd', w, z)


class _UnionFind:
    """Small union-find helper for atlas vertex equivalence classes."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


class CubeAtlasTopology:
    """Cube-face atlas topology with shared edge and corner vertices."""
    FACE_NAMES = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')
    FACE_INDEX = {name: idx for idx, name in enumerate(FACE_NAMES)}

    FACE_EMBEDDINGS = {
        '+X': lambda u, v: np.array([1.0, 1.0 - 2.0 * v, 2.0 * u - 1.0], dtype=np.float64),
        '-X': lambda u, v: np.array([-1.0, 2.0 * v - 1.0, 2.0 * u - 1.0], dtype=np.float64),
        '+Y': lambda u, v: np.array([2.0 * u - 1.0, 1.0, 2.0 * v - 1.0], dtype=np.float64),
        '-Y': lambda u, v: np.array([2.0 * u - 1.0, -1.0, 1.0 - 2.0 * v], dtype=np.float64),
        '+Z': lambda u, v: np.array([2.0 * u - 1.0, 1.0 - 2.0 * v, 1.0], dtype=np.float64),
        '-Z': lambda u, v: np.array([2.0 * u - 1.0, 2.0 * v - 1.0, -1.0], dtype=np.float64),
    }

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_faces = len(self.FACE_NAMES)
        self.local_vertices_per_face = (n_rows + 1) * (n_cols + 1)
        self.edge_glue = self._build_edge_glue()
        self.vertex_map, self.n_vertices = self._build_vertex_map()

    def _local_linear_index(self, face: int, row: int, col: int) -> int:
        return face * self.local_vertices_per_face + row * (self.n_cols + 1) + col

    def face_edge_vertices(self, face_name: str, edge_name: str):
        face = self.FACE_INDEX[face_name]
        if edge_name == 'top':
            return [(face, 0, c) for c in range(self.n_cols + 1)]
        if edge_name == 'bottom':
            return [(face, self.n_rows, c) for c in range(self.n_cols + 1)]
        if edge_name == 'left':
            return [(face, r, 0) for r in range(self.n_rows + 1)]
        if edge_name == 'right':
            return [(face, r, self.n_cols) for r in range(self.n_rows + 1)]
        raise ValueError(f"Unknown edge_name: {edge_name}")

    def _face_xyz(self, face_name: str, u: float, v: float):
        return self.FACE_EMBEDDINGS[face_name](u, v)

    def _edge_xyz_samples(self, face_name: str, edge_name: str):
        ts = np.linspace(0.0, 1.0, max(self.n_rows, self.n_cols) + 1)
        samples = []
        for t in ts:
            if edge_name == 'top':
                samples.append(self._face_xyz(face_name, t, 0.0))
            elif edge_name == 'bottom':
                samples.append(self._face_xyz(face_name, t, 1.0))
            elif edge_name == 'left':
                samples.append(self._face_xyz(face_name, 0.0, t))
            elif edge_name == 'right':
                samples.append(self._face_xyz(face_name, 1.0, t))
            else:
                raise ValueError(f"Unknown edge_name: {edge_name}")
        return np.stack(samples, axis=0)

    def _build_edge_glue(self):
        edge_glue = defaultdict(dict)
        all_edges = [(face, edge) for face in self.FACE_NAMES for edge in ('top', 'bottom', 'left', 'right')]
        used = set()

        for face_name, edge_name in all_edges:
            if (face_name, edge_name) in used:
                continue
            samples_a = self._edge_xyz_samples(face_name, edge_name)
            found = False
            for nbr_face_name, nbr_edge_name in all_edges:
                if nbr_face_name == face_name:
                    continue
                samples_b = self._edge_xyz_samples(nbr_face_name, nbr_edge_name)
                if np.allclose(samples_a, samples_b, atol=1e-6):
                    edge_glue[face_name][edge_name] = (nbr_face_name, nbr_edge_name, False)
                    edge_glue[nbr_face_name][nbr_edge_name] = (face_name, edge_name, False)
                    used.add((face_name, edge_name))
                    used.add((nbr_face_name, nbr_edge_name))
                    found = True
                    break
                if np.allclose(samples_a, samples_b[::-1], atol=1e-6):
                    edge_glue[face_name][edge_name] = (nbr_face_name, nbr_edge_name, True)
                    edge_glue[nbr_face_name][nbr_edge_name] = (face_name, edge_name, True)
                    used.add((face_name, edge_name))
                    used.add((nbr_face_name, nbr_edge_name))
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Could not find glued neighbor for cube edge {face_name}:{edge_name}")
        return edge_glue

    def _build_vertex_map(self):
        total_local = self.n_faces * self.local_vertices_per_face
        uf = _UnionFind(total_local)

        processed = set()
        for face_name, edge_map in self.edge_glue.items():
            for edge_name, (nbr_face_name, nbr_edge_name, reverse) in edge_map.items():
                key = tuple(sorted(((face_name, edge_name), (nbr_face_name, nbr_edge_name))))
                if key in processed:
                    continue
                processed.add(key)

                edge_a = self.face_edge_vertices(face_name, edge_name)
                edge_b = self.face_edge_vertices(nbr_face_name, nbr_edge_name)
                if reverse:
                    edge_b = list(reversed(edge_b))

                if len(edge_a) != len(edge_b):
                    raise ValueError(
                        f"Mismatched edge lengths for {face_name}:{edge_name} and {nbr_face_name}:{nbr_edge_name}"
                    )

                for (fa, ra, ca), (fb, rb, cb) in zip(edge_a, edge_b):
                    uf.union(
                        self._local_linear_index(fa, ra, ca),
                        self._local_linear_index(fb, rb, cb),
                    )

        root_to_global = {}
        vertex_map = {}
        next_idx = 0
        for face in range(self.n_faces):
            for row in range(self.n_rows + 1):
                for col in range(self.n_cols + 1):
                    local_idx = self._local_linear_index(face, row, col)
                    root = uf.find(local_idx)
                    if root not in root_to_global:
                        root_to_global[root] = next_idx
                        next_idx += 1
                    vertex_map[(face, row, col)] = root_to_global[root]
        return vertex_map, next_idx


class SixSheetFeatureComplex(nn.Module):
    """Six-face cube atlas with shared edge/corner vertex features."""
    def __init__(self, n_rows: int, n_cols: int, d_features: int = 64):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.d_features = d_features
        self.n_faces = 6
        self.topology = CubeAtlasTopology(n_rows, n_cols)
        self.n_vertices = self.topology.n_vertices

        self.vertex_features = nn.Parameter(
            torch.randn(self.n_vertices, d_features) * 0.1
        )

        corner_uv = torch.tensor(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]], dtype=torch.float32)
        self.register_buffer('corner_uv', corner_uv)

    def _vertex_index(self, face, row, col):
        if torch.is_tensor(face) or torch.is_tensor(row) or torch.is_tensor(col):
            face_t = torch.as_tensor(face, dtype=torch.long, device=self.vertex_features.device)
            row_t = torch.as_tensor(row, dtype=torch.long, device=self.vertex_features.device)
            col_t = torch.as_tensor(col, dtype=torch.long, device=self.vertex_features.device)
            out = torch.empty_like(face_t)
            flat_face = face_t.reshape(-1).tolist()
            flat_row = row_t.reshape(-1).tolist()
            flat_col = col_t.reshape(-1).tolist()
            mapped = [self.topology.vertex_map[(int(f), int(r), int(c))] for f, r, c in zip(flat_face, flat_row, flat_col)]
            out = torch.tensor(mapped, dtype=torch.long, device=self.vertex_features.device).reshape(face_t.shape)
            return out
        return self.topology.vertex_map[(int(face), int(row), int(col))]

    def _corner_indices(self, face, row, col):
        i00 = self._vertex_index(face, row, col)
        i01 = self._vertex_index(face, row, col + 1)
        i10 = self._vertex_index(face, row + 1, col)
        i11 = self._vertex_index(face, row + 1, col + 1)
        return i00, i01, i10, i11

    def interpolate(self, face, row, col, uv):
        i00, i01, i10, i11 = self._corner_indices(face, row, col)
        vf = self.vertex_features
        z = torch.stack([vf[i00], vf[i10], vf[i11], vf[i01]], dim=1)
        polys = self.corner_uv.unsqueeze(0).expand(uv.shape[0], -1, -1)
        w = mvc_weights_torch(uv, polys)
        return torch.einsum('bk,bkd->bd', w, z)


class MultiPatchForwardMap(nn.Module):
    """
        Vectorized multi-patch forward map.

        Shared features and global UV coordinates keep neighboring patches
        continuous across boundaries.
    """
    def __init__(self, n_rows: int, n_cols: int, d_features: int = 64,
                 L: int = 8, W: int = 256, D: int = 6, beta: float = 5.0):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_patches = n_rows * n_cols
        self.d_features = d_features
        self.L = L

        self.complex = FeatureComplex(n_rows, n_cols, d_features)

        if L > 0:
            self.pe = PositionalEncoding(2, L)
            d_pe = self.pe.d_out
        else:
            self.pe = None
            d_pe = 0

        self.decoder = SkipMLP(d_features + d_pe, 3, W, D, beta=beta)

        # Optional flat-plane initialization.
        # nn.init.zeros_(self.decoder.head.weight)
        # nn.init.zeros_(self.decoder.head.bias)

    def patch_idx_to_rowcol(self, patch_idx):
        """Convert patch indices to row and column indices."""
        row = patch_idx // self.n_cols
        col = patch_idx % self.n_cols
        return row, col

    def forward(self, patch_idx, uv: torch.Tensor):
        """
        Args:
            patch_idx: Patch index or batch of patch indices.
            uv: Local UV coordinates.
        Returns:
            Predicted 3D points.
        """
        B = uv.shape[0]
        if not torch.is_tensor(patch_idx):
            patch_idx = torch.full((B,), int(patch_idx),
                                   dtype=torch.long, device=uv.device)

        row = patch_idx // self.n_cols
        col = patch_idx % self.n_cols

        features = self.complex.interpolate(row, col, uv)

        u = uv[:, 0:1]
        v = uv[:, 1:2]
        # Convert local UV to global UV over the full patch grid.
        global_u = (row.unsqueeze(1).float() + u) / self.n_rows
        global_v = (col.unsqueeze(1).float() + v) / self.n_cols

        if self.pe is not None:
            pe = self.pe(torch.cat([global_u, global_v], dim=1))
            dec_in = torch.cat([features, pe], dim=1)
        else:
            dec_in = features

        # Optional explicit flat-plane embedding.
        # flat = torch.cat([
        #     2 * global_u - 1,
        #     2 * global_v - 1,
        #     torch.zeros_like(global_u)
        # ], dim=1)

        correction = self.decoder(dec_in)

        return correction


class TwoSheetForwardMap(nn.Module):
    """
    Two-sheet forward map with shared boundary features and one shared decoder.

    To keep both sheets on the same front-facing orientation convention, the
    second sheet uses a flipped local-u coordinate. This reverses the chart
    orientation for side 1 before interpolation/decoding, which flips the
    induced normal direction and helps align both sheets face orientation.
    """
    def __init__(self, n_rows: int, n_cols: int, d_features: int = 64,
                 L: int = 8, W: int = 256, D: int = 6, beta: float = 5.0,
                 n_sides: int = 2):
        super().__init__()
        if n_sides != 2:
            raise ValueError(f"TwoSheetForwardMap currently supports n_sides=2, got {n_sides}")

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_sides = n_sides
        self.patches_per_side = n_rows * n_cols
        self.n_patches = self.patches_per_side * n_sides
        self.d_features = d_features
        self.L = L

        self.complex = TwoSheetFeatureComplex(
            n_rows=n_rows,
            n_cols=n_cols,
            d_features=d_features,
            n_sides=n_sides,
        )

        if L > 0:
            self.pe = PositionalEncoding(2, L)
            d_pe = self.pe.d_out
        else:
            self.pe = None
            d_pe = 0

        self.decoder = SkipMLP(d_features + d_pe, 3, W, D, beta=beta)

    def patch_idx_to_side_rowcol(self, patch_idx):
        """Convert flattened patch indices to side, row, and column."""
        side = patch_idx // self.patches_per_side
        local_patch_idx = patch_idx % self.patches_per_side
        row = local_patch_idx // self.n_cols
        col = local_patch_idx % self.n_cols
        return side, row, col

    def forward(self, patch_idx, uv: torch.Tensor):
        """Evaluate the two-sheet forward map on local UV samples."""
        B = uv.shape[0]
        if not torch.is_tensor(patch_idx):
            patch_idx = torch.full((B,), int(patch_idx), dtype=torch.long, device=uv.device)
        else:
            patch_idx = patch_idx.to(device=uv.device, dtype=torch.long)

        side, row, col = self.patch_idx_to_side_rowcol(patch_idx)

        uv_oriented = uv.clone()
        side1_mask = side == 1
        if side1_mask.any():
            uv_oriented[side1_mask, 0] = 1.0 - uv_oriented[side1_mask, 0]

        features = self.complex.interpolate(side, row, col, uv_oriented)

        u = uv_oriented[:, 0:1]
        v = uv_oriented[:, 1:2]
        global_u = (row.unsqueeze(1).float() + u) / self.n_rows
        global_v = (col.unsqueeze(1).float() + v) / self.n_cols

        dec_parts = [features]
        if self.pe is not None:
            dec_parts.append(self.pe(torch.cat([global_u, global_v], dim=1)))

        dec_in = torch.cat(dec_parts, dim=1)
        return self.decoder(dec_in)


class SixSheetForwardMap(nn.Module):
    """Six-face cube atlas forward map with shared edge/corner features."""
    def __init__(self, n_rows: int, n_cols: int, d_features: int = 64,
                 L: int = 8, W: int = 256, D: int = 6, beta: float = 5.0,
                 n_faces: int = 6):
        super().__init__()
        if n_faces != 6:
            raise ValueError(f"SixSheetForwardMap currently supports n_faces=6, got {n_faces}")

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_sides = n_faces
        self.n_faces = n_faces
        self.patches_per_side = n_rows * n_cols
        self.n_patches = self.patches_per_side * n_faces
        self.d_features = d_features
        self.L = L

        self.complex = SixSheetFeatureComplex(n_rows=n_rows, n_cols=n_cols, d_features=d_features)

        if L > 0:
            self.pe = PositionalEncoding(2, L)
            d_pe = self.pe.d_out
        else:
            self.pe = None
            d_pe = 0

        self.decoder = SkipMLP(d_features + d_pe, 3, W, D, beta=beta)
        self.register_buffer('face_uv_transforms', torch.tensor([
            [0, 0, 0],  # +X: identity
            [0, 1, 0],  # -X: flip u
            [1, 0, 0],  # +Y: swap
            [1, 1, 0],  # -Y: swap + flip u
            [0, 0, 0],  # +Z: identity
            [0, 1, 1],  # -Z: flip u and v
        ], dtype=torch.long))
        self.register_buffer('face_orientation_sign', torch.tensor([
            -1,   # +X
            1,   # -X
            -1,  # +Y
            1,  # -Y
            1,   # +Z
            1,   # -Z
        ], dtype=torch.long))

    def patch_idx_to_face_rowcol(self, patch_idx):
        face = patch_idx // self.patches_per_side
        local_patch_idx = patch_idx % self.patches_per_side
        row = local_patch_idx // self.n_cols
        col = local_patch_idx % self.n_cols
        return face, row, col

    def forward(self, patch_idx, uv: torch.Tensor):
        B = uv.shape[0]
        if not torch.is_tensor(patch_idx):
            patch_idx = torch.full((B,), int(patch_idx), dtype=torch.long, device=uv.device)
        else:
            patch_idx = patch_idx.to(device=uv.device, dtype=torch.long)

        face, row, col = self.patch_idx_to_face_rowcol(patch_idx)
        uv_oriented = uv.clone()
        transforms = self.face_uv_transforms[face]
        swap_mask = transforms[:, 0] == 1
        flip_u_mask = transforms[:, 1] == 1
        flip_v_mask = transforms[:, 2] == 1

        if swap_mask.any():
            uv_oriented[swap_mask] = uv_oriented[swap_mask][:, [1, 0]]
        if flip_u_mask.any():
            uv_oriented[flip_u_mask, 0] = 1.0 - uv_oriented[flip_u_mask, 0]
        if flip_v_mask.any():
            uv_oriented[flip_v_mask, 1] = 1.0 - uv_oriented[flip_v_mask, 1]

        orientation_sign = self.face_orientation_sign[face]
        flipped_faces = orientation_sign < 0
        if flipped_faces.any():
            uv_oriented[flipped_faces, 0] = 1.0 - uv_oriented[flipped_faces, 0]

        features = self.complex.interpolate(face, row, col, uv_oriented)

        u = uv_oriented[:, 0:1]
        v = uv_oriented[:, 1:2]
        global_u = (row.unsqueeze(1).float() + u) / self.n_rows
        global_v = (col.unsqueeze(1).float() + v) / self.n_cols

        dec_parts = [features]
        if self.pe is not None:
            dec_parts.append(self.pe(torch.cat([global_u, global_v], dim=1)))

        dec_in = torch.cat(dec_parts, dim=1)
        return self.decoder(dec_in)


class MultiPatchInverseMap(nn.Module):
    """
    Multi-patch inverse map from 3D points to local UV coordinates.

    NOTE (Phase 1): `de_interpolate` analytically inverts BILINEAR interpolation
    and therefore no longer matches the forward map now that F uses MVC. Per the
    project decision this inverse map is a deprecated experiment and is NOT used
    by the training loop, so it is left untouched. Do not rely on cycle
    consistency through G while MVC is active.
    """
    def __init__(self, feature_complex: FeatureComplex, d_features: int = 64,
                 L: int = 0, W: int = 256, D: int = 6, beta: float = 5.0):
        super().__init__()
        self.d_features = d_features
        self.L = L

        # Keep a plain reference to avoid duplicate optimizer parameters.
        self._fc_ref = [feature_complex]

        if L > 0:
            self.pe = PositionalEncoding(3, L)
            d_in = self.pe.d_out
        else:
            self.pe = None
            d_in = 3

        self.encoder = SkipMLP(d_in, d_features, W, D, beta=beta)

    @property
    def feature_complex(self) -> FeatureComplex:
        return self._fc_ref[0]

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        """Encode 3D points into feature space."""
        x = self.pe(xyz) if self.pe is not None else xyz
        return self.encoder(x)

    def de_interpolate(self, patch_idx, z_pred: torch.Tensor) -> torch.Tensor:
        """
        Recover UV coordinates from predicted features (bilinear inverse).

        DEPRECATED under MVC — see class docstring.
        """
        B = z_pred.shape[0]
        if not torch.is_tensor(patch_idx):
            patch_idx = torch.full((B,), int(patch_idx),
                                   dtype=torch.long, device=z_pred.device)

        fc = self.feature_complex
        row = patch_idx // fc.n_cols
        col = patch_idx % fc.n_cols
        i00, i01, i10, i11 = fc._corner_indices(row, col)

        vf = fc.vertex_features
        z00 = vf[i00]
        z01 = vf[i01]
        z10 = vf[i10]
        z11 = vf[i11]

        A = z01 - z00
        Bd = z10 - z00
        C = z11 - z10 - z01 + z00
        R = z_pred - z00

        # Per-sample basis matrix M = [B | A].
        M_mat = torch.stack([Bd, A], dim=2)
        Mt = M_mat.transpose(1, 2)
        MtM = Mt @ M_mat
        reg = 1e-5 * torch.eye(2, device=MtM.device, dtype=MtM.dtype)
        MtM = MtM + reg

        # Initial solve without the bilinear cross term.
        rhs = Mt @ R.unsqueeze(-1)
        params = torch.linalg.solve(MtM, rhs)
        u_est = params[:, 0]
        v_est = params[:, 1]

        # One refinement step with the estimated cross term.
        R_ref = R - (u_est * v_est) * C
        rhs2 = Mt @ R_ref.unsqueeze(-1)
        params2 = torch.linalg.solve(MtM, rhs2)
        u = params2[:, 0]
        v = params2[:, 1]

        return torch.cat([u, v], dim=1)

    def forward(self, patch_idx, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_idx: Patch index or batch of patch indices.
            xyz: 3D points.
        Returns:
            UV coordinates.
        """
        z_pred = self.encode(xyz)
        return self.de_interpolate(patch_idx, z_pred)