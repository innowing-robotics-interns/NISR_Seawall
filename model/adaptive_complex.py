#!/usr/bin/env python3
# model/adaptive_complex.py
"""
Adaptive cube-atlas feature complex (quadtree with hanging/T-vertices).

Data model
----------
* A VERTEX is abstract: an integer id + one learnable row of the (V, d)
  feature matrix. Vertices have NO coordinates of any kind.
* A PATCH is a quadtree cell on one of the 6 cube-face charts. Its record is
  pure bookkeeping: (face, dyadic rectangle in that face's [0,1]^2 domain,
  4 corner vertex ids). Hanging vertices on its edges are DISCOVERED at
  query time from the global key map, never stored redundantly.
* Gluing is done with exact integer keys: every vertex is created at a dyadic
  parametric position, which maps to an exact integer point of the reference
  cube. Touching charts produce the same key -> same vertex id -> same
  feature row. This is what makes seams (including cross-face seams and
  T-junctions) share features automatically.

Continuity: along any patch edge the MVC interpolant reduces to piecewise
linear interpolation between the ordered vertices on that edge. Both sides
of a seam see the same ordered vertex ids, hence the same feature curve,
hence (through the shared decoder) the same 3D curve.

Subdivision inserts new vertices whose features equal the parent patch's own
MVC value at that parametric point, so the represented surface does not move
at the moment of subdivision — it only gains capacity.
"""

import torch
import torch.nn as nn

from .model import mvc_weights_torch

MAX_LEVEL = 20                 # dyadic resolution of the parametric domain
S_INT = 1 << MAX_LEVEL         # integer size of a root face domain

FACE_NAMES = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')

# xyz = A @ (u, v, 1) with u, v in [0,1]  (rows: x, y, z).
# The same integer matrix applied to (ui, vi, S_INT) gives EXACT integer keys.
_FACE_AFFINE = [
    [[0, 0,  1], [0, -2,  1], [2, 0, -1]],   # +X: ( 1,   1-2v, 2u-1)
    [[0, 0, -1], [0,  2, -1], [2, 0, -1]],   # -X: (-1,   2v-1, 2u-1)
    [[2, 0, -1], [0,  0,  1], [0, -2, 1]],   # +Y: (2u-1,  1,   1-2v)
    [[2, 0, -1], [0,  0, -1], [0, 2, -1]],   # -Y: (2u-1, -1,   2v-1)
    [[2, 0, -1], [0, 2, -1], [0, 0,  1]],    # +Z: (2u-1, 2v-1,  1  )
    [[2, 0, -1], [0, -2,  1], [0, 0, -1]],   # -Z: (2u-1, 1-2v, -1  )
]
    

class _Patch:
    """Quadtree cell. Leaf iff children is None."""
    __slots__ = ('face', 'u0', 'v0', 'size', 'depth', 'corner_ids', 'children')

    def __init__(self, face, u0, v0, size, depth, corner_ids):
        self.face = face
        self.u0 = u0
        self.v0 = v0
        self.size = size
        self.depth = depth
        self.corner_ids = list(corner_ids)   # CCW: (0,0),(1,0),(1,1),(0,1)
        self.children = None

    @property
    def is_leaf(self):
        return self.children is None

    def rect_key(self):
        return (self.face, self.u0, self.v0, self.size)


class AdaptiveCubeComplex(nn.Module):
    def __init__(self, d_features: int = 88, base_subdivisions: int = 0,
                 init_scale: float = 0.1):
        super().__init__()
        self.d_features = d_features
        self.init_scale = init_scale

        self._key_to_id = {}
        self.vertex_keys = []          # id -> integer cube key
        self.patches = []              # creation order (deterministic)
        self._rect_to_patch = {}
        self.subdiv_log = []           # replayable list of (face,u0,v0,size)

        self.vertex_features = nn.Parameter(torch.zeros(0, d_features))
        self.register_buffer('face_affine',
                             torch.tensor(_FACE_AFFINE, dtype=torch.float32))

        # 6 root patches; corner keys automatically merge into the 8 cube
        # corners (this is exactly the NPS box: 6 patches, 8 shared vertices).
        for face in range(6):
            pts = [(0, 0), (S_INT, 0), (S_INT, S_INT), (0, S_INT)]
            ids = [self._get_or_create_vertex(face, pt) for pt in pts]
            self._add_patch(face, 0, 0, S_INT, 0, ids)

        for _ in range(max(0, base_subdivisions)):
            for p in [q for q in self.patches if q.is_leaf]:
                self.subdivide(p)

        self.rebuild_flat()

    # ── vertex bookkeeping ──────────────────────────────────────────────
    def _xyz_key(self, face, ui, vi):
        A = _FACE_AFFINE[face]
        return tuple(A[r][0] * ui + A[r][1] * vi + A[r][2] * S_INT
                     for r in range(3))

    def _append_row(self, row: torch.Tensor) -> int:
        vid = self.vertex_features.shape[0]
        with torch.no_grad():
            row = row.reshape(1, self.d_features).to(
                dtype=self.vertex_features.dtype,
                device=self.vertex_features.device)
            data = torch.cat([self.vertex_features.data, row], dim=0)
        # NOTE: replaces the Parameter object. Any optimizer must be rebuilt
        # by the caller after a subdivision round.
        self.vertex_features = nn.Parameter(data)
        return vid

    def _create_vertex(self, key, feat_row) -> int:
        vid = self._append_row(feat_row)
        self._key_to_id[key] = vid
        self.vertex_keys.append(key)
        return vid

    def _get_or_create_vertex(self, face, pt, feat_row=None) -> int:
        key = self._xyz_key(face, pt[0], pt[1])
        vid = self._key_to_id.get(key)
        if vid is not None:
            return vid
        if feat_row is None:
            feat_row = torch.randn(1, self.d_features) * self.init_scale
        return self._create_vertex(key, feat_row)

    # ── patch bookkeeping ───────────────────────────────────────────────
    def _add_patch(self, face, u0, v0, size, depth, corner_ids) -> _Patch:
        p = _Patch(face, u0, v0, size, depth, corner_ids)
        self.patches.append(p)
        self._rect_to_patch[p.rect_key()] = p
        return p

    def _local_uv(self, p, pt):
        return ((pt[0] - p.u0) / p.size, (pt[1] - p.v0) / p.size)

    def _collect_hanging(self, p, a, b, ids, uvs):
        """Recursively collect existing vertices strictly between a and b."""
        su, sv = a[0] + b[0], a[1] + b[1]
        if (su & 1) or (sv & 1):
            return
        m = (su >> 1, sv >> 1)
        vid = self._key_to_id.get(self._xyz_key(p.face, m[0], m[1]))
        if vid is None:
            return   # quadtree property: no midpoint => nothing finer inside
        self._collect_hanging(p, a, m, ids, uvs)
        ids.append(vid)
        uvs.append(self._local_uv(p, m))
        self._collect_hanging(p, m, b, ids, uvs)

    def _polygon(self, p):
        """Ordered CCW polygon of a leaf: corners + hanging vertices."""
        s = p.size
        c = [(p.u0, p.v0), (p.u0 + s, p.v0),
             (p.u0 + s, p.v0 + s), (p.u0, p.v0 + s)]
        ids, uvs = [], []
        for i in range(4):
            ids.append(p.corner_ids[i])
            uvs.append(self._local_uv(p, c[i]))
            self._collect_hanging(p, c[i], c[(i + 1) % 4], ids, uvs)
        return ids, uvs

    def validate_leaf_polygons(self, atol: float = 1e-8):
        """Return diagnostics for leaf MVC polygons after subdivision.

        Checks:
        - duplicate non-consecutive vertices
        - signed area <= 0
        - vertices outside local [0,1]^2
        - edge vertices not lying on the expected boundary edge
        - non-monotone ordering along each boundary edge
        """
        reports = []
        for leaf_idx, p in enumerate([q for q in self.patches if q.is_leaf]):
            ids, uvs = self._polygon(p)
            uv_t = torch.tensor(uvs, dtype=torch.float64)

            issues = []

            # signed area
            x = uv_t[:, 0]
            y = uv_t[:, 1]
            area2 = torch.sum(x * torch.roll(y, -1) - torch.roll(x, -1) * y).item()
            if area2 <= atol:
                issues.append(f"non_positive_signed_area:{area2:.3e}")

            # bounds
            if ((uv_t < -atol) | (uv_t > 1.0 + atol)).any():
                issues.append("vertex_out_of_local_bounds")

            # duplicate non-consecutive vertices
            seen = {}
            n = len(uvs)
            for i, uv in enumerate(uvs):
                key = (round(float(uv[0]), 12), round(float(uv[1]), 12))
                if key in seen:
                    j = seen[key]
                    if not (abs(i - j) == 1 or {i, j} == {0, n - 1}):
                        issues.append(f"duplicate_nonconsecutive_vertex:{j}->{i}:{key}")
                else:
                    seen[key] = i

            # per-edge checks against expected square boundary walk
            edge_specs = [
                (0.0, 'u', 1.0),  # bottom: v=0, u increasing
                (1.0, 'v', 1.0),  # right:  u=1, v increasing
                (1.0, 'u', -1.0), # top:    v=1, u decreasing
                (0.0, 'v', -1.0), # left:   u=0, v decreasing
            ]
            cursor = 0
            for edge_idx in range(4):
                start = cursor
                end = start + 1
                while end < len(uvs):
                    if end < len(uvs) - 1 and ids[end] == p.corner_ids[(edge_idx + 1) % 4]:
                        break
                    end += 1
                segment = uvs[start:end + 1]
                const_val, axis, direction = edge_specs[edge_idx]

                vals = []
                for uv in segment:
                    u, v = float(uv[0]), float(uv[1])
                    if edge_idx == 0 and abs(v - 0.0) > atol:
                        issues.append(f"edge{edge_idx}_off_boundary:{uv}")
                    elif edge_idx == 1 and abs(u - 1.0) > atol:
                        issues.append(f"edge{edge_idx}_off_boundary:{uv}")
                    elif edge_idx == 2 and abs(v - 1.0) > atol:
                        issues.append(f"edge{edge_idx}_off_boundary:{uv}")
                    elif edge_idx == 3 and abs(u - 0.0) > atol:
                        issues.append(f"edge{edge_idx}_off_boundary:{uv}")
                    vals.append(u if axis == 'u' else v)

                diffs = [direction * (vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
                if any(d < -atol for d in diffs):
                    issues.append(f"edge{edge_idx}_nonmonotone:{vals}")
                cursor = end

            if issues:
                reports.append({
                    'leaf_idx': leaf_idx,
                    'face': int(p.face),
                    'depth': int(p.depth),
                    'rect': (int(p.u0), int(p.v0), int(p.size)),
                    'corner_ids': list(map(int, p.corner_ids)),
                    'polygon_ids': list(map(int, ids)),
                    'polygon_uv': [tuple(map(float, uv)) for uv in uvs],
                    'issues': issues,
                })
        return reports

    # ── subdivision ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _interp_feature_at(self, p, local_uv):
        """Current MVC feature of patch p at a local uv (surface-preserving
        initialization for new vertices)."""
        ids, uvs = self._polygon(p)
        dev = self.vertex_features.device
        z = self.vertex_features.data[torch.tensor(ids, dtype=torch.long, device=dev)]
        poly = torch.tensor(uvs, dtype=torch.float32, device=dev).unsqueeze(0)
        q = torch.tensor([local_uv], dtype=torch.float32, device=dev)
        w = mvc_weights_torch(q, poly)          # (1, K)
        return w @ z                            # (1, d)

    def subdivide(self, p: _Patch):
        """Split a leaf into 4 children. New vertices (edge midpoints +
        center) reuse existing ids when a finer neighbor already created
        them; otherwise they are initialized to the parent's own boundary/
        interior value so the surface is unchanged at insertion time.
        Callers must rebuild_flat() (and their optimizer) afterwards."""
        if not p.is_leaf:
            raise ValueError("subdivide() called on a non-leaf patch")
        if p.size < 2:
            raise ValueError(f"patch at max dyadic resolution (level {MAX_LEVEL})")

        u0, v0, s = p.u0, p.v0, p.size
        h = s // 2
        targets = {
            'mB': (u0 + h, v0), 'mR': (u0 + s, v0 + h),
            'mT': (u0 + h, v0 + s), 'mL': (u0, v0 + h),
            'ct': (u0 + h, v0 + h),
        }

        # Compute init features for MISSING vertices first (from the parent's
        # polygon as it exists right now), then insert.
        missing = {}
        for name, pt in targets.items():
            key = self._xyz_key(p.face, pt[0], pt[1])
            if key not in self._key_to_id:
                missing[name] = (key, self._interp_feature_at(p, self._local_uv(p, pt)))
        new_ids = {}
        for name, pt in targets.items():
            key = self._xyz_key(p.face, pt[0], pt[1])
            if name in missing:
                new_ids[name] = self._create_vertex(*missing[name])
            else:
                new_ids[name] = self._key_to_id[key]

        c0, c1, c2, c3 = p.corner_ids
        mB, mR, mT, mL, ct = (new_ids['mB'], new_ids['mR'], new_ids['mT'],
                              new_ids['mL'], new_ids['ct'])
        d = p.depth + 1
        p.children = [
            self._add_patch(p.face, u0,     v0,     h, d, [c0, mB, ct, mL]),
            self._add_patch(p.face, u0 + h, v0,     h, d, [mB, c1, mR, ct]),
            self._add_patch(p.face, u0 + h, v0 + h, h, d, [ct, mR, c2, mT]),
            self._add_patch(p.face, u0,     v0 + h, h, d, [mL, ct, mT, c3]),
        ]
        self.subdiv_log.append(p.rect_key())

    def subdivide_patches(self, patch_list):
        for p in patch_list:
            self.subdivide(p)
        self.rebuild_flat()

    # ── flat query tensors ──────────────────────────────────────────────
    def rebuild_flat(self):
        device = self.vertex_features.device
        self.leaf_patches = [p for p in self.patches if p.is_leaf]
        polys = [self._polygon(p) for p in self.leaf_patches]
        k_max = max(len(ids) for ids, _ in polys)
        vid_rows, uv_rows = [], []
        for ids, uvs in polys:
            pad = k_max - len(ids)
            # Padding by repeating the last vertex is EXACT under MVC:
            # zero-length edges get zero half-angle tangents, and duplicated
            # positions gather the same feature row.
            vid_rows.append(ids + [ids[-1]] * pad)
            uv_rows.append(uvs + [uvs[-1]] * pad)
        self.leaf_poly_vid = torch.tensor(vid_rows, dtype=torch.long, device=device)
        self.leaf_poly_uv = torch.tensor(uv_rows, dtype=torch.float32, device=device)
        self.leaf_face = torch.tensor([p.face for p in self.leaf_patches],
                                      dtype=torch.long, device=device)
        self.leaf_rect = torch.tensor(
            [[p.u0 / S_INT, p.v0 / S_INT, p.size / S_INT] for p in self.leaf_patches],
            dtype=torch.float32, device=device)

    def _sync_device(self):
        if self.leaf_poly_vid.device != self.vertex_features.device:
            self.rebuild_flat()

    @property
    def n_leaves(self):
        return len(self.leaf_patches)

    @property
    def n_vertices(self):
        return self.vertex_features.shape[0]

    # ── queries ─────────────────────────────────────────────────────────
    def interpolate(self, leaf_idx: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """MVC-interpolate vertex features. leaf_idx: (B,) long, uv: (B,2)."""
        self._sync_device()
        polys = self.leaf_poly_uv[leaf_idx]           # (B, K, 2)
        vids = self.leaf_poly_vid[leaf_idx]           # (B, K)
        z = self.vertex_features[vids]                # (B, K, d)
        w = mvc_weights_torch(uv, polys)              # (B, K)
        return torch.einsum('bk,bkd->bd', w, z)

    def face_uv(self, leaf_idx, uv):
        self._sync_device()
        rect = self.leaf_rect[leaf_idx]
        return rect[:, :2] + uv * rect[:, 2:3]

    def cube_xyz(self, leaf_idx, uv):
        """Reference-cube embedding of a query — used only for PE
        conditioning, box pretraining targets and seam diagnostics."""
        self._sync_device()
        fuv = self.face_uv(leaf_idx, uv)
        ones = torch.ones_like(fuv[:, :1])
        homog = torch.cat([fuv, ones], dim=1).unsqueeze(-1)     # (B,3,1)
        A = self.face_affine[self.leaf_face[leaf_idx]]          # (B,3,3)
        return torch.bmm(A, homog).squeeze(-1)                  # (B,3)

    # ── serialization ───────────────────────────────────────────────────
    def serialize(self):
        return {
            'd_features': self.d_features,
            'subdiv_log': list(self.subdiv_log),
            'n_vertices': self.n_vertices,
            'vertex_keys': list(self.vertex_keys),
        }

    def replay(self, topo):
        """Rebuild topology on a FRESH complex (base_subdivisions=0)."""
        if self.subdiv_log:
            raise RuntimeError("replay() requires a fresh complex")
        for rect in topo['subdiv_log']:
            self.subdivide(self._rect_to_patch[tuple(rect)])
        self.rebuild_flat()
        if self.n_vertices != topo['n_vertices']:
            raise RuntimeError("replay vertex-count mismatch")
        if list(self.vertex_keys) != [tuple(k) for k in topo['vertex_keys']]:
            raise RuntimeError("replay vertex-key mismatch")

    def stats(self):
        depths = [p.depth for p in self.leaf_patches]
        hist = {}
        for d in depths:
            hist[d] = hist.get(d, 0) + 1
        return {'n_leaves': self.n_leaves, 'n_vertices': self.n_vertices,
                'depth_hist': hist, 'k_max': int(self.leaf_poly_vid.shape[1])}