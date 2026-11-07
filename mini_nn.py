## this file is only for testing the neural network fitting

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneField(nn.Module):
    """A continuous 3D feature read from three orthogonal 2D grids.

    The three planes are indexed XY, XZ, YZ. A query point is projected onto
    each, bilinearly sampled, and the three samples are concatenated, so the
    feature is a function of position rather than a row looked up by loop id.
    Two loops close in space therefore start from similar features even if the
    network has never fitted either of them; the grid is what makes the feature
    spatial and the bilinear interpolation is what makes it continuous across a
    cell boundary.

    The grids live on the unit cube, `[-UNIT_CUBE_HALF, UNIT_CUBE_HALF]^3`,
    subdivided uniformly and the same for every shape. `centre` and `half` say
    which world box that cube is laid over -- the cloud's own box when fitting, so
    the shape fills the grid -- and a query is carried into the cube by
    (x - centre) / half, then clamped at the border. That map is affine and the
    field is only a feature map, so stretching its domain changes where a cell sits
    in the world, not what the field is: a grid learned on one shape reads the same
    way on another once the frame is known. The frame is stored with the grid, so a
    checkpoint is queried in world coordinates.

    The grids start at zero, so a cell the warm-up never reads stays the zero
    vector; every query reading only such cells shares one condition. The warm-up
    is what puts structure on the cells it reads.
    """

    def __init__(self, centre=None, half=None, resolution=32, channels=16, tv=0.0,
                 lo=None, hi=None):
        super().__init__()
        ## `lo`/`hi` are the older spelling of the same frame, kept so a checkpoint
        ## written before the cube became explicit still loads
        if centre is None or half is None:
            if lo is None or hi is None:
                raise ValueError('a triplane field needs the world box it is mapped '
                                 'onto, as centre and half (or lo and hi)')
            lo = torch.as_tensor(lo, dtype=torch.float32).reshape(3)
            hi = torch.as_tensor(hi, dtype=torch.float32).reshape(3)
            if torch.any(hi <= lo):
                raise ValueError(f'degenerate triplane box {lo.tolist()} .. '
                                 f'{hi.tolist()}')
            centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        centre = torch.as_tensor(centre, dtype=torch.float32).reshape(3)
        half = torch.as_tensor(half, dtype=torch.float32).reshape(3)
        if torch.any(half <= 0):
            raise ValueError(f'triplane half-extents must be positive, got '
                             f'{half.tolist()}')
        ## `box_` not `centre`/`half`: `half` is a method on nn.Module
        self.register_buffer('box_centre', centre)
        self.register_buffer('box_half', half)
        self.resolution = int(resolution)
        self.channels = int(channels)
        ## (plane, channel, row, col); XY samples (x, y), XZ (x, z), YZ (y, z).
        ## Zero, so every loop starts from the same curve until the warm-up or the
        ## fit moves the cells it reads
        self.grid = nn.Parameter(torch.zeros(3, self.channels, self.resolution,
                                             self.resolution))
        self.tv = float(tv)
        self.pairs = ((0, 1), (0, 2), (1, 2))

    @property
    def feat_dim(self):
        return 3 * self.channels

    @property
    def lo(self):
        """World corner of the box the cube is mapped onto."""
        return self.box_centre - self.box_half

    @property
    def hi(self):
        """World corner of the box the cube is mapped onto."""
        return self.box_centre + self.box_half

    def sample(self, xyz):
        """B x 3 world points -> B x (3 * channels) features."""
        xyz = xyz.reshape(-1, 3)
        ## the world box onto the unit cube's [-1, 1], and held there so a query
        ## outside the box reads the border rather than an extrapolated value
        p = (xyz - self.box_centre) / self.box_half
        p = p.clamp(-1.0, 1.0)
        out = []
        for plane, (a, b) in enumerate(self.pairs):
            ## grid_sample takes x first (width) and y second (height)
            g = torch.stack([p[:, a], p[:, b]], dim=-1).view(1, -1, 1, 2)
            s = F.grid_sample(self.grid[plane:plane + 1], g, mode='bilinear',
                              padding_mode='border', align_corners=True)
            out.append(s.reshape(self.channels, -1).T)
        return torch.cat(out, dim=-1)

    def forward(self, xyz):
        return self.sample(xyz)

    def tv_penalty(self):
        """Mean squared difference between neighbouring cells, both directions."""
        g = self.grid
        if g.shape[-1] < 2:
            return g.new_zeros(())
        dx = (g[..., 1:, :] - g[..., :-1, :]).pow(2).mean()
        dy = (g[..., :, 1:] - g[..., :, :-1]).pow(2).mean()
        return 0.5 * (dx + dy)


## define the neural network
class miniNN(nn.Module):
    def __init__(self, num_models, num_neurons=64, feat_dim=32, pe=True,
                 feature_mode='embedding', triplane=None, num_layers=2,
                 axis_condition=False):
        super(miniNN, self).__init__()

        self.feature_mode = feature_mode
        self.num_layers = int(num_layers)
        self.axis_condition = bool(axis_condition)
        if self.num_layers < 1:
            raise ValueError('the network needs at least one hidden layer')
        self.triplane = None
        if feature_mode == 'embedding':
            self.embedding = nn.Embedding(num_models, feat_dim)
            ## zero weights, so every loop starts from the same curve; the fit is
            ## what moves each row off it and onto its own loop
            nn.init.zeros_(self.embedding.weight)
            self.embedding_dim = feat_dim
        elif feature_mode == 'triplane':
            if not triplane:
                raise ValueError('triplane mode needs the field configuration')
            self.triplane = TriplaneField(**triplane)
            self.embedding_dim = self.triplane.feat_dim
        else:
            raise ValueError(f'unknown feature mode {feature_mode!r}')

        ## the sine encoding swaps the single coord for sin/cos, so the first layer
        ## must be built for that width: it cannot be a forward-time switch
        self.pe = pe
        width = self.embedding_dim + (2 if pe else 1) + (3 if self.axis_condition else 0)
        blocks = [nn.Linear(width, num_neurons),
                  nn.Softplus(beta=200)]
        for _ in range(self.num_layers - 1):
            blocks += [nn.Linear(num_neurons, num_neurons), nn.Softplus(beta=200)]
        blocks.append(nn.Linear(num_neurons, 3))
        self.model = nn.Sequential(*blocks)

    def forward(self, z, idx=None, feat=None, axis=None):
        """
        z: B x N x 1 parameter coords in [0-1], or N x 1 for a single part
        idx: part indices, one per sample, or None for the base curve, which is
             conditioned on an all-zero feature instead of a part's own
        feat: B x F feature vectors, when the condition comes from a field rather
             than the embedding table; they override idx and may carry gradients
             back into the field
        axis: B x 3 unit plane normal, one per sample, concatenated to the
             condition when the network is axis-conditioned. Which way round a
             loop runs is then the network's reading of "counter-clockwise about
             this axis", a rule that is the same on the cube and on a shape,
             rather than something it has to infer from the feature -- which at
             load is an untrained cell and carries no slice identity
        """
        if z.dim() == 2:
            ## a bare coord list belongs to one part, so give it the batch axis
            z = z.unsqueeze(0)

        if self.pe:
            ## [0-1] to [0-2pi] and on to sin/cos: the two ends of the parameter then
            ## map to the same input, so the loop closes instead of leaving a seam;
            ## the raw coord is dropped on purpose, it would break that equality
            ang = 2 * np.pi * z
            z = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)        # B x N x 2
        if feat is not None:
            ## a field supplies one vector per sample, so repeat it over the N coords
            emb = feat[:, None, :].expand(-1, z.shape[1], -1)
        elif idx is None:
            ## no part at all: a zero feature in the embedding's place. That is the
            ## one condition every loop shares, so a curve trained on it is the
            ## starting curve for all of them rather than for one of them
            emb = torch.zeros(z.shape[0], z.shape[1], self.embedding_dim,
                              dtype=z.dtype, device=z.device)
        else:
            if self.feature_mode != 'embedding':
                raise ValueError('a triplane network needs feat, not idx')
            ## the part embedding is a single vector per sample, so repeat it over the
            ## N coords before the concat, which needs equal dims on every axis but -1
            emb = self.embedding(idx)[:, None, :].expand(-1, z.shape[1], -1)
        parts = [z, emb]
        if self.axis_condition:
            if axis is None:
                raise ValueError('this network is axis-conditioned; pass one unit '
                                 'plane normal per sample as axis')
            axis = torch.as_tensor(axis, dtype=z.dtype, device=z.device).reshape(-1, 3)
            if axis.shape[0] != z.shape[0]:
                raise ValueError(f'{axis.shape[0]} axes for {z.shape[0]} samples')
            ## unit, so the condition is the direction alone and not its length
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            parts.append(axis[:, None, :].expand(-1, z.shape[1], -1))
        return self.model(torch.cat(parts, dim=-1))


class CD_Loss(nn.Module):
    def __init__(self):
        super(CD_Loss, self).__init__()
    
    def forward(self, z, target):
        """
        Chamfer distance, averaged over the batch, and the pairing it used.

        z: B x N x 3
        target: B x M x 3

        Returns (loss, near). near is B x N: the target point each predicted point was
        matched to. Any per-point term has to use that same pairing, or the two halves
        of the loss pull at different correspondences -- and asking for it here costs
        nothing, since the distance matrix is already there.
        """
        ## save the gpu memory: only the nearest-neighbour indices come back
        dm = torch.cdist(z.detach(), target.detach())
        near = torch.argmin(dm, dim=2)
        near_target = target.gather(1, near[..., None].expand(-1, -1, 3))
        near_z = z.gather(1, torch.argmin(dm, dim=1)[..., None].expand(-1, -1, 3))
        ## squared distances: a mean over the xyz components would cancel out
        loss_1 = ((z - near_target) ** 2).sum(-1).mean()
        loss_2 = ((target - near_z) ** 2).sum(-1).mean()
        return loss_1 + loss_2, near


## the tangent of a predicted curve, component by component, with respect to z
def curve_tangent(out, z):
    """
    d out/dz, B x N x 3.

    z is one scalar per point, so the derivative there is a 3-vector and one backward
    pass cannot hold it: differentiating the sum of all three components returns the
    rate at which they rise together, which is a number with no direction in it and
    reads zero on a curve that turns without climbing. Each component is therefore
    summed and differentiated on its own.

    create_graph keeps the tangent inside the graph. Without it the tangent is a
    detached number and the second derivative cannot be taken from it, so the curve
    could be pulled but not straightened.

    out: B x N x 3 predictions
    z: B x N x 1 parameter coords, a leaf of the graph that produced out
    """
    t = []
    for k in range(out.shape[-1]):
        ## one component summed leaves a scalar per point, and the gradient of that
        ## w.r.t. the single input value at the point is that component's own rate
        t.append(torch.autograd.grad(out[..., k].sum(), z, create_graph=True)[0])
    return torch.cat(t, -1)


## the rate the tangent of a predicted curve changes, with respect to z
def curve_curvature(d1, z):
    """
    d2 out/dz2, B x N x 3, taken from a tangent that is still inside the graph.

    Separate from curve_tangent so a loss that only needs the direction of the curve
    does not pay for a second set of backward passes.

    d1: B x N x 3 tangent, as curve_tangent returns it
    z: B x N x 1 parameter coords, a leaf of the graph that produced out
    """
    d2 = []
    for k in range(d1.shape[-1]):
        d2.append(torch.autograd.grad(d1[..., k].sum(), z, create_graph=True)[0])
    return torch.cat(d2, -1)


## how much a predicted curve bends, relative to how fast it sweeps round
def bend_penalty(d1, d2, eps=1e-12):
    """
    Mean squared second derivative of each curve over its own mean squared speed,
    averaged over the batch.

    Both derivatives scale with the curve, so their ratio does not: a circle reads
    4*pi^2 = 39.5 whatever its radius and whatever number of points describes it, and
    a curve that zig-zags or doubles back over itself reads far above that. One
    weight therefore means the same thing on a 0.05 loop and a 20 unit one, and on a
    100-point loop and a 4000-point one, which is what makes it usable as a bare
    number rather than a per-shape knob.

    The derivative is taken with respect to the parameter, not the arc length, so
    the same curve parameterised unevenly reads higher too -- which is what a wiggly
    fit usually is: points bunch up where the curve turns, and a fit that chases
    them has to speed up and slow down.

    d1, d2: B x N x 3, as curve_tangent and curve_curvature return them
    """
    speed = (d1 ** 2).sum(-1).mean(1)
    bend = (d2 ** 2).sum(-1).mean(1)
    ## the speed is the scale the bend is read against, not something to lower:
    ## differentiating through it would let the penalty be paid by shrinking the curve
    ## instead of straightening it, and a shrinking loop is the one way the fit loses
    ## the shape altogether
    return (bend / (speed.detach() + eps)).mean()


## how far a fitted curve's implied normal sits from the one the cloud gives
def normal_cosine(d1, n_target, axis, eps=1e-12):
    """
    Cos(predicted normal, given normal) at every point, B x N.

    The same construction as normal_penalty, kept separate because the per-loop
    direction is read off it: the orientation gauge and the report both need one
    cosine per loop rather than one mean over the batch.
    """
    a = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    t = d1 / (d1.norm(dim=-1, keepdim=True) + eps)
    n_pred = torch.cross(t, a[:, None, :].expand_as(t), dim=-1)
    n_pred = n_pred / (n_pred.norm(dim=-1, keepdim=True) + eps)
    ## the given normal read in the loop's own plane. The predicted normal lies in that
    ## plane by construction, and a cloud normal leans out of it -- on laurent_hand only
    ## 0.69 of its length is in-plane -- so the leaning part is not something a curve
    ## can ever match, and comparing against it would leave the term stuck at about
    ## 0.3 whatever the weight
    n_given = n_target - (n_target * a[:, None, :]).sum(-1, keepdim=True) * a[:, None, :]
    n_given = n_given / (n_given.norm(dim=-1, keepdim=True) + eps)
    return (n_pred * n_given).sum(-1)


def normal_penalty(d1, n_target, axis, eps=1e-12, per_curve=False):
    """
    Mean 1 - cos(predicted normal, given normal) over the paired points, averaged over
    the batch -- or over the points of each curve with per_curve, as a B vector --
    where the predicted normal is the one the fit is drawn with:

        predicted normal = tangent x plane normal

    That is the loss written as its definition rather than as an equivalent turn. The
    network predicts points, not normals, so the normal of a fitted curve is read off
    it: a contour lies in the surface, so its tangent is square to both the surface
    normal and the slice's plane normal, which leaves one in-plane direction a
    quarter turn from the tangent -- tangent x plane normal, the same turn the figures
    draw and the pipeline exports.

    Both normals are compared in the loop's own plane, because that is where the
    predicted one lies: the given normal is projected into the plane before the angle
    is taken, so what is measured is the direction of the surface normal round the
    loop rather than how far the cloud's normal leans out of the slice.

    The term is signed: 0 when the curve implies the normals it was given, 1 when its
    normal sits square to them (the curve cutting straight into the surface), and 2
    when it runs the other way round, which reads as the normal pointing inward.
    Squaring the dot instead would throw the sense away and leave the curve free to run
    either way, which is exactly the wiggling and doubling back the term is there to
    stop.

    d1: B x N x 3 tangent, as curve_tangent returns it
    n_target: B x N x 3 outward normals of the target points, one per point, as
              chamfer paired them to the predicted points
    axis: B x 3 plane normals, one per curve; the loop's own plane, read off its
          points, is what picks tangent x axis out of the circle of directions square
          to the tangent
    per_curve: average the points of each curve separately and return B, instead of
          one number over the batch
    """
    ## the cosine of the two normals, so the penalty is 0 only when the curve implies
    ## exactly the normal it was handed
    cos = normal_cosine(d1, n_target, axis, eps)
    return (1.0 - cos).mean(-1) if per_curve else (1.0 - cos).mean()


## how many times a closed curve turns about a point, and how much turning it spends
def curve_winding(points, axis, centre=None, eps=1e-12):
    """
    Net turn of each closed curve about `centre`, in turns, and the total turn it
    spends getting there. B each.

    The angle of each point is read in the plane's own frame -- u the world axis
    least aligned with the plane normal, v = axis x u -- and the increments are
    wrapped into (-pi, pi] before they are summed, including the seam from the last
    point back to the first. A simple loop reads +1 counter-clockwise seen from
    +axis, -1 clockwise, and a curve that runs out and back reads |1| with the
    out-and-back cancelling. That is the sense a per-point cosine cannot report: on
    a retrace the forward and backward stretches pair to the same target points and
    average to zero, while the net winding still says which way the loop closes.

    `foldover` is the same sum with the increments taken in absolute value. A loop
    that turns once reads 1, a retrace adds the angle it walks back, so a curve that
    covers its own path reads above 1. It is a reading, not a gauge: a very
    non-convex loop can also read above 1.

    points: B x N x 3, the curve in closed order
    axis: B x 3 plane normals; the basis is chosen the same way `in_plane_u` chooses
          the warm-up square's x, so the cube and a fit agree on the sense
    centre: B x 3 reference point; default the curve's own mean
    """
    if centre is None:
        centre = points.mean(1)
    axis = axis.to(points.dtype)
    centre = centre.to(points.dtype)
    a = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    world = torch.eye(3, device=points.device, dtype=points.dtype)
    ## ties broken by the lower index, as in_plane_u does: a plane normal exactly on
    ## a world axis leaves two axes equally least aligned, and an arbitrary pick would
    ## let a floating point wobble decide the frame
    pick = (a.abs() <= a.abs().min(dim=-1, keepdim=True).values + 1e-9)
    u = world[pick.float().argmax(-1)]
    u = u - (u * a).sum(-1, keepdim=True) * a
    u = u / (u.norm(dim=-1, keepdim=True) + eps)
    v = torch.cross(a, u, dim=-1)
    rel = points - centre[:, None, :]
    theta = torch.atan2((rel * v[:, None, :]).sum(-1),
                        (rel * u[:, None, :]).sum(-1))
    two_pi = 2.0 * np.pi
    turns = theta[:, 1:] - theta[:, :-1]
    turns = torch.remainder(turns + np.pi, two_pi) - np.pi
    seam = torch.remainder(theta[:, :1] - theta[:, -1:] + np.pi, two_pi) - np.pi
    turns = torch.cat([turns, seam], dim=1)
    return turns.sum(1) / two_pi, turns.abs().sum(1) / two_pi


def winding_number(points, axis, centre=None):
    """`curve_winding` for one curve in numpy, returning (winding, foldover)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    a = a / (np.linalg.norm(a) + 1e-12)
    u = in_plane_u(a)
    u = u - (u @ a) * a
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    c = pts.mean(0) if centre is None else np.asarray(centre, dtype=np.float64)
    rel = pts - c
    theta = np.arctan2(rel @ v, rel @ u)
    two_pi = 2.0 * np.pi
    turns = (np.diff(theta) + np.pi) % two_pi - np.pi
    seam = (theta[0] - theta[-1] + np.pi) % two_pi - np.pi
    turns = np.append(turns, seam)
    return turns.sum() / two_pi, np.abs(turns).sum() / two_pi


## how much of a term to take at step t of a run that lasts `total` steps: full at the
## first one, nothing at the last, along half a cosine
def cosine_anneal(t, total, eps=0):
    """
    1.0 at step 0, 0.0 at the last step of `total`, following half a cosine, so the
    weight leaves and arrives flat rather than at a corner.

    Read at a step that is not a step count but a fraction of the run -- a term that
    should hold early and then stop pulling -- it is the shape to use: a linear ramp
    spends as much of the run at half strength as at anything else, while this one is
    still near its start over the middle and only falls away at the end.

    t: the step, counted from 0
    total: steps in the whole run, so t = total - 1 lands on 0.0
    eps: what is left at the last step, for a term that should keep a residue
    """
    return eps + (1.0 - eps) * 0.5 * (1.0 + np.cos(np.pi * t / max(total - 1, 1)))


## The universal cube the warm-up target is cut from: side 1, centred at the origin,
## the same for every shape and every loop.  It is a definition, not a measurement --
## nothing reads a cloud, or a box around one, to size it.  A shape larger than the
## cube is warmed on the cube anyway; the fit is what puts the curve on the shape.
UNIT_CUBE_HALF = 0.5


def in_plane_u(axis):
    """The square's x for a plane of the given normal: the world axis least aligned
    with it, ties broken by the lower index.

    A normal that lies exactly on a world axis is why this exists: two world axes
    are then equally least-aligned, and choosing by the smaller floating point
    component would let a 1e-4 wobble in the normal decide the frame. The lower
    index is a convention, so every loop cut by the same plane gets the same
    square and a phase is read against one origin.

    axis: unit normal of the plane the square lies in
    """
    mag = np.abs(np.asarray(axis, dtype=np.float64).reshape(3))
    return np.eye(3)[int(np.flatnonzero(mag <= mag.min() + 1e-9)[0])]


## the one curve whose parameterisation is known exactly: a square, p(z), held as
## data -- how big it is and where it stands -- so every question the warm-up asks
## about it is answered by the same object
class Square:
    """
    A square: half-width, centre, and the plane it lies in, given as its unit normal
    `axis` and the in-plane direction `u` its own x runs along. `v` is not taken, it is
    closed by v = axis x u, so a walk that is counter-clockwise in (u, v) is
    counter-clockwise seen from the plane normal.

    That is the whole point of carrying the placement in the data rather than drawing
    a square at the origin and moving it later: the walk's sense and the plane's normal
    then agree by construction, so the square's own outward edge normals are the ones
    its walk implies, which is what the direction term rewards and what lets a loop's
    fit start already the right way round.

    The default is the unit square at the origin in z = 0, u along x, v along y: one
    cross-section of the unit cube, which is what the cube warm-up fits. A loop's own
    square is built with `Square.from_points`, which reads the placement off the loop.

    `at(t)` is the curve itself at any parameter -- so a coordinate drawn anywhere gets
    the point that really sits there, with no correspondence search and no quantising
    to a sample -- and `loop(n)` is that same curve laid down as samples, with the
    outward normal of the edge each sits on and its arc-length parameter.
    """

    ## the four edges, walked counter-clockwise seen from the plane normal and starting
    ## at a corner: up the right, across the top, down the left, along the bottom, each
    ## as (start corner, unit walk, outward unit normal) on a square of half-width 1
    EDGES = (
        ((1.0, -1.0), (0.0, 1.0), (1.0, 0.0)),
        ((1.0, 1.0), (-1.0, 0.0), (0.0, 1.0)),
        ((-1.0, 1.0), (0.0, -1.0), (-1.0, 0.0)),
        ((-1.0, -1.0), (1.0, 0.0), (0.0, -1.0)),
    )

    def __init__(self, half=1.0, centre=None, axis=None, u=None):
        """
        half: centre to edge, so the side is 2 * half
        centre: the square's centre in world coordinates; default the origin
        axis: unit normal of the plane the square lies in; default +z
        u: the in-plane direction the square's own x runs along; default +x, and only
           its part inside the plane is used, the rest being dropped
        """
        if half <= 0:
            raise ValueError(f'half must be positive, got {half}')
        self.half = float(half)
        self.centre = (np.zeros(3) if centre is None else
                       np.asarray(centre, dtype=np.float64).reshape(3))
        axis = np.array([0.0, 0.0, 1.0]) if axis is None else axis
        axis = np.asarray(axis, dtype=np.float64).reshape(3)
        self.axis = axis / np.linalg.norm(axis)
        u = np.array([1.0, 0.0, 0.0]) if u is None else u
        u = np.asarray(u, dtype=np.float64).reshape(3)
        ## u has to lie in the plane, or the square would not: keep its in-plane part
        u = u - (u @ self.axis) * self.axis
        if np.linalg.norm(u) < 1e-9:
            raise ValueError('u must not run along the plane normal')
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(self.axis, self.u)

    @classmethod
    def from_points(cls, points, half=None):
        """
        The square standing in a loop's place: the loop's own centre and plane, and --
        unless `half` is given -- the loop's own mean radius.

        The plane normal's sign is fixed here rather than left to the SVD, so the same
        loop always gets the same square and the same way round: the largest component
        is made positive. The square's x then runs along the world axis the normal is
        least aligned with, which is the one that projects into the plane with the most
        length left; a loop in the z = 0 plane therefore comes out as u = x, v = y,
        exactly the square `Square(half)` describes.

        points: S x 3 points of one loop, in its plane
        half: centre to edge. Default: the mean distance of the loop's points from
              their centre, so every square is the size of the loop it stands for
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        centre = pts.mean(0)
        axis = np.linalg.svd(pts - centre)[2][-1]
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
        if half is None:
            half = float(np.linalg.norm(pts - centre, axis=1).mean())
        return cls(half, centre, axis, np.eye(3)[int(np.argmin(np.abs(axis)))])

    def at(self, t):
        """
        k x 3 points p(z) at the k parameters t, and the index of the edge each one
        lands on.

        The walk is counter-clockwise seen from the plane normal, which is the sense
        the direction term rewards: it asks the tangent to be axis x normal, and with
        an in-plane outward normal that is this way round. A parameter landing exactly
        on a corner takes the edge that starts there.

        t: parameters in turns, of any shape; wrapped into [0, 1), so 1.0 reads as 0.0

        Returns (points, edges): k x 3 points and k edge indices.
        """
        q = np.mod(np.asarray(t, dtype=np.float64).reshape(-1), 1.0)
        side = np.minimum(np.floor(4 * q).astype(int), len(self.EDGES) - 1)
        frac = 4 * q - side
        starts = np.array([e[0] for e in self.EDGES])
        walks = np.array([e[1] for e in self.EDGES])
        ## the square in its own two directions, then carried into the plane
        xy = self.half * (starts[side] + walks[side] * 2 * frac[:, None])
        points = self.centre + xy[:, :1] * self.u + xy[:, 1:2] * self.v
        return points, side

    def loop(self, n):
        """
        n x 3 points round the square, one per even split of [0, 1), the n x 3 outward
        normals of the edges they sit on, and the n arc-length parameters they sit at.

        The points are `at` read at the even parameters k / n; the normals and
        parameters travel with them so a fit can be shown which way the surface faces
        there. What holds a coordinate to where it belongs is `at` itself, since a
        square's parameterisation is known in closed form.

        n: samples round the perimeter, laid down evenly by arc length, so parameter
           step and distance travelled agree

        Returns (points, normals, params); a sample landing on a corner is given the
        edge that starts there.
        """
        if n < 1:
            raise ValueError(f'a square needs at least one sample, got {n}')
        points, side = self.at(np.arange(n) / n)
        out = np.array([e[2] for e in self.EDGES])[side]
        normals = out[:, :1] * self.u + out[:, 1:2] * self.v
        ## the parameter is the distance walked, as a fraction of the whole perimeter:
        ## read off the samples themselves, so it comes out at k / n for even spacing
        ## and would still be right for any other spacing
        steps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        params = np.concatenate([[0.0], np.cumsum(steps)[:-1]]) / steps.sum()
        return points, normals, params

    def normals_at(self, t):
        """
        k x 3 outward normals of the edges the k parameters t land on.

        The same normals `loop` hands back with its samples, read at arbitrary
        parameters: constant along an edge and jumping a right angle at each corner.
        They are what the pre-training direction term is held against, so the curve
        is taught the square's own sense of outward even where it is still far from
        the square in position.

        t: parameters in turns, of any shape; wrapped into [0, 1)
        """
        _, side = self.at(t)
        out = np.array([e[2] for e in self.EDGES])[side]
        return out[:, :1] * self.u + out[:, 1:2] * self.v


def cube_loops(planes=4, samples=256, half=UNIT_CUBE_HALF, fill=0.8):
    """
    The unit cube cut by `planes` axis-aligned planes per axis, as the warm-up's
    loop set.

    Each loop is the cube's cross-section on the plane `axis = d`: a square of
    half-side `half` centred at `d * axis`, walked counter-clockwise seen from
    `+axis`. The target handed back is in the plane's own frame -- the world point
    minus the plane's centre -- so every loop's target is the same unit square; the
    offset is carried by the query point, which is where the field is read. That is
    the split the real fit uses, where a loop's target is in its plane's frame and
    the loop's place in the shape is its query.

    Loops adjacent in offset along one axis are paired, so the seam distance D has
    the pairs that "z = 0 close to the neighbours" is asked over.

    There is no plane through the exact centre: the three centre cross-sections,
    x = 0, y = 0 and z = 0, all have their centroid at the origin, so one field
    feature would be asked to produce three different squares. A real shape does not
    repeat that coincidence -- its centre cuts have different centroids -- so the
    cube skips the centre and every loop's query is its own.

    planes: planes per axis, an even number, spread evenly over +-fill * half
    samples: points round each loop, at the parameters k / samples
    half: the cube's half-side, `UNIT_CUBE_HALF` by default -- a definition, never
          a measurement
    fill: how far the outermost plane sits from the centre, as a fraction of `half`.
          0.8 keeps every plane off the cube's faces

    Returns a dict:
      targets: K x samples x 3 plane-local points on the cube
      normals: K x samples x 3 outward normals of the faces there
      queries: K x 3 world points at the loops' centres
      axes:    K x 3 plane normals
      pairs:   P x 2 loop indices adjacent along one axis
    """
    if planes < 2 or planes % 2:
        raise ValueError(f'the cube needs an even number of planes per axis, none '
                         f'at the centre, got {planes}')
    if samples < 4:
        raise ValueError(f'a square loop needs at least four samples, got {samples}')
    if half <= 0:
        raise ValueError(f'the cube half-side must be positive, got {half}')
    if not 0.0 < fill < 1.0:
        raise ValueError(f'fill must be inside (0, 1), got {fill}')
    offsets = np.linspace(-fill * half, fill * half, planes)
    targets, normals, queries, axes, pairs = [], [], [], [], []
    for a in range(3):
        axis = np.eye(3)[a]
        square = Square(half, centre=None, axis=axis, u=in_plane_u(axis))
        points, outward, _ = square.loop(samples)
        base = a * planes
        for k, d in enumerate(offsets):
            targets.append(points)
            normals.append(outward)
            axes.append(axis)
            queries.append(d * axis)
            if k:
                pairs.append((base + k - 1, base + k))
    return dict(targets=np.stack(targets), normals=np.stack(normals),
                queries=np.stack(queries), axes=np.stack(axes),
                pairs=np.asarray(pairs, dtype=np.int64))


def warm_on_cube(model, steps=300, lr=3e-2, align=1.0, seam=1.0, planes=4,
                 samples=512):
    """
    Train `model` on the unit cube and nothing else.

    The cube's loops are `cube_loops`: every cross-section is a unit square, in its
    own plane's frame, walked counter-clockwise seen from the plane normal, so the
    target is exact and needs no correspondence search. Each step draws all of them
    and takes three terms:

    - fit: |f(z) - p(z)| against the cube's cross-sections, so the curves lie on the
      cube;
    - direction: the normal the curve implies, tangent x plane normal, against the
      cube's outward face normal, so every loop along an axis circulates the same
      way;
    - seam: the seam distance D between loops adjacent along an axis -- their mean
      squared separation over their own mean squared radius, both centred -- taken
      with the phase at zero and differentiated through the network. The point is
      that the network itself puts z = 0 where its neighbours put it, so the
      placement is carried by the weights rather than by a phase buffer.

    The triplane frame is set to the unit cube for the run and put back afterwards.
    The grids are a function of unit-cube position, so a field warmed here reads the
    same way once the frame is moved onto a shape's box; the frame belongs to the
    caller, not to the warm-up.

    steps: optimisation steps, every loop in every step
    lr: starting rate, cosine-annealed to 5% over the run
    align: weight of the direction term, 0 to train position and seam only
    seam: weight of the seam term, 0 to leave z = 0 unconstrained
    planes: even planes per axis on the cube, which sets the neighbouring pairs;
            none cuts the centre (see `cube_loops`)
    samples: points round every loop, and so the parameter grid

    Returns a dict with the three reported readings: `fit` and `fit_max` (distance
    to the cube's cross-sections, same units as the fit), `cos` (mean direction
    cosine against the outward normals) and `seam_median` / `seam_max` (D between
    neighbouring loops at z = 0).
    """
    if getattr(model, 'feature_mode', None) != 'triplane':
        raise ValueError('the cube warm-up conditions the network on the triplane '
                         'field; embedding rows name loop planes, not cube ones')
    if steps < 1:
        raise ValueError(f'steps must be at least 1, got {steps}')
    if lr <= 0:
        raise ValueError(f'lr must be positive, got {lr}')
    for name, weight in (('align', align), ('seam', seam)):
        if weight < 0:
            raise ValueError(f'{name} must be non-negative, got {weight}')
    if planes < 2 or planes % 2:
        raise ValueError(f'the cube needs an even number of planes per axis, none '
                         f'at the centre, got {planes}')
    if samples < 4:
        raise ValueError(f'every loop needs at least four samples, got {samples}')

    device = next(model.parameters()).device
    cube = cube_loops(planes, samples)
    targets = torch.from_numpy(cube['targets']).float().to(device)
    normals = torch.from_numpy(cube['normals']).float().to(device)
    axes = torch.from_numpy(cube['axes']).float().to(device)
    queries = torch.from_numpy(cube['queries']).float().to(device)
    pairs = torch.as_tensor(cube['pairs'], device=device)
    count = targets.shape[0]

    ## the denominator is the pair's size on the cube, not the predicted pair's own:
    ## the predicted size is the fit's business, and letting the seam term see it in
    ## the gradient gives it a cheaper way down than placing the seam -- grow one
    ## curve towards the other. Against a fixed target scale, D falls only when the
    ## two curves are read at the same place, which is the property being trained.
    plane = targets - targets.mean(1, keepdim=True)
    pair_scale = 0.5 * ((plane[pairs[:, 0]] ** 2).sum(-1).mean(1)
                        + (plane[pairs[:, 1]] ** 2).sum(-1).mean(1))

    def seam_gap(out):
        """D between the paired loops, each centred on its own points."""
        v = out - out.mean(1, keepdim=True)
        vi, vj = v[pairs[:, 0]], v[pairs[:, 1]]
        return ((vi - vj) ** 2).sum(-1).mean(1) / pair_scale.clamp_min(1e-12)

    triplane = model.triplane
    saved_centre = triplane.box_centre.detach().clone()
    saved_half = triplane.box_half.detach().clone()
    coords = torch.linspace(0, 1, samples + 1, device=device)[:-1, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    every = max(1, steps // 12)
    try:
        with torch.no_grad():
            triplane.box_centre.zero_()
            triplane.box_half.fill_(UNIT_CUBE_HALF)
        z = coords[None].expand(count, -1, -1).clone().requires_grad_(True)
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            ## read again every step: the field's sample is a graph input and the
            ## step before freed it
            feat = triplane(queries)
            out = model(z, None, feat, axes)
            fit = (out - targets).norm(dim=-1).mean()
            loss = fit
            direction = None
            if align:
                direction = normal_penalty(curve_tangent(out, z), normals, axes)
                loss = loss + align * direction
            gap = None
            if seam:
                gap = seam_gap(out)
                loss = loss + seam * gap.mean()
            loss.backward()
            optimizer.step()
            for group in optimizer.param_groups:
                group['lr'] = lr * cosine_anneal(step - 1, steps, eps=0.05)
            if step % every == 0 or step == steps:
                line = f'  cube {step}/{steps}: fit {fit.item():.4f}'
                if direction is not None:
                    line += f', direction {1.0 - direction.item():+.3f} cos'
                if gap is not None:
                    line += f', seam D {float(gap.median()):.4f}'
                print(line)

        ## the three readings the task asks for, on the state the run ended on
        feat = triplane(queries)
        zz = coords[None].expand(count, -1, -1).clone().requires_grad_(True)
        out = model(zz, None, feat, axes)
        with torch.no_grad():
            per = (out - targets).norm(dim=-1).mean(1)
            gap = seam_gap(out)
        cos = 1.0 - float(normal_penalty(curve_tangent(out, zz), normals, axes))
        report = dict(steps=steps, planes=planes, samples=samples, loops=count,
                      fit=float(per.mean()), fit_max=float(per.max()), cos=cos,
                      seam_median=float(gap.median()), seam_max=float(gap.max()))
        print(f'  cube warm-up {steps} steps, {count} loops ({planes} per axis): fit '
              f'{report["fit"]:.4f} mean, {report["fit_max"]:.4f} max; direction '
              f'{cos:+.3f} cos; seam D median {report["seam_median"]:.4f}, max '
              f'{report["seam_max"]:.4f}')
        return report
    finally:
        with torch.no_grad():
            triplane.box_centre.copy_(saved_centre)
            triplane.box_half.copy_(saved_half)


def save_warmup(path, model, **meta):
    """Write a cube-warmed network and field, so a shape's fit can load it as its
    initialisation.

    The checkpoint holds the shared MLP and the triplane grid, and the cube frame
    the grid was warmed in. Nothing shape-specific is written: there are no rows and
    no phases, because the warm-up had no shape to name.
    """
    triplane = getattr(model, 'triplane', None)
    if triplane is None:
        raise ValueError('a cube warm-up checkpoint needs a triplane model')
    payload = dict(
        kind='cube_warmup',
        feature_mode='triplane',
        num_neurons=int(model.model[0].out_features),
        num_layers=int(model.num_layers),
        pe=bool(model.pe),
        embedding_dim=int(model.embedding_dim),
        axis_condition=bool(getattr(model, 'axis_condition', False)),
        num_models=1,
        codes=np.zeros(1, dtype=np.int64),
        queries=None,
        triplane=dict(centre=triplane.box_centre.detach().cpu().numpy(),
                      half=triplane.box_half.detach().cpu().numpy(),
                      resolution=triplane.resolution, channels=triplane.channels,
                      tv=triplane.tv),
        phases=None,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
    )
    payload.update(meta)
    torch.save(payload, path)
    return path


def load_warmup(path, model):
    """Copy a cube warm-up checkpoint into `model`, keeping the model's own triplane
    frame.

    The frame says where the unit cube is laid on this shape's box, and the grid is a
    function of unit-cube position, so the frame belongs to the fit rather than to
    the warm-up: copying it would put the warmed cells where this shape never reads.
    The shapes are checked first, so a checkpoint warmed on a different network or
    grid is refused rather than silently loaded in part.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('kind') != 'cube_warmup':
        raise ValueError(f'{path} is not a cube warm-up checkpoint')
    if getattr(model, 'feature_mode', None) != 'triplane':
        raise ValueError('a cube warm-up transfers through the triplane field, and '
                         'this model conditions on embedding rows')
    if bool(ckpt.get('axis_condition', False)) != bool(model.axis_condition):
        raise ValueError(
            f'{path} was warmed {"with" if ckpt.get("axis_condition") else "without"} '
            f'axis conditioning and this network is '
            f'{"not" if model.axis_condition else "axis-conditioned"}; build the '
            f'model to match, or retrain the warm-up')
    source = ckpt['state_dict']
    own = model.state_dict()
    shared = {name: value for name, value in source.items()
              if not name.endswith(('box_centre', 'box_half'))}
    bad = [name for name, value in shared.items()
           if name not in own or own[name].shape != value.shape]
    if bad:
        shown = ', '.join(f'{name} {tuple(shared[name].shape)} vs '
                          f'{tuple(own[name].shape) if name in own else "missing"}'
                          for name in bad[:4])
        raise ValueError(f'{path} does not fit this network or grid: {shown}'
                         f'{" ..." if len(bad) > 4 else ""}. Warm up with the same '
                         f'neurons, layers, resolution and channels')
    model.load_state_dict(shared, strict=False)
    return ckpt


## fixed-size batches out of point sets holding different numbers of points
class DataGenerator:
    def __init__(self, point_sets, max_points=None, batch_size=1, seed=0, normals=None):
        """
        Batches of one shape from targets of varying size.

        point_sets: list of S_i x 3 arrays, e.g. the sliced cross-section loops
        max_points: M, points per target after sampling (default: median S_i)
        batch_size: targets per batch
        normals: list of S_i x 3 unit normals, one per point of each set, or None.
                 They are drawn with the same indices as the points, so a normal never
                 comes loose from the point it describes

        A target with more than M points is subsampled without replacement, one
        with fewer is padded by repeating its points, so M is the same on every
        iteration and a batch is always batch_size x M x 3.
        """
        self.point_sets = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in point_sets]
        if any(len(p) == 0 for p in self.point_sets):
            raise ValueError('every point set needs at least one point')

        self.normals = None if normals is None else [
            np.asarray(n, dtype=np.float64).reshape(-1, 3) for n in normals]
        if self.normals is not None and any(len(n) != len(p) for n, p in
                                            zip(self.normals, self.point_sets)):
            raise ValueError('every normal set needs one normal per point')

        sizes = np.array([len(p) for p in self.point_sets])
        self.max_points = int(np.median(sizes)) if max_points is None else int(max_points)
        if self.max_points < 1:
            raise ValueError(f'max_points must be positive, got {self.max_points}')
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        ## batches in one pass over the point sets
        return -(-len(self.point_sets) // self.batch_size)

    def pick(self, i):
        """M indices into point set i, repeating points if it holds fewer than M."""
        pts = self.point_sets[i]
        if len(pts) >= self.max_points:
            return self.rng.choice(len(pts), self.max_points, replace=False)
        ## tile shuffled copies so the repeats spread out instead of clustering
        reps = -(-self.max_points // len(pts))
        return np.tile(self.rng.permutation(len(pts)), reps)[:self.max_points]

    def index_batches(self):
        """Yields (point set indices, batch, normal batch or None), so callers can tell
        the sets apart."""
        ## reshuffle the point sets every pass, so the batches differ per epoch
        order = self.rng.permutation(len(self.point_sets))
        ## cycle the tail back to the start, every batch keeps the full batch_size
        order = np.resize(order, len(self) * self.batch_size)
        for start in range(0, len(order), self.batch_size):
            ids = order[start:start + self.batch_size]
            ## one draw of indices feeds the points and their normals both, so a normal
            ## never comes loose from the point it describes
            picks = [self.pick(i) for i in ids]
            batch = np.stack([self.point_sets[i][p] for i, p in zip(ids, picks)])
            yield (ids, batch,
                   None if self.normals is None else
                   np.stack([self.normals[i][p] for i, p in zip(ids, picks)]))


class Trainer:
    def __init__(self, point_sets, num_models=1, max_points=None, batch_size=None,
                 latent_points=None, lr=1e-2, seed=0, num_neurons=64, num_layers=2,
                 feat_dim=32,
                 smooth=0.0, normals=None, align=0.0, codes=None,
                 code_pairs=None, code_weights=None, code_smooth=0.0,
                 feature_mode='embedding', triplane=None, query_points=None,
                 feature_pairs=None, feature_tv=0.0, axis_condition=False,
                 distill=0.0, distill_warmup=None, distill_pairs=None,
                 distill_weights=None,
                 seam_align=0.0, seam_pairs=None, seam_weights=None, seam_keys=None,
                 seam_every=10, seam_start=0, seam_steps=100, seam_lam=0.02,
                 seam_lr=2e-2, seam_mu=1e-3):
        """
        Fit a list of point sets, one target per sample.

        point_sets: list of S_i x 3 targets of varying size, e.g. the loop slices
        num_models: rows in the embedding table, i.e. how many curves the shared
                     network is conditioned on
        codes: the row each target is conditioned on, one per point set. This is what
                     ties a target to a feature code that outlives the list: a plane
                     keeps its row when the set around it is rebuilt, so the fit stays
                     on the loop it was put on and nothing has to be matched up by
                     position. None gives one row per target, i % num_models
        max_points: M, points per target (default: median S_i), see DataGenerator
        batch_size: targets per iteration (default: all of them, one batch)
        latent_points: N, parameter coords the network is evaluated at (default: M)
        num_neurons: width of the shared MLP; the whole set shares one, so a target
                     with a lot of structure in it needs a wider one
        feat_dim: length of the latent vector each target is conditioned on, which is
                     what lets one MLP serve many different targets
        smooth: weight of the bend penalty on d2 out/dz2. The fit alone is free to
                     fold a curve back over itself or leave it wiggling between two
                     branches of a point set, since chamfer distance only asks for the
                     nearest point; the penalty is what makes the second derivative
                     small, and 0 leaves the fit unconstrained
        normals: list of S_i x 3 outward unit normals, one per point of each set.
                     Chamfer distance pairs points by distance alone, so a curve can
                     cut across the surface it is meant to run along wherever two
                     stretches of it come close; the outward normals are what say which
                     way the surface faces and which way round the loop runs
        align: weight of the normal term, 0 to fit by position only. It reads 0 when a
                     curve runs the way the outward normals say, 1 when it runs across
                     them, and 2 when it runs the other way round, so it is a fraction
                     of the loss rather than a distance
        code_pairs: P x 2 embedding rows that should stay close, e.g. the codes of
                     loops on neighbouring planes. The term is the weighted mean of
                     their squared code distance, so it constrains the latent table
                     rather than the curves directly
        code_weights: one non-negative weight per pair, so a neighbour that is close
                     in the plane counts more than one at the edge of the gate
        code_smooth: weight of the code-smoothness term, 0 to learn every row freely.
                     It is in the same units as the chamfer loss, a squared distance
                     summed over xyz, so it competes with the fit rather than replacing
                     it; the row still moves to its own loop, just less independently
                     of its neighbours
        feature_mode: 'embedding' keeps the per-target row table. 'triplane' replaces
                     it with a continuous field: every target is conditioned on the
                     feature sampled at its query point, and there is no row to hand
                     out, inherit or blend. Neighbouring loops then start from similar
                     features because the field is continuous, not because a term made
                     their rows similar
        triplane: configuration for the field: the world box its unit cube is
                     mapped onto (centre, half), resolution and channels; required
                     in triplane mode
        query_points: one xyz per target, in world coordinates, only in triplane
                     mode. A loop's own centre is the natural query; the target is in
                     its plane's local frame, so the caller adds that frame's origin
        feature_pairs: P x 2 target indices whose sampled features should stay close,
                     the triplane analogue of code_pairs. It is applied to the sampled
                     feature vectors, so gradients reach the grid cells the two loops
                     read
        feature_tv: weight of a total-variation penalty on the three grids, 0 to leave
                     every cell free
        axis_condition: concatenate each loop's slice axis to the network's
                     condition, so the curve's direction is read from the axis
                     rather than inferred from the feature. The axis is the unit
                     normal of the plane the target lies in, read off the target
                     with `Square.from_points`, and is the same +x/+y/+z on the
                     cube warm-up, so the direction rule learned there carries to
                     a shape whose field is still untrained
        distill: weight of a teacher-student seam distillation, 0 to train on the
                     shape alone. The warm-up teacher already draws the shape and
                     runs it the right way round; what the shape fit is free to lose
                     is where z = 0 falls, since no fit term sees it. The teacher is
                     therefore read once at the shape's own queries and axes, its
                     per-pair seam distance D is the target, and every step's student
                     D is pulled to that profile. The curves themselves are never
                     matched pointwise: a common deformation costs nothing, so shape
                     and orientation stay with the fit and the warm-up. The term is
                     the weighted mean absolute difference of the two D profiles,
                     carried back to a squared length by each pair's target radius,
                     so it is in the same units as the chamfer loss and is exactly
                     zero when the student's seam agreement is the teacher's
        distill_warmup: the cube-warm-up checkpoint the teacher is built from.
                     `rosa_main` passes its own `--warmup`, so the teacher is the
                     checkpoint the fit started from unless another one is named
        distill_pairs, distill_weights: P x 2 loop indices whose seams the profile
                     is read over, and one non-negative weight per pair, e.g. the
                     same neighbour graph the phase solve uses
        seam_align: weight of the seam-alignment energy, 0 to leave every loop's
                     parameterisation alone. The fit cannot see where z = 0 falls on a
                     closed curve -- sliding a loop through its own parameter changes
                     no chamfer, bend or normal term -- so that phase is one number per
                     loop, learned against its neighbours and nothing else. The curves
                     are read once, detached, and only the phases take a gradient, so
                     the shape never moves: what changes is where the first sample of
                     an exported loop sits
        seam_pairs: P x 2 target indices whose seams are held together, e.g. the
                     mutual-nearest loops of `code_neighbour_graph` in `rosa_main`
        seam_weights: one non-negative weight per pair, as for `code_pairs`
        seam_keys: one key per target, used only by `inherit` to carry a phase over
                     to the next plane set by what the loop is rather than by where
                     it sits in the list
        seam_every: solve the phases every this many epochs, 0 to never solve them
        seam_start: the first epoch the solve runs at, counted from 1
        seam_steps: Adam steps per solve; each solve starts from the phases the last
                     one left, so ten steps every ten epochs is one continuing
                     optimisation rather than ten cold starts
        seam_lam: the scale that decides how much a pair votes, in the same units as
                     the seam distance D (mean squared separation over the two loops'
                     mean squared radius). Each pair is matched at its best circular
                     shift first, and its vote is lam / (lam + D_best): a pair that
                     agrees after a shift is the same cross-section with its seams in
                     the wrong place and votes nearly in full, while two different
                     loops -- a torso against a limb -- vote at nearly nothing. The
                     energy itself is D, so a seam far out is pulled in proportion to
                     how far out it is rather than sinking into a saturating switch
        seam_lr: Adam step size on the phases, in turns
        seam_mu: weight of a zero-mean prior on the phases, which is what keeps the
                     whole run from rotating together: only differences are observable,
                     and this picks the one nearest the phases it started from
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_mode = feature_mode
        if feature_mode == 'triplane':
            if query_points is None:
                raise ValueError('triplane mode needs one query point per target')
            query_points = np.asarray(query_points, dtype=np.float32)
            if query_points.shape != (len(point_sets), 3):
                raise ValueError(f'one xyz query per target: {len(point_sets)} '
                                 f'targets, shape {query_points.shape}')
            self.queries = torch.as_tensor(query_points, device=self.device)
            self.num_models = len(point_sets)
            self.codes = np.arange(len(point_sets), dtype=np.int64)
        else:
            self.queries = None
            self.num_models = num_models
            if codes is None:
                self.codes = np.arange(len(point_sets)) % num_models
            else:
                self.codes = np.asarray(codes, dtype=np.int64).reshape(-1)
                if len(self.codes) != len(point_sets):
                    raise ValueError(f'one code per target: {len(point_sets)} targets, '
                                     f'{len(self.codes)} codes')
                if self.codes.min(initial=0) < 0 or self.codes.max(initial=-1) >= num_models:
                    raise ValueError(f'codes must name rows of the {num_models}-row '
                                     f'table, got {self.codes.min()}..{self.codes.max()}')
        ## one batch of every target by default, not one per row: with a code per plane
        ## the table is larger than the target list, and a batch sized by the table
        ## would repeat targets to fill it
        self.dataloader = DataGenerator(point_sets, max_points,
                                        batch_size or len(point_sets), seed, normals)
        
        self.model = miniNN(1 if feature_mode == 'triplane' else num_models,
                            num_neurons=num_neurons, num_layers=num_layers,
                            feat_dim=feat_dim, feature_mode=feature_mode,
                            triplane=triplane,
                            axis_condition=axis_condition).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.smooth = smooth
        self.align = align
        self.code_smooth = float(code_smooth)
        self.code_pairs = None
        self.code_weights = None
        self.feature_pairs = None
        self.feature_weights = None
        self.feature_tv = float(feature_tv)
        if feature_mode == 'triplane':
            if code_pairs is not None and len(code_pairs):
                raise ValueError('triplane mode takes feature_pairs, not code_pairs')
            if feature_pairs is not None and len(feature_pairs):
                self.feature_pairs, self.feature_weights = self._pair_tensors(
                    feature_pairs, code_weights, len(point_sets), 'feature')
        elif feature_pairs is not None and len(feature_pairs):
            raise ValueError('feature_pairs only apply in triplane mode')
        else:
            if code_pairs is not None and len(code_pairs):
                self.code_pairs, self.code_weights = self._pair_tensors(
                    code_pairs, code_weights, num_models, 'code')
        self.loss_fn = CD_Loss()
        self.scheduler = None

        ## the plane each target lies in, read off the target itself: a loop is a
        ## cross-section, so the direction it is flattest in is the plane it was cut
        ## with, and the normal term's tangent x axis needs it. The sign is the one
        ## Square.from_points fixes, so this is the very axis the warm-up's square is
        ## walked counter-clockwise around, and the two agree on the way round. With
        ## axis_condition the same vector is the network's input as well, so the
        ## condition and the direction term read one axis, not two
        if align and self.dataloader.normals is None:
            raise ValueError('align needs the normals of every point set')
        self.axes = (torch.as_tensor(
            np.stack([Square.from_points(p).axis for p in self.dataloader.point_sets]),
            dtype=torch.float32, device=self.device)
            if (align or axis_condition) else None)

        ## self.z are the [0-1] parameter coords; the latent vector of a part is
        ## appended inside forward, from that part's embedding, so it stays learnable
        latent_points = self.dataloader.max_points if latent_points is None else latent_points
        self.z = torch.linspace(0, 1, latent_points + 1, device=self.device)[:-1, None]  # N x 1

        ## the teacher-student seam distillation: the warm-up maps the shape's own
        ## queries to curves, and its per-pair D is the ceiling the fit is held
        ## under. The curves are never matched, so this term sees only the seam
        self.distill = float(distill)
        if self.distill < 0:
            raise ValueError(f'distill must be non-negative, got {self.distill}')
        self.distill_pairs = None
        self.distill_weights = None
        self.distill_scale = None
        self.distill_target = None
        self.distill_count = None
        if self.distill:
            self._build_distill(distill_warmup, distill_pairs, distill_weights)

        ## where z = 0 falls on each loop, in turns. Read at predict time only: the
        ## training forward pass keeps the base coords, so with the phases at zero the
        ## fit is bit for bit the run without this, and a solved phase is a
        ## reparameterisation of the same curve rather than a different one
        self.seam_align = float(seam_align)
        self.seam_every = int(seam_every)
        self.seam_start = int(seam_start)
        self.seam_steps = int(seam_steps)
        self.seam_lam = float(seam_lam)
        self.seam_lr = float(seam_lr)
        self.seam_mu = float(seam_mu)
        self.phase = None
        self.phase_optimizer = None
        self.seam_keys = None
        self.seam_pairs = None
        self.seam_weights = None
        if self.seam_align:
            if seam_pairs is None or not len(seam_pairs):
                raise ValueError('seam_align needs seam_pairs to align across')
            if self.seam_lam <= 0:
                raise ValueError(f'seam_lam must be positive, got {self.seam_lam}')
            if self.seam_lr <= 0:
                raise ValueError(f'seam_lr must be positive, got {self.seam_lr}')
            if self.seam_mu < 0:
                raise ValueError(f'seam_mu must be non-negative, got {self.seam_mu}')
            if self.seam_every < 0 or self.seam_steps < 1:
                raise ValueError(f'seam_every must be non-negative and seam_steps at '
                                 f'least 1, got {self.seam_every} and {self.seam_steps}')
            if self.seam_start < 0:
                raise ValueError(f'seam_start must be non-negative, got '
                                 f'{self.seam_start}')
            if seam_keys is not None:
                self.seam_keys = list(seam_keys)
                if len(self.seam_keys) != len(point_sets):
                    raise ValueError(f'one seam key per target: {len(point_sets)} '
                                     f'targets, {len(self.seam_keys)} keys')
            self.seam_pairs, self.seam_weights = self._pair_tensors(
                seam_pairs, seam_weights, len(point_sets), 'seam')
            self.phase = nn.Parameter(torch.zeros(len(point_sets),
                                                 device=self.device))
            self.phase_optimizer = torch.optim.Adam([self.phase], lr=self.seam_lr)
        elif seam_keys is not None:
            raise ValueError('seam_keys only mean something with seam_align set')

        ## one row of readings per epoch, split by slice axis, and one winding per
        ## loop per epoch. The epoch line is a mean over every loop, so a family that
        ## runs the other way round is invisible in it; these say which family and
        ## when it turned over
        self.epoch_stats = []
        self.winding_history = []
        self._last_reversed = None

    def _epoch_diagnostics(self, out, target, near, pair, d1, d2, axis):
        """Per-loop fit, direction, bend, winding and fold for one batch, by axis."""
        with torch.no_grad():
            near_t = target.gather(1, near[..., None].expand(-1, -1, 3))
            winding, fold = curve_winding(out, axis, centre=target.mean(1))
            loop = dict(fit=((out - near_t) ** 2).sum(-1).mean(1),
                        winding=winding, fold=fold)
            if d1 is not None and d2 is not None:
                loop['bend'] = ((d2 ** 2).sum(-1).mean(1)
                                / ((d1 ** 2).sum(-1).mean(1).detach()
                                   + 1e-12))
            if d1 is not None and pair is not None:
                loop['align'] = (1.0 - normal_cosine(d1, pair, axis)).mean(1)
        return {name: value.cpu().numpy() for name, value in loop.items()}

    def _build_distill(self, warmup, pairs, weights):
        """
        Read the teacher's seam profile at the shape's own conditions.

        The warm-up is a teacher of shape and orientation, and both are already in
        the student's weights. What the fit cannot see is where z = 0 falls: the
        chamfer, bend and normal terms are invariant to sliding a closed curve
        through its own parameter. The teacher can see it, so the teacher is read
        once at this shape's own queries and axes, and the per-pair seam distance D
        of its curves is the ceiling the student is held under: the term charges
        only the excess over the teacher, so it can lower D and never raise it.

        The teacher's field lives on the unit cube, so a shape query is carried into
        the teacher's own frame before it is sampled; the shape's frame and the
        teacher's then name the same cell, which is what makes the profile the one
        the student starts from. The curves themselves are never matched: only their
        D is, so a common deformation is free and shape and orientation stay with
        the fit. D is scale-free, so each side reads its own curve size -- the
        teacher draws at the warm-up's cube scale, the student at the shape's -- and
        the two relative disagreements are what is compared. At step 0 the curves
        are the same and the two D values are equal. The pair's fixed target radius
        only carries the dimensionless gap into chamfer units, so neither D can be
        lowered by shrinking a curve.

        warmup: path of the cube-warm-up checkpoint the teacher is built from
        pairs: P x 2 loop indices whose seams the profile is read over
        weights: one non-negative weight per pair
        """
        if self.feature_mode != 'triplane':
            raise ValueError('the seam distillation reads the teacher through the '
                             'triplane field; fit with feature_mode="triplane"')
        if warmup is None:
            raise ValueError('the seam distillation needs the warm-up checkpoint to '
                             'build its teacher; pass distill_warmup')
        try:
            ckpt = torch.load(warmup, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            raise ValueError(f'the seam distillation needs a warm-up checkpoint, and '
                             f'there is none at {warmup}') from None
        if ckpt.get('kind') != 'cube_warmup':
            raise ValueError(f'{warmup} is not a cube warm-up checkpoint')
        if bool(ckpt.get('axis_condition', False)) != self.model.axis_condition:
            raise ValueError(
                f'{warmup} was warmed '
                f'{"with" if ckpt.get("axis_condition") else "without"} axis '
                f'conditioning and this network is '
                f'{"axis-conditioned" if self.model.axis_condition else "not"}; '
                f'the teacher and the student must read the same condition')
        if int(ckpt['embedding_dim']) != self.model.embedding_dim:
            raise ValueError(
                f'the teacher in {warmup} supplies {int(ckpt["embedding_dim"])}-D '
                f'features and this network reads {self.model.embedding_dim}-D; '
                f'warm up with the same triplane channels')

        teacher = miniNN(1, num_neurons=int(ckpt['num_neurons']),
                         num_layers=int(ckpt['num_layers']),
                         pe=bool(ckpt['pe']), feature_mode='triplane',
                         triplane=dict(ckpt['triplane']),
                         axis_condition=bool(ckpt['axis_condition']))
        teacher.load_state_dict(ckpt['state_dict'])
        teacher.to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        pairs, weights = self._pair_tensors(
            pairs, weights, len(self.dataloader.point_sets), 'distill')
        ## the target scale is each loop's own mean squared radius on its points,
        ## fixed once: it carries the dimensionless D gap into chamfer units, so the
        ## term cannot be lowered by shrinking a curve
        radius = np.asarray([
            float(((p - p.mean(0)) ** 2).sum(-1).mean())
            for p in self.dataloader.point_sets])
        index = pairs.cpu().numpy()
        scale = torch.as_tensor(0.5 * (radius[index[:, 0]] + radius[index[:, 1]]),
                                dtype=torch.float32, device=self.device)

        ## the teacher's grid is laid on the unit cube, so the shape's normalized
        ## query is carried into the teacher's own frame before sampling
        triplane = self.model.triplane
        unit = (self.queries - triplane.box_centre) / triplane.box_half
        points = (teacher.triplane.box_centre
                  + unit * teacher.triplane.box_half)
        z = self.z[None].expand(len(self.queries), -1, -1)
        with torch.no_grad():
            feat = teacher.triplane(points)
            out = teacher(z, None, feat, self.axes)
            v = out - out.mean(1, keepdim=True)
            vi, vj = v[pairs[:, 0]], v[pairs[:, 1]]
            ## D is a ratio, so each side reads its own curve size: the teacher
            ## draws at the warm-up's cube scale, and a shape-target denominator
            ## would inflate its D wherever the shape loop is small
            t_radius = (v ** 2).sum(-1).mean(1)
            t_scale = 0.5 * (t_radius[pairs[:, 0]] + t_radius[pairs[:, 1]])
            target = (((vi - vj) ** 2).sum(-1).mean(1)
                      / t_scale.clamp_min(1e-12))

        self.distill_pairs = pairs
        self.distill_weights = weights
        self.distill_scale = scale
        self.distill_target = target.detach()
        self.distill_count = len(self.dataloader.point_sets)
        print(f'  seam distillation: teacher {warmup}, {len(pairs)} pairs over '
              f'{len(self.queries)} loops, teacher D mean '
              f'{float(target.mean()):.6f} max {float(target.max()):.6f}, '
              f'weight {self.distill:g}')

    def align_phases(self):
        """
        Move where z = 0 falls on every loop until neighbouring seams line up.

        The curves are read once from the network and detached, and every sample
        after that is a periodic linear interpolation of that fixed array, so the
        only thing the energy differentiates against is `self.phase`. The shape
        terms are therefore untouched by construction rather than by a small weight:
        the fitted curves come out of this exactly as they went in, drawn from a
        different starting point.

        The energy is a pair-weighted seam distance: for each pair the two seams are
        compared as they stand, D being their mean squared separation over the two
        loops' mean squared radius. A pair's vote is not the same for every pair,
        though: the loops are first matched at their best circular shift, and the
        distance they reach there says whether they are the same cross-section at
        all. A pair that only agrees after a shift is one where the seams are simply
        in the wrong place, and it votes at full weight; a pair that disagrees even
        at its best shift is two different loops -- a torso against a limb -- and
        votes at lam / (lam + D_best), which is nearly nothing. The pull itself is D
        and not the switched value, so a badly misplaced seam is pulled as hard as a
        slightly misplaced one rather than falling into the tail of a saturating
        function. The zero-mean prior on the phases fixes the one gauge the pairs
        cannot: every phase moved by the same amount changes no difference at all.
        """
        if self.phase is None:
            raise ValueError('align_phases needs seam_align and seam_pairs')
        count = len(self.dataloader.point_sets)
        n = self.z.shape[0]
        with torch.no_grad():
            idx = (None if self.feature_mode == 'triplane'
                   else torch.as_tensor(self.codes, device=self.device))
            feat = self.feature_values() if self.feature_mode == 'triplane' else None
            curves = self.model(self.z[None].expand(count, -1, -1),
                                idx, feat, self.axes).detach()
        ## the centroid is read before any phase is applied, so the centred curve a
        ## phase draws is the same set of points slid through the coords: on a dense
        ## loop the sampled mean barely moves, and fixing it keeps the distance about
        ## shape and seam rather than about where this sample set's centre landed
        centres = curves.mean(1, keepdim=True)

        def drawn(ph):
            u = torch.remainder(self.z[:, 0][None, :] + ph[:, None], 1.0) * n
            lo = torch.floor(u)
            k0 = lo.long() % n
            w = (u - lo)[..., None]
            c0 = curves.gather(1, k0[..., None].expand(-1, -1, 3))
            c1 = curves.gather(1, ((k0 + 1) % n)[..., None].expand(-1, -1, 3))
            return c0 * (1.0 - w) + c1 * w - centres

        def seam_distance(v, scale_grad=True):
            vi, vj = v[self.seam_pairs[:, 0]], v[self.seam_pairs[:, 1]]
            s2 = 0.5 * ((vi ** 2).sum(-1).mean(1) + (vj ** 2).sum(-1).mean(1))
            if not scale_grad:
                ## the normaliser is the pair's own size, which the phase does not
                ## change; letting it into the gradient would let a pair grow its way
                ## out of a mismatch
                s2 = s2.detach()
            return ((vi - vj) ** 2).sum(-1).mean(1) / s2.clamp_min(1e-12)

        with torch.no_grad():
            start = seam_distance(drawn(self.phase), scale_grad=False)
        ## how much each pair votes, read once: the circular cross-correlation of the
        ## two centred loops over every shift gives the separation they could reach if
        ## the seams were placed freely, so a pair of the same cross-section is told
        ## apart from a pair of two different loops without depending on where the
        ## phases happen to be when the solve starts
        with torch.no_grad():
            vi = curves[self.seam_pairs[:, 0]] - centres[self.seam_pairs[:, 0]]
            vj = curves[self.seam_pairs[:, 1]] - centres[self.seam_pairs[:, 1]]
            corr = torch.fft.irfft((torch.fft.rfft(vj, dim=1)
                                    * torch.fft.rfft(vi, dim=1).conj()).sum(-1),
                                   n=n, dim=1)
            var = (vi ** 2).sum(-1).mean(1) + (vj ** 2).sum(-1).mean(1)
            best = (var[:, None] - 2.0 * corr / n).min(1).values
            agreement = (best / (0.5 * var).clamp_min(1e-12)).clamp_min(0.0)
            weight = self.seam_weights * self.seam_lam / (self.seam_lam + agreement)
            weight = weight / weight.sum()
        for _ in range(self.seam_steps):
            self.phase_optimizer.zero_grad()
            d = seam_distance(drawn(self.phase))
            energy = (weight * d).sum()
            energy = energy + self.seam_mu * self.phase.pow(2).mean()
            (self.seam_align * energy).backward()
            self.phase_optimizer.step()
            ## the phases are angles in turns: a step is taken in the unwrapped
            ## parameterisation and then folded back, which the geometry cannot see
            with torch.no_grad():
                self.phase.copy_(torch.remainder(self.phase + 0.5, 1.0) - 0.5)
        with torch.no_grad():
            end = seam_distance(drawn(self.phase), scale_grad=False)
        print(f'  seam align {self.seam_steps} steps on {len(self.seam_pairs)} pairs: '
              f'D median {float(start.median()):.4f} -> {float(end.median()):.4f}, '
              f'phase median |p| {float(self.phase.abs().median()):.4f} turns')
        return float(end.median())

    def _pair_tensors(self, pairs, weights, count, what):
        """Validate a P x 2 pair list and its weights, as device tensors."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.min() < 0 or pairs.max() >= count:
            raise ValueError(f'{what} pairs name entries outside 0..{count - 1}, '
                             f'got {pairs.min()}..{pairs.max()}')
        weights = (np.ones(len(pairs), dtype=np.float32) if weights is None
                   else np.asarray(weights, dtype=np.float32).reshape(-1))
        if len(weights) != len(pairs):
            raise ValueError(f'one weight per {what} pair: {len(pairs)} pairs, '
                             f'{len(weights)} weights')
        if not np.all(weights >= 0.0) or weights.sum() <= 0.0:
            raise ValueError(f'{what} pair weights must be non-negative and not all '
                             f'zero')
        return (torch.as_tensor(pairs, dtype=torch.long, device=self.device),
                torch.as_tensor(weights, dtype=torch.float32, device=self.device))

    def feature_values(self, ids=None):
        """The field feature of every target, or of the targets named by ids."""
        if self.feature_mode != 'triplane':
            raise ValueError('feature_values only exists in triplane mode')
        feat = self.model.triplane(self.queries)
        if ids is not None:
            feat = feat[torch.as_tensor(ids, dtype=torch.long, device=self.device)]
        return feat


    def train(self, epochs=1, every=None, on_epoch=None):
        """
        One pass over the sampled batches per epoch; returns the mean loss per epoch.

        The cosine anneal spans this call, so building it here rather than in
        __init__ is what makes T_max the epochs actually run.

        The direction term's weight is annealed over the same call: it starts at
        `align` on the first batch and is down to nothing on the last, by half a
        cosine. The term is there to settle which way round each loop runs, and that
        is decided while the fit is still coarse -- early, two stretches of a contour
        passing close are told apart by the way they run rather than by where they
        are. Later it can only pull a fit that already sits on its points, so its
        weight is spent where it is needed and given back afterwards. The printed
        `align` value is the term itself, unweighted, so the columns stay comparable
        across epochs; the weight it was taken at is printed beside it.

        every, on_epoch: `on_epoch(n)` is called after epoch n (counted from 1)
        whenever n is a multiple of `every`, which is where a caller watches a run
        rather than only judging it: the state handed over is the one the epoch
        printed, so a picture taken there is the picture of that line. Both default
        to None, which watches nothing.
        """
        ## a second call would otherwise anneal from the dead eta_min the first one
        ## ended on, leaving every later step with a frozen learning rate
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=epochs, eta_min=1e-6)
        ## the anneal counts batches, not epochs, so it is smooth within one and a run
        ## of one epoch reaches the end of the schedule just as a run of fifty does
        total = epochs * len(self.dataloader)
        step = 0
        history = []
        for e in range(epochs):
            losses, fits, bends, aligns, weights = [], [], [], [], []
            code_pens, tvs, distills = [], [], []
            diag = None
            if self.axes is not None:
                count = len(self.dataloader.point_sets)
                diag = {name: np.zeros(count) for name in
                        ('fit', 'winding', 'fold')}
                diag_seen = np.zeros(count)
            for ids, target, n_target in self.dataloader.index_batches():
                self.optimizer.zero_grad()

                target = torch.from_numpy(target).float().to(self.device)      # B x M x 3
                ## the row is each point set's own feature code, so a target keeps its
                ## curve however the list around it is rebuilt; every sample shares the
                ## same coords, so one call covers the batch
                idx = torch.as_tensor(self.codes[ids], device=self.device)
                ## the slice axis each target was cut on, by its own id: the network's
                ## condition and the direction term read the same vector
                ids_t = torch.as_tensor(ids, device=self.device)
                axis = None if self.axes is None else self.axes[ids_t]
                feat = self.feature_values(ids) if self.feature_mode == 'triplane' else None
                ## a leaf of its own every step: the penalties differentiate with
                ## respect to the coords, so they have to be a graph input, and a view
                ## of the shared self.z would pile every step's gradient onto one leaf
                z = self.z.expand(target.shape[0], -1, -1).clone().requires_grad_(True)
                out = self.model(z, idx, feat, axis)       # B x N x 3
                fit, near = self.loss_fn(out, target)
                loss = fit
                ## one set of backward passes serves both penalties when they are on
                d1 = curve_tangent(out, z) if self.smooth or self.align else None
                d2 = curve_curvature(d1, z) if self.smooth else None
                pair = None
                if self.smooth:
                    ## both derivatives of the curve at the very coords it is drawn at,
                    ## so the term acts on the curve the fit figure shows
                    bend = bend_penalty(d1, d2)
                    loss = loss + self.smooth * bend
                    bends.append(bend.item())
                if self.align:
                    ## the outward normal of the target point chamfer matched to each
                    ## predicted one, against the direction the curve travels there:
                    ## the normals pick out one way round the loop, and two stretches
                    ## of a contour passing close are told apart by the way they run,
                    ## not by where they are
                    n_t = torch.from_numpy(n_target).float().to(self.device)
                    pair = n_t.gather(1, near[..., None].expand(-1, -1, 3))
                    ## the plane normal is the one of the slice each target was cut on,
                    ## so it is read by the target's own id and not by its row
                    align = normal_penalty(d1, pair, axis)
                    ## --normal is the weight it starts at, not the weight it keeps
                    weight = self.align * cosine_anneal(step, total)
                    loss = loss + weight * align
                    aligns.append(align.item())
                    weights.append(weight)

                if diag is not None:
                    ## read off the same curves and the same pairing the step just
                    ## used, so the record is the step's own state rather than a new
                    ## draw of it
                    reading = self._epoch_diagnostics(out, target, near, pair,
                                                      d1, d2, axis)
                    where = np.asarray(ids)
                    diag_seen[where] += 1
                    for name, value in reading.items():
                        if name not in diag:
                            diag[name] = np.zeros(count)
                        diag[name][where] += value

                if self.code_smooth and self.feature_pairs is not None:
                    ## the field is a function of position, so two nearby loops read
                    ## overlapping cells and this term holds those samples together;
                    ## gradients reach the grid rather than a row table. A pair's
                    ## weight falls with its in-plane distance, so a close neighbour
                    ## is held and a distant one is free to differ
                    all_feat = self.feature_values()
                    diff = (all_feat[self.feature_pairs[:, 0]]
                            - all_feat[self.feature_pairs[:, 1]])
                    code_pen = (self.feature_weights *
                                diff.pow(2).sum(1)).sum() / self.feature_weights.sum()
                    loss = loss + self.code_smooth * code_pen
                    code_pens.append(code_pen.item())
                elif self.code_smooth and self.code_pairs is not None:
                    ## the rows are shared by every sample, so this is one term over
                    ## the table rather than one per batch element. A pair's weight
                    ## falls with its in-plane distance, so a close neighbour is held
                    ## to its code and a distant one is free to differ
                    table = self.model.embedding.weight
                    diff = table[self.code_pairs[:, 0]] - table[self.code_pairs[:, 1]]
                    code_pen = (self.code_weights *
                                diff.pow(2).sum(1)).sum() / self.code_weights.sum()
                    loss = loss + self.code_smooth * code_pen
                    code_pens.append(code_pen.item())

                if self.feature_tv and self.feature_mode == 'triplane':
                    tv = self.model.triplane.tv_penalty()
                    loss = loss + self.feature_tv * tv
                    tvs.append(tv.item())

                if self.distill:
                    ## the student's own seam profile, read on the curves this step
                    ## already drew and paired only where both loops are in the
                    ## batch. Holding it under the teacher's ceiling leaves the curves
                    ## themselves free: a common deformation, or a rotation, costs
                    ## nothing, and only the relative seam placement is held
                    position = np.full(self.distill_count, -1, dtype=np.int64)
                    position[np.asarray(ids)] = np.arange(len(ids))
                    local = position[self.distill_pairs.cpu().numpy()]
                    keep = (local[:, 0] >= 0) & (local[:, 1] >= 0)
                    if keep.any():
                        li = torch.as_tensor(local[keep, 0], dtype=torch.long,
                                             device=self.device)
                        lj = torch.as_tensor(local[keep, 1], dtype=torch.long,
                                             device=self.device)
                        v = out - out.mean(1, keepdim=True)
                        ## D is a ratio, so the student reads its own curve size
                        ## just as the teacher does; with the teacher's curves
                        ## reproduced at step 0 the two D values are then equal
                        r = (v ** 2).sum(-1).mean(1)
                        own = 0.5 * (r[li] + r[lj])
                        seen = (((v[li] - v[lj]) ** 2).sum(-1).mean(1)
                                / own.clamp_min(1e-12))
                        w = self.distill_weights[keep]
                        ## the teacher's D is a ceiling, not a target: carry the
                        ## dimensionless excess back to a squared length by the
                        ## pair's fixed target radius before it competes with the fit
                        scale = self.distill_scale[keep]
                        gap = (seen - self.distill_target[keep]).clamp_min(0.0)
                        hold = (w * scale * gap).sum() / w.sum()
                        loss = loss + self.distill * hold
                        distills.append(hold.item())

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                fits.append(fit.item())
                step += 1
            # self.scheduler.step()
            # lr = self.scheduler.get_last_lr()[0]
            ## the terms separately: a weight is chosen by what it costs in the fit
            ## against what it buys in smoothness, which one total cannot show
            parts = [f'fit {np.mean(fits):.4f}']
            if bends:
                parts.append(f'bend {np.mean(bends):.1f}')
            if aligns:
                parts.append(f'align {np.mean(aligns):.3f} '
                             f'(w {np.mean(weights):.4f})')
            if code_pens:
                ## the same term over either condition: a row table in embedding
                ## mode, the sampled field vectors in triplane mode
                parts.append(f'{"feat" if self.feature_mode == "triplane" else "code"} '
                             f'{np.mean(code_pens):.4f}')
            if tvs:
                parts.append(f'tv {np.mean(tvs):.5f}')
            if distills:
                parts.append(f'distill {np.mean(distills):.6f}')
            print(f' {e} epoch: loss: {np.mean(losses):.4f} ({", ".join(parts)})')
            history.append(float(np.mean(losses)))
            if diag is not None:
                ## the axis a loop belongs to is the plane it was cut on, so the
                ## labels come from the trainer's own axes, not from a name in the
                ## data that a caller could order differently
                diag_seen_safe = np.maximum(diag_seen, 1)
                for name in diag:
                    diag[name] = np.where(diag_seen > 0,
                                          diag[name] / diag_seen_safe, np.nan)
                labels = self.axes.abs().argmax(-1).cpu().numpy()
                per_axis = {}
                for a, name in enumerate(('x', 'y', 'z')):
                    mask = labels == a
                    if not mask.any():
                        continue
                    row = dict(loops=int(mask.sum()),
                               fit=float(np.nanmean(diag['fit'][mask])),
                               winding=float(np.nanmean(diag['winding'][mask])),
                               fold=float(np.nanmean(diag['fold'][mask])),
                               reversed=int(np.nansum(diag['winding'][mask] < 0)))
                    if 'align' in diag:
                        row['align'] = float(np.nanmean(diag['align'][mask]))
                    if 'bend' in diag:
                        row['bend'] = float(np.nanmean(diag['bend'][mask]))
                    per_axis[name] = row
                self.epoch_stats.append(dict(epoch=e + 1,
                                             loss=float(np.mean(losses)),
                                             axes=per_axis))
                ## kept per loop rather than per axis: which loop turned over, and at
                ## which epoch, is the question this record exists to answer
                self.winding_history.append(diag['winding'].copy())
                now = tuple(per_axis.get(name, {}).get('reversed', 0)
                            for name in ('x', 'y', 'z'))
                if now != self._last_reversed:
                    print(f'   loops wound the other way round (winding < 0) '
                          f'x/y/z: {now[0]}/{now[1]}/{now[2]}')
                    self._last_reversed = now
            ## the phases follow the curves: a solve is cheap and starts where the
            ## last one left, so re-solving on the way through keeps the seam where
            ## the current shape puts it. Before the watcher, so a picture of epoch n
            ## is drawn with the phases that epoch ended on
            if (self.phase is not None and self.seam_every
                    and e + 1 >= self.seam_start
                    and (e + 1) % self.seam_every == 0):
                self.align_phases()
            ## after the line it belongs to, and at the last epoch too: a watcher gets
            ## the run at 50, 100, ... including whatever count the run ends on, so
            ## the pictures stand in order without a hole where the last one fell
            if on_epoch is not None and every and (e + 1) % every == 0:
                on_epoch(e + 1)
        return history

    def inherit(self, previous, source=None):
        """
        Continue from another trainer's fit: its shared network, and every embedding
        row the two tables have in common.

        previous: the trainer the stage before this one ended on
        source: one row of `previous` per row here, or -1 for a row with no
                  counterpart, which keeps the zero feature. None is the plane-code
                  case: a row is a plane's code and codes are only ever appended, so
                  row i here is row i there, and the table of this trainer starts
                  with the whole of the last one

        The embedding table is the one thing that cannot be copied whole, because a
        refinement stage has rows the stage before it did not. When rows are handed
        out per plane code and never moved, the rows of the earlier stage are a prefix
        of the later one and this is just a copy of that prefix: no loop is matched up
        by position or by where its points sit, so nothing can be handed the wrong
        curve when a stage inserts planes among the ones already cut.

        Nothing else moves: the optimiser was built on these parameters and they are
        copied into in place, but its moments start again, so a stage is a fresh set of
        steps rather than a continuation of the last stage's.
        """
        if self.feature_mode != previous.feature_mode:
            raise ValueError('the two networks condition on features in different '
                             f'ways ({previous.feature_mode} and {self.feature_mode})')
        ## where z = 0 falls belongs to the loop, not to a row: the stages are matched
        ## by seam key, which names the plane and the slot on it and survives planes
        ## being inserted around it. A loop this stage did not have last time starts at
        ## zero, and is moved by the first solve
        if self.phase is not None and self.seam_keys:
            carried = {}
            if previous.phase is not None and previous.seam_keys:
                carried = {key: float(previous.phase[i])
                           for i, key in enumerate(previous.seam_keys)}
            with torch.no_grad():
                for i, key in enumerate(self.seam_keys):
                    if key in carried:
                        self.phase[i] = carried[key]
            print(f'  {sum(1 for k in self.seam_keys if k in carried)} of '
                  f'{len(self.seam_keys)} loops carry their seam phase over')
        if self.feature_mode == 'triplane':
            ## the field is a function of position, so there is no row to match up:
            ## copying the grids and the decoder copies the whole condition, and a
            ## loop on a plane both stages cut reads the same feature at the same
            ## query point. Added planes need no seed for the same reason.
            if previous.model.triplane.grid.shape != self.model.triplane.grid.shape:
                raise ValueError('the two triplanes have different grid shapes '
                                 f'({tuple(previous.model.triplane.grid.shape)} and '
                                 f'{tuple(self.model.triplane.grid.shape)})')
            self.model.load_state_dict(previous.model.state_dict())
            print(f'  {len(self.codes)} of {len(self.codes)} loops read the inherited '
                  f'triplane; added planes need no row seed')
            return self
        if previous.model.embedding.embedding_dim != self.model.embedding.embedding_dim:
            raise ValueError('the two networks condition on features of different '
                             f'length ({previous.model.embedding.embedding_dim} and '
                             f'{self.model.embedding.embedding_dim})')
        rows = previous.model.embedding.weight.detach()
        if source is None:
            ## row i is row i: what this table is short of is the planes this stage
            ## added, which are the rows past the end of the last one
            source = np.arange(self.num_models)
        else:
            source = np.asarray(source, dtype=np.int64).reshape(-1)
            if len(source) != self.num_models:
                raise ValueError(f'one source per row: {self.num_models} rows, '
                                 f'{len(source)} named')
            if source.max(initial=-1) >= len(rows):
                raise ValueError(f'source names row {int(source.max())} of a trainer '
                                 f'that has {len(rows)}')
        shared = {name: value for name, value in previous.model.state_dict().items()
                  if name != 'embedding.weight'}
        ## strict off, since the one table left out on purpose is the one that differs
        self.model.load_state_dict(shared, strict=False)
        have = np.zeros(self.num_models, dtype=bool)
        with torch.no_grad():
            for i, src in enumerate(source):
                if -1 < src < len(rows):
                    self.model.embedding.weight[i] = rows[src]
                    have[i] = True
        ## counted over the loops, not the rows: the rows without a loop here are the
        ## codes of planes this plane set does not cut, and they are not a fit either
        kept = int(have[self.codes].sum())
        print(f'  {kept} of {len(self.codes)} loops carry their fit over from the '
              f'previous plane set, {len(self.codes) - kept} start from a square')
        return self

    def phase_coords(self, i=None):
        """
        The parameter coords point set i is drawn at, with its seam phase applied.

        This is the one place the phase is read into a prediction, so every caller
        -- the fit report, the normals, the exported loops -- draws the same curve
        from the same starting point. Point set None is the base curve, which has
        no seam to move.
        """
        if i is None or self.phase is None:
            return self.z
        return torch.remainder(self.z + self.phase[i], 1.0)

    def predict(self, i=None):
        """
        N x 3 points the network predicts for point set i, or for the base curve when
        i is None.
        """
        if i is None:
            idx, feat = None, None
        elif self.feature_mode == 'triplane':
            idx, feat = None, self.feature_values([i])
        else:
            idx = torch.as_tensor([self.codes[i]], device=self.device)
            feat = None
        axis = None
        if self.axes is not None:
            if i is None:
                raise ValueError('an axis-conditioned network has no base curve: '
                                 'pass the loop whose slice axis to read')
            axis = self.axes[i:i + 1]
        with torch.no_grad():
            return self.model(self.phase_coords(i), idx, feat, axis)[0]

    def predict_tangent(self, i=None):
        """
        N x 3 analytic tangent of the curve predicted for point set i, or for the base
        curve when i is None.

        This is d(prediction)/dz at the same parameter values the points are drawn
        at, so it is the exact tangent of the fitted curve rather than a central
        difference between neighbouring points.  Its length is not thrown away when
        predict_with_normals turns it into an in-plane normal: it becomes that
        normal's length, which is what keeps the drawn arrows in proportion.  Read on
        the warm-up's own curve it is how fast that curve travels per step of the
        parameter, which the square says should be its own perimeter.
        """
        if i is None:
            idx, feat = None, None
        elif self.feature_mode == 'triplane':
            idx, feat = None, self.feature_values([i])
        else:
            idx = torch.as_tensor([self.codes[i]], device=self.device)
            feat = None
        axis = None
        if self.axes is not None:
            if i is None:
                raise ValueError('an axis-conditioned network has no base curve: '
                                 'pass the loop whose slice axis to read')
            axis = self.axes[i:i + 1]
        ## predict() runs under no_grad, but a derivative needs the graph, so this
        ## repeats the forward pass with a fresh leaf. The explicit batch axis keeps
        ## curve_tangent's result 1 x N x 3; a bare N x 1 coord drops that axis. The
        ## coords carry the loop's seam phase, so the tangent is read at the same
        ## points predict draws rather than one loop's worth further round
        z = self.phase_coords(i)[None].clone().requires_grad_(True)
        return curve_tangent(self.model(z, idx, feat, axis), z)[0].detach().cpu().numpy()

    def predict_with_normals(self, i, axis=None):
        """
        N x 3 predicted points and their N x 3 in-plane normals.

        The network predicts points, not normals, so the normal is read off the fitted
        curve: the curve lies in its slice, which leaves one in-plane direction square
        to its tangent, and the quarter turn is taken in the sense the direction term
        asks the tangent for, so the arrow is the surface normal the curve implies
        rather than its negation. The order of the cross product is that whole choice:
        crossing the tangent with the plane normal is one quarter turn, crossing the
        plane normal with the tangent is the same turn the other way, and the direction
        term asks for the tangent to be axis x normal. A quarter turn applied twice is
        a half turn, axis x (axis x t) = -t, so the surface normal is t x axis; the
        other order turns the tangent the wrong way for a surface the curve is a
        contour of.

        That length is left exactly as the turn gives it rather than scaled to one: the
        curve runs in its slice, so it is |tangent|, which is how far the fit travels
        per step of the parameter. A loop drawn larger, or one the network sweeps
        faster, therefore carries longer vectors, and only a tangent lifted out of the
        plane shortens them. On a fit that satisfies the direction term this is the
        cloud's own outward normal; a loop running against it draws inward with it,
        which is the state that term is there to correct.

        i: point set index, or None for the curve conditioned on no feature at all
        axis: plane normal of that set's slice, in the prediction's frame; required,
                 since nothing else in a bare point list says which plane it is in
        """
        if axis is None:
            raise ValueError('predict_with_normals needs the plane normal to cross with')
        points = self.predict(i).cpu().numpy()
        tangent = self.predict_tangent(i)
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        if len(points) < 2:
            return points, np.zeros_like(points)
        return points, np.cross(tangent, axis[None, :])

    def bend(self):
        """
        The bend penalty over every fitted curve as it now stands, all sets at once.

        A circle reads 39.5, so the number says how many times round the fitted curves
        run in the course of one, which is the thing the weight is buying: it is the
        only way to tell a straightened fit from an equally well fitting wiggly one.
        """
        idx = torch.as_tensor(self.codes, device=self.device)
        feat = self.feature_values() if self.feature_mode == 'triplane' else None
        z = self.z.expand(len(idx), -1, -1).clone().requires_grad_(True)
        ## a derivative needs the graph, so this cannot sit under no_grad like predict;
        ## the graph is dropped with the return value
        d1 = curve_tangent(self.model(z, idx, feat, self.axes), z)
        return float(bend_penalty(d1, curve_curvature(d1, z)))

    def nearest_distance(self, points, target):
        """Distance from each row of points to the nearest row of target."""
        p = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        ## chunked: a full slice against M predicted points does not fit as one cdist
        return torch.cat([torch.cdist(chunk, t).min(1).values
                          for chunk in p.split(2048)]).cpu().numpy()

    ## blue -> cyan -> yellow -> red, so a good fit reads blue and a bad one red
    RAMP = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]], np.float64)

    def error_colors(self, errors):
        """RGBA colours for an error array, ramped over its own range."""
        span = errors.max() - errors.min()
        pos = (errors - errors.min()) / (span if span > 0 else 1) * (len(self.RAMP) - 1)
        lo = np.floor(pos).astype(int)
        frac = (pos - lo)[:, None]
        rgb = (self.RAMP[lo] * (1 - frac)
               + self.RAMP[np.minimum(lo + 1, len(self.RAMP) - 1)] * frac)
        return np.concatenate([rgb, np.full((len(errors), 1), 255)], axis=1).astype(np.uint8)


def save_model(path, trainer, **meta):
    """Write the fitted network so it can be replayed without the training run.

    The file holds the shared MLP, the condition it reads (embedding table or
    triplane grid), the parameter grid length, the per-loop codes/queries and the
    seam phase each loop was drawn from. Anything a caller needs to locate the
    curves (keys, plane normals and plane points, target counts) is passed through
    `meta` and stored beside them.
    """
    tp = getattr(trainer.model, 'triplane', None)
    triplane = None
    if tp is not None:
        triplane = dict(centre=tp.box_centre.detach().cpu().numpy(),
                        half=tp.box_half.detach().cpu().numpy(),
                        resolution=tp.resolution, channels=tp.channels, tv=tp.tv)
    payload = dict(
        feature_mode=trainer.feature_mode,
        num_neurons=int(trainer.model.model[0].out_features),
        num_layers=int(trainer.model.num_layers),
        pe=bool(trainer.model.pe),
        embedding_dim=int(trainer.model.embedding_dim),
        axis_condition=bool(trainer.model.axis_condition),
        num_models=int(trainer.num_models),
        latent_points=int(trainer.z.shape[0]),
        codes=np.asarray(trainer.codes, dtype=np.int64),
        queries=(None if trainer.queries is None
                 else trainer.queries.detach().cpu().numpy()),
        axes=(None if trainer.axes is None
              else trainer.axes.detach().cpu().numpy()),
        triplane=triplane,
        phases=(None if trainer.phase is None
                else trainer.phase.detach().cpu().numpy()),
        state_dict={k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
    )
    payload.update(meta)
    torch.save(payload, path)
    return path


def load_model(path, device='cpu'):
    """Rebuild the network `save_model` wrote, as (model, meta) with meta on CPU.

    The returned model is in eval mode and needs only the parameter grid and a
    feature: `z = torch.linspace(0, 1, meta['latent_points'] + 1)[:-1, None]`.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_layers = ckpt.get('num_layers', 2)
    if ckpt['feature_mode'] == 'triplane':
        model = miniNN(1, num_neurons=ckpt['num_neurons'], num_layers=num_layers,
                       pe=ckpt['pe'], feature_mode='triplane',
                       triplane=ckpt['triplane'],
                       axis_condition=bool(ckpt.get('axis_condition', False)))
    else:
        model = miniNN(ckpt['num_models'], num_neurons=ckpt['num_neurons'],
                       num_layers=num_layers, feat_dim=ckpt['embedding_dim'],
                       pe=ckpt['pe'], feature_mode='embedding')
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != 'state_dict'}
    meta['device'] = device
    return model, meta