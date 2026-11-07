"""Train the cube warm-up and save the checkpoint a shape's fit starts from.

The warm-up sees a unit cube and nothing else. Its loops are the cube's
cross-sections -- exact unit squares on the planes it is cut by -- and the network
is trained so that every loop runs the same way round, puts z = 0 where its
neighbouring loops put theirs, and lies on the cube. No cloud, no bounding box and
no shape is read: the cube is `mini_nn.cube_loops`, a definition.

The checkpoint holds the shared MLP and the triplane grid. `rosa_main` loads it as
the initialisation of every run (`--warmup`, on by default), so this is run once,
not per shape:

    python warmup_cube.py --out warmup/cube_warmup.pt

Warm with the network and grid the fit uses: `--neurons`, `--layers`,
`--resolution` and `--channels` have to match `rosa_main`'s, or the load is
refused rather than applied in part.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import mini_nn


def build_parser():
    parser = argparse.ArgumentParser(
        description='Warm a triplane network on the unit cube and save the '
                    'checkpoint rosa_main loads.')
    parser.add_argument('--steps', type=int, default=300,
                        help='optimisation steps; every cube loop is drawn in every '
                             'step')
    parser.add_argument('--planes', type=int, default=4,
                        help='even planes per axis on the cube, so the seam term '
                             'has 3 * (planes - 1) neighbouring pairs; the exact '
                             'centre is skipped, where the three centre cuts would '
                             'share one query')
    parser.add_argument('--samples', type=int, default=512,
                        help='points round every cube loop, and the parameter grid the '
                             'curve is drawn at')
    parser.add_argument('--normal', dest='align', type=float, default=1.0,
                        help='weight of the direction term, held against the cube\'s '
                             'outward face normals; 0 trains position and seam only')
    parser.add_argument('--seam', type=float, default=1.0,
                        help='weight of the seam term, the distance D between loops '
                             'adjacent along an axis, differentiated through the '
                             'network; 0 leaves z = 0 free')
    parser.add_argument('--lr', type=float, default=3e-2,
                        help='starting learning rate, cosine-annealed over the run')
    parser.add_argument('--neurons', type=int, default=64,
                        help='width of the shared network; must match the fit')
    parser.add_argument('--layers', type=int, default=2,
                        help='hidden layers; must match the fit')
    parser.add_argument('--resolution', type=int, default=32,
                        help='triplane cells per side; must match the fit')
    parser.add_argument('--channels', type=int, default=16,
                        help='triplane channels per grid; must match the fit')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for the initial weights and the grid')
    parser.add_argument('--out', default='warmup/cube_warmup_axis.pt',
                        help='checkpoint path; rosa_main loads '
                             'warmup/cube_warmup_axis.pt by default')
    return parser


def main(args=None):
    args = build_parser().parse_args(args)
    if args.steps < 1:
        raise SystemExit(f'--steps must be at least 1, got {args.steps}')
    if args.planes < 2 or args.planes % 2:
        raise SystemExit(f'--planes must be even and at least 2, got {args.planes}')
    if args.samples < 4:
        raise SystemExit(f'--samples must be at least 4, got {args.samples}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ## the grids live on the unit cube itself: the warm-up's object and the field's
    ## domain are the same object. A fit later lays this cube on its cloud's box.
    frame = dict(centre=np.zeros(3), half=np.full(3, mini_nn.UNIT_CUBE_HALF),
                 resolution=args.resolution, channels=args.channels, tv=0.0)
    model = mini_nn.miniNN(1, num_neurons=args.neurons, num_layers=args.layers,
                           feature_mode='triplane', triplane=frame,
                           axis_condition=True).to(device)
    loops = 3 * args.planes
    print(f'cube warm-up: {loops} loops ({args.planes} per axis), {args.samples} '
          f'samples each, axis-conditioned {args.layers}x{args.neurons} net, '
          f'{args.resolution} x {args.resolution} x 3 grids of {args.channels} '
          f'channels, on {device}')

    report = mini_nn.warm_on_cube(model, steps=args.steps, lr=args.lr,
                                  align=args.align, seam=args.seam,
                                  planes=args.planes, samples=args.samples)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mini_nn.save_warmup(out, model, seed=args.seed, lr=args.lr,
                        align=args.align, seam=args.seam, warmup=report)
    print(f'  checkpoint -> {out}')
    print(f'  load it in a fit with --warmup {out} (the default path); '
          f'--no-warmup fits from a blank network')
    return report


if __name__ == '__main__':
    main()
