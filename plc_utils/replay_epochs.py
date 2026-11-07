"""Replay a run's first two epochs and keep the states.

The run writes one checkpoint, at the end, so the curves at epochs 1 and 2 are
not on disk.  This re-runs the same command -- same data, same warm-up, same
seed, and the same 1000-epoch schedule, since the anneal spans the call -- and
records the model after each of the first two epochs, then stops.  The schedule
is why this trains under epochs=1000 and is cut short rather than being asked
for two epochs: a two-epoch call would anneal the direction term to nothing by
its second update, which is not what the run did.

  python plc_utils/replay_epochs.py [out_dir]

The default out dir is log/axis_seam10_horse_repro/replay; the saved states are
what `plot_epoch_matching.py --state e1=... --state e2=...` reads.
"""
import sys
from pathlib import Path

import torch

out = Path(sys.argv[1] if len(sys.argv) > 1 else
           'log/axis_seam10_horse_repro/replay')
out.mkdir(parents=True, exist_ok=True)

sys.argv = [
    'rosa_main.py',
    '--path', 'normalized_shapes_cuboid/horse_points_cubic_lam3.ply',
    '--train',
    '--epochs', '1000',
    '--samples', '29',
    '--normal', '0.01',
    '--distill', '10',
    '--distill-warmup', 'warmup/cube_warmup_axis.pt',
    '--out', str(out),
    '--snapshot-every', '0',
]

import _paths  # noqa: E402
import mini_nn
import rosa_main


class Stop(Exception):
    pass


original = mini_nn.Trainer.train


def train(self, epochs=1, every=None, on_epoch=None):
    def wrap(n):
        if on_epoch is not None:
            on_epoch(n)
        state = {k: v.detach().cpu().clone()
                 for k, v in self.model.state_dict().items()}
        torch.save(state, out / f'state_e{n}.pt')
        print(f'  saved {out / f"state_e{n}.pt"}')
        if n >= 2:
            raise Stop()
    return original(self, epochs, every, wrap)


mini_nn.Trainer.train = train
try:
    rosa_main.main()
except Stop:
    print('stopped after epoch 2')
