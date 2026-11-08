# Agent notes

Read [PROJECT_STATE.md](PROJECT_STATE.md) before starting work, then
[README_ROSA.md](README_ROSA.md) and the tail of [todo.md](todo.md). The state
file is the shared, cross-device summary; `todo.md` is the full experiment log.

## Environment

Use `conda run -n seawall python ...` or
`/home/lyangiw/anaconda3/envs/seawall/bin/python`. Plain `python3` fails on
`libcudnn.so.8` on this machine.

## Standing method rules

- Loop orientation must be learned by the network. Never flip loops per loop or
  on export, and never replace the direction term with a gauge fix.
- Keep the published objective and all of its terms. New regularisation is
  opt-in, and side terms must not move the fit they are meant to observe.
- Validate a change on the shape panel with matched controls, plus a fresh
  shape; one shape is not a result. Quote the ~4e-5 training nondeterminism
  floor when comparing seeded runs.
- Read direction disagreements together with signed winding, not alone.

## Documentation

- Append every experiment to `todo.md`: command, table, figures, caveats, next
  test. Do not rewrite earlier entries.
- Update `PROJECT_STATE.md` (date, branch/commit, results snapshot) in the same
  commit, then push so the other device sees it on `git pull`.

## Not in git

`log/`, `*.ply` (including the input datasets) and `*.pt` (including the warm-up
checkpoint) are gitignored. Copy them between devices separately.
