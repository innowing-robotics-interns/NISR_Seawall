#!/usr/bin/env python3
"""Export the latest NISR_Seawall checkpoint as checkerboard-textured patches."""

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


def find_repo_root(start):
    for path in (start, *start.parents):
        if (path / "utils" / "patch_vis.py").is_file():
            return path
    return None


def checkpoint_epoch(path):
    match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir):
    checkpoints = [p for p in run_dir.glob("checkpoint*.pt") if p.is_file()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: (p.stat().st_mtime, checkpoint_epoch(p)))


def matching_run_dirs(logs_root, query):
    seen = set()
    query = (query or "").lower()
    for checkpoint in logs_root.rglob("checkpoint*.pt"):
        run_dir = checkpoint.parent
        if run_dir in seen:
            continue
        seen.add(run_dir)
        if query and query not in str(run_dir).lower():
            continue
        yield run_dir


def pick_run(logs_root, query):
    best = None
    for run_dir in matching_run_dirs(logs_root, query):
        checkpoint = latest_checkpoint(run_dir)
        if checkpoint is None:
            continue
        key = checkpoint.stat().st_mtime
        if best is None or key > best[0]:
            best = (key, run_dir, checkpoint)
    return best


def training_running(run_dir):
    marker = run_dir.name
    proc = Path("/proc")
    if not proc.is_dir():
        return False
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="ignore"
            )
        except OSError:
            continue
        if "main.py" in cmdline and marker in cmdline:
            return True
    return False


def resolve_python(explicit):
    candidates = [
        explicit,
        __import__("os").environ.get("NISR_SEAWALL_PYTHON"),
        str(Path.home() / "anaconda3" / "envs" / "seawall" / "bin" / "python"),
        sys.executable,
        shutil.which("python3"),
        shutil.which("python"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def exported_paths(out_dir, checkpoint):
    stem = checkpoint.stem
    folder = out_dir / stem
    return (
        folder,
        folder / f"{stem}_checkerboard_{stem}.ply",
        folder / f"{stem}_svd.ply",
    )


def format_size(path):
    if not path.is_file():
        return "not written"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return ""


def summarize(folder, checkpoint):
    stem = checkpoint.stem
    patch_plys = sorted(folder.glob(f"patch_*_*_{stem}.ply"))
    depths = Counter()
    for path in patch_plys:
        match = re.search(r"_d(\d+)_", path.name)
        if match:
            depths[int(match.group(1))] += 1
    depth_text = " ".join(f"d{depth}={count}" for depth, count in sorted(depths.items()))
    _, combined_ply, svd_ply = exported_paths(folder.parent, checkpoint)
    print(
        f"  Patches     : {len(patch_plys)}"
        + (f" ({depth_text})" if depth_text else "")
    )
    print(f"  Combined PLY: {combined_ply} ({format_size(combined_ply)})")
    print(f"  SVD PLY     : {svd_ply} ({format_size(svd_ply)})")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the latest checkpoint of the newest NISR_Seawall run."
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name substring (e.g. xyzrgb_dragon) or path to a run/checkpoint.",
    )
    parser.add_argument("--logs-root", default=None, help="Root of run folders (default: <repo>/logs).")
    parser.add_argument("--out-dir", default=None, help="Export root (default: the run folder).")
    parser.add_argument("--n-images", type=int, default=4, help="Checkerboard textures to load.")
    parser.add_argument("--resolution", type=int, default=None, help="Per-patch UV resolution.")
    parser.add_argument(
        "--subdivision-depth",
        default="-1",
        help="Passed to patch_vis.py; -1 samples every UV (default), 0 keeps active patches only.",
    )
    parser.add_argument("--no-ply", action="store_true", help="Skip vertex-colored PLY exports.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device passed to patch_vis.py (default: its own choice, CUDA when available). "
        "Use 'cpu' to export while a training run occupies the GPU.",
    )
    parser.add_argument("--force", action="store_true", help="Re-export even if outputs are current.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without exporting.")
    parser.add_argument("--python", default=None, help="Python interpreter that has torch/trimesh.")
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__).resolve().parent)
    if repo is None:
        print("ERROR: run this script from inside the NISR_Seawall repository.", file=sys.stderr)
        return 2

    logs_root = Path(args.logs_root).resolve() if args.logs_root else repo / "logs"
    if not logs_root.is_dir():
        print(f"ERROR: logs root not found: {logs_root}", file=sys.stderr)
        return 2

    checkpoint = None
    if args.model:
        target = Path(args.model)
        if target.is_file() and target.suffix == ".pt":
            checkpoint = target.resolve()
            run_dir = checkpoint.parent
        elif target.is_dir():
            run_dir = target.resolve()
            if latest_checkpoint(run_dir) is None:
                print(f"ERROR: no checkpoint*.pt in {run_dir}", file=sys.stderr)
                return 2
        elif target.exists():
            print(f"ERROR: unsupported path: {target}", file=sys.stderr)
            return 2
        else:
            best = pick_run(logs_root, args.model)
            if best is None:
                print(f"ERROR: no run matching {args.model!r} under {logs_root}", file=sys.stderr)
                return 1
            _, run_dir, _ = best
    else:
        best = pick_run(logs_root, None)
        if best is None:
            print(f"ERROR: no runs with checkpoints under {logs_root}", file=sys.stderr)
            return 1
        _, run_dir, _ = best

    if checkpoint is None:
        checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        print(f"ERROR: no checkpoint*.pt in {run_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    folder, combined_ply, _ = exported_paths(out_dir, checkpoint)
    epoch = checkpoint_epoch(checkpoint)
    label = f"epoch {epoch}" if epoch >= 0 else "final checkpoint"

    print(f"Run folder : {run_dir}")
    print(f"Checkpoint : {checkpoint} ({label})")
    if training_running(run_dir):
        print("Training   : still running - a newer checkpoint may appear later.")
    else:
        print("Training   : stopped")
    print(f"Output dir : {folder}")

    current = combined_ply.is_file()
    if current and not args.force:
        newest = combined_ply
        if newest.stat().st_mtime >= checkpoint.stat().st_mtime:
            print("Already exported and newer than the checkpoint (use --force to redo).")
            summarize(folder, checkpoint)
            return 0

    python = resolve_python(args.python)
    if python is None:
        print("ERROR: no Python interpreter found; pass --python.", file=sys.stderr)
        return 2

    command = [
        python,
        str(repo / "utils" / "patch_vis.py"),
        "--ckpt",
        str(checkpoint),
        "--out_dir",
        str(out_dir),
        "--n_images",
        str(args.n_images),
        "--subdivision_depth",
        str(args.subdivision_depth),
        "--no_obj",
    ]
    if args.resolution is not None:
        command += ["--resolution", str(args.resolution)]
    if args.device:
        command += ["--device", args.device]
    if args.no_ply:
        command.append("--no_ply")

    if args.dry_run:
        print("Dry run; command:")
        print("  " + " ".join(command))
        return 0

    folder.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"patch_vis_{checkpoint.stem}.log"
    print(f"Exporting   : {log_path} (this can take several minutes for dense runs)")
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            import subprocess

            result = subprocess.run(
                command, cwd=repo, stdout=log_file, stderr=subprocess.STDOUT, text=True
            )
    except OSError as exc:
        print(f"ERROR: could not launch patch_vis.py: {exc}", file=sys.stderr)
        return 2

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        print(f"ERROR: patch_vis.py failed (exit {result.returncode}).", file=sys.stderr)
        print("Last lines of the log:", file=sys.stderr)
        for line in log_text.splitlines()[-20:]:
            print(f"  {line}", file=sys.stderr)
        print(f"Full log: {log_path}", file=sys.stderr)
        return result.returncode

    print("Exported visualization")
    summarize(folder, checkpoint)
    svd_line = next((line.strip() for line in log_text.splitlines() if line.strip().startswith("SVD (")), None)
    if svd_line:
        print(f"  {svd_line}")
    print(f"  Log         : {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
