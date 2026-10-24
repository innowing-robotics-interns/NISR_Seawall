#!/usr/bin/env python3
"""Build a consolidated visual check folder for NISR_Seawall checkpoints.

For every checkpoint this orchestrator runs the three standalone inspection
scripts and then assembles a multi-run comparison sheet and a Markdown report:

  <out>/<run>/gallery/       target vs prediction, 16 deterministic views
  <out>/<run>/checkerboard/  per-patch checkerboard renders
  <out>/<run>/metrics/       Chamfer, coverage, normals, UV stretch
  <out>/comparison.png       target + one column per run
  <out>/inspection.md        metrics table and image links
  <out>/check_summary.json   all parsed numbers

The checkpoint path may point at a file or at a run directory (the newest
``checkpoint*.pt`` is used).  Rendering stages are reused when their summaries
are current; pass ``--force`` to redo them.

The checkerboard stage runs for every check, including interim renders during
training.  ``--no-checkerboard`` exists for deliberately reduced debugging
runs only; a reported check should always include the checkerboard images.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEAWALL_PYTHON = Path('/home/lyangiw/anaconda3/envs/seawall/bin/python')


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / 'scripts' / 'render_gallery.py').is_file():
            return parent
    raise SystemExit('run this script from inside the NISR_Seawall repository')


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
GALLERY = REPO_ROOT / 'scripts' / 'render_gallery.py'
CHECKERBOARD = REPO_ROOT / 'scripts' / 'render_checkerboard_patches.py'
METRICS = REPO_ROOT / 'scripts' / 'eval_sheet_metrics.py'


def _default_python() -> str:
    override = os.environ.get('SEAWALL_PYTHON')
    if override:
        return override
    if SEAWALL_PYTHON.is_file():
        return str(SEAWALL_PYTHON)
    return sys.executable


def _resolve_checkpoint(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        candidates = sorted(path.glob('checkpoint*.pt'),
                            key=lambda item: item.stat().st_mtime)
        if not candidates:
            raise SystemExit(f'no checkpoint*.pt under {path}')
        return candidates[-1].resolve()
    if not path.is_file():
        raise SystemExit(f'checkpoint not found: {path}')
    return path.resolve()


def _label_for(checkpoint: Path) -> str:
    label = re.sub(r'[^A-Za-z0-9._-]+', '_', checkpoint.parent.name)
    return label.strip('._') or 'run'


def _unique_labels(checkpoints):
    seen = {}
    labels = []
    for checkpoint in checkpoints:
        base = _label_for(checkpoint)
        count = seen.get(base, 0) + 1
        seen[base] = count
        labels.append(base if count == 1 else f'{base}_{count}')
    return labels


def _run(command, log_path: Path) -> None:
    line = '$ ' + ' '.join(str(part) for part in command)
    print(line, flush=True)
    # Calling the interpreter by absolute path does not activate its env, so
    # nvdiffrast's JIT cannot find the env's ninja.  Prepend the interpreter's
    # bin directory for PATH lookups.
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    interpreter_dir = str(Path(str(command[0])).parent)
    env['PATH'] = interpreter_dir + os.pathsep + env.get('PATH', '')
    with open(log_path, 'a') as handle:
        handle.write(line + '\n')
        handle.flush()
        process = subprocess.Popen(
            [str(part) for part in command], cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env)
        assert process.stdout is not None
        for text in process.stdout:
            sys.stdout.write(text)
            handle.write(text)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f'command failed with exit code {code}; see {log_path}')


def _load_json(path: Path):
    with open(path) as handle:
        return json.load(handle)


def _reuse(summary_path: Path, output_paths, checks, checkpoint: Path,
           force: bool):
    """Return parsed summary when an existing stage still matches."""
    if force or not summary_path.is_file():
        return None
    if not all(path.is_file() for path in output_paths):
        return None
    if summary_path.stat().st_mtime < checkpoint.stat().st_mtime:
        return None
    try:
        data = _load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return None
    if os.path.abspath(data.get('checkpoint', '')) != str(checkpoint):
        return None
    for key, expected in checks.items():
        if data.get(key) != expected:
            return None
    return data


def _stage_gallery(checkpoint, run_dir, args, log_path):
    out = run_dir / 'gallery'
    summary_path = out / 'summary.json'
    outputs = [out / 'target_vs_pred_16.png', summary_path]
    checks = {'resolution': args.resolution}
    data = _reuse(summary_path, outputs, checks, checkpoint, args.force)
    if data is not None and ('l1_normal' not in data
                             or 'rendered_normal' not in data):
        data = None
    if data is None:
        out.mkdir(parents=True, exist_ok=True)
        command = [args.python, GALLERY,
                   '--checkpoint', checkpoint, '--out', out,
                   '--device', args.device,
                   '--resolution', str(args.resolution)]
        _run(command, log_path)
        data = _load_json(summary_path)
    else:
        print(f'  gallery reused: {out}', flush=True)
    return data


def _stage_checkerboard(checkpoint, run_dir, args, log_path):
    out = run_dir / 'checkerboard'
    summary_path = out / 'summary.json'
    outputs = [out / 'checkerboard_all_views.png', summary_path]
    checks = {'render_resolution': args.checker_resolution}
    data = _reuse(summary_path, outputs, checks, checkpoint, args.force)
    if data is not None and not (out / data.get(
            'layout_image', 'checkerboard_atlas.png')).is_file():
        data = None
    if data is not None and len(data.get('views_rendered', [])) != args.checker_views:
        data = None
    if data is None:
        out.mkdir(parents=True, exist_ok=True)
        command = [args.python, CHECKERBOARD,
                   '--checkpoint', checkpoint, '--out', out,
                   '--device', args.device,
                   '--resolution', str(args.checker_resolution),
                   '--views', str(args.checker_views)]
        _run(command, log_path)
        data = _load_json(summary_path)
    else:
        print(f'  checkerboard reused: {out}', flush=True)
    return data


def _stage_metrics(checkpoint, run_dir, args, log_path):
    out = run_dir / 'metrics'
    summary_path = out / 'metrics.json'
    outputs = [summary_path]
    if args.metrics_mesh_res > 0:
        checks = {'mesh_resolution': args.metrics_mesh_res}
    else:
        checks = {'mesh_resolution_auto': True}
    data = _reuse(summary_path, outputs, checks, checkpoint, args.force)
    if data is not None and 'visible_layer_normal' not in data:
        data = None
    if data is None:
        out.mkdir(parents=True, exist_ok=True)
        command = [args.python, METRICS,
                   '--checkpoint', checkpoint, '--out', out,
                   '--device', args.device,
                   '--mesh-res', str(args.metrics_mesh_res)]
        _run(command, log_path)
        data = _load_json(summary_path)
    else:
        print(f'  metrics reused: {out}', flush=True)
    return data


def _font(size: int):
    for name in ('DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(draw, text: str, font, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + '...', font=font) > width:
        text = text[:-1]
    return text + '...'


def _comparison_indices(view_count: int, rows: int):
    if view_count <= 0:
        return []
    if rows >= view_count:
        return list(range(view_count))
    indices = [int(round((row + 0.75) * view_count / rows))
               for row in range(rows)]
    return [min(view_count - 1, index) for index in indices]


def _open_tile(path: Path, size):
    with Image.open(path) as image:
        tile = image.convert('RGB')
    if tile.size != size:
        tile = tile.resize(size, Image.LANCZOS)
    return tile


def _build_comparison(records, args, out: Path):
    primary = records[0]
    gallery = primary['gallery']
    view_count = int(gallery.get('views', 0))
    indices = _comparison_indices(view_count, args.rows)
    if not indices:
        print('  comparison skipped: no gallery views', flush=True)
        return None

    first_path = Path(primary['run_dir']) / 'gallery' / 'views' / \
        f'target_{indices[0]:04d}.png'
    if not first_path.is_file():
        print(f'  comparison skipped: missing {first_path}', flush=True)
        return None
    with Image.open(first_path) as image:
        tile_size = image.size

    columns = 1 + len(records)
    header = 48
    canvas = Image.new('RGB', (columns * tile_size[0],
                               header + len(indices) * tile_size[1]),
                       (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    small_font = _font(15)
    inner = tile_size[0] - 20

    headings = [('orbit target', f'{view_count} deterministic views')]
    for record in records:
        l1 = record['gallery'].get('mean_full_l1')
        subtitle = f'L1 {l1:.4f}' if l1 is not None else 'L1 n/a'
        headings.append((record['label'], subtitle))
    for column, (title, subtitle) in enumerate(headings):
        x = column * tile_size[0] + 10
        draw.text((x, 5), _fit(draw, title, title_font, inner),
                  font=title_font, fill=(20, 20, 20))
        draw.text((x, 27), subtitle, font=small_font, fill=(80, 80, 80))
    draw.line([(0, header - 1), (canvas.width, header - 1)],
              fill=(190, 190, 195))

    missing = Image.new('RGB', tile_size, (218, 218, 222))
    missing_draw = ImageDraw.Draw(missing)
    missing_draw.text((12, 12), 'view unavailable', font=small_font,
                      fill=(90, 90, 90))

    for row, view_index in enumerate(indices):
        y = header + row * tile_size[1]
        target_path = Path(primary['run_dir']) / 'gallery' / 'views' / \
            f'target_{view_index:04d}.png'
        tile = _open_tile(target_path, tile_size) if target_path.is_file() \
            else missing
        canvas.paste(tile, (0, y))
        for column, record in enumerate(records, start=1):
            pred_path = Path(record['run_dir']) / 'gallery' / 'views' / \
                f'pred_{view_index:04d}.png'
            tile = _open_tile(pred_path, tile_size) if pred_path.is_file() \
                else missing
            canvas.paste(tile, (column * tile_size[0], y))

    path = out / 'comparison.png'
    canvas.save(path)
    print(f'  comparison -> {path}', flush=True)
    return path, indices


def _fmt(value, digits=4):
    if value is None:
        return 'n/a'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f'{float(value):.{digits}f}'


def _write_report(out: Path, records, comparison, indices, warnings):
    primary = records[0]
    gallery = primary.get('gallery') or {}
    lines = [
        f"# Render check: {primary['label']}",
        '',
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by the "
        '`render-check` skill.',
        '',
        f"- cloud: `{gallery.get('cloud', 'n/a')}`",
        f"- atlas: `{gallery.get('atlas_mode', 'n/a')}` "
        f"({gallery.get('atlas_faces', 'n/a')} faces), "
        f"{gallery.get('active_patches', 'n/a')} active patches",
        f"- orbit views: {gallery.get('views', 'n/a')} at "
        f"{gallery.get('resolution', 'n/a')} px",
    ]
    per_patch = (primary.get('metrics') or {}).get('per_patch') or {}
    depths = Counter(entry['depth'] for entry in per_patch.values()
                     if entry.get('depth') is not None)
    if depths:
        detail = ', '.join(f'd{depth} {depths[depth]}'
                           for depth in sorted(depths))
        lines.append(f'- adaptive leaves by depth: {detail}')
    lines.append('')
    if warnings:
        lines.append('## Warnings')
        lines.append('')
        for warning in warnings:
            lines.append(f'- {warning}')
        lines.append('')

    if comparison is not None:
        comparison_path, compared_views = comparison
        view_text = ', '.join(f'v{index}' for index in compared_views)
        lines += [
            '## Target vs runs',
            '',
            f'Views {view_text} of '
            f'{gallery.get("views", "n/a")}. Left column is the point-cloud '
            'target; each run column is the rendered prediction with its '
            'full-frame six-channel L1. The orbit cameras are deterministic: '
            'when `lambda_render > 0` they are the same views the render loss '
            'trained on, so this is a training-view check; for runs without a '
            'render loss the views are not used in training.',
            '',
            f'![comparison]({comparison_path.relative_to(out)})',
            '',
        ]

    lines += [
        '## Render error',
        '',
        '| run | full L1 | normal L1 | position L1 | per-view min-max |',
        '|---|---|---|---|---|',
    ]
    for record in records:
        data = record.get('gallery') or {}
        per_view = data.get('per_view_full_l1') or []
        span = (f'{min(per_view):.4f}-{max(per_view):.4f}'
                if per_view else 'n/a')
        lines.append(
            f"| {record['label']} | {_fmt(data.get('mean_full_l1'))} | "
            f"{_fmt(data.get('l1_normal'))} | "
            f"{_fmt(data.get('l1_position'))} | {span} |")
    lines.append('')

    primary_metrics = primary.get('metrics') or {}
    near_threshold = _fmt(primary_metrics.get('near_threshold'), 3)
    lines += [
        '## Geometry',
        '',
        'Chamfer is measured against the input cloud. It is a held-out metric '
        'when the run did not optimize it. On the adaptive atlas '
        '`lambda_render > 0` replaces Chamfer even if `lambda_chamfer` is '
        'still 1; on fixed-grid atlases the render term is added to Chamfer, '
        'so CD is a fit diagnostic when `lambda_chamfer > 0` there. Distances '
        'are in normalized cloud coordinates. `rendered normal` compares the '
        'normal channels on covered pixels; `cloud normal` allows a nearby '
        'opposite layer, which matters for closed shells.',
        '',
        '| run | surface term | Chamfer sym | cloud->sheet | sheet->cloud | '
        f'coverage <{near_threshold} | rendered normal med / p95 | cloud '
        'normal med / p95 | checker edge p95 |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for record in records:
        data = record.get('metrics') or {}
        args = data.get('args') or {}
        gallery = record.get('gallery') or {}
        visible = data.get('visible_layer_normal') or {}
        rendered = gallery.get('rendered_normal') or {}
        uses_render = float(args.get('lambda_render') or 0.0) > 0
        uses_cd = float(args.get('lambda_chamfer') or 0.0) > 0
        # metrics.json stores only the objective-related args, not `adaptive`;
        # atlas_mode is always written, so use it as the authoritative flag.
        is_adaptive = bool(args.get('adaptive')) or \
            data.get('atlas_mode') == 'adaptive_cube'
        if uses_render and is_adaptive:
            surface = 'render (CD replaced)'
        elif uses_render and uses_cd:
            surface = 'render + CD'
        elif uses_render:
            surface = 'render'
        elif uses_cd:
            surface = 'CD'
        else:
            surface = 'none'
        lines.append(
            f"| {record['label']} | {surface} | "
            f"{_fmt(data.get('chamfer_symmetric'))} | "
            f"{_fmt(data.get('chamfer_cloud_to_sheet'))} | "
            f"{_fmt(data.get('chamfer_sheet_to_cloud'))} | "
            f"{_fmt(data.get('covered_fraction_near'), 3)} | "
            f"{_fmt(rendered.get('median_deg'), 2)} / "
            f"{_fmt(rendered.get('p95_deg'), 2)} deg | "
            f"{_fmt(visible.get('normal_angle_median_deg'), 2)} / "
            f"{_fmt(visible.get('normal_angle_p95_deg'), 2)} deg | "
            f"{_fmt(data.get('long_edge_global_p95'), 3)} |")
    lines.append('')

    lines += [
        '## Objective settings',
        '',
        '| run | lambda_render | start | warmup | lap weight | lambda_chamfer '
        '| schedule |',
        '|---|---|---|---|---|---|---|',
    ]
    for record in records:
        metrics = record.get('metrics') or {}
        data = metrics.get('args') or {}
        replaces_cd = (float(data.get('lambda_render') or 0.0) > 0
                       and metrics.get('atlas_mode') == 'adaptive_cube')
        if replaces_cd and data.get('lambda_chamfer') is not None:
            chamfer_display = f"{float(data['lambda_chamfer']):.4f} (logged only)"
        else:
            chamfer_display = _fmt(data.get('lambda_chamfer'), 4)
        lines.append(
            f"| {record['label']} | {_fmt(data.get('lambda_render'), 4)} | "
            f"{_fmt(data.get('render_start_epoch'), 0)} | "
            f"{_fmt(data.get('render_warmup_epochs'), 0)} | "
            f"{_fmt(data.get('render_laplacian_weight'), 3)} | "
            f"{chamfer_display} | "
            f"{data.get('schedule', 'n/a')} |")
    lines.append('')

    lines += ['## Images', '']
    for record in records:
        run_rel = Path(record['run_dir']).relative_to(out)
        lines += [f"### {record['label']}", '']
        if record.get('gallery'):
            lines.append(
                f"- [target vs prediction]"
                f"({run_rel / 'gallery' / 'target_vs_pred_16.png'})")
        if record.get('checkerboard'):
            checker = record['checkerboard']
            layout_image = checker.get('layout_image',
                                       'checkerboard_atlas.png')
            layout_title = ('checkerboard palette'
                            if 'palette' in layout_image
                            else 'checkerboard atlas key')
            lines += [
                f"- [checkerboard views]"
                f"({run_rel / 'checkerboard' / 'checkerboard_all_views.png'})",
                f"- [{layout_title}]"
                f"({run_rel / 'checkerboard' / layout_image})",
            ]
        if record.get('metrics'):
            lines.append(
                f"- [metrics JSON]({run_rel / 'metrics' / 'metrics.json'})")
        lines.append('')
    path = out / 'inspection.md'
    path.write_text('\n'.join(lines))
    print(f'  report -> {path}', flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Render and evaluate one or more NISR_Seawall checkpoints.')
    parser.add_argument('--checkpoint', required=True,
                        help='checkpoint file or run directory (newest ckpt)')
    parser.add_argument('--compare', nargs='*', default=[],
                        help='additional checkpoints or run directories')
    parser.add_argument('--out', required=True, help='check folder to write')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution', type=int, default=256,
                        help='gallery render resolution')
    parser.add_argument('--rows', type=int, default=4,
                        help='view rows in the comparison sheet')
    parser.add_argument('--checker-views', type=int, default=4)
    parser.add_argument('--checker-resolution', type=int, default=512)
    parser.add_argument('--metrics-mesh-res', type=int, default=0,
                        help='UV samples per patch side for the metric mesh; '
                             '0 = auto (96 fixed-grid, 16 adaptive)')
    parser.add_argument('--no-gallery', action='store_true')
    parser.add_argument('--no-checkerboard', action='store_true')
    parser.add_argument('--no-metrics', action='store_true')
    parser.add_argument('--python', default=_default_python(),
                        help='interpreter with torch/nvdiffrast')
    parser.add_argument('--force', action='store_true',
                        help='re-render even when stage summaries are current')
    args = parser.parse_args()

    if args.no_gallery and args.compare:
        parser.error('--compare needs galleries; remove --no-gallery')
    if args.no_checkerboard:
        print('[warn] checkerboard stage skipped; a reported check should '
              'include it', flush=True)
    if args.rows < 1:
        parser.error('--rows must be at least 1')

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'render_check.log'
    if args.force and log_path.exists():
        log_path.unlink()

    checkpoints = [_resolve_checkpoint(args.checkpoint)]
    checkpoints += [_resolve_checkpoint(value) for value in args.compare]
    labels = _unique_labels(checkpoints)

    print(f'repo: {REPO_ROOT}', flush=True)
    print(f'check: {out}', flush=True)
    print(f'python: {args.python}', flush=True)
    for label, checkpoint in zip(labels, checkpoints):
        print(f'  {label}: {checkpoint}', flush=True)

    records = []
    for label, checkpoint in zip(labels, checkpoints):
        run_dir = out / label
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n[{label}] {checkpoint}', flush=True)
        record = {
            'label': label,
            'checkpoint': str(checkpoint),
            'run_dir': str(run_dir),
            'gallery': None,
            'checkerboard': None,
            'metrics': None,
        }
        if not args.no_gallery:
            record['gallery'] = _stage_gallery(
                checkpoint, run_dir, args, log_path)
        if not args.no_checkerboard:
            record['checkerboard'] = _stage_checkerboard(
                checkpoint, run_dir, args, log_path)
        if not args.no_metrics:
            record['metrics'] = _stage_metrics(
                checkpoint, run_dir, args, log_path)
        records.append(record)

    warnings = []
    clouds = {record['gallery'].get('cloud') for record in records
              if record['gallery']}
    if len(clouds) > 1:
        warnings.append('runs were trained on different clouds; the '
                        'comparison columns are not directly comparable')
    view_counts = {record['gallery'].get('views') for record in records
                   if record['gallery']}
    if len(view_counts) > 1:
        warnings.append(f'runs have different gallery view counts: '
                        f'{sorted(view_counts)}')
    if any((record['gallery'] or {}).get('atlas_mode') == 'six_sheet'
           for record in records):
        warnings.append('a six-sheet closed shell wraps the scan; symmetric '
                        'Chamfer penalizes the unseen back layer, so prefer '
                        'rendered normals and cloud coverage; the cloud-space '
                        'normal check can pick the wrong layer')
    for record in records:
        metrics = record.get('metrics') or {}
        aspect = metrics.get('aspect_global_p95')
        if aspect is not None and aspect > 10:
            warnings.append(
                f"{record['label']}: UV aspect p95 {aspect:.1f} - one UV "
                'direction is strongly compressed in many patches; the '
                'checkerboard shows streaks. Geometry metrics are unaffected.')

    comparison = None
    if records and all(record['gallery'] for record in records):
        comparison = _build_comparison(records, args, out)

    report_path = _write_report(out, records, comparison, [], warnings)

    summary = {
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'primary': records[0]['label'],
        'comparison': (str(comparison[0]) if comparison else None),
        'comparison_views': (comparison[1] if comparison else []),
        'report': str(report_path),
        'warnings': warnings,
        'runs': records,
    }
    summary_path = out / 'check_summary.json'
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(f'\nsummary -> {summary_path}', flush=True)
    print(f'log     -> {log_path}', flush=True)


if __name__ == '__main__':
    main()
