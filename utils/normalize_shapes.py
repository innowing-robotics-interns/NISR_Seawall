#!/usr/bin/env python3
"""Normalize every PLY point cloud in a folder to a shared scale and origin."""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_TARGET_DIAG = 0.2629781166957031  # bbox diagonal of 3d_test_models/horse_rotate.ply
PLY_TYPES = {
    'char': 'i1', 'int8': 'i1',
    'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2',
    'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4',
    'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4',
    'double': 'f8', 'float64': 'f8',
}


def read_ply_vertices(path):
    """Return a structured vertex array plus its PLY property names."""
    with open(path, 'rb') as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError('unexpected end of PLY header')
            text = line.decode('ascii', errors='ignore').strip()
            header.append(text)
            if text == 'end_header':
                break
        payload = handle.read()

    if not header or header[0] != 'ply':
        raise ValueError('not a PLY file')
    fmt = next((line.split()[1] for line in header if line.startswith('format ')), None)
    if fmt != 'binary_little_endian':
        raise ValueError(f'unsupported PLY format: {fmt}')

    elements = []
    current = None
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'element':
            current = {'name': parts[1], 'count': int(parts[2]), 'props': []}
            elements.append(current)
        elif parts[0] == 'property' and current is not None:
            if parts[1] == 'list':
                current['props'].append(('list', parts[2], parts[3], parts[4]))
            else:
                current['props'].append((parts[2], parts[1]))

    vertex = next((e for e in elements if e['name'] == 'vertex'), None)
    if vertex is None or vertex['count'] <= 0 or not vertex['props']:
        raise ValueError('no vertex element')
    for element in elements:
        if element is not vertex and element['count'] > 0:
            raise ValueError(f"unsupported non-empty '{element['name']}' element")
    if any(prop[0] == 'list' for prop in vertex['props']):
        raise ValueError('list properties are not supported in point clouds')
    for name, ply_type in vertex['props']:
        if ply_type not in PLY_TYPES:
            raise ValueError(f'unsupported vertex property type: {ply_type}')

    dtype = np.dtype([
        (name, '<' + PLY_TYPES[ply_type]) for name, ply_type in vertex['props']
    ])
    expected = vertex['count'] * dtype.itemsize
    if len(payload) < expected:
        raise ValueError(
            f'truncated vertex data: expected {expected} bytes, found {len(payload)}'
        )
    data = np.frombuffer(payload, dtype=dtype, count=vertex['count']).copy()
    for axis in ('x', 'y', 'z'):
        if axis not in data.dtype.names:
            raise ValueError(f'missing vertex property: {axis}')
    return data


def write_ply_vertices(path, data):
    """Write a structured vertex array as binary PLY, keeping property order."""
    fields = []
    for name in data.dtype.names:
        if name in ('x', 'y', 'z', 'nx', 'ny', 'nz'):
            fields.append((name, '<f4'))
        else:
            fields.append((name, data.dtype.fields[name][0]))
    out = np.empty(data.shape[0], dtype=np.dtype(fields))
    for name in data.dtype.names:
        out[name] = data[name]

    lines = [
        'ply',
        'format binary_little_endian 1.0',
        'comment normalized with utils/normalize_shapes.py',
        f'element vertex {len(data)}',
    ]
    for name, descr in fields:
        dtype_str = np.dtype(descr).str
        ply_type = {
            '<f4': 'float', '<f8': 'double',
            '<i1': 'char', '<u1': 'uchar',
            '<i2': 'short', '<u2': 'ushort',
            '<i4': 'int', '<u4': 'uint',
        }.get(dtype_str)
        if ply_type is None:
            raise ValueError(f'cannot map dtype {descr} back to a PLY type')
        lines.append(f'property {ply_type} {name}')
    lines.append('end_header')

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    with open(tmp_path, 'wb') as handle:
        handle.write(('\n'.join(lines) + '\n').encode('ascii'))
        handle.write(out.tobytes(order='C'))
    os.replace(tmp_path, path)


def normalize_vertices(data, target_value, target_kind, center_mode):
    xyz = np.stack([data[name] for name in ('x', 'y', 'z')], axis=1).astype(np.float64)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = float((hi - lo).max())
    diag = float(np.linalg.norm(hi - lo))
    if extent < 1e-12:
        raise ValueError('degenerate bounding box (zero extent)')
    center = 0.5 * (lo + hi) if center_mode == 'bbox' else xyz.mean(axis=0)
    scale = float(target_value) / (extent if target_kind == 'extent' else diag)
    xyz = (xyz - center) * scale

    out = data.copy()
    for i, name in enumerate(('x', 'y', 'z')):
        out[name] = xyz[:, i].astype(np.float32)

    if 'nx' in data.dtype.names and 'ny' in data.dtype.names and 'nz' in data.dtype.names:
        normals = np.stack([data[name] for name in ('nx', 'ny', 'nz')], axis=1)
        norms = np.linalg.norm(normals.astype(np.float64), axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normals = normals / norms
        for i, name in enumerate(('nx', 'ny', 'nz')):
            out[name] = normals[:, i].astype(np.float32)

    stats = {
        'n': int(data.shape[0]),
        'extent_before': extent,
        'diag_before': diag,
        'scale': scale,
        'center': center,
        'has_normals': 'nx' in data.dtype.names,
    }
    return out, stats


def reference_diagonal(reference):
    data = read_ply_vertices(reference)
    xyz = np.stack([data[name] for name in ('x', 'y', 'z')], axis=1).astype(np.float64)
    return float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))


def main():
    parser = argparse.ArgumentParser(
        description='Center every PLY point cloud in a folder and scale it to a '
                    'shared bounding-box diagonal or into the [-1,1] cube.'
    )
    parser.add_argument('--input-dir', default='3d_test_models',
                        help='Folder to scan for *.ply (default: 3d_test_models)')
    parser.add_argument('--output-dir', default=None,
                        help='Where to write normalized files (default: '
                             '<input-dir>_normalized). Ignored with --inplace.')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite the input files atomically instead of '
                             'writing a sibling folder.')
    parser.add_argument('--reference', default=None,
                        help='PLY whose bbox diagonal is the target (default: '
                             '<input-dir>/horse_rotate.ply when present)')
    parser.add_argument('--target-diag', type=float, default=None,
                        help=f'Target bbox diagonal (default: reference diagonal, '
                             f'else {DEFAULT_TARGET_DIAG:.6f})')
    parser.add_argument('--cube', action='store_true',
                        help='Fit each shape into [-1,1]^3: center on the bbox and '
                             'map the longest bbox axis to [-1,1]. Overrides '
                             '--target-diag/--reference and forces --center bbox.')
    parser.add_argument('--center', choices=['bbox', 'mean'], default='bbox',
                        help='Translation pivot: bounding-box center (default) or '
                             'vertex mean')
    parser.add_argument('--dry-run', action='store_true', help='Report scales without writing')
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f'ERROR: input folder not found: {input_dir}', file=sys.stderr)
        return 2

    if args.inplace:
        output_dir = input_dir
    elif args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_dir.with_name(input_dir.name + '_normalized')

    if args.cube:
        target_kind = 'extent'
        target_value = 2.0
        center_mode = 'bbox'
        source = '[-1,1]^3 cube'
    elif args.target_diag is not None:
        target_kind = 'diag'
        target_value = args.target_diag
        center_mode = args.center
        source = '--target-diag'
    else:
        target_kind = 'diag'
        center_mode = args.center
        reference = Path(args.reference).resolve() if args.reference else input_dir / 'horse_rotate.ply'
        if reference.is_file():
            target_value = reference_diagonal(reference)
            source = str(reference)
        else:
            target_value = DEFAULT_TARGET_DIAG
            source = 'built-in horse diagonal'
    if target_value <= 0:
        print(f'ERROR: target {target_kind} must be positive', file=sys.stderr)
        return 2

    files = sorted(input_dir.glob('*.ply'))
    if not files:
        print(f'ERROR: no .ply files in {input_dir}', file=sys.stderr)
        return 2

    print(f'Input      : {input_dir}')
    print(f'Output     : {"(in place)" if args.inplace else output_dir}')
    print(f'Target {target_kind:6s}: {target_value:.6f} ({source})')
    print(f'Center     : {center_mode}')
    skipped = []
    written = 0
    for path in files:
        try:
            data = read_ply_vertices(path)
            normalized, stats = normalize_vertices(data, target_value, target_kind, center_mode)
            if not args.dry_run:
                write_ply_vertices(output_dir / path.name, normalized)
            written += 1
            normals = 'normals' if stats['has_normals'] else 'NO normals'
            print(f'  {path.name:34s} {stats["n"]:>7d} pts  '
                  f'extent {stats["extent_before"]:11.6f}  '
                  f'diag {stats["diag_before"]:12.6f} -> x{stats["scale"]:.6g}  {normals}')
        except Exception as exc:
            skipped.append(path.name)
            print(f'  {path.name:34s} SKIPPED: {exc}', file=sys.stderr)

    action = 'Would normalize' if args.dry_run else 'Normalized'
    print(f'{action} {written}/{len(files)} shapes'
          + ('' if args.dry_run else f' -> {output_dir}'))
    if skipped:
        print(f'Skipped {len(skipped)} file(s): {", ".join(skipped)}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
