#!/usr/bin/env python3
"""Export tracked vertex trajectories as PLY line/tube meshes."""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_HERE), _HERE, os.getcwd()):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import utils as utils  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Export tracked vertex trajectories to PLY.')
    parser.add_argument('--json', required=True, help='Path to the tracked vertex JSON file')
    parser.add_argument('--vertices', nargs='+', required=True,
                        help='Vertex names to export, e.g. v1 v2 v10')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: JSON file directory)')
    parser.add_argument('--output_name', default=None,
                        help='Output PLY filename (default derived from vertex names)')
    parser.add_argument('--line_radius', type=float, default=5.0,
                        help='Tube radius for the exported trajectory lines')
    parser.add_argument('--position_key', type=str, default='position',
                        choices=['position', 'position_normalized', 'position_denormalized'],
                        help='Which position field to read from the JSON file')
    parser.add_argument('--color_mode', type=str, default='epoch',
                        choices=['epoch', 'arc_length'],
                        help='Color progression mode: by epoch or by cumulative traveled distance')
    parser.add_argument('--colormap', type=str, default='turbo',
                        help='Matplotlib colormap name used for the spectrum coloring')
    args = parser.parse_args()

    json_path = os.path.abspath(args.json)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(json_path)
    os.makedirs(output_dir, exist_ok=True)

    if args.output_name:
        output_name = args.output_name
    elif len(args.vertices) == 1:
        output_name = f'{args.vertices[0]}_trajectory.ply'
    else:
        output_name = 'vertex_trajectories.ply'

    output_path = os.path.join(output_dir, output_name)
    utils.export_vertex_trajectories_ply(
        json_path=json_path,
        vertex_names=args.vertices,
        output_path=output_path,
        position_key=args.position_key,
        line_radius=args.line_radius,
        color_mode=args.color_mode,
        colormap=args.colormap,
    )


if __name__ == '__main__':
    main()
