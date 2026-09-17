#!/usr/bin/env python3
"""Generate a native MORAI I-key spawn configuration; does not move the car."""
import argparse
import json
from pathlib import Path

from unita_visualization.spawn_config import make_start_spawn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--waypoint-file', required=True, help='local XYZ waypoint CSV')
    parser.add_argument('--output', required=True, help='new MapInitSetting JSON path')
    args = parser.parse_args()
    try:
        with Path(args.waypoint_file).expanduser().open() as source:
            config = make_start_spawn(source)
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Preserve existing spawn settings; choose another filename to regenerate.
        with output.open('x') as destination:
            json.dump(config, destination, indent=2, allow_nan=False)
            destination.write('\n')
    except (OSError, ValueError) as error:
        parser.exit(1, 'Cannot create spawn config: %s\n' % error)
    print('Created %s\nENU start: %s\nYaw: %.3f deg' %
          (output, config['m_InitPos'], config['m_InitRot']['z']))


if __name__ == '__main__':
    main()
