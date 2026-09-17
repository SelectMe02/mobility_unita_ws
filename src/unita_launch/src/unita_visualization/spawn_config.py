"""Build MORAI MapInitSetting JSON from a local XYZ waypoint path.

The installed SIM's MapInitSetting stores ENU vectors m_InitPos/m_InitRot;
rotation components are roll/pitch/yaw in degrees. Loading this file configures
the existing SIM initialization key, rather than sending a vehicle command.
"""
import math
import re


def make_start_spawn(lines):
    start = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        tokens = re.split(r'[,\s]+', line)
        if len(tokens) < 2:
            continue
        try:
            point = tuple(float(value) for value in tokens[:3])
        except ValueError:
            continue  # CSV column header
        if not all(math.isfinite(value) for value in point):
            raise ValueError('waypoint contains non-finite coordinates')
        if start is None:
            if len(point) != 3:
                raise ValueError('first waypoint needs local x, y, z')
            start = point
            continue
        dx, dy = point[0] - start[0], point[1] - start[1]
        if math.hypot(dx, dy) <= 1e-6:
            continue
        yaw = math.degrees(math.atan2(dy, dx))
        return {'m_InitPos': dict(zip(('x', 'y', 'z'), start)),
                'm_InitRot': {'x': 0.0, 'y': 0.0, 'z': yaw}}
    raise ValueError('need a start XYZ and another distinct XY waypoint')
