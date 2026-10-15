"""Route points from a waypoint CSV or a Map Tuner project JSON.

A JSON project is read the same way as the waypoint follower: the saved
Raceline 1 of every sector, consecutive duplicates removed. Arc length along
it equals the old waypoints.csv, so surveyed s values stay valid.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np


def is_project(path):
    return Path(path).suffix.lower() == '.json'


def _project_route(path):
    import rospkg
    scripts = str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from map_tunner_core import load_project, primary_route
    route, _ = primary_route(load_project(path))
    return np.array([(point['x'], point['y']) for point in route], float)


def load_route_points(path):
    """Return route points as rows starting with x, y."""
    return _project_route(path) if is_project(path) else np.loadtxt(path)


def route_digest(path):
    """sha256 identifying the route a survey was derived from.

    A CSV is hashed as bytes. A JSON project is hashed over its Raceline 1
    coordinates only, so editing speeds or other racelines keeps the survey.
    """
    if not is_project(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    points = np.ascontiguousarray(_project_route(path), dtype='<f8')
    return hashlib.sha256(points.tobytes()).hexdigest()
