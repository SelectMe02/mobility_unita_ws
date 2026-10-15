"""Route points from a waypoint CSV or a Map Tuner project JSON.

A JSON project is read as the Leo route: the saved Raceline 1 of every
sector, consecutive duplicates removed (route_racelines is ignored; R2/R3
remain avoidance lanes).

Surveyed route features (intersections, lane-change intervals, the tunnel
blackout window) are stored as map anchors {x, y, yaw} and projected onto the
current route at start-up, so a Map Tuner edit moves their index and arc
length with the route instead of invalidating them.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


# An anchor further than this from the current route is a survey that no
# longer matches the map; the node refuses to start instead of guessing.
ANCHOR_MAX_ERROR_M = 3.0


def is_project(path):
    return Path(path).suffix.lower() == '.json'


def _map_tunner_core():
    import rospkg
    scripts = str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import map_tunner_core
    return map_tunner_core


def _project_route(path):
    core = _map_tunner_core()
    route, _ = core.primary_route(core.load_project(path), raceline1_only=True)
    return np.array([(point['x'], point['y']) for point in route], float)


def load_route_speeds(path):
    """Map Tuner point speed_kmh for every Leo route point (None for a CSV)."""
    if not is_project(path):
        return None
    core = _map_tunner_core()
    route, _ = core.primary_route(core.load_project(path), raceline1_only=True)
    return np.array([float(point['speed_kmh']) for point in route], float)


def load_lane_points(path):
    """x, y of every saved Raceline 2/3 point (avoidance lanes), or None."""
    if not is_project(path):
        return None
    project = _map_tunner_core().load_project(path)
    points = [(point['x'], point['y']) for sector in project['sectors']
              for lane, line in sector['racelines'].items() if lane != '1'
              for point in line]
    return np.array(points, float) if points else None


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


def checkpoint_indices(path):
    """Route index of every checkpoint ('start', '1'..'14') on the Leo route.

    A checkpoint is the first Raceline 1 point of the sector that starts
    there, so a Raceline 1 cut short before it does not move it.
    """
    core = _map_tunner_core()
    project = core.load_project(path)
    route, _ = core.primary_route(project, raceline1_only=True)
    ids = {point['id']: index for index, point in enumerate(route)}
    xy = np.array([(point['x'], point['y']) for point in route], float)
    result = {}
    for sector in project['sectors']:
        first = sector['racelines']['1'][0]
        index = ids.get(first['id'])
        if index is None:  # dropped as a duplicate of the previous sector's end
            index = int(np.argmin(np.hypot(*(xy - (first['x'], first['y'])).T)))
        result[str(sector['from'])] = 0 if sector['from'] == 'start' else index
    return result


def anchor_on_route(points, anchor, max_error=ANCHOR_MAX_ERROR_M):
    """Project an {x, y, yaw} anchor onto the route.

    Returns (index, s): the segment start index and the arc length of the
    projection. Only segments within 60 degrees of yaw are considered, so a
    route that crosses itself resolves to the surveyed pass.
    """
    xy = np.asarray(points, float)[:, :2]
    x, y, yaw = float(anchor['x']), float(anchor['y']), float(anchor['yaw'])
    if not all(math.isfinite(v) for v in (x, y, yaw)):
        raise ValueError('invalid route anchor {}'.format(anchor))
    delta = np.diff(xy, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    s = np.r_[0., np.cumsum(lengths)]
    t = np.clip(np.sum((np.array((x, y)) - xy[:-1]) * delta, axis=1)
                / np.maximum(lengths ** 2, 1e-12), 0., 1.)
    error = np.linalg.norm(xy[:-1] + t[:, None] * delta - (x, y), axis=1)
    aligned = (delta @ np.array((math.cos(yaw), math.sin(yaw)))
               / np.maximum(lengths, 1e-9)) > .5
    error[~aligned | (lengths < 1e-6)] = np.inf
    index = int(error.argmin())
    if error[index] > max_error:
        raise ValueError('route anchor ({:.2f}, {:.2f}) is {:.2f} m from the route'.format(
            x, y, float(error[index])))
    return index, float(s[index] + t[index] * lengths[index])


def load_survey(path, points):
    """Intersections with route s and waypoint_index from their anchors."""
    survey = json.loads(Path(path).read_text())
    stops = []
    for stop in survey['intersections']:
        index, s = anchor_on_route(points, stop['anchor'])
        stops.append(dict(stop, s=s, waypoint_index=index))
    return stops


def load_lane_changes(path, points):
    """Dashed-line intervals with route start_s/end_s from their anchors."""
    changes = []
    for change in json.loads(Path(path).read_text())['lane_changes']:
        _, start = anchor_on_route(points, change['start'])
        _, end = anchor_on_route(points, change['end'])
        item = {key: value for key, value in change.items() if key not in ('start', 'end')}
        changes.append(dict(item, start_s=start, end_s=end))
    return changes
