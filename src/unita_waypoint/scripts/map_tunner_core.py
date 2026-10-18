#!/usr/bin/env python3
"""Persistent, ROS-independent data model for the K-City RViz map tuner."""

import copy
from bisect import bisect_left
import json
import math
import os
import re
import tempfile
from pathlib import Path

import yaml


CHECKPOINT_ORDER = ['start'] + [str(number) for number in range(1, 15)]
MAX_LANES = 3
# Sector -> raceline that forms the driven route. A project may override this
# with its own "route_racelines" entry; unlisted sectors use Raceline 1.
DEFAULT_ROUTE_RACELINES = {'11': 3}


def load_checkpoints(path):
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)['checkpoints']
    if set(data) != set(CHECKPOINT_ORDER):
        raise ValueError('checkpoints must contain start and 1 through 14')
    for name in CHECKPOINT_ORDER:
        point = data[name]['position']
        if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError('invalid checkpoint position: ' + name)
    return data


def load_csv(path):
    points = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = re.split(r'[,\s]+', line)
            if len(values) < 3:
                raise ValueError('waypoint line {} needs x y z'.format(line_number))
            point = [float(value) for value in values[:3]]
            if not all(math.isfinite(value) for value in point):
                raise ValueError('non-finite waypoint on line {}'.format(line_number))
            points.append(point)
    if len(points) < 16:
        raise ValueError('waypoint CSV has too few points')
    return points


def closest_index(points, target, start=0):
    return min(range(start, len(points)),
               key=lambda i: ((points[i][0] - target[0]) ** 2
                              + (points[i][1] - target[1]) ** 2))


def create_project(waypoint_file, checkpoint_file, default_speed_kmh=20):
    points = load_csv(waypoint_file)
    checkpoints = load_checkpoints(checkpoint_file)
    speed = int(default_speed_kmh)
    if speed < 0 or speed > 200:
        raise ValueError('default speed must be 0..200 km/h')
    boundaries = []
    previous = 0
    for name in CHECKPOINT_ORDER:
        index = closest_index(points, checkpoints[name]['position'], previous)
        distance = math.dist(points[index][:2], checkpoints[name]['position'][:2])
        if distance > 1.0:
            raise ValueError('checkpoint {} is {:.2f} m from the CSV'.format(name, distance))
        boundaries.append(index)
        previous = index + 1
    if boundaries[0] != 0:
        raise ValueError('start checkpoint is not first CSV waypoint')
    if math.dist(points[-1][:2], points[0][:2]) > 1.0:
        raise ValueError('last CSV waypoint must return to start')
    boundaries.append(len(points) - 1)

    sectors = []
    next_id = 1
    for number in range(1, 16):
        first, last = boundaries[number - 1:number + 1]
        end_name = CHECKPOINT_ORDER[number] if number < 15 else 'start'
        lane = []
        for x, y, z in points[first:last + 1]:
            lane.append({'id': next_id, 'x': x, 'y': y, 'z': z,
                         'speed_kmh': speed})
            next_id += 1
        sectors.append({'number': number,
                        'from': CHECKPOINT_ORDER[number - 1],
                        'to': end_name,
                        'source_indices': [first, last],
                        'lane_count': 1,
                        'racelines': {'1': lane}})
    project = {'version': 1, 'map': 'R_KR_PR_K-city_2025', 'frame_id': 'map',
               'speed_unit': 'km/h', 'next_point_id': next_id,
               'sectors': sectors}
    validate_project(project)
    return project


def validate_project(project):
    if project.get('version') != 1 or len(project.get('sectors', [])) != 15:
        raise ValueError('unsupported map tuner project')
    if project.get('frame_id') != 'map' or project.get('speed_unit') != 'km/h':
        raise ValueError('unexpected coordinate frame or speed unit')
    ids = set()
    for number, sector in enumerate(project['sectors'], 1):
        count = sector.get('lane_count')
        if sector.get('number') != number or count not in range(1, MAX_LANES + 1):
            raise ValueError('invalid sector number or lane count')
        if set(sector.get('racelines', {})) != {str(i) for i in range(1, count + 1)}:
            raise ValueError('racelines do not match lane count in sector {}'.format(number))
        for lane in sector['racelines'].values():
            for point in lane:
                if not all(math.isfinite(float(point[key])) for key in ('x', 'y', 'z', 'speed_kmh')):
                    raise ValueError('non-finite waypoint data')
                if not 0 <= point['speed_kmh'] <= 200:
                    raise ValueError('speed outside 0..200 km/h')
                if point['id'] in ids:
                    raise ValueError('duplicate waypoint id')
                ids.add(point['id'])
    if project.get('next_point_id', 0) <= max(ids, default=0):
        raise ValueError('next_point_id is stale')
    for sector, lane in project.get('route_racelines', {}).items():
        if (not str(sector).isdecimal() or not 1 <= int(sector) <= 15
                or type(lane) is not int or not 1 <= lane <= MAX_LANES):
            raise ValueError('invalid route_racelines entry: {}={}'.format(sector, lane))


def sector_route(project, sector, raceline1_only=False):
    """Points and raceline numbers that form one sector's driven route.

    Raceline 1 wherever it is drawn: moving onto another raceline is a lane
    change for the planner (gap-checked), never a bend baked into the route.
    A sector listed in project['route_racelines'] continues on that raceline
    only after Raceline 1 ends (a lane drop), so the route still reaches the
    sector's end checkpoint.
    """
    r1 = sector['racelines']['1']
    choices = ({} if raceline1_only else
               dict(DEFAULT_ROUTE_RACELINES, **project.get('route_racelines', {})))
    lane = int(choices.get(str(sector['number']), 1))
    chosen = sector['racelines'].get(str(lane), []) if lane <= sector['lane_count'] else []
    if lane == 1 or len(chosen) < 2:
        return list(r1), [1]*len(r1)
    if len(r1) < 2:
        return list(chosen), [lane]*len(chosen)
    end = r1[-1]
    tx, ty = end['x']-r1[-2]['x'], end['y']-r1[-2]['y']
    beyond = [p for p in chosen if (p['x']-end['x'])*tx + (p['y']-end['y'])*ty > 0.]
    return list(r1) + beyond, [1]*len(r1) + [lane]*len(beyond)


# Same limits as the planner's lane change (lateral accel, max heading).
ROUTE_BLEND_LATERAL_ACCEL = 1.5
ROUTE_BLEND_MAX_HEADING_DEG = 10.


def blend_toward(points, target, reverse):
    """Shift the end of a polyline smoothly onto target (start if reverse).

    The anchor point (last, or first if reverse) moves fully onto target;
    earlier points move less with a smoothstep over the lane-change length.
    """
    if not points:
        return []
    order = list(reversed(points)) if not reverse else list(points)
    anchor = order[0]
    dx, dy = target['x']-anchor['x'], target['y']-anchor['y']
    lateral = math.hypot(dx, dy)
    if lateral < .3:
        return list(points)
    speed = max(0., float(target['speed_kmh']))/3.6
    length = max(speed*math.sqrt(6.*lateral/ROUTE_BLEND_LATERAL_ACCEL),
                 1.5*lateral/math.tan(math.radians(ROUTE_BLEND_MAX_HEADING_DEG)))
    result, travelled = [], 0.
    for k, point in enumerate(order):
        if k:
            travelled += math.hypot(point['x']-order[k-1]['x'], point['y']-order[k-1]['y'])
        t = max(0., min(1., 1.-travelled/length))
        weight = t*t*(3.-2.*t)
        moved = dict(point)
        moved['x'], moved['y'] = point['x']+weight*dx, point['y']+weight*dy
        result.append(moved)
    return list(reversed(result)) if not reverse else result


def load_project(path):
    with Path(path).open() as stream:
        project = json.load(stream)
    validate_project(project)
    return project


def primary_route(project, with_lanes=False, raceline1_only=False):
    """Return the driven route and its sector for each point.

    with_lanes also returns the raceline number each route point came from.
    raceline1_only ignores route_racelines: Raceline 1 in every sector (the
    Leo signal/tunnel route).
    """
    validate_project(project)
    route, sectors, lanes = [], [], []
    for sector in project['sectors']:
        points, numbers = sector_route(project, sector, raceline1_only)
        if len(points) < 2:
            raise ValueError('sector {} needs at least two route points'.format(
                sector['number']))
        for point, number in zip(points, numbers):
            if route and math.hypot(route[-1]['x'] - point['x'],
                                    route[-1]['y'] - point['y']) < 1e-6:
                continue
            route.append(point)
            sectors.append(sector['number'])
            lanes.append(number)
    if len(route) < 2:
        raise ValueError('project route has too few points')
    return (route, sectors, lanes) if with_lanes else (route, sectors)


def raceline_layout_settings(project, config_file, raceline_width=None):
    """Prefer the saved lane width, then YAML; an explicit width overrides both."""
    with Path(config_file).open() as stream:
        defaults = yaml.safe_load(stream) or {}
    saved = project.get('generated_racelines', {}).get('settings', {})
    width = (raceline_width if raceline_width not in (None, '') else
             saved.get('raceline_width', defaults.get('raceline_width', 3.5)))
    return float(width)


def checkpoint_route_indices(project, route, sectors, checkpoint_file=None):
    """Route index of each sector-end checkpoint at its fixed surveyed position.

    The position comes from map_tunner_checkpoints.yaml, not from the saved
    points, so deleting or replacing a sector's end point does not move it.
    """
    path = (Path(checkpoint_file) if checkpoint_file is not None else
            Path(__file__).resolve().parents[1] / 'config/map_tunner_checkpoints.yaml')
    checkpoints = load_checkpoints(path)
    result = {}
    for sector in project['sectors']:
        name = str(sector['to'])
        x, y = checkpoints[name]['position'][:2]
        candidates = [i for i, s in enumerate(sectors) if s == sector['number']]
        result[name] = min(candidates, key=lambda i: math.hypot(
            route[i]['x'] - x, route[i]['y'] - y))
    return result


def saved_lane_indices(project, route, sectors, route_lanes=None):
    """Map saved lanes to route progress, including manual southbound S11 points.

    A route point never maps the raceline it was taken from (that lane IS the
    route there), so in an R3-driven S11 Raceline 1 becomes a side lane.
    """
    route_ids = {p['id']: i for i, p in enumerate(route)}
    if route_lanes is None:
        route_lanes = [1]*len(route)
    result = {n: {} for n in (1, 2, 3)}
    for sector in project['sectors']:
        for n in (1, 2, 3):
            points = sector['racelines'].get(str(n), [])
            result[n].update({route_ids[p['source_id']]: (p['x'], p['y'])
                              for p in points if p.get('source_id') in route_ids
                              and route_lanes[route_ids[p['source_id']]] != n})
            if sector['number'] != 11 or len(points) < 2:
                continue
            # Manual points have no source_id. Interpolate inside their saved
            # Y range; never truncate the lane at the last generated point.
            ordered = sorted(points, key=lambda p: p['y'])
            ys = [p['y'] for p in ordered]
            for i, point in enumerate(route):
                y = point['y']
                if sectors[i] != 11 or route_lanes[i] == n or not ys[0] <= y <= ys[-1]:
                    continue
                j = bisect_left(ys, y)
                if j == 0 or ys[j] == y:
                    result[n][i] = (ordered[j]['x'], y)
                else:
                    a, b = ordered[j-1], ordered[j]
                    t = (y-a['y'])/(b['y']-a['y'])
                    result[n][i] = (a['x']+t*(b['x']-a['x']), y)
    return result


def generate_racelines(project, raceline_width=3.5, waypoint_file=None):
    """Keep Sector 11 Raceline 1 on the CSV and translate R2/R3 to its left."""
    width = float(raceline_width)
    if not math.isfinite(width) or width <= 0:
        raise ValueError('raceline width must be positive and finite')
    validate_project(project)
    result = copy.deepcopy(project)
    sector = result['sectors'][10]
    csv_path = (Path(waypoint_file).expanduser() if waypoint_file is not None else
                Path(__file__).resolve().parents[1] / 'config/waypoints.csv')
    csv_points = load_csv(csv_path)
    first, last = sector['source_indices']
    reference = csv_points[first:last + 1]
    current_r1 = sector['racelines']['1']
    # Raceline 1 may end early where its lane drops before CP11 (its tail
    # trimmed in the project); it must never extend past the CSV.
    if not 0 < len(current_r1) <= len(reference):
        raise ValueError('Sector 11 Raceline 1 point count differs from waypoints.csv')
    reference = reference[:len(current_r1)]
    settings = {'raceline_width': width}
    existing = project.get('generated_racelines')
    if (existing and existing.get('settings') == settings
            and existing.get('layout_version') == 2
            and all((point['x'], point['y'], point['z']) == tuple(xyz)
                    for point, xyz in zip(current_r1, reference))):
        return result

    # Keep waypoint IDs and edited speed limits; restore the CSV's exact XYZ.
    base = copy.deepcopy(current_r1)
    for point, (x, y, z) in zip(base, reference):
        point.update(x=x, y=y, z=z)
    anchor = min(range(len(base)), key=lambda i:
                 math.hypot(base[i]['x'] - 63.47, base[i]['y'] - 146.13))
    anchor_x = base[anchor]['x']
    start = next((i for i in range(anchor + 1, len(base))
                  if base[i]['x'] > anchor_x), None)
    if start is None:
        raise ValueError('Sector 11 never crosses the reference waypoint X')
    stop = next((i for i in range(start, len(base))
                 if base[i]['y'] < -122.90), None)
    if stop is None or stop - start < 2:
        raise ValueError('Sector 11 must continue south across y=-122.90')
    if any(base[i + 1]['y'] > base[i]['y'] for i in range(start, stop - 1)):
        raise ValueError('three-lane reference must run south')
    if not existing and any(sector['racelines'].get(str(n)) for n in (2, 3)):
        raise ValueError('reference sector already contains manual R2/R3; use a separate project file')

    lanes = {'1': base}
    for number in (2, 3):
        previous = {p.get('source_id'): p['id']
                    for p in sector['racelines'].get(str(number), [])}
        lane = []
        for original in base[start:stop]:
            point = copy.deepcopy(original)
            identifier = previous.get(original['id'])
            if identifier is None:
                identifier = result['next_point_id']
                result['next_point_id'] += 1
            point.update(id=identifier, source_id=original['id'],
                         x=original['x'] + (number - 1) * width)
            lane.append(point)
        lanes[str(number)] = lane
    sector['lane_count'] = 3
    sector['racelines'] = lanes
    result['generated_racelines'] = {
        'settings': settings, 'layout_version': 2, 'sector': sector['number'],
        'start_index': start, 'stop_index': stop,
        'reference': [63.47, 146.13], 'end_y': -122.90,
        'base_raceline1': copy.deepcopy(base),
    }
    validate_project(result)
    return result


def save_project(path, project):
    validate_project(project)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + target.name + '.',
                                              suffix='.tmp', dir=str(target.parent))
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(project, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ProjectEditor:
    def __init__(self, project):
        validate_project(project)
        self.project = copy.deepcopy(project)
        self.history = []
        self.dirty = False

    def sector(self, number):
        if number not in range(1, 16):
            raise ValueError('sector must be 1..15')
        return self.project['sectors'][number - 1]

    def lane(self, sector, lane):
        data = self.sector(sector)
        key = str(lane)
        if key not in data['racelines']:
            raise ValueError('raceline {} does not exist in sector {}'.format(lane, sector))
        return data['racelines'][key]

    def _checkpoint(self):
        self.history.append(copy.deepcopy(self.project))
        self.history = self.history[-30:]
        self.dirty = True

    def undo(self):
        if not self.history:
            return False
        self.project = self.history.pop()
        self.dirty = True
        return True

    def add_lane(self, sector):
        data = self.sector(sector)
        if data['lane_count'] >= MAX_LANES:
            raise ValueError('sector already has three lanes')
        self._checkpoint()
        data['lane_count'] += 1
        data['racelines'][str(data['lane_count'])] = []
        return data['lane_count']

    def remove_lane(self, sector):
        data = self.sector(sector)
        if data['lane_count'] <= 1:
            raise ValueError('raceline1 cannot be removed')
        self._checkpoint()
        removed = data['lane_count']
        del data['racelines'][str(removed)]
        data['lane_count'] -= 1
        return removed

    def add_point(self, sector, lane, x, y, z, speed_kmh, index=None):
        values = (x, y, z, speed_kmh)
        if not all(math.isfinite(float(v)) for v in values) or not 0 <= speed_kmh <= 200:
            raise ValueError('invalid point or speed')
        points = self.lane(sector, lane)
        if index is None:
            index = len(points)
        if not 0 <= index <= len(points):
            raise ValueError('insert index outside raceline')
        self._checkpoint()
        point = {'id': self.project['next_point_id'], 'x': float(x), 'y': float(y),
                 'z': float(z), 'speed_kmh': int(speed_kmh)}
        self.project['next_point_id'] += 1
        points.insert(index, point)
        return point

    def change_point(self, sector, lane, point_id, **updates):
        points = self.lane(sector, lane)
        point = next((item for item in points if item['id'] == point_id), None)
        if point is None:
            raise ValueError('waypoint not found in selected raceline')
        changed = dict(point)
        changed.update(updates)
        if not all(math.isfinite(float(changed[key])) for key in ('x', 'y', 'z', 'speed_kmh')):
            raise ValueError('non-finite waypoint data')
        if not 0 <= changed['speed_kmh'] <= 200:
            raise ValueError('speed outside 0..200 km/h')
        self._checkpoint()
        point.update(changed)
        return point

    def set_speed_range(self, sector, lane, start_id, end_id, speed_kmh):
        """Set one speed on an inclusive span of one sector raceline."""
        points = self.lane(sector, lane)
        indices = {point['id']: index for index, point in enumerate(points)}
        if start_id not in indices or end_id not in indices:
            raise ValueError('시작점과 끝점이 현재 Raceline에 있어야 합니다.')
        if isinstance(speed_kmh, bool) or not isinstance(speed_kmh, int) or not 0 <= speed_kmh <= 200:
            raise ValueError('속도는 0~200 km/h 정수로 입력하세요.')
        first, last = sorted((indices[start_id], indices[end_id]))
        if all(point['speed_kmh'] == speed_kmh for point in points[first:last + 1]):
            return last - first + 1
        self._checkpoint()
        for point in points[first:last + 1]:
            point['speed_kmh'] = speed_kmh
        return last - first + 1

    def remove_point(self, sector, lane, point_id):
        points = self.lane(sector, lane)
        index = next((i for i, point in enumerate(points) if point['id'] == point_id), None)
        if index is None:
            raise ValueError('waypoint not found in selected raceline')
        self._checkpoint()
        return points.pop(index)

    def remove_range(self, sector, lane, start_id, end_id):
        """Delete an inclusive span of one sector raceline as one undo step."""
        if lane == 1 and sector != 11:
            # Raceline 1 is the driven route and checkpoint reference.
            raise ValueError('Raceline 1 구간 삭제는 S11에서만 할 수 있습니다.')
        points = self.lane(sector, lane)
        indices = {point['id']: index for index, point in enumerate(points)}
        if start_id not in indices or end_id not in indices:
            raise ValueError('시작점과 끝점이 현재 Raceline에 있어야 합니다.')
        first, last = sorted((indices[start_id], indices[end_id]))
        self._checkpoint()
        del points[first:last + 1]
        return last - first + 1

    def append_continuous_points(self, sector, lane, start_xy, end_xy, interior_count):
        """Append both endpoints and the requested number of evenly spaced points."""
        if (isinstance(interior_count, bool) or not isinstance(interior_count, int)
                or not 0 <= interior_count <= 10000):
            raise ValueError('사이 점 개수는 0~10000 정수로 입력하세요')
        coordinates = tuple(start_xy) + tuple(end_xy)
        if len(coordinates) != 4 or not all(math.isfinite(float(v)) for v in coordinates):
            raise ValueError('유효한 시작점과 끝점을 찍어주세요')
        x0, y0, x1, y1 = map(float, coordinates)
        if math.hypot(x1 - x0, y1 - y0) < 1e-6:
            raise ValueError('시작점과 끝점은 서로 다른 위치에 찍어주세요')
        points = self.lane(sector, lane)
        reference = points or self.lane(sector, 1)
        if not reference:
            raise ValueError('높이 기준 경로점이 필요합니다')
        first, _ = nearest_point(reference, x0, y0)
        last, _ = nearest_point(reference, x1, y1)
        z0, z1 = reference[first]['z'], reference[last]['z']
        speed = reference[first]['speed_kmh']
        count = interior_count + 2
        self._checkpoint()
        for index in range(count):
            fraction = index / (count - 1)
            points.append({'id': self.project['next_point_id'],
                           'x': x0 + (x1 - x0) * fraction,
                           'y': y0 + (y1 - y0) * fraction,
                           'z': z0 + (z1 - z0) * fraction,
                           'speed_kmh': speed})
            self.project['next_point_id'] += 1
        return count

    def smooth_neighborhood(self, sector, lane, point_id, radius=15):
        """Smooth XY near one point using up to 15 neighbours on each side."""
        points = self.lane(sector, lane)
        center = next((i for i, point in enumerate(points)
                       if point['id'] == point_id), None)
        if center is None:
            raise ValueError('선택한 점이 Raceline에 없습니다')
        if len(points) < 3:
            raise ValueError('스무딩하려면 점이 세 개 이상 필요합니다')
        first = max(0, center - radius)
        last = min(len(points) - 1, center + radius)
        original = [(point['x'], point['y']) for point in points]
        updated = {}
        sigma = max(1.0, radius / 3.0)
        for index in range(first + 1, last):
            low = max(0, index - radius)
            high = min(len(points), index + radius + 1)
            weights = [math.exp(-0.5 * ((i - index) / sigma) ** 2)
                       for i in range(low, high)]
            total = sum(weights)
            average_x = sum(original[i][0] * weights[i - low]
                            for i in range(low, high)) / total
            average_y = sum(original[i][1] * weights[i - low]
                            for i in range(low, high)) / total
            fade = math.sin(math.pi * (index - first) / (last - first)) ** 2
            updated[index] = (original[index][0] * (1 - fade) + average_x * fade,
                              original[index][1] * (1 - fade) + average_y * fade)
        self._checkpoint()
        for index, (x, y) in updated.items():
            points[index]['x'], points[index]['y'] = x, y
        return len(updated)


def nearest_point(points, x, y):
    if not points:
        return None, float('inf')
    index = min(range(len(points)),
                key=lambda i: (points[i]['x'] - x) ** 2 + (points[i]['y'] - y) ** 2)
    return index, math.hypot(points[index]['x'] - x, points[index]['y'] - y)


def insert_index(points, x, y):
    """Insert after the closest path segment; append when fewer than two points."""
    if len(points) < 2:
        return len(points)
    best = None
    for i, (first, last) in enumerate(zip(points, points[1:])):
        dx, dy = last['x'] - first['x'], last['y'] - first['y']
        length2 = dx * dx + dy * dy
        t = 0 if length2 == 0 else max(0, min(1, ((x - first['x']) * dx + (y - first['y']) * dy) / length2))
        distance2 = (x - first['x'] - t * dx) ** 2 + (y - first['y'] - t * dy) ** 2
        if best is None or distance2 < best[0]:
            best = (distance2, i + 1)
    return best[1]
