#!/usr/bin/env python3
"""Persistent, ROS-independent data model for the K-City RViz map tuner."""

import copy
import json
import math
import os
import re
import tempfile
from pathlib import Path

import yaml


CHECKPOINT_ORDER = ['start'] + [str(number) for number in range(1, 15)]
MAX_LANES = 3


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


def load_project(path):
    with Path(path).open() as stream:
        project = json.load(stream)
    validate_project(project)
    return project


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

    def fill_interpolated_lane(self, sector, lane, start_xy, end_xy):
        """Fill an empty raceline with the raceline 1 point count, including endpoints."""
        points = self.lane(sector, lane)
        reference = self.lane(sector, 1)
        if points:
            raise ValueError('선 보간은 빈 Raceline에서만 가능합니다')
        if len(reference) < 2:
            raise ValueError('Raceline 1에 기준점이 두 개 이상 필요합니다')
        coordinates = tuple(start_xy) + tuple(end_xy)
        if len(coordinates) != 4 or not all(math.isfinite(float(v)) for v in coordinates):
            raise ValueError('유효한 시작점과 끝점을 찍어주세요')
        x0, y0, x1, y1 = map(float, coordinates)
        count = len(reference)
        z0, z1 = reference[0]['z'], reference[-1]['z']
        self._checkpoint()
        for index, original in enumerate(reference):
            fraction = index / (count - 1)
            points.append({'id': self.project['next_point_id'],
                           'x': x0 + (x1 - x0) * fraction,
                           'y': y0 + (y1 - y0) * fraction,
                           'z': z0 + (z1 - z0) * fraction,
                           'speed_kmh': original['speed_kmh']})
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
