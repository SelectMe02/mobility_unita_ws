"""Finite raceline strips: full lane width, shared by perception and debug."""
import math

import numpy as np
from scipy.spatial import cKDTree


def project_lane_geometry(project, width, loop=True):
    lanes = []
    for sector in project['sectors']:
        for number, points in sector['racelines'].items():
            if points:
                lanes.append(dict(sector=sector['number'], raceline=int(number),
                                  points=[[float(p['x']), float(p['y'])] for p in points]))
    return dict(frame_id='map', raceline_width=float(width), loop=bool(loop), lanes=lanes)


class LaneCorridor:
    def __init__(self, geometry):
        self.width = float(geometry['raceline_width'])
        if geometry['frame_id'] != 'map' or not math.isfinite(self.width) or self.width <= 0.:
            raise ValueError('invalid map lane width/frame')
        primary, separate = [], []
        for lane in geometry['lanes']:
            if type(lane['sector']) is not int or not 1 <= lane['sector'] <= 15:
                raise ValueError('invalid lane sector')
            points = np.asarray(lane['points'], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError('invalid raceline coordinates')
            if lane['raceline'] not in (1, 2, 3):
                raise ValueError('invalid raceline number')
            if lane['raceline'] == 1:
                primary.extend(points.tolist())
            else:
                # Never join adjacent lines across an unmapped interval.
                separate.append(points)
        if primary:
            if geometry.get('loop', True):
                primary.append(primary[0])
            separate.append(np.asarray(primary))
        starts, ends = [], []
        for points in separate:
            if len(points) > 1:
                starts.extend(points[:-1])
                ends.extend(points[1:])
        if not starts:
            raise ValueError('lane geometry has no segments')
        a, b = np.asarray(starts), np.asarray(ends)
        squared = np.sum((b-a)**2, axis=1)
        valid = squared > 1e-12
        a, vectors, squared = a[valid], (b-a)[valid], squared[valid]
        if not len(a):
            raise ValueError('lane geometry has no length')
        # Split long saved segments only for spatial indexing. Their exact
        # line/width stays the same, and sparse manual waypoints remain fast.
        counts = np.maximum(1, np.ceil(np.sqrt(squared)).astype(int))
        indices = np.repeat(np.arange(len(a)), counts)
        offsets = np.repeat(np.cumsum(counts)-counts, counts)
        t = (np.arange(len(indices))-offsets)/counts[indices]
        self.starts = a[indices]+t[:, None]*vectors[indices]
        self.vectors = vectors[indices]/counts[indices, None]
        self.squared = np.sum(self.vectors*self.vectors, axis=1)
        self.tree = cKDTree(self.starts+.5*self.vectors)
        self.search_radius = self.width*.5+.5*float(np.sqrt(self.squared).max())

    def contains(self, xy):
        points = np.asarray(xy, dtype=float).reshape((-1, 2))
        result = np.zeros(len(points), dtype=bool)
        # Bound temporary allocations for dense point clouds.
        for start in range(0, len(points), 2048):
            batch = points[start:start+2048]
            finite = np.isfinite(batch).all(axis=1)
            indices = np.flatnonzero(finite)
            neighbors = self.tree.query_ball_point(batch[finite], self.search_radius+1e-9)
            counts = np.asarray([len(n) for n in neighbors], dtype=int)
            if not counts.sum():
                continue
            point_ids = np.repeat(indices, counts)
            segment_ids = np.concatenate([n for n in neighbors if n]).astype(int)
            offset = batch[point_ids]-self.starts[segment_ids]
            vectors = self.vectors[segment_ids]
            t = np.sum(offset*vectors, axis=1)/self.squared[segment_ids]
            lateral = offset-t[:, None]*vectors
            inside = ((t >= -1e-9) & (t <= 1.+1e-9) &
                      (np.sum(lateral*lateral, axis=1) <= (self.width*.5)**2+1e-9))
            np.logical_or.at(result, start+point_ids, inside)
        return result

    def filter_obstacles(self, obstacles):
        if not obstacles:
            return []
        mask = self.contains([(o['x'], o['y']) for o in obstacles])
        return [o for o, inside in zip(obstacles, mask) if inside]
