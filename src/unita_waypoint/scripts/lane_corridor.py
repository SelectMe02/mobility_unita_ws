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
        primary, separate, labelled = [], [], []
        for lane in geometry['lanes']:
            if type(lane['sector']) is not int or not 1 <= lane['sector'] <= 15:
                raise ValueError('invalid lane sector')
            points = np.asarray(lane['points'], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError('invalid raceline coordinates')
            if lane['raceline'] not in (1, 2, 3):
                raise ValueError('invalid raceline number')
            labelled.extend((a, b, lane['sector'], lane['raceline'])
                            for a, b in zip(points, points[1:])
                            if np.linalg.norm(b-a) > 1e-6)
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
        # Keep lane identities separate from the union used by contains().
        # Adjacent finite lanes must never be extrapolated across a gap.
        starts = np.asarray([item[0] for item in labelled]).reshape(-1, 2)
        vectors = np.asarray([item[1]-item[0] for item in labelled]).reshape(-1, 2)
        lengths = np.linalg.norm(vectors, axis=1)
        counts = np.maximum(1, np.ceil(lengths).astype(int))
        indices = np.repeat(np.arange(len(starts)), counts)
        offsets = np.repeat(np.cumsum(counts)-counts, counts)
        t = (np.arange(len(indices))-offsets)/counts[indices]
        self.label_starts = starts[indices]+t[:, None]*vectors[indices]
        self.label_vectors = vectors[indices]/counts[indices, None]
        self.label_lengths = lengths[indices]/counts[indices]
        self.label_ids = [(labelled[i][2], labelled[i][3]) for i in indices]
        self.label_tree = cKDTree(self.label_starts+.5*self.label_vectors)

    def label_box(self, x, y, length, width, yaw):
        """Assign a box midpoint to a finite raceline; retain boundary evidence."""
        unknown = dict(raceline=0, lane_sector=0, lane_offset_m=0.,
                       lane_boundary_crossing=True, lane_label_confident=False)
        if not all(math.isfinite(v) for v in (x, y, length, width, yaw)):
            return unknown
        nearby = self.label_tree.query_ball_point([x, y], self.width*.5+.5+1e-9)
        if not nearby:
            return unknown
        nearby = np.asarray(nearby)
        offset = np.array([x, y])-self.label_starts[nearby]
        tangent = self.label_vectors[nearby]/self.label_lengths[nearby, None]
        along = np.sum(offset*tangent, axis=1)
        lateral = tangent[:, 0]*offset[:, 1]-tangent[:, 1]*offset[:, 0]
        valid = ((along >= -1e-9) & (along <= self.label_lengths[nearby]+1e-9)
                 & (np.abs(lateral) <= self.width*.5+1e-9))
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            return unknown
        nearest = min(candidates, key=lambda i: abs(lateral[i]))
        sector, lane = self.label_ids[nearby[nearest]]
        other = [abs(lateral[i]) for i in candidates
                 if self.label_ids[nearby[i]][1] != lane]
        confident = not other or min(other)-abs(lateral[nearest]) >= .15
        heading = math.atan2(tangent[nearest, 1], tangent[nearest, 0])
        half_span = (abs(math.sin(yaw-heading))*length
                     + abs(math.cos(yaw-heading))*width)*.5
        return dict(raceline=lane, lane_sector=sector,
                    lane_offset_m=float(lateral[nearest]),
                    lane_boundary_crossing=bool(abs(lateral[nearest])+half_span
                                                > self.width*.5),
                    lane_label_confident=bool(confident))

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

    def depth(self, xy):
        """Distance inside the lane strip from its nearest edge.

        Negative just outside the strip; -inf far from every saved segment.

        Measured across each saved segment, so a barrier following a bend
        stays at the same depth along its whole length.
        """
        points = np.asarray(xy, dtype=float).reshape((-1, 2))
        result = np.full(len(points), -np.inf)
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
            # Distance to the segment itself, so points outside a polyline
            # vertex on a bend are measured from that vertex.
            t = np.clip(np.sum(offset*vectors, axis=1)/self.squared[segment_ids], 0., 1.)
            depth = self.width*.5-np.linalg.norm(offset-t[:, None]*vectors, axis=1)
            np.maximum.at(result, start+point_ids, depth)
        return result

    def direction(self, xy):
        """Unit tangent of the nearest saved lane segment (travel direction)."""
        _, index = self.tree.query(np.asarray(xy, dtype=float))
        vector = self.vectors[index]
        return vector/max(float(np.hypot(*vector)), 1e-9)

    def filter_obstacles(self, obstacles):
        if not obstacles:
            return []
        mask = self.contains([(o['x'], o['y']) for o in obstacles])
        return [o for o, inside in zip(obstacles, mask) if inside]
