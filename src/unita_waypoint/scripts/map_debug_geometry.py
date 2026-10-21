"""Geometry for the debug view only; never feeds vehicle control."""
import math

import numpy as np


def obstacle_z_bounds(points, x, y, length, width, yaw):
    """Observed map-frame Z inside a detector footprint, or no measurement.

    This is display geometry from the selected scan, not a new clustering or
    ground estimate. Small XY tolerance covers float32 cloud serialization.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return None
    dx, dy = points[:, 0]-x, points[:, 1]-y
    c, s = math.cos(yaw), math.sin(yaw)
    inside = (np.isfinite(points).all(axis=1)
              & (np.abs(c*dx+s*dy) <= length*.5+.01)
              & (np.abs(-s*dx+c*dy) <= width*.5+.01))
    z = points[inside, 2]
    return (float(z.min()), float(z.max())) if len(z) else None


def nearby_lane_edges(corridor, position, radius=70.):
    """Edges of the same finite strips used by the detector (no extrapolation)."""
    offset = np.asarray(position[:2])-corridor.starts
    t = np.clip(np.sum(offset*corridor.vectors, axis=1)/corridor.squared, 0., 1.)
    near = np.linalg.norm(offset-t[:, None]*corridor.vectors, axis=1) <= radius
    starts, vectors = corridor.starts[near], corridor.vectors[near]
    if not len(starts):
        return np.empty((0, 2)), np.empty((0, 2))
    normals = np.column_stack((-vectors[:, 1], vectors[:, 0]))
    normals *= (corridor.width*.5/np.linalg.norm(vectors, axis=1))[:, None]
    edges = np.stack((starts+normals, starts+vectors+normals,
                      starts-normals, starts+vectors-normals), axis=1).reshape(-1, 2)
    centers = np.stack((starts, starts+vectors), axis=1).reshape(-1, 2)
    return edges, centers


def vehicle_box_corners(pose, front, rear, half_width, base, height):
    """Rear axle origin, x forward and y left, matching perception geometry."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x+c*dx-s*dy, y+s*dx+c*dy, z)
            for z in (base, base+height)
            for dx, dy in ((-rear, -half_width), (front, -half_width),
                           (front, half_width), (-rear, half_width))]


BOX_EDGE_PAIRS = ([(i, (i+1) % 4) for i in range(4)]
                  + [(i+4, (i+1) % 4+4) for i in range(4)]
                  + [(i, i+4) for i in range(4)])
