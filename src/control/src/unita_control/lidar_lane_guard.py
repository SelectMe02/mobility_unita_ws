"""Conservative LiDAR marking extraction and steering clearance checks.

The waypoint follower remains responsible for navigation.  These functions only
look for reflective, road-height lines near the vehicle and reject commands
whose short-horizon swept corridor intersects a detected line.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MarkingLine:
    slope: float
    intercept: float
    x_min: float
    x_max: float
    support: int
    x_bins: int

    def lateral_at(self, forward):
        return self.slope * forward + self.intercept


def extract_marking_lines(points, max_lines=5):
    """Fit locally straight reflective markings in vehicle-frame XY.

    ``points`` is N x 4: forward, left, up, return intensity.  A line is a
    candidate, not a semantic solid-line label.  Ground and contrast gates keep
    high returns from buildings and roadside objects out of the fit.
    """
    cloud = np.asarray(points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 4:
        raise ValueError('points must be an N x 4 array')
    if not len(cloud):
        raise ValueError('empty LiDAR scan')
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if not len(cloud):
        raise ValueError('no finite LiDAR points')
    x, y, z, intensity = cloud.T
    region = ((x >= 3.0) & (x <= 18.0) & (np.abs(y) <= 7.5)
              & (z > -3.0) & (z < 0.0))
    central_ground = region & (x <= 12.0) & (np.abs(y) <= 1.5)
    if np.count_nonzero(central_ground) < 30:
        raise ValueError('no central road-height returns')
    ground_z = float(np.median(z[central_ground]))
    road = region & (np.abs(z - ground_z) <= 0.25)
    if np.count_nonzero(road) < 80:
        raise ValueError('too few road-height returns')
    # In this MORAI sensor profile yellow paint is often 45-55 intensity,
    # while asphalt is near 10.  A 55 floor missed the solid yellow shoulder
    # line around 750-800 m of the K-city route.
    threshold = max(45.0, float(np.median(intensity[road])) + 35.0)
    bright = road & (intensity >= threshold) & (np.abs(y) >= 0.8)
    xy = np.column_stack((x[bright], y[bright]))
    if len(xy) < 8:
        return []

    # A fixed seed makes the RANSAC result repeatable for the same scan.
    random = np.random.default_rng(20261014)
    remaining = xy
    lines = []
    for _ in range(max_lines):
        if len(remaining) < 8:
            break
        bx, by = remaining.T
        best = None
        for _ in range(400):
            pair = random.choice(len(remaining), size=2, replace=False)
            first, second = pair
            separation = float(bx[second] - bx[first])
            if abs(separation) < 3.0:
                continue
            slope = float((by[second] - by[first]) / separation)
            # Over this 15 m horizon a lane edge is locally near-parallel to
            # the vehicle.  Steep painted arrows and transverse markings must
            # not trigger a spurious evasive turn.
            if abs(slope) > 0.35:
                continue
            intercept = float(by[first] - slope * bx[first])
            inliers = np.abs(by - (slope * bx + intercept)) <= 0.17
            count = int(np.count_nonzero(inliers))
            if count < 8:
                continue
            selected_x = bx[inliers]
            span = float(np.ptp(selected_x))
            bins = len(np.unique(np.floor(selected_x).astype(int)))
            if span < 5.0 or bins < 4:
                continue
            score = count * min(span, 12.0)
            if best is None or score > best[0]:
                best = (score, inliers)
        if best is None:
            break
        inliers = best[1]
        fit_x, fit_y = remaining[inliers].T
        slope, intercept = np.polyfit(fit_x, fit_y, 1)
        if (math.isfinite(slope) and math.isfinite(intercept)
                and abs(slope) <= 0.35):
            lines.append(MarkingLine(float(slope), float(intercept),
                                     float(np.min(fit_x)), float(np.max(fit_x)),
                                     len(fit_x),
                                     len(np.unique(np.floor(fit_x).astype(int)))))
        remaining = remaining[~inliers]
    return lines


def clearance_for_steering(lines, steering, wheelbase=2.7, half_width=0.90,
                           margin=0.20):
    """Smallest predicted vehicle clearance to any observed marking.

    The 0.90 m half width approximates the Niro body, while the 0.20 m margin
    covers the LiDAR fit and tracking error.  An effective 1.30 m on each side
    falsely trapped the car in an observed 2.8-3.0 m lane.
    """
    curvature = math.tan(float(steering)) / wheelbase
    clearance = float('inf')
    for line in lines:
        start = max(2.0, line.x_min - 0.5)
        end = min(10.0, line.x_max + 0.5)
        if end <= start:
            continue
        distances = np.linspace(start, end, 12)
        # The exact circle matters little over this short 10 m horizon, but
        # avoids optimistic clearance at larger steering angles.
        if abs(curvature) < 1e-6:
            predicted_y = np.zeros_like(distances)
        else:
            radicand = 1.0 - np.square(curvature * distances)
            if np.any(radicand <= 0):
                return -float('inf')
            predicted_y = (1.0 - np.sqrt(radicand)) / curvature
        gap = np.abs(line.slope * distances + line.intercept - predicted_y)
        clearance = min(clearance, float(np.min(gap)) - half_width - margin)
    return clearance


def choose_steering(requested, lines, max_delta=0.12, max_steering=0.632681853):
    """Return a nearby steering angle that does not intersect detected lines.

    A negative clearance means every sampled alternative intersects a line;
    callers must command a held stop rather than guessing a lane change.
    """
    requested = float(requested)
    if not math.isfinite(requested):
        raise ValueError('non-finite steering')
    requested = max(-max_steering, min(max_steering, requested))
    if not lines:
        return requested, float('inf')
    options = np.linspace(-max_delta, max_delta, 25) + requested
    options = np.clip(options, -max_steering, max_steering)
    best = None
    for steering in options:
        clearance = clearance_for_steering(lines, steering)
        if clearance < 0.0:
            continue
        cost = abs(float(steering) - requested)
        if best is None or cost < best[0]:
            best = (cost, float(steering), clearance)
    if best is None:
        return requested, -1.0
    return best[1], best[2]
