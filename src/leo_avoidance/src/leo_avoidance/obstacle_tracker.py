"""Track LiDAR obstacle clusters across scans to separate moving from static.

Cluster centres are placed in the odometry frame with the Ego pose captured
at each scan, so Ego motion does not look like obstacle motion. Velocity is a
least-squares slope over a short history; classification needs several
consistent scans and releases with hysteresis.

A static object's visible face changes as Ego approaches or passes it, which
moves the cluster centre. Motion therefore requires the opposite measured
edges to translate together in the world frame. Geometry remains measured;
only the motion estimate rejects changes of visible extent.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from leo_avoidance.ftg_visualization import cluster_boxes


@dataclass(frozen=True)
class TrackedObstacle:
    """One cluster seen in the current scan, in the current LiDAR frame."""
    track_id: int
    x_m: float
    y_m: float
    size_x_m: float
    size_y_m: float
    vx_mps: float
    vy_mps: float
    velocity_known: bool
    dynamic: bool
    # Consecutive scans with lateral speed above the threshold; signed +y/-y.
    lateral_scans: int = 0


@dataclass
class _Track:
    track_id: int
    history: list = field(default_factory=list)  # (stamp, world_x, world_y)
    velocity: tuple = (0., 0.)
    velocity_known: bool = False
    moving_scans: int = 0
    still_scans: int = 0
    dynamic: bool = False
    size: tuple = None
    lateral_scans: int = 0
    velocity_se: tuple = (math.inf, math.inf)
    bounds_history: list = field(default_factory=list)
    last_trusted_motion: tuple = None
    velocity_held: bool = False


class ObstacleTracker:
    def __init__(self, lidar_to_rear_x_m, gate_m=1.5, history_s=.6,
                 min_span_s=.2, min_samples=3, dynamic_speed_mps=.8,
                 static_speed_mps=.4, confirm_scans=2, lost_s=.5,
                 min_points=3, max_size_m=6., region_x_m=(-6., 25.),
                 region_half_width_m=12., lateral_speed_mps=.5,
                 max_size_jump_m=.5, significance=2.5, min_noise_m=.05):
        self.lidar_to_rear_x_m = float(lidar_to_rear_x_m)
        self.gate_m = float(gate_m)
        self.history_s = float(history_s)
        self.min_span_s = float(min_span_s)
        self.min_samples = int(min_samples)
        self.dynamic_speed_mps = float(dynamic_speed_mps)
        self.static_speed_mps = float(static_speed_mps)
        self.confirm_scans = int(confirm_scans)
        self.lost_s = float(lost_s)
        self.min_points = int(min_points)
        self.max_size_m = float(max_size_m)
        self.region_x_m = tuple(float(v) for v in region_x_m)
        self.region_half_width_m = float(region_half_width_m)
        self.lateral_speed_mps = float(lateral_speed_mps)
        self.max_size_jump_m = float(max_size_jump_m)
        # Motion counts only when the velocity exceeds this many standard
        # errors of its own fit; measurement noise is never below min_noise_m.
        self.significance = float(significance)
        self.min_noise_m = float(min_noise_m)
        if (not all(math.isfinite(v) and v > 0 for v in
                    (self.gate_m, self.history_s, self.min_span_s,
                     self.dynamic_speed_mps, self.static_speed_mps,
                     self.lost_s, self.max_size_m, self.region_half_width_m,
                     self.lateral_speed_mps, self.max_size_jump_m,
                     self.significance, self.min_noise_m))
                or self.min_span_s > self.history_s
                or self.static_speed_mps >= self.dynamic_speed_mps
                or self.min_samples < 2 or self.confirm_scans < 1
                or self.min_points < 1
                or not self.region_x_m[0] < self.region_x_m[1]):
            raise ValueError('invalid obstacle tracker limits')
        self.reset()

    def reset(self):
        self.tracks = []
        self.last_stamp = None
        self._next_id = 1

    def _clusters(self, points):
        points = points[(points[:, 0] >= self.region_x_m[0])
                        & (points[:, 0] <= self.region_x_m[1])
                        & (np.abs(points[:, 1]) <= self.region_half_width_m)]
        clusters = []
        edge_m = .3
        for low, high, count, _, _, _, _, group in cluster_boxes(points, oriented=True):
            size = high[:2] - low[:2]
            if count < self.min_points or np.max(size) > self.max_size_m:
                continue
            if (low[0] <= self.region_x_m[0] + edge_m
                    or high[0] >= self.region_x_m[1] - edge_m
                    or np.max(np.abs((low[1], high[1])))
                    >= self.region_half_width_m - edge_m):
                # Cut by the region: its centre slides as Ego moves.
                continue
            clusters.append(((low[:2] + high[:2]) / 2., size, group[:, :2]))
        return clusters

    def update(self, points_lidar_m, ego_pose, stamp):
        """Associate this scan's clusters; return them with motion estimates."""
        if (ego_pose is None or stamp is None or len(ego_pose) != 3
                or not all(math.isfinite(v) for v in (*ego_pose, stamp))):
            self.reset()
            return []
        if (self.last_stamp is not None
                and not 0. < stamp - self.last_stamp <= self.lost_s):
            # A scan gap or time jump makes world positions incomparable.
            self.reset()
        points = np.asarray(points_lidar_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            self.last_stamp = stamp
            self.tracks = [t for t in self.tracks
                           if stamp - t.history[-1][0] <= self.lost_s]
            return []
        points = points[np.isfinite(points).all(axis=1)]
        x0, y0, yaw = ego_pose
        c, s = math.cos(yaw), math.sin(yaw)
        clusters = self._clusters(points)
        world, bounds = [], []
        for center, _, group in clusters:
            rear_x = center[0] + self.lidar_to_rear_x_m
            world.append((x0 + c * rear_x - s * center[1],
                          y0 + s * rear_x + c * center[1]))
            rear_x = group[:, 0] + self.lidar_to_rear_x_m
            world_points = np.column_stack((x0 + c * rear_x - s * group[:, 1],
                                            y0 + s * rear_x + c * group[:, 1]))
            bounds.append((*world_points.min(axis=0), *world_points.max(axis=0)))
        predicted = []
        for track in self.tracks:
            t, x, y = track.history[-1]
            vx, vy = track.velocity if track.velocity_known else (0., 0.)
            predicted.append((x + vx * (stamp - t), y + vy * (stamp - t)))
        pairs = sorted(
            (math.hypot(w[0] - p[0], w[1] - p[1]), ci, ti)
            for ci, w in enumerate(world) for ti, p in enumerate(predicted))
        assigned = {}
        used = set()
        for distance, ci, ti in pairs:
            if distance > self.gate_m:
                break
            if ci in assigned or ti in used:
                continue
            assigned[ci] = ti
            used.add(ti)
        output = []
        survivors = [t for i, t in enumerate(self.tracks)
                     if i in used or stamp - t.history[-1][0] <= self.lost_s]
        for ci, (center, size, _) in enumerate(clusters):
            if ci in assigned:
                track = self.tracks[assigned[ci]]
            else:
                track = _Track(self._next_id)
                self._next_id += 1
                survivors.append(track)
            reliable = (track.size is None or np.max(
                np.abs(np.asarray(track.size) - size)) <= self.max_size_jump_m)
            track.size = tuple(size)
            track.history.append((stamp, *world[ci]))
            track.history = [h for h in track.history
                             if stamp - h[0] <= self.history_s]
            track.bounds_history.append((stamp, *bounds[ci]))
            track.bounds_history = [h for h in track.bounds_history
                                    if stamp - h[0] <= self.history_s]
            self._estimate(track, reliable)
            vx, vy = track.velocity
            lateral_mps = -s * vx + c * vy
            lateral_se = math.hypot(s * track.velocity_se[0],
                                    c * track.velocity_se[1])
            if (track.velocity_held and track.dynamic and track.velocity_known
                    and abs(lateral_mps) >= self.lateral_speed_mps
                    and track.lateral_scans * lateral_mps > 0.):
                # A changed visible footprint must not cancel a confirmed
                # crossing. Held evidence is neither renewed nor incremented.
                pass
            elif (reliable and track.velocity_known
                    and abs(lateral_mps) >= self.lateral_speed_mps
                    and abs(lateral_mps) >= self.significance * lateral_se):
                direction = 1 if lateral_mps > 0. else -1
                track.lateral_scans = (track.lateral_scans + direction
                                       if track.lateral_scans * direction > 0
                                       else direction)
            else:
                track.lateral_scans = 0
            output.append(TrackedObstacle(
                track.track_id, float(center[0]), float(center[1]),
                float(size[0]), float(size[1]),
                c * vx + s * vy, lateral_mps,
                track.velocity_known, track.dynamic, track.lateral_scans))
        self.tracks = survivors
        self.last_stamp = stamp
        return output

    def _estimate(self, track, reliable=True):
        track.velocity_held = False
        history = np.asarray(track.history)
        span = history[-1, 0] - history[0, 0]
        if len(history) < self.min_samples or span < self.min_span_s:
            track.velocity = (0., 0.)
            track.velocity_known = False
            return
        t = history[:, 0] - history[:, 0].mean()
        denominator = float(t @ t)
        def fit(values):
            centred = values - values.mean()
            slope = float(t @ centred) / denominator
            # Standard error of the slope; a 3-sample fit has one residual
            # degree of freedom and is never trusted as motion evidence.
            if len(history) >= 4:
                residual = centred - slope * t
                noise = max(self.min_noise_m,
                            math.sqrt(float(residual @ residual)
                                      / (len(history) - 2)))
                error = noise / math.sqrt(denominator)
            else:
                error = math.inf
            return slope, error

        velocity, se = [], []
        bounds = np.asarray(track.bounds_history)
        for axis in (1, 2):
            low, low_se = fit(bounds[:, axis])
            high, high_se = fit(bounds[:, axis + 2])
            # A newly visible face moves one edge and the box midpoint even
            # when the object is parked. Translation must move both edges in
            # the same direction; use the common part of their displacement.
            slope = (math.copysign(min(abs(low), abs(high)), low)
                     if low * high > 0. else 0.)
            error = max(low_se, high_se)
            if abs(slope) < self.significance * error:
                # Do not feed a noisy slope into collision prediction while
                # merely hiding its DYNAMIC label in the diagnostic view.
                slope = 0.
            velocity.append(slope)
            se.append(error)
        track.velocity = tuple(velocity)
        track.velocity_se = tuple(se)
        track.velocity_known = True
        if not reliable:
            # A split/merge or newly exposed surface has no trustworthy
            # translation for this scan. Keep its full measured geometry.
            self._hold_or_clear_motion(track)
            track.moving_scans = track.still_scans = 0
            return
        speed = math.hypot(*track.velocity)
        speed_se = math.hypot(*se) if all(map(math.isfinite, se)) else math.inf
        if (speed >= self.dynamic_speed_mps
              and speed < self.significance * speed_se):
            # Fast but not distinguishable from noise: no evidence.
            self._hold_or_clear_motion(track)
            track.moving_scans = track.still_scans = 0
        elif speed >= self.dynamic_speed_mps:
            track.last_trusted_motion = (float(history[-1, 0]),
                                         track.velocity, track.velocity_se)
            track.moving_scans += 1
            track.still_scans = 0
        elif speed <= self.static_speed_mps:
            track.still_scans += 1
            track.moving_scans = 0
        else:
            track.last_trusted_motion = (float(history[-1, 0]),
                                         track.velocity, track.velocity_se)
            track.moving_scans = track.still_scans = 0
        if track.moving_scans >= self.confirm_scans:
            track.dynamic = True
        elif track.still_scans >= self.confirm_scans:
            track.dynamic = False

    def _hold_or_clear_motion(self, track):
        """Retain confirmed traffic briefly through an unreliable footprint."""
        previous = track.last_trusted_motion
        if (track.dynamic and previous is not None
                and 0. <= track.history[-1][0] - previous[0] <= self.lost_s):
            track.velocity, track.velocity_se = previous[1:]
            track.velocity_known = track.velocity_held = True
        else:
            track.velocity = (0., 0.)
            track.velocity_known = False
