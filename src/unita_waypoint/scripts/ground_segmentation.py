"""IMU levelling and conservative, connected local road planes (no ROS I/O).

Coordinates are rear-axle, metres. Quaternion order is x,y,z,w, IMU->world.
Unknown patches retain the original height gate; they never erase objects.
"""
import math
import numpy as np


def rotation_rpy(roll=0., pitch=0., yaw=0.):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                            math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def normalized_quaternion(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError('invalid IMU orientation')
    return q / np.linalg.norm(q)


def quaternion_at(samples, stamp, max_skew):
    """SLERP brackets if BOTH endpoints are fresh; nearest sample otherwise.

    No extrapolation of angular velocity or reuse of stale orientation.
    Returns (quaternion or None, worst contributing timestamp difference).
    """
    if not samples:
        return None, None
    times = np.asarray([s[0] for s in samples])
    index = int(np.searchsorted(times, stamp))
    if 0 < index < len(samples):
        a, b = samples[index-1], samples[index]
        skew = max(stamp-a[0], b[0]-stamp)
        if skew <= max_skew:
            qa, qb = normalized_quaternion(a[1]), normalized_quaternion(b[1])
            dot = float(np.dot(qa, qb))
            if dot < 0.:
                qb, dot = -qb, -dot
            t = (stamp-a[0])/(b[0]-a[0])
            if dot > .9995:
                return normalized_quaternion(qa+t*(qb-qa)), skew
            angle = math.acos(np.clip(dot, -1., 1.))
            return (math.sin((1.-t)*angle)*qa+math.sin(t*angle)*qb)/math.sin(angle), skew
    nearest = min(samples, key=lambda s: abs(s[0]-stamp))
    skew = abs(nearest[0]-stamp)
    return (normalized_quaternion(nearest[1]) if skew <= max_skew else None), skew


def level_rotation(q, body_from_imu=None):
    x, y, z, w = normalized_quaternion(q)
    world_from_imu = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    world_from_body = world_from_imu @ (np.eye(3) if body_from_imu is None else body_from_imu.T)
    yaw = math.atan2(world_from_body[1, 0], world_from_body[0, 0])
    pitch = math.atan2(-world_from_body[2, 0], math.hypot(world_from_body[0, 0], world_from_body[1, 0]))
    roll = math.atan2(world_from_body[2, 1], world_from_body[2, 2])
    return rotation_rpy(yaw=-yaw) @ world_from_body, math.degrees(roll), math.degrees(pitch)


def _plane(points, tolerance, slope_deg, minimum, accept):
    """Bounded RANSAC over spatially distributed lowest returns, then LS refit."""
    if len(points) < minimum:
        return None
    cells = np.floor(points[:, :2]/.5).astype(np.int32)
    order = np.argsort(points[:, 2], kind='stable')
    stride = int(np.ptp(cells[:, 1]))+1
    codes = cells[:, 0]*stride+cells[:, 1]-cells[:, 1].min()
    _, first = np.unique(codes[order], return_index=True)
    seeds = points[order[first]]
    if len(seeds) < minimum:
        return None
    if len(seeds) > 256:
        seeds = seeds[np.linspace(0, len(seeds)-1, 256).astype(int)]
    design = np.column_stack((seeds[:, :2], np.ones(len(seeds))))
    rng = np.random.default_rng(37)
    slope_limit = math.tan(math.radians(slope_deg))
    subsets = rng.integers(0, len(seeds), size=(80, 3))
    mats = design[subsets]
    usable = np.abs(np.linalg.det(mats)) >= .05
    if not usable.any():
        return None
    planes = np.linalg.solve(mats[usable], seeds[subsets[usable], 2])
    planes = planes[(np.linalg.norm(planes[:, :2], axis=1) <= slope_limit) & accept(planes)]
    if not len(planes):
        return None
    # Broadcasting avoids thread-pool overhead from many small BLAS calls.
    distance = np.abs(seeds[:, 0, None]*planes[None, :, 0] +
                      seeds[:, 1, None]*planes[None, :, 1] +
                      planes[None, :, 2]-seeds[:, 2, None]) / np.sqrt(1+(planes[:, :2]**2).sum(axis=1))
    counts = (distance <= tolerance).sum(axis=0)
    if counts.max() < minimum:
        return None
    best = planes[int(np.argmax(counts))]
    # Refit original supported returns, not just lowest seeds: otherwise
    # dense/noisy ground would bias the final height towards the lower tail.
    refit_points = points
    if len(points) > 4096:
        refit_points = points[np.linspace(0, len(points)-1, 4096).astype(int)]
    design = np.column_stack((refit_points[:, :2], np.ones(len(refit_points))))
    for _ in range(2):
        inlier = np.abs(refit_points[:, 0]*best[0]+refit_points[:, 1]*best[1]+best[2]-refit_points[:, 2]) <= tolerance*math.sqrt(1+np.dot(best[:2], best[:2]))
        supported = refit_points[inlier]
        if len(supported) < minimum or np.ptp(supported[:, 0]) < 2. or np.ptp(supported[:, 1]) < 1.:
            return None
        best = np.linalg.lstsq(design[inlier], supported[:, 2], rcond=None)[0]
        if np.linalg.norm(best[:2]) > slope_limit or not accept(best[None, :])[0]:
            return None
    return best


def segment_ground(points, eligible, tolerance=.04, max_slope_deg=20., minimum=12,
                   min_height=.22, max_height=2.6, reference_z=0.):
    """8x6 m patches grown from near-ego road, with height/slope continuity.

    Only directly supported planes are used (never extrapolated into an
    occluded patch). Roofs/walls cannot seed distant road on their own.
    Returns ground mask, obstacle mask, and diagnostics. Low objects above
    a trusted plane survive even below min_height; that bound applies only
    to unknown road patches. max_height applies in both cases.
    """
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    ground = np.zeros(len(points), dtype=bool)
    height = points[:, 2]-reference_z
    obstacle = eligible & (height > min_height) & (height < max_height)
    if not len(points) or not eligible.any():
        return ground, obstacle, dict(planes=0, supported_points=0, residual_p95_m=None)
    indices = np.flatnonzero(eligible)
    labels = np.floor((points[indices, :2]+[4., 3.])/[8., 6.]).astype(np.int32)
    stride = int(np.ptp(labels[:, 1]))+1
    codes = labels[:, 0]*stride+labels[:, 1]-labels[:, 1].min()
    _, first, inverse, counts = np.unique(codes, return_index=True, return_inverse=True, return_counts=True)
    groups = np.split(indices[np.argsort(inverse, kind='stable')], np.cumsum(counts)[:-1])
    patches = {tuple(labels[i]): group for i, group in zip(first, groups)}
    accepted = {}
    remaining = sorted(patches, key=lambda k: abs(k[0])+abs(k[1]))
    # Several passes let neighbouring lateral patches acquire an anchor.
    for _ in range(3):
        next_remaining = []
        for key in remaining:
            neighbours = [k for k in accepted if abs(k[0]-key[0])+abs(k[1]-key[1]) == 1]
            near = abs(key[0]) <= 1 and key[1] == 0
            if not near and not neighbours:
                next_remaining.append(key)
                continue
            def accept(plane):
                valid = np.abs(plane[:, 2]-reference_z) <= .18 if near else np.zeros(len(plane), dtype=bool)
                for other in neighbours:
                    # Both ends of the shared edge must agree, including bank.
                    mid = (np.asarray(key)+np.asarray(other))*.5*[8., 6.]
                    tangent = np.array([0., 2.5]) if key[0] != other[0] else np.array([3.5, 0.])
                    edge = np.vstack((mid-tangent, mid+tangent))
                    delta = plane-accepted[other]
                    edge_delta = (edge[:, 0, None]*delta[None, :, 0]+
                                  edge[:, 1, None]*delta[None, :, 1]+delta[None, :, 2])
                    valid |= ((np.max(np.abs(edge_delta), axis=0) <= .12) &
                              (np.linalg.norm(delta[:, :2], axis=1) <= math.tan(math.radians(10.))))
                return valid
            indices = patches[key]
            fit = _plane(points[indices], tolerance, max_slope_deg, minimum, accept)
            if fit is None:
                next_remaining.append(key)
            else:
                accepted[key] = fit
        if len(next_remaining) == len(remaining):
            break
        remaining = next_remaining
    residuals, supported_count = [], 0
    for key, plane in accepted.items():
        indices = patches[key]
        distance = (points[indices, 2]-points[indices, 0]*plane[0]-points[indices, 1]*plane[1]-plane[2])/math.sqrt(1+np.dot(plane[:2], plane[:2]))
        inlier = np.abs(distance) <= tolerance
        ground[indices[inlier]] = True
        obstacle[indices] = (distance > tolerance) & (distance < max_height)
        residuals.extend(np.abs(distance[inlier]).tolist())
        supported_count += len(indices)
    return ground, obstacle, dict(planes=len(accepted), supported_points=supported_count,
                                 residual_p95_m=float(np.percentile(residuals, 95)) if residuals else None)
