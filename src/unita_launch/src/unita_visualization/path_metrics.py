import math


def nearest_segment(points, x, y, yaw):
    """Project onto the nearest nonzero segment; left of travel is positive.

    Returns closest x/y, signed lateral distance, heading error, distance along
    the reference and segment index. At crossings the nearest segment can jump.
    """
    best = None
    along = 0.0
    for index, (a, b) in enumerate(zip(points, points[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        t = max(0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / length ** 2))
        px, py = a[0] + t * dx, a[1] + t * dy
        distance = math.hypot(x - px, y - py)
        if best is None or distance < best[0]:
            cross = dx * (y - py) - dy * (x - px)
            signed = -distance if cross < 0 else distance
            heading = yaw - math.atan2(dy, dx)
            heading = math.atan2(math.sin(heading), math.cos(heading))
            best = (distance, px, py, signed, heading, along + t * length, index)
        along += length
    if best is None:
        raise ValueError('reference needs at least one nonzero segment')
    return best[1:]
