"""Map-local stop-line geometry and conservative straight-signal decision."""

import math


# Tune against the competition vehicle in MORAI before enabling signal release.
FRONT_OFFSET_M = 3.0       # TODO: 측정 필요 - GPS 기준점에서 앞범퍼까지 [m]
A_BRAKE_MPS2 = 3.0         # TODO: 측정 필요 - 실측 최대 감속의 70~80% [m/s^2]
A_MAX_MPS2 = 1.5           # TODO: 측정 필요 - 통과 가속 상한 [m/s^2]
T_DELAY_S = 0.3            # TODO: 측정 필요 - 인식·제어·제동 지연 [s]
YELLOW_DURATION_S = 3.0    # TODO: 측정 필요 - 노란불 지속 시간 [s]
GO_MARGIN_S = 0.3          # TODO: 측정 필요 - 전환·시간 오차 여유 [s]
STOP_BUFFER_M = 0.5        # TODO: 측정 필요 - 정지선 앞 여유 [m]
LATERAL_TOLERANCE_M = 2.0  # TODO: 측정 필요 - 동일 차로 횡방향 허용폭 [m]


def project_stopline(stopline, x_m, y_m, yaw_rad,
                     front_offset_m=FRONT_OFFSET_M):
    """Return signed front-bumper distance and lateral error in map-local m."""
    values = (stopline['x'], stopline['y'], x_m, y_m, yaw_rad, front_offset_m)
    if not all(math.isfinite(value) for value in values) or front_offset_m < 0:
        raise ValueError('invalid stop-line projection input')
    dx, dy = stopline['x'] - x_m, stopline['y'] - y_m
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    distance_m = c * dx + s * dy - front_offset_m
    lateral_m = -s * dx + c * dy
    return distance_m, lateral_m


def nearest_ahead(stoplines, x_m, y_m, yaw_rad, traffic_light_id,
                  node_idx=None, front_offset_m=FRONT_OFFSET_M,
                  lateral_tolerance_m=LATERAL_TOLERANCE_M):
    """Select an ahead node from the controlling light and current lane only."""
    if not traffic_light_id or not math.isfinite(lateral_tolerance_m) or lateral_tolerance_m <= 0:
        raise ValueError('controlling light and positive lateral tolerance required')
    candidates = []
    for stopline in stoplines:
        if stopline['traffic_light_id'] != traffic_light_id:
            continue
        if node_idx is not None and stopline['idx'] != node_idx:
            continue
        distance_m, lateral_m = project_stopline(stopline, x_m, y_m, yaw_rad,
                                                  front_offset_m)
        if distance_m > 0 and abs(lateral_m) <= lateral_tolerance_m:
            candidates.append((distance_m, abs(lateral_m), stopline['idx'], stopline))
    return min(candidates) if candidates else None


def camera_signal_state(detections, roi, confidence=0.7, min_width=0.0):
    """Classify only one reviewed camera region; uncertainty means unknown."""
    if not math.isfinite(min_width) or not 0 <= min_width < 1:
        return 'unknown'
    labels = []
    for detection in detections:
        try:
            x0, y0, x1, y1 = detection['bbox_normalized']
            score = float(detection['confidence'])
            if (not all(math.isfinite(v) for v in (x0, y0, x1, y1, score))
                    or not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1):
                return 'unknown'
            x, y = (x0 + x1) / 2, (y0 + y1) / 2
            if roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]:
                if x1 - x0 < min_width:
                    continue
                if score < confidence:
                    return 'unknown'
                labels.append(detection['label'])
        except (KeyError, TypeError, ValueError):
            return 'unknown'
    if not labels or len(set(labels)) != 1:
        return 'unknown'
    if labels[0] in ('traffic_light_green', 'traffic_light_green_left'):
        return 'green'
    if labels[0] == 'traffic_light_yellow':
        return 'yellow'
    if labels[0] in ('traffic_light_red', 'traffic_light_red_and_yellow'):
        return 'red'
    return 'unknown'


def stopping_speed_mps(distance_m, brake_mps2=A_BRAKE_MPS2,
                       delay_s=T_DELAY_S, buffer_m=STOP_BUFFER_M):
    """Largest target speed satisfying v*delay + v²/(2*brake) <= space."""
    if not all(math.isfinite(v) for v in (distance_m, brake_mps2, delay_s, buffer_m)):
        raise ValueError('non-finite stopping input')
    if brake_mps2 <= 0 or delay_s < 0 or buffer_m < 0:
        raise ValueError('invalid stopping configuration')
    available_m = max(0.0, distance_m - buffer_m)
    return max(0.0, math.sqrt((brake_mps2 * delay_s) ** 2
                              + 2 * brake_mps2 * available_m)
               - brake_mps2 * delay_s)


def decide_signal(signal, distance_m, speed_mps, yellow_remaining_s,
                  brake_mps2=A_BRAKE_MPS2, delay_s=T_DELAY_S,
                  go_margin_s=GO_MARGIN_S, accel_mps2=A_MAX_MPS2,
                  use_acceleration=False):
    """Return (decision, can_go, can_stop, target_speed_mps).

    `yellow_remaining_s` must account for time since the light changed and
    sensing latency. Unknown/stale signal is treated as stop.
    """
    values = (distance_m, speed_mps, yellow_remaining_s, brake_mps2,
              delay_s, go_margin_s, accel_mps2)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('non-finite yellow decision input')
    if (speed_mps < 0 or brake_mps2 <= 0 or delay_s < 0
            or go_margin_s < 0 or accel_mps2 < 0):
        raise ValueError('invalid yellow decision configuration')
    can_stop = distance_m >= speed_mps * delay_s + speed_mps ** 2 / (2 * brake_mps2)
    usable_s = max(0.0, yellow_remaining_s - go_margin_s)
    if use_acceleration:
        can_go = (usable_s > 0 and distance_m > 0 and
                  distance_m <= speed_mps * usable_s
                  + 0.5 * accel_mps2 * usable_s ** 2)
    else:
        can_go = (usable_s > 0 and speed_mps > 0 and distance_m > 0
                  and distance_m / speed_mps <= usable_s)
    if signal == 'green':
        return 'go', False, can_stop, math.inf
    if signal == 'yellow' and can_go:
        return 'go', can_go, can_stop, math.inf
    return ('stop', can_go if signal == 'yellow' else False, can_stop,
            stopping_speed_mps(distance_m, brake_mps2, delay_s))
