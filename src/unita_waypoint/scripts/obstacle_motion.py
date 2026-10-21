"""Estimate object translation without mistaking changing visible faces for motion."""
import math

import numpy as np


def box_bounds(x, y, length, width, yaw):
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    dx, dy = .5*(c*length+s*width), .5*(s*length+c*width)
    return (x-dx, y-dy, x+dx, y+dy)


def update_motion(previous, stamp, bounds, threshold=.8):
    # A growing visible face moves the midpoint but leaves an edge still.
    # Compare both world-frame edges over time; keep the measured geometry
    # separate so an expanding footprint still participates in collision tests.
    history = [h for h in previous.get('motion_history', ()) if 0. < stamp-h[0] <= .8+1e-6]
    history.append((stamp, *bounds))
    state = dict(motion_history=history, motion_kind=previous.get('motion_kind', 'unknown'),
                 moving_scans=previous.get('moving_scans', 0),
                 still_scans=previous.get('still_scans', 0),
                 motion_threshold=threshold, vx=0., vy=0., velocity_reliable=False,
                 motion_velocity_stamp=previous.get('motion_velocity_stamp'))
    if len(history) < 3 or stamp-history[0][0]+1e-6 < .2:
        state.update(moving_scans=0, still_scans=0)
        return state
    data = np.asarray(history, dtype=float)
    times = data[:, 0]-data[0, 0]
    times -= times.mean()
    denominator = float(times @ times)
    centred = data[:, 1:]-data[:, 1:].mean(axis=0)
    slopes = times @ centred / denominator
    residual = centred-times[:, None]*slopes
    noise = np.maximum(.08, np.sqrt(np.sum(residual**2, axis=0)/max(1, len(data)-2)))
    errors = noise/math.sqrt(denominator)
    velocity = np.zeros(2)
    for axis in range(2):
        low, high = slopes[axis], slopes[axis+2]
        # Translation has the same direction at both edges. The slower edge
        # bounds the motion supported by the observations when a face changes.
        if low*high > 0.:
            candidate = math.copysign(min(abs(low), abs(high)), low)
            if abs(candidate) > 2.5*max(errors[axis], errors[axis+2]):
                velocity[axis] = candidate
    reliable = bool(np.max(noise) <= .25)
    if not reliable:
        trusted_stamp = previous.get('motion_velocity_stamp')
        velocity = (np.array([previous.get('vx', 0.), previous.get('vy', 0.)])
                    if previous.get('motion_kind') == 'dynamic' and trusted_stamp is not None
                    and stamp-trusted_stamp <= .4 else np.zeros(2))
    else:
        state['motion_velocity_stamp'] = stamp
    state.update(vx=float(velocity[0]), vy=float(velocity[1]), velocity_reliable=reliable)
    speed = float(np.linalg.norm(velocity))
    if previous.get('motion_threshold', threshold) != threshold:
        # A live setting change applies to the same observed track immediately.
        state.update(motion_kind='dynamic' if reliable and speed > threshold+1e-6 else 'static',
                     moving_scans=0, still_scans=0)
    elif reliable and speed > threshold+1e-6:
        state['moving_scans'] += 1
        state['still_scans'] = 0
        if state['moving_scans'] >= 2:
            state['motion_kind'] = 'dynamic'
    elif reliable and speed <= .5*threshold:
        state['still_scans'] += 1
        state['moving_scans'] = 0
        if state['still_scans'] >= 2 or state['motion_kind'] == 'unknown':
            state['motion_kind'] = 'static'
    else:
        # Neither an isolated jump nor separated noisy scans confirm motion.
        state['moving_scans'] = state['still_scans'] = 0
        if reliable and state['motion_kind'] == 'unknown':
            state['motion_kind'] = 'static'
    return state
