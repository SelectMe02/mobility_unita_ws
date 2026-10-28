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
                 candidate_vx=0., candidate_vy=0.,
                 prediction_confirmed=previous.get('prediction_confirmed', False),
                 prediction_moving_scans=previous.get('prediction_moving_scans', 0),
                 prediction_consistent_scans=previous.get('prediction_consistent_scans', 0),
                 prediction_still_scans=previous.get('prediction_still_scans', 0),
                 prediction_stop_threshold=previous.get('prediction_stop_threshold', .5*threshold),
                 motion_velocity_stamp=previous.get('motion_velocity_stamp'))
    if len(history) < 3 or stamp-history[0][0]+1e-6 < .2:
        state.update(moving_scans=0, still_scans=0,
                     prediction_moving_scans=0, prediction_consistent_scans=0,
                     prediction_still_scans=0)
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
    # Broad returns changing visible face are not evidence of translation.
    # Small pedestrian footprints retain the existing motion thresholds.
    if len(history) >= 2:
        previous_size = np.asarray(history[-2][3:5]) - np.asarray(history[-2][1:3])
        current_size = np.asarray(bounds[2:]) - np.asarray(bounds[:2])
        if (max(np.max(previous_size), np.max(current_size)) > 1.
                and np.max(np.abs(current_size-previous_size)) > .35
                and np.linalg.norm(velocity) > 1e-6):
            reliable = False
    # Keep the fitted observation separate from the velocity accepted for
    # collision prediction. One noisy scan must not move a static obstacle
    # into the adjacent lane while its motion classification stays static.
    state.update(candidate_vx=float(velocity[0]), candidate_vy=float(velocity[1]))
    if not reliable:
        trusted_stamp = previous.get('motion_velocity_stamp')
        velocity = (np.array([previous.get('vx', 0.), previous.get('vy', 0.)])
                    if state['prediction_confirmed'] and trusted_stamp is not None
                    and stamp-trusted_stamp <= .4 else np.zeros(2))
    state.update(vx=float(velocity[0]), vy=float(velocity[1]), velocity_reliable=reliable)
    speed = float(np.linalg.norm(velocity))
    broad = max(bounds[2]-bounds[0], bounds[3]-bounds[1]) > 1.
    static_footprint = broad and (previous.get('static_footprint', False)
                                 or (previous.get('motion_kind') == 'static'
                                     and previous.get('still_scans', 0) >= 2
                                     and not previous.get('prediction_confirmed', False)
                                     and math.hypot(previous.get('candidate_vx', 0.),
                                                    previous.get('candidate_vy', 0.)) < .05))
    # A previously stationary broad surface needs sustained translation;
    # two adjacent face shifts must not move it into the passing lane.
    motion_scans_required = 4 if static_footprint else 2
    state['static_footprint'] = static_footprint
    if previous.get('motion_threshold', threshold) != threshold:
        # A live setting change applies to the same observed track immediately.
        state.update(motion_kind='dynamic' if reliable and speed > threshold+1e-6 else 'static',
                     moving_scans=0, still_scans=0)
    elif reliable and speed > threshold+1e-6:
        state['moving_scans'] += 1
        state['still_scans'] = 0
        if state['moving_scans'] >= motion_scans_required:
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
    # Classification is a configurable label. Collision prediction separately
    # needs consecutive physical motion evidence, and a changed label threshold
    # must not erase an already established moving vehicle's velocity.
    if reliable:
        if state['prediction_confirmed']:
            if speed <= state['prediction_stop_threshold']:
                state['prediction_still_scans'] += 1
                if state['prediction_still_scans'] >= 2:
                    state['prediction_confirmed'] = False
                    state['prediction_moving_scans'] = 0
                    state['prediction_consistent_scans'] = 0
                else:
                    state.update(vx=previous.get('vx', 0.), vy=previous.get('vy', 0.))
            else:
                state['prediction_still_scans'] = 0
                state['motion_velocity_stamp'] = stamp
        else:
            same_direction = (previous.get('velocity_reliable', False) and
                              state['candidate_vx']*previous.get('candidate_vx', 0.) +
                              state['candidate_vy']*previous.get('candidate_vy', 0.) > 0.)
            if previous.get('motion_threshold', threshold) != threshold:
                state['prediction_moving_scans'] = 0
            state['prediction_moving_scans'] = (
                (state['prediction_moving_scans']+1 if same_direction else 1)
                if speed > threshold+1e-6 else 0)
            # Slow real traffic also moves. Below the display's dynamic
            # threshold, require four consecutive significant fits in the
            # same direction instead of promoting a short residual drift.
            state['prediction_consistent_scans'] = (
                (state['prediction_consistent_scans']+1 if same_direction else 1)
                if speed > 1e-6 else 0)
            state['prediction_still_scans'] = 0
            if (state['prediction_moving_scans'] >= motion_scans_required or
                    state['prediction_consistent_scans'] >= (6 if static_footprint else 4)):
                state['prediction_confirmed'] = True
                state['static_footprint'] = False
                # A confirmed 0.36 m/s mover must not immediately satisfy
                # the ordinary 0.4 m/s stop test on its next observation.
                state['prediction_stop_threshold'] = .5*min(threshold, speed)
                state['motion_velocity_stamp'] = stamp
    else:
        state['prediction_moving_scans'] = state['prediction_consistent_scans'] = 0
        state['prediction_still_scans'] = 0
    if not state['prediction_confirmed']:
        # Keep the whole measured footprint occupied while motion is uncertain.
        state.update(vx=0., vy=0.)
    return state
