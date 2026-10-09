#!/usr/bin/env python3
"""Explain blackout drift from an odometry_drift_recorder.py CSV.

1. Clock/latency: for each pose source, the time shift that best aligns its
   trajectory with /competition/ego_pose (MORAI ground truth), by header
   stamp and by receive time.
2. Speed scale: dead-reckoned /leo/ego_odom path length vs ground truth.
3. Blackout error: map_pose minus ground truth (forward/left) against
   travelled distance; a constant offset means an anchor timing problem,
   a growing one means scale or heading drift.
"""
import csv
import math
import sys
from collections import defaultdict

import numpy as np


def load(path):
    rows = defaultdict(list)
    with open(path, errors='replace') as handle:
        lines = (line for line in handle if '\0' not in line)
        for r in csv.DictReader(lines):
            try:  # skip rows corrupted by an interrupted write
                float(r['recv_s'])
                for key in ('recv_s', 'stamp_s', 'x', 'y', 'yaw', 'travelled_m'):
                    if r[key]:
                        float(r[key])
            except (TypeError, ValueError):
                continue
            rows[r['topic']].append(r)
    return rows


def series(rows, topic, clock):
    out = [(float(r[clock]), float(r['x']), float(r['y']), float(r['yaw']))
           for r in rows.get(topic, []) if r[clock] and r['x']]
    return np.array(sorted(out)) if out else np.empty((0, 4))


def interp(track, t):
    return np.c_[np.interp(t, track[:, 0], track[:, 1]),
                 np.interp(t, track[:, 0], track[:, 2])]


def best_shift(source, truth, moving):
    """Shift (s) to add to the source time so it best matches ground truth."""
    shifts = np.arange(-1.5, 1.5001, .02)
    t = source[moving, 0]
    errors = [np.mean(np.linalg.norm(interp(truth, t - s) - source[moving, 1:3], axis=1))
              for s in shifts]
    i = int(np.argmin(errors))
    return shifts[i], errors[i], errors[len(shifts) // 2]


def main(path):
    rows = load(path)
    for clock in ('stamp_s', 'recv_s'):
        truth = series(rows, '/competition/ego_pose', clock)
        if len(truth) < 20:
            print('not enough ground truth samples (%s)' % clock)
            continue
        speed = np.r_[0, np.linalg.norm(np.diff(truth[:, 1:3], axis=0), axis=1)
                      / np.maximum(np.diff(truth[:, 0]), 1e-3)]
        print('\n== time alignment against ego_pose by %s' % clock)
        for topic in ('/localization/pose', '/leo/driving_pose'):
            source = series(rows, topic, clock)
            if len(source) < 20:
                continue
            moving = np.interp(source[:, 0], truth[:, 0], speed) > 1.
            if moving.sum() < 20:
                continue
            shift, err, err0 = best_shift(source, truth, moving)
            print('%-22s best shift %+5.2f s -> mean error %.2f m (unshifted %.2f m)'
                  % (topic, shift, err, err0))
    # Header stamp vs receive time per topic: different clocks show up here.
    print('\n== receive time minus header stamp (ms)')
    for topic, items in sorted(rows.items()):
        lag = [float(r['recv_s']) - float(r['stamp_s']) for r in items if r['stamp_s']]
        if lag:
            print('%-28s mean %8.1f  min %8.1f  max %8.1f'
                  % (topic, 1e3 * np.mean(lag), 1e3 * np.min(lag), 1e3 * np.max(lag)))
    truth = series(rows, '/competition/ego_pose', 'recv_s')
    odom = series(rows, '/leo/ego_odom', 'recv_s')
    if len(truth) > 20 and len(odom) > 20:
        start, end = max(truth[0, 0], odom[0, 0]), min(truth[-1, 0], odom[-1, 0])
        sel = lambda a: a[(a[:, 0] >= start) & (a[:, 0] <= end)]
        length = lambda a: float(np.sum(np.linalg.norm(np.diff(a[:, 1:3], axis=0), axis=1)))
        t_len, o_len = length(sel(truth)), length(sel(odom))
        if t_len > 10:
            print('\n== speed scale: ego_odom %.1f m vs ground truth %.1f m -> scale %.4f'
                  % (o_len, t_len, o_len / t_len))
    blackout = [r for r in rows.get('map_pose', []) if r['mode'] in ('BLACKOUT', 'SIM_EXIT')]
    if blackout and len(truth) > 20:
        print('\n== blackout map_pose error vs ground truth')
        print('travelled_m  forward_m  left_m  yaw_deg')
        last = -10.
        for r in blackout:
            travelled = float(r['travelled_m'] or 0)
            if travelled - last < 5.:
                continue
            last = travelled
            t = float(r['recv_s'])
            ex, ey = interp(truth, np.array([t]))[0]
            eyaw = float(np.interp(t, truth[:, 0], np.unwrap(truth[:, 3])))
            dx, dy = float(r['x']) - ex, float(r['y']) - ey
            c, s = math.cos(eyaw), math.sin(eyaw)
            print('%10.1f  %+8.2f  %+6.2f  %+6.2f' % (
                travelled, c * dx + s * dy, -s * dx + c * dy,
                math.degrees(math.atan2(math.sin(float(r['yaw']) - eyaw),
                                        math.cos(float(r['yaw']) - eyaw)))))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'drift.csv')
