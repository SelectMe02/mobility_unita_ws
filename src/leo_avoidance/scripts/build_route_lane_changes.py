#!/usr/bin/env python3
"""Write where the route may change lanes, from the MORAI MGeo link set.

A pass may only use an adjacent same-direction lane whose MGeo link allows a
lane change (dashed marking) toward it; everything else, including the
centre line, is treated as a solid line. Output intervals are along the CSV
route: {start_s, end_s, side, offset_m} with offset_m to the left (+).

Usage: build_route_lane_changes.py LINK_SET_JSON ROUTE_CSV OUTPUT_JSON
"""
import json
import sys

import numpy as np


def segments(links):
    starts, ends, owners = [], [], []
    for link in links:
        p = np.asarray(link['points'], float)[:, :2]
        starts.extend(p[:-1])
        ends.extend(p[1:])
        owners.extend([link['idx']] * (len(p) - 1))
    return np.asarray(starts), np.asarray(ends), owners


def project(point, polyline):
    """Signed left offset of the polyline from point (sign by its direction)."""
    a, b = polyline[:-1], polyline[1:]
    d = b - a
    length2 = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((point - a) * d).sum(1) / length2, 0., 1.)
    foot = a + t[:, None] * d
    k = int(np.argmin(np.linalg.norm(point - foot, axis=1)))
    tangent = d[k] / np.sqrt(length2[k])
    rel = foot[k] - point
    return float(tangent[0] * rel[1] - tangent[1] * rel[0])


def main(link_file, route_file, output_file):
    links = json.load(open(link_file))
    by_id = {link['idx']: link for link in links}
    route = np.loadtxt(route_file)[:, :2]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    starts, ends, owners = segments(links)
    delta = ends - starts
    length2 = np.maximum((delta ** 2).sum(1), 1e-12)
    rows = []
    for i, point in enumerate(route):
        heading = route[min(i + 1, len(route) - 1)] - route[max(i - 1, 0)]
        heading /= max(np.linalg.norm(heading), 1e-9)
        t = np.clip(((point - starts) * delta).sum(1) / length2, 0., 1.)
        distance = np.linalg.norm(point - (starts + t[:, None] * delta), axis=1)
        distance[(delta @ heading) / np.sqrt(length2) < .7] = np.inf
        k = int(np.argmin(distance))
        link = by_id[owners[k]]
        allowed = {}
        for side, flag, target in (('left', 'can_move_left_lane', 'left_lane_change_dst_link_idx'),
                                   ('right', 'can_move_right_lane', 'right_lane_change_dst_link_idx')):
            if link.get(flag) and link.get(target) in by_id:
                polyline = np.asarray(by_id[link[target]]['points'], float)[:, :2]
                offset = project(point, polyline)
                if 2. <= abs(offset) <= 5. and (offset > 0) == (side == 'left'):
                    allowed[side] = offset
        rows.append((float(distance[k]), allowed))
    intervals = []
    for side in ('left', 'right'):
        current = None
        for i, (_, allowed) in enumerate(rows):
            if side in allowed:
                if current is None:
                    current = dict(start_s=float(s[i]), side=side, offsets=[])
                current['end_s'] = float(s[i])
                current['offsets'].append(allowed[side])
            elif current is not None:
                intervals.append(current)
                current = None
        if current is not None:
            intervals.append(current)
    result = [dict(start_s=round(v['start_s'], 2), end_s=round(v['end_s'], 2), side=v['side'],
                   offset_m=round(float(np.median(v['offsets'])), 3))
              for v in intervals if v['end_s'] - v['start_s'] >= 5.]
    json.dump(dict(source='MORAI MGeo link_set can_move_*_lane; route %s' % route_file.split('/')[-1],
                   route_length_m=round(float(s[-1]), 3),
                   max_link_match_error_m=round(max(r[0] for r in rows), 3),
                   lane_changes=sorted(result, key=lambda v: v['start_s'])),
              open(output_file, 'w'), indent=2)
    for v in sorted(result, key=lambda v: v['start_s']):
        print('%8.1f - %8.1f m  %-5s %+.2f m' % (v['start_s'], v['end_s'], v['side'], v['offset_m']))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
