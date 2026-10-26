"""Compare guard decisions on recorded poses; not a closed-loop driving test."""
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/unita_waypoint/scripts'))
from map_tunner_core import primary_route


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


old = module('before_departure', ROOT / 'rollback/2026-10-26_cp8_roundabout_departure_timing_before/src/unita_waypoint/scripts/local_avoidance_core.py')
new = module('after_departure', ROOT / 'src/unita_waypoint/scripts/local_avoidance_core.py')
results = {}
for run in ('195016', '195054'):
    folder = ROOT / ('src/unita_waypoint/log/run_20261026_' + run)
    project = json.loads((folder / 'config/map_tunner_project.json').read_text())
    route, sectors = primary_route(project)
    points = [(p['x'], p['y']) for p in route]
    params = yaml.safe_load((folder / 'rosparams.yaml').read_text())['gps_waypoint_follower']
    kwargs = {k: v for k, v in params.items() if k.startswith('roundabout_')
              and k in old.LocalAvoidancePlanner.__init__.__code__.co_varnames}
    planners = [old.LocalAvoidancePlanner(points, sectors, {}, **kwargs),
                new.LocalAvoidancePlanner(points, sectors, {},
                    **dict(kwargs, roundabout_ego_lead_min_speed_kmh=2.5))]
    stamp = None
    rows = []
    for line in (folder / 'events.jsonl').open():
        event = json.loads(line)
        d = event['data']
        if event['topic'] == '/waypoint_debug/obstacles':
            stamp = d.get('stamp', event['ros_time'])
        if event['topic'] != '/waypoint_debug/control_decision':
            continue
        if d.get('roundabout_state') in (None, 'OUTSIDE'):
            continue
        row = dict(time=event['time'], recorded_state=d['roundabout_state'],
                   recorded_speed_kmh=d['current_speed_kmh'],
                   recorded_accel=d['command']['accel'], recorded_brake=d['command']['brake'])
        for label, m, p in zip(('before', 'after'), (old, new), planners):
            fields = {f.name for f in dataclasses.fields(m.Obstacle)}
            obstacles = []
            for entry in d['obstacles']:
                entry = dict(entry, track_id=entry['id'])
                obstacles.append(m.Obstacle(**{k: v for k, v in entry.items() if k in fields}))
            index, xy = d['waypoint_index'], (d['x'], d['y'])
            speed = d['current_speed_kmh']/3.6
            stopping = speed**2/(2.*p.decel)+speed*p.reaction_s
            near = p.front_extent+max(6., stopping)+p.margin
            obstacles = [o for o in obstacles if o.confirmed or o.radius >= .35
                         or m.distance(xy, (o.x, o.y))-o.radius <= near]
            progress = p._progress_at(index, xy)
            decision = p._roundabout_guard(index, progress, d['current_speed_kmh']/3.6,
                d['speed_limits']['profile_kmh']/3.6, [], obstacles, float('inf'), stamp)
            row[label] = dict(state=p.roundabout_state, votes=p.roundabout_clear_votes,
                              yielded=p.roundabout_yield_seen,
                              cap=None if decision is None or decision.speed_cap_kmh == float('inf')
                              else decision.speed_cap_kmh)
        rows.append(row)
    results[run] = rows
    print('Run', run)
    for label in ('before', 'after'):
        release = next(r for r in rows if r[label]['yielded'] and r[label]['state'] == 'ENTRY_CLEAR')
        print(label, 'first release', release['time'])
    assert any(r['before']['state'] == 'APPROACH' and r['after']['state'] == 'ENTRY_CLEAR' for r in rows)
    # Both runs previously rejected the first gap again at 2.8--4.4 km/h.
    assert any(r['before']['state'] == 'YIELD' and r['after']['state'] == 'ENTRY_CLEAR' for r in rows)
    previous = None
    for r in rows:
        states = (r['before']['state'], r['after']['state'])
        if states != previous:
            print(r['time'][11:23], states)
        previous = states
(Path(__file__).parent / 'guard_replay.json').write_text(json.dumps(results, indent=2) + '\n')
