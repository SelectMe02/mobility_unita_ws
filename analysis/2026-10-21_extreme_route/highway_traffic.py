"""Three bounded, proximity-triggered S11 cars; serialized speeds are km/h."""
import math
from pathlib import Path
import json
import numpy as np


def make_highway_traffic(project):
    sector = next(s for s in project['sectors'] if s['number'] == 11)
    lanes = {int(k): np.array([[p[c] for c in 'xyz'] for p in v])
             for k, v in sector['racelines'].items()}
    def at(y, lane):
        a = lanes[lane]
        assert a[:, 1].min() <= y <= a[:, 1].max()
        return np.array([np.interp(y, a[::-1, 1], a[::-1, k]) for k in range(3)])
    links = {v['idx']: v for v in json.loads(Path('/home/unita/Downloads/link_set.json').read_text())}
    exit_points = np.array(links['A2256W000179']['points'] + links['A2256W001105']['points'][1:])
    def exit_at(y):
        a = exit_points[exit_points[:, 1] >= -240]
        return np.array([np.interp(y, a[::-1, 1], a[::-1, k]) for k in range(3)])
    def vec(p):
        return {**dict(zip('xyz', map(float, p))), **dict(zip(('_x','_y','_z'), ('%.3f' % v for v in p)))}
    def unity(p): return vec((-p[0], p[2], -p[1]))
    def rot(yaw=0.): return dict(roll='0.000', pitch='0.000', yaw='%.3f' % yaw)
    specs = [
        (1101, 35., [(80,2),(65,2),(35,3),(15,3),(-10,2),(-60,2),(-85,1)], -163., 0.),
        (1102, 37., [(60,1),(45,1),(15,2),(-10,3),(-35,3),(-60,2),(-85,1)], -188., 3.),
        (1103, 40., [(40,2),(25,2),(-5,3),(-25,3),(-55,2),(-85,1)], -213., 6.),
    ]
    objects, paths, actors = [], [], []
    for uid, speed, keys, end_y, padding in specs:
        points = []
        for (ya, la), (yb, lb) in zip(keys, keys[1:]):
            for y in np.linspace(ya, yb, int(math.ceil(ya-yb))+1)[:-1]:
                t = (ya-y)/(ya-yb); t = t*t*(3-2*t)
                points.append(at(y, la)*(1-t)+at(y, lb)*t)
        # Merge into the actual mapped right-hand exit, then stop off the Ego route.
        for y in np.linspace(-85., -120., 36)[:-1]:
            t = (-85-y)/35; t = t*t*(3-2*t)
            points.append(at(y, 1)*(1-t)+exit_at(y)*t)
        points.extend(exit_at(y) for y in np.linspace(-120., end_y, int(-120-end_y)+1))
        points = np.array(points)
        radius = float(np.linalg.norm(points[0]-at(115., 1))+padding)
        delta = points[1]-points[0]
        path = [dict(pos=unity(p), rot=rot()) for p in points]
        objects.append(dict(DataID=40100020, UNIQUEID=uid, m_eobstacleObjType=4,
            pos=vec(points[0]), rot=rot(math.degrees(math.atan2(delta[1],delta[0]))),
            scale=vec((1.,1.,1.)), initPos=unity(points[0]), initRot=rot(),
            isWaypointMode=True, moveSpeed=speed, activeDistance=radius,
            waypointEventList=[dict(index=-1,waypointEventData=[]) for _ in path]))
        paths.append(dict(pathName='ObstaclePath - %d' % uid, waypointData=path))
        actors.append(dict(id=uid,label='S11 %gkm/h 차선 변경 차량' % speed,sector=11,
            kind='moving_object',model=40100020,speed_kmh=speed,active_distance_m=radius,
            start=points[0].tolist(),end=points[-1].tolist(),path_points=len(path),
            lane_sequence=[lane for _,lane in keys],exit_links=['A2256W000179','A2256W001105']))
    return objects, paths, actors
