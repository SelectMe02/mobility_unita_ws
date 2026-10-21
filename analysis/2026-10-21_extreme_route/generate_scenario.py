#!/usr/bin/env python3
"""Build a full-route stress scene from the saved route and native MORAI samples.

Writes only beside this script. Installation into MORAI is a separate step.
No controller settings, map points, sensor profiles, or existing scenes change.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from highway_traffic import make_highway_traffic

OUT = Path(__file__).resolve().parent
WS = OUT.parents[1]
SOURCE = Path('/home/unita/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Scenario/R_KR_PR_K-city_2025')
PROJECT = WS / 'src/unita_waypoint/config/map_tunner_project.json'
NAME = 'UNITA_EXTREME_ROUTE_20261021'
REMOVED_ACTOR_IDS = {1002, 1006, 1008, 1012, 1014}


def read(path):
    return json.loads(path.read_text())


def vec(p):
    return dict(zip(('x', 'y', 'z'), map(float, p)),
                **dict(zip(('_x', '_y', '_z'), ('%.3f' % v for v in p))))


def xyz(v):
    return np.array([v[k] for k in ('x', 'y', 'z')], dtype=float)


def unity(p):
    return vec((-p[0], p[2], -p[1]))


def rotation(yaw):
    return dict(roll='0.000', pitch='0.000', yaw='%.3f' % math.degrees(yaw))


class Line:
    def __init__(self, points):
        points = np.asarray(points, dtype=float)
        self.p = points[np.r_[True, np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) > 1e-5]]
        self.s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.p[:, :2], axis=0), axis=1))]

    def at(self, station, lateral=0.):
        assert -.001 <= station <= self.s[-1]+.001, (station, self.s[-1])
        i = min(len(self.s)-2, max(0, int(np.searchsorted(self.s, station, side='right')-1)))
        t = (station-self.s[i])/(self.s[i+1]-self.s[i])
        p = self.p[i]+t*(self.p[i+1]-self.p[i])
        d = self.p[i+1, :2]-self.p[i, :2]
        yaw = math.atan2(d[1], d[0])
        p[:2] += np.array([-math.sin(yaw), math.cos(yaw)])*lateral
        return p, yaw

    def station_y(self, y):
        i = min(range(len(self.p)-1), key=lambda k: abs((self.p[k, 1]+self.p[k+1, 1])*.5-y))
        a, b = self.p[i:i+2]
        assert min(a[1], b[1])-.01 <= y <= max(a[1], b[1])+.01, ('y outside saved lane', y)
        t = (y-a[1])/(b[1]-a[1])
        return float(self.s[i]+t*(self.s[i+1]-self.s[i]))

    def follow(self, keys):
        """(station,left-offset) keys, smooth lane shifts sampled every <=1m."""
        points = []
        for (a, da), (b, db) in zip(keys, keys[1:]):
            assert b > a
            for t in np.linspace(0., 1., max(2, int(math.ceil(b-a))+1))[:-1]:
                blend = t*t*(3.-2.*t)
                points.append(self.at(a+t*(b-a), da+blend*(db-da))[0])
        points.append(self.at(*keys[-1])[0])
        return points


project = read(PROJECT)
lanes = {(s['number'], int(k)): Line([[p['x'], p['y'], p['z']] for p in points])
         for s in project['sectors'] for k, points in s['racelines'].items() if len(points) >= 2}
reference = read(SOURCE / '00_official_reference.json')
scene = copy.deepcopy(read(SOURCE / 'LEO_start_parked.json'))
for key in ('vehicleList', 'objectList', 'pedestrianList', 'spawnPointList',
            'waypointDataList', 'pedspawnPointList', 'aiPedstrianList', 'frictionChangeMaterialList'):
    scene[key] = []
# Preserve the native intersection phases. They are not distance-trigger events.
scene['tlList'] = copy.deepcopy(reference['tlList'])
scene['shadedAreaList'] = copy.deepcopy(read(SOURCE / 'TUN07_car_lane2_cuts_in_to_lane1_ahead_of_ego.json')['shadedAreaList'])
start, yaw = lanes[1, 1].at(0.)
ego = scene['egoVehicle']
ego['initPosition']['pos'] = vec(start+np.array([0., 0., .34]))
ego['initPosition']['rot'] = rotation(yaw)
ego['initPosition']['gear'] = 4
ego['currentVelocity'] = 0.
save = ego['initPosition']['vehicleSaveData']
save['velocity'] = 0.
save['inputData'][6] = 4
save['gearbox']['gearMode'] = 4
save['vehicle']['lastVelocity'] = dict(x=0., y=0., z=0.)
save['lastVelocity'] = dict(x=0., y=0., z=0.)

manifest = dict(name=NAME, route=str(PROJECT), route_sha256=hashlib.sha256(PROJECT.read_bytes()).hexdigest(),
                map=scene['mapInfo'], actors=[], spawners=[], notes=[
                    'Full lap from the current saved start. Road Z is taken from the current saved route.',
                    'Scripted proximity actors use existing native object/pedestrian waypoint schemas.',
                    'Signal phases are the official snapshot; no unverified timed/yellow event schema is invented.',
                    'S5-end obstruction is temporary crossing. Compact static blockers use surveyed passing windows.',
                    'S12/S13 tollgate waypoint-only corridor is left clear.',
                    'Object moveSpeed and pedestrian speed are km/h. Installed ObstacleObject.SetMoveSpeed divides by 3.6 internally.',
                    'Bounded load: 13 scripted actors, including 3 proximity-triggered S11 lane-changing cars; no NPC spawners.',
                    'Moving cars outside S11 are set to 20 km/h; static objects and pedestrians retain their roles.',
                    'Asset mesh footprints and sensor mounting are not inferred from scale factors.',
                    'Native scenario loading, exact NPC timing and drivability still require a simulator run.'])
uid = 1000


def add_object(label, sector, points, yaw=None, speed=0., active=0., model=40100020, scale=(1., 1., 1.)):
    global uid
    uid += 1
    points = [np.asarray(p, dtype=float) for p in points]
    if yaw is None:
        d = points[1]-points[0]
        yaw = math.atan2(d[1], d[0])
    moving = speed > 0.
    path = [dict(pos=unity(p), rot=rotation(0.)) for p in points] if moving else []
    obj = dict(DataID=model, UNIQUEID=uid, m_eobstacleObjType=4,
               pos=vec(points[0]), rot=rotation(yaw), scale=vec(scale),
               initPos=unity(points[0]) if moving else vec((0., 0., 0.)), initRot=rotation(0.),
               isWaypointMode=True, moveSpeed=float(speed), activeDistance=float(active),
               waypointEventList=[dict(index=-1, waypointEventData=[]) for _ in path])
    scene['objectList'].append(obj)
    scene['waypointDataList'].append(dict(pathName='ObstaclePath - %d' % uid, waypointData=path))
    manifest['actors'].append(dict(id=uid, label=label, sector=sector, kind='moving_object' if moving else 'static',
                                   model=model, speed_kmh=float(speed), active_distance_m=float(active),
                                   start=list(map(float, points[0])), end=list(map(float, points[-1])),
                                   path_points=len(path)))


def static_at(label, sector, lane, y, model=40100001, scale=(2., 3., 2.)):
    line = lanes[sector, lane]
    p, yaw = line.at(line.station_y(y))
    add_object(label, sector, [p], yaw=yaw, model=model, scale=scale)


def pedestrian(label, sector, station, lateral_start, lateral_end, speed, active):
    global uid
    uid += 1
    line = lanes[sector, 1]
    points = [line.at(station, d)[0] for d in np.linspace(lateral_start, lateral_end, 31)]
    p = copy.deepcopy(reference['pedestrianList'][0])
    d = points[1]-points[0]
    p.update(UNIQUEID=uid, pos=vec(points[0]), rot=rotation(math.atan2(d[1], d[0])),
             activeDistance=float(active), speed=float(speed), active=True, loop=False,
             initPos=unity(points[0]), standardPos=unity(points[0]), movingDistanceAmount=0.,
             stateNormalizedTime=0., isWaypointMode=True,
             waypointEventList=[dict(index=-1, waypointEventData=[]) for _ in points])
    scene['pedestrianList'].append(p)
    scene['waypointDataList'].append(dict(pathName='PedPath - %d' % uid,
        waypointData=[dict(pos=unity(q), rot=rotation(0.)) for q in points]))
    manifest['actors'].append(dict(id=uid, label=label, sector=sector, kind='pedestrian',
                                   speed_kmh=float(speed), active_distance_m=float(active),
                                   start=list(map(float, points[0])), end=list(map(float, points[-1])),
                                   path_points=len(points)))


# C077 signal: crossing after the stop line, activated on approach even after a red wait.
pedestrian('C077 신호 통과 직후 횡단', 3, 39., 3.5, -6., 7.5, 15.)
# C091 static-obstacle zone: adjacent parked vehicle + slow departing lead vehicle.
static_at('C091 인접 차로 주차 장애물', 4, 2, -142., model=40100012, scale=(1., 1., 1.))
line = lanes[4, 1]
add_object('C091 20km/h 선행차', 4,
           line.follow([(38., 0.), (78., 0.), (94., 0.), (109., 7.)]), speed=20., active=32.)
# S5/S6 static objects leave the adjacent same-direction lane unobstructed.
static_at('S5 점선 구간 차로 중앙 적재물', 5, 1, -65.)
pedestrian('S5 끝 1차 횡단', 5, lanes[5, 1].station_y(-19.), 3.5, -6., 7., 13.5)
pedestrian('S5 끝 반대쪽 후속 횡단', 5, lanes[5, 1].station_y(-10.), -4., 6., 5., 18.)
static_at('S6 교차로 뒤 연속 정적 장애물', 6, 1, 25.)
line = lanes[6, 1]
add_object('S6 회피 차로 저속 이륜차', 6,
           line.follow([(47., -3.2), (70., -3.2), (86., -8.)]),
           speed=2.2, active=45., model=40200014)
pedestrian('C052 좌회전 중 추가 횡단', 7, 47., -3.5, 6., 6., 15.)
# The roundabout conflict follows actual MGeo circulating links, not a line through the island.
links = {x['idx']: x for x in read(Path('/home/unita/Downloads/link_set.json'))}
circle = links['A2256W000134']['points']+links['A2256W000135']['points'][1:]
# Continue through the native northbound exit before leaving traffic. A
# waypoint actor stays at its last point: stopping on link135 would queue
# the periodic circulating NPCs back across Ego's merge.
circle += links['A2256W000102']['points'][1:]
circle_exit = Line(links['A2256W000848']['points'])
circle += circle_exit.follow([(0., 0.), (4., 0.), (24., -6.), (28., -6.)])[1:]
add_object('회전교차로 20km/h 순환 차량', 9, circle, speed=20., active=30.)
# Three staggered highway obstructions; no station blocks every lane.
static_at('S11 R3 정차 차량', 11, 3, 85., model=40100020, scale=(1., 1., 1.))
static_at('S11 R2 연속 정차 차량', 11, 2, 25., model=40100013, scale=(1., 1., 1.))
static_at('S11 R1 후속 정차 차량', 11, 1, -35., model=40100018, scale=(1., 1., 1.))
# GPS-only denied tunnel: a slow cut-in plus one adjacent parked car.
line = lanes[14, 1]
p, yaw = line.at(59., 3.5)
add_object('GPS 음영 터널 인접 차로 정차 차량', 14, [p], yaw=yaw, model=40100012)
add_object('GPS 음영 진입 20km/h 끼어들기 후 차로 이탈', 14,
           line.follow([(24., 3.5), (43., 0.), (95., 0.), (118., 3.5)]),
           speed=20., active=30.)

# Retain the original IDs and geometry while thinning the same scenario.
for key in ('objectList', 'pedestrianList'):
    scene[key] = [a for a in scene[key] if a['UNIQUEID'] not in REMOVED_ACTOR_IDS]
scene['waypointDataList'] = [p for p in scene['waypointDataList']
                             if int(p['pathName'].rsplit(' - ', 1)[1]) not in REMOVED_ACTOR_IDS]
manifest['actors'] = [a for a in manifest['actors'] if a['id'] not in REMOVED_ACTOR_IDS]

# Proximity cars remain available until Ego reaches S11, with a fixed actor count.
highway_objects, highway_paths, highway_actors = make_highway_traffic(project)
scene['objectList'].extend(highway_objects)
scene['waypointDataList'].extend(highway_paths)
manifest['actors'].extend(highway_actors)

# Current semi UDP ports. This saved template differs from latest test_scenario
# only by disabling unused publishers/services; sensors are a separate saved profile.
network_source = SOURCE / 'EgoNetwork/UNITA_raceline_dynamic_traffic_MN.json'
network_bytes = network_source.read_bytes()
manifest['network_template'] = str(network_source)
manifest['sensor_profile'] = str(SOURCE.parents[1] / 'Sensor/25.S4.MolitComp03/SensorInfo_2023_Hyundai_Ioniq5.json')
manifest['network_note'] = 'Existing semi-compatible command UDP9093, vehicle-status UDP9096. Not a certification of competition protocol compliance.'

# Structural and geometric validation of the actual serialized artifacts.
all_objects = [ego]+sum([scene[k] for k in ('vehicleList', 'objectList', 'pedestrianList', 'spawnPointList', 'shadedAreaList')], [])
ids = [o['UNIQUEID'] for o in all_objects]
assert len(ids) == len(set(ids)), 'duplicate actor UID'
paths = {p['pathName']: p['waypointData'] for p in scene['waypointDataList']}
assert len(paths) == len(scene['waypointDataList'])
for key, prefix in (('objectList', 'ObstaclePath'), ('pedestrianList', 'PedPath')):
    for obj in scene[key]:
        path = paths['%s - %d' % (prefix, obj['UNIQUEID'])]
        assert len(path) == len(obj['waypointEventList'])
        if path:
            np.testing.assert_allclose(xyz(path[0]['pos']), xyz(obj['initPos']), atol=1e-6)
            p = xyz(obj['pos'])
            np.testing.assert_allclose(xyz(path[0]['pos']), [-p[0], p[2], -p[1]], atol=1e-6)
            assert np.linalg.norm(p[:2]-start[:2]) > obj['activeDistance']+5., 'premature trigger at spawn'
        assert 26. < obj['pos']['z'] < 31., 'invalid road height'
for spawner in scene['spawnPointList']:
    for key in ('startLinkInfo', 'endLinkInfo'):
        ref = spawner[key]
        assert ref['linkIdx'] in links
        assert 0 <= ref['pointIdx'] < len(links[ref['linkIdx']]['points'])
    np.testing.assert_allclose(xyz(spawner['pos'])[:2],
        links[spawner['startLinkInfo']['linkIdx']]['points'][spawner['startLinkInfo']['pointIdx']][:2], atol=.03)
assert all(n['sensorType'] == 4 and n['noiseType'] == 1 and n['isActivated']
           for area in scene['shadedAreaList'] for n in area['noiseDatas'])
assert not any(a['sector'] in (12, 13) for a in manifest['actors'])
net = json.loads(network_bytes)['egoNetworkData'][0]['listEgoNetworkInfo']
assert {n['PUBSUB_TYPE'] for n in net if n['netType']} == {770, 771}
assert next(n for n in net if n['PUBSUB_TYPE'] == 771)['EgoCtrlConfig']['udpConfig']['hostPort'] == 9093
assert next(n for n in net if n['PUBSUB_TYPE'] == 770)['config']['udpConfig']['destinationPort'] == 9096

manifest['counts'] = {k: len(scene[k]) for k in ('objectList', 'pedestrianList', 'spawnPointList', 'tlList', 'shadedAreaList')}
manifest['checks'] = dict(unique_ids=True, waypoint_references=True, coordinate_conversion=True,
                         road_heights=True, no_activation_at_ego_spawn=True,
                         spawn_graph_links=True, gps_only_denial=True, tollgate_clear=True,
                         paired_udp_ports=True, native_load_tested=False)
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'EgoNetwork').mkdir(exist_ok=True)
(OUT/(NAME+'.json')).write_text(json.dumps(scene, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
(OUT/'EgoNetwork'/(NAME+'_MN.json')).write_bytes(network_bytes)
(OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(name=NAME, counts=manifest['counts'], checks=manifest['checks']), ensure_ascii=False))
