#!/usr/bin/env python3
"""MORAI 신호등 교차로 시나리오 생성기 (R_KR_PR_K-city_2025, 비음영 구간).

대회 코스 구간에 맞춰 배치한다(경로 순서, START 기준):
  C077  신호등             -> 노란불 급변 / 노란불 유지
  C091  신호등 & 정적 장애물 -> 정지선 앞/교차로 출구 정적 장애물
  C016  신호등 & 동적 장애물 -> 초록불/노란불 무단횡단
  (C123 우회전, C052 좌회전 신호등은 아직 시나리오 없음)

  SIG01  초록불 무단횡단 (C016): 지정된 횡단보도 위치에서 -X로 보행자 이동
  SIG02  노란불 무단횡단 (C016): 출발 거리 스윕 + SIG01 보행자
  SIG03  노란불 출발 거리 스윕 (C077): 정지선 도착 시점에 따라
         - 정지선 가까이에서 노란불로 바뀜 -> 노란불 급변(딜레마)
         - 멀리서 노란불이 켜짐 -> 정지선 앞 정지, 빨간불 동안 유지, 초록불 출발
  SIG04  정적 장애물 정지선 20 m 앞, 자기 차로 (C091)
  SIG05  정적 장애물 교차로 출구 뒤 15 m, 자기 차로 (C091)
  SIG06  정적 장애물 정지선 뒤 60 m, 자기 차로 (C016 출구 뒤, 오른쪽 점선)
  SIG07  지정된 C016 횡단보도 위치에서 -X로 보행자 이동 (C016, C077)

모든 보행자는 같은 ENU 좌표에서 출발해 X가 감소하는 방향으로 걷는다.
C077용 SIG07도 같은 절대 좌표를 사용하므로 C077 차량과는 멀리 떨어져 있다.

MORAI 신호 위상은 시나리오 tlList(공식 값) 그대로라 매 실행 같다. 노란불이
언제 켜지는지는 아직 측정하지 않았으므로 노란불 시나리오는 출발 거리만 바꾼
여러 파일로 만든다. 한 번 돌려 보고 급변/유지에 맞는 파일을 고른다.

사용법:
  python3 generate_signal_scenarios.py              # MORAI 폴더에 설치 (있는 파일은 건너뜀)
  python3 generate_signal_scenarios.py --force      # 덮어쓰기
  python3 generate_signal_scenarios.py --dry-run    # 좌표만 출력
  python3 generate_signal_scenarios.py --uninstall  # 이 스크립트가 만든 SIG 파일 제거
"""
import argparse
import copy
import json
import math
import os
import shutil

HOME = os.path.expanduser("~")
MORAI_DIR = os.path.join(
    HOME, "MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Scenario/R_KR_PR_K-city_2025")
PATH_FILE = os.path.join(HOME, "Downloads/2026_molit_comp_global_path.txt")
INTERSECTIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "config", "intersections.json")

SIGNAL_TEMPLATE = "LEO_start_parked.json"          # 공식 tlList, 장애물 없음
PED_TEMPLATE = "00_official_reference.json"        # 보행자 블록 형식
MN_TEMPLATE = "EgoNetwork/blackout_obstactle five_MN.json"

# 공식 예제의 정적 장애물 (CargoBox 확대)
STATIC_BOX = (40100001, (2.0, 3.0, 2.0))
# 신호등 바로 앞에서 출발한다(신호 판단과 교차로 인근 회피가 목적).
APPROACH_M = 55.                          # 정지선까지 출발 거리 [m]
C077_YELLOW_START_M = range(45, 106, 15)  # 노란불 스윕: 정지선까지 출발 거리 [m]
C016_YELLOW_START_M = range(45, 106, 15)  # 100 m 이상은 C091 교차로 안에서 출발


# ---------------------------------------------------------------- 경로 기하
class Route:
    def __init__(self, path_file):
        self.p = [list(map(float, l.split()))[:3] for l in open(path_file) if l.strip()]
        self.s = [0.0]
        for a, b in zip(self.p, self.p[1:]):
            self.s.append(self.s[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))

    def nearest_s(self, x, y, lo=0, hi=None):
        hi = len(self.p) if hi is None else hi
        k = min(range(lo, hi), key=lambda i: (self.p[i][0] - x) ** 2 + (self.p[i][1] - y) ** 2)
        return self.s[k]

    def pose(self, s, lat=0.0):
        """route 호장 s, 왼쪽(+) 횡오프셋 lat -> (x, y, z, heading_rad)."""
        i = max(0, min(len(self.s) - 2, next((j for j, v in enumerate(self.s) if v > s), len(self.s) - 1) - 1))
        a, b = self.p[i], self.p[i + 1]
        seg = max(self.s[i + 1] - self.s[i], 1e-6)
        t = (s - self.s[i]) / seg
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        x = a[0] + t * (b[0] - a[0]) - math.sin(h) * lat
        y = a[1] + t * (b[1] - a[1]) + math.cos(h) * lat
        z = a[2] + t * (b[2] - a[2])
        return x, y, z, h


def straight(p0, p1, step=1.0):
    n = max(1, int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step))
    return [tuple(p0[j] + (p1[j] - p0[j]) * k / n for j in range(3)) for k in range(n + 1)]


# ---------------------------------------------------------------- JSON 블록
def v3(x, y, z):
    return {"x": x, "y": y, "z": z, "_x": "%.3f" % x, "_y": "%.3f" % y, "_z": "%.3f" % z}


def rot(yaw_deg=0.0):
    return {"roll": "0.000", "pitch": "0.000", "yaw": "%.3f" % yaw_deg}


def unity(p):
    """시나리오 ENU -> waypoint 저장 좌표(Unity): (-x, z, -y). PedPath 예시와 동일."""
    return v3(-p[0], p[2], -p[1])


def heading_deg(pts):
    if len(pts) < 2:
        return 0.0
    return math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))


class Builder:
    def __init__(self, base, ped_template):
        self.d = copy.deepcopy(base)
        for key in ("vehicleList", "objectList", "pedestrianList", "spawnPointList",
                    "waypointDataList"):
            self.d[key] = []
        self.ped_template = ped_template
        self.uid = 101

    def _next(self):
        self.uid += 1
        return self.uid

    def ego(self, x, y, z, yaw_deg):
        ip = self.d["egoVehicle"]["initPosition"]
        ip["pos"] = v3(x, y, z)
        ip["rot"] = rot(yaw_deg)
        ip["initLink"] = ""
        sd = ip.get("vehicleSaveData")
        if sd:
            sd["velocity"] = 0.0
            sd["vehicle"]["lastVelocity"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.d["egoVehicle"]["currentVelocity"] = 0.0

    def parked(self, data_id, p, yaw_deg, scale=(1.0, 1.0, 1.0)):
        uid = self._next()
        self.d["objectList"].append({
            "DataID": data_id, "UNIQUEID": uid, "m_eobstacleObjType": 4,
            "pos": v3(*p), "rot": rot(yaw_deg), "scale": v3(*scale),
            "initPos": v3(0.0, 0.0, 0.0), "initRot": rot(),
            "isWaypointMode": True, "moveSpeed": 0.0, "activeDistance": 0.0,
            "waypointEventList": [],
        })
        self.d["waypointDataList"].append({"pathName": "ObstaclePath - %d" % uid,
                                           "waypointData": []})

    def ped(self, pts, speed_kmh, active_m):
        uid = self._next()
        p = copy.deepcopy(self.ped_template)
        p.update({
            "UNIQUEID": uid, "pos": v3(*pts[0]), "rot": rot(heading_deg(pts)),
            "activeDistance": float(active_m), "speed": float(speed_kmh),
            "active": True, "loop": False,
            "initPos": unity(pts[0]), "standardPos": unity(pts[0]),
            "isWaypointMode": True,
            "waypointEventList": [{"index": -1, "waypointEventData": []} for _ in pts],
        })
        self.d["pedestrianList"].append(p)
        self.d["waypointDataList"].append({
            "pathName": "PedPath - %d" % uid,
            "waypointData": [{"pos": unity(q), "rot": rot()} for q in pts],
        })


# ---------------------------------------------------------------- 시나리오
# 사용자 지정 MORAI 화면의 ENU 좌표. 모든 보행자 시나리오가 이 점에서 출발한다.
PED_SPAWN_ENU = (-57.49, -27.62, 28.57)
PED_END_X = PED_SPAWN_ENU[0] - 20.0
PED_SPEED_KMH = 7.0
# At 7 m the walker entered the route after Ego had begun crossing it.
# At 25 m the walker reaches the route roughly 1 s before a 30 km/h Ego.
PED_JAYWALK_ACTIVE_M = 25.0


def shared_pedestrian(speed_kmh, active_m):
    end = (PED_END_X, PED_SPAWN_ENU[1], PED_SPAWN_ENU[2])
    return straight(PED_SPAWN_ENU, end), speed_kmh, active_m


# 횡단보도 무단횡단(보행자 신호 빨간불, 차량 신호 초록불): 정지선 바로 뒤
# 횡단보도를 시나리오 시작과 함께 천천히 건너므로, Ego가 횡단보도에 닿기 전에
# 이미 차로 안에 있다. Ego는 반드시 횡단보도 앞에서 멈췄다가 다 건넌 뒤 출발해야 한다.
CROSSWALK_PED_SPEED_KMH = 4.0


def signal_scenarios(route, ints):
    by_tl = {i["traffic_light_id"]: i for i in ints if i.get("traffic_light_id")}

    def stop_s(tl):
        i = by_tl[tl]
        x, y = i["stop_point"][:2]
        k = i["waypoint_index"]
        return route.nearest_s(x, y, max(0, k - 40), min(len(route.p), k + 40)), i

    def box(s, lat=0.0):
        x, y, z, h = route.pose(s, lat)
        return (x, y, z), math.degrees(h), STATIC_BOX

    sc = {}

    # 신호등 & 동적 장애물 (C016): 지정된 위치에서 -X로 횡단.
    s0, _ = stop_s("C1256W000016")
    ped = shared_pedestrian(PED_SPEED_KMH, PED_JAYWALK_ACTIVE_M)
    sc["SIG01_green_jaywalk_ped_after_intersection_C016"] = dict(ego_s=s0 - APPROACH_M, peds=[ped])
    for d in C016_YELLOW_START_M:
        sc["SIG02_yellow_jaywalk_ped_after_intersection_C016_start_%03dm" % d] = dict(
            ego_s=s0 - d, peds=[ped])

    # 신호등 (C077): 노란불 출발 거리 스윕.
    s0, _ = stop_s("C1256W000077")
    for d in C077_YELLOW_START_M:
        sc["SIG03_yellow_change_C077_start_%03dm" % d] = dict(ego_s=s0 - d)

    # 신호등 & 정적 장애물 (C091, 단일 차로 접근로).
    s0, i = stop_s("C1256W000091")
    # C091 주변은 지도상 차선 변경 불가(실선): 장애물 앞 대기가 정상 동작.
    sc["SIG04_static_obstacle_20m_before_stopline_C091"] = dict(
        ego_s=s0 - 20 - APPROACH_M, parked=[box(s0 - 20)])
    sc["SIG05_static_obstacle_15m_after_intersection_C091"] = dict(
        ego_s=s0 - APPROACH_M, parked=[box(s0 + i["clearance_m"] + 15)])

    # C016 출구 뒤는 오른쪽 같은 방향 차로로 변경 가능(점선, 경로 471~551 m):
    # 오른쪽 차로로 비켜 지나간 뒤 복귀해야 한다.
    s0, _ = stop_s("C1256W000016")
    sc["SIG06_static_obstacle_60m_after_stopline_dashed_right_lane_C016"] = dict(
        ego_s=s0 - APPROACH_M, parked=[box(s0 + 60)])

    # 보행자 신호가 아닌데 지정된 횡단보도에서 -X로 걷는 보행자.
    s0, _ = stop_s("C1256W000016")
    sc["SIG07_crosswalk_jaywalk_ped_on_red_must_stop_C016"] = dict(
        ego_s=s0 - APPROACH_M,
        peds=[shared_pedestrian(CROSSWALK_PED_SPEED_KMH, APPROACH_M + 40.)])
    s0, _ = stop_s("C1256W000077")
    sc["SIG07_crosswalk_jaywalk_ped_on_red_must_stop_C077"] = dict(
        ego_s=s0 - APPROACH_M,
        peds=[shared_pedestrian(CROSSWALK_PED_SPEED_KMH, APPROACH_M + 40.)])
    return sc


# ---------------------------------------------------------------- 출력
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="이미 있는 파일도 덮어쓰기")
    ap.add_argument("--uninstall", action="store_true", help="이 스크립트의 SIG 파일 제거")
    ap.add_argument("--morai-dir", default=MORAI_DIR)
    args = ap.parse_args()

    load = lambda f: json.load(open(os.path.join(args.morai_dir, f)))
    route = Route(PATH_FILE)
    ints = json.load(open(INTERSECTIONS))["intersections"]
    base = load(SIGNAL_TEMPLATE)
    ped_template = load(PED_TEMPLATE)["pedestrianList"][0]

    for name, spec in signal_scenarios(route, ints).items():
        target = os.path.join(args.morai_dir, name + ".json")
        network = os.path.join(args.morai_dir, "EgoNetwork", name + "_MN.json")
        if args.uninstall:
            for path in (target, network):
                if os.path.exists(path):
                    os.remove(path)
            print("removed", name)
            continue
        b = Builder(base, ped_template)
        x, y, z, h = route.pose(spec["ego_s"])
        b.ego(x, y, z, math.degrees(h))
        for p, yaw, (data_id, scale) in spec.get("parked", []):
            b.parked(data_id, p, yaw, scale)
        for pts, speed, active in spec.get("peds", []):
            b.ped(pts, speed, active)
        print("%-62s ego(%.1f, %.1f)" % (name, x, y))
        for pts, speed, active in spec.get("peds", []):
            print("    ped (%.1f, %.1f) -> (%.1f, %.1f) %.0f km/h active %.0f m"
                  % (pts[0][0], pts[0][1], pts[-1][0], pts[-1][1], speed, active))
        if args.dry_run:
            continue
        if os.path.exists(target) and not args.force:
            print("    exists, skipped (use --force)")
            continue
        with open(target, "w") as f:
            json.dump(b.d, f, ensure_ascii=False, indent=2)
        shutil.copyfile(os.path.join(args.morai_dir, MN_TEMPLATE), network)


if __name__ == "__main__":
    main()
