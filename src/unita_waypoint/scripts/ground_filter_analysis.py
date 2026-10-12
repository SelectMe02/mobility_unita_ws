#!/usr/bin/env python3
"""Offline braking-bag inspection using permitted IMU/LiDAR only.

No ROS master or simulator connection; does not issue driving commands.
Use --config with rosparam dump of /lidar_obstacles for calibrated extrinsics.
"""
import argparse
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np
import rosbag
import rospkg
import yaml
from sensor_msgs import point_cloud2

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from ground_segmentation import normalized_quaternion, rotation_rpy
from lidar_obstacles import LidarObstacles, road_obstacle_mask


def summarize(rows):
    corrected = [r for r in rows if r['imu_corrected']]
    pitches = [r['pitch_deg'] for r in corrected]
    residuals = [r['residual_p95_m'] for r in rows if r['residual_p95_m'] is not None]
    skews = [r['imu_skew_s'] for r in rows if r['imu_skew_s'] is not None]
    return dict(scans=len(rows), corrected_scans=len(corrected),
                pitch_min_deg=min(pitches) if pitches else None,
                pitch_max_deg=max(pitches) if pitches else None,
                imu_skew_p95_s=float(np.percentile(skews, 95)) if skews else None,
                supported_road_residual_p95_m=float(np.percentile(residuals, 95)) if residuals else None,
                legacy_ground_false_candidate_points=sum(r['legacy_ground_candidates'] for r in rows),
                warning='Residuals are conditional on confirmed road and the configured tolerance. '
                        'Check point colours and real obstacles before changing tolerance. '
                        'Pitch variation alone does not prove simulator body pitch accuracy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag')
    parser.add_argument('--config', default=str(Path(rospkg.RosPack().get_path('unita_waypoint')) /
                                               'config/ground_filter.yaml'))
    parser.add_argument('--stride', type=int, default=10, help='Analyse every Nth cloud (default 10)')
    parser.add_argument('--output', help='Write summary and per-scan diagnostics as JSON')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    with open(args.config) as stream:
        config = yaml.safe_load(stream) or {}
    detector = LidarObstacles.__new__(LidarObstacles)
    detector.lock, detector.imus = threading.Lock(), []
    detector.imu_frame_id = config.get('imu_frame_id', 'imu')
    def mount(prefix):
        return rotation_rpy(*(math.radians(float(config.get(prefix+'_'+axis+'_deg', 0.)))
                              for axis in ('roll', 'pitch', 'yaw')))
    detector.body_from_imu, detector.body_from_lidar = mount('imu_mount'), mount('lidar_mount')
    for attr, key, default in [
            ('lidar_x', 'lidar_to_rear_x_m', 1.4), ('lidar_y', 'lidar_to_rear_y_m', 0.),
            ('lidar_z', 'lidar_to_rear_z_m', 1.23), ('front_range', 'front_range_m', 55.),
            ('rear_range', 'rear_range_m', 55.), ('side_range', 'side_range_m', 12.),
            ('half_width', 'vehicle_half_width_m', .946), ('front_extent', 'front_extent_m', 3.845),
            ('rear_extent', 'rear_extent_m', .790), ('imu_max_skew_s', 'imu_max_skew_s', .05),
            ('ground_plane_tolerance_m', 'ground_plane_tolerance_m', .04),
            ('ground_max_slope_deg', 'ground_max_slope_deg', 20.),
            ('obstacle_min_height_m', 'obstacle_min_height_m', .22),
            ('obstacle_max_height_m', 'obstacle_max_height_m', 2.6)]:
        setattr(detector, attr, float(config.get(key, default)))
    detector.ground_reference_z_m = float(config.get('ground_reference_z_m', 0.))
    with rosbag.Bag(args.bag) as bag:
        for _, message, _ in bag.read_messages(topics=['/imu']):
            if message.header.frame_id != detector.imu_frame_id or message.orientation_covariance[0] == -1:
                continue
            q = message.orientation
            try:
                orientation = normalized_quaternion((q.x, q.y, q.z, q.w))
            except ValueError:
                continue
            stamp = message.header.stamp.to_sec()
            if detector.imus and stamp < detector.imus[-1][0]:
                parser.error('IMU time reset in bag; analyse each clock epoch separately')
            if stamp > 0. and (not detector.imus or stamp > detector.imus[-1][0]):
                detector.imus.append((stamp, orientation))
        imu_samples = detector.imus
        imu_times = np.asarray([s[0] for s in imu_samples])
        rows = []
        for index, (_, message, _) in enumerate(bag.read_messages(topics=['/lidar3D'])):
            if index % args.stride or message.header.frame_id != 'lidar':
                continue
            raw = np.asarray(list(point_cloud2.read_points(message,
                field_names=('x', 'y', 'z'), skip_nans=True)), dtype=float).reshape((-1, 3))
            nearest = int(np.searchsorted(imu_times, message.header.stamp.to_sec()))
            detector.imus = imu_samples[max(0, nearest-2):nearest+2]
            _, ground, _, status = detector.filter_ground(raw, message.header.stamp.to_sec())
            old = raw @ detector.body_from_lidar.T + [detector.lidar_x, detector.lidar_y, detector.lidar_z]
            legacy = road_obstacle_mask(old, detector.front_range, detector.rear_range,
                                       detector.side_range, detector.half_width,
                                       detector.front_extent, detector.rear_extent,
                                       detector.obstacle_min_height_m, detector.obstacle_max_height_m)
            status['legacy_ground_candidates'] = int(np.count_nonzero(legacy & ground))
            rows.append(status)
    result = dict(bag=args.bag, config=args.config, stride=args.stride, summary=summarize(rows), scans=rows)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
