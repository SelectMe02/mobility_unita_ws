#!/usr/bin/env python3
"""Session archive: raw sensors in rosbag, decisions/configuration in JSONL.

Launched ONLY by diagnostic_log=true. Simulator truth is evidence for offline
analysis, never a driving/perception input. No simulator control commands.
"""
import collections
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import rospkg
import rospy
import rosnode
import yaml
from std_msgs.msg import String

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from driving_log_session import new_run_directory, announce


JSON_TOPICS = [
    '/waypoint_debug/ground_status', '/waypoint_debug/tracking_state',
    '/waypoint_debug/control_decision', '/waypoint_debug/obstacles',
    '/waypoint_debug/lane_geometry', '/waypoint_debug/avoidance_config/lidar_obstacles',
    '/waypoint_debug/avoidance_config/gps_waypoint_follower',
    '/control/udp_status', '/diagnostics/simulator/ego_status',
    '/diagnostics/simulator/ego_status_udp', '/diagnostics/control_tx',
    '/diagnostics/sensors/imu_udp', '/diagnostics/sensors/gps_udp',
    '/map_debug/avoidance_parameter_edit']
BAG_TOPICS = JSON_TOPICS + [
    '/lidar3D', '/velodyne_packets', '/imu', '/gps', '/localization/pose',
    '/localization/valid', '/ctrl_cmd', '/competition/ego_speed_kmh',
    '/control/sim_mode', '/waypoint_path', '/waypoint_debug/local_path',
    '/waypoint_debug/obstacle_horizon', '/waypoint_debug/ground_points',
    '/waypoint_debug/obstacle_points', '/waypoint_debug/target_speed_kmh',
    '/waypoint_debug/current_speed_kmh', '/waypoint_debug/speed_zone',
    '/waypoint_debug/path_curvature', '/rosout_agg', '/diagnostics',
    '/leo/waypoint_debug/obstacles', '/leo/route_avoidance',
    '/leo/waypoint_debug/local_path', '/leo/raw_ctrl_cmd',
    '/leo/signal_status', '/leo/speed_limit_kmh', '/leo/driving_pose']


def json_safe(value):
    """Normalize numpy-like numbers, nonfinite values, and XMLRPC data."""
    import math
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def record_command(directory):
    return ['rosbag', 'record', '--lz4', '--split', '--size=512', '-b', '256',
            '-o', str(Path(directory) / 'sensors')] + BAG_TOPICS


class DrivingLogRecorder:
    def __init__(self):
        explicit = rospy.get_param('~directory', '')
        if explicit:
            self.directory = Path(explicit).resolve()
            self.directory.mkdir(parents=True, exist_ok=True)
        else:
            # Default: <root>/run_YYYYmmdd_HHMMSS, shared with the follower.
            self.directory = new_run_directory(rospy.get_param('~root'))
        announce(rospy, self.directory, rospy.get_param('~session', ''))
        self.lock = threading.Lock()
        self.closed = False
        self.counts = collections.Counter()
        self.last_samples = {}
        self.last_parameters = None
        self.events = (self.directory / 'events.jsonl').open('a', buffering=1)
        self.manifest = dict(created=datetime.now().astimezone().isoformat(),
            ros_start=rospy.Time.now().to_sec(), diagnostic_only=True,
            directory=str(self.directory), bag_topics=BAG_TOPICS,
            truth_topics=['/diagnostics/simulator/ego_status', '/diagnostics/simulator/ego_status_udp'],
            missing_topics=[], snapshots=[], notes=[
                'Raw simulator fields are diagnostic evidence only.',
                'Sensor mounting settings must match MORAI; ROS params cannot read the SIM GUI.',
                'rosbag subscribes to point debug topics even when RViz is absent.'])
        self.write_manifest()
        self.process_output = (self.directory / 'rosbag_record.log').open('a', buffering=1)
        # Start synchronously, so rosbag's own signal finalizes .bag on shutdown.
        self.process = subprocess.Popen(record_command(self.directory),
            stdout=self.process_output, stderr=subprocess.STDOUT, start_new_session=True)
        rospy.on_shutdown(self.close)
        self.subscribers = [rospy.Subscriber(topic, String, self.on_json,
            callback_args=topic, queue_size=100) for topic in JSON_TOPICS]
        self.sensor_subscribers = [rospy.Subscriber(topic, rospy.AnyMsg, self.on_sensor,
            callback_args=topic, queue_size=1, buff_size=2**24)
            for topic in ('/lidar3D', '/imu', '/gps', '/localization/pose', '/ctrl_cmd')]
        self.snapshot_sources()
        self.timer = rospy.Timer(rospy.Duration(2.), self.health)
        rospy.loginfo('Driving log session (log=true): %s', self.directory)

    def write_manifest(self):
        with self.lock:
            temporary = self.directory / 'manifest.tmp'
            temporary.write_text(json.dumps(json_safe(self.manifest), ensure_ascii=False, indent=2)+'\n')
            temporary.replace(self.directory / 'manifest.json')

    def append(self, topic, data):
        with self.lock:
            if self.closed:
                return
            self.counts[topic] += 1
            self.last_samples[topic] = time.monotonic()
            self.events.write(json.dumps(json_safe(dict(
                time=datetime.now().astimezone().isoformat(timespec='milliseconds'),
                ros_time=rospy.Time.now().to_sec(), topic=topic, data=data)),
                ensure_ascii=False, allow_nan=False)+'\n')

    def on_json(self, message, topic):
        try:
            data = json.loads(message.data)
        except (ValueError, TypeError):
            data = message.data
        self.append(topic, data)

    def on_sensor(self, _message, topic):
        with self.lock:
            self.counts[topic] += 1
            self.last_samples[topic] = time.monotonic()

    def snapshot_sources(self):
        pack = Path(rospkg.RosPack().get_path('unita_waypoint'))
        config = self.directory / 'config'
        config.mkdir(exist_ok=True)
        paths = [Path(rospy.get_param('~waypoint_file', str(pack / 'config/map_tunner_project.json'))),
                 pack / 'config/ground_filter.yaml', pack / 'config/raceline_layout.yaml',
                 pack / 'config/morai_sensor_settings.json',
                 pack / 'config/competition_speed_profile.yaml',
                 pack / 'config/map_tunner_calibration.yaml', pack / 'config/map_tunner_checkpoints.yaml',
                 Path(rospkg.RosPack().get_path('control')) / 'config/competition_udp.yaml',
                 Path(rospkg.RosPack().get_path('localization')) / 'config/gps_imu.yaml',
                 Path(rospkg.RosPack().get_path('sensor_bridge')) / 'config/sensors.yaml']
        for source in paths:
            if source.is_file():
                content = source.read_bytes()
                target = config / source.name
                target.write_bytes(content)
                self.manifest['snapshots'].append(dict(source=str(source), copy=str(target),
                    sha256=hashlib.sha256(content).hexdigest()))
        code = self.directory / 'code'
        code.mkdir(exist_ok=True)
        sources = [pack / 'scripts' / name for name in ('ground_segmentation.py', 'lidar_obstacles.py',
                'local_avoidance_core.py', 'gps_waypoint_follower.py', 'raceline_test_follower.py', 'avoidance_runtime.py', 'pedal_guard.py')]
        control = Path(rospkg.RosPack().get_path('control'))
        sources += [control / 'scripts/competition_udp_control.py',
                    control / 'src/unita_control/competition_protocol.py',
                    control / 'src/unita_control/udp_protocol.py',
                    Path(rospkg.RosPack().get_path('sensor_bridge')) / 'scripts/udp_sensor_bridge.py',
                    Path(rospkg.RosPack().get_path('localization')) / 'scripts/gps_imu_localization.py']
        for source in sources:
            content = source.read_bytes()
            (code / source.name).write_bytes(content)
            self.manifest['snapshots'].append(dict(source=str(source), copy=str(code / source.name),
                                                   sha256=hashlib.sha256(content).hexdigest()))
        self.write_manifest()

    def health(self, _event):
        try:
            # get_param('/') can return rospy's partially populated local
            # cache. Read the authoritative master tree for a full snapshot.
            code, message, params = rospy.get_master().getParam('/')
            if code != 1:
                raise rospy.ROSException(message)
            signature = yaml.safe_dump(params, allow_unicode=True, sort_keys=True)
            if signature != self.last_parameters:
                (self.directory / 'rosparams.yaml').write_text(signature)
                lidar = params.get('lidar_obstacles', {})
                (self.directory / 'lidar_params.yaml').write_text(yaml.safe_dump(lidar, allow_unicode=True))
                self.last_parameters = signature
                self.append('parameters', params)
            published = dict(rospy.get_published_topics('/'))
            now = time.monotonic()
            with self.lock:
                counts = dict(self.counts)
                ages = {topic: now-stamp for topic, stamp in self.last_samples.items()}
            health = dict(counts=counts, receive_age_s=ages,
                missing_topics=[topic for topic in BAG_TOPICS if topic not in published],
                published_topics=published, recorder_exit_code=self.process.poll(),
                free_disk_bytes=shutil.disk_usage(self.directory).free,
                nodes=rosnode.get_node_names())
            self.append('session_health', health)
            self.manifest.update(missing_topics=health['missing_topics'],
                                 last_health_ros_time=rospy.Time.now().to_sec())
            self.write_manifest()
            if health['recorder_exit_code'] is not None:
                rospy.logerr_throttle(5., 'rosbag recorder exited; inspect %s/rosbag_record.log', self.directory)
        except (OSError, rospy.ROSException, rosnode.ROSNodeIOException) as error:
            rospy.logwarn_throttle(5., 'Diagnostic snapshot failed: %s', error)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        if hasattr(self, 'timer'):
            self.timer.shutdown()
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=6.)
            except subprocess.TimeoutExpired:
                # Do not truncate the bag by SIGKILL; child continues finalizing.
                rospy.logwarn('rosbag is still finalizing in %s', self.directory)
        self.process_output.close()
        self.events.close()
        self.manifest.update(closed=datetime.now().astimezone().isoformat(),
                             ros_end=rospy.Time.now().to_sec(), counts=dict(self.counts),
                             bag_files=[path.name for path in sorted(self.directory.glob('*.bag*'))])
        self.write_manifest()


if __name__ == '__main__':
    rospy.init_node('driving_log_recorder')
    DrivingLogRecorder()
    rospy.spin()
