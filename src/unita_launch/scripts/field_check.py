#!/usr/bin/env python3
"""One-shot, read-only MORAI field checks. Never publishes vehicle commands."""
import datetime
import json
import math
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

try:
    from unita_visualization.field_diagnostics import Check, SensorStats, aggregate, ip_check
except ImportError:
    # Also let an unsourced source checkout explain the missing ROS environment.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from unita_visualization.field_diagnostics import Check, SensorStats, aggregate, ip_check


SOURCE_HINT = ('source /opt/ros/noetic/setup.bash',
               'source ~/catkin_ws/devel/setup.bash',
               'source ~/unita_ws/devel/setup.bash')
SERVICE_HINT = ('Check MORAI ROS Bridge IP = Ubuntu Ethernet IPv4 and port 9090.',
                'Keep Ego Network Service on ROS and Connect ON.',
                'Check /Service_MoraiEventCmd and both PCs\' firewall settings.')


class Report:
    def __init__(self):
        name = 'unita_field_check_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
        self.path = Path('/tmp') / name
        if self.path.exists():
            self.path = self.path.with_name(self.path.stem + '_' + datetime.datetime.now().strftime('%f') + '.log')
        self.file = self.path.open('x', encoding='utf-8')
        self.checks = []

    def line(self, text=''):
        print(text, flush=True)
        self.file.write(text + '\n')
        self.file.flush()

    def add(self, check):
        self.checks.append(check)
        self.line('[{}] {:25} {}'.format(check.status, check.name, check.detail))
        if check.causes:
            self.line('  Possible causes / actions:')
            for cause in check.causes:
                self.line('  - ' + cause)

    def finish(self):
        result, code = aggregate(self.checks)
        self.line('=' * 60)
        self.line('RESULT: ' + result)
        self.line('Diagnostic exit code: ' + str(code))
        self.line('Log: ' + str(self.path))
        self.line('Read-only check: no CtrlCmd publication and no control UDP sends.')
        self.line('=' * 60)
        self.file.close()
        return code


def bounded_call(function, timeout):
    """Bound rosbridge service / XMLRPC waits without changing global sockets."""
    result = queue.Queue(maxsize=1)

    def worker():
        try:
            result.put((True, function()))
        except Exception as error:
            result.put((False, error))

    threading.Thread(target=worker, daemon=True).start()
    try:
        success, value = result.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError('No response within {:.1f}s'.format(timeout))
    if not success:
        raise value
    return value


def command(args, timeout=3):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'command failed: ' + ' '.join(args))
    return result.stdout


def ethernet_interfaces(addresses):
    interfaces = {}
    for item in addresses:
        name = item['ifname']
        base = Path('/sys/class/net') / name
        if ('UP' not in item.get('flags', []) or 'LOWER_UP' not in item.get('flags', [])
                or not (base / 'device').exists() or (base / 'wireless').exists()):
            continue
        ips = [addr['local'] for addr in item.get('addr_info', [])
               if addr.get('family') == 'inet' and addr.get('scope') == 'global']
        if ips:
            interfaces[name] = ips
    return interfaces


def network(report, address, field_mode):
    check = ip_check(address, field_mode)
    report.add(check)
    if check.status == 'FAIL':
        return None
    source_ip = None
    try:
        addresses = json.loads(command(['ip', '-j', '-4', 'addr']))
        wired = ethernet_interfaces(addresses)
        report.line('Active Ethernet IPv4: ' + (json.dumps(wired) if wired else 'none'))
        if field_mode:
            report.add(Check('Ethernet', 'PASS' if wired else 'FAIL',
                             json.dumps(wired) if wired else 'No active wired interface with IPv4',
                             () if wired else ('Check the LAN cable and Ethernet IPv4/netmask.',
                                               'Use compatible IPv4 subnets agreed with the venue.')))
        else:
            report.add(Check('Ethernet', 'PASS', 'Local test mode; wired Ethernet is not required'))
        report.line('IPv4 routes:\n' + command(['ip', '-4', 'route']).rstrip())
        route = json.loads(command(['ip', '-j', '-4', 'route', 'get', address]))[0]
        source_ip = route.get('prefsrc', route.get('src'))
        device = route.get('dev', '?')
        status = 'PASS'
        detail = 'to {} via {}, source={}'.format(address, device, source_ip)
        if field_mode and device not in wired:
            status = 'WARN' if address.startswith('127.') else 'FAIL'
            detail += '; route does not use the wired Ethernet link'
        report.add(Check('Network route', status, detail,
                         () if status == 'PASS' else ('Check morai_ip, Ethernet address/netmask and route.',)))
    except (OSError, ValueError, KeyError, IndexError, RuntimeError, subprocess.TimeoutExpired) as error:
        report.add(Check('Network route', 'FAIL', str(error),
                         ('Run ip -4 addr and ip route; verify the Windows Client IPv4.',)))
    try:
        command(['ping', '-n', '-c', '1', '-W', '1', address], timeout=2)
        report.add(Check('Client ping', 'PASS', address))
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report.add(Check('Client ping', 'WARN', 'ICMP failed: ' + str(error),
                         ('Windows may block ICMP. Use sensor/service results to judge connectivity.',)))
    return source_ip


def udp_listeners():
    ports = set()
    for line in command(['ss', '-H', '-u', '-l', '-n']).splitlines():
        fields = line.split()
        if len(fields) >= 5:
            try:
                ports.add(int(fields[3].rsplit(':', 1)[-1]))
            except ValueError:
                pass
    return ports


def service_check(report, rospy, name, timeout):
    try:
        from morai_msgs.srv import MoraiEventCmdSrv, MoraiEventCmdSrvRequest
        query = MoraiEventCmdSrvRequest()
        query.request.option = 0
        query.request.gear = -1

        def read_only_query():
            rospy.wait_for_service(name, timeout=timeout)
            return rospy.ServiceProxy(name, MoraiEventCmdSrv)(query)

        response = bounded_call(read_only_query, timeout).response
        label = {1: 'Manual', 3: 'ExternalCtrl', 6: 'Built-in autonomy'}.get(response.ctrl_mode, 'Other')
        report.add(Check('MORAI Event Service', 'PASS',
                         '{}; ctrl_mode={} ({}); option=0, gear=-1 query only'.format(name, response.ctrl_mode, label)))
    except Exception as error:
        report.add(Check('MORAI Event Service', 'FAIL', str(error), SERVICE_HINT))


def control_check(report, rospy, paths, topics, nodes, address, port):
    path = paths.get('control')
    try:
        if not path:
            raise ValueError('control package is unavailable')
        launch = Path(path) / 'launch/morai_udp_control.launch'
        arguments = {arg.attrib['name'] for arg in ET.parse(str(launch)).getroot().findall('arg')}
        if not {'morai_ip', 'morai_port'} <= arguments:
            raise ValueError('UDP launch does not expose morai_ip/morai_port')
        if not 1 <= port <= 65535:
            raise ValueError('Control UDP port must be 1..65535')
        actual_type = topics.get('/ctrl_cmd')
        if actual_type and actual_type != 'morai_msgs/CtrlCmd':
            raise ValueError('/ctrl_cmd has type ' + actual_type)
        for key, expected in (('morai_ip', address), ('morai_port', port)):
            if '/morai_udp_control' in nodes:
                actual = rospy.get_param('/morai_udp_control/' + key, None)
                if actual != expected:
                    raise ValueError('Running sender {}={} differs from expected {}'.format(key, actual, expected))
        detail = '{}:{}; morai_ip is configurable; /ctrl_cmd {}'.format(
            address, port, actual_type or 'not published (expected before driving)')
        report.add(Check('Control configuration', 'PASS' if port == 9093 else 'WARN', detail))
        report.line('Control configuration does not verify Windows UDP receipt or vehicle actuation (no ACK).')
    except (OSError, ET.ParseError, ValueError) as error:
        report.add(Check('Control configuration', 'FAIL', str(error),
                         ('Cmd Control receive IP/port must match Windows Ethernet IPv4 and morai_port.',)))


def run(report):
    report.line('=' * 60)
    report.line('UNITA MORAI FIELD CHECK')
    report.line('=' * 60)
    distro = os.environ.get('ROS_DISTRO', '')
    report.add(Check('ROS Noetic', 'PASS' if distro == 'noetic' else 'FAIL',
                     'ROS_DISTRO=' + (distro or '(unset)'), () if distro == 'noetic' else SOURCE_HINT))
    report.line('ROS_MASTER_URI=' + os.environ.get('ROS_MASTER_URI', 'http://localhost:11311'))
    report.line('Recommended source order:\n' + '\n'.join(SOURCE_HINT))
    try:
        import rosgraph
        import rospkg
        import rospy
        import yaml
    except ImportError as error:
        report.add(Check('ROS Python environment', 'FAIL', str(error), SOURCE_HINT))
        return
    paths = {}
    rospack = rospkg.RosPack()
    for package in ('morai_msgs', 'sensor_bridge', 'localization', 'control',
                    'unita_launch', 'unita_waypoint', 'rosbridge_server'):
        try:
            paths[package] = rospack.get_path(package)
            report.add(Check(package, 'PASS', paths[package]))
        except rospkg.ResourceNotFound:
            hint = ('~/catkin_ws underlay가 source되지 않았을 가능성',) + SOURCE_HINT
            report.add(Check(package, 'FAIL', 'Package not found', hint if package == 'morai_msgs' else SOURCE_HINT))
    master = rosgraph.Master('/field_check')
    try:
        bounded_call(master.getPid, 2)
        report.add(Check('ROS Master', 'PASS', master.master_uri))
    except Exception as error:
        report.add(Check('ROS Master', 'FAIL', str(error), ('Start roscore or roslaunch; check ROS_MASTER_URI.',)))
        return
    rospy.init_node('field_check', disable_signals=True, disable_rosout=True)
    get = lambda name, default: rospy.get_param('~' + name, default)
    address = str(get('morai_ip', ''))
    field_mode = get('field_mode', True)
    window = float(get('window', 5))
    timeout = float(get('stamp_timeout', 1))
    service_timeout = float(get('service_timeout', 2))
    if not all(math.isfinite(value) and 0 < value <= 60 for value in (window, timeout, service_timeout)):
        raise ValueError('window/stamp_timeout/service_timeout must be finite and within (0,60] seconds')
    port, bridge_port = int(get('morai_port', 9093)), int(get('rosbridge_port', 9090))
    rear_required, lidar_required = get('require_rear_camera', False), get('require_lidar', True)
    sensor_config = get('sensor_config', str(Path(paths.get('sensor_bridge', '')) / 'config/sensors.yaml'))
    with open(sensor_config) as file:
        sensors = yaml.safe_load(file)['sensors']
    sensors['lidar3d']['destination_port'] = int(get('lidar_port', 9210))
    sensors['lidar3d']['topic'] = get('lidar_topic', '/lidar3D')
    source_ip = network(report, address, field_mode)
    report.line('-' * 60)
    report.line('MORAI Windows Client: {} | Cmd Control UDP receive: {}'.format(address, port))
    display_ip = source_ip or '<Ubuntu Ethernet IPv4: inspect ip -4 addr>'
    report.line('Ubuntu Algorithm PC: {} | ROS Bridge TCP: {}'.format(display_ip, bridge_port))
    report.line('Sensor Destination settings (entered on the Windows SIM):')
    for name, sensor in sensors.items():
        report.line('  {:16} {}:{} -> {}'.format(name, display_ip, sensor['destination_port'], sensor['topic']))
    report.line('Sensor Host ports stay on Windows; ROS receivers bind Destination ports.')
    report.line('-' * 60)
    try:
        if not 1 <= bridge_port <= 65535:
            raise ValueError('ROS Bridge port must be 1..65535')
        # TCP connect verifies a listener, not an authenticated Windows link.
        with socket.create_connection(('127.0.0.1', bridge_port), timeout=1):
            pass
        report.add(Check('ROS Bridge listener', 'PASS', 'Local TCP {}; client link checked by Event Service'.format(bridge_port)))
    except (OSError, ValueError) as error:
        report.add(Check('ROS Bridge listener', 'FAIL', str(error),
                         ('roslaunch rosbridge_server rosbridge_websocket.launch port:=' + str(bridge_port),)))
    topics = dict(bounded_call(lambda: master.getPublishedTopics('/'), 2))
    system_state = bounded_call(master.getSystemState, 2)
    nodes = {node for group in system_state for topic, owners in group for node in owners}
    control_check(report, rospy, paths, topics, nodes, address, port)
    service_check(report, rospy, get('mode_service', '/Service_MoraiEventCmd'), service_timeout)
    from sensor_msgs.msg import Image, Imu, PointCloud2
    from morai_msgs.msg import GPSMessage
    types = {'camera': Image, 'velodyne': PointCloud2, 'imu': Imu, 'gps': GPSMessage}
    names = {'camera_front': 'Camera Front', 'camera_left': 'Camera Left', 'camera_right': 'Camera Right',
             'camera_rear': 'Camera Rear (optional)' if not rear_required else 'Camera Rear',
             'lidar3d': 'LiDAR' if lidar_required else 'LiDAR (optional)', 'gps': 'GPS UDP -> ROS', 'imu': 'IMU'}
    stats, subscribers = {}, []
    lock = threading.Lock()
    bad_types = set()

    def subscribe(key, name, kind, topic, cls, required):
        stats[key] = SensorStats(name, kind, required)
        if topic in topics and topics[topic] != cls._type:
            bad_types.add(key)
            report.add(Check(name, 'FAIL' if required else 'WARN',
                             '{}: expected {}, got {}'.format(topic, cls._type, topics[topic])))
            return

        def callback(message):
            with lock:
                stats[key].observe(message, time.monotonic())

        subscribers.append(rospy.Subscriber(topic, cls, callback, queue_size=1, buff_size=16 * 1024 * 1024))

    for key in names:
        config = sensors[key]
        required = rear_required if key == 'camera_rear' else lidar_required if key == 'lidar3d' else True
        subscribe(key, names[key], config['kind'], config['topic'], types[config['kind']], required)
    monitor = get('monitor_control', False)
    if monitor:
        from morai_msgs.msg import CtrlCmd
        from std_msgs.msg import String, UInt8
        for key, kind, topic, cls in (('ctrl', 'control', '/ctrl_cmd', CtrlCmd),
                                     ('mode', 'mode', '/control/sim_mode', UInt8),
                                     ('status', 'udp_status', '/control/udp_status', String)):
            subscribe(key, key, kind, topic, cls, True)
    report.line('Measuring all topics together for {:.1f}s (wall time)...'.format(window))
    end = time.monotonic() + window
    while not rospy.is_shutdown() and time.monotonic() < end:
        time.sleep(min(.05, max(0, end - time.monotonic())))
    if rospy.is_shutdown():
        report.add(Check('Measurement', 'FAIL', 'ROS shutdown interrupted the measurement'))
    for subscriber in subscribers:
        subscriber.unregister()
    now = time.monotonic()
    ros_now = rospy.Time.now().to_sec()
    if rospy.get_param('/use_sim_time', False) and ros_now == 0:
        report.add(Check('ROS clock', 'FAIL', 'use_sim_time=true but /clock has not started'))
    try:
        listeners = udp_listeners()
        for key in names:
            probe = stats[key]
            expected = int(sensors[key]['destination_port'])
            if expected not in listeners:
                report.add(Check(probe.name + ' UDP port', 'FAIL' if probe.required else 'WARN',
                                 'No local UDP listener on ' + str(expected),
                                 ('Start sensor_bridge or check Destination port; do not bind Host ports.',)))
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report.add(Check('UDP listeners', 'WARN', str(error)))
    with lock:
        for key, probe in stats.items():
            if key in bad_types:
                continue
            config = sensors.get(key)
            causes = ('Sensor Connect OFF or incorrect Destination IP.',
                      'Destination Port must be ' + str(config['destination_port']),
                      'Firewall or sensor_bridge/Velodyne not running.',
                      'Check sensor clock / ROS time if timestamps are stale or frozen.') if config else (
                          'Start driving separately only when ready; inspect /control/udp_status.',)
            report.add(probe.result(now, ros_now, timeout, causes))
        gps = stats['gps'].last
        if gps is None:
            report.add(Check('GPS Map Offset', 'WARN', 'No GPS sample to inspect; transport and map offsets are separate'))
        else:
            east, north = gps.eastOffset, gps.northOffset
            expected = (float(get('east_offset', 0)), float(get('north_offset', 0)))
            match = all(math.isfinite(v) for v in (east, north)) and (east, north) != (0, 0) and (east, north) == expected
            report.add(Check('GPS Map Offset', 'PASS' if match else 'WARN',
                             'east={}, north={}; expected={}; {}'.format(
                                 east, north, expected, 'matches configuration; confirm actual map values' if match else
                                 'zero/non-finite/mismatched offsets; confirm actual map before driving'),
                             () if match else ('GPS UDP does not carry map offsets. Set east_offset and north_offset.',)))
        if monitor and stats['status'].last is not None:
            status = stats['status'].last.data
            if status.startswith('BRAKE'):
                report.add(Check('Control watchdog', 'WARN', status))
            report.line('SENDING only confirms local UDP send; verify SIM receipt/gear/vehicle behavior separately.')


def main():
    report = Report()
    try:
        run(report)
    except KeyboardInterrupt:
        report.add(Check('Diagnostic', 'FAIL', 'Interrupted before completion'))
    except Exception as error:
        report.add(Check('Diagnostic', 'FAIL', '{}: {}'.format(type(error).__name__, error)))
    return report.finish()


if __name__ == '__main__':
    sys.exit(main())
