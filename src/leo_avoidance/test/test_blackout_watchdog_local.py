from leo_avoidance.blackout_watchdog_local import BlackoutWatchdog


STOP = b'brake'
SHARED = b'guarded_pp_pid'


def gate():
    g = BlackoutWatchdog(fallback_timeout=.35, max_blackout_distance_m=20.)
    g.set_command(SHARED, 10.)
    g.set_sensor('gps', True, 10., 10.)
    g.set_raw_gps(37., 127., 1, 10., 10.)
    g.set_sensor('imu', True, 10., 10.)
    g.set_lidar(True, 10., 10.)
    g.set_odom(True, 10., 10.)
    g.set_speed(1., 10.)
    g.set_drive_mode('GPS', 10.)
    g.set_drive_valid(True, 10.)
    g.set_avoidance_valid(True, 10.)
    g.set_guard(True, 50., 10.)
    assert g.select(10., 10., STOP)[0] == SHARED
    assert g.armed
    return g


def blackout(g):
    g.set_sensor('gps', False, 10.1, 10.1)
    g.set_raw_gps(0., 0., 0, 10.1, 10.1)
    g.set_drive_mode('BLACKOUT', 10.1)
    g.set_drive_valid(True, 10.1)
    g.set_lidar(True, 10.1, 10.1)
    g.set_odom(True, 10.1, 10.1)
    g.set_speed(1., 10.1)
    g.set_avoidance_valid(True, 10.1)
    g.set_guard(True, 49., 10.1)


def test_same_command_in_gps_and_blackout():
    g = gate()
    blackout(g)
    assert g.select(10.15, 10.15, STOP)[0] == SHARED


def test_simulator_pose_mode_is_not_a_blackout_source():
    g = gate()
    blackout(g)
    g.set_drive_mode('SIM_EXIT', 10.1)
    assert g.select(10.15, 10.15, STOP)[0] == STOP


def test_no_gap_brakes():
    g = gate()
    blackout(g)
    g.set_avoidance_valid(False, 10.1)
    assert g.select(10.15, 10.15, STOP)[0] == STOP


def test_blackout_requires_fresh_zero_gps_trigger():
    g = gate()
    blackout(g)
    g.set_raw_gps(37., 127., 1, 10.1, 10.1)
    assert g.select(10.15, 10.15, STOP)[0] == STOP
    g.set_raw_gps(0., 0., 0, 9., 10.1)
    assert g.select(10.15, 10.15, STOP)[0] == STOP


def test_missing_gps_stream_uses_same_fallback_gate():
    g = gate()
    assert not g._gps_blackout_trigger(10.5, 10.5)
    g.set_command(SHARED, 10.8)
    g.set_sensor('imu', True, 10.8, 10.8)
    g.set_lidar(True, 10.8, 10.8)
    g.set_odom(True, 10.8, 10.8)
    g.set_speed(1., 10.8)
    g.set_drive_mode('BLACKOUT', 10.8)
    g.set_drive_valid(True, 10.8)
    g.set_avoidance_valid(True, 10.8)
    g.set_guard(True, 49., 10.8)
    packet, reason = g.select(10.8, 10.8, STOP)
    assert packet == SHARED and 'LiDAR odometry' in reason


def test_invalid_nonzero_fix_does_not_trigger_blackout():
    g = gate()
    g.set_raw_gps(37., 127., 0, 10.1, 10.1)
    assert not g._gps_blackout_trigger(11., 11.)
    g.set_raw_gps(90., 127., 1, 10.1, 10.1)
    assert not g._gps_blackout_trigger(11., 11.)


def test_no_odom_brakes():
    g = gate()
    blackout(g)
    g.set_odom(False, 10.1, 10.1)
    assert g.select(10.15, 10.15, STOP)[0] == STOP


def test_blackout_travel_budget_brakes():
    g = gate()
    g.distance_budget_m = .1
    blackout(g)
    assert g.select(10.15, 10.15, STOP)[0] == STOP
    assert not g.armed


def test_gps_approach_does_not_consume_blackout_budget():
    g = gate()
    g.max_blackout_distance_m = 100.
    for step in range(1, 31):
        now = 10. + step * .1
        g.set_command(SHARED, now)
        g.set_sensor('gps', True, now, now)
        g.set_sensor('imu', True, now, now)
        g.set_raw_gps(37., 127., 1, now, now)
        g.set_lidar(True, now, now)
        g.set_odom(True, now, now)
        g.set_speed(4., now)
        g.set_drive_mode('GPS', now)
        g.set_guard(True, 50. - 4. * (now - 10.), now)
        assert g.select(now, now, STOP)[0] == SHARED
    assert g.armed
    assert g.travelled_m == 0.
    assert abs(g.distance_budget_m - 28.) < 1e-8
    g.set_command(SHARED, 13.1)
    g.set_sensor('gps', False, 13.1, 13.1)
    g.set_sensor('imu', True, 13.1, 13.1)
    g.set_raw_gps(0., 0., 0, 13.1, 13.1)
    g.set_lidar(True, 13.1, 13.1)
    g.set_odom(True, 13.1, 13.1)
    g.set_speed(1., 13.1)
    g.set_drive_mode('BLACKOUT', 13.1)
    g.set_drive_valid(True, 13.1)
    g.set_avoidance_valid(True, 13.1)
    g.set_guard(True, 38., 13.1)
    assert g.select(13.1, 13.1, STOP)[0] == SHARED
    assert 0. < g.travelled_m < .2


def test_fast_blackout_keeps_shared_command():
    g = gate()
    blackout(g)
    assert g.select(10.15, 10.15, STOP)[0] == SHARED
    g.set_command(SHARED, 10.2)
    g.set_lidar(True, 10.2, 10.2)
    g.set_odom(True, 10.2, 10.2)
    g.set_speed(11.1, 10.2)
    g.set_drive_mode('BLACKOUT', 10.2)
    g.set_drive_valid(True, 10.2)
    g.set_avoidance_valid(True, 10.2)
    g.set_guard(True, 48., 10.2)
    assert g.select(10.21, 10.21, STOP)[0] == SHARED


def test_missing_guard_brakes():
    g = gate()
    blackout(g)
    g.set_guard(False, 49., 10.1)
    assert g.select(10.15, 10.15, STOP)[0] == STOP


def test_normal_gps_keeps_original_behavior_at_high_speed():
    g = gate()
    g.set_speed(15., 10.1)
    assert g.select(10.15, 10.15, STOP)[0] == SHARED


def test_blackout_entry_speed_follows_route_profile():
    # The project file owns the approach speed; entry does not brake for it.
    g = gate()
    blackout(g)
    g.set_speed(15., 10.1)
    assert g.select(10.15, 10.15, STOP)[0] == SHARED


def test_tunnel_budget_survives_uncontrolled_crossing_and_guard_handoff():
    g = BlackoutWatchdog(fallback_timeout=.35,
                         max_blackout_distance_m=160.)
    g.set_command(SHARED,10.)
    g.set_sensor('gps',True,10.,10.)
    g.set_sensor('imu',True,10.,10.)
    g.set_raw_gps(37.,127.,1,10.,10.)
    g.set_lidar(True,10.,10.)
    g.set_odom(True,10.,10.)
    g.set_speed(2.8,10.)
    g.set_drive_mode('GPS',10.)
    g.set_guard(True,190.,10.)  # Next controlled light, not tunnel entrance.
    assert g.select(10.,10.,STOP)[0]==SHARED
    assert g.distance_budget_m==160.
    blackout(g)
    g.set_guard(False,float('nan'),10.1)  # Safe brake during a pose handoff.
    assert g.select(10.15,10.15,STOP)[0]==STOP
    assert g.armed
    g.set_guard(True,180.,10.2)
    g.set_speed(1.,10.2)
    g.set_lidar(True,10.2,10.2)
    g.set_odom(True,10.2,10.2)
    g.set_drive_mode('BLACKOUT',10.2)
    g.set_drive_valid(True,10.2)
    g.set_avoidance_valid(True,10.2)
    g.set_command(SHARED,10.2)
    assert g.select(10.21,10.21,STOP)[0]==SHARED


def test_gps_blip_during_blackout_keeps_fallback_armed():
    g = gate()
    blackout(g)
    assert g.select(10.15, 10.15, STOP)[0] == SHARED
    # One valid fix inside the tunnel; the driving state stays BLACKOUT.
    g.set_command(SHARED, 10.2)
    g.set_sensor('gps', True, 10.2, 10.2)
    g.set_raw_gps(37., 127., 1, 10.2, 10.2)
    g.set_drive_mode('BLACKOUT', 10.2)
    g.set_drive_valid(True, 10.2)
    g.set_lidar(True, 10.2, 10.2)
    g.set_odom(True, 10.2, 10.2)
    g.set_speed(1., 10.2)
    g.set_avoidance_valid(True, 10.2)
    g.set_guard(True, 48., 10.2)
    packet, reason = g.select(10.25, 10.25, STOP)
    assert packet == SHARED and 'LiDAR odometry' in reason and g.armed
    # The blackout gates still apply during the blip.
    g.set_avoidance_valid(False, 10.26)
    assert g.select(10.27, 10.27, STOP)[0] == STOP
    g.set_avoidance_valid(True, 10.28)
    # GPS returns to zero: the fallback is still armed and keeps sending.
    g.set_command(SHARED, 10.3)
    g.set_sensor('gps', False, 10.3, 10.3)
    g.set_raw_gps(0., 0., 0, 10.3, 10.3)
    g.set_drive_mode('BLACKOUT', 10.3)
    g.set_drive_valid(True, 10.3)
    g.set_lidar(True, 10.3, 10.3)
    g.set_odom(True, 10.3, 10.3)
    g.set_speed(1., 10.3)
    g.set_guard(True, 47., 10.3)
    assert g.select(10.35, 10.35, STOP)[0] == SHARED


def test_gps_return_accepted_by_driving_state_rearms_from_gps():
    g = gate()
    blackout(g)
    assert g.select(10.15, 10.15, STOP)[0] == SHARED
    g.set_command(SHARED, 10.2)
    g.set_sensor('gps', True, 10.2, 10.2)
    g.set_raw_gps(37., 127., 1, 10.2, 10.2)
    g.set_drive_mode('GPS', 10.2)
    g.set_speed(1., 10.2)
    g.set_guard(True, 48., 10.2)
    packet, reason = g.select(10.25, 10.25, STOP)
    assert packet == SHARED and reason.startswith('SENDING: fresh command')
    assert not g.blackout_started
