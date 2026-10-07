# LiDAR iterative odometry and obstacle avoidance

This GPS blackout stack is available on `local-state-2026-10-05` through
`signal_tracking_avoidance.launch`. It uses the same Pure Pursuit steering
and pedal PID for ordinary waypoint driving and LiDAR avoidance.

## Data flow

1. `/competition/ego_speed_kmh` and `/competition/heading_imu` produce
   `/leo/ego_odom` by relative integration. These inputs come from Ego UDP
   status and do not require GPS position.
2. `/lidar3D` scans are matched iteratively with a planar ICP algorithm. Ego
   odometry supplies the initial motion estimate. Fresh GPS map pose anchors
   this relative trajectory before signal loss.
3. The single `/leo_driving_state` node switches `/leo/driving_mode` from
   `GPS` to `BLACKOUT` when a fresh `/gps` message has **both** `latitude` and
   `longitude` equal to `0.0`, or when an anchored, previously valid GPS
   stream stops for `gps_silence_timeout_s` (default 0.75 s). ICP, odometry,
   Ego speed, the anchored pose and a safe LiDAR gap must remain fresh. Other
   GPS faults, failed matching, blocked gaps and excessive travel switch to
   `HOLD`.
4. In that same state machine, the gap planner uses the existing waypoint route as its
   preferred heading, then chooses a LiDAR gap wide enough for the car. It
   outputs `/leo/avoidance_target` in
   `base_link` and `/leo/avoidance_speed_limit_kmh`. The existing waypoint
   follower feeds that target into its **same** Pure Pursuit steering function
   and applies the cap through its **same** pedal PID. The Leo signal guard
   receives the switched map pose and remains the single command gate.
   This is reactive gap selection. It does not track an individual moving car
   or run a lane-change and return maneuver.
   Each scan scores safe gaps against a lookahead point on the original
   waypoint path. After passing an obstacle, the preferred bearing therefore
   points back toward that path. Route association remains available up to
   4 m from its centerline during the tunnel pass; a wider deviation, an
   unsafe gap, or stale localization commands a stop. GPS rejoin still checks
   position and heading against the original route.
5. The alternative UDP node remains the only UDP owner in this launch. It
   sends the same guarded `/ctrl_cmd` during GPS and blackout. GPS loss is
   allowed through only while LiDAR ICP, Ego odometry, avoidance, route guard
   and travel-distance limits agree. Missing evidence sends full brake.

```bash
source /opt/ros/noetic/setup.bash
source /home/unita/catkin_ws/devel/setup.bash
source /home/unita/unita_ws/devel/setup.bash
roslaunch leo_avoidance signal_tracking_avoidance.launch morai_ip:=<SIM_IP>
```

This optional launch caps ordinary GPS route driving at **40 km/h**. Its
route speed profile slows to **18 km/h** from waypoint 3730 through 4100
while GPS is available. During blackout, the LiDAR gap target can reach
**30 km/h** only when the observed forward range, wall clearance, route
curvature and steering allow it. A near wall or obstacle reduces the limit.
The blackout avoidance scan covers **-135° to +135°** around the vehicle's
forward axis. Rear-quarter returns within that scan cannot establish forward
clearance; steering remains limited by `max_target_bearing_rad`.
The transport gate checks the 18 km/h entry speed at the
blackout transition. Its 160 m travel allowance starts when GPS is lost,
and the next controlled stop line still limits that allowance. A GPS pose
return within 1.5 s switches the state machine back to ordinary GPS driving;
LiDAR blackout mode is restricted to route indices 3780 through 4100.

Steering-based speed reduction is continuous to avoid step changes in the
target speed. At an open tunnel exit with no forward-sector LiDAR hits, the
planner may continue at at most 2.0 m/s toward the route only when at least
10 other finite forward-half-plane returns remain. Sparse or empty
scans still stop. LiDAR ICP, route, UDP and signal guard checks remain active;
if GPS does not return before the blackout travel limit, the car stops.
ICP scan matching is checked against Ego odometry, while the Ego heading
remains the heading reference to avoid repeated wall-induced yaw drift.
`/leo/driving_status` reports `sim_pose_error_left_m` and
`sim_pose_error_yaw_rad` against MORAI's Ego pose for diagnosis only.

The original `signal_tracking.launch` must not run at the same time because
both launches use the Ego UDP control port. The Leo signal guard still needs
its usual `set_armed` service before driving. The blackout transition itself
does not need an additional service.

## RViz: FTG obstacle candidates

While `signal_tracking_avoidance.launch` is running, start the separate
visualization with:

```bash
source /home/unita/unita_ws/devel/setup.bash
roslaunch leo_avoidance ftg_visualization.launch
```

The RViz fixed frame is `lidar`. Cyan points on `/leo/ftg_obstacle_points`
are the returns FTG keeps after ground removal and the ±135°/range filter.
Magenta points on `/leo/ftg_rear_obstacle_points` are the separate rear
±15° returns used to confirm a completed pass; they do not affect FTG's
forward collision gap. They also pass the close-range, lateral and height
filters described below.
`/leo/ftg_obstacle_boxes` groups compact returns into translucent boxes and
shows long narrow candidate groups as lines; red groups cross the forward
vehicle corridor and yellow groups lie to its sides. Each label shows the
group's point count and nearest LiDAR distance.
`/leo/ftg_planning` shows actual FTG gap candidates as thin blue direction
lines and the selected steering target as a green arrow. A red label reports
when there is no valid target. These are short steering directions, not
predicted multi-step vehicle trajectories.
Single points are shown because FTG also reacts to single returns. Walls or
road patches appear if FTG currently treats them as obstacle returns. The
visualization publishes no control commands and does not change FTG planning.
FTG now accepts shorter observed ground patches when their fitted plane is
near the configured LiDAR ground height (`expected_ground_z_m`); broad low
objects above that plane remain obstacle returns. Tunnel walls remain FTG
collision boundaries and can still make a pass impossible.
For a central obstacle, FTG selects a gap around the nearest obstacle rather
than requiring one straight LiDAR ray to remain clear all the way to the far
tunnel wall. The full scan still limits speed. The target bearing may reach
35 degrees at low speed. Each obstacle return excludes the vehicle half width
plus 0.4 m of side clearance. A front obstacle within the current stopping
distance commands braking while keeping a passable steering target.
Only returns within the configurable vehicle collision height
(`collision_min_z_m` through `collision_max_z_m`) are used by FTG. This omits
tunnel ceilings while retaining low boxes near the road. Coherent side-wall
returns are fitted as left and right track boundaries, not obstacle bubbles.
If a wall fit is unreliable its returns remain obstacle candidates. RViz
shows fitted boundaries as green lines and boxes only for remaining obstacle
points. Candidate targets must remain inside the fitted boundaries with the
vehicle width and side margin included.
During GPS blackout, FTG targets 2.9 m of right wall clearance from the
vehicle side, with no offset correction inside the 2.8-3.0 m band.
A forward obstacle within 20 m starts `PASSING`; the wall
spacing correction stays off until the raw LiDAR cloud detects at least three
returns in the near rear sector (180 degrees +/-15 degrees) on two consecutive
scans, after the front corridor is clear. Rear detection uses a 12 m range
and a 2.8 m lateral limit to avoid distant tunnel-wall returns. This rear
sector is separate from the forward FTG scan (-135 to +135 degrees). FTG
still checks LiDAR gaps and wall collision clearance during the pass. A
return search after the front corridor clears considers +/-50 degrees and
plans against the nearest 20 m of LiDAR returns, while the full scan remains
available for clearance checks; return speed is capped at 2 m/s. A
reliable wall resumes leading steering after the pass. FTG handles distant
tunnel returns conservatively: when a far return
fills every FTG gap but a right wall is measured and current-speed stopping
distance remains available, the car slows to at most 2 m/s while following
that wall. A close blockage or insufficient stopping distance still stops.
The wall-to-route handoff starts 15 m before the photographed exit, using the simulator's
ENU pose to time that handoff. Within 35 m of the exit, a fresh simulator pose
may correct accumulated LiDAR map drift by at most 1 m per valid scan only if
it is within 1.9 m of the surveyed route and aligned within 0.35 rad; speed
is then capped at 18 km/h.
LiDAR matching, FTG and the obstacle checks remain required. This is a
MORAI-specific recovery and does not provide real-world GPS-free localization.
If that pose is unavailable, wall guidance stays active. A route bearing
more than 15 degrees from the measured wall
direction retains at least 90% wall guidance; without a visible wall and
before GPS recovery, route bearing is limited to 7 degrees. The speed cap
decreases near the exit. Near the exit the waypoint target is
shifted about 0.41 m toward the photographed ENU coordinate
`(-78.35, -545.63)`. The exit handoff uses that coordinate and its route
tangent; LiDAR odometry error can still affect the real position reached.
LiDAR ICP validates scans and corrects lateral motion, while Ego speed and heading
provide forward distance because longitudinal ICP on repetitive walls drifts.
`/leo/avoidance_status` reports `right_wall_clearance_m`,
`right_wall_target_clearance_m`, `right_wall_heading_rad`,
`avoidance_phase`, `rear_scan_points`, `rear_obstacle_points`, and
`rear_confirm_scans`. If the wall
cannot be identified across several forward strips, the values are `null`
and FTG uses its route and obstacle gap steering without wall correction.

## Tunnel GPS blackout scenario

`scenarios/UNITA_tunnel_gps_blackout.json` copies the currently used
`idiot1.json` scenario and adds one GPS-only Denied Area. The same file is in
the local MORAI `SaveFile/Scenario/R_KR_PR_K-city_2025` directory. In MORAI,
open **Edit → Scenario → Load Scenario** and choose
`UNITA_tunnel_gps_blackout`; confirm GPS is checked and `Noise Type` is
`Blackout`, while LiDAR and IMU remain unchecked.

The matching `scenarios/EgoNetwork/UNITA_tunnel_gps_blackout_MN.json` is also
installed under MORAI's `Scenario/.../EgoNetwork` directory. It restores
`MoraiInfoPublisher` UDP to `127.0.0.1:9096` and Cmd Control to port 9093.
When loading the scenario, enable the option to load network connection data,
then verify **Ego Network → Publisher → Ego Vehicle Status** is connected.
MORAI may disconnect Ego Network during scenario reload; reconnect it and
resume the paused simulation before arming the ROS guard.

For a repeatable moving-vehicle test, load
`UNITA_tunnel_gps_blackout_periodic_traffic` instead. It keeps the same ego
start and GPS Denied Area, removes the previously saved NPC, and spawns one
NPC on the tunnel route every 15 seconds (at most four active). The spawn
point is MGeo link `A2256W000126` point 0, near the tunnel entrance; its
destination is link `A2256W000128` point 16, just beyond the exit. NPCs
start at 10 km/h and target 12 km/h in the lane center. Its matching
`EgoNetwork/UNITA_tunnel_gps_blackout_periodic_traffic_MN.json` keeps the
same UDP settings. In the MORAI scenario editor, confirm the red path reaches
the exit and spawned cars actually drive through the tunnel. This scenario
tests moving obstacles; the current gap planner does not implement a tracked
overtake and return maneuver.

The area covers route waypoints 3803–4050 (about 122 m) with a valid-GPS
approach before its entrance. Its size follows the screenshot (129.43 m long,
8.84 m wide, 14.84 m high); the saved sample Denied Area supplies the
scenario-coordinate yaw. This placement is checked against the waypoint
geometry, but the simulator must confirm the actual tunnel entrance and exit.
The test launch now permits up to 160 m without GPS, leaving 38 m beyond the
122 m route span for the actual sensor boundary and approach. This is a test
setting: compare ICP position to GPS when it returns and measure drift before
using a longer or faster blackout. The signal guard keeps its brake applied
during a pose handoff of at most 2 s, then disarms if the pose stays unavailable.
The next **controlled** stop line still bounds the blackout distance. The
surveyed uncontrolled point at the tunnel entrance does not block entry.

## Limits and simulator checks

- Measure the `lidar_to_rear_x_m` mount offset, LiDAR ICP height crop,
  road width, actual brake response and suitable low-speed entry profile.
  Values in `config/avoidance.yaml`, `config/odometry.yaml` and
  `config/blackout.yaml` are provisional.
- In open road, compare `/leo/driving_pose` with `/localization/pose` while
  GPS is valid. In MORAI, add a `Denied Area` around the tunnel and select
  GPS `Blackout` only; leave LiDAR available. MORAI documents 0/0 output in
  this mode. Check `/leo/driving_status` changes to `BLACKOUT` as the vehicle
  enters. Alternatively, stop GPS UDP delivery after the initial GPS anchor;
  the missing stream triggers after 0.75 s. Verify a nonzero invalid fix
  produces `HOLD`.
  Compare integrated distance
  with a known map landmark. No jump should occur at the transition.
- Put one obstacle ahead. Check `/leo/avoidance_target` points into a gap
  wide enough for the vehicle and the follower's target speed is capped.
  An occupied or narrow gap must command a stop.
- Freeze LiDAR, Ego heading or odometry separately. Each must produce `HOLD`
  or a UDP brake within the configured timeout. Travel beyond 160 m without
  GPS must brake. If GPS returns far from the ICP pose, the state machine
  latches `HOLD`; restart after reviewing the mismatch.
- A failed ICP match brakes for that scan and moves the scan reference using
  Ego odometry so the next scan can match locally. More than 3 m of unmatched
  motion prevents recovery until a new GPS anchor is available. The map pose
  stops advancing after this limit. A delayed Ego heading sample does not
  reset the odometry frame; gaps up to 0.6 s are integrated from Ego speed,
  while longer gaps withhold odometry until samples resume. LiDAR scans use
  the closest timestamped odometry/GPS sample and the nearest Ego speed
  sample, so callback order alone does not invalidate a fresh scan.
- Check `next_controlled_stop_distance_m` in `/leo/signal_status` before the
  blackout. The UDP gate brakes when the next controlled stop line is within
  the configured 10 m margin. Confirm the tunnel entrance is reported as a
  reviewed uncontrolled crossing.
- Check the 40 km/h GPS route and 18 km/h entry profile in MORAI with fresh
  LiDAR and Ego status.
  The UDP gate requires at most 5.1 m/s at the blackout transition; a faster
  entry brakes until the car slows. Sensor or ICP loss still commands brake.

Real simulator validation is still required. Recovery after an obstacle
avoidance failure is deliberately deferred.
