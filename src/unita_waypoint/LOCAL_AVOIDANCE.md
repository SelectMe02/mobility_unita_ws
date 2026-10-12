# Map route obstacle avoidance

`competition_udp_tracking.launch` starts `lidar_obstacles.py` with the follower
when `enable_lidar:=true`. `test_map_raceline.launch` enables local avoidance
and prioritizes Raceline 3 in Sector 11 through checkpoint 11 (inclusive),
when a safe transition and saved coverage exist. After checkpoint 11 it returns
to Raceline 1 as soon as the merge gap is safe. The boundary is the saved JSON
Sector 11 endpoint (CP11 Y=-217.855053961277), shared with Map Tuner/Debug.
The speed profile's legacy CP11 at Y=+89.1979 is not used for lane priority.
Crossing requires the entire vehicle footprint to have Y strictly below the
finish line; heading, rear overhang and half-width determine the trailing
corner Y. Equality retains R3 priority, including when the nearest route
sample has already entered S12. Manual R3 extension points are interpolated
onto route progress within their saved Y coverage. At the saved R3/R1 join,
the path follows the existing R1 continuation while the rear clears the line.
Short/missing R3 geometry causes slowing/holding rather than early R1 priority.
Tracking state includes finish-line Y, trailing Y, crossing status, `waypoint_index` and
`preferred_raceline_reason` (`priority_interval`, `default_lane`,
`saved_lane_unavailable`, or `saved_lane_ending`) to distinguish the triggers.
Its ten-second lane sequence has been removed.
The generic competition launch defaults to Raceline 1 and also accepts
`preferred_raceline`. Disabling local avoidance follows the global Raceline 1.

The global `/waypoint_path` stays the saved Raceline 1 route, providing route
progress and point speed limits. `/waypoint_debug/local_path` carries the path
actually followed while driving in another lane, changing lanes, or returning.
With a blocked R3, the planner tries the nearest free saved passing lane
(usually R2). It maintains that lane until R3 has a clear forward corridor and
safe rear clearance; only then does it return to R3 while inside the priority
interval. After checkpoint 11 the recovery target is R1. If no passing gap exists,
it follows the leader in its current lane with a stopping-distance speed cap.
The merge end is anchored to route progress, so replanning does not keep moving
it forward. A committed transition is checked continuously; a newly predicted
collision stops the car rather than selecting a different target mid-merge.

## Lane-only obstacle detection

`raceline_width` is the full lane width in meters. With width 3.2, the recognized
road is the union of strips within +/-1.6 m of every mapped Raceline 1/2/3.
Distances are measured perpendicularly to waypoint segments, so sparse points
and rotated roads work too. Separate saved adjacent-lane intervals are not
connected or extended. Global R1 follows the same continuous loop as the follower.

The follower publishes the actual centerlines and resolved width on the latched
`/waypoint_debug/lane_geometry` topic. Perception and map_debug consume that same
geometry for obstacle filtering. The follower reports the resolved width as
`effective_raceline_width`, separate from the input `raceline_width` parameter.
The saved Map Tuner project owns the width, including when an old launch or ROS
width override remains. The YAML fallback in `config/raceline_layout.yaml` is
used only when the project has no saved width.

```bash
roslaunch unita_launch test_map_raceline.launch morai_ip:=127.0.0.1
roslaunch unita_waypoint map_debug.launch
```

After changing width in Map Tuner, save the JSON. The running follower checks
the save every 0.5 s and republishes its lane filter with the saved width;
perception consumes that geometry immediately. Map Debug reloads its saved
project within 1 s. Width synchronization does not regenerate or replace a
route that is being driven. Centerline edits require restarting the driving
launch to load the new route. Map Debug draws the saved JSON paths.

Roadside LiDAR points are dropped before clustering. Out-of-lane object centers
and predicted tracks are discarded before publishing obstacles, and the follower
and debug also apply the corridor filter. An out-of-lane detection never becomes
a passing threat or debug box merely because its collision radius is large.
A briefly occluded track is retained only while its predicted center stays in a
mapped lane. Missing/invalid lane geometry holds perception until a valid map
arrives rather than recognizing unrestricted surroundings.

The removed experimental 10 cm cutoff and `ground_ignore_height_m` setting
remain absent. The replacement uses timestamp-matched **raw `/imu` orientation**
and `/lidar3D` only, plus the existing GPS/IMU-derived map pose for tracking.
It does not read full Ego Vehicle Status, true simulator coordinates, other
vehicle states, or tire/acceleration fields from vehicle status. These inputs
fit the allowed IMU, 3D LiDAR and GPS networks in the supplied competition rules;
this statement covers this filter's data path, not an audit of the entire vehicle.
**Diagnostic simulator truth and the remaining launch compliance issue:**
`decode_ego_status` now reads simulator position at byte 77 only when
`diagnostics=True`, selected by the launch's `log:=true` option. Raw status
packets and parsed truth are published on `/diagnostics/simulator/*` only in
this mode, for offline analysis. Normal `/competition/ego_status` omits position,
and driving/perception never subscribe to the diagnostic truth topics.
The receiver still accepts full `#MoraiInfo$` packets. Selecting
`heading_source:=imu` does not remove this network input. Running the whole
current launch cannot be certified compliant until that independent receiver
and the simulator network settings are migrated to the organizer's actual
Competition Vehicle Status protocol.
Do not guess byte offsets for the restricted packet; obtain its specification
or a representative packet first.

## IMU posture compensation and local road separation

The detector normalizes quaternion orientation, interpolates fresh bracketing
samples using SLERP, or uses the nearest fresh sample. Both interpolation
endpoints must meet `imu_max_skew_s` (default 0.05 s). Invalid orientation,
unsupported frame, orientation covariance -1, and old samples are rejected.
Clock reversal clears the IMU buffer. Timestamp differences describe header
alignment, not guaranteed sensor latency: the current UDP bridge defaults to
receipt timestamps, so a real braking recording must verify that alignment.

Fixed sensor mounting rotations (`imu_mount_*_deg`, `lidar_mount_*_deg`)
use sensor-to-body Rz(yaw)*Ry(pitch)*Rx(roll). Translation is applied to the
LiDAR points before posture rotation, using the existing rear-axle mount
(1.4, 0, 1.23 m). IMU translation is not used because this filter reads only
orientation. The leveling rotation removes roll/pitch while retaining vehicle
heading; GPS/IMU map yaw is applied once, when producing map coordinates.

Leveled points form 8 m longitudinal by 6 m lateral patches. RANSAC fits
spatially distributed low returns with at least 12 support cells spanning
2 m longitudinally and 1 m laterally. Road slope is limited to 20 degrees.
Near-ego planes must intersect `ground_reference_z_m` within +/-0.18 m.
This launch starts at -0.38 m, measured in the supplied braking recording;
the generic detector default remains 0 m. Mounting extrinsics match the
supplied MORAI settings and are not shifted. Other patches
require adjacent confirmed road, with <=0.12 m height difference at both
ends of their common edge and <=10 degrees of slope-vector difference.
Planes are never extrapolated over unseen patches. These bounds prevent a
disconnected roof or vertical wall from independently becoming distant road.

Only points within `ground_plane_tolerance_m` of a **confirmed local plane**
are removed. Default 0.04 m is an initial value, not a measured MORAI noise
calibration; it can be changed live up to 0.08 m. This is a plane residual
tolerance, not a blanket object-height cutoff. Objects above a confirmed
plane survive even below the former 0.22 m absolute threshold. When IMU is
missing/stale, no plane removal or posture correction is performed. Unsupported
patches retain the original `0.22 < z < 2.6 m` gate; false ground obstacles can
therefore still occur there. Ground status explicitly reports these conditions.
The live panel now exposes this gate as `obstacle_min_height_m` (0.22 m) and
`obstacle_max_height_m` (2.6 m). The lower limit is used only for fallback or
unconfirmed road; confirmed road still keeps low objects above its plane
tolerance. The upper limit also bounds confirmed plane-relative object height.
The lower limit may be zero and must be strictly less than the upper limit;
invalid range edits preserve the previously applied pair.
Fallback heights are relative to `ground_reference_z_m`, while confirmed
patches use their own fitted plane. Physical debug point heights stay unchanged.

`map_debug.launch` shows fresh pitch, roll, worst contributing IMU time skew,
supported patch count and removed/retained point counts. The left table edits
`ground_plane_tolerance_m`, `imu_max_skew_s` and `ground_max_slope_deg` using
the same owning-node acknowledgement as the other avoidance settings. Mount
extrinsics are startup settings in `config/ground_filter.yaml`. Removed road
is recorded with physical heights on `/waypoint_debug/ground_points`; Map Debug
projects a display copy to z=0.15 m on `/map_debug/ground_points` so green points
are visible above the map image. Orange points on
`/waypoint_debug/obstacle_points` are retained candidates before lane filtering.
Point displays expire after 0.35 s; panel status expires after 0.5 s. Cloud
publication is skipped without subscribers, and capped to 15,000 points each.
The panel also reports missing clouds, lane geometry, map pose, wrong cloud
frames and cloud clock mismatch. Detailed rejection reasons, sample counts,
time skew and near-ego height percentiles are stored in ground status.

### Track confidence and smooth braking

Tracks expose confirmation, hit count, observation duration, measurement stamp
and velocity reliability. Confirmation requires >=3 hits and >=0.2 s. Size
changes beyond a factor of three reset association. A short velocity history
must agree within 3 m/s to permit extrapolation; otherwise points keep their
observed position. Unknown rear traffic still blocks lane changes.
Far tiny provisional objects do not select a passing lane. All objects inside
the reaction/stopping envelope, and objects with radius >=0.35 m, remain active.

A transition collision check reports earliest predicted contact time and ego
travel distance. Contact within 1 s, within 1 m, or inside the reaction/stopping
distance causes immediate `transition_emergency` and full braking. More distant
contacts use `transition_slowing` and the existing stopping-speed equation.
The soft brake pedal limit is 0.35; it is not a calibrated physical deceleration.
Clear transitions still impose no passing speed cap. Missing paths stop immediately.

`PedalGuard` keeps a full-stop release pending for 0.4 s and three distinct new
perception samples. Any new full-stop decision resets the confirmation. Throttle
increases at <=0.5 per second; braking/throttle release is immediate. Runtime
values are in the existing panel, and diagnostics include collision risk and
requested versus held target speed. See DRIVING_LOG.md for calibration and replay limits.

### Record and validate actual braking

For the requested driving launch, logging is integrated; no separate recorder
is needed. It captures raw sensors, filter output, lane/stop decisions, actual
control transmission, live parameters and diagnostic simulator truth:

```bash
roslaunch unita_launch test_map_raceline.launch morai_ip:=127.0.0.1 log:=true
roslaunch unita_waypoint map_debug.launch
```

Each run creates `log/driving_run_*/` with bags, JSONL events, parameter history,
route/config/source snapshots and a manifest of missing topics. See
[DRIVING_LOG.md](DRIVING_LOG.md) for collection steps and contents.
Stop normally with Ctrl-C and let the bag close. For offline analysis of a run
with unchanged filter settings:

```bash
rosrun unita_waypoint ground_filter_analysis.py /path/to/driving_run_.../sensors_..._0.bag \
  --config /path/to/driving_run_.../lidar_params.yaml --stride 1 --output /tmp/ground_analysis.json
```

The smaller allowed-sensor-only recorder remains available for a dedicated
ground-filter recording:

```bash
roslaunch unita_waypoint ground_filter_record.launch
# Drive the planned acceleration -> hard stop scenario, then Ctrl-C recorder.
rosparam dump /tmp/ground_filter_run.yaml /lidar_obstacles
rosrun unita_waypoint ground_filter_analysis.py /path/to/ground_braking_TIMESTAMP_0.bag \
  --config /tmp/ground_filter_run.yaml --stride 1 --output /tmp/ground_analysis.json
```

The recorder sends no control command and uses an exact topic whitelist:
`/imu`, `/lidar3D`, GPS/IMU-derived `/localization/pose`,
and `/waypoint_debug/ground_status`. It splits files
at 512 MB. The analyzer uses only raw IMU/LiDAR and the specified static settings;
leave the live filter settings unchanged while recording. Analyze split files
separately; first/last scans may lack an IMU bracket at a file boundary.

The report compares old height candidates against newly confirmed road,
reports pitch range, timestamp skew and conditional road residuals, and saves
per-scan diagnostics. Residuals inside the current tolerance are truncated;
they are insufficient by themselves to set a new tolerance or prove simulator
pitch correctness. Inspect both point colours against the physical scene.
Check empty-road braking, stationary and moving cars during braking, slopes,
road crests, IMU delay/missing data, and runtime changes. Confirm actual objects
remain in orange and track boxes, and ground does not create new boxes. The
automated tests cover these with synthetic data; actual MORAI recordings are
still required before claiming real braking performance.

## Rear and side traffic

The detector uses `/lidar3D` and capture-time `/localization/pose` to track
map-frame positions and velocities. Its default ROI extends 55 m forward,
55 m rearward and 12 m to either side. It excludes the ego body separately,
so nearby side and rear cars are retained. Long trucks are represented by
several bounded circles instead of being dropped by the old 8 m size filter.
Motion prediction is used for track association. A missing object remains as
a predicted, expanded obstacle for up to 0.4 s to cover brief occlusion.

Each lane-change candidate checks target-lane and intermediate-lane traffic.
Rear bumper clearance must exceed 8 m plus 2 s of positive closing speed,
and rear time to collision must exceed the estimated merge duration plus the
reaction time. An unclassified rear track blocks a new merge until its motion
is established. Side vehicles are checked against the swept front/rear vehicle
footprint over the transition and a buffer after completion. Predictions use
actual arrival time, without freezing obstacle positions after 3 s. These are
constant-velocity predictions; they cannot see a vehicle fully hidden from the
LiDAR or guarantee that another driver will maintain its motion.

`/waypoint_debug/obstacles` is a timestamped JSON String in the `map` frame.
The follower stops if this stream is missing or stale. Rear perception uses
the 360-degree LiDAR; it does not infer rear vehicles from camera images.

After at least three matched observations, an obstacle is dynamic when its
smoothed map-frame speed exceeds `lidar_obstacles/dynamic_speed_threshold_mps`
(default 0.8 m/s, or 2.88 km/h). At or below the threshold it is static;
new tracks and briefly occluded predictions are unknown. Raising the threshold
classifies more slow-moving tracks as static. The estimated velocity and
collision prediction remain available regardless of this label. This setting
controls classification; it does not remove ground returns from detection.

## Saved geometry and settings

Raceline 2/3 currently exist only through part of Sector 11. Source IDs restrict
selection to the saved interval. The planner reduces speed approaching a lane
end and returns to R1 while the saved transition corridor still exists. A
blocked exit produces a stop; it never extrapolates R3 beyond its saved end.

Vehicle dimensions use this workspace's MORAI/LEO configuration: half-width
0.946 m, front extent from rear axle 3.845 m, rear overhang 0.790 m, and LiDAR
mount 1.4 m forward / 1.23 m high. Planning settings on `gps_waypoint_follower`:

- `preferred_raceline` (3 in test launch, 1 in generic launch)
- `preferred_raceline_sector` (11 in test launch, 0 = unrestricted)
- `preferred_raceline_end_checkpoint` ("11" in test launch, empty = unrestricted)
- `preferred_raceline_after_checkpoint` (1 outside the priority interval)
- `rear_extent_m` (0.790)
- `lane_change_min_rear_gap_m` (8.0)
- `lane_change_rear_headway_s` (2.0)
- `lane_change_smoothness_s` (2.5, minimum transition length = speed in m/s × this value)
- `avoidance_margin_m` (0.35)
- `avoidance_decel_mps2` (3.0)
- `avoidance_reaction_s` (0.4)
- `obstacle_horizon_m` (45.0, arc distance to the obstacle center's projection)

Clear lane changes, overtakes and returns follow the existing point speed
profile. There is no separate transition speed limit or pre-merge 20 km/h
slowdown. Rear gap and swept-path predictions use the actual current speed.
Blocked passing lanes still cause leader-following with a stopping-distance
cap; blocked committed transitions stop. Missing saved lane coverage still
imposes a stopping cap while holding that lane.

## Debug and validation

`map_debug.launch` subscribes to `/waypoint_debug/obstacles` and publishes
`/map_debug/obstacles` as a MarkerArray. The default RViz display shows filled
boxes and outlines: red for dynamic, blue for static, yellow for unknown.
Only lane-filtered detections are displayed. Labels include track ID and, for
dynamic objects, estimated speed. Boxes enclose
the tracker's 2D radius; their height is a display setting, not measured vehicle
height. Rear and side detections are shown too. Removed IDs are explicitly
deleted, and stale input expires after 0.5 s. Each marker also has a 0.5 s
lifetime so stopped visualization nodes do not leave permanent boxes.

The left `map_debug` panel displays estimated current lane, target lane and
preferred lane, plus `RACELINE`, `LANE_CHANGING`, `LANE_WAIT`, `TAILING`,
`OVERTAKING`, `RETURNING`, `BLOCKED` or `SENSOR_HOLD`. During a merge, the
current lane is the nearest saved lane, not a claim that the car has already
reached the target. `lane_change_reason` distinguishes rear/side traffic and
a predicted obstruction.

The panel's real-time settings table shows explanations, the actual applied
values and editable text boxes for:

| ROS node | Parameter | Default | Meaning |
| --- | --- | --- | --- |
| `lidar_obstacles` | `front_range_m` | 55 m | Forward ROI from the rear axle |
| `lidar_obstacles` | `rear_range_m` | 55 m | Rearward ROI from the rear axle |
| `lidar_obstacles` | `side_range_m` | 12 m | ROI on each side, without automatic widening |
| `lidar_obstacles` | `dynamic_speed_threshold_mps` | 0.8 m/s | Speed above which a track with at least 3 observations is dynamic |
| `gps_waypoint_follower` | `obstacle_horizon_m` | 45 m | Threat horizon along the current path |
| `gps_waypoint_follower` | `avoidance_margin_m` | 0.35 m | Vehicle-to-obstacle margin |
| `gps_waypoint_follower` | `lane_change_min_rear_gap_m` | 8 m | Base rear bumper gap |
| `gps_waypoint_follower` | `lane_change_rear_headway_s` | 2 s | Extra gap per positive closing speed |
| `gps_waypoint_follower` | `lane_change_smoothness_s` | 2.5 s | Larger values make high-speed lane changes longer and gentler |
| `gps_waypoint_follower` | `avoidance_reaction_s` | 0.4 s | Reaction time for braking/prediction |
| `gps_waypoint_follower` | `avoidance_decel_mps2` | 3 m/s² | Deceleration assumed by stopping calculation |

Enter a positive finite number and press Enter or Apply. Each owning node polls
its private ROS namespace every 0.2 s, applies changes on its processing loop,
and publishes acknowledgements on `/waypoint_debug/avoidance_config/<node>`.
The panel displays Applied only while that node's acknowledgement is fresh,
and retains a pending value until the node reports it. Invalid edits preserve
the last valid value. Ordinary `rosparam set` edits also work. Settings are
saved to `config/avoidance_settings_lidar_obstacles.json` or
`config/avoidance_settings_gps_waypoint_follower.json` after the owning node
accepts them. Saved values are loaded on restart and override launch defaults.
`raceline_width` is a read-only text box synchronized to the tuner save.

The purple `obstacle_horizon` display uses the follower's actual current path,
clipped at the configured arc distance, rather than a circle around the LiDAR.
Its ribbon half-width is vehicle half-width plus safety margin. Threat checks
also account for each obstacle's radius. The end label shows the configured
distance and, if shorter, the available saved path length. Markers expire
after 0.5 s when driving decisions stop arriving.

Tests cover fast/close/unknown rear tracks, intermediate and parallel traffic,
truck occupancy, brief occlusion, fixed transition endpoints, R3 overtaking
and return clearance, saved lane endings, and the follower's local-path control.
Run the updated launch and map_debug again to load changes into running nodes.
Unit tests and the panel build do not replace a MORAI run with moving traffic.
