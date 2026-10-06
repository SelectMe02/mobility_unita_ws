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
Set the launch argument on the driving launch to change the filter width.
The saved project width is preferred when no override is supplied; the YAML
fallback in `config/raceline_layout.yaml` is 3.2 m.

```bash
roslaunch unita_launch test_map_raceline.launch morai_ip:=127.0.0.1 raceline_width:=3.2
# Example alternate full width (half-width is then 1.8 m):
roslaunch unita_launch test_map_raceline.launch morai_ip:=127.0.0.1 raceline_width:=3.6
```

After a width change, restart the driving launch. Driving always loads saved
JSON centerlines, including manual extensions, without regenerating them from
the width parameter. Edit centerlines in Map Tuner and save the JSON. The
default launch resets the width override to empty so stale ROS values cannot
override the saved width. Map Debug always draws the saved JSON paths.

Roadside LiDAR points are dropped before clustering. Out-of-lane object centers
and predicted tracks are discarded before publishing obstacles, and the follower
and debug also apply the corridor filter. An out-of-lane detection never becomes
a passing threat or debug box merely because its collision radius is large.
A briefly occluded track is retained only while its predicted center stays in a
mapped lane. Missing/invalid lane geometry holds perception until a valid map
arrives rather than recognizing unrestricted surroundings.

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
- `avoidance_margin_m` (0.35)
- `avoidance_decel_mps2` (3.0)
- `avoidance_reaction_s` (0.4)
- `pass_speed_kmh` (20.0, during transitions)

The point speed profile remains an upper bound. Faster leaders may require
following instead of passing; the planner does not increase the configured
transition speed to force an overtake. Keep the launch's forward sensor range
and stopping distance in mind when validating the existing high-speed profile.

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
reached the target. `lane_change_reason` distinguishes rear/side traffic,
slowing before merging, and a predicted obstruction.

Tests cover fast/close/unknown rear tracks, intermediate and parallel traffic,
truck occupancy, brief occlusion, fixed transition endpoints, R3 overtaking
and return clearance, saved lane endings, and the follower's local-path control.
Run the updated launch and map_debug again to load changes into running nodes.
Unit tests and the panel build do not replace a MORAI run with moving traffic.
