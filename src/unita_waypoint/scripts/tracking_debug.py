"""Read-only snapshots of the path and decisions used by the followers."""
import math


def saved_route_window(points, index, loop, horizon_m=60.):
    """Return actual saved waypoints ahead, respecting the follower's loop rule."""
    count = len(points)
    if loop and count > 1 and math.dist(points[0], points[-1]) <= 1e-6:
        count -= 1
    if not 0 <= index < count:
        return []
    indices = ((index + offset) % count for offset in range(count)) if loop else range(index, count)
    result, travelled = [], 0.
    for i in indices:
        point = points[i]
        if result:
            travelled += math.dist(result[-1], point)
        result.append(point)
        if travelled >= horizon_m:
            break
    return result


def local_plan_trace(planner, decision, obstacles):
    """Copy an executed planner result; never run the planner for display."""
    # This early return precedes branch selection; highway_active can still
    # describe the previous tick at the end of a saved path.
    before_selection = decision.state == 'BLOCKED' and not decision.path and decision.reason == 'saved_path_ends'
    highway = bool(getattr(planner, 'highway_active', False)) and not before_selection
    transition = getattr(planner, 'transition', None)
    blocker = getattr(planner, 'gap_blocker', None)
    return dict(
        planner_mode='waiting' if before_selection else 'highway' if highway else 'local_avoidance',
        planner_state=decision.state,
        planner_reason=decision.reason,
        planner_speed_cap_kmh=(decision.speed_cap_kmh
                               if math.isfinite(decision.speed_cap_kmh) else None),
        planner_obstacle_input_count=len(obstacles),
        confirmed_obstacle_count=(sum(bool(o.confirmed) for o in obstacles)
                                  if all(hasattr(o, 'confirmed') for o in obstacles) else None),
        highway_active=highway,
        highway_sectors=list(getattr(planner, 'highway_sectors', ())),
        highway_emergency=bool(getattr(planner, 'highway_emergency', False)) if highway else False,
        highway_road_returns_ignored=(getattr(planner, 'highway_road_returns_ignored', 0)
                                      if highway else 0),
        obstacle_horizon_m=getattr(planner, 'highway_obstacle_horizon_m', None) if highway
                           else getattr(planner, 'obstacle_horizon_m', None),
        transition=dict(transition) if transition is not None and not before_selection else None,
        gap_blocker=(dict(id=getattr(blocker[0], 'track_id', None), x=blocker[0].x, y=blocker[0].y,
                          front_gap_m=blocker[1]) if blocker is not None else None))


def tracking_debug_fields(follower):
    data = dict(getattr(follower, 'debug_trace', {}))
    data.setdefault('planner_mode', 'waiting')
    data.setdefault('local_path_source', 'none')
    data['lookahead_m'] = getattr(follower, 'lookahead_distance', None)
    data['drive_mode'] = getattr(follower, 'drive_mode', None)
    return data
