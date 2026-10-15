import numpy as np

from leo_avoidance.road_points import obstacle_points

LIDAR_Z = -1.23  # road under the LiDAR on flat ground


def road(slope=0., camber=0., x=(3., 55.), y=(-6., 6.), step=.4):
    """Road returns rising ahead with slope (rise/run) and sideways with camber."""
    xs, ys = np.meshgrid(np.arange(*x, step), np.arange(*y, step))
    xs, ys = xs.ravel(), ys.ravel()
    return np.column_stack((xs, ys, LIDAR_Z + slope * xs + camber * ys))


def box(cx, cy, ground_z, size=(1.5, 1.5, 1.5), step=.15):
    """Vertical faces of a box standing on the road."""
    pts = []
    for z in np.arange(ground_z + .1, ground_z + size[2], step):
        for y in np.arange(cy - size[1] / 2, cy + size[1] / 2, step):
            pts.append((cx - size[0] / 2, y, z))
        for x in np.arange(cx - size[0] / 2, cx + size[0] / 2, step):
            pts.append((x, cy - size[1] / 2, z))
    return np.asarray(pts)


def test_flat_road_has_no_obstacles_and_a_box_on_it_is_kept():
    assert len(obstacle_points(road())) == 0
    kept = obstacle_points(np.vstack((road(), box(20., 0., LIDAR_Z))))
    assert len(kept) > 20
    assert np.all(np.abs(kept[:, 0] - 20.) < 1.)


def test_uphill_and_cambered_road_ahead_is_not_an_obstacle():
    # 8 % uphill: 4 m higher at 50 m ahead; the tunnel plane fit rejects this.
    assert len(obstacle_points(road(slope=.08))) == 0
    assert len(obstacle_points(road(camber=.04))) == 0


def test_box_on_a_slope_is_kept():
    ground = LIDAR_Z + .08 * 30.
    kept = obstacle_points(np.vstack((road(slope=.08), box(30., 0., ground))))
    assert len(kept) > 20


def test_kerb_is_not_an_obstacle_but_a_pedestrian_is():
    kerb = np.column_stack((np.arange(5., 50., .3), np.full(150, -3.),
                            np.full(150, LIDAR_Z + .15)))
    assert len(obstacle_points(np.vstack((road(), kerb)))) == 0
    walker = box(15., -3., LIDAR_Z, size=(.5, .5, 1.7))
    assert len(obstacle_points(np.vstack((road(), walker)))) > 10


def test_ego_body_returns_are_removed():
    body = np.column_stack((np.full(20, 1.), np.linspace(-.8, .8, 20), np.full(20, -.5)))
    assert len(obstacle_points(np.vstack((road(), body)))) == 0


def test_road_turning_uphill_ahead_is_not_an_obstacle():
    # Flat, then 3 % up from 15 m: the tunnel plane fit extends the near
    # plane and reports the far road as thousands of obstacle points.
    points = road()
    points[:, 2] += np.where(points[:, 0] > 15., .03 * (points[:, 0] - 15.), 0.)
    assert len(obstacle_points(points)) == 0


def test_guardrail_along_a_bend_is_a_boundary_but_cars_and_walkers_are_not():
    from leo_avoidance.road_points import boundary_mask
    s = np.arange(0., 40., .2)
    rail = np.column_stack((s, np.full(len(s), -2.6)))           # 40 m rail, 2.6 m right
    car_x = np.arange(20., 24.8, .2)
    car_side = np.column_stack((car_x, np.full(len(car_x), 3.0)))
    car_back = np.column_stack((np.full(10, 20.), np.linspace(3., 5., 10)))
    walker = np.column_stack((np.full(6, 15.), np.linspace(-2.1, -1.9, 6)))
    box_in_lane = np.column_stack((np.full(8, 30.), np.linspace(-.7, .7, 8)))
    sd = np.vstack((rail, car_side, car_back, walker, box_in_lane))
    mask = boundary_mask(sd[:, 0], sd[:, 1])
    n = len(rail)
    assert mask[:n].mean() > .95
    assert not mask[n:].any()


def test_short_partly_occluded_guardrail_is_removed_but_car_side_is_kept():
    from leo_avoidance.road_points import boundary_mask
    rail_s = np.r_[np.arange(10., 13., .2), np.arange(13.8, 17.2, .2)]
    car_s = np.arange(25., 29.8, .2)
    s = np.r_[rail_s, car_s]
    d = np.r_[np.full(len(rail_s), -3.), np.full(len(car_s), 3.)]
    mask = boundary_mask(s, d)
    assert mask[:len(rail_s)].mean() > .9
    assert not mask[len(rail_s):].any()


def test_long_parked_truck_in_the_next_lane_is_not_a_boundary():
    from leo_avoidance.road_points import boundary_mask
    s, d = np.meshgrid(np.arange(10., 22., .2), np.arange(2.3, 4.7, .2))
    assert not boundary_mask(s.ravel(), d.ravel()).any()
