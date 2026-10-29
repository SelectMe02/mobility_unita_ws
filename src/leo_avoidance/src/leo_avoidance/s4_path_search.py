"""Bounded S4-only spline search; callers must run the exact swept-body check."""
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


def search_offsets(start, end, initial, slope, lane_offset, items, cfg, budget=.18):
    if end-start < 8.:
        return None
    knots = np.linspace(start, end, max(5, min(10, int((end-start)/4.)+1)))
    samples = np.arange(start, end, .25)
    low, high = min(0., lane_offset), max(0., lane_offset)
    half_l = (cfg.front_extent_m+cfg.rear_extent_m)/2.+.15
    half_w = cfg.vehicle_half_width_m+cfg.side_margin_m+.15
    center = (cfg.front_extent_m-cfg.rear_extent_m)/2.
    deadline = time.monotonic()+budget
    best = None

    def evaluate(values):
        nonlocal best
        if time.monotonic() > deadline:
            raise TimeoutError
        y = np.r_[initial, values, 0.]
        curve = CubicSpline(knots, y, bc_type=((1, slope), (1, 0.)))
        d, derivative = curve(samples), curve(samples, 1)
        curvature = np.abs(curve(samples, 2))/(1.+derivative**2)**1.5
        angle = np.arctan(derivative)
        c, sn = np.cos(angle), np.sin(angle)
        es, ed = samples+c*center, d+sn*center
        constraints = [d-low, high-d, .135-curvature]
        for o in items:
            ds, dd = o['s']-es, o['d']-ed
            ol, ow = o['l'], o['r']
            separation = np.maximum.reduce([
                abs(ds)-(half_l*abs(c)+half_w*abs(sn)+ol),
                abs(dd)-(half_l*abs(sn)+half_w*abs(c)+ow),
                abs(ds*c+dd*sn)-(half_l+ol*abs(c)+ow*abs(sn)),
                abs(-ds*sn+dd*c)-(half_w+ol*abs(sn)+ow*abs(c))])
            constraints.append(separation-.04)
        margins = np.concatenate(constraints)
        if np.min(margins) >= -1e-5:
            best = (knots.tolist(), y.tolist(), curve(knots, 1).tolist())
        return curve, margins

    def cost(values):
        curve, _ = evaluate(values)
        return float(np.mean(curve(samples)**2)*.02 + np.mean(curve(samples, 2)**2)*30.)

    # Different initial offsets expose alternate sides of staggered obstacles.
    for fraction in (.65, 1., .3):
        try:
            seed = np.full(len(knots)-2, lane_offset*fraction)
            minimize(cost, seed, method='SLSQP', bounds=[(low, high)]*len(seed),
                     constraints=[dict(type='ineq', fun=lambda v: evaluate(v)[1])],
                     options=dict(maxiter=40, ftol=1e-5))
        except TimeoutError:
            break
        if best is not None:
            break
    return best
