"""Route-associated camera signal gating. No simulator signal-state input."""
import math
import numpy as np
from .core_local import SignalLatch, signal_permission, speed_for_clearance
from .yellow_stop import FRONT_OFFSET_M, camera_signal_state


class RouteProgress:
    def __init__(self, points, max_error=2.0, backward_window=10):
        self.points = np.asarray(points, float)[:, :2]
        if len(self.points) < 3 or not np.isfinite(self.points).all():
            raise ValueError('invalid route')
        self.delta = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=1)
        self.s = np.r_[0., np.cumsum(self.lengths)]
        self.total = float(self.s[-1]); self.index = None
        self.max_error = max_error
        self.backward_window = int(backward_window)
        if self.backward_window < 1:
            raise ValueError('backward_window must be positive')

    def locate(self, x, y, yaw):
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError('invalid pose')
        n = len(self.delta)
        indices = np.arange(n) if self.index is None else np.array(
            [(self.index+j) % n for j in range(
                -self.backward_window, min(160, n-self.backward_window))])
        delta = self.delta[indices]; lengths = self.lengths[indices]
        t = np.sum((np.array([x,y])-self.points[indices])*delta, axis=1)/np.maximum(lengths**2, 1e-12)
        projected = self.points[indices]+np.clip(t,0,1)[:,None]*delta
        error = np.linalg.norm(projected-[x,y], axis=1)
        aligned = (delta @ np.array([math.cos(yaw),math.sin(yaw)])) / np.maximum(lengths,1e-9) > .5
        error[~aligned | (lengths < 1e-6)] = np.inf
        selected = int(error.argmin())
        if error[selected] > self.max_error:
            raise ValueError('vehicle is outside route corridor or facing backwards')
        self.index = int(indices[selected])
        return float(self.s[self.index]+np.clip(t[selected],0,1)*lengths[selected]), float(error[selected])


class TrafficDecision:
    def __init__(self, stops, route_length, front_margin=5.,
                 decel=3., latency=.5, stoplines=None, missed_signal_hold_s=0.):
        self.stops = sorted(stops, key=lambda v:v['s'])
        # A frame with no detection in the head ROI keeps the last permitted
        # signal of the same stop for this long. Red, yellow, an uncertain
        # head and a stale camera still stop at once. 0 disables it.
        if not math.isfinite(missed_signal_hold_s) or missed_signal_hold_s < 0:
            raise ValueError('invalid missed signal hold')
        self.missed_signal_hold_s = missed_signal_hold_s
        self.last_permit = None
        if not math.isfinite(route_length) or route_length<=0:
            raise ValueError('invalid route length')
        if any(not math.isfinite(v) or v<=0 for v in (front_margin,decel,latency)):
            raise ValueError('invalid traffic braking configuration')
        for stop in self.stops:
            if stop.get('signal_camera','front') not in ('front','up'):
                raise ValueError('unsupported surveyed signal camera')
            if (not isinstance(stop['s'],(float,int)) or not math.isfinite(stop['s'])
                    or not 0<=stop['s']<route_length or stop['clearance_m']<=0
                    or not math.isfinite(stop['clearance_m'])):
                raise ValueError('invalid intersection position/exit')
            if stop.get('reviewed') and stop.get('controlled') is True:
                roi=stop.get('signal_roi')
                if (not isinstance(roi,list) or len(roi)!=4 or not all(math.isfinite(v) for v in roi)
                        or not 0<=roi[0]<roi[2]<=1 or not 0<=roi[1]<roi[3]<=1):
                    raise ValueError('reviewed signal requires a valid controlling-head ROI')
            for key in ('signal_min_width', 'signal_activation_distance_m', 'stop_margin_m'):
                if key in stop and (not isinstance(stop[key],(float,int))
                        or not math.isfinite(stop[key]) or stop[key]<=0):
                    raise ValueError('invalid surveyed head association bound')
            if stop.get('signal_min_width',0)>=1:
                raise ValueError('invalid surveyed head size')
        self.route_length = route_length
        # No route speed of its own: the Map Tuner point speed is the target,
        # this gate only lowers it to stop before an unpermitted line.
        self.front_margin = front_margin; self.maximum = float('inf')
        self.decel = decel; self.latency = latency
        self.latch = SignalLatch(0.,1.,min_frames=1)
        self.committed = None
        self.active = None
        self.last_s = None
        self.stoplines = None if stoplines is None else {node['idx']: node for node in stoplines}
        if self.stoplines is not None:
            for stop in self.stops:
                if stop.get('controlled') is True:
                    node = self.stoplines.get(stop['id'])
                    if node is None or node['traffic_light_id'] != stop.get('traffic_light_id'):
                        raise ValueError('survey/stoplines mismatch at {}'.format(stop['id']))
        self.approach_go_selected = None

    def reset(self):
        self.latch = SignalLatch(0.,1.,min_frames=1); self.committed = None; self.last_s = None; self.active = None
        self.approach_go_selected = None
        self.last_permit = None

    def held_permit(self, stop, label, now):
        """Signal state of a recent permit when this frame shows no relevant head."""
        if (label != 'no relevant signal' or self.last_permit is None
                or self.missed_signal_hold_s <= 0):
            return None
        stop_id, state, seen = self.last_permit
        if stop_id != stop['id'] or not 0 <= now - seen <= self.missed_signal_hold_s:
            return None
        return state

    def distance_to_next_controlled_stop(self, progress, front_offset=0.):
        """Meters from the front axle to the next signal-controlled stop line."""
        if not math.isfinite(progress) or not math.isfinite(front_offset):
            raise ValueError('invalid route progress or front offset')
        distances = [((stop['s'] - progress) % self.route_length)
                     for stop in self.stops if stop.get('controlled') is True]
        return max(0., min(distances) - front_offset) if distances else None

    def decide(self, progress, signals, now, front_offset=0., yield_clear=False,
               pose=None, speed_mps=None, ros_now=None):
        return self._decide(progress, signals, now, front_offset, yield_clear,
                            pose, speed_mps, ros_now)

    def _decide(self, progress, signals, now, front_offset=0., yield_clear=False,
                pose=None, speed_mps=None, ros_now=None):
        if not math.isfinite(front_offset) or not 0<=front_offset<=5.:
            raise ValueError('invalid front-wheel crossing offset')
        if self.last_s is not None:
            jump = (progress-self.last_s+self.route_length/2)%self.route_length-self.route_length/2
            if abs(jump)>10.:
                self.reset(); raise ValueError('route progress reset: manual re-arm required')
            if self.active is not None:
                self.active[0] -= jump
        self.last_s = progress
        if self.committed is not None:
            stop, start, clearance = self.committed
            advanced = (progress-start)%self.route_length
            if advanced < clearance:
                return self.maximum, {'state':'clearing','intersection_id':stop['id'],
                                      'reason':'entered on permitted camera signal'}
            self.committed = None; self.active = None; self.latch = SignalLatch(0.,1.,min_frames=1)
            self.approach_go_selected = None
        if not self.stops:
            return 0., {'state':'hold','reason':'intersection survey is empty'}
        if self.active is None:
            distances = [((v['s']-progress)%self.route_length,v) for v in self.stops]
            self.active = list(min(distances,key=lambda v:v[0]))
            self.approach_go_selected = None
        distance, stop = self.active
        # No signal ignored merely because it is outside the camera frame.
        detail = dict(state='approach', intersection_id=stop['id'],
                      signal_camera=stop.get('signal_camera','front'),
                      stop_distance=distance, front_wheel_distance=distance-front_offset,
                      maneuver=stop['maneuver'], release_stable=False)
        if stop.get('reviewed') and stop.get('controlled') is False:
            if distance < -5.:
                self.active = None
                return self._decide(progress, signals, now, front_offset, yield_clear,
                                   pose, speed_mps, ros_now)
            detail.update(state='clear',reason='surveyed uncontrolled crossing')
            return self.maximum, detail
        allowed = False; label = 'intersection/head association not reviewed'
        signal_state = 'unknown'
        if stop.get('reviewed') and stop.get('controlled') is True:
            if distance > stop.get('signal_activation_distance_m',float('inf')):
                self.latch.update(False,None,None,now);self.last_permit=None
                label='outside surveyed signal observation range'
            elif signals and signals.get('camera','front')!=stop.get('signal_camera','front'):
                self.latch.update(False,None,None,now);self.last_permit=None;label='wrong camera for surveyed signal head'
            elif signals and signals.get('valid'):
                signal_state = camera_signal_state(
                    signals.get('detections', []), stop['signal_roi'],
                    min_width=stop.get('signal_min_width', 0.))
                allowed,label = signal_permission(signals.get('detections',[]),
                    'right_green_only' if stop['maneuver']=='right_unprotected' else stop['maneuver'],stop['signal_roi'],
                    min_width=stop.get('signal_min_width',0.))
                if stop['maneuver']=='right_unprotected' and yield_clear is not True:
                    allowed=False;label='right turn requires completed stop and fresh observed yield clearance'
                held = self.held_permit(stop, label, now)
                if held is not None:
                    # Keep the latch; the next detection continues its evidence.
                    signal_state=held;allowed=True;label='permit held through missed detection'
                else:
                    allowed=self.latch.update(allowed,(stop['id'],stop['maneuver'],label),signals.get('frame'),now)
                    self.last_permit=(stop['id'],signal_state,now) if allowed else None
            else:
                self.latch.update(False,None,None,now);self.last_permit=None;label='camera result missing or stale'
        else:
            self.latch.update(False,None,None,now);self.last_permit=None
        detail.update(release_stable=allowed,reason=label)
        if self.stoplines is not None and stop['maneuver'] == 'straight' and stop.get('controlled') is True:
            result = self.decide_yellow(stop, progress, distance, signals, now,
                                        pose, speed_mps, ros_now, signal_state, allowed)
            if result is not None:
                if label == 'permit held through missed detection':
                    result = (result[0], dict(result[1], signal_held=True))
                return result
        if allowed:
            # Use the measured GPS-to-front-wheel projection, never the
            # braking margin. Keep permission through the surveyed exit.
            if distance <= front_offset:
                self.committed=(stop,progress,stop['clearance_m']+max(0.,distance))
            return self.maximum, detail
        if distance < 0:
            return 0.,dict(detail,state='hold',reason='crossed unpermitted/unreviewed stop line: reset required')
        available=max(0.,distance-max(self.front_margin,stop.get('stop_margin_m',0.),front_offset))
        return min(self.maximum,speed_for_clearance(available,self.decel,self.latency)), detail

    def decide_yellow(self, stop, progress, route_distance_m, signals, now,
                      pose, speed_mps, ros_now, signal_state, allowed):
        """Override only the reviewed straight approach; Leo owns all other turns."""
        node = self.stoplines.get(stop['id'])
        if not stop.get('reviewed'):
            self.approach_go_selected = None
            return None
        if node is None or pose is None or speed_mps is None or ros_now is None:
            self.approach_go_selected = None
            return 0., dict(state='hold', intersection_id=stop['id'],
                            signal_camera=stop.get('signal_camera','front'),
                            signal='unknown', can_go=False, can_stop=False,
                            decision='stop', reason='mapped pose/speed/time unavailable')
        if (not all(math.isfinite(value) for value in (*pose, speed_mps, ros_now))
                or speed_mps < 0):
            self.approach_go_selected = None
            return 0., dict(state='hold', intersection_id=stop['id'],
                            signal_camera=stop.get('signal_camera','front'),
                            signal='unknown', can_go=False, can_stop=False,
                            decision='stop', reason='invalid mapped pose/speed/time')
        # Distance along the global route, so curved approaches stay exact.
        # Being in another lane is the route corridor check's job (the
        # caller's RouteProgress), not a straight-line projection's.
        d_m = route_distance_m - FRONT_OFFSET_M
        detail = dict(state='approach', intersection_id=stop['id'],
                      signal_camera=stop.get('signal_camera','front'),
                      traffic_light_id=stop['traffic_light_id'],
                      stop_distance=route_distance_m, d_m=d_m, v_mps=speed_mps,
                      signal=signal_state, can_go=False, can_stop=False,
                      decision='stop', release_stable=allowed)
        if d_m <= 0 and self.approach_go_selected == stop['id']:
            self.committed = (stop, progress, stop['clearance_m'] + max(0., route_distance_m))
            self.approach_go_selected = None
            return self.maximum, dict(detail, state='clearing', decision='go',
                                      reason='front already crossed on permitted signal')
        # Green goes; yellow, red and an unseen signal stop at the line.
        decision = 'go' if signal_state == 'green' and allowed else 'stop'
        detail.update(decision=decision, reason='reviewed camera signal')
        if decision == 'go':
            self.approach_go_selected = stop['id']
            if d_m <= 0:
                self.committed = (stop, progress,
                                  stop['clearance_m'] + max(0., route_distance_m))
            return self.maximum, detail
        self.approach_go_selected = None
        available_m = max(0.,route_distance_m-max(
            self.front_margin,stop.get('stop_margin_m',0.),FRONT_OFFSET_M))
        cap = min(self.maximum, speed_for_clearance(available_m,self.decel,self.latency))
        return cap, dict(detail, state='hold' if d_m <= 0 else 'approach',
                         reason='stop line reached without permission' if d_m <= 0
                                else detail['reason'])
