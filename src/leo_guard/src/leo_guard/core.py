"""Experimental perception and conservative command constraints, independent of ROS.

Coordinates: rear axle origin, X forward, Y left. Polynomial boundaries are
models of observed paint candidates, NOT proof of ground-truth wheel clearance.
All distances and timestamps are validated before an output can permit motion.
"""
import math
import numpy as np


def project_ground_pixels(pixels, intrinsics, optical_to_rear, ground_z=0.):
    """Calibrated optical rays intersect a ground plane; horizon rays rejected.

    optical_to_rear is 4x4: optical x right, y down, z forward. No guessed
    camera height, pitch or rear-axle offset is supplied by this function.
    """
    k=np.asarray(intrinsics,dtype=float);t=np.asarray(optical_to_rear,dtype=float)
    uv=np.asarray(pixels,dtype=float)
    if k.shape!=(3,3) or t.shape!=(4,4) or uv.ndim!=2 or uv.shape[1]!=2:
        raise ValueError('invalid camera calibration/pixels')
    if not all(np.isfinite(v).all() for v in (k,t,uv)) or not math.isfinite(ground_z):
        raise ValueError('nonfinite camera calibration')
    if not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5):
        raise ValueError('invalid camera rigid transform')
    rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(k).T@t[:3,:3].T
    good=rays[:,2]<-1e-4
    scale=np.zeros(len(rays));scale[good]=(ground_z-t[2,3])/rays[good,2]
    good &= scale>0
    points=t[:3,3]+rays*scale[:,None]
    good &= (points[:,0]>-12)&(points[:,0]<65)&(abs(points[:,1])<12)
    return points[good]


class PaintAccumulator:
    """Motion-compensated paint candidates with bounded time and pose matching.

    Poses refer to calibrated rear axle, matched to scan capture time. Never
    substitutes reception-time pose or indefinitely retains invisible paint.
    """
    def __init__(self,max_age=.8,max_pose_skew=.04,max_points=15000):
        self.max_age=max_age;self.max_pose_skew=max_pose_skew
        self.max_points=max_points;self.frames=[];self.last_stamp=None

    def add(self,points,scan_stamp,pose_stamp,pose):
        p=np.asarray(points,float);x,y,yaw=pose
        if p.ndim!=2 or p.shape[1]<4 or not np.isfinite(p).all():raise ValueError('invalid paint points')
        if not all(math.isfinite(v) for v in (scan_stamp,pose_stamp,x,y,yaw)):
            raise ValueError('invalid paint pose/stamp')
        if abs(scan_stamp-pose_stamp)>self.max_pose_skew:raise ValueError('unmatched scan/pose')
        if self.last_stamp is not None and scan_stamp<=self.last_stamp:
            self.frames=[];self.last_stamp=None;raise ValueError('non-increasing capture stamp')
        world=p.copy();c,s=math.cos(yaw),math.sin(yaw)
        world[:,:2]=p[:,:2]@np.array([[c,s],[-s,c]])+[x,y]
        self.frames.append((scan_stamp,world));self.last_stamp=scan_stamp
        self.frames=[f for f in self.frames if scan_stamp-f[0]<=self.max_age]
        while len(self.frames)>1 and sum(len(f[1]) for f in self.frames)>self.max_points:self.frames.pop(0)

    def current(self,now,pose):
        if not math.isfinite(now) or not all(math.isfinite(v) for v in pose):raise ValueError('invalid current pose')
        self.frames=[f for f in self.frames if 0<=now-f[0]<=self.max_age]
        if not self.frames:return np.zeros((0,4))
        p=np.concatenate([f[1] for f in self.frames]);x,y,yaw=pose;c,s=math.cos(yaw),math.sin(yaw)
        p[:,:2]=(p[:,:2]-[x,y])@np.array([[c,-s],[s,c]])
        return p


def speed_for_clearance(distance, decel, latency):
    """Solve v*latency + v^2/(2*decel) <= available metres."""
    if not all(math.isfinite(v) for v in (distance, decel, latency)):
        raise ValueError('nonfinite braking input')
    if decel <= 0 or latency < 0:
        raise ValueError('invalid braking input')
    return max(0., math.sqrt((decel*latency)**2 + 2*decel*max(0.,distance))
               - decel*latency)


def signal_permission(detections, maneuver, roi, confidence=.7, min_width=0.):
    """Require unanimous relevant head states; ambiguity never releases a stop.

    ROI is camera-calibrated/configured, normalized [x0,y0,x1,y1]. A ROI alone
    does not establish relevance; the deployment gate requires recorded review.
    """
    if maneuver not in ('straight', 'left', 'right_green_only'):
        return False, 'unsupported maneuver'
    if not math.isfinite(min_width) or not 0 <= min_width < 1:
        return False, 'invalid head size bound'
    labels=[]
    for d in detections:
        try:
            x0,y0,x1,y1 = d['bbox_normalized']
            values=(x0,y0,x1,y1,float(d['confidence']))
            if not all(math.isfinite(v) for v in values):
                return False, 'invalid detection'
            if not (0<=x0<x1<=1 and 0<=y0<y1<=1):
                return False, 'invalid detection box'
            x,y=(x0+x1)/2,(y0+y1)/2
            if roi[0]<=x<=roi[2] and roi[1]<=y<=roi[3]:
                # A distant intersection can appear in the same image region.
                # Size is an additional surveyed bound, never head identity.
                if x1-x0 < min_width:
                    continue
                if d['confidence'] < confidence:
                    return False, 'uncertain relevant head'
                labels.append(d['label'])
        except (KeyError,TypeError,ValueError):
            return False, 'invalid detection'
    if not labels:
        return False, 'no relevant signal'
    allowed={'traffic_light_green','traffic_light_green_left'} if maneuver in ('straight','right_green_only') else {
        'traffic_light_left','traffic_light_green_left'}
    if len(set(labels)) != 1:
        return False, 'conflicting heads'
    return all(label in allowed for label in labels), 'signal='+labels[0]


class SignalLatch:
    """Fresh, consecutive observations required to release; stop is immediate."""
    def __init__(self, stable_seconds=.6, max_gap=.35):
        self.stable_seconds=stable_seconds;self.max_gap=max_gap
        self.since=None;self.last=None;self.key=None;self.frame=None;self.count=0

    def update(self, permitted, key, frame, now):
        if not permitted:
            self.since=None;self.last=None;self.key=None;self.frame=None;self.count=0
            return False
        # Do not count repeat handling of the same frame as independent evidence.
        if self.key != key or self.last is None or now-self.last>self.max_gap:
            self.since=now;self.count=0
        if frame != self.frame:
            self.last=now;self.frame=frame;self.key=key;self.count+=1
        return self.count>=3 and self.last-self.since>=self.stable_seconds and now-self.last<=self.max_gap


def fit_boundary(points, min_span=6., residual=.10):
    """Quadratic RANSAC with longitudinal coverage and no long unobserved gaps."""
    if len(points)<12:return None
    rng=np.random.RandomState(7);best=None;score=-1
    for _ in range(100):
        sample=points[rng.choice(len(points),3,replace=False)]
        if np.ptp(sample[:,0])<min_span*.6:continue
        design=np.c_[sample[:,0]**2,sample[:,0],np.ones(3)]
        if np.linalg.cond(design)>1e6:continue
        coeff=np.linalg.solve(design,sample[:,1])
        mask=np.abs(points[:,1]-np.polyval(coeff,points[:,0]))<residual
        inliers=points[mask]
        if len(inliers)<12 or np.ptp(inliers[:,0])<min_span:continue
        occupied=np.unique(np.floor(inliers[:,0]/1.5))
        if len(occupied)<5 or np.max(np.diff(occupied))>3:continue
        # Coverage matters more than density in a single scan ring.
        value=len(occupied)*100 + min(len(inliers),99)
        if value>score:best=inliers;score=value
    if best is None:return None
    coeff=np.polyfit(best[:,0],best[:,1],2)
    error=float(np.max(np.abs(best[:,1]-np.polyval(coeff,best[:,0]))))
    return {'coeff':coeff.tolist(),'x_min':float(best[:,0].min()),
            'x_max':float(best[:,0].max()),'uncertainty':max(.10,error),
            'inliers':len(best)}


def extract_scene(points, ground_z, center_coeff=(0.,0.,0.), intensity_floor=65.):
    """Extract both observed paint boundaries and transverse stop candidates.

    No fixed intensity IDs/semantic labels, synthetic lane width or one-sided
    inferred boundary. Candidates are still experimental until field reviewed.
    Input is rear-axle XYZ, raw intensity, optionally ring.
    """
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]<4:raise ValueError('XYZI required')
    p=p[np.isfinite(p).all(axis=1)]
    p=p[(p[:,0]>-12)&(p[:,0]<65)&(abs(p[:,1])<12)]
    empty={'valid':False,'reason':'missing two observed boundaries','stop_candidates':[],
           'obstacle_distance':None,'candidate_count':0}
    g=p[np.abs(p[:,2]-ground_z)<.35]
    if len(g)<100:return dict(empty,reason='insufficient ground')
    # Robust near-horizontal ground plane; reject a strongly sloped false fit.
    coeff=np.array([0.,0.,ground_z])
    for _ in range(5):
        mask=abs(g[:,2]-np.c_[g[:,:2],np.ones(len(g))]@coeff)<.10
        if mask.sum()<60:return dict(empty,reason='ground plane unavailable')
        coeff=np.linalg.lstsq(np.c_[g[mask,:2],np.ones(mask.sum())],g[mask,2],rcond=None)[0]
    if np.linalg.norm(coeff[:2])>.15:return dict(empty,reason='ground slope rejected')
    heights=p[:,2]-np.c_[p[:,:2],np.ones(len(p))]@coeff
    ground=p[abs(heights)<.07]
    # Elevation above ground is a conservative obstacle candidate, not a class.
    route_y=np.polyval(center_coeff,p[:,0]);obstacle=p[(heights>.20)&(heights<3.)&
                                                     (abs(p[:,1]-route_y)<1.4)&(p[:,0]>0)]
    distance=float(obstacle[:,0].min()) if len(obstacle)>=3 else None
    bright=ground[ground[:,3]>=intensity_floor]
    empty.update(obstacle_distance=distance,candidate_count=len(bright))
    empty['ground_valid']=True
    if len(bright)<24:return dict(empty,reason='insufficient paint candidates')
    offsets=bright[:,1]-np.polyval(center_coeff,bright[:,0])
    left=fit_boundary(bright[(offsets>.7)&(offsets<6.)])
    right=fit_boundary(bright[(offsets<-.7)&(offsets>-6.)])
    # Transverse paint is not automatically a verified stop line: road markings
    # can also span the corridor. Guard requires a reviewed stop-line association.
    stops=[]
    for bucket in np.unique(np.floor(bright[:,0]/.3)):
        row=bright[np.floor(bright[:,0]/.3)==bucket]
        if len(row)>=8 and np.ptp(row[:,1])>=2.5 and 0<row[:,0].mean()<35:
            stops.append(float(row[:,0].mean()))
    empty['stop_candidates']=stops
    if left is None or right is None:return empty
    # A dense outer marking must not hide a nearer solid/center marking.
    interior=bright[(bright[:,1]>np.polyval(right['coeff'],bright[:,0])+.25)&
                    (bright[:,1]<np.polyval(left['coeff'],bright[:,0])-.25)]
    if fit_boundary(interior) is not None:
        return dict(empty,reason='additional longitudinal paint inside candidate corridor')
    xmin=max(left['x_min'],right['x_min']);xmax=min(left['x_max'],right['x_max'])
    if xmax-xmin<6:return dict(empty,reason='insufficient shared coverage')
    x=np.linspace(xmin,xmax,30)
    width=np.polyval(left['coeff'],x)-np.polyval(right['coeff'],x)
    if np.min(width)<2.4 or np.max(width)>6 or np.ptp(width)>1.:
        return dict(empty,reason='implausible or branching boundary pair')
    return dict(empty,valid=True,reason='two paint boundary candidates',left=left,right=right,
                x_min=xmin,x_max=xmax,ground_plane=coeff.tolist())


def swept_clearance(scene, speed, steering, geometry, latency, decel, margin):
    """Check a sampled bicycle stopping trajectory of an enlarged body rectangle.

    Entire body (including wheel area) constrained, no unobserved extrapolation.
    This is a model check, not a dynamics/ground-truth safety proof.
    """
    if not scene.get('valid'):return False,'lane unavailable'
    if not all(math.isfinite(v) for v in (speed,steering,latency,decel,margin)):
        return False,'invalid motion'
    if speed<0 or decel<=0 or latency<0 or margin<0:return False,'invalid motion'
    stop=speed*latency+speed*speed/(2*decel)
    left,right=scene['left'],scene['right']
    inflate=margin+max(left['uncertainty'],right['uncertainty'])
    # Dense perimeter sampling also checks concave polynomial boundaries.
    gx=np.linspace(-geometry['rear_overhang'],geometry['front_extent'],20)
    gy=np.linspace(-geometry['half_width'],geometry['half_width'],10)
    body=np.r_[np.array([(x,y) for x in gx for y in (-geometry['half_width'],geometry['half_width'])]),
               np.array([(x,y) for x in (-geometry['rear_overhang'],geometry['front_extent']) for y in gy])]
    curvature=math.tan(steering)/geometry['wheelbase']
    for s in np.linspace(0,stop,max(2,int(stop/.15)+1)):
        yaw=s*curvature
        x=math.sin(yaw)/curvature if abs(curvature)>1e-8 else s
        y=(1-math.cos(yaw))/curvature if abs(curvature)>1e-8 else 0.
        q=body@np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])+[x,y]
        if q[:,0].min()<scene['x_min'] or q[:,0].max()>scene['x_max']:
            return False,'unobserved stopping footprint'
        upper=np.polyval(left['coeff'],q[:,0])-inflate
        lower=np.polyval(right['coeff'],q[:,0])+inflate
        if np.any(q[:,1]>=upper) or np.any(q[:,1]<=lower):return False,'predicted boundary contact'
    return True,'stopping footprint inside observed corridor'


def constrain_command(scene, speed, steering, geometry, traffic, detections,
                      approved, latency=.4, decel=3., margin=.25):
    """Return (permit, speed cap m/s, reason). Never assume missing semantics.

    traffic is a fresh, independently verified ahead-of-route context. Its
    stop_distance is rear-axle to stop line; front extent deliberately stops
    earlier than the front wheels. A false signal permission cannot release.
    No obstacle or blackout exception is automatically granted here.
    """
    if not approved:return False,0.,'calibration/perception validation incomplete'
    if not scene.get('valid'):return False,0.,scene.get('reason','lane unavailable')
    if not traffic or not traffic.get('association_verified'):
        return False,0.,'route signal/stop-line association unavailable'
    if traffic.get('state') not in ('clear','approach'):
        return False,0.,'unsupported traffic context'
    available=scene['x_max']-geometry['front_extent']-margin
    obstacle=scene.get('obstacle_distance')
    if obstacle is not None:
        available=min(available,obstacle-geometry['front_extent']-1.)
    if traffic['state']=='approach':
        distance=traffic.get('stop_distance')
        if not isinstance(distance,(float,int)) or not math.isfinite(distance) or distance<0:
            return False,0.,'invalid stop-line distance'
        if not traffic.get('release_stable',False):
            available=min(available,distance-geometry['front_extent']-1.)
    cap=speed_for_clearance(available,decel,latency)
    ok,reason=swept_clearance(scene,speed,steering,geometry,latency,decel,margin)
    if not ok:return False,0.,reason
    if available<=0:return False,0.,'stop clearance exhausted'
    return True,min(cap,58./3.6),'corridor and stopping clearance available'
