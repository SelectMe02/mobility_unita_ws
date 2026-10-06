"""Observed LiDAR corridor occupancy, not semantic or occlusion-free truth."""
import numpy as np


def corridor_observation(points, route, half_width=2.0, horizon=20.0):
    if not np.isfinite(horizon) or horizon<12 or horizon>32:
        raise ValueError('invalid observation horizon')
    p=np.asarray(points,float);r=np.asarray(route,float)
    if p.ndim!=2 or p.shape[1]<3 or r.ndim!=2 or r.shape[1]!=2 or len(r)<3:
        raise ValueError('invalid cloud/route shape')
    if not np.isfinite(p[:,:3]).all() or not np.isfinite(r).all():
        raise ValueError('nonfinite cloud/route')
    p=p[(np.linalg.norm(p[:,:2],axis=1)<45)&(p[:,0]>-8)]
    ground=p[(p[:,0]>3)&(p[:,0]<22)&(abs(p[:,1])<8)&(abs(p[:,2]+.344)<2.)]
    if len(ground)<100:return dict(valid=False,clear=False,reason='ground unavailable')
    design=np.c_[ground[:,:2],np.ones(len(ground))]
    # Acceleration changes body pitch. Fit a bounded robust plane rather
    # than selecting inliers against a fixed horizontal seed.
    rng=np.random.RandomState(0);best=None;best_count=0
    for _ in range(64):
        sample=rng.choice(len(ground),3,replace=False)
        try:candidate=np.linalg.solve(design[sample],ground[sample,2])
        except np.linalg.LinAlgError:continue
        if np.linalg.norm(candidate[:2])>.15 or abs(candidate[2]+.344)>.4:continue
        count=int(np.sum(abs(ground[:,2]-design@candidate)<.12))
        if count>best_count:best=candidate;best_count=count
    if best is None or best_count<80:return dict(valid=False,clear=False,reason='ground fit unavailable')
    coeff=best
    for _ in range(3):
        mask=abs(ground[:,2]-design@coeff)<.12
        if mask.sum()<80:return dict(valid=False,clear=False,reason='ground fit unavailable')
        coeff=np.linalg.lstsq(design[mask],ground[mask,2],rcond=None)[0]
    if np.linalg.norm(coeff[:2])>.15 or abs(coeff[2]+.344)>.4:
        return dict(valid=False,clear=False,reason='ground slope rejected')
    height=p[:,2]-np.c_[p[:,:2],np.ones(len(p))]@coeff
    # Consecutive duplicate waypoints occur at surveyed link joins.
    r=r[np.r_[True,np.linalg.norm(np.diff(r,axis=0),axis=1)>1e-5]]
    if len(r)<3:return dict(valid=False,clear=False,reason='route coverage invalid')
    delta=np.diff(r,axis=0);length=np.linalg.norm(delta,axis=1)
    s=np.r_[0.,np.cumsum(length)]
    if s[-1]<15 or np.any(length<1e-5):return dict(valid=False,clear=False,reason='route coverage invalid')
    offsets=p[:,:2,None]-r[:-1].T[None,:,:]
    # N x segments x 2; nearest finite segment supplies along-route distance.
    offsets=offsets.transpose(0,2,1)
    t=np.clip(np.sum(offsets*delta[None,:,:],axis=2)/(length**2)[None,:],0,1)
    error=np.linalg.norm(offsets-t[:,:,None]*delta[None,:,:],axis=2)
    nearest=error.argmin(axis=1);idx=np.arange(len(p))
    along=s[nearest]+t[idx,nearest]*length[nearest]
    near=(error[idx,nearest]<half_width)&(along>4)&(along<min(horizon,s[-1]))
    # The measured vertical scan leaves gaps between ground rings.
    # Require ground in each 8 m band, plus the final partial band. Empty data
    # cannot be turned into permission by an absence of obstacle returns.
    ends=np.arange(4,min(horizon,s[-1]),8.).tolist()+[min(horizon,s[-1])]
    covered=[int(np.sum(near&(abs(height)<.10)&(along>=a)&(along<b))) for a,b in zip(ends[:-1],ends[1:])]
    if not covered or min(covered)<8:return dict(valid=False,clear=False,reason='unobserved route ground',coverage=covered)
    obstacle=near&(height>.20)&(height<3.5)
    # A single elevated return is enough to hold; no class/size exemption.
    distance=float(np.min(along[obstacle])) if obstacle.any() else None
    return dict(valid=True,clear=distance is None,reason='observed corridor occupied' if distance is not None else 'observed corridor clear',
                obstacle_distance=distance,ground_plane=coeff.tolist(),coverage=covered)
