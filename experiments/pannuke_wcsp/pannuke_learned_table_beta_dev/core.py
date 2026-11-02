"""Standalone six-label pixel WCSP. No imports from earlier experiments."""
import math
import numpy as np
from numba import njit

CLASSES = ['Background','Neoplastic','Inflammatory','Connective','Dead','Non-Neoplastic Epithelial']
OFFSETS = np.array([(dy,dx) for dy in range(-2,3) for dx in range(-2,3) if (dy,dx)!=(0,0)],dtype=np.int64)
HALF = [(int(dy),int(dx)) for dy,dx in OFFSETS if dy>0 or (dy==0 and dx>0)]
WEIGHTS = np.array([1/math.hypot(dy,dx) for dy,dx in OFFSETS])

def semantic(mask):
    assert mask.shape == (256,256,6)
    crop = mask[46:210,46:210,:5]>0
    # Ascending channels overwrite: highest biological class wins overlaps.
    out = np.zeros((164,164),np.uint8)
    for k in range(5): out[crop[...,k]]=k+1
    return out, int((crop.sum(axis=-1)>1).sum())

def pairs(labels,dy,dx):
    h,w=labels.shape
    return (labels[max(0,dy):min(h,h+dy),max(0,dx):min(w,w+dx)],
            labels[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)])

def adjacency(labels):
    counts=np.zeros((6,6),np.int64)
    for dy,dx in HALF:
        a,b=pairs(labels,dy,dx)
        n=np.bincount((a.astype(int)*6+b).ravel(),minlength=36).reshape(6,6)
        counts+=n+n.T
    return counts

def normalize(c):
    mean=float(c[~np.eye(len(c),dtype=bool)].mean())
    if mean<=0: raise ValueError('Degenerate zero off-diagonal costs')
    return c/mean,mean

def fit(counts):
    joint=(counts+1)/(counts+1).sum()
    marginal=joint.sum(1)
    raw=np.maximum(0,-np.log(joint/np.outer(marginal,marginal)))
    np.fill_diagonal(raw,0)
    data,factor=normalize(raw)
    bio=np.array([[0,1,1,1,10],[1,0,2,1,6],[1,2,0,4,1],[1,1,4,0,10],[10,6,1,10,0]],float)
    bio,bfactor=normalize(bio)
    prior=data.copy(); prior[1:,1:]=bio
    blend=.8*data+.2*prior
    final,ffactor=normalize(blend)
    return dict(counts=counts,joint=joint,marginals=marginal,raw_cost=raw,data_cost=data,
                biological_prior=bio,prior=prior,unnormalized_blend=blend,final=final,
                normalization=dict(data_off_diagonal_mean=factor,biological_off_diagonal_mean=bfactor,final_off_diagonal_mean=ffactor))

def probabilities(logits):
    assert logits.shape==(164,164,6) and np.isfinite(logits).all()
    z=logits.astype(np.float64)-logits.max(axis=-1,keepdims=True)
    p=np.exp(z); p/=p.sum(axis=-1,keepdims=True)
    validate_probabilities(p)
    return p

def validate_probabilities(p):
    assert p.shape==(164,164,6)
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    assert np.allclose(p.sum(-1),1,rtol=0,atol=1e-7)

def unary(p):
    u=-np.log(np.maximum(p,1e-12)); return u-u.min(axis=-1,keepdims=True)

def energy(y,u,c,beta):
    total=float(np.take_along_axis(u,y[...,None],axis=-1).sum())
    for dy,dx in HALF:
        a,b=pairs(y,dy,dx)
        total+=beta/math.hypot(dy,dx)*c[a,b].sum()
    return float(total)

@njit(cache=True)
def sweep(y,u,c,beta,offsets,weights):
    h,w=y.shape; changes=0; delta=0.
    for row in range(h):
        for col in range(w):
            costs=u[row,col].copy()
            for n in range(len(offsets)):
                r=row+offsets[n,0]; s=col+offsets[n,1]
                if 0<=r<h and 0<=s<w:
                    for k in range(6): costs[k]+=beta*weights[n]*c[k,y[r,s]]
            old=y[row,col]; best=old
            for k in range(6):
                if costs[k]<costs[best]: best=k
            if costs[best]<costs[old]:
                y[row,col]=best; changes+=1; delta+=costs[best]-costs[old]
    return changes,delta

def improving_moves(y,u,c,beta):
    """Independent vectorized full-grid conditional energies; no solver sweep reuse."""
    local=u.copy()
    for dy,dx in OFFSETS:
        dy=int(dy); dx=int(dx); h,w=y.shape
        target=(slice(max(0,dy),min(h,h+dy)),slice(max(0,dx),min(w,w+dx)))
        neighbor=y[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]
        local[target]+=beta/math.hypot(dy,dx)*c[:,neighbor].transpose(1,2,0)
    current=np.take_along_axis(local,y[...,None],axis=-1)[...,0]
    gain=current-local.min(-1)
    # Floating-point audit tolerance is separate from strict solver acceptance.
    return int((gain>1e-10).sum()),float(gain.max()),int((gain>0).sum())

def solve(p,c,beta=.30,max_rounds=100):
    u=unary(p); y=p.argmax(-1).astype(np.uint8)
    energies=[energy(y,u,c,beta)]; rounds=[]
    for r in range(max_rounds):
        n,delta=sweep(y,u,c,beta,OFFSETS,WEIGHTS)
        e=energy(y,u,c,beta)
        assert e<=energies[-1]+1e-7
        assert abs((e-energies[-1])-delta)<1e-6
        energies.append(e); rounds.append(int(n))
        if n==0: break
    n,g,strict=improving_moves(y,u,c,beta)
    return y,dict(rounds=len(rounds),changes_per_round=rounds,energy=energies,
        nonincreasing_energy=True,remaining_improving_pixels=n,max_remaining_gain=g,
        strict_floating_point_candidates=strict,verified_local_minimum=n==0,
        audit_tolerance=1e-10,stop_reason='no_changes' if rounds[-1]==0 else 'round_limit',
        global_optimum_claimed=False)

def confusion(gt,pred):
    return np.bincount((gt.astype(int)*6+pred).ravel(),minlength=36).reshape(6,6)

def metrics(cm):
    cm=np.asarray(cm); tp=np.diag(cm); union=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,union,out=np.zeros(6,float),where=union>0)
    return dict(iou=iou.tolist(),biological_miou=float(iou[1:].mean()),six_class_miou=float(iou.mean()),
                pixel_accuracy=float(tp.sum()/cm.sum()),gt_absent_classes=np.where(cm.sum(1)==0)[0].tolist(),
                zero_union_classes=np.where(union==0)[0].tolist())

def bootstrap(a,b,seed=20261102,replicates=5000):
    rng=np.random.default_rng(seed); values=[]
    for _ in range(replicates):
        ids=rng.integers(0,len(a),len(a))
        values.append(metrics(b[ids].sum(0))['biological_miou']-metrics(a[ids].sum(0))['biological_miou'])
    return dict(seed=seed,replicates=replicates,method='paired patch percentile; pool sampled confusion matrices before IoU',
                ci95=np.percentile(values,[2.5,97.5]).tolist(),replicate_differences=values)
