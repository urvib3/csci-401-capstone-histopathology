"""Six-level soft WCSP; scalar energy with explicit family/scale costs."""
import numpy as np
from numba import njit

SIZES = [1, 2, 4, 8, 16, 32]
CLASSES = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Non-Neoplastic Epithelial']
PRIOR = np.array([[0,1,1,1,10],[1,0,2,1,6],[1,2,0,4,1],[1,1,4,0,10],[10,6,1,10,0]], float)

def normalize(a):
    m = a[~np.eye(len(a), dtype=bool)].mean()
    return a/m if m > 0 else a.copy()

def majority(gt, size):
    h,w=gt.shape
    c=np.stack([(gt==k).reshape(h//size,size,w//size,size).sum((1,3)) for k in range(6)],-1)
    return c.argmax(-1).astype(np.uint8)

def counts(y):
    a=np.zeros((6,6), np.int64)
    for x,z in [(y[:-1],y[1:]),(y[:,:-1],y[:,1:])]:
        n=np.bincount((x.astype(int)*6+z).ravel(),minlength=36).reshape(6,6)
        a+=n+n.T
    return a

def fit_table(n):
    j=(n+1)/(n+1).sum(); m=j.sum(1)
    d=np.maximum(0,-np.log(j/np.outer(m,m)));np.fill_diagonal(d,0);d=normalize(d)
    prior=d.copy();prior[1:,1:]=normalize(PRIOR)
    return normalize(.8*d+.2*prior),dict(counts=n.tolist(),joint=j.tolist(),marginals=m.tolist(),learned=d.tolist(),prior=prior.tolist())

def block_probabilities(p):
    h,w,k=p.shape
    return [p.reshape(h//s,s,w//s,s,k).mean((1,3)) for s in SIZES]

def graph(side=256):
    shapes=[side//s for s in SIZES]; offsets=np.cumsum([0]+[n*n for n in shapes])
    aa=[];bb=[];tt=[]; edge_counts=[]
    for level,n in enumerate(shapes):
        ids=np.arange(offsets[level],offsets[level+1]).reshape(n,n)
        a=np.concatenate([ids[:-1].ravel(),ids[:,:-1].ravel()]);b=np.concatenate([ids[1:].ravel(),ids[:,1:].ravel()])
        aa.append(a);bb.append(b);tt.append(np.full(len(a),level));pc=0
        if level<5:
            parents=np.arange(offsets[level+1],offsets[level+2]).reshape(n//2,n//2).repeat(2,0).repeat(2,1)
            aa.append(ids.ravel());bb.append(parents.ravel());tt.append(np.full(n*n,6+level));pc=n*n
        edge_counts.append(dict(level=level+1,block_size=SIZES[level],variables=n*n,sibling_edges=len(a),parent_child_edges=pc))
    a=np.concatenate(aa);b=np.concatenate(bb);t=np.concatenate(tt)
    src=np.r_[a,b];dst=np.r_[b,a];typ=np.r_[t,t];order=np.argsort(src,kind='stable')
    indptr=np.r_[0,np.cumsum(np.bincount(src,minlength=offsets[-1]))]
    return dict(a=a,b=b,t=t,ptr=indptr,dst=dst[order],typ=typ[order],offsets=offsets,counts=edge_counts)

def costs(tables,sibling,parent):
    potts=1-np.eye(6)
    # At a given scale the maximum grid degree is four. Divide by four so
    # sibling strength controls a per-node budget rather than one edge.
    return np.stack([sibling*np.asarray(x)/4 for x in tables]+[parent*potts for _ in range(5)])

def initial_unary(p, gamma=0):
    ps=block_probabilities(p)
    y=np.concatenate([x.argmax(-1).ravel() for x in ps]).astype(np.uint8)
    u=np.concatenate([(-np.log(np.maximum(x,1e-12))).reshape(-1,6)*(1 if i==0 else gamma) for i,x in enumerate(ps)])
    u-=u.min(1,keepdims=True)
    return y,u

def energy(y,u,c,g):
    return float(u[np.arange(len(y)),y].sum()+c[g['t'],y[g['a']],y[g['b']]].sum())

@njit(cache=True)
def sweep(y,u,c,ptr,dst,typ,reverse):
    changes=0
    for index in range(len(y)):
        i=len(y)-1-index if reverse else index
        v=u[i].copy()
        for e in range(ptr[i],ptr[i+1]):
            for k in range(6):v[k]+=c[typ[e],k,y[dst[e]]]
        old=y[i];best=old
        for k in range(6):
            if v[k]<v[best]-1e-12:best=k
        if best!=old:y[i]=best;changes+=1
    return changes

def audit(y,u,c,g):
    # Independent edge-list accumulation, rather than reusing CSR sweep.
    v=u.copy()
    for k in range(6):
        np.add.at(v[:,k],g['a'],c[g['t'],k,y[g['b']]])
        np.add.at(v[:,k],g['b'],c[g['t'],y[g['a']],k])
    gain=v[np.arange(len(y)),y]-v.min(1)
    return dict(remaining_improving_variables=int((gain>1e-9).sum()),max_remaining_gain=float(gain.max()))

def solve(p,tables,g,sibling,parent,gamma=0,max_rounds=100):
    import time
    start=time.perf_counter();y,u=initial_unary(p,gamma);init=y.copy();c=costs(tables,sibling,parent)
    energies=[energy(y,u,c,g)];changes=[]
    # Each iteration includes both directions, allowing bottom-up and top-down influence.
    for _ in range(max_rounds):
        n=sweep(y,u,c,g['ptr'],g['dst'],g['typ'],False)+sweep(y,u,c,g['ptr'],g['dst'],g['typ'],True)
        e=energy(y,u,c,g);assert e<=energies[-1]+1e-7
        energies.append(e);changes.append(n)
        if n==0:break
    check=audit(y,u,c,g)
    levels=[]
    for l in range(6):
        sl=slice(g['offsets'][l],g['offsets'][l+1]);mask=g['t']==6+l
        levels.append(dict(level=l+1,changed=int((y[sl]!=init[sl]).sum()),variables=int(g['offsets'][l+1]-g['offsets'][l]),transitions=confusion(init[sl],y[sl]).tolist(),parent_disagreement_before=int((init[g['a'][mask]]!=init[g['b'][mask]]).sum()),parent_disagreement_after=int((y[g['a'][mask]]!=y[g['b'][mask]]).sum())))
    return y,dict(rounds=len(changes),changes_per_round=changes,energy=energies,stop='no_changes' if changes[-1]==0 else 'round_limit',verified_local_minimum=check['remaining_improving_variables']==0,**check,levels=levels,seconds=time.perf_counter()-start)

def confusion(a,b):
    return np.bincount((a.astype(np.int64)*6+b).ravel(),minlength=36).reshape(6,6)

def metrics(cm):
    tp=np.diag(cm);union=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,union,out=np.zeros(6,float),where=union>0)
    return dict(iou=iou.tolist(),biological_miou=float(iou[1:].mean()),six_class_miou=float(iou.mean()),pixel_accuracy=float(tp.sum()/cm.sum()))

def bootstrap(a,b,seed=20261109):
    rng=np.random.default_rng(seed);ds=[]
    for _ in range(5000):
        ix=rng.integers(len(a),size=len(a));ds.append(metrics(b[ix].sum(0))['biological_miou']-metrics(a[ix].sum(0))['biological_miou'])
    return np.percentile(ds,[2.5,97.5]).tolist()
