"""Deterministic constraint-aware NSGA-II survival, all objectives minimized."""
import numpy as np

def select(F, npop, G=None, n_partitions=None):
    F=np.asarray(F,dtype=float)
    if F.ndim!=2 or not 0<npop<=len(F): raise ValueError('Invalid population shape or size')
    F=np.where(np.isfinite(F),F,np.inf)
    violation=np.zeros(len(F)) if G is None else np.maximum(np.asarray(G,dtype=float),0).sum(axis=1)
    violation=np.where(np.isfinite(violation),violation,np.inf)
    remaining=list(range(len(F))); chosen=[]
    while remaining and len(chosen)<npop:
        front=[]
        for i in remaining:
            dominated=any((violation[j]<violation[i]) or (violation[j]==violation[i] and np.all(F[j]<=F[i]) and np.any(F[j]<F[i])) for j in remaining if j!=i)
            if not dominated: front.append(i)
        if len(chosen)+len(front)<=npop: chosen.extend(front)
        else:
            distance={i:0.0 for i in front}
            for col in range(F.shape[1]):
                order=sorted(front,key=lambda i:(F[i,col],i))
                width=F[order[-1],col]-F[order[0],col]
                if not np.isfinite(width) or width<=0: continue
                distance[order[0]]=distance[order[-1]]=np.inf
                for k in range(1,len(order)-1): distance[order[k]]+=(F[order[k+1],col]-F[order[k-1],col])/width
            chosen.extend(sorted(front,key=lambda i:(-distance[i],i))[:npop-len(chosen)])
        removed=set(front); remaining=[i for i in remaining if i not in removed]
    return np.asarray(chosen,dtype=np.int32)
