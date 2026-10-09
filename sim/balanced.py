# Two-party polarization = structurally balanced signed network: + ties inside a party, - ties across.
# Gauge transform s_i -> tau_i s_i maps it exactly onto a ferromagnet, so "polarization" = |m| in the
# party frame. Non-reciprocity: reverse direction redrawn with prob eps as a coin-flip sign.
import numpy as np
from multiprocessing import Pool
from asym import run_dyn
def graph(N,c,eps,rng):
    M=rng.poisson(c*N/2); a=rng.integers(0,N,M); b=rng.integers(0,N,M); k=a!=b; a,b=a[k],b[k]
    lo,hi=np.minimum(a,b),np.maximum(a,b); key=np.unique(lo.astype(np.int64)*N+hi)
    lo,hi=(key//N).astype(np.int64),(key%N).astype(np.int64); E=len(lo)
    Jf=np.ones(E,np.int64)                      # gauge frame: balanced => all +
    Jb=np.where(rng.random(E)<eps, np.where(rng.random(E)<0.5,-1,1), Jf)
    src=np.concatenate([hi,lo]); dst=np.concatenate([lo,hi]); J=np.concatenate([Jf,Jb])
    o=np.argsort(src,kind="stable"); src,dst,J=src[o],dst[o],J[o]
    ip=np.zeros(N+1,np.int64); np.add.at(ip,src+1,1); return np.cumsum(ip),dst,J
def job(a):
    N,eps,T,seed=a; g=graph(N,4,eps,np.random.default_rng(seed)); return run_dyn(*g,1/T,2000,2000,seed+3)
if __name__=="__main__":
    with Pool(2) as p:
        for T in [2.0,3.0]:
            for eps in [0,0.2,0.5,1.0]:
                r=np.array(p.map(job,[(2000,eps,T,777+s) for s in range(20)]))
                print(f"T={T} eps={eps}: polarization |m|={r[:,0].mean():.3f}+-{r[:,0].std(ddof=1)/np.sqrt(20):.3f} persist={r[:,4].mean():.3f}",flush=True)
