import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, time, numpy as np
from multiprocessing import Pool
from asym import make_graph_asym, run_dyn
OUT=_os.path.join(ROOT, "data", "asym"); os.makedirs(OUT, exist_ok=True)
def job(a):
    N,c,rho,eps,T,seed=a
    rng=np.random.default_rng(seed)
    g=make_graph_asym(N,c,rho,eps,rng)
    return run_dyn(*g,1.0/T,2000,2000,seed+5)
if __name__=="__main__":
    EPS=[0,0.05,0.1,0.2,0.35,0.5,0.75,1.0]
    plan=[]
    for N,S in [(1000,40),(2000,30),(4000,16)]:
        for T in [1.0,1.3,1.6,2.4]: plan+= [(N,4,0.5,e,T,S) for e in EPS]
        for T in [2.0]: plan+= [(N,4,0.1,e,T,S) for e in EPS]
    rows=[]
    t0=time.time()
    with Pool(2) as p:
        for (N,c,rho,e,T,S) in plan:
            res=p.map(job,[(N,c,rho,e,T,90000+s*13+int(e*1000)+int(T*100)) for s in range(S)])
            rows.append((N,c,rho,e,T,np.array(res)))
            print(f"N={N} rho={rho} T={T} eps={e}: persist={np.mean([r[4] for r in res]):.3f} |m|={np.mean([r[0] for r in res]):.3f} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/asym.npy",np.array(rows,dtype=object),allow_pickle=True)
    print("ALL DONE",flush=True)
