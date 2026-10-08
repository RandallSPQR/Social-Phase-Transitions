# Control: symmetric ties with the SAME average loyalty as the non-reciprocal balanced runs.
# Redrawn reverse ties have mean 0, so average directed coupling = 1 - eps/2  <=>  symmetric rho = eps/4.
import numpy as np
from multiprocessing import Pool
from vbsim import make_graph
from asym import run_dyn
def job(a):
    N,rho,T,seed=a; g=make_graph(N,4,rho,np.random.default_rng(seed)); return run_dyn(*g,1/T,2000,2000,seed+3)
if __name__=="__main__":
    with Pool(2) as p:
        for T in [2.0,3.0]:
            for eps in [0,0.2,0.5,1.0]:
                r=np.array(p.map(job,[(2000,eps/4,T,777+s) for s in range(20)]))
                print(f"T={T} eps-matched rho={eps/4}: |m|={r[:,0].mean():.3f}+-{r[:,0].std(ddof=1)/np.sqrt(20):.3f} persist={r[:,4].mean():.3f}",flush=True)
