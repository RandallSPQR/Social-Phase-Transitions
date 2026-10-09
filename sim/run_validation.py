import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, time
import numpy as np
from multiprocessing import Pool
from vbsim import make_graph, run_sample

OUT = _os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)


def job(args):
    N, c, rho, Ts, n_pow, seed = args
    rng = np.random.default_rng(seed)
    g = make_graph(N, c, rho, rng)
    betas = 1.0 / np.asarray(Ts)[::-1]          # ascending beta
    acc, cnt, sw = run_sample(*g, betas, n_pow, seed + 7)
    return acc[:, ::-1, :], cnt, sw[::-1]      # back to ascending T


def run(tag, N, c, rho, Tlo, Thi, K, n_pow, S, base_seed):
    fn = f"{OUT}/{tag}_N{N}.npz"
    if os.path.exists(fn):
        return
    Ts = np.linspace(Tlo, Thi, K)
    args = [(N, c, rho, Ts, n_pow, base_seed + 1000 * s) for s in range(S)]
    t0 = time.time()
    with Pool(2) as p:
        res = p.map(job, args, chunksize=1)
    acc = np.stack([r[0] for r in res])   # [S, bins, K, obs]
    cnt = res[0][1]
    sw = np.mean([r[2] for r in res], axis=0)
    np.savez(fn, acc=acc, cnt=cnt, Ts=Ts, swap=sw, N=N, c=c, rho=rho)
    print(f"{tag} N={N} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    plan = []
    # 1. Spin-glass transition, rho = 0.5, c = 4 (predicted T_SG = 1.820)
    for N, npow, K, S in [(250, 12, 16, 400), (500, 12, 16, 300), (1000, 13, 16, 200), (2000, 13, 20, 160)]:
        plan.append(("sg_c4", N, 4, 0.5, 1.3, 2.6, K, npow, S, 11))
    # 2. Ferromagnetic transitions, c = 4
    for rho, lo, hi, tag in [(0.0, 3.0, 5.0, "f_r00"), (0.1, 2.4, 3.9, "f_r10"), (0.2, 1.7, 2.9, "f_r20")]:
        for N, npow, S in [(250, 11, 150), (500, 11, 120), (1000, 12, 100), (2000, 12, 80)]:
            plan.append((tag, N, 4, rho, lo, hi, 16, npow, S, 22))
    # 3. Spin-glass line vs connectivity: c = 3 (T_SG=1.519), c = 6 (T_SG=2.307)
    for c, lo, hi in [(3, 1.1, 2.2), (6, 1.7, 3.2)]:
        for N, npow, K, S in [(500, 12, 16, 200), (1000, 13, 16, 150), (2000, 13, 20, 100)]:
            plan.append((f"sg_c{c}", N, c, 0.5, lo, hi, K, npow, S, 33))
    for p in plan:
        run(*p)
    print("ALL DONE", flush=True)
