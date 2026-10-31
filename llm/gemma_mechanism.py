"""Exploratory mechanism test for Gemma-31B's frozen, start-dependent regime (plan: llm/GEMMA_MECHANISM_PLAN.md).

  python llm/gemma_mechanism.py

Simulation only. Outputs: results/llm/confirm/gemma_mechanism/ (per-unit csv.gz, summary.json).
"""
import json, os, sys, time, math
from multiprocessing import Pool
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2, surrogate as S, kernel_test as K, amplification_curve as A, confirm_sim as C

OUT = os.path.join(K.ROOT, "results", "llm", "confirm", "gemma_mechanism")
TS = [2.0, 2.163, 2.3392]
NS = [100, 400, 1600]
R = 50
seed = A.seed


def decode(i):
    s0 = 1 if i % 2 else -1; c = i // 2
    return (c // 125, (c // 25) % 5, (c // 5) % 5, c % 5), s0


def rival_knockout(P):
    Q = P.copy()
    for c, s0 in K.configs(4, 0.25):
        aP, aQ, rP, rQ = c; nr = rP + rQ
        w = [math.comb(nr, x) / 2 ** nr for x in range(nr + 1)]
        Q[K.idx(c, s0)] = sum(wx * P[K.idx((aP, aQ, x, nr - x), s0)] for x, wx in zip(range(nr + 1), w))
    return Q


def additive(P):
    rows = [(K.idx(c, s0), c, s0) for c, s0 in K.configs(4, 0.25)]
    X = np.array([[1.0, c[0] - c[1], c[2] - c[3], s0] for _, c, s0 in rows]); y = np.array([P[i] for i, _, _ in rows])
    b = K.logit_fit(X, y, np.ones(len(y)))
    Q = np.full(K.NT, np.nan)
    for (i, c, s0), x in zip(rows, X):
        Q[i] = 1 / (1 + np.exp(-x @ b))
    return Q, b


def table(kernel, T):
    P = C.table("gemma", T)
    if kernel == "original":
        return P, None
    if kernel == "rival_ko":
        return rival_knockout(P), None
    return additive(P)


def logits(P):
    return np.clip(A.lg(np.nan_to_num(P, nan=0.5)), -A.CLIP, A.CLIP)


def unit_len(args):
    kernel, N, T, W, start = args
    path = os.path.join(OUT, "len", f"{kernel}_N{N}_T{T:.4f}_W{W}_{start}.csv.gz")
    if os.path.exists(path):
        return path
    G, _ = C.graphs("gemma", N); L = logits(table(kernel, T)[0]); rows = []
    for r in range(R):
        res, _ = A.sim_eq(*G[r % 5], L, seed("gemma-mech", N, T, W, start, r), W, W, A.PERS_WIN, start == "c")
        rows.append(dict(run=r, **dict(zip(A.MEAS, res))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False, compression="gzip"); os.replace(path + ".tmp", path)
    return path


@njit(cache=True)
def run_glass(ip, nb, tie, L, start, dyn_seed, W):
    """W warm-up + W measured sweeps from `start`; returns (time-avg |m|, time-avg unsatisfied fraction), final state."""
    n = ip.shape[0] - 1; s = start.copy(); np.random.seed(dyn_seed); P = 1.0 / (1.0 + np.exp(-L))
    am = 0.0; un = 0.0
    for t in range(2 * W):
        for _u in range(n):
            i = np.random.randint(n); u = np.random.random(); c = 0
            for q in range(ip[i], ip[i + 1]):
                if tie[q] > 0:
                    c += 125 if s[nb[q]] > 0 else 25
                else:
                    c += 5 if s[nb[q]] > 0 else 1
            c = c * 2 + (1 if s[i] > 0 else 0)
            s[i] = 1 if u < P[c] else -1
        if t >= W:
            m = 0.0; x = 0.0
            for i in range(n):
                m += s[i]
                for e in range(ip[i], ip[i + 1]):
                    agree = s[i] == s[nb[e]]
                    if (tie[e] > 0 and not agree) or (tie[e] < 0 and agree):
                        x += 1.0
            am += abs(m / n); un += x / ip[n]
    return am / W, un / W, s


def unit_glass(args):
    N, T = args
    path = os.path.join(OUT, "glass", f"N{N}_T{T:.4f}.csv.gz")
    if os.path.exists(path):
        return path
    G, _ = C.graphs("gemma", N); L = logits(table("original", T)[0]); rows = []
    for r in range(R):
        g = G[r % 5]
        st = np.where(np.random.default_rng(seed("gemma-glass-start", N, T, r)).random(N) < 0.5, 1, -1).astype(np.int64)
        st2 = np.where(np.random.default_rng(seed("gemma-glass-start2", N, T, r)).random(N) < 0.5, 1, -1).astype(np.int64)
        ma, ua, sa = run_glass(*g, L, st, seed("gemma-glass-dyn-a", N, T, r), 1000)
        mb, ub, sb = run_glass(*g, L, st, seed("gemma-glass-dyn-b", N, T, r), 1000)
        mc, uc, sc = run_glass(*g, L, st2, seed("gemma-glass-dyn-c", N, T, r), 1000)
        fl = stage2.S.run(*g, np.zeros(N, np.int64), S.ising_row(6.0)[None, :], 200, 50, seed("gemma-floor", N, r % 5), False, 0.0)[7]
        rows.append(dict(run=r, graph=r % 5, abs_m_a=ma, abs_m_b=mb, abs_m_c=mc, unsat_a=ua, unsat_b=ub, floor=fl,
                         q_same_start=abs(float((sa * sb).mean())), q_indep_start=abs(float((sa * sc).mean()))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False, compression="gzip"); os.replace(path + ".tmp", path)
    return path


def summarise():
    out = {"label": "exploratory (not confirmatory); plan llm/GEMMA_MECHANISM_PLAN.md"}
    co = {}
    for T in TS:
        P = C.table("gemma", T); _, b = additive(P); co[str(T)] = dict(zip(["intercept", "ally", "rival", "own"], map(float, b)))
    out["additive_coefficients"] = co
    rows = []
    for kernel in ("original", "rival_ko", "additive"):
        for N in NS:
            for T in TS:
                for W in (1000, 4000):
                    h = pd.read_csv(os.path.join(OUT, "len", f"{kernel}_N{N}_T{T:.4f}_W{W}_h.csv.gz")).abs_m.mean()
                    c = pd.read_csv(os.path.join(OUT, "len", f"{kernel}_N{N}_T{T:.4f}_W{W}_c.csv.gz")).abs_m.mean()
                    rows.append(dict(kernel=kernel, N=N, T=T, W=W, hot=float(h), cold=float(c), gap=float(abs(h - c))))
    D = pd.DataFrame(rows); out["runlength"] = rows
    persist = {}
    for kernel in ("original", "rival_ko", "additive"):
        g = D[(D.kernel == kernel) & (D.N == 1600)].pivot_table(index="T", columns="W", values="gap")
        per_T = {str(T): bool(g.loc[T, 4000] > 0.02 and g.loc[T, 4000] >= 0.5 * g.loc[T, 1000]) for T in TS}
        persist[kernel] = dict(per_T=per_T, any=any(per_T.values()), gaps_1600={str(T): [float(g.loc[T, 1000]), float(g.loc[T, 4000])] for T in TS})
    out["persistence"] = persist
    ko_rival_removes = not persist["rival_ko"]["any"]; ko_add_removes = not persist["additive"]["any"]
    if not persist["original"]["any"]:
        reading = "original kernel does not reproduce persistence at 4x in this rerun; knockouts read against that"
    elif ko_rival_removes and not ko_add_removes:
        reading = "frustration-driven (rival knockout removes start-dependence; saturation knockout does not)"
    elif ko_add_removes and not ko_rival_removes:
        reading = "higher-order (saturation)-driven (saturation knockout removes it; rival knockout does not)"
    elif ko_rival_removes and ko_add_removes:
        reading = "both knockouts remove it: no single mechanism; each ingredient necessary"
    else:
        reading = "neither knockout removes it: not frustration or saturation alone; mechanism not identified (inertia/other untested)"
    out["reading"] = reading
    G = pd.concat([pd.read_csv(os.path.join(OUT, "glass", f"N{N}_T{T:.4f}.csv.gz")).assign(N=N, T=T) for N in (400, 1600) for T in TS])
    G["excess_unsat"] = 0.5 * (G.unsat_a + G.unsat_b) - G.floor
    out["glass"] = G.groupby(["N", "T"]).agg(abs_m=("abs_m_a", "mean"), q_same_start=("q_same_start", "mean"), q_indep_start=("q_indep_start", "mean"),
                                           unsat=("unsat_a", "mean"), floor=("floor", "mean"), excess_unsat=("excess_unsat", "mean"),
                                           excess_unsat_sd=("excess_unsat", "std")).reset_index().round(4).to_dict(orient="records")
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("additive_coefficients", "persistence", "reading")}, indent=1, default=float))
    print(pd.DataFrame(out["glass"]).to_string())


def main():
    J = [(k, N, T, W, s) for W in (4000, 1000) for N in (1600, 400, 100) for k in ("original", "rival_ko", "additive") for T in TS for s in "hc"]
    Jg = [(N, T) for N in (1600, 400) for T in TS]; t0 = time.time()
    with Pool(4) as p:
        for i, _ in enumerate(p.imap_unordered(unit_len, J), 1):
            if i % 12 == 0:
                print(f"[{time.time() - t0:6.0f}s] len {i}/{len(J)}", flush=True)
        for i, _ in enumerate(p.imap_unordered(unit_glass, Jg), 1):
            print(f"[{time.time() - t0:6.0f}s] glass {i}/{len(Jg)}", flush=True)
    summarise()


if __name__ == "__main__":
    main()
