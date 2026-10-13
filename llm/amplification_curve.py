"""Amplification curve near the order-disorder transition (plan: llm/AMPLIFICATION_PLAN.md). Simulation only, no API calls.

  python llm/amplification_curve.py bench               # timing, no results
  python llm/amplification_curve.py locate [--R 200]    # stage A: base kernels on the coarse grid, Binder crossing
  python llm/amplification_curve.py run [--R 400]       # stage B: perturbations on the final grid (resumable per job)
  python llm/amplification_curve.py analyse             # amplification, CIs, pre-registered verdict, figures

Kernel at temperature T: every per-prompt letter logit is divided by T, then marginalised over mapping/order/contact
order exactly as at T = 1 (K_avg: k_avg_calls.csv; K_fixed: the Stage 1 battery; A3: the fitted rule with eta/T).
Outputs: results/llm/amplification/.
"""
import argparse, hashlib, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2, stage3, surrogate as S, kernel_test as K, k_avg as KA
from endpoints import E

OUT = os.path.join(K.ROOT, "results", "llm", "amplification")
PTS = ["P1", "P2", "P4"]                     # P1 test point; P2, P4 controls
NS = [100, 400, 1600]
N_GRAPHS = 5
W_EQ, M_EQ, PERS_WIN = 200, 200, 10         # equilibrium protocol: warm-up, measured sweeps, persistence window
COARSE = [round(x, 4) for x in np.geomspace(0.2, 5.0, 15)]
DELTAS = [0.01, 0.03, 0.1, 0.2, 0.5]
LAMBDAS = [0.25, 0.5, 1.0]
N_NOISE = 3
CLIP = 12.0


def seed(*p):
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:15], 16) % 2**31


lg = lambda p: np.clip(np.log(np.clip(p, 1e-300, 1) / np.clip(1 - p, 1e-300, 1)), -CLIP, CLIP)
sg = lambda x: 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- kernels at temperature T
def _calls():
    c = pd.read_csv(os.path.join(K.OUT, "k_avg_calls.csv"), dtype={"counts": str, "seq": str})
    c["counts"] = c.counts.str.zfill(4)
    return c


def _tempered(pa, T):
    return sg(lg(np.asarray(pa, float)) / T)


def battery_T(pid, T):
    """K.battery_kernel with each per-prompt P(A) tempered before the mapping/order mean."""
    key, framing, rho, _ = stage3.POINTS[pid]
    d = pd.read_csv(os.path.join(K.ROOT, f"results/llm/stage1/main/{key}.csv"), dtype={"counts": str})
    d = d[(d.arm == stage2.FRAME_ARM[framing]) & (d.perm == 0)].copy(); d["counts"] = d.counts.str.zfill(4)
    pa = d.groupby(["counts", "s0", "mapping", "order"]).pA.mean()
    pa = pd.Series(_tempered(pa.values, T), index=pa.index).to_dict()
    Tb = np.full(K.NT, np.nan)
    for c, s0 in K.configs(4, rho):
        vals = []
        for m in (0, 1):
            for o in (0, 1):
                if framing == "neutral":
                    if m == 0:
                        vals.append(pa[("".join(map(str, c)), s0, 0, o)])
                    else:
                        sw = (c[1], c[0], c[3], c[2]); vals.append(1 - pa[("".join(map(str, sw)), -s0, 0, o)])
                else:
                    v = pa[("".join(map(str, c)), s0, m, o)]; vals.append(v if m == 0 else 1 - v)
        Tb[K.idx(c, s0)] = np.mean(vals)
    return Tb


def kavg_T(pid, T, calls):
    """k_avg.kavg_table with each per-prompt P(A) tempered; configurations never seen live fall back to battery_T."""
    key, framing, rho, _ = stage3.POINTS[pid]
    d = calls[(calls.point == pid) & calls.valid]
    pp = d.groupby(["counts", "s0", "mapping", "order", "seq"]).pA.mean().reset_index()
    pp["pA"] = _tempered(pp.pA.values, T); pp["P"] = np.where(pp.mapping == 0, pp.pA, 1 - pp.pA)
    byc = pp.groupby(["counts", "s0"]).P.mean().to_dict()
    Tk = battery_T(pid, T); filled = np.zeros(K.NT, bool)
    for c, s0 in K.configs(4, rho):
        cs = "".join(map(str, c))
        if framing == "neutral":
            sw = "".join(map(str, (c[1], c[0], c[3], c[2])))
            if (cs, s0) in byc and (sw, -s0) in byc:
                Tk[K.idx(c, s0)] = 0.5 * byc[(cs, s0)] + 0.5 * (1 - byc[(sw, -s0)]); filled[K.idx(c, s0)] = True
        elif (cs, s0) in byc:
            Tk[K.idx(c, s0)] = byc[(cs, s0)]; filled[K.idx(c, s0)] = True
    return Tk, filled


def a3_T(pid, T):
    key, framing, rho, _ = stage3.POINTS[pid]
    row = stage2.param_row(stage2.load_fit(key, framing), framing, None, "fitted").copy()
    for j in (S.B_A, S.B_R, S.GAMMA, S.H_C, S.H_L, S.H_O):
        row[j] /= T
    return K.a3_kernel(row, 4, rho)


def social_sign():
    h = np.zeros(K.NT)
    for i in range(K.NT):
        c = i // 2; naP, naQ, nrP, nrQ = c // 125, (c // 25) % 5, (c // 5) % 5, c % 5
        h[i] = np.sign((naP - naQ) - (nrP - nrQ))
    return h


def stay_sign():
    return np.array([1.0 if i % 2 else -1.0 for i in range(K.NT)])


def noise_shape(k):
    return np.random.default_rng(seed("amp-noise", k)).standard_normal(K.NT)


def settings(pid, T, calls, stage):
    """name -> (logit table of the perturbed kernel, delta logit). Base first."""
    base, filled = kavg_T(pid, T, calls); Lb = lg(np.nan_to_num(base, nan=0.5))
    out = {"base": (Lb, np.zeros(K.NT))}
    if stage == "locate":
        return out, filled
    for d in DELTAS:
        out[f"stay:{d}"] = (Lb + d * stay_sign(), d * stay_sign())
        out[f"majority:{d}"] = (Lb + d * social_sign(), d * social_sign())
        for k in range(N_NOISE):
            out[f"noise{k}:{d}"] = (Lb + d * noise_shape(k), d * noise_shape(k))
    for nm, Tx in (("fixed", battery_T(pid, T)), ("a3", a3_T(pid, T))):
        dL = np.nan_to_num(lg(Tx) - Lb)
        for lam in LAMBDAS:
            out[f"{nm}:{lam}"] = (Lb + lam * dL, lam * dL)
    return out, filled


# ---------------------------------------------------------------- simulator
@njit(cache=True)
def sim_eq(ip, nb, tie, L, sd, n_warm, n_meas, pwin):
    """One replica, random 50/50 start, random-sequential updates with P(+1) = sigma(L[config]).
    Returns (mean |m|, mean m^2, mean m^4, persistence over the first pwin measured sweeps, flip rate), visit counts.
    The same seed gives the same site sequence, start and uniforms for every kernel (common random numbers)."""
    np.random.seed(sd)
    N = ip.shape[0] - 1
    P = 1.0 / (1.0 + np.exp(-L))
    s = np.empty(N, np.int64)
    for i in range(N):
        s[i] = 1 if np.random.random() < 0.5 else -1
    ref = s.copy(); vis = np.zeros(L.shape[0]); a = np.zeros(5); nfl = 0.0; nup = 0.0
    for t in range(n_warm + n_meas):
        for _u in range(N):
            i = np.random.randint(N); u = np.random.random()
            c = 0
            for q in range(ip[i], ip[i + 1]):
                if tie[q] > 0:
                    c += 125 if s[nb[q]] > 0 else 25
                else:
                    c += 5 if s[nb[q]] > 0 else 1
            old = s[i]; c = c * 2 + (1 if old > 0 else 0)
            new = 1 if u < P[c] else -1
            s[i] = new
            if t >= n_warm:
                vis[c] += 1.0; nup += 1.0
                if new != old:
                    nfl += 1.0
        if t == n_warm - 1:
            for i in range(N):
                ref[i] = s[i]
        if t >= n_warm:
            m = 0.0
            for i in range(N):
                m += s[i]
            m /= N
            a[0] += abs(m); a[1] += m * m; a[2] += m ** 4
            if t < n_warm + pwin:
                o = 0.0
                for i in range(N):
                    o += s[i] * ref[i]
                a[3] += o / N
    res = np.empty(5)
    res[0] = a[0] / n_meas; res[1] = a[1] / n_meas; res[2] = a[2] / n_meas; res[3] = a[3] / min(pwin, n_meas)
    res[4] = nfl / max(nup, 1.0)
    return res, vis


MEAS = ["abs_m", "m2", "m4", "persistence", "flip_rate"]
GRAPH_FP = {}


def graphs(pid, N):
    base = pid.split("-")[0]; rho = stage3.POINTS[base][2]
    if N == stage3.N:
        G = [stage3.graph(base, g) for g in range(N_GRAPHS)]
    else:
        G = [S.make_graph(N, rho, 0.0, np.random.default_rng(seed("amp-graph", base, N, g)), "rrg", d=stage3.D) for g in range(N_GRAPHS)]
    for g, (ip, nb, tie) in enumerate(G):
        GRAPH_FP[f"{base}:{N}:{g}"] = hashlib.sha256(np.concatenate([ip, nb, tie]).astype(np.int64).tobytes()).hexdigest()[:16]
    return G


def job(args):
    """One unit = (stage, protocol, point, N, T): every setting, R runs each, common random numbers across settings."""
    stage, proto, pid, N, T, R = args
    path = os.path.join(OUT, stage, proto, f"{pid}_N{N}_T{T:.4f}.parquet")
    if os.path.exists(path):
        return path
    calls = _calls(); G = graphs(pid, N)
    nw, nm, pw = (W_EQ, M_EQ, PERS_WIN) if proto == "eq" else (stage3.N_WARM, stage3.N_MEAS, stage3.N_MEAS)
    sets, filled = settings(pid, T, calls, stage)
    rows, vis = [], np.zeros(K.NT)
    for name, (L, dL) in sets.items():
        Lc = np.clip(L, -CLIP, CLIP)
        for r in range(R):
            ip, nb, tie = G[r % N_GRAPHS]
            res, v = sim_eq(ip, nb, tie, Lc, seed("amp-run", proto, pid, N, T, r), nw, nm, pw)
            rows.append(dict(setting=name, run=r, graph=r % N_GRAPHS, **dict(zip(MEAS, res))))
            if name == "base":
                vis += v
    w = vis / vis.sum()
    eps = {name: float((w * np.abs(dL)).sum()) for name, (L, dL) in sets.items()}
    df = pd.DataFrame(rows); df["eps"] = df.setting.map(eps); df["filled_share"] = float(w[filled].sum())
    df["point"], df["N"], df["T"], df["proto"] = pid, N, T, proto
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp", index=False); os.replace(path + ".tmp", path)
    np.save(path.replace(".parquet", "_visits.npy"), vis)
    return path


def run_jobs(jobs, procs=4):
    t0 = time.time()
    with Pool(procs) as p:
        for i, path in enumerate(p.imap_unordered(job, jobs), 1):
            print(f"[{time.time() - t0:7.0f}s] {i}/{len(jobs)} {os.path.basename(path)}", flush=True)


# ---------------------------------------------------------------- stage A: locate T_c
def binder(df):
    return 1 - df.m4.mean() / (3 * df.m2.mean() ** 2)


def crossing(Ts, ua, ub):
    """First T (ascending) where ua - ub changes sign, linear interpolation; None if no sign change."""
    d = np.asarray(ua) - np.asarray(ub)
    for j in range(len(Ts) - 1):
        if d[j] == 0:
            return Ts[j]
        if d[j] * d[j + 1] < 0:
            x0, x1 = math.log(Ts[j]), math.log(Ts[j + 1])
            return float(math.exp(x0 + (x1 - x0) * d[j] / (d[j] - d[j + 1])))
    return None


def locate(R):
    run_jobs([("locate", "eq", p, N, T, R) for p in PTS for N in NS for T in COARSE])
    D = pd.concat([pd.read_parquet(os.path.join(OUT, "locate", "eq", f"{p}_N{N}_T{T:.4f}.parquet")) for p in PTS for N in NS for T in COARSE])
    out = {}
    rng = np.random.default_rng(seed("amp-binder-boot"))
    for p in PTS:
        U = {N: [binder(D[(D.point == p) & (D.N == N) & (D["T"] == T)]) for T in COARSE] for N in NS}
        tc = {f"{a}-{b}": crossing(COARSE, U[a], U[b]) for a, b in ((100, 400), (400, 1600), (100, 1600))}
        boots = []
        for _ in range(200):
            Ub = {}
            for N in (400, 1600):
                Ub[N] = []
                for T in COARSE:
                    x = D[(D.point == p) & (D.N == N) & (D["T"] == T)]
                    Ub[N].append(binder(x.iloc[rng.integers(len(x), size=len(x))]))
            boots.append(crossing(COARSE, Ub[400], Ub[1600]))
        ok = [b for b in boots if b is not None]
        out[p] = dict(binder={str(N): [float(u) for u in U[N]] for N in NS}, crossings=tc,
                      Tc=tc["400-1600"], Tc_ci=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) > 20 else None,
                      boot_share_crossing=len(ok) / len(boots))
        print(p, json.dumps({k: v for k, v in out[p].items() if k != "binder"}), flush=True)
    out["coarse"] = COARSE
    json.dump(out, open(os.path.join(OUT, "locate.json"), "w"), indent=1)
    return out


def final_grid(Tc):
    """Plan section 3: 9 geometric points over [Tc/1.6, 1.6 Tc], plus the coarse points outside that band, plus T = 1."""
    fine = [round(x, 4) for x in np.geomspace(Tc / 1.6, Tc * 1.6, 9)] if Tc else []
    lo, hi = (Tc / 1.6, Tc * 1.6) if Tc else (np.inf, -np.inf)
    return sorted(set(fine + [t for t in COARSE if not lo <= t <= hi] + [1.0]))


def run(R):
    loc = json.load(open(os.path.join(OUT, "locate.json")))
    grid = final_grid(loc["P1"]["Tc"])
    json.dump(dict(grid=grid, Tc_P1=loc["P1"]["Tc"]), open(os.path.join(OUT, "grid.json"), "w"), indent=1)
    jobs = [("perturb", proto, p, N, T, R) for proto in ("eq", "s3") for N in sorted(NS, reverse=True) for p in PTS for T in grid]
    run_jobs(jobs)
    json.dump(GRAPH_FP, open(os.path.join(OUT, "graph_fingerprints.json"), "w"), indent=1)


def bench():
    calls = _calls(); sets, _ = settings("P1", 1.0, calls, "locate"); L = sets["base"][0]
    for N in NS:
        ip, nb, tie = graphs("P1", N)[0]
        sim_eq(ip, nb, tie, L, 1, 2, 2, 1)
        t0 = time.time(); sim_eq(ip, nb, tie, L, 2, W_EQ, M_EQ, PERS_WIN); print(N, f"{time.time() - t0:.3f}s per eq run")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--R", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "bench":
        bench()
    elif a.cmd == "locate":
        locate(a.R or 200)
    elif a.cmd == "run":
        run(a.R or 400)
    elif a.cmd == "analyse":
        import amplification_analyse; amplification_analyse.main()
