"""Confirmatory amplification phase, Steps 4-5 + amendment 3 runs (plan: llm/AMPLIFICATION_CONFIRM_PLAN.md §5-6, A3).

  python llm/confirm_step45.py run        # all units (resumable per unit), 4 processes
  python llm/confirm_step45.py gemma      # amendment 3 exploratory Gemma checks (a) run length, (b) outcome sensitivity

Unit = (step, model, N, T, W, R): every setting (base, stay, majority, noise0-2 at delta = 0.1; real fixed-order battery
shape at lambda = 1 where it exists) with common random numbers (same seed per run across settings).
Outputs: results/llm/confirm/step45/<step>/*.csv.gz (+ _visits.npy).
"""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2, stage3, kernel_test as K, amplification_curve as A, confirm_sim as C

ROOT = K.ROOT
OUT = os.path.join(ROOT, "results", "llm", "confirm", "step45")
SIM = os.path.join(ROOT, "results", "llm", "confirm", "sim")
DELTA = 0.1
seed = A.seed


# ---------------------------------------------------------------- real error shapes (fixed-order battery)
def battery_sampled_T(key, T):
    """Stage 1 battery kernel of a sampled endpoint at temperature T: per (cell, mapping, order) Jeffreys P(A), tempered,
    then the battery_kernel mean over mapping x order (neutral: relabelled from mapping 0)."""
    d = pd.read_csv(os.path.join(ROOT, f"results/llm/stage1/main/{key}.csv"), dtype={"counts": str})
    d = d[(d.arm == "neutral_own") & (d.perm == 0)].copy(); d["counts"] = d.counts.str.zfill(4)
    g = d.groupby(["counts", "s0", "mapping", "order"]).agg(x=("answer", lambda a: (a == "A").sum()), n=("answer", lambda a: a.notna().sum()))
    pa = pd.Series(A._tempered(((g.x + 0.5) / (g.n + 1)).values, T), index=g.index).to_dict()
    Tb = np.full(K.NT, np.nan)
    for c, s0 in K.configs(4, 0.25):
        vals = []
        for o in (0, 1):
            vals.append(pa[("".join(map(str, c)), s0, 0, o)])
            sw = (c[1], c[0], c[3], c[2]); vals.append(1 - pa[("".join(map(str, sw)), -s0, 0, o)])
        Tb[K.idx(c, s0)] = np.mean(vals)
    return Tb


def real_shape(model, T):
    if model == "P1":
        return A.battery_T("P1", T)
    if model == "gemini":
        return battery_sampled_T("gemini-lite", T)
    if model == "luna":
        return battery_sampled_T("gpt6-luna", T)
    return None


def settings(model, T, real):
    Lb = A.lg(np.nan_to_num(C.table(model, T), nan=0.5))
    out = {"base": (Lb, np.zeros(K.NT)), f"stay:{DELTA}": (Lb + DELTA * A.stay_sign(), DELTA * A.stay_sign()),
           f"majority:{DELTA}": (Lb + DELTA * A.social_sign(), DELTA * A.social_sign())}
    for k in range(3):
        xi = np.random.default_rng(seed("confirm-noise", k)).standard_normal(K.NT)
        out[f"noise{k}:{DELTA}"] = (Lb + DELTA * xi, DELTA * xi)
    if real:
        Tx = real_shape(model, T)
        if Tx is not None:
            dL = np.nan_to_num(A.lg(Tx) - Lb); out["fixed:1.0"] = (Lb + dL, dL)
    return out


def unit(args):
    step, model, N, T, W, R, real = args
    path = os.path.join(OUT, step, f"{model}_N{N}_T{T:.4f}_W{W}.csv.gz")
    if os.path.exists(path):
        return path
    G, _ = C.graphs(model, N); S = settings(model, T, real); rows, vis = [], np.zeros(K.NT)
    for name, (L, dL) in S.items():
        Lc = np.clip(L, -A.CLIP, A.CLIP)
        for r in range(R):
            res, v = A.sim_eq(*G[r % 5], Lc, seed("confirm-step4" if step != "step5" else "confirm-freeze", model, N, T, W, r), W, W, A.PERS_WIN, False)
            rows.append(dict(setting=name, run=r, **dict(zip(A.MEAS, res))))
            if name == "base":
                vis += v
    w = vis / vis.sum(); eps = {n: float((w * np.abs(dL)).sum()) for n, (L, dL) in S.items()}
    df = pd.DataFrame(rows); df["eps"] = df.setting.map(eps)
    df["model"], df["N"], df["T"], df["W"], df["step"] = model, N, T, W, step
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path + ".tmp", index=False, compression="gzip"); os.replace(path + ".tmp", path)
    return path


def jobs():
    J = []
    loc = {m: json.load(open(os.path.join(SIM, f"locate_{m}.json"))) for m in ("P1", "gemini", "luna")}
    for m in ("P1", "gemini", "luna"):
        band = loc[m]["step4_band"]; tc = band[6]; W, R = loc[m]["step4"]["W"], loc[m]["step4"]["R"]
        J += [("step4", m, 1600, T, W, R, True) for T in band] + [("step4", m, 100, tc, W, R, True)]
        J += [("step4", c, N, tc, W, R, False) for c in ("P2", "P4") for N in (1600, 100)]
    for m in ("P1", "gemini"):        # amendment 3: GPT-4o-mini negative control on the confirmatory bands
        J += [("step4", "selected", N, T, 500, 400, False) for T in loc[m]["step4_band"] for N in (1600, 100)]
    s5 = [round(x, 4) for x in np.geomspace(0.5, 0.9, 7)]
    J += [("step5", "P1", N, T, W, 200, False) for W in (500, 1000, 2000) for N in (1600, 400, 100) for T in s5]
    J = list(dict.fromkeys(J))
    return sorted(J, key=lambda j: -(j[2] * j[4] * j[5]))          # longest first


def run():
    J = jobs(); t0 = time.time(); print(len(J), "units", flush=True)
    with Pool(4) as p:
        for i, path in enumerate(p.imap_unordered(unit, J), 1):
            print(f"[{time.time() - t0:6.0f}s] {i}/{len(J)} {os.path.relpath(path, OUT)}", flush=True)


# ---------------------------------------------------------------- amendment 3: Gemma exploratory
@njit(cache=True)
def sim_basin(ip, nb, tie, L, start_seed, dyn_seed, W, last):
    N = ip.shape[0] - 1
    np.random.seed(start_seed)
    s = np.empty(N, np.int64)
    for i in range(N):
        s[i] = 1 if np.random.random() < 0.5 else -1
    np.random.seed(dyn_seed)
    P = 1.0 / (1.0 + np.exp(-L)); acc = 0.0
    for t in range(W):
        for _u in range(N):
            i = np.random.randint(N); u = np.random.random(); c = 0
            for q in range(ip[i], ip[i + 1]):
                if tie[q] > 0:
                    c += 125 if s[nb[q]] > 0 else 25
                else:
                    c += 5 if s[nb[q]] > 0 else 1
            c = c * 2 + (1 if s[i] > 0 else 0)
            s[i] = 1 if u < P[c] else -1
        if t >= W - last:
            m = 0.0
            for i in range(N):
                m += s[i]
            acc += m / N
    return acc / last


def gemma_basin_unit(args):
    N, T, R = args
    path = os.path.join(OUT, "gemma_basin", f"N{N}_T{T:.4f}.csv.gz")
    if os.path.exists(path):
        return path
    G, _ = C.graphs("gemma", N); S = settings("gemma", T, False)
    keep = ["base", f"stay:{DELTA}", f"majority:{DELTA}", f"noise0:{DELTA}"]; rows = []
    for r in range(R):
        g = G[r % 5]; ss, ds = seed("gemma-basin-start", N, T, r), seed("gemma-basin-dyn", N, T, r)
        row = dict(run=r)
        for k in keep:
            row[k] = sim_basin(*g, np.clip(S[k][0], -A.CLIP, A.CLIP), ss, ds, 1000, 100)
        row["ref_otherseed"] = sim_basin(*g, np.clip(S["base"][0], -A.CLIP, A.CLIP), ss, seed("gemma-basin-dyn2", N, T, r), 1000, 100)
        rows.append(row)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).assign(N=N, T=T).to_csv(path + ".tmp", index=False, compression="gzip"); os.replace(path + ".tmp", path)
    return path


def gemma():
    Ts = [round(x, 4) for x in np.geomspace(2.0, 3.2, 7)]
    la = [("gemma_len", "gemma", N, T, s, W, 50, None) for W in (4000, 2000, 1000) for N in (1600, 400, 100) for T in Ts for s in "hc"]
    lb = [(N, T, 200) for N in (1600, 400, 100) for T in Ts + [1.0]]
    t0 = time.time()
    with Pool(4) as p:
        for i, path in enumerate(p.imap_unordered(C.unit, la), 1):
            if i % 20 == 0:
                print(f"[{time.time() - t0:6.0f}s] len {i}/{len(la)}", flush=True)
        for i, path in enumerate(p.imap_unordered(gemma_basin_unit, lb), 1):
            print(f"[{time.time() - t0:6.0f}s] basin {i}/{len(lb)} {os.path.basename(path)}", flush=True)


if __name__ == "__main__":
    {"run": run, "gemma": gemma}[sys.argv[1]]()
