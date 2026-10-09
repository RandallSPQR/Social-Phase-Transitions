"""EXPLORATORY (post-data, EXECUTION_LOG step 1c): voter vs logit response models, and the mixture
p·(copy a random neighbour) + (1−p)·logit used by the Stage 2 surrogates.

Models, fitted by soft-label cross-entropy on the main battery (perm 0), compared by held-out CE with the
same grouped folds as Stage 1 (10-fold, grouped by neighbour multiset):
  A0   additive logit (Stage 1)
  A3   logit with k^-α scaling of the neighbour field (Stage 1 H5 model)
  V    signed voter with lapse: P = (1−ε)·V + ε/2, V = (n_ally,P + n_rival,Q)/k
       (pick a uniformly random contact; copy an ally, take the opposite of a rival). No own-position or field terms.
  M    mixture: P = p·V + (1−p)·σ(η_A3)
  MF   first-contact mixture: P = p·F + (1−p)·σ(η_A3), F = 1 if the FIRST-LISTED contact pulls toward P
       (order-specific; under random order E[F] = V).
Also writes, per endpoint × arm, point estimates and 100 cluster-bootstrap draws (cells) of A3 and MF to
results/llm/fits/surrogate/<key>.json for the Stage 2 simulator, plus the surrogate choice (rule fixed before
any simulation, EXECUTION_LOG step 1c): use MF if its held-out CE beats A3 by >= 2% relative, else A3 (p = 0).
Under random listing order (Stage 3 protocol) copying the first-listed contact IS a random-neighbour voter, so
MF's p is the voter weight of the network rule. M is reported but not used: its p is not identified apart from α.

  python llm/voter.py            # all endpoints with main-battery CSVs
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["neutral_own", "neutral_noown", "political", "workplace"]
EPS = 1e-9


def parts(P):
    X0, names = analysis.design(P, "A0")
    Z, znames = X0[:, 2:], names[2:]
    Ma, Mr = P["Ma"].values.astype(float), P["Mr"].values.astype(float)
    k = P["k"].values.astype(float)
    V = (P["naP"].values + P["nrQ"].values) / k
    F = (P["F1"].values + 1) / 2.0
    return dict(Z=Z, znames=znames, Ma=Ma, Mr=Mr, lk=np.log(k), V=V, F=F, y=P["y"].values, w=P["w"].values.astype(float))


def nll(Pp, y, w):
    Pp = np.clip(Pp, EPS, 1 - EPS)
    return -(w * (y * np.log(Pp) + (1 - y) * np.log(1 - Pp))).sum()


def fit_mix(D, comp="V", x0=None, fix_p=None):
    """Mixture p·comp + (1−p)·σ(k^-α(b_a Ma − b_r Mr) + Zθ). fix_p=0 gives A3. Parameters: [logit p, α, b_a, b_r, θ...]."""
    C = D[comp] if comp else 0.0
    nz = D["Z"].shape[1]
    def f(t):
        lp = t[0]; p = expit(lp) if fix_p is None else fix_p
        a, ba, br, th = t[1], t[2], t[3], t[4:]
        s = np.exp(-a * D["lk"]); core = ba * D["Ma"] - br * D["Mr"]
        eta = s * core + D["Z"] @ th; sg = expit(eta)
        Pp = np.clip(p * C + (1 - p) * sg, EPS, 1 - EPS)
        dLdP = -D["w"] * (D["y"] / Pp - (1 - D["y"]) / (1 - Pp))
        dPde = (1 - p) * sg * (1 - sg)
        r = dLdP * dPde
        g = np.concatenate([[(dLdP * (C - sg)).sum() * p * (1 - p) if fix_p is None else 0.0,
                             (r * (-D["lk"] * s * core)).sum(), (r * s * D["Ma"]).sum(), (r * (-s * D["Mr"])).sum()],
                            D["Z"].T @ r])
        return nll(Pp, D["y"], D["w"]), g
    t0 = np.concatenate([[-1.0, 0.5, 0.5, 0.2], np.zeros(nz)]) if x0 is None else x0
    bounds = [(-12, 12), (-3, 4)] + [(None, None)] * (2 + nz)
    best = None
    for start in ([t0] if x0 is not None else [t0, np.concatenate([[1.0, 1.0, 1.0, 0.5], np.zeros(nz)])]):
        r = minimize(f, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 4000, "gtol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def predict_mix(D, t, comp="V", fix_p=None):
    p = expit(t[0]) if fix_p is None else fix_p
    C = D[comp] if comp else 0.0
    eta = np.exp(-t[1] * D["lk"]) * (t[2] * D["Ma"] - t[3] * D["Mr"]) + D["Z"] @ t[4:]
    return p * C + (1 - p) * expit(eta)


def fit_voter(D):
    r = minimize(lambda e: nll((1 - expit(e[0])) * D["V"] + expit(e[0]) / 2, D["y"], D["w"]), [-2.0], method="L-BFGS-B")
    return r.x


def sub(D, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in D.items()}


def cv(P, D):
    f = analysis.folds_by_multiset(P)
    pred = {m: np.empty(len(P)) for m in ("A0", "A3", "V", "M", "MF")}
    X0, _ = analysis.design(P, "A0")
    for k in range(10):
        tr, te = f != k, f == k
        if te.sum() == 0: continue
        Dtr, Dte = sub(D, tr), sub(D, te)
        pred["A0"][te] = expit(X0[te] @ analysis.fit_ce(X0[tr], D["y"][tr], D["w"][tr]))
        pred["A3"][te] = predict_mix(Dte, fit_mix(Dtr, None, fix_p=0.0), None, fix_p=0.0)
        e = fit_voter(Dtr); pred["V"][te] = (1 - expit(e[0])) * Dte["V"] + expit(e[0]) / 2
        pred["M"][te] = predict_mix(Dte, fit_mix(Dtr, "V"), "V")
        pred["MF"][te] = predict_mix(Dte, fit_mix(Dtr, "F"), "F")
    return {m: analysis.ce(D["y"], v, D["w"]) for m, v in pred.items()}


def run_endpoint(path, B=100, seed=0):
    key = os.path.basename(path)[:-4]
    d = analysis.load(path)
    rows, sur = [], {"key": key, "model": "p*copy(first-listed contact) + (1-p)*sigmoid(k^-alpha (b_a Ma - b_r Mr) + Z theta); A3 = p fixed at 0", "arms": {}}
    rng = np.random.default_rng(seed)
    for arm in ARMS:
        da = d[d["arm"] == arm]
        if da.empty: continue
        P, st = analysis.prompts(da); P = P[P["perm"] == 0].reset_index(drop=True)
        if len(P) < 20: continue
        D = parts(P)
        tM = fit_mix(D, "V"); tF = fit_mix(D, "F"); tA = fit_mix(D, None, fix_p=0.0); tA[0] = -50.0
        c = cv(P, D)
        row = {"key": key, "arm": arm, "n_prompts": len(P), **{f"cvCE_{m}": v for m, v in c.items()},
               "label_entropy": analysis.ce(D["y"], D["y"], D["w"]),
               "p_M": float(expit(tM[0])), "alpha_M": float(tM[1]), "p_MF": float(expit(tF[0])), "alpha_MF": float(tF[1])}
        # cluster bootstrap (cells) of A3 and MF for intervals + Stage 2 draws
        members = analysis.cluster_index(P)
        dA, dF = [], []
        for _ in range(B):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            Db = sub(D, idx)
            a_ = fit_mix(Db, None, x0=tA, fix_p=0.0); a_[0] = -50.0; dA.append(a_)
            dF.append(fit_mix(Db, "F", x0=tF))
        dA, dF = np.array(dA), np.array(dF)
        pf = expit(dF[:, 0])
        row.update(p_MF_lo=float(np.percentile(pf, 2.5)), p_MF_hi=float(np.percentile(pf, 97.5)))
        use_mf = (c["A3"] - c["MF"]) / c["A3"] >= 0.02
        row["surrogate"] = "MF" if use_mf else "A3"
        names = ["logit_p", "alpha", "b_a", "b_r"] + D["znames"]
        sur["arms"][arm] = {"names": names, "choice": row["surrogate"],
                            "A3": {"est": tA.tolist(), "draws": dA.round(5).tolist()},
                            "MF": {"est": tF.tolist(), "draws": dF.round(5).tolist()},
                            "repeat_sd": st.get("repeat_sd"), "n_prompts": len(P), "cvCE": c}
        rows.append(row)
        print(f"{key:20s} {arm:14s} " + " ".join(f"{m}={v:.4f}" for m, v in c.items()) +
              f"  p_M={row['p_M']:.2f} α_M={row['alpha_M']:.2f}  p_MF={row['p_MF']:.2f}[{row['p_MF_lo']:.2f},{row['p_MF_hi']:.2f}] -> {row['surrogate']}", flush=True)
    out = os.path.join(ROOT, "results", "llm", "fits", "surrogate"); os.makedirs(out, exist_ok=True)
    json.dump(sur, open(os.path.join(out, key + ".json"), "w"))
    return rows


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "results", "llm", "stage1", "main", "*.csv")))
    rows = []
    for p in paths:
        rows += run_endpoint(p)
    od = os.path.join(ROOT, "results", "llm", "voter"); os.makedirs(od, exist_ok=True)
    for k in {r["key"] for r in rows}:          # one file per endpoint (parallel-safe), then re-merge all
        pd.DataFrame([r for r in rows if r["key"] == k]).to_csv(os.path.join(od, f"{k}.csv"), index=False)
    allp = sorted(glob.glob(os.path.join(od, "*.csv")))
    pd.concat([pd.read_csv(p) for p in allp]).to_csv(os.path.join(ROOT, "results", "llm", "voter_vs_logit.csv"), index=False)
