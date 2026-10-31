"""Exploratory probe follow-ups (owner request 2026-10-11). Exploratory only: never in a headline, cannot alter H1–H5.

  python llm/probes_followup.py --acts DIR/activations --key gemma-31b-local [--B 1000]

Same probe as llm/probes.py (standardise + L2 logistic, C = 1, 10-fold GroupKFold by multiset, AUC from the decision
function), letter-space targets, arms neutral_own / political / workplace, targets ally_majority and rival_majority.
 a) CI on AUC(probe) − AUC(letter-count control): the held-out scores of both come from the same grouped folds; the
    95% CI resamples multisets (the CV groups) with replacement, B times, recomputing both AUCs.
 b) Deconfounded test: AUCs restricted to configurations where the target side (rival- or ally-majority letter)
    DISAGREES with the overall letter majority across all contact lines (ally + rival). There a probe that only tracks
    "how many A's are in the prompt" must fail; the overall-A-count score itself is reported as that baseline.
 c) Readout baselines at the last layer: cosine with W_U[A] − W_U[B] (tokens 'A', 'B') of (i) the rival-majority
    probe direction, (ii) the ally-majority and own-position probe directions, (iii) 1,000 isotropic random
    directions, (iv) 100 shuffled-label probe directions (same pipeline). Directions in raw activation units.
Saves results/llm/amendment5/probes_followup_<key>.csv per layer as it goes (resumable) and a summary json.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probes as PR

ROOT = PR.ROOT
OUT = os.path.join(ROOT, "results/llm/amendment5")
ARMS = ["neutral_own", "political", "workplace"]


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def cv_scores(X, y, g, folds=10):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, g):
        s[te] = model().fit(X[tr], y[tr]).decision_function(X[te])
    return s


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def boot_diff(y, s1, s2, g, B, rng):
    ug = np.unique(g); idx = {u: np.flatnonzero(g == u) for u in ug}; d = []
    for _ in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if len(np.unique(y[ii])) == 2:
            d.append(roc_auc_score(y[ii], s1[ii]) - roc_auc_score(y[ii], s2[ii]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True); ap.add_argument("--key", required=True); ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    d = os.path.join(a.acts, a.key)
    idx = pd.read_csv(os.path.join(d, "index.csv"), dtype={"counts": str, "seq": str})
    M = idx[idx["phase"] == "main"].copy(); T = PR.targets(M)
    # overall letter majority across all contact lines (1 = more A than B, 0 = more B, NaN = tie)
    nA = T["f_ally"] + T["f_riv"]; k = M["counts"].map(lambda c: sum(int(x) for x in c.zfill(4)))
    overall = np.where(nA * 2 > k, 1.0, np.where(nA * 2 < k, 0.0, np.nan))
    layers = sorted(int(f[6:9]) for f in os.listdir(d) if f.startswith("layer_"))
    os.makedirs(OUT, exist_ok=True)
    part = os.path.join(OUT, f"probes_followup_{a.key}.csv")
    done = pd.read_csv(part) if os.path.exists(part) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    for L in layers:
        if len(done) and L in set(done["layer"]):
            print(f"{a.key} L{L} done earlier (resumed)", flush=True); continue
        H = np.load(os.path.join(d, f"layer_{L:03d}.npy")).astype(np.float32)
        rng = np.random.default_rng(L)
        for arm in ARMS:
            for tgt in ("ally_majority", "rival_majority"):
                m = ((M["arm"] == arm) & T[tgt].notna()).values
                ii = np.flatnonzero(m); y = T[tgt].values[ii].astype(int); g = M["counts"].values[ii]
                X = H[M["row"].values[ii]]; F = T[["f_ally", "f_riv", "f_own"]].values[ii].astype(float)
                sp, sl = cv_scores(X, y, g), cv_scores(F, y, g)
                lo, hi = boot_diff(y, sp, sl, g, a.B, rng)
                dis = ~np.isnan(overall[ii]) & (overall[ii] != y)              # target side != overall A-majority
                r = dict(key=a.key, layer=L, arm=arm, target=tgt, n=int(len(y)),
                         auc_probe=auc(y, sp), auc_letter=auc(y, sl), diff=auc(y, sp) - auc(y, sl), diff_lo=lo, diff_hi=hi,
                         n_disagree=int(dis.sum()), auc_probe_disagree=auc(y[dis], sp[dis]),
                         auc_letter_disagree=auc(y[dis], sl[dis]), auc_overallA_disagree=auc(y[dis], nA.values[ii][dis]))
                if dis.sum() > 20 and len(np.unique(y[dis])) == 2:
                    r["diff_disagree_lo"], r["diff_disagree_hi"] = boot_diff(y[dis], sp[dis], sl[dis], g[dis], a.B, rng)
                rows.append(r)
                print(f"{a.key} L{L:02d} {arm:12s} {tgt:15s} probe {r['auc_probe']:.3f} letter {r['auc_letter']:.3f} "
                      f"diff {r['diff']:+.3f} [{lo:+.3f},{hi:+.3f}] | disagree n={r['n_disagree']} probe {r['auc_probe_disagree']:.3f} "
                      f"letter {r['auc_letter_disagree']:.3f} overallA {r['auc_overallA_disagree']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(part + ".tmp", index=False); os.replace(part + ".tmp", part)
    # c) readout baselines, last layer, neutral_own
    L = layers[-1]; H = np.load(os.path.join(d, f"layer_{L:03d}.npy")).astype(np.float32)
    z = np.load(os.path.join(ROOT, "results/llm/amendment4", f"unembed_AB_{PR.MODEL_ID[a.key].split('/')[1]}.npz"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PR.MODEL_ID[a.key])
    wd = z[f"A_{tok.convert_tokens_to_ids('A')}"] - z[f"B_{tok.convert_tokens_to_ids('B')}"]
    cos = lambda v: float(v @ wd / np.linalg.norm(v) / np.linalg.norm(wd))

    def direction(tgt, shuffle_rng=None):
        m = ((M["arm"] == "neutral_own") & T[tgt].notna()).values
        X = H[M["row"].values[m]]; y = T[tgt].values[m].astype(int)
        if shuffle_rng is not None:
            y = shuffle_rng.permutation(y)
        p = model().fit(X, y)
        return p[1].coef_[0] / p[0].scale_

    rng = np.random.default_rng(99)
    obs = {t: cos(direction(t)) for t in ("rival_majority", "ally_majority", "own")}
    iso = np.array([cos(rng.standard_normal(len(wd))) for _ in range(1000)])
    shuf = np.array([cos(direction("rival_majority", np.random.default_rng(1000 + i))) for i in range(100)])
    summ = dict(key=a.key, layer=L, cos_observed=obs,
                isotropic=dict(mean=float(iso.mean()), sd=float(iso.std()), q95_abs=float(np.quantile(np.abs(iso), 0.95))),
                shuffled_label=dict(mean=float(shuf.mean()), sd=float(shuf.std()), q95_abs=float(np.quantile(np.abs(shuf), 0.95))),
                percentile_abs_vs_shuffled={t: float((np.abs(shuf) < abs(v)).mean()) for t, v in obs.items()},
                percentile_abs_vs_isotropic={t: float((np.abs(iso) < abs(v)).mean()) for t, v in obs.items()})
    json.dump(summ, open(os.path.join(OUT, f"probes_followup_{a.key}_readout.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
