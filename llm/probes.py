"""Amendment 4 exploratory annex: linear probes on last-token residual activations (labelled exploratory; cannot alter
H1–H5).

  python llm/probes.py --acts DIR/activations --key gemma-31b-local [--folds 10] [--out results/llm/amendment4]

Rows: main battery (all arms, perm 0 and the order subset), from the activation index. Targets are in LETTER space
(1 = the letter A), because the probe reads what the model sees and the readout is A vs B:
  ally_majority  side held by the strict majority of ally contacts (cells with M_a = 0 dropped)
  rival_majority side held by the strict majority of rival contacts (M_r = 0 dropped)
  own            the agent's own current position (arms that show it)
Probe: standardise + L2 logistic regression (C = 1), 10-fold CV grouped by multiset (the counts string), pooled
held-out AUC, per layer and arm.
Controls: (1) letter-count probe: features = number of 'A' among ally lines, number of 'A' among rival lines, own
letter is A (same CV); (2) shuffled labels (one permutation, same CV); (3) transfer: trained on neutral_own, tested
on political (held-out by construction).
'Represented but not used' (descriptive): rival target decodable (AUC ≥ 0.9 at some layer and above the letter-count
control) while |b_r| is small relative to b_a (fitted A0, local endpoint, neutral_own) — the ratio is reported.
Readout alignment: cosine between the last-layer rival-majority probe direction (mapped back to raw activation units)
and W_U[A] − W_U[B] (rows fetched by llm/unembed_rows.py; ids = the tokenizer's single tokens 'A', 'B').
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["neutral_own", "neutral_noown", "political", "workplace"]
MODEL_ID = {"gemma-26b-local": "google/gemma-4-26b-a4b-it", "gemma-31b-local": "google/gemma-4-31b-it"}


def targets(idx):
    """Letter-space targets and letter-count control features from the index (counts, s0, mapping, seq)."""
    rows = []
    for r in idx.itertuples():
        let = battery.letters(int(r.mapping))                     # content key -> letter
        c = [int(x) for x in str(r.counts).zfill(4)]
        Ma, Mr = c[0] - c[1], c[2] - c[3]
        isA = lambda content: 1 if let[content] == "A" else 0
        seq = [r.seq[i:i + 2] for i in range(0, len(r.seq), 2)]
        nA_ally = sum(isA(t[1]) for t in seq if t[0] == "a"); nA_riv = sum(isA(t[1]) for t in seq if t[0] == "r")
        own = None if int(r.s0) == 0 else isA("P" if int(r.s0) > 0 else "Q")
        rows.append(dict(ally_majority=None if Ma == 0 else isA("P" if Ma > 0 else "Q"),
                         rival_majority=None if Mr == 0 else isA("P" if Mr > 0 else "Q"),
                         own=own, f_ally=nA_ally, f_riv=nA_riv, f_own=-1 if own is None else own))
    return pd.DataFrame(rows, index=idx.index)


def cv_auc(X, y, groups, folds, seed=0, return_model=False):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, p))
    if return_model:
        return auc, make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X, y)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--folds", type=int, default=10); ap.add_argument("--out", default=os.path.join(ROOT, "results/llm/amendment4"))
    a = ap.parse_args()
    d = os.path.join(a.acts, a.key)
    idx = pd.read_csv(os.path.join(d, "index.csv"), dtype={"counts": str, "seq": str})
    main_rows = idx[idx["phase"] == "main"].copy()
    T = targets(main_rows)
    layers = sorted(int(f[6:9]) for f in os.listdir(d) if f.startswith("layer_"))
    os.makedirs(a.out, exist_ok=True)
    part = os.path.join(a.out, f"probes_{a.key}_partial.csv")                  # one block of rows per finished layer
    done = pd.read_csv(part) if os.path.exists(part) else pd.DataFrame()
    res, models = (done.to_dict("records") if len(done) else []), {}
    for L in layers:
        rng = np.random.default_rng(L)                                          # per layer, so resuming is exact
        last = L == layers[-1]
        if len(done) and L in set(done["layer"]) and not last:
            print(f"{a.key} L{L:02d} done earlier (resumed)", flush=True); continue
        if last and len(done) and L in set(done["layer"]):
            res = [r for r in res if r["layer"] != L]                          # refit last layer: its model is needed
        H = np.load(os.path.join(d, f"layer_{L:03d}.npy")).astype(np.float32)
        for arm in ARMS:
            sel = main_rows["arm"] == arm
            for tgt in ("ally_majority", "rival_majority", "own"):
                m = sel & T[tgt].notna()
                if m.sum() < 50:
                    continue
                ii = np.flatnonzero(m.values); rows_idx = main_rows.index[ii]
                X = H[main_rows.loc[rows_idx, "row"].values]; y = T.loc[rows_idx, tgt].astype(int).values
                g = main_rows.loc[rows_idx, "counts"].values
                keep_model = (L == layers[-1] and tgt == "rival_majority" and arm == "neutral_own")
                auc = cv_auc(X, y, g, a.folds, return_model=keep_model)
                if keep_model:
                    auc, models["rival_last"] = auc
                F = T.loc[rows_idx, ["f_ally", "f_riv", "f_own"]].values.astype(float)
                res.append(dict(key=a.key, layer=L, arm=arm, target=tgt, n=int(len(y)), auc=auc,
                                auc_letter_count=cv_auc(F, y, g, a.folds),
                                auc_shuffled=cv_auc(X, rng.permutation(y), g, a.folds)))
                print(f"{a.key} L{L:02d} {arm:13s} {tgt:15s} n={len(y):5d} AUC {res[-1]['auc']:.3f}  "
                      f"letter-count {res[-1]['auc_letter_count']:.3f}  shuffled {res[-1]['auc_shuffled']:.3f}", flush=True)
            # transfer: neutral_own -> political
        for tgt in ("ally_majority", "rival_majority", "own"):
            tr = (main_rows["arm"] == "neutral_own") & T[tgt].notna(); te = (main_rows["arm"] == "political") & T[tgt].notna()
            Xtr, ytr = H[main_rows.loc[tr, "row"].values], T.loc[tr, tgt].astype(int).values
            Xte, yte = H[main_rows.loc[te, "row"].values], T.loc[te, tgt].astype(int).values
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Xtr, ytr)
            res.append(dict(key=a.key, layer=L, arm="neutral_own->political", target=tgt, n=int(len(yte)),
                            auc=float(roc_auc_score(yte, mdl.predict_proba(Xte)[:, 1]))))
            print(f"{a.key} L{L:02d} transfer neutral_own->political {tgt:15s} AUC {res[-1]['auc']:.3f}", flush=True)
        pd.DataFrame(res).to_csv(part + ".tmp", index=False); os.replace(part + ".tmp", part)
    R = pd.DataFrame(res)
    # readout alignment and 'represented but not used'
    summary = {"key": a.key, "layers": layers}
    z = np.load(os.path.join(a.out, f"unembed_AB_{MODEL_ID[a.key].split('/')[1]}.npz"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID[a.key])
    iA, iB = tok.convert_tokens_to_ids("A"), tok.convert_tokens_to_ids("B")
    wdiff = z[f"A_{iA}"] - z[f"B_{iB}"]
    pipe = models["rival_last"]; sc, lr = pipe[0], pipe[1]
    direction = lr.coef_[0] / sc.scale_                                  # raw activation units
    summary["readout_token_ids"] = {"A": int(iA), "B": int(iB)}
    summary["cos_rival_probe_vs_WU_A_minus_B_last_layer"] = float(direction @ wdiff / np.linalg.norm(direction) / np.linalg.norm(wdiff))
    fit = json.load(open(os.path.join(ROOT, "results/llm/fits/main", a.key + ".json")))
    c = {x["arm"]: x["coef"] for x in fit["arms"]}["neutral_own"]
    rv = R[(R.target == "rival_majority") & (R.arm == "neutral_own")]
    best = rv.loc[rv.auc.idxmax()]
    summary["rival_neutral_own_best_layer"] = int(best.layer); summary["rival_best_auc"] = float(best.auc)
    summary["rival_best_auc_letter_count"] = float(best.auc_letter_count)
    summary["b_a"] = c["b_a"]["est"]; summary["b_r"] = c["b_r"]["est"]; summary["abs_b_r_over_b_a"] = abs(c["b_r"]["est"] / c["b_a"]["est"])
    summary["rival_decodable"] = bool(best.auc >= 0.9 and best.auc > best.auc_letter_count)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, f"probes_{a.key}.csv"), index=False)
    json.dump(summary, open(os.path.join(a.out, f"probes_{a.key}_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
