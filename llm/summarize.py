"""Confirmatory summary across endpoints (PREREG §6 + Amendments 1–3).

  python ../llm/summarize.py            # reads results/llm/fits/main/*.json, results/llm/comprehension_summary.csv

Confirmatory set: WORKHORSES that pass the comprehension check (≥90% on each question type) and the 0.3-nat
repeat-noise gate. Endpoints failing comprehension are reported descriptively ("does not parse task") and are
excluded from every Holm family and from H4. Bridge (gpt4o-mini) and provider replication (llama-70b-pa) are
descriptive. Holm at FWER 0.05, separately per hypothesis family, across endpoint × framing cells.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import holm
from endpoints import H4_PAIRS, BACKUP_KEY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKHORSES = ["llama-8b", "llama-70b", "qwen-9b", "qwen-122b", "gemma-26b", "gemma-31b", "nemo-12b",
              "mistral-small", "mistral-large", "gemma-26b-cw", "gemma-31b-nv"]
NOISE_GATE = 0.30


def pooled_sd(arms):
    """Endpoint-level repeat SD (PREREG §4): RMS over all prompts of all arms."""
    if any(a.get("repeat_sd") is None for a in arms.values()):
        return float("nan")                                  # sampled endpoint: gate not applicable
    n = np.array([a["prompts"] for a in arms.values()]); sd = np.array([a["repeat_sd"] for a in arms.values()])
    return float(np.sqrt((n * sd ** 2).sum() / n.sum()))


def resolve(key, F, gate):
    """Endpoint used for a model in confirmatory analysis: the primary if it passes the noise gate,
    else its pre-listed backup if that passes, else None (model dropped)."""
    if key in F and gate.get(key):
        return key
    b = BACKUP_KEY.get(key)
    return b if b in F and gate.get(b) else None
CONF_ARMS = ["neutral_own", "political", "workplace"]
ALPHA = 0.05


def load(fits_dir):
    F = {}
    for p in glob.glob(os.path.join(fits_dir, "*.json")):
        j = json.load(open(p)); F[os.path.basename(p)[:-5]] = {a["arm"]: a for a in j["arms"]}   # key = file stem
    return F


def main(fits_dir=os.path.join(ROOT, "results", "llm", "fits", "main"), comp_path=None, out=None):
    F = load(fits_dir)
    comp_path = comp_path or os.path.join(ROOT, "results", "llm", "comprehension_summary.csv")
    comp = pd.read_csv(comp_path).set_index("key") if os.path.exists(comp_path) else pd.DataFrame()
    gate = {k: not (pooled_sd(a) > NOISE_GATE) for k, a in F.items()}
    used = {resolve(k, F, gate) for k in WORKHORSES if k not in BACKUP_KEY.values()} - {None}
    rows = []
    for key, arms in F.items():
        for arm, r in arms.items():
            c = r["coef"]
            row = {"key": key, "arm": arm, "n_prompts": r["n_prompts"], "excluded_frac": r["excluded_frac"],
                   "repeat_sd": r.get("repeat_sd"), "perm_sd": r.get("perm_sd"),
                   "endpoint_repeat_sd": pooled_sd(arms), "noise_ok": gate[key],
                   "comprehension_ok": bool(comp.loc[key, "passes"]) if key in comp.index else None}
            for n in ("b_a", "b_r", "gamma", "h", "h_C", "h_L", "h_O", "beta", "b_r_minus_b_a", "w_r_over_w_a", "alpha"):
                if n in c:
                    row[n], row[n + "_lo"], row[n + "_hi"], row["p_" + n] = c[n]["est"], c[n]["lo"], c[n]["hi"], c[n]["p"]
            for blk, t in r["H3"].items():
                row[f"H3_{blk}_p"] = t["p"]; row[f"H3_{blk}_cvrel"] = t["cv_rel_improvement"]
                row[f"H3_{blk}_fracdP"] = t["frac_cells_dP_ge_0.05"]; row[f"H3_{blk}_mineff"] = t["passes_min_effect"]
            row["confirmatory"] = (key in used and arm in CONF_ARMS and row["comprehension_ok"] is True)
            rows.append(row)
    T = pd.DataFrame(rows)
    C = T[T["confirmatory"]].copy()
    # Holm families
    def fam(col, mask=None, name=None):
        sub = C if mask is None else C[mask]
        adj = holm(list(sub[col].astype(float)))
        T.loc[sub.index, name or (col + "_holm")] = adj
    fam("p_b_r_minus_b_a", name="H1_holm")
    # letter field: neutral h and content-framing h_L form one family; content field h_C its own family
    letter = pd.concat([C.loc[C["arm"] == "neutral_own", "p_h"], C.loc[C["arm"] != "neutral_own", "p_h_L"]])
    T.loc[letter.index, "H2_letter_holm"] = holm(list(letter.astype(float)))
    cont = C.loc[C["arm"] != "neutral_own", "p_h_C"]
    T.loc[cont.index, "H2_content_holm"] = holm(list(cont.astype(float)))
    for blk in ("pairwise", "cubic"):
        fam(f"H3_{blk}_p", name=f"H3_{blk}_holm")
        T[f"H3_{blk}_supported"] = (T[f"H3_{blk}_holm"] < ALPHA) & T[f"H3_{blk}_mineff"].astype(bool)
    T["H1_supported"] = T["H1_holm"] < ALPHA
    T["H5_departs_from_sum"] = (T["alpha_lo"] > 0) | (T["alpha_hi"] < 0)
    T["H5_interpretable"] = ~(T["H3_pairwise_supported"].fillna(False) | T["H3_cubic_supported"].fillna(False))
    # H4: one-sided bootstrap test of beta_large > beta_small, per family x framing, comprehension-gated
    h4 = []
    rng = np.random.default_rng(4)
    for small0, large0 in H4_PAIRS:
        small, large = resolve(small0, F, gate) or small0, resolve(large0, F, gate) or large0
        for arm in CONF_ARMS:
            ok = all(k in F and arm in F[k] for k in (small, large))
            gated = [k for k in (small, large) if k in comp.index and not bool(comp.loc[k, "passes"])]
            row = {"family_pair": f"{small} -> {large}", "arm": arm}
            dropped = [k for k in (small0, large0) if resolve(k, F, gate) is None and (k in F)]
            if dropped:
                row["status"] = "excluded: noise gate failed, no passing backup (" + ", ".join(dropped) + ")"; h4.append(row); continue
            if not ok:
                row["status"] = "missing"; h4.append(row); continue
            bs, bl = np.array(F[small][arm]["draws"]["beta"]), np.array(F[large][arm]["draws"]["beta"])
            diff = rng.permutation(bl) - rng.permutation(bs)
            row.update(beta_small=F[small][arm]["coef"]["beta"]["est"], beta_large=F[large][arm]["coef"]["beta"]["est"],
                       diff_lo=float(np.percentile(diff, 2.5)), diff_hi=float(np.percentile(diff, 97.5)),
                       p_one_sided=float(max((diff <= 0).mean(), 1 / len(diff))),
                       status="excluded: does not parse task (" + ", ".join(gated) + ")" if gated else "tested")
            h4.append(row)
    H4 = pd.DataFrame(h4)
    tested = H4["status"] == "tested"
    H4.loc[tested, "holm"] = holm(list(H4.loc[tested, "p_one_sided"]))
    H4["significant"] = H4["holm"] < ALPHA
    n_neu = int(((H4["arm"] == "neutral_own") & H4["significant"].fillna(False)).sum())
    n_testable = int(((H4["arm"] == "neutral_own") & tested).sum())
    verdict = ("supported" if n_neu >= 3 else "not supported") + f" ({n_neu} of {n_testable} testable families significant in neutral; ≥3 of 4 required)"
    out = out or os.path.join(ROOT, "results", "llm")
    T.drop(columns=[c for c in T if c.endswith("_mineff")]).to_csv(os.path.join(out, "stage1_cells.csv"), index=False)
    H4.to_csv(os.path.join(out, "stage1_H4.csv"), index=False)
    json.dump({"H4_verdict": verdict}, open(os.path.join(out, "stage1_verdicts.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    show = ["key", "arm", "confirmatory", "beta", "b_a", "b_r", "b_r_minus_b_a", "H1_holm", "gamma", "alpha",
            "H3_pairwise_holm", "H3_pairwise_supported", "H3_cubic_holm", "H3_cubic_supported", "repeat_sd", "excluded_frac"]
    print(T[[c for c in show if c in T]].sort_values(["key", "arm"]).round(3).to_string(index=False))
    print(H4.round(4).to_string(index=False)); print("H4:", verdict)
    return T, H4


if __name__ == "__main__":
    main(*sys.argv[1:])
