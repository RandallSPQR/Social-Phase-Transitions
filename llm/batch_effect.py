"""Amendment 5, exploratory: the batch-composition effect in local Gemma-26B (MoE), per prompt.

  python llm/batch_effect.py

Compares gemma-26b-local (Amendment 4: batched, two replicates with different batch composition) with
gemma-26b-local-bs1 (Amendment 5: batch size 1, deterministic; replicates identical) on the same prompts.
Logit scale (clipped at ±6, as analysis.py). Reports: the spread of each batched replicate around the batch-size-1
value, whether the batched mean is biased relative to it, and which cells move most (by arm, by number of contacts k,
by own position shown). Output: results/llm/amendment5/batch_effect.json + batch_effect_by_cell.csv.
"""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDC = ["arm", "counts", "s0", "mapping", "order", "perm"]


def lg(p):
    p = np.clip(p.astype(float), 1e-12, 1 - 1e-12)
    return np.clip(np.log(p / (1 - p)), -6, 6)


def main():
    a = pd.read_csv(os.path.join(ROOT, "results/llm/stage1/main/gemma-26b-local.csv"), dtype={"counts": str})
    b = pd.read_csv(os.path.join(ROOT, "results/llm/stage1/main/gemma-26b-local-bs1.csv"), dtype={"counts": str})
    A = a.pivot_table(index=IDC, columns="rep", values="pA"); Bv = b[b.rep == 0].set_index(IDC)["pA"]
    J = pd.DataFrame({"r0": lg(A[0]), "r1": lg(A[1]), "bs1": lg(Bv.reindex(A.index))}).dropna()
    J["d0"], J["d1"] = J.r0 - J.bs1, J.r1 - J.bs1
    J["dmean"] = (J.r0 + J.r1) / 2 - J.bs1
    J = J.reset_index(); J["k"] = J["counts"].map(lambda c: sum(int(x) for x in c.zfill(4)))
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    out = {
        "n_prompts": int(len(J)),
        "rms_batched_minus_bs1_logit": rms(np.r_[J.d0, J.d1]),
        "rms_rep0_minus_rep1_over_sqrt2": rms((J.r0 - J.r1) / np.sqrt(2)),
        "mean_batched_minus_bs1_logit": float(np.r_[J.d0, J.d1].mean()),
        "corr_batched_mean_vs_bs1": float(np.corrcoef((J.r0 + J.r1) / 2, J.bs1)[0, 1]),
        "share_prompts_majority_flip": float((((J.r0 + J.r1) / 2 > 0) != (J.bs1 > 0)).mean()),
        "by_arm": {k: rms(np.r_[g.d0, g.d1]) for k, g in J.groupby("arm")},
        "by_k": {int(k): rms(np.r_[g.d0, g.d1]) for k, g in J.groupby("k")},
        "by_own_shown": {("own shown" if k else "no own"): rms(np.r_[g.d0, g.d1]) for k, g in J.groupby(J.s0 != 0)},
    }
    os.makedirs(os.path.join(ROOT, "results/llm/amendment5"), exist_ok=True)
    J.groupby(["arm", "counts", "s0"]).apply(lambda g: pd.Series({"rms_dev": rms(np.r_[g.d0, g.d1]), "n": len(g)})) \
     .sort_values("rms_dev", ascending=False).to_csv(os.path.join(ROOT, "results/llm/amendment5/batch_effect_by_cell.csv"))
    json.dump(out, open(os.path.join(ROOT, "results/llm/amendment5/batch_effect.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
