"""Score the comprehension battery (PREREG Amendment 3a).

  python ../llm/comprehension.py results/llm/stage1/comprehension/*.csv

A prompt is correct if the renormalised P(correct letter) > 0.5 and leak <= 5%; a leaked prompt counts as
incorrect. An endpoint PASSES iff accuracy >= 90% on each of the three question types (own, ally_majority,
rival_majority), pooled over k = 1..6 and both letter orders. Writes results/llm/comprehension_summary.csv.
"""
import os, sys
import pandas as pd

PASS = 0.90
LEAK_MAX = 0.05


def score(path):
    d = pd.read_csv(path, dtype={"counts": str})
    d["k"] = d["counts"].str.zfill(4).map(lambda s: sum(int(c) for c in s))
    p_correct = d["pA"].where(d["correct"] == "A", 1 - d["pA"])
    d["ok"] = (p_correct > 0.5) & (d["leak"] <= LEAK_MAX) & d["pA"].notna()
    d["p_correct"] = p_correct
    row = {"key": d["key"].iloc[0], "model": d["model"].iloc[0], "tag": d["tag"].iloc[0], "n": len(d)}
    for q, g in d.groupby("question"):
        row[f"acc_{q}"] = round(g["ok"].mean(), 4)
        row[f"meanP_{q}"] = round(g["p_correct"].mean(), 4)
        row[f"worst_k_{q}"] = round(g.groupby("k")["ok"].mean().min(), 3)
    row["leak_gt5"] = int((d["leak"] > LEAK_MAX).sum())
    row["passes"] = all(row.get(f"acc_{q}", 0) >= PASS for q in ("own", "ally_majority", "rival_majority"))
    return row


if __name__ == "__main__":
    rows = [score(p) for p in sys.argv[1:]]
    out = pd.DataFrame(rows)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out.to_csv(os.path.join(root, "results", "llm", "comprehension_summary.csv"), index=False)
    print(out.to_string(index=False))
