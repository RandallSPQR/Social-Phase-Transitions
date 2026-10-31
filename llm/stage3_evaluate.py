"""Pre-stated Stage 3 evaluation (llm/STAGE3_PREDICTIONS.md, 'Pre-stated evaluation' + addendum b). Mechanical only:
observed mean of 5 graphs vs the committed 90% mean-of-5 intervals.   python llm/stage3_evaluate.py"""
import json, os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
obs = pd.read_csv(os.path.join(ROOT, "results/llm/stage3/summary.csv")).set_index("point")
pred = json.load(open(os.path.join(ROOT, "results/llm/stage2/stage3_predictions.json")))
nc = json.load(open(os.path.join(ROOT, "results/llm/stage2/stage3_predictions_nocontacts.json")))
M = ["abs_m", "m", "persistence", "excess_unsat", "flip_rate", "abs_q"]
inside = lambda p, mode, m, x: p[f"{mode}_{m}_mean5_lo"] <= x <= p[f"{mode}_{m}_mean5_hi"]
rows, verdict = [], {}
for p in pred:
    pid = p["id"].split(" ")[0]
    for m in M:
        x = obs.loc[pid, m]
        rows.append(dict(point=pid, measure=m, primary=m in p["primary"], observed=round(x, 3),
                         surrogate=f"[{p[f'fitted_{m}_mean5_lo']:.2f}, {p[f'fitted_{m}_mean5_hi']:.2f}]",
                         surrogate_pass=inside(p, "fitted", m, x),
                         fields_only=f"[{p[f'fields_only_{m}_mean5_lo']:.2f}, {p[f'fields_only_{m}_mean5_hi']:.2f}]",
                         outside_fields_only=not inside(p, "fields_only", m, x),
                         ising=f"[{p[f'ising_{m}_mean5_lo']:.2f}, {p[f'ising_{m}_mean5_hi']:.2f}]",
                         outside_ising=not inside(p, "ising", m, x)))
    prim = [r for r in rows if r["point"] == pid and r["primary"]]
    passes = all(r["surrogate_pass"] for r in prim)
    if pid == "P3":
        disc = prim[0]["outside_fields_only"] and prim[0]["outside_ising"] and prim[1]["outside_ising"]
    else:
        disc = all(r["outside_fields_only"] and r["outside_ising"] for r in prim)
    verdict[pid] = dict(surrogate_passes_both_primary=passes, discriminating=disc,
                        ising_rejected=all(r["outside_ising"] for r in prim))
for p in nc:
    pid = p["id"]
    for m in M:
        x = obs.loc[pid, m]
        rows.append(dict(point=pid, measure=m, primary=m in p["primary"], observed=round(x, 3),
                         surrogate=f"[{p[f'predicted_{m}_mean5_lo']:.2f}, {p[f'predicted_{m}_mean5_hi']:.2f}]",
                         surrogate_pass=p[f"predicted_{m}_mean5_lo"] <= x <= p[f"predicted_{m}_mean5_hi"]))
    verdict[pid] = dict(passes_both_primary=all(r["surrogate_pass"] for r in rows if r["point"] == pid and r["primary"]))
n_pass = sum(verdict[k]["surrogate_passes_both_primary"] for k in ("P1", "P2", "P3", "P4"))
out = {"per_point": verdict, "P1_P4_points_passing_both_primary": n_pass,
       "overall_surrogate_supported (>=3 of 4)": n_pass >= 3,
       "amplification_observed": {k: {m: round(obs.loc[k, m] - obs.loc[k + "-nc", m], 3) for m in ("m", "persistence")} for k in ("P2", "P4")}}
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/llm/stage3/evaluation.csv"), index=False)
json.dump(out, open(os.path.join(ROOT, "results/llm/stage3/evaluation.json"), "w"), indent=1, default=bool)
print(pd.DataFrame(rows)[lambda d: d.primary].to_string(index=False)); print(json.dumps(out, indent=1, default=bool))
