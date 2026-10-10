"""Null operating characteristic of the Amendment 4 local-vs-provider verdict rule (synthetic data only).

  python llm/compare_local_null.py [--R 20] [--B 100] [--noise 0.3]

R independent pairs of synthetic endpoints with IDENTICAL true coefficients (llm/test_compare_local.synth, same row
layout as the real local CSVs). For each, compare_local.compare_pair; reports how often the pre-registered rule
(≥ 90% of coefficients agree AND slope CI contains 1) returns 'jitter averages out', and the distribution of the
agreement fraction. Writes results/llm/amendment4/compare_rule_null.json.
"""
import argparse, json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_local as C, test_compare_local as TC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=20); ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.3)
    a = ap.parse_args()
    src = os.path.join(C.ROOT, "results/llm/stage1/main/gemma-31b-local.csv")
    tmp = tempfile.mkdtemp()
    runs_f = os.path.join(C.ROOT, "results/llm/amendment4/compare_rule_null_runs.jsonl")     # appended per run
    os.makedirs(os.path.dirname(runs_f), exist_ok=True)
    out = [json.loads(l) for l in open(runs_f)] if os.path.exists(runs_f) else []
    out = [o for o in out if o.get("B") == a.B and o.get("noise") == a.noise]
    for r in range(len(out), a.R):
        TC.synth(src, 1.0, a.noise, 1000 + 2 * r).to_csv(os.path.join(tmp, "l.csv"), index=False)
        TC.synth(src, 1.0, a.noise, 1001 + 2 * r).to_csv(os.path.join(tmp, "p.csv"), index=False)
        _, s = C.compare_pair("l", "p", "null", a.B, tmp, tmp, 4, seed=50 + r)
        out.append(dict(run=r, B=a.B, noise=a.noise, frac_agree=s["frac_agree"],
                        slope_ci_contains_1=s["slope_ci_contains_1"], verdict=s["verdict"]))
        open(runs_f, "a").write(json.dumps(out[-1]) + "\n")
        print(r, out[-1], flush=True)
    f = np.array([o["frac_agree"] for o in out])
    summ = dict(R=a.R, B=a.B, noise=a.noise, p_verdict_averages_out=float(np.mean([o["verdict"] == "jitter averages out" for o in out])),
                p_frac_agree_ge_0_9=float((f >= 0.9).mean()), p_slope_ci_contains_1=float(np.mean([o["slope_ci_contains_1"] for o in out])),
                frac_agree_mean=float(f.mean()), frac_agree_min=float(f.min()), runs=out)
    os.makedirs(os.path.join(C.ROOT, "results/llm/amendment4"), exist_ok=True)
    json.dump(summ, open(os.path.join(C.ROOT, "results/llm/amendment4/compare_rule_null.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
