"""Null operating characteristic of the Amendment 4 local-vs-provider verdict rule (synthetic data only).

  python llm/compare_local_null.py [--R 100] [--B 1000] [--noise 0.3] [--workers 4]

R independent pairs of synthetic endpoints with IDENTICAL true coefficients (llm/test_compare_local.synth, same row
layout as the real local CSVs). For each, compare_local.compare_pair with the same B as the real comparison;
reports how often the pre-registered rule (≥ 90% of coefficients agree AND slope CI contains 1) returns 'jitter
averages out', and the distribution of the agreement fraction (so the real pairs' agreement can be placed in it).
Replications run in parallel (one per worker, each sequential inside) and each is appended to
results/llm/amendment4/compare_rule_null_runs.jsonl the moment it finishes; a rerun skips finished replications.
Summary: results/llm/amendment4/compare_rule_null_B{B}.json.
"""
import argparse, json, os, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_local as C, test_compare_local as TC

RUNS = os.path.join(C.ROOT, "results/llm/amendment4/compare_rule_null_runs.jsonl")


def one(args):
    r, B, noise = args
    src = os.path.join(C.ROOT, "results/llm/stage1/main/gemma-31b-local.csv")
    tmp = tempfile.mkdtemp()
    TC.synth(src, 1.0, noise, 1000 + 2 * r).to_csv(os.path.join(tmp, "l.csv"), index=False)
    TC.synth(src, 1.0, noise, 1001 + 2 * r).to_csv(os.path.join(tmp, "p.csv"), index=False)
    _, s = C.compare_pair("l", "p", "null", B, tmp, tmp, 1, seed=50 + r)
    return dict(run=r, B=B, noise=noise, frac_agree=s["frac_agree"], slope=s["slope"],
                slope_ci_contains_1=s["slope_ci_contains_1"], verdict=s["verdict"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=100); ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.3); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(RUNS), exist_ok=True)
    done = [json.loads(l) for l in open(RUNS)] if os.path.exists(RUNS) else []
    done = {o["run"]: o for o in done if o.get("B") == a.B and o.get("noise") == a.noise}
    todo = [(r, a.B, a.noise) for r in range(a.R) if r not in done]
    print(f"null check B={a.B}: {len(done)} done, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, t) for t in todo]):
            o = f.result(); done[o["run"]] = o
            with open(RUNS, "a") as fh:
                fh.write(json.dumps(o) + "\n")
            print(f"run {o['run']}: agree {o['frac_agree']:.3f} slope {o['slope']:.3f} -> {o['verdict']} "
                  f"({len(done)}/{a.R})", flush=True)
    out = [done[r] for r in sorted(done) if r < a.R]
    f = np.array([o["frac_agree"] for o in out])
    summ = dict(R=len(out), B=a.B, noise=a.noise,
                p_verdict_averages_out=float(np.mean([o["verdict"] == "jitter averages out" for o in out])),
                p_frac_agree_ge_0_9=float((f >= 0.9).mean()),
                p_slope_ci_contains_1=float(np.mean([o["slope_ci_contains_1"] for o in out])),
                frac_agree_mean=float(f.mean()), frac_agree_quantiles={q: float(np.quantile(f, q)) for q in (0.01, 0.05, 0.1, 0.5)},
                p_frac_agree_le_observed_0_759=float((f <= 22 / 29 + 1e-9).mean()), runs=out)
    json.dump(summ, open(os.path.join(C.ROOT, f"results/llm/amendment4/compare_rule_null_B{a.B}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
