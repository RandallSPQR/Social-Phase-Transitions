"""Pre-registered predictions for the fresh live replicate (llm/RESIDUAL_TEST_PLAN.md). No API calls.

  python llm/replicate_predictions.py

Same 5 graphs per point, fresh starts (stage3.initial_state with run_tag "rep1"). Reference kernel: K_avg for P1–P5
(k_avg.kavg_table from the order-marginalised battery). For the nc arms: K_live from the original Stage 3 run
(secondary; the primary nc reference, the k = 0 K_avg battery, is computed after its 32 calls and committed before
any replicate chain). R = 400 runs; mean, SD and the 90% interval of the mean of 5 per measure.
Output: results/llm/replicate/predictions.json
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3, kernel_test as K, k_avg as KA
from endpoints import E

TAG, R = "rep1", 400
OUT = os.path.join(K.ROOT, "results", "llm", "replicate")


def starts(pid):
    return [np.stack([stage3.initial_state(pid, g, r, TAG) for r in (0, 1)]) for g in range(stage3.N_GRAPHS)]


def summarise(df):
    return {m: dict(mean=float(df[m].mean()), sd=float(df[m].std(ddof=1)), lo=float(np.percentile(df[m], 5)),
                    hi=float(np.percentile(df[m], 95))) for m in K.MEAS}


def main():
    os.makedirs(OUT, exist_ok=True)
    calls = pd.read_csv(os.path.join(K.OUT, "k_avg_calls.csv"), dtype={"counts": str, "seq": str}); calls["counts"] = calls.counts.str.zfill(4)
    out = {"run_tag": TAG, "R": R, "points": {}}
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        S0 = starts(pid)
        if contacts:
            T0 = KA.kavg_table(pid, calls)
            if E[key][2] == "sample":
                rng = np.random.default_rng(stage3._seed("rep1-kavg-beta", pid)); tabs = [KA.kavg_table(pid, calls, rng) for _ in range(R)]
                fn = lambda run, g: tabs[run]
            else:
                fn = lambda run, g, T0=T0: T0
            ref = "K_avg"
        else:
            d = K.live_rows(pid); T0, _ = K.live_kernel(d.groupby("idx").agg(S1=("p", "sum"), n=("p", "size")), K.N0_MAIN)
            fn = lambda run, g, T0=T0: T0; ref = "K_live (original run; secondary)"
        df = K.simulate(pid, fn, R, "rep1-pred", starts=S0)
        out["points"][pid] = dict(reference=ref, starts_m=[float(s.mean()) for s in S0], **summarise(df))
        p = out["points"][pid]
        print(pid, ref, {m: f"{p[m]['mean']:.3f} [{p[m]['lo']:.3f}, {p[m]['hi']:.3f}]" for m in ("abs_m", "m", "persistence", "flip_rate")}, flush=True)
    json.dump(out, open(os.path.join(OUT, "predictions.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
