import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import glob, re
import numpy as np
from vbsim import analytic_TF, analytic_TSG

D = _os.path.join(ROOT, "data")
rng = np.random.default_rng(0)


def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{D}/{tag}_N*.npz")):
        z = np.load(f)
        N = int(z["N"])
        acc, cnt = z["acc"], z["cnt"]
        last = len(cnt) - 1
        out[N] = dict(
            Ts=z["Ts"], c=float(z["c"]), rho=float(z["rho"]), swap=z["swap"],
            m=acc[:, last] / cnt[last],          # [S, K, obs] thermal means, last log bin
            prev=acc[:, last - 1] / cnt[last - 1],
        )
    return out


def binder(m, i2, i4, idx=None):
    if idx is not None:
        m = m[idx]
    a2 = m[:, :, i2].mean(0)
    a4 = m[:, :, i4].mean(0)
    return 0.5 * (3 - a4 / a2 ** 2)


def scaled(m, i2, N, power, idx=None):
    if idx is not None:
        m = m[idx]
    return N ** power * m[:, :, i2].mean(0)


def crossing(T, y1, y2):
    d = y1 - y2
    s = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(s) == 0:
        return np.nan
    k = s[0]
    return T[k] - d[k] * (T[k + 1] - T[k]) / (d[k + 1] - d[k])


def boot_cross(r1, r2, kind, i2, i4, power=None, B=1000):
    T = r1["Ts"]; T2 = r2["Ts"]
    S1, S2 = len(r1["m"]), len(r2["m"])
    def f(r, N, idx):
        if kind == "binder":
            y = binder(r["m"], i2, i4, idx)
        else:
            y = scaled(r["m"], i2, N, power, idx)
        return np.interp(T, r["Ts"], y)
    N1, N2 = r1["N"], r2["N"]
    est = crossing(T, f(r1, N1, None), f(r2, N2, None))
    bs = []
    for _ in range(B):
        bs.append(crossing(T, f(r1, N1, rng.integers(0, S1, S1)), f(r2, N2, rng.integers(0, S2, S2))))
    bs = np.array(bs)
    ok = np.isfinite(bs)
    lo, hi = np.percentile(bs[ok], [2.5, 97.5]) if ok.sum() > 20 else (np.nan, np.nan)
    return est, lo, hi, ok.mean()


def equil_check(r, i2):
    """Relative change of disorder-averaged <x^2> between the last two log bins, per T."""
    a = r["m"][:, :, i2]
    b = r["prev"][:, :, i2]
    d = a - b
    return d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d)), a.mean(0)


def summarize(tag, kind):
    runs = load(tag)
    if not runs:
        return None
    Ns = sorted(runs)
    for N in Ns:
        runs[N]["N"] = N
    any_r = runs[Ns[0]]
    c, rho = any_r["c"], any_r["rho"]
    pred = analytic_TSG(c) if kind == "sg" else analytic_TF(c, rho)
    i2, i4 = (0, 1) if kind == "sg" else (3, 4)
    power = 2 / 3 if kind == "sg" else 1 / 2   # mean-field FSS: chi ~ N^{1/3} (SG), N^{1/2} (F)
    print(f"\n== {tag}: c={c} rho={rho}  predicted T = {pred:.3f}  sizes {Ns}")
    for N in Ns:
        r = runs[N]
        d, se, a = equil_check(r, i2)
        k = 0  # lowest T is hardest
        print(f"  N={N:5d} S={len(r['m'])} min swap={r['swap'].min():.2f} "
              f"equil lowest-T: <x2>={a[k]:.4f} last-prev={d[k]:+.4f}±{se[k]:.4f}")
    res = []
    for N1, N2 in zip(Ns[:-1], Ns[1:]):
        b = boot_cross(runs[N1], runs[N2], "binder", i2, i4)
        s = boot_cross(runs[N1], runs[N2], "scaled", i2, i4, power)
        print(f"  cross {N1}/{N2}: Binder {b[0]:.3f} [{b[1]:.3f},{b[2]:.3f}] (found {b[3]:.0%})"
              f" | scaled-chi {s[0]:.3f} [{s[1]:.3f},{s[2]:.3f}] (found {s[3]:.0%})")
        res.append((N1, N2, b, s))
    return dict(runs=runs, pred=pred, res=res, c=c, rho=rho, kind=kind)


if __name__ == "__main__":
    import sys
    for tag in sys.argv[1:]:
        summarize(tag, "sg" if tag.startswith("sg") else "f")
