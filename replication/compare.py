"""Compare regenerated result files with the committed reference copies.

Profiles (stated tolerances; |new - ref| <= atol + rtol * |ref| for every numeric value):
  exact   rtol 1e-12, atol 1e-12   deterministic code on the pinned environment (simulations, fits, tables)
  fit     rtol 1e-6,  atol 1e-9    iterative optimisers whose last digits may move with BLAS thread scheduling
  print4  rtol 0,     atol 6e-5    values that were transcribed at 4 decimals
  boot    per-column               bootstrap CI bounds whose generator state depends on an unrecorded call order
  figure  file exists, is a non-empty image, and has the reference's pixel size (bytes are not compared: fonts,
          codecs and library versions change them without changing the content)
  media   file exists and is non-empty (MP4/GIF)
  text    the file exists and is non-empty (logs, Markdown tables)

JSON fields present only in the regenerated file (added by a later version of the producing script) are listed as
"extra" in the record, not counted as differences; every committed field must be reproduced.
Strings must match exactly except for keys that record where or when a file was made (IGNORE_KEYS) and values that
are absolute paths.
"""
import json, math, os, struct
import numpy as np
import pandas as pd

PROFILES = {"exact": (1e-12, 1e-12), "fit": (1e-6, 1e-9), "print4": (0.0, 6e-5)}
IGNORE_KEYS = {"source", "path", "paths", "created", "created_utc", "time", "timestamp", "generated", "generated_utc",
               "git", "commit", "elapsed", "elapsed_s", "runtime", "runtime_s", "wall", "wall_s", "host", "date",
               "written", "seconds",
               "gif_bytes"}   # encoded media sizes depend on the ffmpeg build


def _num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _close(a, b, rtol, atol):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= atol + rtol * abs(a)


def _pathlike(s):
    return isinstance(s, str) and (s.startswith("/") or "\\" in s)


def compare_json(ref, new, rtol, atol):
    """Return (n_values, n_bad, worst_abs_diff, first_problems)."""
    st = {"n": 0, "bad": 0, "worst": 0.0, "probs": [], "extra": []}

    def walk(a, b, p):
        if isinstance(a, dict):
            if not isinstance(b, dict):
                return bad(p, "type")
            for k in a:
                if k in IGNORE_KEYS:
                    continue
                if k not in b:
                    bad(p + "/" + str(k), "missing"); continue
                walk(a[k], b[k], p + "/" + str(k))
            for k in b:   # fields added by later code versions are reported, not failed (committed fields must match)
                if k not in a and k not in IGNORE_KEYS:
                    st["extra"].append(p + "/" + str(k))
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                return bad(p, f"length {len(a)} vs {len(b) if isinstance(b, list) else type(b).__name__}")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{p}[{i}]")
        elif _num(a) or a is None:
            st["n"] += 1
            if _num(a) and _num(b):
                d = abs(float(a) - float(b))
                if not math.isnan(d):
                    st["worst"] = max(st["worst"], d)
            if not ((a is None and b is None) or (_num(a) and _num(b) and _close(a, b, rtol, atol))):
                bad(p, f"{a!r} vs {b!r}")
        else:
            st["n"] += 1
            if a != b and not (_pathlike(a) and _pathlike(b)):
                bad(p, f"{str(a)[:60]!r} vs {str(b)[:60]!r}")

    def bad(p, why):
        st["bad"] += 1
        if len(st["probs"]) < 5:
            st["probs"].append(f"{p}: {why}")

    walk(ref, new, "")
    return st


def read_table(path):
    comp = "gzip" if path.endswith(".gz") else None
    return pd.read_csv(path, compression=comp, keep_default_na=False, na_values=["", "nan", "NaN"], low_memory=False)


def compare_csv(ref_path, new_path, rtol, atol, col_tol=None):
    a, b = read_table(ref_path), read_table(new_path)
    st = {"n": 0, "bad": 0, "worst": 0.0, "probs": []}
    if list(a.columns) != list(b.columns):
        miss = [c for c in a.columns if c not in b.columns]
        if miss:
            st["bad"] += 1; st["probs"].append(f"columns missing: {miss[:5]}")
            return st
        b = b[a.columns]
    if len(a) != len(b):
        st["bad"] += 1; st["probs"].append(f"rows {len(a)} vs {len(b)}")
        return st
    for c in a.columns:
        if c in IGNORE_KEYS:
            continue
        x, y = a[c], b[c]
        r_, a_ = (col_tol or {}).get(c, (rtol, atol))
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        numeric = xn.notna().sum() == x.notna().sum() and yn.notna().sum() == y.notna().sum()
        st["n"] += len(x)
        if numeric:
            xv, yv = xn.to_numpy(float), yn.to_numpy(float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            d = np.abs(xv - yv)
            ok = both_nan | (d <= a_ + r_ * np.abs(xv)) | ((xv == yv) & np.isinf(xv))
            fin = np.isfinite(d)
            if fin.any():
                st["worst"] = max(st["worst"], float(d[fin].max()))
            nb = int((~ok).sum())
            if nb:
                st["bad"] += nb
                i = int(np.argmax(~ok))
                st["probs"].append(f"{c}[{i}]: {xv[i]!r} vs {yv[i]!r} ({nb} cells)")
        else:
            xs, ys = x.astype(str), y.astype(str)
            ok = (xs == ys) | (xs.map(_pathlike) & ys.map(_pathlike))
            nb = int((~ok).sum())
            if nb:
                st["bad"] += nb
                i = int(np.argmax(~ok.to_numpy()))
                st["probs"].append(f"{c}[{i}]: {xs.iloc[i][:40]!r} vs {ys.iloc[i][:40]!r} ({nb} cells)")
    return st


def png_size(path):
    with open(path, "rb") as f:
        h = f.read(24)
    if h[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", h[16:24])


def compare_file(ref, new, profile, col_tol=None):
    """Compare one file; returns dict(ok, n, bad, worst, probs, profile)."""
    if not os.path.exists(new):
        return dict(ok=False, n=0, bad=1, worst=None, probs=["not regenerated"], profile=profile)
    if os.path.getsize(new) == 0:
        return dict(ok=False, n=0, bad=1, worst=None, probs=["empty file"], profile=profile)
    if profile == "figure":
        a, b = png_size(ref), png_size(new)
        ok = b is not None and a == b
        return dict(ok=ok, n=1, bad=0 if ok else 1, worst=None, profile=profile,
                    probs=[] if ok else [f"image size {a} vs {b}"])
    if profile in ("media", "text"):
        return dict(ok=True, n=1, bad=0, worst=None, probs=[], profile=profile)
    rtol, atol = PROFILES.get(profile, PROFILES["exact"])
    if new.endswith((".csv", ".csv.gz")):
        st = compare_csv(ref, new, rtol, atol, col_tol)
    elif new.endswith(".json"):
        st = compare_json(json.load(open(ref)), json.load(open(new)), rtol, atol)
    elif new.endswith(".jsonl"):
        r = [json.loads(l) for l in open(ref) if l.strip()]
        n = [json.loads(l) for l in open(new) if l.strip()]
        st = compare_json(r, n, rtol, atol)
    elif new.endswith(".npy"):
        a, b = np.load(ref, allow_pickle=False), np.load(new, allow_pickle=False)
        ok = a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        st = dict(n=int(a.size), bad=0 if ok else 1, worst=float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else None,
                  probs=[] if ok else ["array differs"])
    else:
        same = open(ref, "rb").read() == open(new, "rb").read()
        st = dict(n=1, bad=0 if same else 1, worst=None, probs=[] if same else ["bytes differ"])
    st.update(ok=st["bad"] == 0, profile=profile)
    return st
