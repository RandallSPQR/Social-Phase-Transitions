"""Comparator canary: prove that replication/compare.py notices the smallest corruption, before any result is trusted.

  python replication/canary.py          # writes replication/out/canary.json; exit 1 if any case behaves wrongly

Each case writes a perturbed copy of a committed file to a temporary directory and compares it with the original
under the profile the replication itself uses for that file. A case passes only if the comparator's verdict is the
expected one.

Expected to FAIL the comparison (the comparator must reject them):
  1. byte_in_array    data/sg_c4_N250.npz (a base.sim output), one byte flipped inside one float64 of the `acc`
                      array, re-saved as a valid .npz. The value changes in its low-order bits.
  2. byte_in_file     the same file with one raw byte flipped inside the stored array data. The zip CRC then fails, and
                      the driver records that as a failed comparison, never as a pass.
  3. last_decimal_csv results/llm/kernel_test/k_avg_amplification.csv, the last decimal digit of the shortest
                      round-trip form of one value (P1, K_avg vs K_live |dP|) changed, profile "exact".
  4. last_decimal_json results/llm/confirm/results/step4.json, the last decimal digit of the decisive D (P1,
                      stay-bias) changed, profile "exact".
"Last decimal place" means the smallest change a stored double can undergo: the next representable value (1 ulp),
which changes the last significant digit(s) of its decimal form. A smaller decimal edit would parse to the same
double and be invisible to any comparison.
Expected to PASS (controls, so a comparator that rejects everything cannot pass the canary):
  5. identical_npz, 6. identical_csv: unmodified copies.
"""
import json, os, shutil, sys, tempfile, zipfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "replication"))
import compare  # noqa: E402

NPZ = "data/sg_c4_N250.npz"
CSV = "results/llm/kernel_test/k_avg_amplification.csv"   # compared with profile "exact"
JSN = "results/llm/confirm/results/step4.json"


def verdict(ref, new, profile):
    """The driver's view: a comparator exception counts as a failed comparison (replicate.compare_step)."""
    try:
        r = compare.compare_file(ref, new, profile)
        return bool(r["ok"]), "; ".join(r["probs"][:1])
    except Exception as e:
        return False, f"comparison error: {type(e).__name__}"


def bump_last_digit(s):
    """The smallest possible change to a stored value: the next representable double (one unit in the last place),
    written in its shortest round-trip form. In decimal this changes the last significant digit or two
    (0.0037564763171786066 -> 0.003756476317178607). Any smaller decimal edit would parse to the same double."""
    x = float(s); t = repr(float(np.nextafter(x, np.inf)))
    assert float(t) != x, (s, t)
    return t


def main():
    tmp = tempfile.mkdtemp(prefix="canary_")
    cases = []

    def case(name, ref, new, profile, expect_ok, what):
        ok, why = verdict(ref, new, profile)
        cases.append(dict(case=name, file=os.path.relpath(ref, ROOT), profile=profile, perturbation=what,
                          comparator_ok=ok, expected_ok=expect_ok, correct=ok == expect_ok, detail=why))

    ref_npz = os.path.join(ROOT, NPZ)
    z = dict(np.load(ref_npz))

    # 1. one byte flipped inside one float64, saved as a valid npz
    a = z["acc"].copy(); b = a.view(np.uint8).reshape(-1)
    i = 8 * 12345 + 1                    # 2nd least significant byte of element 12345 (little-endian float64)
    old = float(a.reshape(-1)[12345]); b[i] ^= 0x01
    p1 = os.path.join(tmp, "byte_in_array.npz"); np.savez(p1, **{**z, "acc": a})
    case("byte_in_array", ref_npz, p1, "exact", False,
         f"acc.flat[12345] byte 1 xor 0x01: {old!r} -> {float(a.reshape(-1)[12345])!r}")

    # 2. one raw byte flipped inside the stored data of the committed file
    p2 = os.path.join(tmp, "byte_in_file.npz"); shutil.copy(ref_npz, p2)
    with zipfile.ZipFile(ref_npz) as zf:
        info = zf.getinfo("acc.npy")
    off = info.header_offset + 30 + len(info.filename) + len(info.extra) + 4096   # inside acc.npy's data
    with open(p2, "r+b") as f:
        f.seek(off); x = f.read(1); f.seek(off); f.write(bytes([x[0] ^ 0x01]))
    case("byte_in_file", ref_npz, p2, "exact", False, f"raw byte at offset {off} xor 0x01")

    # 3. last printed decimal digit of one CSV value
    ref_csv = os.path.join(ROOT, CSV); lines = open(ref_csv).read().split("\n")
    head = lines[0].split(","); col = head.index("K_avg_vs_K_live_dP")
    cells = lines[1].split(","); before = cells[col]; cells[col] = bump_last_digit(before)
    lines[1] = ",".join(cells); p3 = os.path.join(tmp, "last_decimal.csv"); open(p3, "w").write("\n".join(lines))
    case("last_decimal_csv", ref_csv, p3, "exact", False, f"P1 K_avg_vs_K_live_dP: {before} -> {cells[col]}")

    # 4. last printed decimal digit of the decisive Rule A statistic
    ref_json = os.path.join(ROOT, JSN); txt = open(ref_json).read()
    d = json.loads(txt)["P1"]["abs_m"]["stay"]["D"]; s = repr(d); t = bump_last_digit(s)
    assert txt.count(s) == 1, "the perturbed value must be unique in the file"
    p4 = os.path.join(tmp, "last_decimal.json"); open(p4, "w").write(txt.replace(s, t, 1))
    case("last_decimal_json", ref_json, p4, "exact", False, f"P1/abs_m/stay/D: {s} -> {t}")

    # 5-6. controls: identical copies must pass
    p5 = os.path.join(tmp, "identical.npz"); shutil.copy(ref_npz, p5)
    case("identical_npz", ref_npz, p5, "exact", True, "none")
    p6 = os.path.join(tmp, "identical.csv"); shutil.copy(ref_csv, p6)
    case("identical_csv", ref_csv, p6, "exact", True, "none")

    shutil.rmtree(tmp)
    out = os.path.join(ROOT, "replication", "out"); os.makedirs(out, exist_ok=True)
    allok = all(c["correct"] for c in cases)
    json.dump(dict(all_correct=allok, cases=cases), open(os.path.join(out, "canary.json"), "w"), indent=1)
    for c in cases:
        print(f"{'ok ' if c['correct'] else 'BAD'} {c['case']:<18} expected {'pass' if c['expected_ok'] else 'FAIL'}, "
              f"comparator {'pass' if c['comparator_ok'] else 'FAIL'}  ({c['perturbation']}){'  ' + c['detail'] if c['detail'] else ''}",
              flush=True)
    print("canary:", "all cases behave as expected" if allok else "COMPARATOR DOES NOT DETECT A PERTURBATION", flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
