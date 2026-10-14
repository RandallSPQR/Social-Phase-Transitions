"""Confirmatory amplification phase, Gemma-4 31B local (plan: llm/AMPLIFICATION_CONFIRM_PLAN.md §2.4, §3). GPU pod.

  python llm/local_gemma_confirm.py [--model-id google/gemma-4-31b-it] [--smoke]

Batch size 1, Amendment 5 deterministic kernels, official chat template, Stage 1 P(A) rules (top-5 primary, exact
full-vocabulary secondary). Units, each written to results/llm/confirm/gemma/ and uploaded + verified on a pod (RUN set):
  fingerprint  20 Stage 1 prompts vs the Amendment 4 gemma-31b-local values (RMS |Δ logit| must be <= 0.3; else stop)
  kavg_rep0    768 K_avg prompts (prompts_kavg.csv), natural order
  kavg_rep1    same prompts, reversed order; identity with rep0 is checked in the analysis
  comprehension  2,116 Stage 1 comprehension prompts
  garbage      first 100 K_avg prompts (SHA-256 order) x 20 sampled replies at T = 1.0, 1.5, 2.0 (softmax(logits/T), no
               top-p / top-k / repetition penalty, <= 8 new tokens) + exact first-token leak at each T
--smoke: 5 prompts per unit and 2 samples (pipeline test).
"""
import argparse, csv, hashlib, json, math, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "llm", "confirm", "gemma")
LETTERS = ("A", "B")


def parse_sample(content):            # = run_stage1.parse_sample (copied: that module imports the API client)
    if content is None:
        return None
    s = content.strip().strip("*").strip().strip("'\"`").strip()
    if s.endswith("."):
        s = s[:-1]
    if s in LETTERS:
        return s
    if len(s) > 1 and s[0] in LETTERS and not s[1].isalpha():
        return s[0]
    return None


def parse_top5(top):
    pa = sum(math.exp(l) for t, l in top if t.strip() == "A")
    pb = sum(math.exp(l) for t, l in top if t.strip() == "B")
    leak = max(0.0, 1.0 - pa - pb); cens = 0; floor = math.exp(min(l for _, l in top))
    if pa == 0: pa, cens = floor, 1
    if pb == 0: pb, cens = floor, 1
    return pa / (pa + pb), leak, cens


def progress(**kw):
    p = os.path.join(ROOT, "results", "llm", "logs", "progress.json"); os.makedirs(os.path.dirname(p), exist_ok=True)
    kw["t"] = time.time(); open(p + ".tmp", "w").write(json.dumps(kw)); os.replace(p + ".tmp", p)


def write(name, rows):
    os.makedirs(D, exist_ok=True); keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(D, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    if os.environ.get("RUN"):
        import pod_tracker
        assert pod_tracker.upload(os.environ["RUN"], "gemma-confirm-" + name), f"upload of {name} not verified"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model-id", default="google/gemma-4-31b-it")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp-gate", type=float, default=0.3, help="fingerprint gate (nats); raised only in CPU tests with random weights")
    a = ap.parse_args()
    open(os.path.join(ROOT, "results/llm/logs/local_gemma.pid"), "w").write(str(os.getpid()))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    progress(stage="loading model", model=a.model_id)
    tok = AutoTokenizer.from_pretrained(a.model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.bfloat16, device_map=a.device)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.model_id, dtype=torch.bfloat16, device_map=a.device)
    model.eval()
    A_ids = [i for i in range(len(tok)) if tok.decode([i]).strip() == "A"]
    B_ids = [i for i in range(len(tok)) if tok.decode([i]).strip() == "B"]
    AB = torch.tensor(A_ids + B_ids, device=a.device)
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, |A|={len(A_ids)} |B|={len(B_ids)}", flush=True)
    n_cap = 5 if a.smoke else None
    tmpl = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    def logits1(prompt):
        enc = tok([tmpl(prompt)], return_tensors="pt", add_special_tokens=False).to(a.device)
        with torch.no_grad():
            return model(**enc).logits[0, -1, :].float(), enc

    def lp_row(prompt):
        lg, _ = logits1(prompt); lp = torch.log_softmax(lg, -1); top = torch.topk(lp, 5)
        t5 = [(tok.decode([int(t)]), float(v)) for t, v in zip(top.indices, top.values)]
        pa, leak, cens = parse_top5(t5)
        pe, pbe = float(lp[A_ids].exp().sum()), float(lp[B_ids].exp().sum())
        return dict(pA=pa, leak=leak, censored=cens, pA_exact=pe / (pe + pbe) if pe + pbe > 0 else None, leak_exact=1 - pe - pbe,
                    top5=" | ".join(f"{t!r}:{v:.4f}" for t, v in t5))

    t0 = time.time()
    def unit(name, items, fn):
        rows = []
        for j, it in enumerate(items[:n_cap] if n_cap else items):
            rows.append({**{k: v for k, v in it.items() if k != "prompt"}, **fn(it)})
            if j % 25 == 0:
                el = time.time() - t0
                progress(stage=name, done=j + 1, total=len(items), elapsed_s=el); print(f"  {name} {j + 1}/{len(items)} {el:.0f}s", flush=True)
        write(name + ("-smoke" if a.smoke else ""), rows); print(f"UNIT {name} done ({len(rows)} rows, {time.time() - t0:.0f}s)", flush=True)
        return rows

    # 1. fingerprint against Amendment 4
    ref = list(csv.DictReader(open(os.path.join(D, "fingerprint_ref.csv"))))
    fp = unit("fingerprint", ref, lambda it: lp_row(it["prompt"]))
    y = lambda p: max(-6, min(6, math.log(p / (1 - p))))
    rms = math.sqrt(np.mean([(y(float(r["pA"])) - y(float(r["pA_ref"]))) ** 2 for r in fp]))
    print(f"FINGERPRINT rms |Δ logit| vs Amendment 4 = {rms:.4f} nats (gate {a.fp_gate})", flush=True)
    json.dump(dict(rms=rms, gate=a.fp_gate, passes=rms <= a.fp_gate), open(os.path.join(D, "fingerprint_result.json"), "w"))
    if rms > a.fp_gate:
        print("FINGERPRINT FAILED: environment differs from Amendment 4; stopping", flush=True); sys.exit(6)
    # 2-3. K_avg two replicates (different order)
    P = list(csv.DictReader(open(os.path.join(D, "prompts_kavg.csv"))))
    unit("kavg_rep0", P, lambda it: lp_row(it["prompt"]))
    unit("kavg_rep1", list(reversed(P)), lambda it: lp_row(it["prompt"]))
    # 4. comprehension
    comp = [dict(question=c["question"], counts="".join(map(str, c["counts"])), s0=c["s0"], order=c["order"], correct=c["correct"],
                 prompt=c["prompt"]) for c in battery.comprehension_items()]
    unit("comprehension", comp, lambda it: lp_row(it["prompt"]))
    # 5. garbage / invalid-token rate by temperature
    G = sorted(P, key=lambda r: hashlib.sha256(r["prompt"].encode()).hexdigest())[:100]
    ns = 2 if a.smoke else 20

    def garbage(it):
        out = {}
        lg, enc = logits1(it["prompt"])
        for T in (1.0, 1.5, 2.0):
            p = torch.softmax(lg / T, -1); out[f"leak_exact_T{T}"] = float(1 - p[AB].sum())
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=T, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                   max_new_tokens=8, num_return_sequences=ns, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            txt = [tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True) for s in g]
            ans = [parse_sample(t) for t in txt]
            out[f"invalid_T{T}"] = sum(x is None for x in ans) / len(ans); out[f"nA_T{T}"] = sum(x == "A" for x in ans)
            out[f"examples_T{T}"] = " || ".join(repr(t) for t, x in zip(txt, ans) if x is None)[:300]
        return out
    unit("garbage", [dict(prompt_id=r["prompt_id"], prompt=r["prompt"]) for r in G], garbage)
    progress(stage="done", elapsed_s=time.time() - t0); print(f"ALL UNITS DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
