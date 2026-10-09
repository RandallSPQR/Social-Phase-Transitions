"""Self-hosted bf16 battery runner (PREREG Amendment 4). Runs on a GPU pod.

  python llm/local_gemma.py --model-id google/gemma-4-31b-it --key gemma-31b-local [--batch-size 16] [--max-prompts 40]

For every prompt of the Stage 1 battery (5 arms + order-sensitivity subset) and the comprehension battery:
official chat template (user turn, add_generation_prompt=True), ONE batched forward pass, next-token
distribution at the last position. Two replicates with different batch order/size (measures batch-
composition numerics for the 0.3-nat gate). Replicate 0 also saves last-token residual activations at 6 layers.

Outputs (same schema as run_stage1.py, so analysis.py / summarize.py run unchanged):
  results/llm/stage1/main/<key>.csv, results/llm/stage1/comprehension/<key>.csv
  results/llm/activations/<key>/layer_<L>.npy (fp16, [n_prompts, d_model]) + index.csv
"""
import argparse, math, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def all_items(max_prompts=None):
    its = []
    for arm in battery.ARMS:
        its += list(battery.items(arm))
    its += list(battery.items("neutral_own", extra_perms=True))
    comp = list(battery.comprehension_items())
    if max_prompts:
        rng = random.Random(0)
        its, comp = rng.sample(its, max_prompts), rng.sample(comp, max(4, max_prompts // 4))
    return its, comp


def letter_ids(tok):
    """All vocab ids whose stripped decode is exactly 'A' / 'B' (for the exact full-vocabulary P(A))."""
    A, B = [], []
    for i in range(len(tok)):
        s = tok.decode([i]).strip()
        if s == "A": A.append(i)
        elif s == "B": B.append(i)
    return A, B


def parse_top5(top):
    """Stage 1 rule (run_stage1.parse_lp) on [(token_text, logprob)] top-5."""
    pa = sum(math.exp(l) for t, l in top if t.strip() == "A")
    pb = sum(math.exp(l) for t, l in top if t.strip() == "B")
    leak = max(0.0, 1.0 - pa - pb); cens = 0; floor = math.exp(min(l for _, l in top))
    if pa == 0: pa, cens = floor, 1
    if pb == 0: pb, cens = floor, 1
    return pa / (pa + pb), leak, cens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--layer-fracs", default="0.1667,0.3333,0.5,0.6667,0.8333,1.0")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model_id); tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.bfloat16, device_map=a.device)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText      # multimodal checkpoints (text-only use)
        model = AutoModelForImageTextToText.from_pretrained(a.model_id, torch_dtype=torch.bfloat16, device_map=a.device)
    model.eval()
    A_ids, B_ids = letter_ids(tok)
    # padding check (abort if left-padding changes next-token logprobs): 4 prompts batched with a long
    # neighbour vs run alone; tolerance 0.05 nats in bf16
    probe = [it["prompt"] for it in list(battery.items("political"))[:4]]
    pt = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in probe]
    long_t = tok.apply_chat_template([{"role": "user", "content": probe[0] + " " + "filler " * 60}], tokenize=False, add_generation_prompt=True)
    with torch.no_grad():
        eb = tok([long_t] + pt, return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
        lb = torch.log_softmax(model(**eb).logits[:, -1, :].float(), -1)
        worst = 0.0
        for i, t in enumerate(pt):
            e1 = tok([t], return_tensors="pt", add_special_tokens=False).to(a.device)
            l1 = torch.log_softmax(model(**e1).logits[:, -1, :].float(), -1)
            top = torch.topk(l1[0], 20).indices
            worst = max(worst, float((lb[i + 1, top] - l1[0, top]).abs().max()))
    print(f"padding check: max |batched − single| over top-20 tokens = {worst:.4f} nats", flush=True)
    if worst > 0.05:
        sys.exit(f"ABORT: left padding changes logprobs by {worst:.3f} nats (> 0.05)")
    its, comp = all_items(a.max_prompts)
    jobs = [("main", it) for it in its] + [("comprehension", it) for it in comp]
    texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True)
             for _, it in jobs]
    n_layers = model.config.get_text_config().num_hidden_layers if hasattr(model.config, "get_text_config") else model.config.num_hidden_layers
    layers = sorted({max(1, round(f * n_layers)) for f in map(float, a.layer_fracs.split(","))})
    acts = {L: np.zeros((len(jobs), 0), np.float16) for L in layers}
    rows = [[None, None] for _ in jobs]
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    print(f"{a.key}: {len(jobs)} prompts, mean {np.mean(lens):.0f} tokens, layers {layers}/{n_layers}, |A ids|={len(A_ids)} |B ids|={len(B_ids)}", flush=True)
    t0 = time.time()
    for rep in (0, 1):
        order = sorted(range(len(jobs)), key=lambda i: lens[i]) if rep == 0 else random.Random(1).sample(range(len(jobs)), len(jobs))
        bs = a.batch_size if rep == 0 else max(1, a.batch_size // 2 + 3)          # different batch composition
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=(rep == 0))
            lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            top = torch.topk(lp, 5, dim=-1)
            pA_exact = lp[:, A_ids].exp().sum(-1); pB_exact = lp[:, B_ids].exp().sum(-1)
            for j, i in enumerate(idx):
                t5 = [(tok.decode([int(t)]), float(v)) for t, v in zip(top.indices[j], top.values[j])]
                pa, leak, cens = parse_top5(t5)
                pe, pbe = float(pA_exact[j]), float(pB_exact[j])
                rows[i][rep] = dict(pA=pa, leak=leak, censored=cens, pA_exact=pe / (pe + pbe) if pe + pbe > 0 else None,
                                    leak_exact=1 - pe - pbe, content=t5[0][0], top5=" | ".join(f"{t!r}:{v:.3f}" for t, v in t5))
            if rep == 0:
                for L in layers:
                    h = out.hidden_states[L][:, -1, :].float().cpu().numpy().astype(np.float16)
                    if acts[L].shape[1] == 0:
                        acts[L] = np.zeros((len(jobs), h.shape[1]), np.float16)
                    acts[L][idx] = h
            if (s // bs) % 50 == 0:
                print(f"  rep {rep} {s + len(idx)}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    import csv
    for phase in ("main", "comprehension"):
        recs = []
        for (ph, it), rr in zip(jobs, rows):
            if ph != phase: continue
            base = {k: v for k, v in it.items() if k != "prompt"}
            base.update(counts="".join(map(str, it["counts"])), key=a.key, model=a.model_id, tag="local/bf16", temperature=1.0,
                        served="local-hf-transformers", cost=0.0)
            for rep in (0, 1):
                recs.append({**base, "rep": rep, **rr[rep]})
        d = os.path.join(ROOT, "results", "llm", "stage1", phase); os.makedirs(d, exist_ok=True)
        keys = list(dict.fromkeys(k for r in recs for k in r))
        with open(os.path.join(d, f"{a.key}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(recs)
    ad = os.path.join(ROOT, "results", "llm", "activations", a.key); os.makedirs(ad, exist_ok=True)
    for L in layers:
        np.save(os.path.join(ad, f"layer_{L:03d}.npy"), acts[L])
    with open(os.path.join(ad, "index.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "phase", "arm", "question", "counts", "s0", "mapping", "order", "perm", "seq"])
        for i, (ph, it) in enumerate(jobs):
            w.writerow([i, ph, it["arm"], it.get("question", ""), "".join(map(str, it["counts"])), it["s0"], it["mapping"], it["order"], it["perm"], it["seq"]])
    print(f"done {a.key}: {len(jobs)} prompts x 2 reps in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
