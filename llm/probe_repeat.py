"""Second probe: (a) repeat-consistency of first-token logprobs, 6 identical calls per
endpoint on 3 prompts; (b) whether reasoning can be switched off for hybrid models;
(c) whether sampled frequencies match logprob P on one cheap endpoint.
Output: results/llm/catalog/probe_repeat_<date>.csv
"""
import csv, datetime, json, math, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client
from probe_providers import PROMPT

P2 = ("You currently hold position B. Your two contacts currently hold these positions:\n"
      "- a rival who usually opposes you: B\n- an ally who agrees with you on most things: A\n\n"
      "Which position do you take now? Reply with exactly one letter, A or B, and nothing else.")
P3 = ("You currently hold position A. Your one contact currently holds this position:\n"
      "- an ally who agrees with you on most things: B\n\n"
      "Which position do you take now? Reply with exactly one letter, A or B, and nothing else.")
PROMPTS = [PROMPT, P2, P3]

LP = [("meta-llama/llama-3.2-3b-instruct", "parasail/bf16"),
      ("meta-llama/llama-3.1-8b-instruct", "novita/fp8"), ("meta-llama/llama-3.1-8b-instruct", "coreweave/bf16"),
      ("meta-llama/llama-3.3-70b-instruct", "novita/bf16"), ("meta-llama/llama-3.3-70b-instruct", "parasail/fp8"),
      ("meta-llama/llama-3.3-70b-instruct", "coreweave/fp16"), ("meta-llama/llama-3.3-70b-instruct", "akashml/fp8"),
      ("qwen/qwen3-30b-a3b-instruct-2507", "streamlake"), ("qwen/qwen3-30b-a3b-instruct-2507", "alibaba"),
      ("qwen/qwen3-235b-a22b-2507", "parasail/fp8"), ("qwen/qwen3-235b-a22b-2507", "alibaba"),
      ("google/gemma-3-27b-it", "parasail/fp8"),
      ("google/gemma-4-26b-a4b-it", "parasail/bf16"), ("google/gemma-4-26b-a4b-it", "coreweave/bf16"),
      ("google/gemma-4-31b-it", "parasail/fp8"),
      ("mistralai/mistral-nemo", "io-net/fp16"), ("mistralai/mistral-nemo", "dekallm/fp8"), ("mistralai/mistral-nemo", "parasail/fp8"),
      ("mistralai/mistral-small-3.2-24b-instruct", "parasail/bf16"),
      ("openai/gpt-4o-mini", "openai"),
      ("qwen/qwen3.5-9b", "parasail/bf16"), ("qwen/qwen3.5-27b", "phala"), ("mistralai/mistral-large-4-0", "mistral")]
NREP = 6


def lp_body(model, tag, prompt):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
         "temperature": 1.0, "logprobs": True, "top_logprobs": 5,
         "provider": {"order": [tag], "allow_fallbacks": False, "require_parameters": True}}
    if "qwen3.5" in model or "large-4" in model:
        b["reasoning"] = {"enabled": False}
    return b


def pA_of(first):
    if not first or not first["top"]:
        return None, None
    pa = sum(math.exp(l) for t, l in first["top"] if t.strip() == "A")
    pb = sum(math.exp(l) for t, l in first["top"] if t.strip() == "B")
    return (pa / (pa + pb) if pa + pb > 0 else None), pa + pb


def run_lp(model, tag):
    rows = []
    for pi, pr in enumerate(PROMPTS):
        b = lp_body(model, tag, pr)
        logits, mass, err, served, content = [], [], None, None, None
        for r in range(NREP):
            try:
                rec = client.chat(b, rep=r)["resp"]
                served, content = rec["provider"], rec["content"]
                p, m = pA_of(rec["first"])
                if p is not None and 0 < p < 1:
                    logits.append(math.log(p / (1 - p))); mass.append(m)
                elif p is not None:
                    logits.append(math.copysign(30, p - 0.5)); mass.append(m)
            except Exception as e:
                err = str(e)[:200]
        sd = (sum((x - sum(logits) / len(logits)) ** 2 for x in logits) / (len(logits) - 1)) ** 0.5 if len(logits) > 1 else None
        rows.append({"model": model, "tag": tag, "served": served, "prompt": pi, "n_ok": len(logits),
                     "content_last": repr(content),
                     "logit_mean": round(sum(logits) / len(logits), 3) if logits else None,
                     "logit_sd": round(sd, 3) if sd is not None else None,
                     "logit_range": round(max(logits) - min(logits), 3) if logits else None,
                     "logits": " ".join(f"{x:.2f}" for x in logits),
                     "mass_AB_min": round(min(mass), 4) if mass else None, "error": err})
    return rows


def run_sampling(model, tag, reasoning, n=1):
    b = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 64, "temperature": 1.0,
         "provider": {"order": [tag], "allow_fallbacks": False}}
    if reasoning is not None:
        b["reasoning"] = reasoning
    out = []
    for r in range(n):
        try:
            rec = client.chat(b, rep=r)["resp"]; u = rec["usage"] or {}
            out.append({"model": model, "tag": tag, "reasoning_param": json.dumps(reasoning), "content_last": repr(rec["content"]),
                        "served": rec["provider"], "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        "completion_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"), "cost": u.get("cost")})
        except Exception as e:
            out.append({"model": model, "tag": tag, "reasoning_param": json.dumps(reasoning), "error": str(e)[:200]})
    return out


if __name__ == "__main__":
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda mt: run_lp(*mt), LP))
    rows = [r for rs in res for r in rs]
    samp = []
    for m, t in [("anthropic/claude-haiku-5.5", "anthropic"), ("openai/gpt-6-luna", "openai"),
                 ("google/gemini-3.1-flash-lite", "google-ai-studio"), ("google/gemini-2.5-flash-lite", "google-ai-studio")]:
        for rp in [{"enabled": False}, {"effort": "none"}, {"max_tokens": 0}]:
            samp += run_sampling(m, t, rp)
    d = datetime.date.today()
    for name, rr in [("probe_repeat", rows), ("probe_reasoning_off", samp)]:
        keys = []
        for r in rr:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(client.ROOT, "results", "llm", "catalog", f"{name}_{d}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rr)
    for r in rows:
        print(f"{r['model'][:36]:36s} {r['tag']:15s} p{r['prompt']} n={r['n_ok']} mean={r['logit_mean']} sd={r['logit_sd']} range={r['logit_range']} mass={r['mass_AB_min']} {r['content_last']} {r['error'] or ''}")
    for r in samp:
        print(json.dumps(r))
    print("spend", client.spend())
