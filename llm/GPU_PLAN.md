# Amendment 4 GPU plan and cost estimate (awaiting owner OK; nothing rented)

**Workload per model**
- Prompts: 6,214 main (5 arms + order subset) + 2,116 comprehension = 8,330, × 2 replicates = 16,660
  forward passes.
- Size: ≈ 155 tokens each with the Gemma chat template, ≈ 2.6 M tokens per model.
- No generation; one forward pass per prompt.

| | Gemma-4 31B (dense) | Gemma-4 26B-A4B (MoE, ~4B active) |
|---|---|---|
| bf16 weights | ≈ 62 GB | ≈ 52 GB |
| forward FLOPs (2·N_active·tokens) | ≈ 1.6e17 | ≈ 2e16 (HF MoE kernels are inefficient; assume 2–4× slower than the FLOPs imply) |
| A100 80GB, ~35% MFU | ≈ 25 min | ≈ 10–20 min |
| H100 80GB | ≈ 8–10 min | ≈ 5–10 min |
| activations (6 layers, fp16) | ≈ 0.54 GB | ≈ 0.3 GB |

**Overheads** (per model unless noted):
- pip install: ~3 min, once.
- Weights: if not cached on the network volume, download ≈ 115 GB total (~5–10 min); load 3–5 min.
- Vocabulary scan for A/B token ids: ~1 min.
- Padding check plus 40-prompt smoke test: ~5 min.

**Estimate** (one pod, both models in sequence):

| GPU (1×, 80 GB) | wall time | list price (RunPod, **unverified, check in console**) | estimate |
|---|---|---|---|
| A100 80GB | 1.5–2.5 h | ~$1.5–1.9 / h | **≈ $3–5** |
| H100 80GB | 1.0–1.5 h | ~$2.7–3.3 / h | **≈ $3–5** |

**Ceiling: $15.** Safeguards:
- `pod_run.sh` has a 4 h hard limit and per-step timeouts.
- It self-terminates via `runpodctl remove pod` on success *or* failure.
- Results go to the HF dataset before termination.
- Network volume: ≥ 150 GB if the weights are not already cached there.

**Recommendation:** 1× A100 80GB (enough memory for 31B bf16 plus batch 16 × ~300 tokens), with H100
as the fallback.

**Prerequisites still missing in this session:**
1. RunPod access: no RunPod connector here, and api.runpod.io is blocked by the network policy.
2. huggingface.co access, also blocked: needed to ship results back and to confirm the Gemma-4 model ids.
3. An HF token with access to the gated Gemma-4 weights, available on the pod.

**Untested on real Gemma-4:**
- the model class (the runner tries `AutoModelForCausalLM`, then `AutoModelForImageTextToText`);
- the chat template (taken from the tokenizer);
- left-padding position handling (the pre-run check aborts if batched and single logprobs differ by
  > 0.05 nats).

The runner was validated end to end on a tiny random-weight model on CPU: padding error < 1e-6, and the
Stage 1 schema loads in `analysis.py`.
