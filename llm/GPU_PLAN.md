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

**Estimate** (one pod, both models in sequence). Live RunPod prices, read 2026-10-09 (EXECUTION_LOG entry 21).
The owner's network volume `u0isne6ams` (150 GB) is in **EUR-IS-1**, and a pod can only mount it in that data
centre. The only ≥ 80 GB GPU with stock there is the RTX PRO 6000 Blackwell Server (96 GB, stock "Low").

| GPU (1×) | where | live price | wall time | estimate |
|---|---|---|---|---|
| **RTX PRO 6000 Blackwell Server 96GB** (with volume) | EUR-IS-1 | $2.49/h | 1.5–2.5 h | **≈ $4–6** |
| A100 80GB (no volume; weights re-downloaded) | elsewhere | $1.79/h secure | 1.5–2.5 h | ≈ $3–5 |
| H100 PCIe 80GB (no volume) | elsewhere | $2.89/h secure | 1.0–1.5 h | ≈ $3–4.5 |

**Ceiling: $15.** Safeguards:
- `pod_run.sh` has a 4 h hard limit (tested: the exit trap still ships results and removes the pod) and per-step
  timeouts. Worst case at 4 h on the RTX PRO 6000: $9.96.
- It self-terminates via `runpodctl remove pod $RUNPOD_POD_ID` (its own pod only) on success *or* failure.
- Outside check-in at +4 h 15 min terminates the pod if still present; spend check against the balance
  baseline net of the owner's other running pod (≈ $1.82/h), stop at $12.
- Results go to the HF dataset before termination.
- **Shared volume:** writes only under `/workspace/social-phase-transitions/` (marker file `.owner`), aborts if
  that directory exists without the marker, never deletes on the volume, and falls back to the container disk if
  the volume has < 130 GB free (weights ≈ 115 GB). The repo, logs and results tarball stay on the container disk.

**Recommendation:** 1× RTX PRO 6000 Blackwell Server 96GB in EUR-IS-1 with the owner's volume (owner's preferred setup); fall back to an A100 80GB elsewhere without the volume if it is out of stock.

**Prerequisites (status 2026-10-09, EXECUTION_LOG entries 16 and 19):**
1. huggingface.co access and an HF token: **done** (token works; both Gemma-4 repos readable, not gated).
2. RunPod access: **done** (API key; REST and GraphQL both work; plugin tools not loaded, not needed).
3. Owner's explicit OK to rent.

**Untested on real Gemma-4:**
- the model class (the runner tries `AutoModelForCausalLM`, then `AutoModelForImageTextToText`);
- the chat template (taken from the tokenizer);
- left-padding position handling (the pre-run check aborts if batched and single logprobs differ by
  > 0.05 nats).

The runner was validated end to end on a tiny random-weight model on CPU: padding error < 1e-6, and the
Stage 1 schema loads in `analysis.py`.
