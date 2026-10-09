# Stage 1 model lineup: proposal (pending owner OK)

Evidence: `results/llm/catalog/` — `models_2026-10-08.json` (catalogue), `endpoints_2026-10-08.json`
(per-provider metadata), `probe_*.csv` (live checks). Probe spend: $0.0063, 513 calls.

## What the probes showed (2026-10-08)

1. **Metadata is unreliable.** Endpoints that claim `logprobs` but return none: Gemma-4-26B @ Novita,
   Gemma-4-31B @ Venice, Qwen3.5-27B @ Phala. Alibaba caps `top_logprobs` at 5 → we request 5 everywhere.
2. **Logit resolution is 0.125 nats** (logit(P_A) lands on multiples of 1/8: bf16 raw logits).
3. **Repeat noise is endpoint-specific.** SD of logit(P_A) over 6 identical calls ranges 0 (Llama-8B @ Novita,
   Qwen3.5 @ Alibaba/Parasail, Mistral-Small @ Parasail) to ≈1 nat (Gemma-4-31B @ Parasail fp8; 4 nats @ CoreWeave fp4).
4. **Same weights, different providers, different answers.** Llama-3.3-70B, one prompt: logit −2.75 (AkashML fp8),
   −0.25 (Parasail fp8), −0.08 (CoreWeave fp16), +2.17 (Novita bf16). Provider is part of the "model".
5. **Temperature semantics differ.** Novita returns post-temperature logprobs (logit × 1/T); Parasail, Alibaba,
   OpenAI return pre-temperature. ⇒ logprob battery runs at T = 1 only; temperature effects measured by sampling.
6. **Reasoning.** Qwen3.5, Haiku 5.5, GPT-6-luna, Gemini 2.5 Flash-Lite reason by default; `reasoning:{enabled:false}`
   turns it off (0 reasoning tokens). Gemini 3.5 Flash-Lite refuses ("reasoning is mandatory") → excluded.
7. **Off-target mass.** A+B mass ≥ 0.98 for most; Mistral Large 4 puts up to 8% on "I" (starts explaining).

## Proposed lineup (all pinned: `provider.order=[tag]`, `allow_fallbacks=false`, `require_parameters=true`)

| Family | Small | Large | Note |
|---|---|---|---|
| Llama | 3.1-8B @ coreweave/bf16 | 3.3-70B @ coreweave/fp16 | same provider, full precision |
| Qwen 3.5 | 9B dense @ parasail/bf16 | 122B-A10B MoE @ alibaba | reasoning off; both deterministic |
| Gemma 4 | 26B-A4B MoE @ parasail/bf16 | 31B dense @ io-net | no ~8B / ~70B Gemma with logprobs; ladder is 4B→31B active |
| Mistral | Nemo 12B @ io-net/fp16 | Large 4 @ mistral | Large 4: open-weight status unverified, 8% off-target; Small 3.2 24B @ parasail/bf16 as mid rung |
| replication | — | Llama-3.3-70B @ parasail/fp8 | provider-robustness check, neutral framing |
| bridge | GPT-4o-mini @ openai | | has logprobs AND samples → validates sampling-based estimates |

Spot checks (sampling, reasoning off, T = 1, pinned first-party): Claude Haiku 5.5 @ anthropic,
GPT-6-luna @ openai, Gemini 3.1 Flash-Lite @ google-ai-studio.
