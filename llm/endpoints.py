"""Pinned endpoints for Stage 1 (see LINEUP.md, PREREG.md §2, §4).
mode: 'lp' = first-token logprobs at T=1; 'sample' = sampled replies.
backup: next endpoint to use if this one fails the 0.3-nat repeat-noise gate."""

E = {
    # key: (model, provider tag, mode, reasoning_off, backup tag)
    "llama-8b":      ("meta-llama/llama-3.1-8b-instruct", "coreweave/bf16", "lp", False, "novita/fp8"),
    "llama-70b":     ("meta-llama/llama-3.3-70b-instruct", "coreweave/fp16", "lp", False, "parasail/fp8"),
    "llama-70b-pa":  ("meta-llama/llama-3.3-70b-instruct", "parasail/fp8", "lp", False, None),   # provider robustness
    "qwen-9b":       ("qwen/qwen3.5-9b", "parasail/bf16", "lp", True, "venice/fp8"),
    "qwen-122b":     ("qwen/qwen3.5-122b-a10b", "alibaba", "lp", True, "novita/bf16"),
    "gemma-26b":     ("google/gemma-4-26b-a4b-it", "parasail/bf16", "lp", False, "coreweave/bf16"),
    "gemma-31b":     ("google/gemma-4-31b-it", "io-net", "lp", False, "novita/bf16"),
    "nemo-12b":      ("mistralai/mistral-nemo", "io-net/fp16", "lp", False, "parasail/fp8"),
    "mistral-small": ("mistralai/mistral-small-3.2-24b-instruct", "parasail/bf16", "lp", False, None),
    "mistral-large": ("mistralai/mistral-large-4-0", "mistral", "lp", True, None),
    "gpt4o-mini":    ("openai/gpt-4o-mini", "openai", "lp", False, None),                       # bridge
    "haiku":         ("anthropic/claude-haiku-5.5", "anthropic", "sample", True, None),
    "gpt6-luna":     ("openai/gpt-6-luna", "openai", "sample", True, None),
    "gemini-lite":   ("google/gemini-3.1-flash-lite", "google-ai-studio", "sample", True, None),
}

H4_PAIRS = [("llama-8b", "llama-70b"), ("qwen-9b", "qwen-122b"), ("gemma-26b", "gemma-31b"), ("nemo-12b", "mistral-large")]
