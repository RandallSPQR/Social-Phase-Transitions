# Licensing

| what | licence | file |
|---|---|---|
| Code: everything under `llm/`, `sim/`, `replication/`, plus `Makefile` and `Dockerfile` | Apache License 2.0 | `LICENSE` |
| Data and results: `results/`, `data/`, `figures/`, the pre-registrations and logs in `llm/*.md`, `paper/`, and the Hugging Face dataset `RandallSPQR/social-phase-transitions-llm-cache` | Creative Commons Attribution 4.0 International (CC BY 4.0) | `LICENSE-DATA` |
| `papers/*.pdf`: five unrefereed proofs by OpenAI (September 2026), kept as the motivating sources | **not ours; not covered by either licence above** | see note 1 |

## Notes (to check before release)

1. **`papers/`.** These PDFs are third-party works. Their redistribution terms are not stated in the files. Before the
   repository or a release archive is made public, either confirm that redistribution is permitted or remove the PDFs
   and cite them instead (the paper already cites them by title and date, `paper/refs.bib`). The replication does not
   read them.
2. **Model outputs in the response cache and the results.** The cache holds request bodies and compact replies (first
   answer token, its top-5 log-probabilities, the text of short replies, the serving provider). These are outputs of
   the models below, obtained through OpenRouter. Publishing them under CC BY 4.0 is subject to each provider's terms
   for outputs. As read on 2026-10-15 (**verify each before release**):

   | model(s) | weights licence / terms | source checked |
   |---|---|---|
   | Gemma-4 26B-A4B, Gemma-4 31B (Google) | Apache 2.0 according to the Hugging Face model cards (metadata `license: apache-2.0`, card text "License: Apache 2.0", not gated). The linked licence page (ai.google.dev/gemma/docs/gemma_4_license) could not be reached from this session: **verify** that no separate prohibited-use policy applies, as it did for Gemma 1–3 | huggingface.co/google/gemma-4-31B-it, huggingface.co/google/gemma-4-26B-A4B-it |
   | Qwen3.5 9B, Qwen3.5 122B-A10B (Alibaba) | Apache 2.0 | Hugging Face model cards |
   | Mistral Nemo 12B, Mistral Small 3.2 24B (Mistral AI) | Apache 2.0 | Hugging Face model cards |
   | Mistral Large 4 (Mistral AI, via the Mistral API) | Mistral's API terms (licence not found on Hugging Face) | **to verify** |
   | Llama-3.3 70B, Llama-3.1 8B (Meta) | Llama 3.3 / Llama 3.1 Community Licence (gated). Its use restrictions and acceptable-use policy travel with redistributed outputs used to build other models; the "Built with Llama" attribution applies to redistributed derivatives | **to verify** |
   | GPT-4o-mini, GPT-6-luna (OpenAI) | OpenAI terms of use (outputs assigned to the user, subject to the usage policies) | **to verify** |
   | Gemini 3.1 Flash-Lite (Google, AI Studio) | Gemini API additional terms | **to verify** |
   | Claude Haiku 5.5 (Anthropic) | Anthropic commercial or consumer terms, as applicable to the account used | **to verify** |

   The cache also holds probe calls (20–460 per model) to candidates screened but not selected: DeepSeek chat
   v3/v3.1/v4.x, Llama-3.2 3B and Llama-4 Maverick, GPT-4o (two dated versions) and the dated GPT-4o-mini, Gemini 2.5
   Flash-Lite, Gemma-3 27B, Qwen3 235B/30B and Qwen3.5 27B. Their providers' terms apply in the same way.

   The outputs are a few tokens per call and serve a research record. Releasing them is common practice, but the
   terms above are the owner's to confirm. A conservative option is to release the dataset under CC BY 4.0 with this
   table copied into the dataset card (done in `replication/hf_dataset_card/README.md`).
3. **Gemma activations** (`runs/e3zan9ygnmqe45/.../activations/`, about 825 MB, in the dataset) are intermediate
   activations of Apache-2.0 models and are covered by the same CC BY 4.0 licence.
4. **Fonts and tools.** Figures are rendered with matplotlib's bundled DejaVu fonts (free licence). MP4/GIF rendering
   uses the system `ffmpeg`, which the Docker image installs from Debian; it is not redistributed by this repository.
