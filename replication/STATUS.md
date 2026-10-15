# Replication package: working status (handoff note, updated 2026-10-15 ~18:40 UTC)

For a session that resumes this work without the earlier conversation. Branch `claude/peaceful-fermi-hz11cv`; all
work is committed and pushed. Background: `llm/EXECUTION_LOG.md` entry 73; package docs in `replication/README.md`.

## Owner's request (2026-10-15)

1. `make replicate` / `make replicate-fast`: fetch HF dataset → verify SHA-256 → rerun every analysis from cache (no
   keys, no GPU) → regenerate figures → compare with committed results (stated tolerances) → pass/fail per claim.
2. Pinned environment (lockfile + Dockerfile), graph-fingerprint assertions kept, runtime/hardware for both modes.
3. README with claims map, tier 1 (exact, from cache) vs tier 2 (re-collection, ~$37, provider drift).
4. Test on a fresh container, fix what breaks, record the clean-run log.
5. Licences: code Apache-2.0, data CC BY 4.0, model-provider output terms (Gemma checked).
6. Prepare GitHub release + Zenodo + HF dataset card. **Publish nothing; the owner flips the switches.**

## Done

- **Items 1–3, 5 and 6:** done. Files: `Makefile`, `Dockerfile`, `.dockerignore`, `replication/*`, `LICENSE`,
  `LICENSE-DATA`, `LICENSES.md`, `CITATION.cff`, `.zenodo.json`, `replication/hf_dataset_card/README.md`,
  `replication/RELEASE_CHECKLIST.md`.
- **Item 4, fast mode:** passes in a fresh container (image built from `git archive` of 38b926f). Every claim passes,
  53/53 headline numbers, 54.6 min, peak 4.6 GB. Record: `replication/logs/clean_run_fast_2026-10-15/` (including 4
  earlier failed attempts and their fixes).
- **HF dataset (still private):** `sim_units/amplification_units.tar` and `confirm_units.tar` uploaded and verified
  (`replication/sim_units_manifest.json`).
- **Secrets audits:** done; results in `replication/RELEASE_CHECKLIST.md`.
- **Paper and REPORT:** §4.1 now states that the asymmetry values are the N = 4,000 rows. The paper's data-availability
  section describes the package.

## The two tiers (as stated in `replication/README.md`)

- **Tier 1: bit-identical recomputation from the cached responses** (`make replicate[-fast]`). No keys, no GPU.
  - Fast mode passes in a fresh container: all claims, 53/53 headline numbers.
  - The comparator canary proves that a one-byte or one-ulp change is rejected.
- **Tier 2: live re-query.** New API calls and GPU runs, about $37. It is **not expected to be identical**. Measured
  reasons:
  - hosted endpoints have repeat noise of 0.034–0.296 nats (gate 0.3), and gate-failed endpoints reach 0.70;
  - provider jitter: format fields shift, coefficients differ 7–9% in one framing, and answers drift over hours;
  - mixture-of-experts batch nondeterminism: RMS 0.86 nats, with 2.5% of majority answers flipped.
  - The deterministic reference is local bf16 Gemma-4 at batch size 1 with deterministic kernels: replicates
    identical, repeat SD 0.114.

## Amendment 4 null distribution

- **Checked:** 20 of the 100 replicates were recomputed in the interrupted clean full run, and all 20 are identical
  to the committed ones.
- **Not yet recomputed:** replicates 20–99.

## Full-mode clean run: interrupted (host restart), needs an owner decision

- **What happened:** the run started 04:54 UTC on 2026-10-15. The cloud host restarted between about 13:40 and 17:35
  UTC, after the session went idle; uptime was 0 at the 17:35 check-in. Docker and the run were lost.
- **What survived:** 8 of 25 steps finished, all PASS. That includes `base.sim`, bit-identical, recomputed with the
  fixed comparator. `s1.null` had finished 20 of 100 replicates, all equal to the committed ones. Record:
  `replication/logs/clean_run_full_2026-10-15_partial/`.
- **Owner decision (2026-10-15): do not relaunch here.** A RunPod CPU-pod recipe for the unmeasured steps (without
  the null distribution) is prepared in `replication/runpod/` but not launched:
  - dedicated `spt-replication-…` network volume;
  - HF heartbeat and per-step verified uploads;
  - end-to-end test mode first;
  - failure-mode table.

  The pod script was tested locally (dry, 3 steps). The owner launches it.
- **Earlier options**, kept for reference: the owner decides. Options:
  1. Relaunch the whole full mode: about 37 h, dominated by `s1.null` at about 21 h.
  2. Relaunch with `--skip s1.null`: about 16 h.
  3. Run only the unmeasured steps with `--only …`: about 13 h.

  Any long run needs the session kept active (the host is reclaimed when idle) or a machine the owner controls.
- **To relaunch:** see `replication/logs/clean_run_full_2026-10-15_partial/run_full.sh`. First start dockerd with
  `llm/bg.sh start dockerd dockerd`. Rebuild the image, which may need a retry on Docker Hub 429s.

## Open decisions for the owner (reported; not acted on)

1. **Generation ids in git history.** Cache files committed before 2026-10-09 still contain OpenRouter generation
   ids. The options (accept; rewrite history; fresh public repo) are in `RELEASE_CHECKLIST.md`. Only "accept" keeps
   the pre-registration commit ids verifiable.
2. **`papers/*.pdf`** (OpenAI's documents): confirm redistribution rights or remove them; they are also in the history.
3. **Author metadata:** "Randall Bennington" (inferred from the email address) in `CITATION.cff`, `.zenodo.json` and
   the card is to be confirmed, and an ORCID added.
4. **Model-output terms:** verify the table in `LICENSES.md`. The Gemma 4 licence page was unreachable; the HF card
   says Apache-2.0.
5. **Publishing:** nothing is published. The HF dataset is private, there is no release or tag, and Zenodo is not
   enabled. Steps are in `RELEASE_CHECKLIST.md`.

## Standing rules (from CLAUDE.md and the owner)

- Background jobs: start with `llm/bg.sh start NAME …` and stop with `llm/bg.sh stop NAME`. Never use `pkill -f`.
- No force-push. No PR unless asked. Never print secrets. No model identifiers in commits.
- Commit trailers: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01HkJZ66L2APs3D5WfJVsYtg`.
- Disk is a fixed allowance; it filled once. Free scratch space before large runs.
