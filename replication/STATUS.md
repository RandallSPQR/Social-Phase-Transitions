# Replication package: status at session close (2026-10-19)

This note is for a session that resumes the work without the earlier conversation.

- **Branch:** `claude/peaceful-fermi-hz11cv`.
- **State:** everything is committed and pushed. Nothing is running, no scheduled check-ins remain, and nothing has
  been published or launched.
- **Background:** `llm/EXECUTION_LOG.md` entry 73. Package documentation: `replication/README.md`.

## Owner's request (2026-10-15) and where each item stands

| # | item | status |
|---|---|---|
| 1 | `make replicate` / `make replicate-fast`: fetch → verify SHA-256 → rerun from cache (no keys, no GPU) → regenerate figures → compare with stated tolerances → pass/fail per claim | **done** |
| 2 | Pinned environment (lock file, Dockerfile), graph fingerprints, runtime and hardware for both modes | **done**. Fast mode measured; full mode partly measured, rest estimated |
| 3 | README: claims map, tier 1 vs tier 2 | **done** |
| 4 | Fresh-container test, fixes, clean-run log | **done for fast mode**. Full mode interrupted after 8 of 25 steps (all pass); the rest stays estimated (no RunPod jobs, owner 2026-10-30) |
| 5 | Licences: code Apache-2.0, data CC BY 4.0, model-output terms | **done**; the terms table is still to be verified by the owner |
| 6 | GitHub release, Zenodo, HF dataset card | **prepared, not published** |

Owner follow-ups, all done:
- comparator canary;
- two-tier statement with the measured drift;
- null-distribution note;
- RunPod recipe with a separate `--job null`;
- release checklist section 0b;
- confirmation that the RunPod secret `HF_TOKEN` already exists.

## Verified results

- **Fast mode, clean container** (image from `git archive` of 38b926f):
  - every claim passes, 53/53 headline numbers;
  - 54.6 min on 4 vCPU, peak 4.6 GB;
  - record: `replication/logs/clean_run_fast_2026-10-15/`, including 4 earlier failed attempts and the fix for each.
- **Full mode, clean container, interrupted by a host restart:**
  - 8 of 25 steps finished, all pass, and all recomputed outputs are bit-identical (baseline sweeps 153.7 min;
    Amendment 4–5 bootstrap caches 79.5 min);
  - the null distribution reached 20 of 100 replicates, all identical to the committed ones;
  - record: `replication/logs/clean_run_full_2026-10-15_partial/`.
- **Comparator** (`replication/canary.py`, step `qc.canary`, first in every run):
  - rejects a one-byte change in a base.sim array, a raw byte flip, and a 1-ulp change in a CSV and in a JSON result;
  - `exact` means bit-identical; CSVs are parsed round-trip;
  - both clean runs pass under this strict comparator.
- **Two tiers** (`replication/README.md`):
  - Tier 1 is bit-identical recomputation from the cached responses.
  - Tier 2 is live re-query, not expected to be identical. Measured reasons: hosted repeat SD 0.034–0.296 nats (up to
    0.70 on gate-failed endpoints); provider jitter (coefficients 7–9%, drift over hours); mixture-of-experts batch
    nondeterminism (0.86 nats RMS, 2.5% of majority answers flip).
  - The deterministic reference is local bf16 Gemma-4 at batch size 1 (identical replicates; repeat SD 0.114).

## Owner decisions of 2026-10-30 (done in this commit)

- **OpenAI PDFs removed** from the tree (`papers/`, now in `.gitignore`); cited by title in `paper/refs.bib`. They
  remain in git history and in 10 HF code bundles, so the original repository and dataset stay private.
- **Author:** Randall S. Bennington (`CITATION.cff`, `.zenodo.json`, dataset card, `paper/paper.md`).
- **Model-output terms** confirmed (`LICENSES.md`, note 2).
- **No more RunPod jobs.** The recipe in `replication/runpod/` is kept for reference only; the remaining full-mode
  times stay estimates.
- **Versioning with the pre-registration timestamps:** `replication/provenance/`. `PREREG_PROVENANCE.md` lists all 22
  pre-registration commits with GitHub's server-side first-push time and the earliest data each governs (all 22:
  plan pushed before data). `verify_pack.py` checks them against a pack of the original history without the PDFs
  and cache blobs (verified 2026-10-30 in a fresh repository: all 22 OK, no PDF, no generation id). The pack is not
  committed (80 MB); it goes into the Zenodo deposit.

## Open for the owner (publishing; nothing published)

Steps in `RELEASE_CHECKLIST.md` §§0–3: rebuild the pack at the release commit; a new public HF dataset without
`code/`; a new public GitHub repository holding a snapshot; a manual Zenodo upload with the snapshot, the pack and the
provenance files. Optional: ORCID; reviewer access to the private originals.

## Environment notes for a resuming session

- **Machine:** the cloud host is reclaimed or restarted when the session is idle. Any long run here dies with it
  (this happened on 2026-10-15). Use a machine the owner controls for multi-hour jobs.
- **Docker:** start it with `llm/bg.sh start dockerd dockerd`. Pulls from Docker Hub can be rate-limited (429); retry
  with backoff. Debian mirrors are blocked by the network policy, which is why ffmpeg comes from PyPI.
- **Disk:** the allowance filled once. Free scratch space before large runs.
- **Scratch outputs:** the clean-run directories under the session scratchpad are not part of the record; the
  committed logs are.

## Standing rules (from CLAUDE.md and the owner)

- Background jobs: start with `llm/bg.sh start NAME …` and stop with `llm/bg.sh stop NAME`. Never use `pkill -f`.
- No force-push. No PR unless asked. Never print secrets. No model identifiers in commits.
- Commit trailers: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01HkJZ66L2APs3D5WfJVsYtg`.
- Never mount the owner's shared network volume or touch their other pods. Deleting anything on a shared resource
  needs the owner's OK.
