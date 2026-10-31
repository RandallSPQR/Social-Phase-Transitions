# Replication package: status at session close (2026-10-31): released

This note is for a session that resumes the work without the earlier conversation.

- **Branch:** `claude/peaceful-fermi-hz11cv`.
- **State:** released (v1.0.1). Everything is committed and pushed. Nothing is running and no scheduled check-ins
  remain.
- **Public:** code https://github.com/RandallSPQR/social-phase-transitions-public (one snapshot commit of private
  `3a26daf` plus later doc commits; release v1.0.1); data https://huggingface.co/datasets/RandallSPQR/social-phase-transitions-data;
  archive DOI 10.5281/zenodo.23821224 (all versions; v1.0.1 = 10.5281/zenodo.23821962; v1.0.0 deleted by the owner).
- **Private, keep unchanged:** this repository and the dataset `RandallSPQR/social-phase-transitions-llm-cache`. Their
  history holds third-party PDFs and generation ids, and their server logs back the pre-registration timestamps.
- **Background:** `llm/EXECUTION_LOG.md` entry 73. Package documentation: `replication/README.md`.

## Owner's request (2026-10-15) and where each item stands

| # | item | status |
|---|---|---|
| 1 | `make replicate` / `make replicate-fast`: fetch → verify SHA-256 → rerun from cache (no keys, no GPU) → regenerate figures → compare with stated tolerances → pass/fail per claim | **done** |
| 2 | Pinned environment (lock file, Dockerfile), graph fingerprints, runtime and hardware for both modes | **done**. Fast mode measured; full mode partly measured, rest estimated |
| 3 | README: claims map, tier 1 vs tier 2 | **done** |
| 4 | Fresh-container test, fixes, clean-run log | **done for fast mode**. Full mode interrupted after 8 of 25 steps (all pass); the rest stays estimated (no RunPod jobs, owner 2026-10-30) |
| 5 | Licences: code Apache-2.0, data CC BY 4.0, model-output terms | **done**; terms confirmed by the owner 2026-10-30 |
| 6 | GitHub release, Zenodo, HF dataset card | **published 2026-10-30** (public copies; see above) |

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

## Owner decisions and release (2026-10-30)

- **OpenAI PDFs removed** from the tree (`papers/`, in `.gitignore`); cited by title in `paper/refs.bib`.
- **Author:** Randall S. Bennington. **Model-output terms** confirmed. **No more RunPod jobs** (recipe kept for
  reference; remaining full-mode times stay estimates).
- **Pre-registration provenance** (`replication/provenance/`): all 22 pre-registration commits were on GitHub
  (server-side push log) before the data they govern. Margins under a minute for 4 rows rest on the client clock
  (`PREREG_PROVENANCE.md`). `verify_pack.py` checks them against `pack/` in the public repository: the original
  history without `papers/`, `results/llm/cache/` and the GitHub-made root commit (it carried the owner's personal
  email). Verified from a fresh clone of the public repository.
- **Public HF dataset:** copy of the private one without `code/` and `cleanup/` (535 files, hashes match; scanned:
  no PDF, generation id or email). Manifests point at it.
- **Public GitHub repository:** history force-pushed once (owner's OK) to a single clean commit; the old v1.0.0
  release and tag were deleted by the owner; v1.0.1 released.

## Open for the owner (optional)

- ORCID in `CITATION.cff`, `.zenodo.json` and the dataset card.
- Reviewer access to the private originals on request (to check the exported logs against GitHub's and HF's own).
- Next study: deposit each pre-registration and amendment on Zenodo/OSF before data collection.
- Changes to the public repository after v1.0.1 (DOI and checklist edits) reach Zenodo only with a new GitHub release.

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
