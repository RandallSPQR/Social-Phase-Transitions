# Stage 1 execution log (events after main data collection began)

These are not pre-registration amendments. They record how pre-registered rules were applied, plus any
judgment calls, in time order (2026-10-08).

1. **Temperature verification (Amendment 3b) FAILED as pre-registered.**
   - GPT-4o-mini, 20 prompts × 50 samples at T = 0.5 and T = 1.5.
   - Slopes 1.67 (95% CI 1.37–2.07; expected 2.0) and 0.63 (0.46–0.82; expected 0.67): both CIs contain 1/T.
   - Pearson χ²: p = 1.4e-12 (T = 0.5) and p = 0.008 (T = 1.5), so the test fails.
   - **Exploratory diagnostic** (`llm/temp_diag.py`): the reference logits themselves jitter by 0.25 nats
     between identical calls. Simulating the 1/T rule plus that noise gives 95th-percentile χ² of 57.2
     (T = 0.5) and 35.7 (T = 1.5). Observed: 63.3 and 36.1. So reference noise explains most, but not all,
     of the misfit.
   - Per the amendment, no sampled temperature sweep has been run; the owner decides.
   - The per-endpoint semantics rule labelled 6 of 11 endpoints "unclear". All of them use saturated test
     prompts (|logit| 8–15) where bf16 jitter exceeds the ±0.2 tolerance. None showed post-temperature
     (1/T) scaling.
2. **Noise gate (PREREG §4) applied at endpoint level.** Repeat SD is the RMS over all prompts of all arms.
   Results:
   - gemma-26b @ Parasail: **0.703, fail**
   - gemma-31b @ Io Net: **0.304, fail**
   - gpt4o-mini: 0.497, fail (bridge; descriptive anyway)
   - llama-70b @ Parasail: 0.404, fail (replication; descriptive anyway)
   - All others pass (0.026–0.137).
   - The first interim summary applied the gate per arm. That was a bug, fixed before any confirmatory
     output was produced.
3. **Backups for the failed Gemma primaries** were run as pre-listed (`llm/endpoints.py`).
   - **gemma-26b → CoreWeave bf16.** A 100-prompt pre-check gave 0.289, but the full battery gave
     **0.354: fail.** No further backup is listed, so gemma-26b is dropped from confirmatory analysis.
   - **gemma-31b → Novita bf16.** **77% of calls returned no logprobs** (2,676 of 3,498), even with
     `require_parameters: true`. Text answers were correct, and when logprobs were present they matched
     Io Net exactly (r = 1.00). This is not a usable logprob endpoint, and the missingness may depend on
     the prompt.
     - The run was **stopped** after the comprehension battery (its 22% "accuracy" reflects missing
       logprobs, not misreading).
     - gemma-31b is dropped from confirmatory analysis.
   - Consequence: the Gemma family is excluded from H4.
   - The Gemma fits are reported descriptively. The Parasail and CoreWeave fits of Gemma-26B agree closely
     (neutral b_ally 1.56 vs 1.59, γ −0.53 vs −0.60; political γ 5.25 vs 6.0). Llama-70B's two providers
     likewise agree (b_ally 0.605 vs 0.607, γ −0.252 vs −0.256). The gate measures per-call jitter that
     largely averages out in fits. **Whether to relax it is an owner decision; it has not been changed.**
4. **Comprehension (Amendment 3a).**
   - **llama-8b fails:** ally majority 0.78, rival majority 0.72, own position 0.99. It is labelled "does
     not parse task".
   - **mistral-small fails by the leak rule:** 1,116 of 2,116 prompts have leak > 5%. On the majority
     questions it starts to reason ("To…", "Let…", "Based…"). Mean renormalised P(correct) is 0.94 (ally)
     and 0.91 (rival), so it is not misreading. It is reported as "answers only after reasoning"; the
     pre-registered verdict (fail) stands.
   - qwen-122b: one comprehension call was lost to an Alibaba rate-limit error (2,115 of 2,116 scored).
5. **Spot checks were started before the last two workhorse runs finished**, to save wall-clock time.
   - Spend at launch: $5.18. The remaining workhorse runs (Mistral Large, Novita Gemma-31B) could cost at
     most about $2.40 at list price, so spend after the workhorses was guaranteed to be ≤ $12, the owner's
     condition.
   - Per-process caps held the worst case to about $11.50.

## Phase 2 (owner instructions of 2026-10-09). All items below are post-data.

6. **Step 1a: noise-gate sensitivity analysis** (`python llm/summarize.py --relax-gate`; outputs carry the
   suffix `_sensitivity_gate_relaxed`).
   - The confirmatory H1–H5 analysis is unchanged: 4 endpoints, 12 cells.
   - **Sensitivity** (gate relaxed; primaries Gemma-26B @ Parasail, Gemma-31B @ Io Net and Mistral Large
     added; 7 endpoints, 21 cells):
     - H1 is significant in 18 of 21 cells, all with rivals weighted less than allies.
     - H2 letter field 18 of 21; content field 11 of 14.
     - H3 pairwise is supported in 4 cells: Llama-70B political and workplace, Mistral Large workplace,
       Gemma-26B workplace.
     - H3 cubic is supported in 4 cells: Gemma-26B neutral and workplace, Gemma-31B neutral and political.
       **All four are on gate-failed (jittery) endpoints.** Per-call jitter plus saturation could produce
       apparent curvature, so they are pending the local bf16 replication (Amendment 4).
     - H5 departs from summation in 12 of 21 cells.
     - **H4 would be supported** (3 of 3 testable families; Llama is still excluded by comprehension).
   - Stage 2 includes the gate-failed endpoints, with their jitter modelled explicitly as per-call logit
     noise.
   - Correction to `results/llm/STAGE1_RESULTS.md`: the content field h_C is significant in 8 of 8
     confirmatory content cells, not "6 of 6".
7. **Step 1b: temperature.** The owner accepts β(T) = β(1)/T provisionally.
   - The clean check reran the pre-registered verification on **Qwen3.5-9B @ Parasail** (deterministic
     logprobs, repeat SD 0.035): 20 prompts × 50 samples at each T, about $0.03
     (`results/llm/temperature_verify_qwen-9b.json`).
   - **It passes at both temperatures:**
     - T = 0.5: slope 2.22 (95% CI 1.87–2.61; expected 2.0), χ² p = 0.13.
     - T = 1.5: slope 0.68 (0.45–0.93; expected 0.67), χ² p = 0.57.
   - This supports reading the GPT-4o-mini failure as reference-logit jitter rather than a failure of the
     1/T rule.
   - Caveat: at T = 1.5, 99 of 1,000 samples were unparseable (stray tokens, other scripts, "C"). They are
     spread across prompts (≤ 16% per prompt; correlation with P₁ −0.2). Sampling above T = 1 needs a
     resample-on-invalid rule in Stage 3. The 1/T rule describes the valid-reply-conditional choice.
8. **Step 1d: repository housekeeping.**
   - **Scrub check:** all 268,884 cached records were scanned (every snapshot). No credentials, headers,
     cookies or e-mail addresses. Records hold only the request body (model, messages, sampling
     parameters, provider pin) and a compact response.
   - The one account-linked field is OpenRouter's generation `id` (`gen-…`). It is not a credential and is
     usable only with the owner's key. It is kept for auditability unless the owner asks to strip it.
   - **Manifest and scripts:**
     - `results/llm/cache_manifest.json`: SHA-256, size and records per snapshot, plus the dataset location.
     - `llm/fetch_cache.py`: download and verify.
     - `llm/upload_cache_hf.py`: upload, then re-download into a temp dir to verify SHA-256.
   - **Hugging Face upload is BLOCKED.** The session's network policy refuses huggingface.co (403 at the
     proxy), and no HF token is available.
   - **Planned dataset id:** `RandallSPQR/social-phase-transitions-llm-cache` (owner to confirm).
   - **Pending:** remove the `.jsonl.gz` snapshots from git (no force-push), then squash-merge into main.
     This only happens after the upload is verified, so the data is never unreachable.
9. **Step 2: Amendment 4** committed (ee18890) before any GPU work.
   - Runner `llm/local_gemma.py` and pod script `llm/pod_run.sh` (4 h hard limit, self-termination,
     results shipped to the HF dataset).
   - Validated end to end on a tiny random-weight Llama on CPU: batched vs single-prompt logprobs agree to
     < 1e-6 with 30 pad tokens; replicates agree to ≤ 0.005 nats; the output loads in the Stage 1
     analysis. A pre-run padding check aborts if this fails on Gemma-4.
   - Estimate (`llm/GPU_PLAN.md`): ≈ $3–5 on 1× A100 80GB, ceiling $15.
   - **No pod rented:** awaiting the owner's OK, plus RunPod and Hugging Face access (both blocked in this
     session).
10. **Step 1c: voter vs logit (exploratory).** `llm/voter.py`, `results/llm/voter_vs_logit.csv`.
    - Models: A0, A3, signed random-neighbour voter with lapse, mixture M = p·voter + (1−p)·A3, and the
      first-contact mixture MF = p·copy(first-listed contact) + (1−p)·A3. Compared on held-out CE with the
      Stage 1 grouped folds.
    - **First pass (before any simulation):**
      - **M is never better than A3 held-out, and its p is not identified apart from α.** Llama-70B neutral:
        p = 0.40 (0.11–0.78) on CoreWeave vs 0.79 (0.71–0.86) on Parasail, for the same weights. A
        random-neighbour voter is nearly collinear with k-averaging.
      - **MF improves held-out CE on some endpoints:**
        - Gemma-31B neutral: 0.261 vs 0.312.
        - Llama-70B neutral: 0.415 vs 0.430 (CoreWeave) and 0.423 vs 0.439 (Parasail).
        - GPT-4o-mini with own position hidden: 0.309 vs 0.375.
        - GPT-6-luna neutral: 0.530 vs 0.547.
    - **Surrogate rule, fixed before any simulation:** use MF if its held-out CE beats A3 by ≥ 2% relative;
      otherwise use A3 (p = 0).
      - Justification: under per-call random listing order (the Stage 3 protocol), copying the first-listed
        contact IS a random-neighbour voter, so MF's p is the network voter weight.
      - Cluster-bootstrap draws (B = 100) of A3 and MF are saved per endpoint × arm in
        `results/llm/fits/surrogate/`.
    - The rerun with saved draws replaced a first pass that saved only M. That pass was stopped before
      completion, and its outputs were overwritten. No simulation used it.
11. **Step 3 (in progress): Stage 2 simulator** `llm/surrogate.py`, driver `llm/stage2.py`.
    - **Validation:** with Ising rows it reproduces HANDOFF §3. On ER c = 4, ρ = 0, consensus switches on
      at Tc = 3.915 (|m| = 0.59 at T = 3.0, 0.09 at T = 3.9). At ρ = 0.1, T = 2.0, |m| = 0.64 (handoff:
      0.62) with reciprocal ties and 0.53 (0.55) at ε = 1.
    - **Jitter:** fitted coefficients are already jitter-marginal, so per-call noise on top would
      double-count. The explicit mode de-attenuates (η/κ) and adds N(0, σ²). It is run as a consistency
      check against the marginal mode.
    - **Frustration floor** (β = 6 quench, `results/llm/stage2/floor.csv`):
      - ρ = 0: ≈ 0 (random-regular) and 0.02 (ER; leftover domain walls).
      - ρ = 0.25 / 0.5: 0.15–0.17.
    - **Extrapolation:** ER c = 4 has ≈ 11% of nodes with k > 6, outside the fitted k ≤ 6. The
      random-regular d = 4 control stays inside.
12. **Step 3: Stage 2 bug found and fixed before reporting.**
    - **Bug:** the "naive Ising reading" baseline took β from the surrogate's k-scaled fit (A3/MF), whose
      coefficients are at the k = 1 scale. For averaging endpoints this inflated β several-fold; for
      Llama-70B neutral it was 6.95 instead of the Stage 1 additive 0.65. This made the Ising baseline
      spuriously ordered (|m| 0.88 at N = 100, although T_eff = 2/0.65 = 3.08 is above the RRG d = 4 Bethe
      Tc = 2.885).
    - **Fix:** the Ising baseline now uses the Stage 1 additive A0 β and its bootstrap draws. The
      Stage-3-matched runs were rerun, and the first FSS run (wrong baseline) was stopped and relaunched.
      Fitted-surrogate and fields-only rows were unaffected.
    - **Effect:** the apparent Llama-70B neutral contrast (fitted 0.41 vs Ising 0.88) disappears
      (0.41 vs 0.32). GPT-6-luna now matches its Ising reading (0.16 vs 0.16).
    - **Explicit-jitter check:** explicit (η/κ + noise) vs marginal agree within chance (5 of 96 contrasts
      at |z| > 2, about 4.8 expected; `results/llm/stage2/jitter_check_z.csv`).
    - **Fields-only control added** (inertia and fields only, no coupling, no voter step). Workplace
      "consensus" is largely field-driven: fields-only |m| is 0.92 for Gemma-26B, 0.72 for Mistral Large,
      0.54–0.55 for Qwen-122B and Nemo.
13. **Step 3 complete: Stage 2 surrogate simulations and written Stage 3 predictions.**
    - `llm/STAGE3_PREDICTIONS.md` and `results/llm/stage2/stage3_predictions.json` were generated from the
      simulation summaries by `llm/make_predictions.py` and committed before any Stage 3 call.
    - **Point choice:** made after seeing the Stage 2 simulations and before any live data. The criteria
      were confirmatory endpoints only, and separation of the fitted surrogate from both controls on
      pre-stated primary measures.
    - **P3 correction:** its primary measures were revised before commit, from |m| + excess unsatisfied to
      |m| + persistence, because excess unsatisfied did not separate the surrogate from the Ising reading.
    - Summary: `results/llm/STAGE2_RESULTS.md`.
    - **STOP:** no Stage 3 call has been made. Awaiting the owner.
14. **Access check and data privacy (owner request, 2026-10-09).**
    - **Network:**
      - `huggingface.co` and its storage hosts (`cas-server.xethub.hf.co`, `cas-bridge.xethub.hf.co`,
        `cdn-lfs.hf.co`) are now reachable.
      - `api.runpod.io` (GraphQL) is reachable.
      - Still denied: `hf.co` (short-link domain, not needed), `rest.runpod.io` and `api.runpod.ai`.
    - **Credentials:** neither an HF token nor a RunPod API key is present, and the proxy injects
      credentials only for openrouter.ai.
    - **The RunPod MCP connector is not active in this session.**
    - **Account-linked IDs stripped.**
      - The OpenRouter generation `id` was removed from every cached record: the local raw cache and all 19
        published snapshots (verified: 0 remaining).
      - `llm/client.py` no longer stores it.
      - Cache hits are unaffected, because records are keyed by a hash of the request body.
    - **Dataset:** `RandallSPQR/social-phase-transitions-llm-cache`, **private**. The upload script creates it
      private and asserts privacy before uploading.
    - **Git history:** earlier commits on this branch still contain the IDs (raw JSONL in the first commits,
      older snapshots). No force-push. The planned squash-merge puts a single clean commit on main; deleting
      the feature branch afterwards removes the old commits from every ref (owner's choice).

15. **RunPod plugin made permanent for this repo (2026-10-09).**
    - `.claude/settings.json` now declares the official RunPod marketplace (`runpod/runpod-plugins-official` on
      GitHub, marketplace name `runpod`) and enables `runpod@runpod`. It loads in the next session, not this one.
    - The file holds no credentials. The plugin's MCP server is hosted at `mcp.getrunpod.io`, and this
      environment's network policy currently blocks that host (proxy 403). It must be allowed in the environment's
      network settings before the MCP tools can work from a cloud session.

16. **Access check (2026-10-09, new session).**
    - **HF:** `HF_TOKEN` is set (37 characters; value not printed). `huggingface_hub.whoami()` succeeds:
      user `RandallSPQR`, fine-grained token. The token can read `google/gemma-4-31b-it` and
      `google/gemma-4-26b-a4b-it` (both exist and are not gated).
    - **Hosts** (plain GET of `/`, HTTP status):
      - `huggingface.co` 200, `api.runpod.io` 404, `rest.runpod.io` 301: reachable (404/301 are the services'
        own replies to a bare `/`).
      - `mcp.getrunpod.io` 405: reached this time (entry 15 had a proxy 403), but a bare GET is not an MCP call.
      - `api.runpod.ai`: no HTTP response (status 000). Treated as still blocked.
      - Follow-up probes (headers, an MCP `initialize` POST, the proxy status page) were refused by this
        session's own command-approval policy, so the cause for `api.runpod.ai` is not confirmed.
    - **RunPod plugin:** `.claude/settings.json` still enables `runpod@runpod`, but **no RunPod tools are loaded**
      in this session (tool search finds none; the account plugin list has none). No RunPod API key is in the
      environment either. So no read-only RunPod call (pods, GPU types, prices) was possible.
    - **Owner action needed:** add a RunPod API key as an environment secret (variable `RUNPOD_API_KEY`), allow
      `api.runpod.ai` in the environment's network settings, and check why the plugin did not load (it may need
      installing/approving on the account, or the MCP host allowed). Steps 2–3 did not depend on RunPod.

17. **Private dataset upload and round-trip check.**
    - `python llm/upload_cache_hf.py` uploaded the 19 snapshots + manifest to
      `RandallSPQR/social-phase-transitions-llm-cache`; its own re-download check PASSED.
    - Independent check (fresh empty directory, `snapshot_download`, SHA-256 against
      `results/llm/cache_manifest.json`): **private = True**; files expected 19, present 19, matched 19,
      mismatched 0, missing 0, extra 0 (top level: `cache/`, `cache_manifest.json`; the remote manifest is
      byte-identical to the committed one). Records 270,884 = manifest total.
    - **Records carrying an OpenRouter generation `id`: 0** (checked `resp.id`, top-level `id`, and any `"gen-`
      string).

18. **Snapshots removed from git (no history rewrite).**
    - `git rm --cached` on the 19 `results/llm/cache/*.jsonl.gz`; `.gitignore` now ignores them.
      `results/llm/cache_manifest.json` stays tracked and is the pointer to the dataset.
    - `llm/client.py`: when a model listed in the manifest has neither a raw cache nor a snapshot (fresh clone),
      it downloads that snapshot from the dataset and verifies its SHA-256 (needs `HF_TOKEN`). If that fails it
      **raises** rather than silently re-querying OpenRouter (paid, and not reproducible). Models not in the
      manifest behave as before. Tested in a scratch copy without the cache: fetch + restore works (4 records),
      no token gives the error, an unknown model gives an empty cache.
    - `llm/fetch_cache.py` now exposes `fetch()` (used by the client; deletes files that fail the hash check).
      `llm/cache_manifest.py` docstring updated.
    - **Older commits still contain the snapshots** (and, before entry 14, the generation ids). The owner's plan
      is unchanged: squash-merge into main, then optionally delete the feature branches (entry 14).

19. **GPU-stage estimate for the owner. STOP: nothing rented.**
    - No pod, endpoint, volume or other billable RunPod resource was created.
    - **Live prices could not be read** (no RunPod tools or key; entry 16). The figures below are the
      `llm/GPU_PLAN.md` list prices and must be checked in the console before launch.
    - **Plan:** 1× A100 80GB (fallback H100 80GB), one pod, Gemma-4 26B-A4B then 31B, bf16, 16,660 forward
      passes per model.
      - A100 80GB: ~$1.5–1.9/h × 1.5–2.5 h ≈ **$3–5**. H100 80GB: ~$2.7–3.3/h × 1.0–1.5 h ≈ $3–5.
      - Network volume ≥ 150 GB for the weights: a few cents per hour of use (delete afterwards).
    - **How the $15 ceiling is enforced:**
      1. In-pod: `pod_run.sh` has a 4 h hard kill (`sleep 14400`), per-step `timeout`s (30 min smoke,
         150 min main), and runs `runpodctl remove pod` on success or failure. Worst case at 4 h:
         A100 ≈ $7.6, H100 ≈ $13.2, both under $15.
      2. Outside the pod (covers a failed self-termination): when the pod is launched, schedule a session
         check-in at +4 h 15 min that lists pods and terminates this one if it is still present.
      3. Spend check: record the RunPod balance before launch, check it at each check-in, and terminate if
         spend reaches $12 (leaves margin for the volume and billing lag).
      4. Recommended owner-side backstop: hold no more than ~$15 of prepaid credit on the RunPod account
         and keep auto-reload off, so the account itself cannot overspend.
    - **Before launch:** RunPod API access for this session (key + plugin/hosts, entry 16), the pod needs
      `HF_TOKEN` and a RunPod key for `runpodctl`, and the owner's explicit OK.

20. **RunPod access re-check (2026-10-09, owner request).**
    - `RUNPOD_API_KEY` is now set (50 characters; value not printed). The RunPod plugin's MCP tools are still
      not loaded, so access is via the REST API directly.
    - `rest.runpod.io` works with the key: read-only `GET /v1/pods` and `GET /v1/networkvolumes` return 200.
    - `api.runpod.io` (GraphQL: GPU types, live prices, account balance) is now **denied by the egress proxy**
      (CONNECT rejected). `api.runpod.ai` and `mcp.getrunpod.io` answer.
    - The account already has one **running pod from another project** (`item10-grader`, 1 GPU, $1.79/h,
      created 16:10 UTC, self-stop deadline 2026-10-10 01:40 UTC) and a 150 GB network volume in EUR-IS-1.
      Not touched. It counts toward account spend, so the spend check in entry 19 must use a baseline.
    - Nothing created.

21. **RunPod access confirmed; live prices; shared-volume safety (2026-10-09).**
    - After the owner allowed it, `api.runpod.io` GraphQL works. Read-only queries only:
      balance $46.52, spend limit $80, current spend $1.82/h (the owner's SAE calibration pod plus its volume).
    - **Owner instruction: the `item10-grader` pod is another experiment. Never touch it.** Nothing in this
      project's tooling refers to it; `pod_run.sh` only removes `$RUNPOD_POD_ID` (its own pod).
    - **Owner preference: use the network volume `u0isne6ams` (150 GB, EUR-IS-1) for weights, without overwriting
      anything on it.** The volume is shared with the SAE work, so `llm/pod_run.sh` was changed:
      - writes only under `/workspace/social-phase-transitions/` (HF cache in `hf/`), marked by an `.owner` file;
      - aborts if that directory exists without the marker; never deletes anything on the volume;
      - repo, logs and results tarball moved to the container disk (previously `/workspace/...` paths, including
        `/workspace/pod_results.tgz` and `/workspace/hf`, which could have collided with existing files);
      - if the volume has < 130 GB free (minus what is already cached by us), the weights go to the container
        disk instead. Free space on the volume cannot be read through the API, only from inside a pod.
      - Tested locally with a fake volume: fresh, rerun, too-full and foreign-directory cases behave as above,
        and other files on it are untouched. The 4 h hard-limit trap was also tested: results shipping and pod
        removal still run.
    - **Live prices** (secure cloud, 1 GPU): A100 80GB $1.79/h, H100 PCIe $2.89/h, H100 SXM $3.99/h. In EUR-IS-1
      (the volume's data centre) the only ≥ 80 GB GPU in stock is the RTX PRO 6000 Blackwell Server 96GB,
      $2.49/h, stock "Low". It fits Gemma-4 31B in bf16.
    - **Updated estimate:** RTX PRO 6000 in EUR-IS-1 with the volume, 1.5–2.5 h, **≈ $4–6**; worst case at the 4 h
      limit $9.96, under the $15 ceiling. Details in `llm/GPU_PLAN.md`.
    - Nothing created. **STOP: awaiting the owner's explicit OK.**

22. **Owner approvals (2026-10-09) and the Stage 3 pre-data addendum.**
    - Approved: Stage 3 on P1–P4 plus P5 (calibration); the GPU battery (~$3–5, ceiling $15, 4 h hard stop,
      auto-terminate), **on an A100** (the card the owner's other work runs on; wait for one if needed); dataset
      private; squash-merge to main after the upload and the removal of the snapshots from git.
    - Framing for all write-ups: Stage 3 tests inertia, content fields and rival weighting, not non-reciprocity.
      The gate-relaxed H4 result stays in the sensitivity section; local bf16 Gemma decides it.
    - **Pre-data addendum** appended to `llm/STAGE3_PREDICTIONS.md` and committed before any Stage 3 call: (a)
      live prompts = Stage 1 template, checked automatically before every call; (b) no-contacts arms P2-nc and
      P4-nc, prediction = fields-only control, flagged as an extrapolation; (c) what a miss at each point would
      imply; (d) resample-on-invalid (≤ 3 attempts), invalid rate per point, flag > 2%; (e) Ising-β regression test.
    - New: `llm/stage3.py`, `llm/test_stage3.py`, `results/llm/stage2/stage3_predictions_nocontacts.json`.
      `battery.render` takes an optional contact order (Stage 1 battery unchanged: same SHA-256 over all 8,330
      prompts). `make_predictions.py` now preserves the addendum and writes the no-contacts json; rerunning it
      leaves the original predictions byte-identical.
    - `python llm/test_stage3.py`: all checks pass (prompts, Ising β for 27 fits, mock-LLM runner reproduces the
      simulator with |z| < 2).

23. **Stage 3 launched; main squash-merged; GPU runner fixed in a dry run; A100 poller.**
    - **Stage 3 live run started** (`llm/stage3.py run`, 70 chains, budget stop $12) after the addendum commit
      (de0ff08) and one smoke call per endpoint (tag `stage3-smoke`, not part of the data; all 7 endpoints served
      by their pinned providers).
    - **main**: squash of this branch at de0ff08 pushed as 23471e7 (no cache snapshots, no generation ids).
    - **GPU runner dry run** on tiny random-weight Gemma-4 checkpoints built from the real configs, tokenizer and
      chat template (`Gemma4ForConditionalGeneration`, MoE and dense variants), CPU, transformers 5.19.0 +
      torch 2.8.0 (the pod image's torch). Two problems found and fixed before any rental:
      1. **Pinned transformers.** Gemma-4 needs transformers ≥ 5.5; the pod script asked for ≥ 4.57. Now pinned
         to the validated 5.19.0.
      2. **Padding.** The left-padding check failed in bf16 (0.57 nats MoE, 0.15 dense) but passes in fp32
         (8e-5 nats): bf16 numerics, not a masking bug. With a 0.05 abort it would have stopped a correct run.
         **Deviation (implementation, pre-data):** the battery now runs in padding-free batches (prompts grouped by
         exact token length; replicate 1 uses another batch size and a shuffled order within each length), with an
         assertion that no batch contains padding. The padding check is kept as a reported diagnostic. Nothing
         else in Amendment 4 changes.
      - After the fixes both variants run end to end: Stage 1 CSV schema (main + comprehension), activations at 6
        layers, replicates agree.
    - **A100 poller** `llm/runpod_launch.py` (owner: A100 only, wait if needed; owner's volume in EUR-IS-1, which
      lists no A100 stock right now). Polls every 3 min, tries a create when stock is listed and every 15 min
      regardless; then polices the pod: terminate at 4.25 h or estimated spend > $12 (balance drop minus the
      owner's other burn measured at launch), or as soon as the pod's results tarball appears in the dataset while the pod is
      still up (failed self-removal). It only ever touches the pod it created. Pod code ships as a tarball
      in the private dataset (`code/pod_code_<rev>.tgz`); HF_TOKEN is passed as a pod env variable.

24. **First A100 launch failed on the volume quota; relaunch without the volume.**
    - Pod `w9m8lg9txkisc5` (A100-SXM4-80GB, EUR-IS-1, owner's volume) ran 17:27–17:31 UTC (≈ $0.12). It started the
      Gemma-4 26B download into `/workspace/social-phase-transitions/hf` and failed with "Disk quota exceeded"; the
      script shipped its log to the dataset and the pod removed itself, as designed.
    - **Cause (my error):** the free-space check used `df` on the network volume, which reports ≈ 494,616 GB free;
      the volume's real quota is 150 GB and most of it holds the owner's SAE data.
    - **Leftover on the owner's volume:** our partial download in `/workspace/social-phase-transitions/` (marked by
      our `.owner` file). It may have used up the remaining quota, which could affect the SAE pod's writes.
      **Not removed:** deleting anything on the shared volume needs the owner's OK (an attempt to have the next pod
      remove it was blocked by the session's permission policy). Owner decision pending.
    - **Fix:** the pod no longer mounts the volume at all (`pod_run.sh`, `runpod_launch.py`); weights go to the
      200 GB container disk. A100 in any secure data centre. Everything else unchanged.

25. **Owner-approved cleanup of our leftover directory on the shared volume.**
    - The owner approved removing `/workspace/social-phase-transitions/`. `llm/runpod_cleanup_volume.py` started pod
      `lr9trys3tx4vtc` (RTX PRO 6000, EUR-IS-1, ≈ 1 min, ≈ $0.05) with the volume mounted; it removed only that
      directory after checking our marker, uploaded its log (`cleanup/lr9trys3tx4vtc.log` in the dataset) and
      removed itself.
    - Log: volume used 114 GB before; our directory 1.7 GB, marker matched, removed; 112 GB after (of 150 GB).
    - So the leftover was small, and the "disk quota exceeded" error in entry 24 most likely came from the
      download's temporary files. The volume has ≈ 38 GB free, which could not hold the 26B weights (≈ 52 GB)
      anyway; the GPU run stays on the container disk.

26. **Stage 3 complete (data only; pre-stated evaluation applied mechanically, no new analysis).**
    - 70 chains (P1–P5, P2-nc, P4-nc; 5 graphs × 2 replicas × 2,000 updates), 140,000 live updates, $4.95 total
      (budget stop $12 not reached), no errors, no prompt-check failures. Invalid rate ≤ 0.02% at every point
      (flag threshold 2%: none flagged); no agent kept its position after 3 invalid draws.
    - Outputs: `results/llm/stage3/chains/` (every update + states per sweep), `measures_by_graph.csv`,
      `summary.csv`, `evaluation.csv` / `evaluation.json` (`llm/stage3_evaluate.py`), `run.log`.
    - **Pre-stated verdict: the surrogate is NOT supported.** It passes both primary measures at 0 of P1–P4
      (rule: ≥ 3 of 4). The calibration point P5 and both no-contacts arms also miss.
    - Observed (mean of 5 graphs) vs fitted-surrogate 90% interval, primary measures:
      - P1 |m| 0.21 [0.24, 0.46], persistence 0.78 [0.66, 0.74];
      - P2 m 0.56 [0.58, 0.66], persistence 0.60 [0.59, 0.67] (pass);
      - P3 |m| 0.10 [0.20, 0.37], persistence 0.99 [0.91, 0.95];
      - P4 m 0.94 [0.87, 0.91], persistence 0.90 [0.75, 0.84];
      - P5 |m| 0.60 [0.14, 0.24], persistence 0.66 [0.11, 0.20];
      - P2-nc m 0.15 [0.34, 0.43], persistence 0.80 [0.47, 0.54]; P4-nc m 0.64 [0.52, 0.58], persistence 0.70
        [0.41, 0.48].
    - Discrimination: at P1, P2, P4 and P5 the observed primary measures lie outside both the Ising-reading and
      the fields-only intervals; at P3 |m| lies inside both control intervals and persistence outside the Ising one.
    - Observed social amplification (P − P-nc): P2 m +0.41, persistence −0.20; P4 m +0.29, persistence +0.20
      (surrogate predicted P2 +0.23 / +0.13, P4 +0.34 / +0.35).
    - STOP for the owner's report; no further analysis until the GPU battery is also done.

27. **GPU Gemma battery (Amendment 4) ran but returned NO results.**
    - Pod `kwzw7yoclbtlq1` (A100-SXM4-80GB, US-MD-1, $1.79/h, container disk only) ran 17:33–20:38 UTC, 3.08 h,
      ≈ $5.50. The watchdog never terminated it (no TERMINATE in `results/llm/logs/runpod_launch_final.log`); the
      pod disappeared on its own between 20:36 and 20:38.
    - **No results tarball reached the dataset** (`pod_results/pod_results_kwzw7yoclbtlq1.tgz` absent; dataset
      commits end at 17:41). The container disk is deleted with the pod, so the outputs and the pod log are lost.
    - Most likely cause: the end-of-job upload failed and `pod_run.sh` removed the pod regardless (design flaw:
      self-removal was not conditional on a successful upload, and nothing was uploaded before the end). Removal
      from outside our tooling cannot be excluded. The run also took longer than the 1.5–2.5 h estimate.
    - GPU spend so far: ≈ $5.50 + $0.12 (first launch) + $0.05 (volume cleanup) ≈ $5.67 of the $15 ceiling.
    - Not relaunched; reported to the owner. A rerun needs: per-model upload as soon as each model finishes,
      the pod log uploaded every few minutes, and self-removal only after a confirmed upload (otherwise leave it
      for the watchdog, which ends it at the time cap).
