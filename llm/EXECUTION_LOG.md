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

28. **Owner: "that was a total waste." Real-time tracking built and tested before any rerun.**
    - Lesson written into `CLAUDE.md` at the repo root (loaded by every Claude session on this repository): remote
      or paid jobs stream a heartbeat to durable storage, upload and verify each unit of output as soon as it exists,
      never destroy the machine before a verified final upload, print to stdout, are visible to the owner while they
      run, and get an end-to-end test of the whole chain before paying. The owner's global `~/.claude/CLAUDE.md`
      is not reachable from this cloud container; the same rule was given to the owner to paste there.
    - **Pod side:** `llm/pod_tracker.py` (heartbeat every 90 s → `runs/<pod>/status.json` + full `pod.log` in the
      private dataset; per-model upload with re-list and size check; `uploads.json` records what is verified).
      `llm/local_gemma.py` writes live progress (stage, replicate, batch, ETA). `llm/pod_run.sh` rewritten: smoke
      and full outputs uploaded per model as soon as each finishes; job deadline 3.6 h; the final upload retries
      until 30 min past the deadline and cannot be interrupted; the pod removes itself only after a verified upload,
      otherwise it reports `upload_failed` and stays up for the watchdog. All output also goes to stdout (RunPod
      console).
    - **Session side:** `llm/run_relay.py` turns heartbeat + pod state into a dashboard document (alerts: stale
      heartbeat, pod gone without `done`, upload failure, caps); `llm/runpod_launch.py` treats a run as finished
      only when the heartbeat says `done`.
    - **Dashboard:** artifact "Gemma GPU Run Monitor" (progress per model, ETA, caps, GPU and progress over time,
      verified uploads, log tail), fed from the relay.
    - **Tests before any spend** (CPU, tiny random Gemma-4 models):
      - failure path (invalid HF token, failing model): uploads and heartbeats fail, retries continue, phase
        `upload_failed`, pod NOT removed, no leftover processes. It exposed two bugs, now fixed: the job deadline's
        TERM aborted the final upload, and the deadline timer was left orphaned.
      - full path: smoke → upload (verified) → full run → upload (verified) per model → `done`; relay and dashboard
        follow it live.

29. **Rerun 1 (`2nt34s8z62hxwt`) failed in 5 min on the weight download; the tracking showed why at once.**
    - The heartbeat and log reached the dataset; the smoke test failed at the first Gemma-4 26B shard with
      "Disk quota exceeded" from the xet download backend while the container disk showed 0.26 GB used of 215 GB.
      The first launch (entry 24) failed with the same xet error on the owner's volume, where our folder held only
      1.7 GB. So the quota error is the xet backend on RunPod storage, not space. The 3-hour run (entry 27) probably
      died the same way early on and then hung in its single end-of-run upload (not provable: its log was lost).
    - The pod uploaded its log (verified), reported `done` (exit 3) and removed itself: ≈ 6 min, ≈ $0.18.
      GPU stage total ≈ $5.85.
    - Fix: `HF_HUB_DISABLE_XET=1` (classic HTTP download); weights download as their own tracked phase with retries
      (`pod_tracker.py download`), and the heartbeat reports the downloaded GB. Tested locally on a small real model.

30. **Rerun 2 (`225e35mmgqhn3d`): download failed at ≈ 21 GB with "No space left on device"; terminated by us.**
    - With xet disabled the download got further, then failed at 21.4 GB while the heartbeat's disk reading (taken
      at `/root`) showed 0.26 GB used of 215 GB: that reading was measuring the wrong filesystem. The pod also had a
      default 20 GB pod volume at `/workspace`; writes for the weights were evidently capped near 20 GB.
    - Terminated after ≈ 7 min (≈ $0.21) since every retry would fail the same way. GPU stage total ≈ $6.06.
    - Fix: the pod gets its own 200 GB pod-local volume at `/workspace` (deleted with the pod; never the owner's
      shared network volume) and the weights go to `/workspace/hf`; the heartbeat reports free space at the real
      download target plus `df` of `/`, `/root`, `/workspace`; the download step aborts at once if the target has
      < 130 GB free (tested locally: aborts with 15 GB free).

31. **Amendment 4 GPU battery complete (rerun 3, `e3zan9ygnmqe45`). Data only; no analysis.**
    - Rerun 2's real cause, confirmed from rerun 3's heartbeat: the pod image presets
      `HF_HOME=/workspace/.cache/huggingface`, which overrode the `/root/hf` default, so the weights went to RunPod's
      default 20 GB pod volume. With a 200 GB pod volume the download ran normally (`df` in the heartbeat: `/workspace`
      200 GB, `/` 50 GB).
    - A100-SXM4-80GB, 21:37–22:18 UTC (≈ 40 min, ≈ $1.16). Sequence, each step uploaded and verified on finishing:
      26B download → smoke → full (1,319 batches) → 31B download → smoke → full → final upload of everything →
      `done` (exit 0) → pod removed itself. The dashboard showed every step live.
    - Retrieved all 42 files from `runs/e3zan9ygnmqe45/` and checked each size against the dataset (0 mismatches).
    - In git: `results/llm/stage1/{main,comprehension}/gemma-{26b,31b}-local[-smoke].csv` and the pod log. The
      activations (≈ 825 MB) stay in the private dataset; `results/llm/activations_manifest.json` lists them.
    - Integrity checks: per model 12,428 main rows (6,214 prompts × 2 replicates) and 4,232 comprehension rows;
      no missing P(A); no prompt with first-token leak > 5%. Replicate disagreement (logit of P(A), rep 0 vs rep 1,
      different batch composition): 26B MoE median 0.50 nats, 99th percentile 4.05; 31B dense median 0.00, 99th
      percentile 1.00. This is the batch-numerics noise the pre-registered 0.3-nat gate measures; the gate has not
      been applied (no analysis yet).
    - **GPU stage total ≈ $7.22** (≈ $5.50 for the lost 3-hour run, $0.12 + $0.18 + $0.21 for the failed launches,
      $0.05 volume cleanup, $1.16 for this run) of the $15 ceiling.
    - STOP: both Stage 3 and the GPU battery are done; reporting to the owner before any new analysis.

32. **Audit before analysis (owner request, 2026-10-10). No analysis run.**
    - **Pods:** all six pods this project created (`w9m8lg9txkisc5`, `kwzw7yoclbtlq1`, `lr9trys3tx4vtc`,
      `2nt34s8z62hxwt`, `225e35mmgqhn3d`, `e3zan9ygnmqe45`) return 404. The account shows only the owner's
      `item10-grader-2`; current spend rate $1.82/h is that pod alone. No volumes or other resources of ours.
    - **Data backup gap found and closed:** the 140,012 Stage 3 raw API responses (140,005 live calls including
      5 invalid-reply re-draws, plus 7 smoke calls) existed only on this container's disk. Snapshots rebuilt
      (`snapshot_cache.py`, 410,896 records in 19 files), manifest regenerated, uploaded and re-verified by SHA-256
      (`upload_cache_hf.py`: PASSED); 0 records carry a generation id; dataset still private. Fixed
      `cache_manifest.py`, which had dropped the `private` flag and still described the stripped `id` field.
    - **GPU run integrity:** pod log clean (no load warnings, no newly-initialised weights, no errors); downloads
      51.6 GB and 62.6 GB; every unit uploaded and verified. Outputs cover the Stage 1 battery exactly (6,214 main
      cells and 2,116 comprehension cells per model, 2 replicates each); git CSVs byte-identical to the dataset.
      Activations: 6 layers per model, 8,330 rows = index, no non-finite values, no all-zero rows.
    - **Validity check (not a hypothesis test):** local vs provider P(A) on the same 6,214 prompts: logit
      correlation 0.987 (26B vs Parasail), 0.993 (26B vs CoreWeave), 0.987 (31B vs Io Net); comprehension accuracy
      100% local and provider. The local models are the real Gemma-4 models, loaded correctly.
    - Real-model padding diagnostic: 0.84 nats (26B), 0.38 nats (31B); the original 0.05 abort would have stopped
      both runs, so the padding-free batching (entry 23) was needed.
    - **Analysis readiness:**
      - Ready as is: `analysis.py` (fits, noise gate from the two replicates, top-5 `pA` as pre-registered) and
        `comprehension.py`; both run end to end on the local CSV format (tested on a copy with randomised P(A)).
      - Not yet written (pre-registered in Amendment 4): (1) extension of `summarize.py` (local keys in the
        confirmatory set, Gemma-family H4 local 26B → local 31B, Holm over extended cells, both versions reported);
        (2) the local-vs-provider paired bootstrap and attenuation test. The existing per-endpoint bootstraps are
        not paired: they resample each endpoint's own cell set, which differs after leak exclusions, so the
        comparison needs its own refits on shared resampling indices; (3) the exploratory activation probes.
      - Deviations to state in the report: padding-free batches instead of left padding (entry 23); activations
        are from the first replicate (`rep` 0 in the files; the amendment says "replicate 1", meaning the first).

33. **Amendment 4 analysis code written and tested (2026-10-10); my comprehension-summary mistake fixed.**
    - **Mistake found and fixed:** yesterday's readiness test ran `comprehension.py` on a synthetic file; that
      script always wrote `results/llm/comprehension_summary.csv`, so the real 11-endpoint Stage 1 summary was
      replaced by one synthetic row and committed in 1d713d7 (my `git add -A`). Restored byte-identical from
      6c40245 (d9f9453). Scan of every results file changed since Stage 1 closed: no other unintended change.
      `comprehension.py` now takes `--out`; all tests write to scratch.
    - **`summarize.py --amendment4`:** extended confirmatory set (local endpoints that pass comprehension and the
      gate), Holm recomputed over all extended cells, H4 Gemma family uses the local pair and is also tested in its
      own right (Holm over its 3 framings); outputs `_amendment4`. The original mode drops the local fits so it
      never sees them. Regression: original and gate-relaxed modes reproduce all six committed Stage 1 outputs
      byte for byte. Logic test with stand-in local fits: gate failed → 12 confirmatory cells, Gemma family
      'excluded'; gate passed → 18 cells, Gemma family tested, 2 testable H4 families (the amendment's maximum).
    - **`compare_local.py`** (paired bootstrap + attenuation test, analysis.py's own fitting functions). Known-answer
      test (`test_compare_local.py`): slope 1.001 [0.994, 1.008] for identical truth, 0.797 [0.791, 0.803] for
      0.8 × truth; local estimates within 0.03 of truth. Calibration: SD of the provider − local difference over 40
      independent synthetic replications vs the paired-bootstrap SE: ratios 0.8–1.15 (political), 0.75–1.53
      (neutral_noown, the smallest arm): roughly calibrated, mild under-coverage on some coefficients in small arms.
    - **Property of the pre-registered verdict rule:** "≥ 90% of ~31 correlated coefficients agree" can fail when the
      truth is identical (one null replication: 72% agree). Its null operating characteristic is measured by
      `compare_local_null.py` and reported with the results; the rule itself is applied unchanged.
    - **Exploratory probes (`probes.py`)**: known-answer test on synthetic activations (ally side encoded): ally AUC
      1.00, rival/own 0.47–0.52, shuffled ≈ 0.5. The pre-registered letter-count control is strong by construction
      (AUC 0.96–1.00 in letter space), a high bar for "decodable above the control".
    - **Unembedding rows** for A/B fetched by HTTP range from the safetensors shards (`unembed_rows.py`; tied
      embeddings, `model.language_model.embed_tokens.weight`); sanity: finite, distinct rows for the real letter
      tokens (ids 236776/236799).

34. **Amendment 4 analysis complete (2026-10-10).** Write-up: `results/llm/AMENDMENT4_RESULTS.md`.
    - Gate: 31B local passes (0.114); 26B local fails (0.592: the MoE's bf16 output moves with batch composition).
    - Confirmatory set 12 → 15 cells (31B local). H1 12/15; H3 cubic replicates the two Gemma-31B positives from the
      gate-relaxed sensitivity analysis (neutral, political) on the clean endpoint; no original H1/H3 verdict
      changes; re-Holm moves Qwen-122B political H2-content from 0.034 to 0.102. H4 Gemma family not testable
      (26B fails the gate); overall H4 still not supported.
    - Local vs provider: all three pairs 76% agreement → pre-registered verdict "jitter biases estimates"; every
      slope CI contains 1 and κ. Measured null operating characteristic of the rule: 80% "averages out" under
      identical truth, minimum agreement 76% → disagreements probably real but small (h_L, h_O political, γ).
    - Probes (exploratory): rival side decodable above the letter-count control in both models (0.986 / 0.976 vs
      0.955); "represented but not used" present for 26B (|b_r|/b_a 0.07; fit from the gate-failing endpoint),
      absent for 31B (0.71); rival direction orthogonal to the A−B readout at the last layer.
    - Process: the chain was killed by a container restart, rebuilt to save per unit, and resumed after a second
      restart; a probe AUC bug (saturated probabilities) was caught in the output and fixed before results were kept.

35. **Owner's next list (2026-10-11) and Amendment 5.**
    - **Process:** `llm/bg.sh` (own process group, PID file, stop by PID only; tested: the whole group stops, nothing
      left over); rule added to CLAUDE.md. The pod runner writes its own PID file and the pod script stops it by
      PID (no more `pkill -f` there either).
    - **Amendment 5** committed before any run (7705d12): rescue rerun of local Gemma-26B at batch size 1 with
      deterministic kernels; summarize `--amendment5` mode (original, gate-relaxed and Amendment 4 outputs
      reproduce byte for byte). End-to-end CPU test of the exact configuration: replicates identical (max |Δ| 0.0).
    - **First A5 pod `5407atbbj7bwit` stopped by us after 8 min (≈ $0.25):** the measured batch-size-1 rate (1.86
      forward passes/s → ≈ 149 min for the main battery, ≈ 2.8 h in all) exceeded the pod's 2.2 h job deadline,
      and the runner wrote its outputs only at the end, so the run would have been cut off with nothing saved.
      Caught from the live ETA within 8 minutes. **Deviation from Amendment 5's caps** (my 40–70 min estimate was
      wrong): job deadline 3.3 h, watchdog 3.5 h / $6.50; worst-case GPU stage total ≈ $13.7, under the owner's
      $15 ceiling.
    - **Runner checkpoint added:** replicate 0's outputs (CSVs + activations) are written and uploaded, verified,
      as soon as replicate 0 finishes (`<key>-rep0`), so a late failure loses at most replicate 1.
    - **Exploratory cubic analysis (`llm/cubic_31b.py`, Gemma-31B local):** political: T_aaa −0.48 [−0.84, −0.37]
      dominates (CV gain alone 4.7%; full cubic 2.4%): a saturating ally response; neutral: T_rrr −0.36, T_aar
      −0.36, T_arr +0.35 (CIs exclude 0), full cubic CV gain 3.0%. A pure majority rule (sign of M_a, M_r) is far
      worse than linear (held-out CE +21% neutral, +55% political); adding it to the linear model gains ≤ 0.5%.
      So the cubic effect is mild saturation, not a switch to majority rule.
    - **Local-vs-provider framing check (31B, the clean pair):** format fields agree 7/10 (h_O ×2, h_L ×1
      disagree); social coefficients agree 9/12, the 3 disagreements all in workplace (b_a, b_r, β; 7–9% lower at
      the provider). The proposed headline needs this qualification.

36. **Amendment 5 and follow-ups complete (2026-10-11).** Write-up: `results/llm/AMENDMENT5_RESULTS.md`; revision note
    added to `AMENDMENT4_RESULTS.md`.
    - Run `4p4a463q78clp0` (≈ 2 h 28 min, ≈ $4.40): 26B at batch size 1, deterministic; replicates identical; replicate
      0 checkpointed and uploaded mid-run. GPU stage total ≈ $11.90 of $15.
    - Rescue results: gate passes (repeat SD 0.000, determinism only); 18 confirmatory cells, no Amendment 4 verdict
      changes; Gemma-family H4 (26B-bs1 → 31B) significant in all three framings (Holm 0.003); overall H4 still not
      supported (2 of 2 testable; ≥ 3 of 4 required).
    - Secondary: Parasail and CoreWeave vs the clean 26B agree at 86% and 83% (vs 76% against the batched 26B).
    - Batch effect (exploratory): batched vs bs1 RMS 0.86 nats (unbiased, r 0.990, 2.5% majority flips); replicate
      difference 0.59 understates it. Stage 2 "batched serving" σ ≈ 0.86.
    - Null recalibrated at B = 1000, 100 replications: rule "averages out" 77% under identical truth; agreement 1st
      percentile 83%, none ≤ 76% → the Io Net 31B pair's differences are real. Qualified headline: format biases
      shift; social coefficients agree in 2 of 3 framings, 7–9% lower at the provider in workplace.
    - Probe follow-ups (exploratory): 26B decodability above the letter-count control does not survive a CI; 31B
      deconfounded rival signal from layer 30 (AUC up to 0.96); readout cosine uninformative in 31B (all directions at
      baseline; earlier "orthogonal" statement withdrawn), discriminative in 26B (own 100th, ally 99th, rival 1st
      percentile).
    - Process: the first A5 pod was stopped after 8 min when its live ETA showed the run would exceed its deadline;
      the null check was found oversubscribed (load 20 on 4 cores) and restarted single-threaded (100 replications
      then took ~1 h); a monitor filter was silent because mawk buffers piped input (fixed with -W interactive).

37. **Stage 3 harness audit (owner request 2026-10-12; plan committed first, 4c3924a).** No rerun, no fix, no API
    call; results in `results/llm/STAGE3_AUDIT_RESULTS.md`, code `llm/stage3_audit.py`.
    - B1–B11: all pass with zero mismatches over 140,005 cached calls (prompt text and raw replies recomputed
      independently; exact replay of all 70 chains from the cache; live measures reproduced to 1e-16 by the
      simulator's own measurement code; one provider per point, no fallback; 5 invalid replies, all resampled).
    - A (surrogate mock through the unchanged harness): passes at P1, P3, P4, P5, P2-nc; fails the pre-stated rule at
      P2 (29/40 repetitions inside, threshold 30) and P4-nc (live-seed run 0.511 vs 0.52). Post-hoc: the simulator on
      the same 5 graphs matches the mock (paired |Δ| ≤ 0.013, |z| ≤ 2.3) and covers those intervals only 80–86%, so
      the failures are interval coverage conditional on the graphs, not the harness. Live misses at P1, P3, P4, P5 and
      the nc arms are 3.6–20 SD from the same-graph surrogate; P2's miss is −1.5/−1.0 SD (consistent with the
      surrogate on these graphs). Pre-stated verdict unchanged.
    - C: calibration-in-the-large within ±0.003 everywhere; slopes flagged at P3 (near-0/1 battery probabilities) and
      P5 (30-sample battery shares). Post-hoc: on calls whose contact order equals the battery's, live pA matches the
      battery (RMS logit 0.06–0.35); with a different order 0.44–0.97: per-call contact-order effects, absent from the
      A3 surrogate.
    - Audit-code bug found before reporting: the invalid-reply CSV kept only the last point's rows; fixed and the log
      audit rerun.

38. **Three-kernel test, K_fixed and K_live (owner request 2026-10-12; plan and outcome map committed first, fe65958).**
    No API calls. Results in `results/llm/KERNEL_TEST_RESULTS.md`; code `llm/kernel_test.py`.
    - Table-lookup simulator = `surrogate.run` with only the update rule replaced; self-test vs `surrogate.run` passes.
      The self-test caught a bug in the new code before any result (nc arms looked up unseen configurations); fixed.
    - Pre-registered verdicts (P1–P4 passing): K_fixed (battery kernel) 2, K_fixed-A3 1, K_live 2 at n0 = 5, 20 and 100
      → **K_live fails** → outcome-map branch "not Markov in configuration; investigate hidden state before claiming
      anything". No claim made.
    - Post-hoc investigation: K_live estimator unbiased in a known-answer test; no per-step hidden state (time
      inhomogeneity, history dependence, lag-1 correlation all ≈ 0 within ±0.002–0.01; identical-prompt repeat noise
      at Stage 1 level; no wall-clock drift). Live persistence nonetheless sits 1.6–2.3 SD above K_live at 5 of 7 points
      (flip rate below at 7 of 7). Unresolved.
    - Amplification: battery vs live kernel per step |ΔP| 0.002–0.010 (0.07–0.36 nats) at P1–P4; A3 vs live 0.2–1.6
      nats → 3–20 SD collectively. Most of the Stage 3 miss is the parametric A3 surrogate, not the battery.
    - K_avg cost estimate: 5,292 calls, ≈ $0.28. Not run; awaiting approval.

39. **Kernel test continued (owner approval 2026-10-12: neighbourhood test first, then K_avg with both options).**
    - Neighbourhood/concurrency test (free): no dependence of live residuals on the 2-hop state, recently changed
      contacts or time since the last update; a small same-time-window offset only at Io Net (ICC 0.003); live
      trajectories visit ordinary configurations. No mechanism for the persistence excess found.
    - K_avg (amendment 1 committed before calls, f6b1922): 10-call end-to-end test, then 11,992 calls, $0.58, 0 errors,
      1 invalid; cache snapshots uploaded to the private HF dataset and verified. (Amendment text said ≈ 7,500 calls;
      the true count is 11,992, same cost.) Two analysis-code bugs fixed before reading (P5 Beta series; fallback
      metric).
    - K_avg vs K_live per step: |ΔP| 0.001–0.004 (0.03–0.07 nats) at P1–P4. Collectively K_avg passes 2 of P1–P4
      (misses P1 |m| z −1.8, P3 persistence by 0.0002) and matches K_live's predictions within 0.9 SD. Outcome map:
      K_live fails → no claim; open item is the live-vs-Markov persistence excess.

40. **Residual test 1, heterogeneity (owner request 2026-10-12; plan and predictions committed first, 9c759f7).**
    Results in `results/llm/RESIDUAL_TEST_RESULTS.md`.
    - Runner gained `run_tag` for replicates (original chains replay byte-identically; pre-data checks pass); numeric
      K_avg predictions for the replicate's starts committed with the plan.
    - a) All 140,005 prompts = deterministic render of (c, s0, mapping, option order, contact order); one request body
      per point; no agent-specific tokens (prediction confirmed).
    - b) Pre-stated permutation test significant (Fisher p 0.008 agent, 0.003 position; P1, P4, P4-nc), contrary to
      the prediction. Post-hoc stratified null (labels shuffled within (c, s0)): only P4-nc remains (p 0.01-0.04).
    - c) Heterogeneous kernel (τ ≤ 0.11 logit) leaves the persistence excess unchanged (Stouffer +2.84 vs +2.72):
      heterogeneity does not explain it. Test 2 (paid replicate) awaits approval.

41. **Residual plan amendment 1 (option B approved; power 0.81; asymmetry; sentinels) and the identical-request test
    (free, before any paid call).** Pre-stated expectations showed "excess" agreement at P2, P4, P2-nc, P4-nc, but they
    ignore that a repeated prompt selects on the earlier choice (stay in 97-100% of pairs). Selection-corrected:
    excess -0.001 to +0.003 (two marginal at logprob points = draw luck). Pre-stated re-simulation: the biased excess
    overshoots (persistence Stouffer -13.4); the corrected one leaves +1.74. No mechanism → option B with sentinels.

42. **Option B replicate launched (2026-10-13 00:46 UTC).** Runner `llm/stage3_replicate.py` proved end to end in mock
    mode (70 chains, 5 streams, sentinels, HF upload verified 148/148, restart skips finished chains). The k = 0 K_avg
    battery (32 calls, $0.0003) ran first and the nc predictions were committed before any chain (93bd950). Live job
    `rep1`: 5 streams, one call in flight per endpoint, sentinels every 5 min, heartbeats + verified HF uploads every
    15 min to runs/stage3-rep1/, $7 cap. The pre-registered analysis code (`llm/stage3_replicate_analyse.py`) is
    committed before the run finishes and was tested on the mock output only. Self check-ins scheduled.

43. **Replicate interrupted by a container restart (2026-10-13 ~00:59 UTC); resumed 01:2x UTC.** The owner's app crashed
    ("memory overload") and the session container restarted; the `rep1` process died 13 min into the run (last cache
    write 00:59:20, ~3,400 chain calls done, all cached). No memory limit inside the container (cgroup peak 2.6 GB,
    0 OOM kills), so the run did not cause it, but its footprint was cut anyway: `CLIENT_SLIM=1` keeps only replies in
    the in-memory cache index (1.6 → 1.1 GB at load); heartbeats now report RSS and peak RSS. Resumed from the cache
    (finished calls replay free; sentinel rounds continue from the same t0, leaving a gap in the drift trace).

44. **Replicate halted: the session container is reclaimed when the session goes idle (2026-10-13).** Second
    interruption at 01:29:20 UTC, about 1–2 min after the session's turn ended; the first (00:59:20) also followed the
    end of a turn. Container uptime reset both times; no memory limit or OOM involved. Background jobs therefore do not
    survive this environment between turns, so a 27-h run cannot be hosted here. Progress kept (local cache, uploaded
    and verified to HF runs/stage3-rep1/): 1 of 70 chains complete, ≈ 9,400 chain calls and the sentinel rounds so far,
    $0.39. Self check-in cancelled; awaiting the owner's choice of host.

45. **Interruption record for the replicate (owner instruction 2026-10-13: log every interruption).** Correction to
    entry 42: the run started 00:51:21 UTC (not 00:46). Segments so far, from the job log and cache timestamps (also in
    `results/llm/stage3_rep1/interruptions.jsonl`, uploaded with every heartbeat; each future (re)start appends itself):
    - segment 1: 00:51:21-00:59:20 UTC (session container reclaimed). Mid-run at the end: P1, P2, P3, P4, P5 (graph 0,
      replica 0 each).
    - segment 2: 01:15:28-01:19:45 UTC (stopped by me to fix the upload verifier). Mid-run: the same five chains.
    - segment 3: 01:19:45-01:29:20 UTC (container reclaimed). Saved by then: P3 g0 r0. Mid-run: P1 g0 r0, P2 g0 r0,
      P3 g0 r1, P4 g0 r0, P5 g0 r0.
    The run moves to a RunPod CPU pod (owner decision): `llm/pod_replicate.sh` + `llm/runpod_cpu_launch.py`; the pod gets
    only references to the owner's two RunPod secrets (HF token, dedicated OpenRouter key with a $10 cap); it restores
    the run state from HF runs/stage3-rep1/ and resumes. Sentinel drift is analysed within continuous segments only
    (analysis amended before any result); the primary persistence test uses the resumed cache as pre-registered.
    Mock-run files deleted from HF and locally (owner OK).

46. **Replicate pod attempts (2026-10-13).** Test pod `tdq1s9av3qnema` (03:27-03:39 UTC) idled after boot with no
    visible output, because the boot step logged to a file only. The boot was rewritten (stdout + boot-log heartbeat to
    HF, secret presence by length) and tested locally end to end. Test pod `4np9j3upf9p17c` (03:49 UTC to its
    termination on the owner's OK) ran but could not write to HF: the `HF_TOKEN` secret was read-only. The owner
    replaced it with a write token. Whatever that pod collected in its ≤ 9-min test window was never uploaded and is
    lost (at most ~9 min of calls, a few cents). It is not part of the replicate: the next run redoes those chain steps
    from the last HF state with the same prompts (deterministic per-chain RNG), so no data are biased. Pod cost of both
    tests ≈ $0.05. A third test pod follows before the full launch.

47. **Correction to entry 46, and the end-to-end test (2026-10-13).** Test pod `4np9j3upf9p17c`'s data were NOT lost:
    its final-upload retry loop failed 7 times while the token was read-only, then verified 16/16 files at 04:06:25 UTC
    once the owner made the token writable; I terminated it at 04:06:27, 2 s after that verification (its start record
    is in interruptions.jsonl). Test pod `ts8ufqgcra5s7i` (04:06:43-04:18 UTC) then proved the whole pod chain:
    secrets resolved (reported by length only), restore from HF, resumed run, heartbeat upload verified 16/16, chain
    P3 g0 r1 saved, deadline stop, final upload verified 18/18 on the first attempt, then self-removal (pod gone, 404).
    Replicate spend so far ≈ $0.97 (API) + ≈ $0.07 (test pods). Full pod launched next (JOB_HOURS 40).

48. **Replicate interruption 4 (pod `ojaz7c51i59doy` disappeared, 2026-10-13 ~07:22 UTC).** The pod ran 04:19:34-~07:22 UTC
    healthy (last heartbeat 07:20:52: 35+ chains, API $4.53, RSS 1.6 GB of 8 GB, every upload verified; boot log ends
    07:22 without an error). At the 09:23 check-in the pod no longer existed (REST 404 = terminated, not stopped). It
    did not remove itself (that requires a finished run; status said not done) and the 06:52 check-in did not
    terminate it (no cap near). Cause unknown: terminated outside this project's code (RunPod host or manual). Lost: at
    most the calls between the 07:20:52 upload and the end (≈ 2 min); those chain steps are redone from the HF state.
    Chain P2 g4 r1 was saved on the pod after the last upload, so it is redone in full. Relaunched from HF at the
    09:2x check-in; the new start record in interruptions.jsonl lists the chains mid-run.

49. **Replicate interruption 5 (pod `cetayr8gsy6xcf` terminated externally ~12:24 UTC).** Same signature as entry 48:
    healthy to the end (last heartbeat 12:16:35, 62/70 chains, API $5.16, RSS 1.9 GB of 8 GB, uploads verified; boot log
    ends 12:24:04 mid-run without the runner's exit line), then REST 404. **Both pods lived ≈ 3 h** (04:19:01-07:22,
    09:25:10-12:24): consistent with an external 3-hour lifetime limit on pods (not in this project's code: the pod only
    removes itself after a finished run, and check-ins terminate only at 44 h / $8 / verified finish). Owner confirms
    it was not them. Relaunched immediately; 8 chains remain (~1 h), inside the 3-h window.

50. **Replicate interruption 6: runner bug on a provider rate limit (pod `xfwih37n1gc1zo`, 13:43-14:30 UTC).** At
    ~14:30 Parasail returned "qwen/qwen3.5-9b is temporarily rate-limited upstream" (HTTP 429) beyond the client's
    retries. The runner (my code) then (a) stopped ALL streams on one stream's error and (b) reported "done" because
    every stream had finished, so the pod made its verified final upload (14:30:24) and removed itself with 67/70
    chains (P2-nc g4 r1, P4-nc g4 r1, P5 g4 r1 incomplete). No data lost. Fixed: a failing chain waits 120 s and resumes
    from the cache (up to 30 times) while the other streams continue; only the budget stop halts all; "done" = every
    chain saved. Tested locally in mock mode with an injected 429. This pod lived 47 min, so the ~3-h pattern
    (entries 48-49) did not apply. Relaunched for the last 3 chains.

51. **Replicate finished (70/70, 16:00 UTC) — and a graph mismatch found before reporting (DEVIATION).** The pre-registered
    analysis ran first (`results/llm/stage3_rep1/analysis.json`: persistence Stouffer Z = -0.48) but excess unsatisfied
    ties sat 3.6-8.9 SD above prediction at P1-P4. Tracing it: the pods built their random-regular graphs with
    networkx 3.3 (preinstalled in the PyTorch image; the unpinned `pip install networkx` left it), this session with
    networkx 3.7 — same generator and seeds, different graphs and tie signs. Verified against every logged configuration:
    69/70 replicate chains ran on the networkx-3.3 graphs, P3 g0 r0 (finished in the session container) on the original
    3.7 graphs. So the replicate ran on 5 OTHER graphs per point from the same ensemble, not the original 5, and the
    pre-registered analysis predicted and measured on the wrong graphs. Graph-independent results stand (sentinel drift
    test; per-step kernel replicate vs original |ΔP| ≤ 0.003). Both graph sets saved as files
    (`results/llm/stage3_rep1/graphs/`, with `chain_graph.json`). Reanalysis `llm/stage3_replicate_reanalyse.py` written and
    committed before running it: same kernel tables, starts and rules, simulated and measured on the actual graphs. (The
    v1 numbers had been seen; the reanalysis has no free choices.) Prevention: pin dependencies in the pod and assert a
    graph hash at start (next commit).

52. **Replicate result (reanalysis on the actual graphs, `results/llm/stage3_rep1/reanalysis.json`).** Persistence
    Stouffer Z = -0.42 (p 0.66): the excess does NOT repeat; both original misses moved (P1 |m| now high, P3
    persistence now slightly low). Excess-unsatisfied artefact gone on the correct graphs. Sentinel drift-chain
    correlation null. Per-step kernel unchanged vs the original run. Map: excess does not repeat → narrowed headline,
    read as "realisation noise or our own concurrency; not separable" (asymmetry rule). Prevention: networkx pinned in the
    pod, graph-fingerprint assertion in `stage3.graph`, rule added to CLAUDE.md.

53. **Replay audit of the replicate: PASS (plan f06db97).** 70/70 chains replayed exactly on their recorded graphs from the
    cache (0 misses over 140,003 calls; logs, pA and states identical; 2,000 updates each; providers pinned; 3 resamples,
    0 kept). 23 resume points, all consecutive, no re-drawn requests. 730 replies in 5 chains were reused from an
    earlier host's byte-identical request (valid, older by up to ~3 h); 6,656 orphaned calls ($0.27). Combined
    persistence effect: original +0.0116 [+0.0033, +0.0200], replicate −0.0051 [−0.0135, +0.0033], pooled +0.0033
    [−0.0026, +0.0093]; between-run heterogeneity Q = 7.6 (p 0.006).

54. **Amplification curve: plan committed before any run (`llm/AMPLIFICATION_PLAN.md`, owner spec).** Temperature axis
    (per-prompt logits / T on K_avg; 1/T rule assumed for Qwen-122B and Nemo-12B), N = 100/400/1600 random-regular
    d = 4, T_c from the Binder crossing; synthetic and real error-shape perturbations; amplification = |Δ| / per-step
    nats with bootstrap CIs; decisive test H1 (N-growth at T_c) + H2 (beyond the controls P2, P4). Simulation only.

55. **Amplification stage A → Amendment 1 (deviation, before any perturbation run).** Base kernels: P1 is ordered only
    for T ≈ 0.8–1.6 (disordered above; disordered or frozen below). The mechanical first-crossing rule returned
    T = 0.36, a spurious crossing in the frozen region. N = 1,600 is not equilibrated in 200 sweeps. Stage A2
    (hot/cold starts, 1,000 + 1,000 sweeps, fine grid [1, 3]) now locates T_c at the order → disorder edge and fixes
    the stage B protocol length by an equilibration rule. Stage A had also failed once on a missing parquet engine;
    units are now stored as csv.gz.
    Stage A2 first start: a stage-name check (`stage == "locate"`) let A2 units simulate every perturbation as well
    (about 30× slower). It was stopped after 4 units at T ≤ 0.25, and those files were deleted unread. The check is now
    `startswith("locate")`, and A2 was restarted from scratch.

56. **Amplification stage A2 (amendment 1), base kernels only.** P1 T_c = 1.93 (95% CI 1.92–1.96) from the
    U(400)–U(1600) crossing. Checks agree: U(100)–U(400) at 2.02, scaled-m² crossing at 2.02. Below T ≈ 0.65 the
    dynamics are frozen (N = 1,600: hot start |m| 0.07–0.45, cold start 0.99), not a disordered phase. P2 and P4 show
    no crossing in [1, 3]. Stage B: W = M = 500 (hot/cold gap ≤ 0.017 on the band), R = 400; band 1.21–3.10, plus the
    coarse points outside it and T = 1. The Stage 3 temperature is T/T_c ≈ 0.52, deep in the ordered phase.

57. **Amplification curve, stage B and analysis (`results/llm/AMPLIFICATION_RESULTS.md`).** 360 units, 4.9 h CPU.
    Pre-registered verdict: **refuted on H2** for both |m| and persistence. H1 (N-growth at T_c = 1.93) holds for all
    three synthetic shapes (|m| G = 3.0, 3.4, 5.5, all CIs > 1). H2 fails because the majority-shift ratio against
    P2 is uninformative: P2's amplification ≈ 0 at T*, so its G has CI 0.9–68. Stay-bias and noise beat both controls
    3–5×. Expectations: (i) peak at T_c holds; (ii) and (iii) fail (symmetry-breaking noise above T_c; freezing edge
    at T ≈ 0.6–0.8; near-zero denominators in the controls); (v) K_fixed shape grows at T_c (G 2.5), A3 does not;
    (vi) linear response holds. Stage 3 sits at T/T_c ≈ 0.52, where the same errors move |m| about 40× less than at
    T_c with N = 1,600.
