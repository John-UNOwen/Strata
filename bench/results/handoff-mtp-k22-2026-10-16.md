# Handoff: k22 + MTP deterministic NaN on the A770 stage (2026-10-16)

Scope note: the cross-card MTP plumbing questions (warmup deaths, queue placement, embedding table) are
resolved on branch `mtp-cross-card` (HEAD 25bb337). What remains is a deterministic NaN in the main-model
forward under `--mtp` on the k22 split, and the standing >4 GiB buffer-layout problem. That is this handoff.

## Fixed here (do not re-investigate)
| Fault | Fix | Proof |
|---|---|---|
| Drafter private queues on the secondary GPU: eager first-kernel UR 40 / graphs DEVICE_LOST+TDR | root: `NativeEmbed` table allocated in the default-context queue, read by A770 kernels (`cross_context` class); + queues now created under `OnDevice(device_)` (`0ae1e47`) | per-context host mirror (`79e5b7f`): warmup survives, `host mirror for a foreign queue context (322 MiB)` prints, k22 mtp boots |
| Sporadic `!!!!!` (token 0) storms, moved run to run (7537 / 7578 / 7587) | gemm staging scratch was one thread_local buffer shared by all queues of one thread; now one buffer per queue (`9953a5b`) | 32k no-mtp control `start-2c-k22-check-graph-nomtp.bat`: died twice pre-fix, then 3/3 clean post-fix |
| Drafter embed gather reading B580-context host USM | CTXCHECK evidence + mirror | `CTXCHECK record_front embed table: ... owned: host USM on Intel(R) Arc(TM) B580 Graphics` |

## Current failure (deterministic, mtp-only)
Config: `strata-iq2_xs-2card-mtp-k22-check-graph.json` (k22 split, `--mtp`, graph mode, `STRATA_DBG_NAN=1`,
max-context 32768). Three consecutive runs with exe `9953a5b` (09:29 build): all died, no race variance left.
Signatures:
- `strata dbg: verify window at position 7578/7630, row 0/2: 248320 of 248320 logits non-finite (token out 0)`
- `strata dbg: layer 27 (mmq 0, types 16/42, ~380-433 experts): non-finite GU <partial> H -1 Dm <full T*N> bo <full T*N> of T <chunk>` - repeated across prefill chunks (T 2048/1393/1022) in the newest run, i.e. NaN already during warmup prefill
- while `fp16 inputs: activations 0 non-finite, dequantized gate/up 0 non-finite` - the expert GEMM inputs are intact
- Controls: same split/env WITHOUT `--mtp`: clean 3/3. Single-card A770 mtp: clean. `--split-device 0` + mtp: clean (slow, 10-11 tok/s). So: mtp presence flips this on, at the A770 stage, layer 27 (5th layer of the 22-47 stage).
Probe fields decode (`prefill.cpp:4258-4289`): counts of non-finite floats in the MoE workspace; `bo` is the attention-output buffer written by `native_proj` (`prefill.cpp:3194`) - fully non-finite means the residual/attention flow is already garbage before the experts.

## Suspect areas (updated 17:05 - region correction + cache64 test)
Region attribution for the DBGALLOC maps (from boot-log anchors): the `FFFFD4...` region is B580 (the `711.78 MiB` = "711 MiB of weights loaded" for CUDA0 layers 0-21; the `7273.88 MiB` = the B580 cache "5421 slots 7.10 GiB"). The `FFFFB8...` region is A770 (the `6518.07 MiB` = CUDA1 cache "4634 slots 6.37 GiB"; the `675.00 MiB` = the drafter's "839 MiB of VRAM (experts 675, dense 111)").

CORRECTION to the 16:50 note below it: the near-identical 7.27 GiB arena is the **B580** one (7273.88 vs 7275.32). The **A770** layout DOES shift with mtp: nomtp has one arena `+7364.92 MiB @ FFFFB80292A00000`; mtp has the drafter experts `+675.00 @ FFFFB80299A00000` inserted and the cache arena moved to `+6518.07 @ FFFFB802C7800000` - A770 bases/offsets move by ~0.85-1.4 GiB. The A770 layout-shift suspect stands (it was wrongly eliminated). What keeps it murky: no GEMM was staged in ANY 32k run (clean or NaN), i.e. no oneMKL operand ends > 4 GiB inside a tracked allocation - the shift only bites through a path that assumes arena offsets or reads an operand the map cannot see.
What still holds from 16:50: the RoPE-table suspect is out (`native_rope_apply` computes angles analytically; `cos_tab`/`sin_tab` are never read with native rope on); no GEMM staged at 32k; the >4 GiB oneMKL bug is only demonstrably live at 64k.
CACHE64 TEST (16:46, `--expert-cache 64` -> 68 slots 0.09 GiB, appended to the check-graph log): request 1 CLEAN (329 tokens, drafts 208/264 accepted, slow 10.7 tok/s as expected), request 2 NaN again (`verify window at position 1325, row 2`). A tiny cache does not prevent the failure. Caveat: the prompt path can still borrow those 68 slots, so the cache-slot carrier is not fully refuted - but this pushes weight toward the reviewer's section-2 reading (stage persistent state) over cache-content poisoning.
Carry-over evidence stands (16:39 run): request 1 HEALTHY after 34 drafter rounds; request 2 is pure main-model prefill (`layer 33, mmq 0 ... of T 1047`) producing `bo/Dm` fully non-finite with finite expert inputs, then window NaN at 1047. Failure position random run to run (1047 / 1325 / 7578 / 7630 / 7587); only the "mtp on k22 split" precondition is stable.

## Repro and instruments
- Fails: `start-2c-mtp-k22-check-graph.bat` (and the eager variant `start-2c-mtp-k22-check.bat` storms too). Warmup dies inside ~60 s; nothing to wait for. Variant `start-2c-mtp-k22-check-graph-cache64.bat` (`--expert-cache 64`) still NaNs; its output lands in the same check-graph log.
- Clean controls: `start-2c-k22-check-graph-nomtp.bat` (32k no-mtp), `start-2c-mtp-k22-check-sd0.bat` (mtp, single device), `start-1c-a770-mtp-check.bat` (single card, eager). The 64k controls (start-2c-k22-ctrl64k*.bat + configs/logs) and all older experiment files live in `run-archive\` (git-excluded) since 16:40.
- New instrument 16:35 (commit 120a404): `STRATA_DBG_ALLOC=1` prints a `DBGALLOC` dump of every tracked device allocation (base + size) at prompt end; both 32k check configs have it on. Compare the maps in the clean vs NaN logs to see what crosses 4 GiB and what mtp adds untracked.
- `STRATA_DBG_NAN=1` window/GU probes; `STRATA_MTP_CHECK_CTX=1` CTXCHECK lines (embed table, R_rows, window_R, forward phases A..G); probe matrix `C:\strata-deps\cross_usm_probe.exe` (modes `shared copy perdevice dev dev9b dev10`).
- Always re-run gates N>=3 times: a single clean run on this split proves nothing (proven).

## Review (2026-10-16, code reading only, nothing run)

### 1. The controls differ from the failing run in the drafter's prompt pass, not only in the second card
The drafter's K/V over the prompt is filled one of two ways:
- **E-9:** batched through the prompt path, `sp.draft_kv` (`generate.cpp:7919`). It runs only when `!multi_gpu`.
- **E-4:** per group, through `MtpDrafter::prefill` (`mtp.cpp:1379`), called from the last stage's `on_chunk` (`prefill.cpp:4522`). On a split, `on_chunk` is moved to the last stage (`generate.cpp:7962`), so it runs on the stage-1 `std::async` thread (`prefill.cpp:4457`).

| Run | `multi_gpu` | Drafter prompt pass | Result |
|---|---|---|---|
| `start-1c-a770-mtp-check.bat` (A770 alone) | false | E-9 batched | clean |
| `start-2c-mtp-k22-check-sd0.bat` (`--split-device 0`) | false | E-9 batched | clean |
| `start-2c-mtp-k22-check-graph.bat` (k22, two cards) | true | **E-4 per group, on the stage-1 thread** | NaN |

So no run yet separates "two cards / two contexts" from "the E-4 path". E-4 has a record of being fragile: `mtp.cpp`'s `per_group_sync` comment logs E-4 hangs under HIP on gfx1201. `STRATA_MTP_BATCH=0` forces E-4 on one card (`prefill.cpp:1309`), so a control needs no code change.

### 2. The NaN pattern points at the stage's persistent state
- `qsa_interval` is 4 (`geometry.hpp:26`). Layer 27 is a full-attention (QSA) layer; layer 33 is a GDN layer. Both carry state across requests: K/V pages and the GDN recurrent state.
- One non-finite value in either makes that layer's output (`bo`) non-finite for every row, even when its inputs are finite. That is the request-2 signature: `bo`/`Dm` fully non-finite, expert inputs clean.
- This fits the carry-over reading above, and narrows it from "a shared buffer" to the A770 stage's GDN states and K/V pages.
- Writers of that state on the A770 stage:
  - the stage's own prompt pass;
  - the verify commit graph;
  - conversation checkpoint restore.
- **Instrument to add:** under `STRATA_DBG_NAN`, a NaN scan of the A770 stage's GDN states and its QSA K/V pages and indexer buffers:
  - at request start;
  - after each `on_chunk` / `mtp.prefill` call;
  - after each commit.

  The first scan that flips names the step that poisons it.

### 3. Checked and ruled out
- **`R_rows` ordering:** `m.cs->wait()` runs before `on_chunk` (`prefill.cpp:4505-4522`), and `mtp.prefill` waits on `cs_` at the end (`sync` defaults to true).
- **Async commit overlapping the drafter:** `set_commit_async(!multi_gpu)` (`generate.cpp:3309`), so it is off on a split.
- **The legacy QSA step buffer (`qsa.dp.cpp` `step_scratch`):** only the legacy direct-launch `kv_append` / `qsa_index` / `topk_512` use it. The engine calls the prompt path's own `kv_append` (`prefill/kernels.dp.cpp`).
- **The drafter's binding:** `mtp.load` and `mtp.bind` use the last stage's session, weights, head and `final_R_all` (`generate.cpp:4150`, `7393`).
- **The drafter's PLE session:** it is CUDA0's (`set_ple_session(&ss)`, `generate.cpp:4151`), but the drafter only calls `prefetch_rows` on it, a host-side page-cache prefetch.

### 4. Wasted VRAM, unrelated to the NaN
- The expert-cache sizing holds back `mtp_bind` on CUDA0 (`generate.cpp:4466`), but on a split the drafter and its draft head live on the last stage.
- That is the "+143 MiB for the draft head" that reshapes the B580 cache in the DBGALLOC maps. It wastes ~143 MiB of B580 cache; it cannot cause the NaN.
- **Fix:** reserve it on the last stage only when `multi_gpu`.

### 5. Is MTP worth finishing here?
- The plan's +47% is Qwen3.8-27B under llama.cpp. On this engine one B580 decoded *slower* with mtp: 6.6 vs 10.4 tok/s, cache hits 40.7% vs 82% (`HANDOFF-WINDOWS-SYCL.md`, run 11).
- Split11 already decodes at 35.5 tok/s, and VicGPT's map calls run ~16:1 prompt to output.
- Measure before more cross-card mtp work.

### Next runs (each N >= 3)
| Run | Config | NaN means | Clean means |
|---|---|---|---|
| A | `start-1c-a770-mtp-check.bat` + `STRATA_MTP_BATCH=0` | the bug is in E-4 on SYCL; two cards are irrelevant | go to B |
| B | `start-2c-mtp-k22-check-sd0.bat` + `STRATA_MTP_BATCH=0` (E-4 on the stage-1 thread, one context) | the thread placement matters | it needs two contexts plus E-4: add the state scan (2) to the failing config |
| C | single-card A770, `--mtp` vs no mtp, same prompt | - | whether mtp pays at all on this engine |
| D | split11 + `--mtp` | - | the measured default has a different layout; k22 may not matter |

### Runs A and B executed 16:53/17:05 (bats `start-1c-a770-mtp-check-batch0.bat`, `start-2c-mtp-k22-check-sd0-batch0.bat`; note both append to the PARENT config's log via the inherited `"log"` field)
- A (single A770, E-4 forced): no storm, no guard hit, decode 19.9-20.7 tok/s. Drafts collapsed after warmup (accepted 1 of 1, suffix 1 of 1) - but the SAME collapse happened in the 08:42 E-9 run of this config, so it is a pre-existing serve-side draft-disabling behavior on this config, not an E-4 symptom. Verdict: E-4 alone on one context does not reproduce.
- B (sd0, E-4 on the stage-1 thread, one context): CLEAN and healthy - drafts 146/249, 182/263, 185/281, 231/310, 235/304 (66-78%), decode 9.4-10.7 tok/s. Verdict per the table: the failure **needs two contexts plus E-4**.
- Caveat: neither batch0 config carried `STRATA_DBG_NAN` (inherited env), so NaN detection there relied only on the serve #606 repeat guard and draft health. Re-running A/B with `STRATA_DBG_NAN=1` would be stricter. UPDATE: both batch0 configs now carry `STRATA_DBG_NAN=1` (17:15), so strict re-runs are ready.

### STATE scan timeline (17:16 run, commit 6671e7f, exe 17:14) - narrows the carrier sharply
- Request 1 (7544): `STATE request start [0-]` and `[22-]`: 0/0. `STATE post on_chunk [22-]` after every chunk (pos 0/2048/4096/6144, each ran mtp.prefill for that chunk): **all 0/0**. First failure anywhere: `verify window at position 7676, row 2` - **mid-DECODE**, request completed (391 tokens, drafts 64/112).
- Request 2 (~1157, single chunk): `STATE request start [0-]` 0/0, `[22-]` 0/0 - **GDN + QSA indexer float state CLEAN after request 1's NaN decode**, then the chunk itself traces `layer 35 ... GU 7273 (partial) Dm 17920 bo 17920 of T 1157` with expert inputs finite.
Conclusion: the reviewer's section-2 candidate (a) GDN recurrent state is CLEAN at every probe point - not the carrier (at least not alone). (b) survives in its non-float parts: the QSA K/V pools (int8, unscannable for NaN), the page table / slot maps (int32), and the checkpoint contents - the poison reaches layer 35 mid-chunk from something the float scans do not cover. And the FIRST poison event in a run is mid-decode (7676): a verify window goes all-NaN head *during* decode with everything scanned clean before it - so the corrupting write is inside the decode loop (verify commit graph / drafter round / checkpoint save).
Note the load-bearing comment found at `verify.cpp:2656`: "set_commit_async: no wait here - the next window runs on the same stream after it, and **the drafter (its own stream) reads only this window's final rows and its own K/V**" - the drafter reading stage-queue-written rows on its private stream is exactly the un-ordered cross-queue edge; whether the rows it reads are fenced is worth a careful read of the mtp record_front sync path vs `g_commit_async`.
### Full event chain (17:25 run, commit-rows probe `7238ab8`) - first bad event is a DECODE window, and later requests self-ignite from stage scratch
Request 1: prefill completely clean (STATE all zero, prompt-end finite). First poison anywhere = the window at pos 7757: its head all-NaN AND its commit rows non-finite (20480/30720) at the SAME position - the window forward produced garbage rows from everything scanned-clean upstream. Every commit after it is 100% non-finite (the storm feeds itself).
Request 2 (1244 prompt): `STATE request start` 0/0 on BOTH stages (clean again!), `layer 24 ... bo 209920 non-finite` traces MID-chunk, and `STATE post on_chunk pos 0: GDN 13871104, QSA indexer 235008 non-finite` - the persistent state is poisoned WITHIN chunk 0, by the request's own forward, from a clean start. Storm again (`126 of 126` fake acceptance at 32.8 tok/s - pollution ledger).
Reading: each dying request is self-ignited by garbage LEFT IN STAGE SCRATCH by the previous request's storm - the scan covers GDN *state* and indexer floats, NOT the reusable scratch carves (`ss.gdn` buffers `session.cpp:100`, `ss.qsa_bufs` `session.cpp:111`, the moe carves `session.cpp:122`, verify's `R_/ple_/emb_...` arena `verify.cpp:573`): those persist across requests and are read by later forwards. The FIRST storm (7757) has no upstream poison in any probed buffer - it is generated by the window forward itself, mid-decode, on the k22 split with the drafter running. Given the `verify.cpp:2656` comment (drafter reads the window's final rows on its own stream, "no wait here") and the A/B result (needs two contexts plus E-4): the cross-queue/cross-context handshake between stage row writes and drafter reads is the prime root-event suspect; a half-read draft row enters the NEXT window's forward as garbage input rows (window input rows are NOT yet scanned).
Next probes (order): (1) scan the WINDOW INPUT rows at capture (window rows / window_R in verify) - catches a corrupt draft row at entry and names the first bad draft round; (2) scan the scratch carves (ss.gdn buffers, ss.qsa_bufs, verify arena) at request start - they should be readable-but-not-initialized: nonzero garbage is the carrier between requests; (3) audit the ordering of the drafter's read of window final rows vs `set_commit_async` + window forward enqueue order on the stage queue (event waits across contexts: do they actually order?).

### Benchmark pollution ledger (the `!!!!!` bug taints speed numbers)
INVALID (storm-contaminated, decode was garbage-token spam): k22 mtp 29.7 tok/s (09:10 graph run, window NaN at 7578 mid-request), 19.3/22.4 tok/s (09:03 eager run), and every k22 mtp run after a window NaN (1047/1325/7578/7630/7676 family).
CLEAN baselines for the mtp-value question: nomtp B580-alone ~10.4 tok/s decode (KB control); mtp sd0 9.4-10.7 (17:05, drafts 66-78%); a770-alone nomtp 14.0/20.7 (KB control C); nomtp k22 64k daily (`strata-iq2_xs-2card-k22-ctx64k.log`): decode 18.0-24.4 tok/s with spec-4 suffix drafts already accepting 86% (276/321, 104/122). mtp has never beaten a clean nomtp baseline on any config that works; the only apparent gain (29.7 on k22) came from the broken one. To justify finishing cross-card mtp, a FIXED k22 mtp must beat 18-24 tok/s clean, ideally while raising the 86% suffix-draft acceptance the nomtp spec-4 path already gets.

## Review 2 (2026-10-16, after runs A/B and the state / commit-row scans; code reading only)

**Corrections to review 1.** Runs A and B rule out "E-4 on one context". The failure needs two contexts plus mtp. The first bad event is a decode window (positions 7578-7757), not the drafter's prompt pass. Section 1 of review 1 is answered.

### The instruments (`6671e7f`, `7238ab8`) are correct
- Both scans copy through the current card's default queue, but only after that card's work has finished:
  - in the prompt pass, after `m.cs->wait()`;
  - in `commit()`, after `run()` has synced (commit is synchronous on a split).
- **Gap: the hand-off rows between the stages are not scanned.** A window whose scanned inputs are clean can still produce NaN at the A770 stage if its *input* rows came in bad. Add under `STRATA_DBG_NAN`:
  - stage 0's `hand_out_` at the end of its window (B580);
  - stage 1's `hand_in_` at the start of its window, after `forward_handoff`'s host copy (A770).

  A non-finite `hand_in_` means the garbage is made on the B580 stage, and the A770 is only where it first shows.

### New prime suspect: the B580's device-wait bound under `--mtp`
`generate.cpp:2364`:
```
strata::spin_drafts_need_long_bound(o.spec >= 2 && o.mtp.empty());   // spin_max: drafting without MTP waits long
```
With it, `spin_max` (`sycl_doorbell.hpp:146-148`) gives a Battlemage card without a sysfs driver (Windows):
- **without `--mtp`** (the clean nomtp spec-4 configs): 2,000,000 reads;
- **with `--mtp`**: 20,000 reads.

The 20,000 bound was measured (#1473) for B-series windows that only wait in a failure: all experts resident, or the device plan. At k22 the B580 stage is not all-resident; its cache holds about half of its layers' experts. So its windows really wait for the CPU pool's experts. A wait that expires lets the window run on without those experts' outputs. `docs/INTEL.md` (~line 708) describes that as the cause of `!!!!` (token 0) and crashes. Garbage leaving the B580 stage first shows up as NaN at the A770 stage, where the probes are.

| Config | B580 bound | B580 stage all-resident? | Result | Fits? |
|---|---|---|---|---|
| k22 nomtp (spec-4) | 2,000,000 | no | clean | yes |
| k22 + mtp | 20,000 | no | NaN | yes |
| A770 alone + mtp | (Alchemist: 2,000,000) | - | clean | yes |
| split11 + mtp (run D, not run) | 20,000 | **yes** (5,632 of 5,632) | predicted clean | - |
| `--split-device 0` + mtp | 20,000 | no (hit 40.7%) | clean | **not explained** |

For the last row: a guess, not verified, is that on one card the drafter and the windows serialize, so the host serves the B580's waits in time.

**Test (cheap, config only):** the failing config (`start-2c-mtp-k22-check-graph.bat`) plus `STRATA_SPIN_MAX=2000000`, N >= 3.
- **Clean:** the fix is to keep the long bound on any Battlemage stage that is not all-resident when `--mtp` is on. The 20,000 bound stays where it was measured: all-resident or device-plan windows.
- **Same NaN:** the bound is not it. Go on with the hand-off scans above.

### Why later requests start poisoned
- The GDN and indexer float scans are clean at request start, yet request 2 is poisoned within chunk 0.
- An unscanned carrier: the fp16 K/V scales (`k_scale` / `v_scale`, and `v_q4` scales under k8v4) of the QSA pools. A storm can write non-finite scales into cells, and a new request reuses those page slots.
- That reaches a fresh request only if an attention kernel reads past `n_kv` (whole pages or tiles) and masks by multiplication instead of selection (NaN x 0 = NaN).
- **Instrument:** scan the scales as half floats at request start.
- This is the spread to later requests, not the root event. Restart between gate runs, as already done.

### Checked by reading, nothing wrong found
- **`verify.cpp:2656` ("no wait here", drafter on its own stream)** only applies with `g_commit_async`, which is off on a split (`generate.cpp:3309`). The commit waits.
- **The drafter's read of the window's rows** (`MtpDrafter::stage_source_R`, `mtp.cpp:1821`) runs after `run()` has synced. Its foreign-owner staging only runs for a source on another card; in decode `src_R_` is the same card's `window_R_`.
- **The drafter shares no scratch with the verifier's head.** It uses its own `xq_` / `head_logits_` and reads only `head_->weights()` (`mtp.cpp:1159`). `NativeHead::scratch_` is used by `NativeHead::run` alone.
- **`pl_launch` / `batch_launch`** (no `forward_handoff`) are only reached under `--pipeline-windows` / `--batch-groups`, which are refused across contexts. mtp decode goes through `ver.run`, which does the hand-off copy.

### Order
1. k22 + mtp + `STRATA_SPIN_MAX=2000000` (N >= 3).
2. Run D: split11 + mtp. That is the measured default, and its B580 stage is all-resident. It also answers the mtp-value question against split11's 35.5 tok/s clean decode.
3. Only if 1 still fails: the hand-off row scans, then the K/V scale scan.

## Response to Review 2 (engine side, same evening)

**Their order item 1 is already negative - the long bound has been on in every k22 run today.** All five k22 configs (every failing mtp one AND the clean nomtp control AND sd0) carry `STRATA_SPIN_MAX=2000000` in their env block (inherited from the daily-config lineage; verified by reading the config JSONs), and `sycl_doorbell.hpp:142` applies the env override over the per-device auto choice unconditionally. The 09:03-17:25 k22 mtp runs already ran the B580 windows at 2,000,000 reads and still NaN'd; the nomtp control at the same bound stays clean. So the `spin_drafts_need_long_bound(o.spec >= 2 && o.mtp.empty())` selection (`generate.cpp:2364`) is overridden on this machine everywhere it was tested. Spin-bound hypothesis: **tested, refuted** - unless the wait in question is some doorbell spin not routed through `spin_max(queue)`. Their matrix's remaining tension (sd0 clean with the same bound and a not-all-resident stage) is the same fact seen from the other side.

**Hand-off scans built** (DBG_NAN-gated, `verify.cpp`): `hand_out rows` scan at the earlier stage's window end (before `forward_handoff`, volatile reads of the mapped buffer) and `hand_in rows` scan at the later stage right after its three `copy_from_mapped` calls (line 859 area). Their reading stands: `hand_in` non-finite with clean `hand_out` places the producer in the hand-off copy itself; `hand_out` non-finite first names the B580 stage as producer.

**Caveat on my own probes:** the `fp16 inputs` scan samples the dq/activation scratch **after** the chunk, but those carves are overwritten per type-group during it - "expert inputs finite at probe time" is not "finite at gemm time". If the hand-off scans stay clean, next instruments: their K/V fp16 scale scan ("Why later requests start poisoned"), plus re-sampling expert inputs INSIDE the per-group loop before each gemm launch.

**Run D:** unexecuted; agreed it is the most informative remaining run (all-resident B580 stage, and it prices mtp against split11's 35.5 tok/s clean nomtp decode). Needs a bat if the next gate run does not close the NaN.
