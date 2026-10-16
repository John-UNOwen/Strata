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

### Benchmark pollution ledger (the `!!!!!` bug taints speed numbers)
INVALID (storm-contaminated, decode was garbage-token spam): k22 mtp 29.7 tok/s (09:10 graph run, window NaN at 7578 mid-request), 19.3/22.4 tok/s (09:03 eager run), and every k22 mtp run after a window NaN (1047/1325/7578/7630 family).
CLEAN baselines for the mtp-value question: nomtp B580-alone ~10.4 tok/s decode (KB control); mtp sd0 9.4-10.7 (17:05, drafts 66-78%); a770-alone nomtp 14.0/20.7 (KB control C); nomtp k22 64k daily (`strata-iq2_xs-2card-k22-ctx64k.log`): decode 18.0-24.4 tok/s with spec-4 suffix drafts already accepting 86% (276/321, 104/122). mtp has never beaten a clean nomtp baseline on any config that works; the only apparent gain (29.7 on k22) came from the broken one. To justify finishing cross-card mtp, a FIXED k22 mtp must beat 18-24 tok/s clean, ideally while raising the 86% suffix-draft acceptance the nomtp spec-4 path already gets.
