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

## Suspect areas (updated 16:50 with DBGALLOC map evidence - see Repro)
ELIMINATED by the clean-vs-NaN allocation maps (16:38/16:39 runs):
- (old 1) >4 GiB layout shift from the drafter: NO. The A770 expert arena is the same single allocation, same base `FFFFD40173C00000`, 7273.88 (nomtp) vs 7275.32 MiB (mtp). No GEMM was staged in ANY 32k run (clean or NaN): at 32k the engine finds no oneMKL operand past 4 GiB; the >4 GiB bug is only live at 64k. mtp's only layout effect is on the B580 side: the expert cache splits `7364 MiB` -> `675 + 6518 MiB` (the +143 MiB draft-head reservation reshapes it).
- (old 3) drafter-borrowed RoPE table: NO. With native rope enabled (this model), `cos_tab`/`sin_tab` are never read - all live call sites use `native_rope_apply` (analytic angles); `rope_neox_apply` (the only reader) is off. Verified all callers.
- (old 2) extra per-queue staging scratch on the full A770: no evidence; maps identical.
STANDING suspect - cross-request carry-over (16:39 run): request 1 completed HEALTHY with 34 drafter rounds; request 2 is ordinary main-model prefill (`layer 33, mmq 0 ... of T 1047`) that immediately produces `bo/Dm` fully non-finite with finite expert inputs, then window NaN at position 1047. The poison persists in a shared persistent buffer that drafter rounds touched. Candidates: the shared expert cache (draft-head experts hold real slots, log line `(+143 MiB for the draft head)`), or session-arena carve reuse (`moe_buffers_init` GU/Dm/bo live in the per-stage session arena) accessed by drafter-private queues without ordering against the stage queues. A NaN scan of session-arena carves at REQUEST start (before any kernel of that request runs) would show carry-over directly - worth adding next to the `STRATA_DBG_NAN` probe family.
Failure position is random run to run (1047 / 7578 / 7630 / 7587), so the corruption point moves; only the "mtp on k22 split" precondition is stable.

## Repro and instruments
- Fails: `start-2c-mtp-k22-check-graph.bat` (and the eager variant `start-2c-mtp-k22-check.bat` storms too). Warmup dies inside ~60 s; nothing to wait for.
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
- The expert-cache sizing holds back `mtp_bind` on CUDA0 (`generate.cpp:4468`), but on a split the drafter and its draft head live on the last stage.
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
