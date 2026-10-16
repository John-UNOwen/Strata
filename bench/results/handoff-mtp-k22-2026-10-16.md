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

## Suspect areas (ranked)
1. >4 GiB layout shift caused by the drafter's 839 MiB on the A770: with mtp loaded, other A770-stage buffers/offsets cross 4 GiB and kernels/copies that per-op staging does not cover misread them (review-leads' exact standing point; `STRATA_SYCL_GEMM_NO_STAGE=1` at 64k NaNs by design, proving the underlying oneMKL bug is live: oneMKL 2026.1 reads the last partial tile of an operand past 4 GiB from the wrong place). Real fix: keep the stage's working buffers under 4 GiB (sub-arena allocation per stage), not per-op staging.
2. The per-queue staging scratch itself adds new A770 device allocations (one scratch per staging queue); on the nearly-full A770 that may push arenas past physical/limit boundaries (relaxed allocation limits on). If the scratch theory matters, size/report it.
3. `m.bo`/`m.Dm` come from session scratch arenas (`prefill.cpp:1169` carve); with mtp the session/verifier borrow chain (drafter borrows the primary QSA RoPE table from the B580-context `SessionLoopScratch`, `session.cpp:107`) is another cross-context surface the CTXCHECK has not yet looked at.

## Repro and instruments
- Fails: `start-2c-mtp-k22-check-graph.bat` (and the eager variant `start-2c-mtp-k22-check.bat` storms too). Warmup dies inside ~60 s; nothing to wait for.
- Clean controls: `start-2c-k22-check-graph-nomtp.bat` (32k no-mtp), `start-2c-mtp-k22-check-sd0.bat` (mtp, single device), `start-1c-a770-mtp-check.bat` (single card, eager). The 64k controls (start-2c-k22-ctrl64k*.bat + configs/logs) and all older experiment files live in `run-archive\` (git-excluded) since 16:40.
- New instrument 16:35 (commit 120a404): `STRATA_DBG_ALLOC=1` prints a `DBGALLOC` dump of every tracked device allocation (base + size) at prompt end; both 32k check configs have it on. Compare the maps in the clean vs NaN logs to see what crosses 4 GiB and what mtp adds untracked.
- `STRATA_DBG_NAN=1` window/GU probes; `STRATA_MTP_CHECK_CTX=1` CTXCHECK lines (embed table, R_rows, window_R, forward phases A..G); probe matrix `C:\strata-deps\cross_usm_probe.exe` (modes `shared copy perdevice dev dev9b dev10`).
- Always re-run gates N>=3 times: a single clean run on this split proves nothing (proven).
