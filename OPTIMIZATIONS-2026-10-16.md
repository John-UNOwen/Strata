# Strata Windows SYCL - optimization and fix backlog (2026-10-16)

Measured on this PC: B580 `level_zero:0` + A770 `level_zero:1`, engine 0.1.44.1, JIT tree `build-sycl-jit`, branch `main`
(bench tools at `01cd9e8`; the certified numbers below are gate-checked, 2026-10-16 evening).
Evidence:
- `HANDOFF-WINDOWS-SYCL.md` sections 13-17;
- `bench/results/handoff-mtp-k22-2026-10-16.md`;
- the split11 benchmark in commit `2ccc945`;
- `docs/INTEL_WINDOWS.md`;
- KB `projects/strata-windows-sycl/`.

## Where things stand (updated 18:40 by the split search; see KB split11-nan-story Update 18:40)

**The reliable baseline is k22, not split11.** The evening ladder with real traffic (text and images) showed the fault
tracks how many layers the A770 side holds:

| A770 layers | Config | Text | With images |
|---|---|---|---|
| 26 | k22 | clean | clean 2/2 (even while staging GEMMs past 4 GiB) |
| 32 | split16 | clean 3/3 | NaN (also with nothing staged past 4 GiB) |
| 37 | split11 | NaN at 64K and at 32K | - |

The split11 bench table below (621/851/35.5, commit `2ccc945`) is kept as the historical ceiling of this PC, but
split11/split12/split16 are the **unstable class** until the >26-layer frontier is understood; measuring optimization
arms on them mixes speed with a moving fault. **k22 daily numbers** (`start-2card-k22-ctx64k.bat`, `STRATA_VRAM_FREE_MIB=0,15000`
now): prompt 260-320 tok/s, decode 18-25, suffix drafts up to ~86% accepted. k22 also serves images (CPU encoder,
280-297 tok/s map prompts): at k22 the bigger B580 share no longer buys prompt speed, so the frontier costs little today.

Against VicGPT's bar and the old llama.cpp Vulkan model (two-card numbers now gate-certified by k22ref):
- **Prompt:** k22 377 (4K) / 628 (16K) tok/s certified; its map-sized 8-18K warm reads run 200-330. llama.cpp ran
  380-420 at 16K, so k22 is ahead but the split11-class 851 remains the number the frontier hunt could restore.
- **Decode:** k22 22.2 tok/s certified (22.9 in the best session) is about 1.4x llama.cpp's ~16.
- **A 10k-token map chunk** in a warm session reads in about 30-50 s at k22 (it was 12-16 s at split11); a cold 4K or
  16K prompt runs at the certified 377 / 628.

**Arms for the k22 baseline**: `python bench/arms/make_arms_k22.py` writes the configs; `BENCH-ARMS.bat` (repo root)
then runs the whole ladder automatically - each arm in **3 separate fresh server sessions** (ports 8191+; 3 is the
default, `set BENCH_REPEATS=5` to change), the suite with the pollution gate in every session, and a final
session-level summary (`tools/arms_summary.py`): aggregates over clean sessions only, spread across sessions
(consistency verdict per arm), deltas vs k22ref; `--newest 3` restricts to the latest round when re-runs pile onto
old sessions, `--sessions` lists them all. One arm alone: `bench/arms/run-arm.bat k22mirror6000`.
`k22ref`, `k22mirror6000`, `k22mirror10000`, `k22prefill16k`, `k22own4096`, `k22own4096mirror`, `k22ctx32k`, `k22ctx32kmirror`.
The old split11-based arms (`make_arms.py`) stay for the frontier hunt itself.

Arm ledger after the first gated rounds (details in the two sections below):
- `k22ref`: **certified baseline**, decode 22.2 (5% over sessions), 4K 377, 16K 628, 3 of 4 sessions clean.
- `k22mirror6000` / `k22mirror10000`: **dropped**. Neutral within 4%, and by construction they only exercise the part of
  the mirror that cannot matter at k22 (speed item 1).
- `k22own4096`: **settled**, 4 sessions, decode 16.1 (7%), 4K 494.5 (4%), 16K 698.1 (6%): -27% decode for +31% / +11%
  prompt. The map-reading profile if VicGPT cares more about 8-18k reads than answer speed.
- `k22own4096mirror`: 17.9 / 500 / 706 in one session - pays less decode than plain own4096 for the same prompt win.
  The one arm still worth 3 sessions.
- `k22prefill16k`, `k22ctx32k`, `k22ctx32kmirror`: **void** (storms in the first two chat runs). Revisit only after the
  frontier moves (robustness item 0), not before.

### First gated ladder (fresh data, exe 17:45 build, run 2026-10-16 19:29-20:25)

Every number below is new, gate-checked, one fresh server per arm:

| Arm | Decode tok/s | 4K | 16K | Polluted replies | Verdict |
|---|---:|---:|---:|---:|---|
| **k22ref** | **22.9** | **377** | **618** | 0 | the valid baseline |
| k22mirror6000 | 22.7 | 361 | 607 | 0 | mirror is neutral (-1 to -4%) |
| k22mirror10000 | 22.1 | 365 | 610 | 0 | neutral |
| k22prefill16k | - | - | - | 7 | void (storms from chat run 2; 4 draft collapses) |
| k22own4096 | 16.3 | 498 | 702 | 0 | real trade: +32%/+14% prompt, -29% decode |
| k22own4096mirror | 17.9 | 500 | 706 | 0 | same prompt win, -22% decode |
| k22ctx32k | - | - | - | 32 | void (storms from chat run 1; its +66% "win" is garbage) |
| k22ctx32kmirror | - | - | - | 22 | void |

What this settles:
1. **k22ref 22.9 / 377 / 618 is the certified baseline.** It nearly repeats the suspect-era k22 row (21.7 / 367 / 616),
   so the k22 arms of the morning era were probably clean even though the split11-era rows were not.
2. **The pinned-RAM mirror is dead *as configured*** at 64K: 6000 and 10000 MiB both land within noise, because
   `mirror_end = split_at[0]` can only mirror the B580 stage's misses and at k22 the expensive ones are the A770
   stage's 5656 host-served experts. This does not refute the mirror lever; it refutes mirroring the wrong stage
   (speed item 1).
3. **own4096 (no borrow, 4096 chunks) is the one measured win for prompt-heavy use**, settled over 4 sessions at
   16.1 / 494.5 / 698.1: +31% at 4K, +11% at 16K, paid for with -27% decode. A VicGPT decision: map reads faster,
   answers slower.
4. **k22 is not shape-invariant clean.** Storms appeared inside 5 of 14 k22 sessions so far: 32K context 1/1, 16384
   chunks 1/1, and 64K auto 1/4. The 64K auto shape is the best one known and it is still 3 of 4, not immune. The
   ">26 layers" frontier story needs a revision: buffer/pool shape, not only the layer count, decides whether the
   A770 stage manufactures NaN windows.
5. **The gate works**: ctx32k's apparent +66%/+58% speed would have been logged as the biggest win of the day by the
   old bench. It was 32 `!`x100 replies.

**Open measurements (next round).** The repeats this originally asked for are done: storm rates are in, the mirror arms
are dead, own4096 is settled. What is left:
1. `BENCH-ARMS.bat k22own4096mirror` for 3 sessions - the last arm with an open verdict.
2. One DBG-capture run aimed at the **agent turn**, not the short chats: a `STRATA_DBG_NAN=1` session whose 11498-token
   agent turn (8577 reused) is the one schedule currently known to ignite (robustness item 0). Accept the masking
   caveat: the scans can hide the fault, so N>=3 and a plain-launcher rate next to it.
3. No more 32K or 16384-chunk arm runs until the frontier moves. Those shapes are 0 for 1 and re-measuring a known
   storm tells us nothing; the chunk-size work in speed item 3 is gated on robustness item 0 first.

### Baseline round (4 k22ref sessions, 19:29-21:06) - the certified baseline and the storm rate

`BENCH-ARMS.bat k22ref` x3 (plus the original 19:29 session): **3 clean / 4 sessions** at the 64K auto-chunk shape,
aggregating to **decode 22.2 (5% spread), 4K 377 (2%), 16K 628 (4%) - consistent**. This is the new baseline.

- **k22 64K auto is ~75% clean, not immune**: the 4th session (21:06) stormed at the last short-chat request and the
  new stop cut the session there before the prompt tests. Storm-rate so far: 32K context 1/1, 16384-chunks 1/1
  (both lit inside the first two chat runs), 64K auto 1/4 (lit late in chat).
- **Short chat is the usual fuse, not the only one**: `own4096`'s 3-session round (21:32-21:45) added a storm in the
  **agent phase** (run 1 turn 4, the 11498-token turn that reuses 8577 tokens). That session's chat and prompt numbers
  stayed valid (the fault does not reach backwards); `tools/arms_summary.py` now voids a session from the polluted
  suite onwards instead of dropping the whole session.
- **The engine log carries the same signature, independently of the reply text**: the junk replies accept every
  suffix draft (`drafts accepted 29 of 29`, `48 of 48`, `30 of 30`) and run 1.3-2x the session's normal pace
  (30.1 tok/s where the session ran 17.9; 36 tok/s where it ran 15.7). `tools/arms_logscan.py` reads the
  `strata serve: prompt ...` lines and flags the same 5 storm sessions the client gate flagged, with no false hit in
  the 9 clean ones. This is the second opinion to use on any log whose results.json predates the gate.
- Deltas against the certified baseline (clean suites): mirrors neutral (within 4%); `own4096` is settled after
  4 sessions (one polluted in the agent phase, chat/prompts valid): **decode 16.1 (7% spread), 4K 494.5 (4%),
  16K 698.1 (6%) - consistent**, i.e. -27% decode for +31% 4K and +11% 16K.

`tools/strata_bench.py` now carries a client-side **pollution gate**: every reply (warmup included) is scored for
storm signatures (long single-character bursts) and degenerate loops; any hit voids the session's numbers, STOPS the
suite at that point (exit code 3; the rest of the tests are marked skipped), prints a POLLUTED banner and lists the
flagged replies in the report; `BENCH-ARMS.bat` aborts the whole ladder on exit 3. It needs no engine debug flags -
whose synchronous scans can mask the fault (2/2 runs). Re-baseline every claim on the fixed engine with this gate:
`run-arm.bat k22ref` first.

Historical baseline table (split11 era, `2ccc945`; pollution-unverified, kept only for shape):

| Config | Prompt 4K | Prompt 16K | Decode |
|---|---|---|---|
| k22 (old reference) | 367 | 616 | 21.7 |
| split10 | 453 | 643 | 33.3 |
| **split11** | **621** | **851** | **35.5** |
| split12 | 628 | 860 | 35.7 (tie, less VRAM) |
| split13 | 343 | 655 | collapsed on turn 5 (do not use) |
| split11 + `STRATA_SYNC_PREAD=1` (old synchronous reads) | 377 | 554 | 34.1 |

**These split11-era numbers remain pollution-unverified** and every row here is shape, not evidence. The morning
baselines were benched 06:2x-07:42 on the engine *before* the scratch-race fix (`9953a5b`, ~09:45), i.e. the version
that provably NaN'd split11-class runs in the same hour (the 07:4x noborrow ladder). One poisoned window makes every
later request in the session fast garbage while staying internally consistent, and `strata_bench.py` stored no reply
text and ran no quality check then, so a storm during the discarded warmup would have tainted every number invisibly.
Tight ranges (620.9 603-622; 37.0 36.0-37.7) are not evidence of cleanliness - a poisoned session is consistently fast.

Against VicGPT's bar and the old llama.cpp Vulkan model:
- **Prompt:** 851 tok/s at 16K beats llama.cpp's 380-420 tok/s.
- **Decode:** 35.5 tok/s is about 2x llama.cpp's ~16.
- **A 10k-token map chunk** reads in about 12-16 s at 64K.

**Open: the A770's arena past 4 GiB at 64K.** The split11 NaN story (KB `references/split11-nan-story-2026-10-16.md`) has two faults:
- the GEMM staging scratch shared across queues: fixed in `9953a5b`;
- the A770's layout past 4 GiB at 64K: open.

`start-strata-split11-ctx32k.bat` runs the same split at 32K to stay under it. VicGPT needs <= 32K. But at 32K the split does not borrow (`generate.cpp:2286`), so prompts read in 2048-token chunks and are slower than the 64K numbers above.

**Settled: k22 is the two-card default.** `docs/INTEL_WINDOWS.md` and the two split11 starter headers used to disagree
about which launcher is the default; the headers now call themselves probes and name `start-2card-k22-ctx64k.bat`, the
certified baseline. The 32K split11 probe still has no gate-checked prompt numbers, and now that k22 storms at 32K
(robustness item 0) it should not get any until the frontier moves.

## Settled (no further work)

- **Split point:** 10-11 makes the B580 stage all-resident; measured above. 11 and 12 tie.
- **Overlapped `fd_pread`:** in `main` (from `20c693c`). It is 2.3x of split11's prompt speed against the old synchronous reads, and neutral at k22.
- **Token-0 (`!!!!`) storms:** these were not greedy loops. They were the per-thread GEMM staging scratch shared by all queues of a thread; it is per queue since `9953a5b`. The 32K no-mtp control died twice before the fix and ran 3/3 clean after it. The `sampling` block in the configs no longer hides anything.
- **The B580's device-wait bound:** every config carries `STRATA_SPIN_MAX=2000000`, which overrides the per-device choice (`sycl_doorbell.hpp:142`).
- **MTP: parked.** KB decision `park-cross-card-mtp-unless-it-beats-18-24.md`:
  - it never beat a clean no-mtp baseline on any working config;
  - the built-in drafts (`--spec 4`) accept 29-86%;
  - the two-card NaN hunt is in `bench/results/handoff-mtp-k22-2026-10-16.md`;
  - speed claims from the broken k22 mtp runs are void (the pollution ledger there).

## How to measure

- Use `BENCH-ARMS.bat` (3 fresh server sessions per arm, one port each) or `BENCH.bat` for a one-off; the suite gates
  every reply and stops the session at the first storm.
- Include a ~10k-token prompt, since VicGPT's map step sends 8-18k.
- Record prompt tok/s, decode tok/s, the hit rate, and each card's dedicated and shared GPU memory peak.
- Compare against split11 at the same context length (32K or 64K, not mixed).
- Read the engine log too: a storm shows up as 100% suffix-draft acceptance plus a decode speed spike
  (`python tools/arms_logscan.py` prints those rows per session). It catches storms in logs whose results.json
  predates the gate, and it corroborates a gate hit without trusting the reply text.

## Speed (no MTP)

1. **Mirror the A770 stage's missed experts in pinned RAM.** The largest remaining lever, and still untested.
   - **The arms did not test it.** `k22mirror6000` / `k22mirror10000` came back neutral because `mirror_end =
     split_at[0]` (below): at k22 the mirror can only hold the **B580 stage's** misses, and the expensive ones - the
     A770 stage's 5656 experts served by the host - never reach it. Neutral mirror arms are the expected result, not a
     refutation.
   - **Measured cache coverage to size it** (k22, 64K auto, own4096 session log): B580 4071 slots of its 13312 profiled
     pairs (31%), A770 5503 slots of 13312 (41%), decode hit rate 78-94%, agent-session reuse 69%. So both stages miss.
   - **Today:**
     - On a split the expert mirror covers only stage 0's layers (`mirror_end = split_at[0]`, `generate.cpp:4941`), because the later stages' caches do not exist yet when it is built (#1054, #1440).
     - At split11 the B580 stage has no misses, so no mirror is built at all.
     - Every miss of layers 11-47 is read from the GGUF: the prompt path streams it with `fd_pread`, and the CPU pool's decode reads go through `GgufExpertSource::blob()`, which checks the mirror first, then the ring + `fd_pread`.
     - Windows' file cache absorbs part of this, but the 2.3x from overlapped reads shows how much the file path costs.
   - **Windows bug on the way:** the default mirror size reads `MemAvailable` from `/proc/meminfo` (`generate.cpp:4974`, still open as of `01cd9e8`). On Windows there is no such file, so the cap is 0 and no mirror is built unless `STRATA_MIRROR_MIB` is set. Use `GlobalMemoryStatusEx` (`ullAvailPhys`) there, the helper already exists for the RAM figure (`generate.cpp:1147`).
   - **Change:**
     - After the stage caches exist, mirror the later stage's misses, hottest first by the profile, through that stage's own queue (`OnDevice(stage dev)`).
     - `GgufExpertSource::mirror` replaces its chunks today. It needs to append segments and remember each chunk's queue, so a chunk is freed through its own context.
     - A mirror read through a copy is fine across contexts (T5). Kernels must only read it in place on its owner's card (item 2).
   - **RAM budget:**
     - Layers 11-47 hold 18,944 experts at ~1.38 MiB each. Most are not in the A770's VRAM, so a full mirror (~19 GiB or more) does not fit next to the OS and the engine in 32 GB.
     - Cap it at ~10-12 GiB, hottest first. Watch Windows RAM, one of this PC's stop signals.
     - `start-strata-split16.bat` already mirrors 7,000 MiB, but of the B580's misses (B580 layers 0-15).
   - **Check first (config only):** `STRATA_MIRROR_MIB=10000` at split11 should change nothing. The log should show no "mirrored in pinned host memory" line, which confirms that the A770's misses are never mirrored today.
2. **The A770 stage reads its missed experts over PCIe again (decode only).** This needs item 1.
   - Today `stage_reads_host` sets the stage's `pcie_num` to 0 when it is in another context than CUDA0, so the CPU computes all of the A770's misses.
   - Once its mirror segment is allocated through the A770's queue, change the gate to compare the stage's device with the mirror segment's owner instead of CUDA0. Then the copy kernels read it in place on the card that owns it.
   - The A770 is on PCIe 4.0 x16. Fresh ladder probes read 12.0-16.0 GB/s (`pcie_frac` 0.33-0.44, and it moves between
     sessions), against 17-22 GB/s in the earlier probe; budget with the lower figure.
3. **Prompt chunk size.** Measured so far: `--prefill 4096` with own buffers is the only chunk lever that paid
   (`k22own4096`, settled at +31% 4K / +11% 16K for -27% decode); `k22prefill16k` (the `auto:16384` case) is **void**,
   it stormed in chat run 2 and never produced a number, and both 32K arms are void too. So the 16K-chunk question is
   still open but it is now a frontier question, not a tuning one: any chunk large enough to help is also a shape that
   lit the fault at 32K. Do robustness item 0 first.
   - **64K:** the split borrows, and `--prefill auto` caps at 8192-token chunks (`prefill_auto_max`). A 16K prompt is then two passes over the experts.
     - Arm: `--prefill auto:16384`, measured on the 16K prompt.
     - Evidence that fewer passes pay: 2048 -> 4096-token chunks gave +49% at k22 in runs 7-8 (pollution-unverified,
       predates the gate); the gate-checked version of that lever is `k22own4096`, +31% at 4K and +11% at 16K.
     - The old gate (the A770 past-4-GiB issue) was demoted by the split search: k22 stages GEMM operands past 4 GiB and stays clean. Arm `k22prefill16k` measures it directly at k22.
   - **32K:** no borrowing, so auto is forced to 2048-token chunks (`generate.cpp:2305-2307`). Arms:
     - `--prefill 4096` with own buffers: ~160 + 4096 x 680/1024 MiB + the ring per card (~3.2 GiB), taken from the expert caches for the whole session;
     - `--prefill-borrow --prefill auto`: lends during prompts only, but puts the A770's borrowed operands back into the past-4-GiB territory 32K was chosen to avoid.
     - Both 32K arms were run and both are void (storms in the first two chat runs), so nothing here is measured yet.
4. **An expert profile from VicGPT's own traffic.** Config only.
   - Measured today: decode hit rates run 78-94% per request and the agent-session reuse is 69%, not the 95-99% an
     ideal profile could reach. A traffic-shaped profile is the cheapest way to close part of that gap.
   - Save one with `expert_profile_save` during Discord-summary requests, then start with `--expert-profile` pointing at it.
   - Fewer A770 misses means less of everything in items 1-3.
5. **Split 12 instead of 11, if the VRAM goes somewhere useful.** It ties split11 on speed and uses less VRAM. Worth it only if that VRAM ends up in the A770's cache, or makes the 32K prompt-chunk arm (item 3) fit.

**Low priority, kept for the record:**
- **The prompt helper across contexts:** both stages read their prompts alone. The helper binds only for a one-chunk
  prompt (`bind_stage_helper`), so it matters for 8-18k prompts only if item 3 makes a chunk as large as the prompt.
  On Windows the in-place version cannot exist: the two cards are in different contexts and a kernel may not read the
  other's buffer (`strata::cross_context`), so a helper would have to stage rows through host memory, paying a PCIe
  round trip per chunk. Not worth building until item 3 has a number.

**Dropped:**
- **"Overlap chunks across stages":** a multi-chunk prompt already runs stage 1 on chunk N while stage 0 reads chunk N+1 (`std::async`, `prefill.cpp:4491`).
- **"Hand-off copy on the reading card":** a few hundred KB per window, not worth the change.

## Robustness

0. **The >26-layer A770 frontier (the gate for restoring split11-class speed).** Facts on the table: clean at 26 A770
   layers (text and images, even staging past 4 GiB), NaN at 32 (images) and 37 (text); per-op staging neither
   necessary nor sufficient; once a session serves a non-finite window every later request stays garbage until
   restart; the `STRATA_DBG_NAN` sync scans masked the fault in 2/2 runs, so gate conclusions on plain launchers,
   N>=3. Evidence: KB `references/split11-nan-story-2026-10-16.md` (Update 18:40) and the probe timelines in
   `bench/results/handoff-mtp-k22-2026-10-16.md`.
   - **The frontier is per-shape and has a rate, not a yes/no.** Measured storm rate so far: 32K context 1/1 void,
     16384-token chunks 1/1, 64K auto 1/4 (75% clean), and the k22 *decode* shape itself stays clean at 22.2 tok/s
     over 3 clean sessions. So "k22 is clean" means "clean at this shape, usually".
   - **Two fuses, both inside the model forward:** short chats (43-51 tokens, 100-token decode, mid-decode verify
     window) and, once, a long agent turn (11498 tokens, 8577 reused, `drafts accepted 29 of 29` at 30.1 tok/s). Both
     are verify-window shapes, which is why the fault also leaves a fingerprint in the log (see How to measure).
   - **A session stops at the first polluted reply** (suite exit 3, ladder aborts) and `arms_summary.py` voids only
     the suites from that point on, so data measured before ignition is still used. Do not "measure through" a storm.
   - **Next probe target:** the agent turn, not the chat. It is the one ignition we can schedule on demand.
1. **Teardown crash on the "no room" refusal.**
   - Two reproductions: the engine refuses the split (`CUDA1 expert cache: no room`, e.g. `STRATA_VRAM_FREE_MIB=0,7000`), then crashes with `0xc0000005` in `ur_loader.dll` instead of exiting cleanly.
   - Suspect: queues and contexts destroyed after the UR loader unloads, at static destruction.
   - Fix: explicit teardown before the loader unloads, or exit before the destructors run.
2. **The A770 reports no free memory.**
   - `ext_intel_free_memory` is unsupported there, so a stated figure is needed; the k22 configs carry `STRATA_VRAM_FREE_MIB=0,15000` (an earlier ladder used `0,13000`).
   - Since `d5b2875` that figure is reduced by this process's DXGI-counted use of the card and capped by the OS budget (`expert_cache.cpp` `stated_free_bytes`).
   - A default without a stated figure could use the DXGI budget less usage directly. That needs one measured run to compare DXGI's budget with Task Manager on the A770.
3. **The draft-head reservation on CUDA0** (`mtp_bind`, `generate.cpp:4466`) is held back on CUDA0 even on a split, where the drafter lives on the last stage. Moot while MTP is parked; fix it if MTP ever returns.
4. **Audit the verify expert-staging ordering (`Verifier::copy_` + flag-B doorbell, `verify.cpp:2523-2553`).**
   - Parked, not refuted: it is the leading remaining suspect for the garbage the A770 stage manufactures inside its
     own forward (hand-off row scans proved the B580 stage and the cross-card copy are clean through the storm; KB
     `references/split11-nan-story-2026-10-16.md`).
   - Read-only audit first: the private copy queue stages experts for the window, then a flag-B doorbell tells the
     window kernels the rows are there. Check the submission and wait ordering for a window that was captured as a graph
     versus one run eagerly, since eager and graphs fail differently (error 40 vs hang).
   - It needs no GPU run, so it is safe work between benchmark rounds.

## Environment notes (this PC, not engine)

- Every GPU is its own SYCL platform under Windows UR; `sycl::context` refuses cross-platform devices (probe T1:
  `Can't add devices across platforms`). Level Zero itself is not the limit - a `ze_context` can serve several devices
  of one driver - but WDDM gives each card its own driver instance, so the platform split is structural on this OS. Do
  not try a shared-context approach again; `strata::cross_context()` is the right primitive. On Linux one context holds
  both cards and the engine keeps in-place hand-offs.
- `SYCL_EXT_ONEAPI_USM_IMPORT` is undefined in the 2026.1 headers, so `import_mem` is not available.
- A single clean run on this split proves nothing. Gate every conclusion behind N >= 3 identical runs.
