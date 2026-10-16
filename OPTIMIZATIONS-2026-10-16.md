# Strata Windows SYCL - optimization and fix backlog (2026-10-16)

Measured on this PC: B580 `level_zero:0` + A770 `level_zero:1`, engine 0.1.44.1, JIT tree `build-sycl-jit`, branch `main` (29f4935).
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

Against VicGPT's bar and the old llama.cpp Vulkan model:
- **Prompt:** k22 benched 367 (4K) / 616 (16K) tok/s (table below), still above llama.cpp's 380-420; its map-sized
  8-18K warm reads run 260-320. The split11-class 851 is the number the frontier hunt could restore.
- **Decode:** k22 18-25 tok/s is about 1.2-1.5x llama.cpp's ~16.
- **A 10k-token map chunk** reads in about 30-38 s at k22 (was 12-16 s at split11).

**Arms for the k22 baseline** (`python bench/arms/make_arms_k22.py`, then `bench/arms/run-arm.bat <arm>`):
`k22ref`, `k22mirror6000`, `k22mirror10000`, `k22prefill16k`, `k22own4096`, `k22own4096mirror`, `k22ctx32k`, `k22ctx32kmirror`.
The old split11-based arms (`make_arms.py`) stay for the frontier hunt itself.

Historical baseline table (split11 era, `2ccc945`):

| Config | Prompt 4K | Prompt 16K | Decode |
|---|---|---|---|
| k22 (old reference) | 367 | 616 | 21.7 |
| split10 | 453 | 643 | 33.3 |
| **split11** | **621** | **851** | **35.5** |
| split12 | 628 | 860 | 35.7 (tie, less VRAM) |
| split13 | 343 | 655 | collapsed on turn 5 (do not use) |
| split11 + `STRATA_SYNC_PREAD=1` (old synchronous reads) | 377 | 554 | 34.1 |

Against VicGPT's bar and the old llama.cpp Vulkan model:
- **Prompt:** 851 tok/s at 16K beats llama.cpp's 380-420 tok/s.
- **Decode:** 35.5 tok/s is about 2x llama.cpp's ~16.
- **A 10k-token map chunk** reads in about 12-16 s at 64K.

**Open: the A770's arena past 4 GiB at 64K.** The split11 NaN story (KB `references/split11-nan-story-2026-10-16.md`) has two faults:
- the GEMM staging scratch shared across queues: fixed in `9953a5b`;
- the A770's layout past 4 GiB at 64K: open.

`start-strata-split11-ctx32k.bat` runs the same split at 32K to stay under it. VicGPT needs <= 32K. But at 32K the split does not borrow (`generate.cpp:2286`), so prompts read in 2048-token chunks and are slower than the 64K numbers above.

**The docs disagree on which launcher is the default.** `docs/INTEL_WINDOWS.md` names `start-strata-split11.bat` (64K); the header of `start-strata-split11-ctx32k.bat` calls itself "the two-card default". Pick one. Measure the 32K launcher with the same `BENCH.bat` suite: it has no prompt numbers yet.

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

- Use `BENCH.bat` / `tools/strata_bench.py`: 3 runs per arm, a restart between runs, the same prompts.
- Include a ~10k-token prompt, since VicGPT's map step sends 8-18k.
- Record prompt tok/s, decode tok/s, the hit rate, and each card's dedicated and shared GPU memory peak.
- Compare against split11 at the same context length (32K or 64K, not mixed).

## Speed (no MTP)

1. **Mirror the A770 stage's missed experts in pinned RAM.** The largest remaining lever.
   - **Today:**
     - On a split the expert mirror covers only stage 0's layers (`mirror_end = split_at[0]`, `generate.cpp:4941`), because the later stages' caches do not exist yet when it is built (#1054, #1440).
     - At split11 the B580 stage has no misses, so no mirror is built at all.
     - Every miss of layers 11-47 is read from the GGUF: the prompt path streams it with `fd_pread`, and the CPU pool's decode reads go through `GgufExpertSource::blob()`, which checks the mirror first, then the ring + `fd_pread`.
     - Windows' file cache absorbs part of this, but the 2.3x from overlapped reads shows how much the file path costs.
   - **Windows bug on the way:** the default mirror size reads `MemAvailable` from `/proc/meminfo` (`generate.cpp:4971`). On Windows there is no such file, so the cap is 0 and no mirror is built unless `STRATA_MIRROR_MIB` is set. Use `GlobalMemoryStatusEx` (`ullAvailPhys`) there.
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
   - The A770 is on PCIe 4.0 x16 (17-22 GB/s measured); `pcie_frac` was 0.48 on its probe.
3. **Prompt chunk size.**
   - **64K:** the split borrows, and `--prefill auto` caps at 8192-token chunks (`prefill_auto_max`). A 16K prompt is then two passes over the experts.
     - Arm: `--prefill auto:16384`, measured on the 16K prompt.
     - Evidence that fewer passes pay: 2048 -> 4096-token chunks gave +49% at k22 (runs 7-8), and split11 reads 16K prompts 37% faster than 4K ones.
     - The old gate (the A770 past-4-GiB issue) was demoted by the split search: k22 stages GEMM operands past 4 GiB and stays clean. Arm `k22prefill16k` measures it directly at k22.
   - **32K:** no borrowing, so auto is forced to 2048-token chunks (`generate.cpp:2305-2307`). Arms:
     - `--prefill 4096` with own buffers: ~160 + 4096 x 680/1024 MiB + the ring per card (~3.2 GiB), taken from the expert caches for the whole session;
     - `--prefill-borrow --prefill auto`: lends during prompts only, but puts the A770's borrowed operands back into the past-4-GiB territory 32K was chosen to avoid.
4. **An expert profile from VicGPT's own traffic.** Config only.
   - Hit rates rise from ~80% to 95-99% during a session.
   - Save one with `expert_profile_save` during Discord-summary requests, then start with `--expert-profile` pointing at it.
   - Fewer A770 misses means less of everything in items 1-3.
5. **Split 12 instead of 11, if the VRAM goes somewhere useful.** It ties split11 on speed and uses less VRAM. Worth it only if that VRAM ends up in the A770's cache, or makes the 32K prompt-chunk arm (item 3) fit.

**Low priority, kept for the record:**
- **The prompt helper across contexts:** both stages read their prompts alone. The helper binds only for a one-chunk prompt (`bind_stage_helper`), so it matters for 8-18k prompts only if item 3 makes a chunk as large as the prompt.

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
1. **Teardown crash on the "no room" refusal.**
   - Two reproductions: the engine refuses the split (`CUDA1 expert cache: no room`, e.g. `STRATA_VRAM_FREE_MIB=0,7000`), then crashes with `0xc0000005` in `ur_loader.dll` instead of exiting cleanly.
   - Suspect: queues and contexts destroyed after the UR loader unloads, at static destruction.
   - Fix: explicit teardown before the loader unloads, or exit before the destructors run.
2. **The A770 reports no free memory.**
   - `ext_intel_free_memory` is unsupported there, so `STRATA_VRAM_FREE_MIB=0,13000` is needed.
   - Since `d5b2875` that figure is reduced by this process's DXGI-counted use of the card and capped by the OS budget (`expert_cache.cpp` `stated_free_bytes`).
   - A default without a stated figure could use the DXGI budget less usage directly. That needs one measured run to compare DXGI's budget with Task Manager on the A770.
3. **The draft-head reservation on CUDA0** (`mtp_bind`, `generate.cpp:4466`) is held back on CUDA0 even on a split, where the drafter lives on the last stage. Moot while MTP is parked; fix it if MTP ever returns.

## Environment notes (this PC, not engine)

- Every GPU is its own SYCL platform under Windows UR; `sycl::context` refuses cross-platform devices (probe T1). Do not try a shared-context approach again; `strata::cross_context()` is the right primitive.
- `SYCL_EXT_ONEAPI_USM_IMPORT` is undefined in the 2026.1 headers, so `import_mem` is not available.
- A single clean run on this split proves nothing. Gate every conclusion behind N >= 3 identical runs.
