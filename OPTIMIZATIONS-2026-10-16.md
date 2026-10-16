# Strata Windows SYCL - optimization and fix backlog (2026-10-16)

Everything below is measured on this PC (B580 `level_zero:0` + A770 `level_zero:1`, engine 0.1.44.1, JIT tree `build-sycl-jit`, HEAD `f4fc1e3`). Evidence: HANDOFF-WINDOWS-SYCL.md sections 13-17; probes `C:\strata-deps\cross_usm_probe.cpp` (T1-T5) and `dpct_ctx_probe.cpp`; KB `projects/strata-windows-sycl/`.

## Where things stand against VicGPT's bar

- **The 2-minute bar for a 10k-token prompt is met.** Prompt 122-182 tok/s on the split is about 55-80 s for 10k tokens (HANDOFF section 7 asked for under ~2 minutes). The VicGPT A/B (section 7, step 4) can start.
- **Decode already matches llama.cpp.** 14-17 tok/s on a fresh start, 21.4 later in a session (95-99% cache hits), against llama.cpp Vulkan's ~16.
- **Prompt speed is the remaining gap.** llama.cpp Vulkan reads 380-420 tok/s, 2-3x ahead. VicGPT's map step sends 8-18k-token prompts, so the list below is ordered for prompt speed first.

## Before measuring: check that the output is right

**The '!' x 256 reply is probably corrupted output, not a greedy loop.** '!' is token 0. In this engine, a run of token 0 is what corrupted output looks like:
- `docs/INTEL.md` (~line 708): the first request after a start answered `!!!!` when the GPU stopped waiting for the CPU's experts;
- the A770's 4 GiB misreads looked the same.

On Windows the B580's wait limit is the short one: 20,000 reads, `sycl/include/strata/sycl_doorbell.hpp:147`. That is the limit the docs blame for the first-request `!!!!`.

The configs' `sampling: {temperature 1.0, presence_penalty 1.5}` would hide that, not fix it. Check first:
1. A short prompt at temperature 0 with `STRATA_DBG_NAN=1` (prints the first non-finite logits per window).
2. If logits are non-finite: the same with `STRATA_SPIN_MAX=2000000`.
3. Only if both are clean is it greedy degeneracy (#606). Then keep the sampling block.

## How to measure

The speed ladder so far does not separate the split points:
- one run per point;
- the results are not monotonic (k14 150.3 / k18 122.6 / k22 160.9 prompt tok/s);
- 3.4k-token prompts, while VicGPT sends 8-18k.

For every item below:
- use the same ~10k-token prompt, 3 runs per arm, and a restart between runs;
- record prompt tok/s, decode tok/s and the hit rate;
- use `BENCH.bat` / `tools/strata_bench.py`;
- compare against the current k22 nomtp config measured the same way.

## Speed (prompt first)

1. **Bigger prompt chunks without giving up VRAM for the whole session.** Config only.
   - **Why the chunk is 2048 today:** in a split with `--max-context` <= 32768 the engine does not borrow (`generate.cpp:2274`), and `--prefill auto` without borrowing is forced to 2048-token chunks (`generate.cpp:2293-2295`).
   - **Measured:** 2048-token chunks 122.6 tok/s, one 4096-token chunk 182.3 (+49%, runs 7-8).
   - **The cost of own buffers:** they are held for the whole session, ~160 + chunk x 680/1024 MiB + the 96-blob ring per card (~3.2 GiB at 4096), taken from the expert cache.
   - **Borrowing:** lends cache slots only while a prompt is read; `--prefill auto` then picks up to 8192.
   - Arms:
     - `--prefill-borrow --prefill auto`;
     - `--prefill 4096` and `--prefill 8192` with own buffers.
   - Caveats:
     - Borrowed operands on the A770 can lie past 4 GiB. The oneMKL staging now applies per card (`a774ac5`), at ~10% of a prompt where it stages.
     - #340: lent slots are streamed and refilled per request. That cost is measured as small on cards that hold half their experts, as these do.
2. **Split at layer 10-11, so that every expert of the B580's layers is in its VRAM.** Config only.
   - 24,576 experts / 48 layers = 512 per layer. The B580's 5,607 slots hold ~11 layers in full.
   - Then nothing of its layers streams over its PCIe 3.0 x4 slot (3.5 GB/s measured), and its stage runs the all-resident window (#646: device plan, no doorbells).
   - The A770 (PCIe 4.0 x16, ~22 GB/s) streams the rest.
   - The ladder never went below 14. Arms: `--layer-split 10`, `11`, and 22 as the reference.
3. **An expert profile saved from VicGPT's own traffic.** Config only.
   - Hit rates rise from ~80% at a fresh start to 95-99% later in a session (decode 21.4 tok/s).
   - Save one with `expert_profile_save` during Discord-summary requests, then start with `--expert-profile` pointing at it.
4. **The A770 stage reads missed experts over PCIe again (decode only).**
   - Today `stage_reads_host` sets `pcie_num=0` for a cross-context stage, so the CPU computes all of the A770 stage's misses.
   - The fix: allocate that stage's share of the expert mirror / arena through its own queue (the T5 pattern), and drop the gate for it.
   - The prompt path is not affected: it copies experts, and copies work across contexts.
5. **Keep `--mtp` off for now.**
   - On a single B580, mtp decoded slower than without it: 6.6 vs 10.4 tok/s at 73% draft acceptance. The hit rate fell from 82% to 40.7%; draft verification thrashes the expert cache.
   - Fixing mtp across contexts (Gate, below) is correctness work. Measure single-card mtp again before counting on it for speed.
   - A cache policy that keeps draft-only lookups from evicting hot decode experts is the mtp speed item, if mtp is ever worth it here.
6. **The prompt helper across contexts.**
   - Both stages read their prompts alone today, because the helper's rows are read in place by both cards.
   - The helper could compute its share and copy the results into the stage's own rows (copy after compute, safe per T5).
   - It only binds for a prompt of one chunk (`bind_stage_helper`), so it matters for 8-18k prompts only if item 1 makes the chunk larger than the prompt. Low priority.
7. **`fd_pread` overlapped I/O (HANDOFF fix item 4).**
   - It only matters with `--stream-experts` (`GgufExpertSource`).
   - The pack source reads through `DirectFile`, which already uses `FILE_FLAG_OVERLAPPED` with a completion port.
   - Check which source the log shows before working on it.

**Corrected / dropped from the first draft of this list:**
- "Overlap chunks across stages": a multi-chunk prompt already runs stage 1 on chunk N while stage 0 reads chunk N+1 (`std::async`, `prefill.cpp:4457`). `--pipeline-windows` pipelines decode windows, not prompts. The 33% gap between 2048- and 4096-token chunks is the chunk size (item 1), not missing overlap.
- "Hand-off copy on the reading card instead of the host": the copy is T x (hc x n_embd + n_embd + hc) floats, a few hundred KB per window, against windows of ~100 ms. Not worth the change.

## Gate: mtp across the two contexts (correctness)

- Split without mtp runs clean (runs 7, 8).
- With mtp: DEVICE_LOST (error 20) + TDR in the mtp prefill draft-spends pass, identical with `STRATA_MTP_PREFILL_SYNC=0/1` (runs 9-10).
- `--split-device 0` + mtp passes (run 11).
- Suspect per the T3 probe: the draft head (137.9 MiB host rows over 106,299 tokens) and the draft layer's hidden-row hand-offs, allocated through CUDA0's `dpct::get_in_order_queue()` while the A770 stage's kernels read or write them in place.
- Fix: the `set_handoff_copy` / per-card allocation pattern from `f4fc1e3`, applied to the mtp warmup and draft verification paths.
- Speed comes after correctness here (item 5).

## Robustness

8. **Teardown crash on the "no room" refusal.**
   - Two reproductions: the engine refuses the split (`CUDA1 expert cache: no room`, e.g. `STRATA_VRAM_FREE_MIB=0,7000`), then crashes with `0xc0000005` in `ur_loader.dll` instead of exiting cleanly.
   - Suspect: queues and contexts destroyed after the UR loader unloads, at static destruction.
   - Fix: explicit teardown before the loader unloads, or exit before the destructors run.
9. **Error string wart.** On mtp failure the client receives the literal status line "mtp prefill: no error" as the error message. The catch path reuses a status string; propagate the caught `sycl::exception` text instead.
10. **No stall dump on the mtp prefill death.**
    - The verify watchdog writes `strata-stall-<pid>.dmp`, but the mtp prefill hang exits via DEVICE_LOST + abort with no dump.
    - A dump on that catch path would fingerprint the stuck kernel directly.
11. **The A770 reports no free memory.**
    - `ext_intel_free_memory` is unsupported there, so `STRATA_VRAM_FREE_MIB=0,13000` is needed for sizing.
    - Since `d5b2875` that figure is reduced by this process's DXGI-counted use of the card and capped by the OS budget (`expert_cache.cpp` `stated_free_bytes`).
    - A default without a stated figure could use the DXGI budget less usage directly. That needs one measured run to check DXGI's budget against Task Manager on the A770.

## Environment notes (this PC, not engine)

- Every GPU is its own SYCL platform under Windows UR; `sycl::context` refuses cross-platform devices (probe T1).
- `STRATA_SYCL_SHARED_CONTEXT` from `97dbfd8` was reverted. Do not reintroduce a shared-context approach; `cross_context()` is the right primitive.
- `SYCL_EXT_ONEAPI_USM_IMPORT` is undefined in the 2026.1 headers, so `import_mem` is not available as a fix vehicle.
- Serve configs now carry `sampling: {temperature 1.0, presence_penalty 1.5}`. The no-sampling default is greedy. See "Before measuring" for why the '!' loop may not be a sampling issue.
