# Strata Windows SYCL - optimization and fix backlog (2026-10-16)

Everything below is measured on this PC (B580 `level_zero:0` + A770 `level_zero:1`, engine 0.1.44.1, JIT tree `build-sycl-jit`, HEAD `f4fc1e3`). Evidence: HANDOFF-WINDOWS-SYCL.md sections 13-17; probes `C:\strata-deps\cross_usm_probe.cpp` (T1-T5) and `dpct_ctx_probe.cpp`; KB `projects/strata-windows-sycl/`.

## Gate: make mtp work across the two contexts

1. **mtp buffers cross contexts - the last broken path.** Split without mtp runs clean (runs 7, 8); with mtp: DEVICE_LOST (error 20) + TDR in the mtp prefill draft-spends pass, identical with `STRATA_MTP_PREFILL_SYNC=0/1` (runs 9-10); `--split-device 0` + mtp PASSES (run 11). Suspect per the T3 probe: the draft head (137.9 MiB host rows over 106,299 tokens) and the draft layer's hidden-row hand-offs, allocated through CUDA0's `dpct::get_in_order_queue()` while stage-1 (A770) kernels write or read them in place. Apply the `set_handoff_copy` / per-card allocation pattern from `f4fc1e3` to the mtp warmup and draft verification paths.

## Speed

2. **Per-stage mirror/arena allocation (the decode recovery).** Today `stage_reads_host` forces `pcie_num=0` for a cross-context stage: the A770 stage computes its expert misses on CPU. Allocate that stage's share of the expert mirror / arena through ITS OWN queue (the T5 pattern) and turn its PCIe share back on. Expected: decode above today's 13.9-16.6 tok/s and higher hit rates.

3. **Split search past 22.** Measured (clean nomtp config, ~3.4K prompts): k14 150.3/16.4, k18 122.6/13.9, k22 160.9/16.6 (prompt/decode tok/s) - 22 best so far, non-monotonic. Search 24-28 and consider picking the split by dense-weight balance rather than a scan.

4. **Multi-chunk prefill is 33% slower than single-chunk on the split.** 4096-token chunk: 182.3 tok/s; `--prefill auto` (2048 chunks, stages unbound): 122.6. Options: raise the chunk on splits; overlap chunks across stages (stage 1 works chunk N while stage 0 does N+1) using per-chunk copied hand-off buffers - `--pipeline-windows` is currently refused cross-context, which is exactly the hole to fill.

5. **Prefill helper unbound across contexts loses help.** Both stages "read their prompts alone". The helper could compute its share and copy results into the peer stage's OWN mapped rows (copy after compute - safe per T5) instead of both reading one shared buffer in place.

6. **Hand-off copy overlap.** `forward_handoff` serializes: stage 0 window sync -> host memcpy -> stage 1 launch. The reading card can do the copy itself (`q1.memcpy` from the foreign buffer is proven correct, T5), which frees the host thread and could overlap with stage 1's first-layer upload.

7. **Draft verification thrashes the expert cache (mtp).** Single-card mtp: hit 40.7% vs 82% nomtp, decode 6.6 vs 10.4 tok/s despite 73% draft acceptance. Candidate policy: don't cache experts looked up only by speculative drafts below an acceptance threshold, or keep a dedicated reservation so verify windows (T=6) do not evict hot decode experts.

## Robustness

8. **Teardown crash on the "no room" refusal.** Two reproductions: engine refuses the split (`CUDA1 expert cache: no room`, e.g. `STRATA_VRAM_FREE_MIB=0,7000`) then crashes `0xc0000005` in `ur_loader.dll` instead of exiting clean. Suspect queues/contexts destroyed after the UR loader unloads at static destruction; explicit teardown before loader unload, or exit before destructors.

9. **Error string wart.** On mtp failure the client receives the literal status line "mtp prefill: no error" as the error message - the catch path reuses a status string; propagate the caught `sycl::exception` text instead.

10. **No stall dump on the mtp prefill death.** The verify watchdog writes `strata-stall-<pid>.dmp`, but the mtp prefill hang exits via DEVICE_LOST + abort with no dump. Adding a dump on that catch path would fingerprint the stuck kernel directly.

11. **A770 reports no free memory.** `ext_intel_free_memory` unsupported, driver claims 0 free, so `STRATA_VRAM_FREE_MIB=0,13000` fakes are needed for sane sizing. Consider defaulting to the DXGI/agreed_account numbers (already used for budgeting) when the L0 free-memory query is unsupported.

## Environment notes (this PC, not engine)

- Every GPU is its own SYCL platform under Windows UR; `sycl::context` refuses cross-platform devices (probe T1). `STRATA_SYCL_SHARED_CONTEXT` from `97dbfd8` was reverted - do not reintroduce a shared-context approach; `cross_context()` is the right primitive.
- `SYCL_EXT_ONEAPI_USM_IMPORT` is undefined in the 2026.1 headers - import_mem is not available as a fix vehicle.
- Serve configs now carry `sampling: {temperature 1.0, presence_penalty 1.5}`; the no-sampling default is greedy and produced a #606 '!' repeat loop (guard worked correctly).
