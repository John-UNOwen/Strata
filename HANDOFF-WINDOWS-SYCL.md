# Handoff: Strata's Intel (SYCL) engine on native Windows

**Status (2026-10-15):** it builds and serves on native Windows on **one Arc B580**. It is too slow to be useful yet.
**Branch:** `windows-sycl` (code commits on upstream `296b59d`; see section 5).
**End goal:** Qwen3.8-Flash-Next IQ2_XS fast enough to summarize Discord chat for VicGPT (`<HOME>\Documents\GitHub\VicGPT`; `<HOME>` is the user's Windows home directory).
- The bar to beat is VicGPT's current model: Qwen3.8-27B on llama.cpp Vulkan across both Arcs, at **~380-420 t/s prompt and ~16 t/s decode**.
- VicGPT's map step sends 8-18k-token prompts, so **prompt speed matters most**.

---

## 1. Machine rules (read first)

This PC bugchecks when GPU memory is overcommitted, and it has a history of hangs under GPU compute.

1. **The user launches every model load**, in a normal terminal, watching Task Manager. An agent may build, run `sycl-ls`, `strata.exe --help` and `--families`, and short kernel parity tests the user has OK'd. **No sustained inference from an agent shell.**
2. **Stop signal:**
   - **shared** GPU memory above ~0.2 GB on either Arc;
   - **dedicated** memory on the B580 near ~11.3 GB;
   - Windows RAM nearly full.
   Ctrl+C immediately.
3. **Close VicGPT's model server first.** It is `llama-server` from VicGPT's `start-llm.bat` and holds most of both cards.
4. Ask the user before any download (name the file, source and size).
5. Never read or print secrets (VicGPT's `.env`: `DISCORD_BOT_TOKEN`, `VICGPT_API_KEY`). No cloud LLM calls.
6. This checkout is owned by `BUILTIN\Administrators`. Use `git -c safe.directory=* ...` rather than changing the global git config.

## 2. Hardware

| | |
|---|---|
| **Arc B580** 12 GB (Battlemage) | PCIe 3.0 **x4** chipset slot (measured 3.5 GB/s host->device), ReBAR on, **no display** |
| **Arc A770** 16 GB (Alchemist) | PCIe 4.0 x16 (~22 GB/s), **drives the desktop** (~0.6-1 GB used) |
| UHD 770 iGPU | never select it |
| Level Zero order (Windows) | `level_zero:0` = B580, `:1` = A770, `:2` = iGPU |
| CPU / RAM | i7-12700K (AVX2, no AVX-512), 32 GB DDR4-4000 |
| Windows Arc driver | 32.0.101.9150 (2026-10-07). Native Level Zero works on both cards with it; driver 6913 crashed in `ze_intel_gpu64.dll`. |
| oneAPI | `C:\Program Files (x86)\Intel\oneAPI` (2026.x), `setvars.bat` |

## 3. Layout and how to run

| What | Where |
|---|---|
| Engine (AOT `bmg-g21`, B580 only) | `build-sycl-aot\strata.exe` (+ `strata.map` for fault triage) |
| Other build dir | `build-sycl-win\` (earlier/parity build) |
| Model | `C:\strata-data\models\IQ2_XS\*.gguf` (2 shards, 68 GB), `C:\strata-data\packs\iq2_xs\`, `C:\strata-data\mtp\rt` |
| Config (not in git; on the PC only) | `strata-iq2_xs-win.json` (env: `ONEAPI_DEVICE_SELECTOR=level_zero:0`, port 8081) |
| Launcher (not in git; on the PC only) | `start-strata.bat` (sources `setvars.bat`, runs `python serve\server.py --engine strata --config strata-iq2_xs-win.json --port 8081`) |
| Engine log | `strata-iq2_xs-win.log` |
| API | `http://127.0.0.1:8081/v1` (OpenAI-compatible). VicGPT can point at it with `VICGPT_LLM_BASE_URL=http://127.0.0.1:8081/v1`. The server honors `chat_template_kwargs.enable_thinking=false`, which VicGPT sends. |

The config also loads the **experimental speed projection** (`--control-vector-scaled ...:1.0`, layers 4-44). This is the refusal-direction projection. **Keep it on**: the user wants uncensored summaries.

## 4. Measured so far (B580 only, IQ2_XS, from `strata-iq2_xs-win.log`)

| | Value |
|---|---|
| Load | ~10-15 s (pack 0.8 s, projections 2.5 s, expert-cache fill 1.2-4.7 s) |
| Expert cache | 2,725 of 24,576 experts in VRAM (3.67 GiB); decode hit rate 46-61% |
| Free VRAM after load | ~2.2-2.3 GB |
| Prompt | **10.7-17.8 t/s** (72 and 44 new tokens) |
| Decode | **7.4-11.6 t/s**, MTP drafts accepted ~50-60% |

That means **~11 minutes for one 10k-token VicGPT chunk**. llama.cpp Vulkan does it in ~30 s.

The cause: a 12 GB card holds ~11% of the experts, so the CPU (AVX2, 9-11 pool workers) computes the rest. The log also says "the prompt path allocates its own buffers (too few cache slots to borrow)".

## 5. What the branch changes (for reviewers)

| Commit | Change |
|---|---|
| `510d71f` | `gguf_expert_source.cpp`: Windows `fd_open`/`fd_close`/`fd_pread`. A HANDLE is stored in an int; `ReadFile` uses an `OVERLAPPED` offset. |
| `a9ffd23`, `3a662c2`, `25a5477` | `windows.h` macro clashes (`small`/`large`/`OUT`/...) via `strata/win_macros.hpp` and local undefs; a `memmem` shim; `_putenv_s` in the parity tests |
| `8bd0e76` | `pinned.dp.cpp`: the device LUID for the WDDM shared-memory budget, matched by name through DXGI (Level Zero exposes no LUID) |
| `9ec1997`, `0433d9f` | CMake: `-fsycl-device-code-split=off` on Windows (per-kernel split made sycl-post-link take 41 min, then run out of memory); link `mkl_sycl_blas_dll`; `/fp:precise` instead of `-fp-model=precise` |
| `c7acd7f` | +4 KiB tail slack on pinned/polled host buffers (verify, session, mtp, expert sources) |
| `072fc0d` | `LINKER:/MAP` link map (Windows only since the review fix) |
| `4857f1d`, `2f205be` | "no free-memory report" fallbacks in `expert_cache.cpp` / `remote_experts.cpp` (since dropped, fix item 2). With `--expert-cache-device1 auto`, `remote_experts` shrinks the tier to fit, as the CUDA engine does; an explicit slot count that does not fit still refuses |
| review fix | `/MAP` gated on `WIN32`; the `remote_experts` fallback gated on `_WIN32` and the shrink gated on `auto`, so the Linux and explicit-count paths match upstream again |

## 6. Fix list (do these first)

1. ~~`LINKER:/MAP` outside `if(WIN32)` broke the Linux link.~~ Fixed: it is inside `if(WIN32)` now.
2. ~~The free-memory fallbacks (`4857f1d`, `2f205be`) assumed the card was empty (`total - 1 GiB`, `total - 2 GiB`).~~ Dropped: Windows Level Zero reports free memory on this PC for the B580 (the log says "6.80 GiB free"). Update 2026-10-16: the A770 answers nothing (no `ext_intel_free_memory` there), and the honest replacement now exists: `STRATA_VRAM_FREE_MIB="0,13000"` states the free MiB per device index for silent drivers (`4b727ff`, see section 9). Nothing is guessed. Since the review fix the figure is what was free **before** Strata started: DXGI's count of this process's use of the card is taken off it on every read, so it shrinks as the caches fill (section 9, item 2).
3. **The `+4096` tail slack hides a real out-of-bounds read.** Host code (the reference gemv, the pool kernels) reads past the end of these buffers. Linux has the same bug; it is just masked there. Find the reader and bound it, or at least file it upstream.
4. **`fd_pread` serializes.** `CreateFileA` is called without `FILE_FLAG_OVERLAPPED`, so Windows serializes I/O on the handle, and the "several threads pread one handle" design (mirror pool, prompt stagers) runs one read at a time.
   - Open with `FILE_FLAG_OVERLAPPED` and wait per call (event in the OVERLAPPED, or `GetOverlappedResult`), or open one handle per thread.
   - Check the effect on prompt speed.
   - Fix the comment above `fd_open` too (it says several threads can read one handle at once), and clamp `n` to `DWORD` per call (a read over 4 GiB would be cut short).
5. ~~**AOT `bmg-g21` only.** The A770 (Alchemist/DG2) can't run this binary.~~ Done: the JIT tree (`build-sycl-jit`, `53c2728` + `4b727ff` needed on top) runs both cards; section 9. A dual-target AOT (`bmg-g21,dg2`) also builds and launches, but its dg2 kernels cannot address past 4 GiB of an allocation (section 9.4).
6. **Config:**
   - `--max-context 65536` in `strata-iq2_xs-win.json` is untested (the last run used 32768, with ~2.3 GB free after load). Go back to 32768 until measured.
   - Remove `STRATA_SYCL_RESIDENT_EXPERTS=0`: nothing in the repo reads it.
   - `STRATA_NO_NORM_ROPE=1` turns off the fused norm+rope (`sycl/src/kernels/cuda/native_rope.dp.cpp`). If the B580 needs it for correct output, that is a parity failure: run `rope_parity` (check 6) on the B580 and record the result.
7. **`start-strata.bat`:**
   - The comment's stop signal is wrong. It is shared GPU memory above ~0.2 GB, or dedicated near ~11.3 GB; not "shared past ~11.3 GB".
   - It runs `python` from PATH. Pin a venv, or check the required packages at startup.

## 7. Next development steps (in order)

1. **Fix list item 6** (config), then commit. Items 1 and 2 are done.
2. **Two-card split, B580 + A770.** This is the main lever: ~26 GB of VRAM instead of ~11, so many more experts on the GPU.
   - **Build (JIT, both cards).** Same configure command as `build-sycl-aot`, in a new directory and without AOT: `-B build-sycl-jit -DSTRATA_SYCL_AOT= -DSTRATA_SYCL_PARITY=OFF`, then `cmake --build build-sycl-jit --target strata`. One build cannot hold both AOT targets: the grouped XMX kernels have one shape per build (sub-group 8 for the A770, 16 for the B580). In the JIT build the B580 skips them (`STRATA_PF_XMX` is ignored there) and uses the other kernels. `STRATA_SYCL_JIT_STATELESS` stays ON: without it the A770 misreads expert caches past 4 GiB (`docs/INTEL.md`).
   - **First start compiles every kernel for both cards** (about 47 s for one card on Linux; expect longer here: two devices, and one module per file on Windows). That is CPU time, not GPU memory. If the first compile crashes, set `SYCL_CACHE_PERSISTENT=0` (it crashed on Xe2 on Linux). Update 2026-10-16: with per-kernel split (section 9.1) the first start here reached "session is up" in well under a minute, with `SYCL_CACHE_PERSISTENT=1` kept on.
   - **Config:** copy `strata-iq2_xs-win.json` to `strata-iq2_xs-2card.json` and change:
     - `ONEAPI_DEVICE_SELECTOR=level_zero:0;level_zero:1` (the semicolon form is the one verified in the config/bat here). The B580 is the first stage, the A770 the last. The last stage runs the head, the draft layer and the verify, so the faster card goes last (`serve/server.py`, `ordered_gpus`).
     - `--layer-split 18` in `args` (layers 0-17 on the B580, 18-47 on the A770). Then try 14 and 22, one change per run. Not `auto`: under WSL it gave the B580 only layers 0-1.
     - `--max-context 32768`.
     - The engine path: `build-sycl-jit\strata.exe`.
     - An `--expert-profile` must be in `args`: a split across cards refuses to start without one.
     - Optional: `STRATA_STAGE_TRIM=1` in `env`. Each card then loads only its own layers' dense weights and gives the rest to its expert cache. It is opt-in upstream, and the output can differ slightly. (Used in the verified 2-card config.)
      - Required on this PC: `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` (without it an A770 device allocation past 4 GiB comes back aliased, section 9.3) and `STRATA_VRAM_FREE_MIB=0,13000` (the driver reports no free memory on the A770, section 9.2). Commas only: `0;13000` is refused with a message, not half-read.
   - **Run it as before:** VicGPT's server closed, launched by the user, Task Manager open. Watch **both** cards' shared and dedicated memory now. The A770 also carries the desktop, so its stop line is dedicated memory near ~15 GB (not measured; scaled from the B580's 11.3 of 12).
   - **Read in the log:** "layer split across 2 GPUs", each stage's experts in VRAM, "experts of the later stages are neither in VRAM nor mirrored", and the decode hit rate. Record prompt and decode t/s for each K beside section 4's numbers.
   - Expect each card to be slower per expert than in an AOT build: the kernels are JIT-compiled, and the B580 has no XMX prompt kernels in this build.
3. **Prompt speed.**
   - Look into why the prompt path computes on the CPU ("too few cache slots to borrow").
   - Measure `fd_pread` with overlapped I/O (item 4).
   - Compare `--prefill` settings and `STRATA_PROMPT_ATTN_XMX=1`.
   - Use `BENCH.bat` / `tools/strata_bench.py` for repeatable numbers.
   - **Target:** a 10k-token prompt in under ~2 minutes before it is worth a VicGPT A/B.
4. **VicGPT A/B.** Run the same 24h `#spenceral` window that the Qwen/Gemma comparisons used; compare time, digest quality and VRAM peaks. Record the result in VicGPT's `AGENTS.md`.
5. **Optional: upstream.** Generic Windows/SYCL fixes (fd wrappers, macro guards, CMake) could go upstream as a PR, but only with the user's OK.
   - Use a clean branch: leave out this file (it describes the user's PC and VicGPT setup), the `/MAP` commit and the tail-slack commit.
   - Use `strata/win_macros.hpp` everywhere: `generate.cpp` and `verify_kernels.dp.cpp` keep their own `#undef` lists (with `medium`, which the header lacks).
   - The DXGI LUID lookup matches by device name, so two identical cards would both get the first card's LUID. Say so in a comment.

## 8. Background: what was tried before this branch

- **WSL2 + Docker (Linux SYCL build): blocked.**
  - Under WSL, the B580's Level Zero aborts on the first kernel launch (`neo/.../linear_stream.h:91`). This happens with compute-runtime 26.09, 26.31 and 26.35, and with Windows drivers 8992 and 9150. The A770 and OpenCL work.
  - Strata over OpenCL loads the whole model, then fails: OpenCL has no SYCL Graph support, and graphs are recorded in about 12 places (`verify.cpp`, `mtp.cpp`, `session.cpp`, `graph.cpp`). `STRATA_VERIFY_EAGER=1` covers only one of them.
  - Full write-up: VicGPT `reports/intel-b580-wsl-level-zero.md`.
  - The WSL artifacts are still on disk: `/root/strata-work` (~75 GB), Docker image `strata-sycl-dev` (13 GB), `%USERPROFILE%\.wslconfig` (26 GB).
- **llama.cpp SYCL vs Vulkan** (same PC, Qwen3.8-27B, both cards): Vulkan b11576 reads prompts at 405-422 t/s and decodes at 16.2. SYCL b11576 read at 196-225 t/s and decoded at 11.0. SYCL also put shared GPU memory on both cards as context grew. Expect Level Zero on this PC to lag Vulkan.
- **Planning doc this branch followed:** VicGPT `reports/strata-windows-sycl-plan.md`.

## 9. The two-card run (2026-10-16): what settled

**The split starts, verifies, and idles.** Startup probe (engine 0.1.44.1, JIT tree, no inference run): CUDA0 (B580) layers 0-17 with a 5607-slot expert cache (7.34 GiB), CUDA1 (A770) layers 18-47 with a 6950-slot cache (9.48 GiB, "slot 0 verified" - the kernels ran on dg2), 12557 resident experts against 3471 single-card, 8410 later-stage experts host-served, "session is up", "2077 MiB of VRAM free with everything loaded", engine stable through a 460 s watch at CPU 3.4 s. Prompt/decode t/s for the split are NOT measured yet - the user launches `start-strata-2card.bat` (git-excluded, repo root; config `strata-iq2_xs-2card.json`).

Four findings made it work; the first two are committed:

1. **Windows JIT needs per-kernel device code split** (`53c2728`). The CMake forced `code-split=off` on every Windows build (the workaround for the AOT post-link blowup), which makes the whole-TU module's JIT compile every kernel the file contains: the A770 died at start with "Unsupported JointMatrix operation ... SG16 ... supported values: 16" on Battlemage joint-matrix kernels it never launches. The rule is `off` only for `WIN32 AND STRATA_SYCL_AOT` now. With per-kernel split the engine reaches "session is up" in well under a minute with `SYCL_CACHE_PERSISTENT=1`.
2. **The Windows driver reports no free memory on the A770** (Level Zero `ext_intel_free_memory` absent there; the B580 has it), so every size-from-free decision saw 0: the split stage got its weights and zero expert slots, and the remote tier refused. `4b727ff` adds `STRATA_VRAM_FREE_MIB="0,13000"`: explicit free MiB per device index for silent drivers, a user-stated figure, never a computed guess (0 entries keep the driver's number; keep the figure honest - A770 is 15.56 GiB total with ~1 GiB used by the desktop). It feeds `device_free_bytes()` and the split's `stage_room`.
   - **Review fix (2026-10-16):**
     - The figure is free MiB **before Strata starts**, by device index after `ONEAPI_DEVICE_SELECTOR` (the N of "CUDAN" in the log).
     - On every read, DXGI's `QueryVideoMemoryInfo` (this process's usage of that card) is taken off it, capped by the OS budget for the process. A fixed figure used to make every after-allocation check pass.
     - The device is found among the DXGI adapters by name. If none has it, or the adapters of that name report different memory (two identical cards in use), the variable is not used for that card, and the log says so. Several adapters that report the same figures are accepted (section 11).
     - Anything but numbers, spaces and commas disables the variable with one message. `0;13000` used to hang the parser and fill RAM.
     - The variable is Windows-only now; Linux keeps its DRM-fdinfo fallback.
     - The split search uses `device_free_bytes()` only when the driver reports 0, so Linux splits size as before.
     - **To check on the first run:** the lines "CUDA1 reports no free memory; STRATA_VRAM_FREE_MIB 13000 MiB less this process's X GiB on it (DXGI)", one per read. X must grow once the A770's cache is allocated (to about the cache size in the log, ~9.5 GiB). If it stays near 0, DXGI does not count Level Zero allocations: stop, and do not trust the figure.
3. **`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` is required on the A770.** Without it `sycl::malloc_device` past 4 GiB returns aliased pages: writes land before 4 GiB, reads past it come back wrong - deterministic across runs (test matrix: `C:\strata-deps\device1_probe*.cpp`, 13 GiB allocations, host-written tags). With the env var plus JIT kernels the matrix reads back clean 2/2. The engine's arena alias guard (`STRATA_ARENA_ALIAS_CHECK`) detects the aliasing but cannot fix it. Set in the 2-card config/bat.
4. **AOT for dg2 is a dead end past 4 GiB addressing** (finding, not committed). A dual-target AOT build (`-device bmg-g21,dg2`) compiles ("Build succeeded for : dg2") and its images launch, but kernels AOT-targeted at dg2 cannot address past 4 GiB of any allocation - host-written tags read correctly to +4095 MiB and everything past +4096 MiB reads wrong - even with relaxed allocation limits. Only usable with arenas under 4 GiB; the JIT tree is the regime for the A770 (the `STRATA_SYCL_JIT_STATELESS` comment above is the same 4 GiB surface-offset trap, approached from the other side).

Tree and files: `build-sycl-jit` = the `build-sycl-aot` recipe with `-DSTRATA_SYCL_AOT=` empty, `-DSTRATA_SYCL_PARITY=OFF`, `-DSTRATA_SYCL_JIT_STATELESS=ON` (default). Startup probe: `C:\strata-deps\probe2j.bat` (bounded, load + verify only). Git-excluded locals: `strata-iq2_xs-2card.json`, `start-strata-2card.bat`, their logs.

Unchanged open work for whoever continues: measure prompt/decode t/s for `--layer-split 18` (then 14 and 22), the "too few cache slots to borrow" prompt path, fix-list items 3 (tail-slack OOB read) and 4 (`fd_pread` overlapped I/O), item 6's single-card config cleanup, then the prompt-speed steps of section 7.

## 10. Status of the two-card launch (2026-10-16, after `d5b2875`)

Section 9's startup success no longer holds on `d5b2875`: the rebuilt engine exits at ~20 s with `layer split, CUDA1: 0.00 GiB free of 15.56, room for experts 0.00 GiB` then `layer split, CUDA1 expert cache: no room` (B580 stages normally, 5607 slots). **Fixed in `060dc6a` (section 11) and verified on this PC 2026-10-16 (rebuilt JIT tree, bounded probe, engine stopped after verification, no inference):** start reaches "session is up" in ~15 s (JIT cache warm); the shrink lines print `STRATA_VRAM_FREE_MIB 13000 MiB less this process's 3.40 GiB on it (DXGI), within the OS budget of 15.13 GiB: 9.30 GiB free` - DXGI counts the Level Zero allocations, the section 11 first-run check passes; `room for experts 6.09 GiB`, CUDA1 (A770) runs layers 18-47 with a 4453-slot cache (6.08 GiB, "slot 0 verified"), 10060 resident experts. The A770 cache is smaller than section 9's 9.48 GiB because the figure is now honestly reduced by what the process already holds - intended. The build itself is clean - `pinned.dp.cpp` (DXGI code) and `expert_cache.cpp` compile without errors (only the usual dpct unused-function warnings); the config/env are correct as listed.

**First real requests hang in mtp prefill (two runs, 2026-10-15 22:17 and 22:20).** The split loads and idles fine; a 1,960-token prompt reads for ~25 s, then: `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` + `strata serve: mtp prefill: no error`. System events: `Display #4101 "Display driver igfxnd stopped responding and has successfully recovered"` at both failures - the TDR watchdog killed a device transaction past its delay on the A770 (`TdrDelay=10`, `TdrDdiDelay=20` already set under `HKLM\...\GraphicsDrivers`, from the llama.cpp era). The engine then dies: run 1 `0xc0000409` in ucrtbase (unhandled exception - `mtp.cpp` reports via `dpct::get_error_string_dummy(0)`, hence the odd "no error" text); run 2 `0xc0000005` in `ur_loader.dll` (accessing the lost device). Dumps: `%LOCALAPPDATA%\CrashDumps\strata.exe.*.dmp`. The driver recovers each time; the desktop is unaffected.

Knobs already ruled out: `STRATA_MTP_PREFILL_SYNC=1` (the old per-group loop) had no effect - both modes end up launching a graph per group (mtp.cpp:1346-1363). `STRATA_MTP_PRIORITY` is a no-op in the SYCL build: dpct drops queue priority and both branches create the same queue (mtp.cpp:483-508). Relevant port notes (docs/INTEL.md, "What the port had to get right"): every device spin is capped (`kSpinMax`), so a 10 s spin kernel is unlikely; host-to-device flag visibility *during* a kernel is unreliable on Intel; graphs are recorded in about 12 places. Suspect family: graph launch or the cross-card hand-off path on dg2 under Windows Level Zero.

Isolation runs prepared (git-excluded, one variable each, separate logs; restart fully between attempts - a live process cannot survive a device loss):

**nomtp run executed (2026-10-15 22:26): mtp is not the root cause.** Without `--mtp` the prompt path ran fine, the window graphs captured cleanly on CUDA1 (`verify: captured the 4-token window (upload no error, sync no error)` x2), and the hang moved to the first decode window launch: `verify: timed out at layer 18; its GPU waits were released but the GPU did not finish within 5 s (#267)` - layer 18 is the first layer of the CUDA1 stage; the engine's per-layer stage watchdog (verify.cpp, `release_gpu_waits(5000)`) fired, then `Display #4101` at 22:27:20 killed the still-busy transaction. No host crash event this time. So both failure families sit in work executing on CUDA1 (the A770): draft prefill first, verify window after; graph capture itself succeeds.

Remaining variants (both without `--mtp` now, since mtp is cleared):
- `start-2c-nomtp.bat` (updated: `STRATA_VERIFY_EAGER=1` - the window runs on the queue instead of as a graph; verify.cpp:822). Completes -> graph launch on dg2/Windows L0 is the culprit; still times out at layer 18 -> the CUDA1 kernels/queues themselves.
**Eager run executed (2026-10-15 22:31): the error class changed to a memory failure.** With `STRATA_VERIFY_EAGER=1` the prompt prefill completed; the engine then died in the generic window catch-all (verify.cpp:2454-2457): `level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`, `std::exit(1)`. A TDR event (#4101) still fired in the same second, so it is not settled whether the eager 48-layer body truly out-allocates the room (the graph path ran on fixed 56.9 MiB window buffers; slack at that point: ~2.08 GiB on B580, ~0.7 GiB on CUDA1 with the stated 13,000 MiB), or a kernel hung, TDR reset the card, and a later allocation reported the odd code. The next nomtp run is instrumented: `STRATA_VERIFY_DEBUG=1` (per-layer residual ladder, verify.cpp:769/2313) and `UR_LOG_LEVEL_ZERO=debug` (the Level Zero plugin prints the failing driver call with sizes). Reading: a ladder that stops at a layer with a `ze*` failure names the allocation; a ladder that hangs before any allocation failure means the hang is unchanged and eager only relabeled it.

**Instrumented eager run (2026-10-15 22:34): window 1 ran clean on both cards; window 2 died mid-launch.** The ladder rang/served layers 0-17 and printed `T=4 window: staging 0.0 ms, gpu 114.4 ms` - the CUDA1 segment executed in milliseconds, no layer-18 timeout. Window 2 logged `staged; launching` (verify.cpp:2023) but never reached `launched` (verify.cpp:2177): a `sycl` exception `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (39) thrown inside the launch block (verify.cpp:2027-2177) reached the catch-all (2454) and exited; a TDR event also fired at 22:35:21. The `UR_LOG_LEVEL_ZERO=debug` value was rejected by the runtime (format is `param:value`, correct form `level:debug`).

**Leading candidate (verify.cpp:2027-2041):** upstream known bug: the first window launched "behind that work still in flight on another queue" after other GPU work (a prefill, the previous request) never starts on Intel Arc (#2194: the same on an A770 under xe; engine dies). The mitigation drains all queues before every window; the default `STRATA_WINDOW_SYNC` is 1 only for Alchemist cards, 0 for Battlemage. In the two-card split each stage's verifier has its own default, and the split creates exactly the cross-card, cross-queue pattern the rule was written for: the B580-side verifier (Battlemage) launches its window behind CUDA1 work with no drain. Next nomtp run adds `STRATA_WINDOW_SYNC=1` (force the drain on both stages) plus the corrected `UR_LOG_LEVEL_ZERO=level:debug`. If decode survives several windows: retest with graphs (drop EAGER) plus the drain, then reintroduce `--mtp` step by step.

**Drain run executed (2026-10-15 22:37, `STRATA_WINDOW_SYNC=1` + `UR_LOG_LEVEL_ZERO=level:debug`): the drain did not save window 2, and the error moved to resource exhaustion.** Same structure: window 1 clean (`T=4 window: staging 0.0 ms, gpu 111.4 ms`), window 2 died between `staged; launching` and `launched` with `error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)`. TDR fired at 22:38:30, about 6 s after the engine exited (something was still running on the A770 after death). Composition of the instrumented L0 trace for that run: 896 `ze_event_pool_desc_t` creations total (384 by the time the session was up, 512 more during the two windows), and the last calls before the failure were `create command list` (immediate, inOrder) on two device handles, then the event-pool line flood ends at the exception.

Environment facts from the same trace: L0 loader 1.33.1, three L0 driver handles, driver versions `1.17.39894+4` and `1.6.34728`, plus "Skipping L0 V1 adapter". **Corrected by the graph_probe series below: both Arcs run on `xe 1.17.39894+4`; the 1.6.34728 i915-era runtime carries the UHD 770 iGPU, which the selector excludes. The A770 is NOT on an old runtime.**

**graph_probe series (2026-10-15 22:4x, source + build: `C:\strata-deps\graph_probe.cpp`, `icx -fsycl -O2 -EHsc -I <repo>\sycl\include graph_probe.cpp -Fe:graph_probe.exe`; run with `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`): graphs are healthy on every card and arrangement.** The probe replays the engine's exact graph path (`dpct::experimental::begin_recording` on a dedicated in-order queue, 2,400 `single_task` nodes mirroring the window graph, `end_recording`, `command_graph<executable>(graph->finalize())`, `ext_oneapi_graph`), 100 replays, sentinel verified per replay, every step printed flushed-first:
- a) `ONEAPI_DEVICE_SELECTOR=level_zero:1`, probe 0 (A770 alone): PASS, driver `1.17.39894+4`
- b) selector `level_zero:0;level_zero:1`, probe 1 (A770 at the engine's position, both cards initialized): PASS, driver `1.17.39894+4`
- c) b plus `SYCL_UR_USE_LEVEL_ZERO_V2=0`: PASS (identical; the v2-adapter theory is not confirmed, and this env is NOT in any config)
- d) b, probe 0 (B580 control): PASS, driver `1.17.39894+4`
- extra) selector unset, probe 2: UHD 770 iGPU, driver `1.6.34728` (the i915 runtime identity)
No TDR events during the series (probe logs kept at `C:\strata-deps\gp_a.log` ... `gp_d.log`). Conclusion: plain graphs capture, finalize, and replay on the A770 even with both cards initialized - the engine's problem is in **what its window graphs contain** (their waits/flags/event plumbing) or in the launch machinery around them, not in graph support itself. Next run per the plan: `start-2c-b580.bat`; the offer on the table from the other agent: a commit making eager mode run without a graph at all so the A770 stage needs no graphs (MTP excepted).

Current picture for whoever picks this up. Hard facts: kernels and the CUDA1 stage itself are healthy (window 1 completes in ~111 ms with both cards, graph capture succeeds); every death is at the start of the SECOND window, inside verify.cpp lines 2023-2177 (the drain block at 2027-2041 and the queue submissions after it); every failure ends with a TDR (the A770 still busy); eager produces error 39 then 40 depending on mode; event pools churn 384 at startup + 512 per request with no evidence of reuse. Ranked suspects: (1) event/semaphore churn in the engine's own plumbing - count the pools and check `create_event_pool` / `delete_event_pool` balance in the stage-stamp and staging paths (verify.cpp:2155 and 2407 print the per-window stage table); a driver cap on pools/semaphores fits OUT_OF_RESOURCES exactly (plain graph replay needs none of this plumbing and passes, so the churn is engine-side); (2) the #2194 "window launched behind work in flight on another queue" family, cross-card variant: the drain per verifier may not cover the OTHER device's queues (drain is per device in dpct); (3) the waits/flags the window graphs embed (host-mapped flag reads inside the graph) as the difference between the engine's graphs and the plain probe graph - the probe graph has no waits or flags and survives 100 replays. Prepared runs not yet executed: `start-2c-b580.bat` (no CUDA1 execution at all; if it survives several windows, the fault is CUDA1-side and suspect 2 or 3 becomes primary). Reproduce: `start-2c-nomtp.bat` (no mtp, EAGER, VERIFY_DEBUG, UR_LOG debug, WINDOW_SYNC 1), any ~2k-token prompt, dies at window 2 within ~45 s; logs in the repo root (`strata-iq2_xs-2card*.log`). Crash dumps: `%LOCALAPPDATA%\CrashDumps\strata.exe.*.dmp` (host-side catch-all exits only; the engine's own catch-all at verify.cpp:2454 exits 1, so no WER dump appears for the 39/40 runs).
- `start-2c-b580.bat` / `strata-iq2_xs-2card-b580.json` (not yet run): `--layer-split 18 --split-device 0 --conversation-cache-mib 0` (every stage runs on the B580; graphs on, no `--mtp`). Completes -> the fault is CUDA1-side (the A770 or the cross-card hand-off); times out at layer 18 again -> the split window machinery on the JIT tree itself.

## 11. Regression in `d5b2875`: the A770 answers twice in DXGI

- `stated_free_bytes()` (expert_cache.cpp, the `_WIN32` block) drops `STRATA_VRAM_FREE_MIB` when `dxgi_local_memory()` fails, and `with_adapter_named()` (pinned.dp.cpp) requires exactly ONE adapter whose `Description` equals the SYCL device name.
- **On this PC DXGI enumerates the A770 twice**: adapter 0 `Intel(R) Arc(TM) A770 Graphics` LUID 0:81183457 and adapter 3 the same name, LUID 0:130438 - same DeviceId 0x56A0, same dedicated 15.88 GiB, same WDDM accounting. WMI shows one PCI device (`VEN_8086&DEV_56A0`): the Intel driver registers two WDDM render adapters for the one physical card. So `matches == 2`, not 0: the guard prints "no single adapter named..." and the stated figure is discarded - free 0, zero slots, engine refuses the split.
- Measured with two probes (sources + compile lines at `C:\strata-deps\dxgi_probe.cpp`, `C:\strata-deps\dxgi_l0_probe.cpp`; run the latter with `ONEAPI_DEVICE_SELECTOR=level_zero:1`): after a 2 GiB `sycl::malloc_device` + fill on the A770, **both** duplicates show **usage 2.01 GiB** (budget 15.13 GiB): the shrink mechanism is sound - Level Zero allocations ARE counted, and the duplicates report the same card identically. Only the adapter selection is wrong.
- Same guard also backs `current_device_luid` / `sliced_pin_limit` (pinned.dp.cpp): on the A770 they silently fall back (`sliced_pin_limit` to RAM/2 - 8 GiB).
- **Fix (`agreed_account`, pinned.dp.cpp):**
  - Every adapter with the device's name is read for the segment the caller needs (local for `STRATA_VRAM_FREE_MIB`, non-local for the pinned arena's LUID).
  - If they all report the same budget and usage, the first one is used: the answer is then the same whichever adapter is the card. This covers the A770's duplicate.
  - If they differ, the read is refused (fail closed). One retry covers an allocation landing between two reads.
  - Equal VendorId/DeviceId/memory size was not used: two identical physical cards have those equal too, but their usage differs once the engine uses both.
  - Not measured: whether the two A770 adapters also agree on the non-local (shared) segment. If not, `sliced_pin_limit` falls back to RAM/2 - 8 GiB as before. The pinned device there is CUDA0 (the B580) anyway.
- **First-run check (unchanged):** the "STRATA_VRAM_FREE_MIB 13000 MiB less this process's X GiB" lines, X growing to ~the cache size (section 9).

## 12. Review of the two-card hangs (2026-10-16): graph work on the A770

**One pattern fits all four failures (a reading of the code, not yet tested).** Every death follows a SYCL graph run on CUDA1 (the A770). The B580 runs the same graphs fine on its own.

| Run | Last graph work on the A770 before the failure |
|---|---|
| `--mtp` (2 runs) | MTP prefill: one graph per group (mtp.cpp:1346-1363). The draft layer sits on the last stage. |
| no MTP, graphs | the first decode window's graph. It times out at layer 18, the first A770 layer. |
| no MTP, `STRATA_VERIFY_EAGER=1` (2 runs) | window 1 runs eagerly and completes (111-114 ms). Then **`commit()` still launches the commit graph** (verify.cpp:2631): `STRATA_VERIFY_EAGER` covers only the window (`capture` returns early, `capture_commit` does not). Window 2 dies at its first submission or drain with 39/40. |

So "window 1 clean, window 2 dies" in eager mode fits a graph hang between the windows. Errors 39 and 40 and the event-pool growth would then be what a stuck queue looks like from the next submission. The A770's kernels themselves are fine: eager window 1 ran them, doorbells included.

Checked and ruled out:
- The window drain is per card. `run()` switches to its stage's device (`OnDevice`), the A770 stage defaults to the drain (Alchemist), and the B580 stage's work is done before the A770 stage starts (`next_->run` after the stage's sync).
- `STRATA_VERIFY_STEPPED` is not involved: it is opt-in and off.

**Prime suspect: the Level Zero v2 adapter on the A770's old driver.**
- The trace says "Skipping L0 V1 adapter", so this process runs the v2 adapter, for both cards.
- The A770 is on driver 1.6.34728 (the older runtime); the B580 is on 1.17.
- `docs/INTEL.md` already records a v2-only hang on Alchemist: the prompt stager deadlocked under v2 and ran under `SYCL_UR_USE_LEVEL_ZERO_V2=0`.
- SYCL graphs run through command buffers, the newest part of the adapter.
- One cost of v1: the B580's copy-engine queue is off under v1 (copy_queue.cpp:96). The legacy adapter puts memcpy on the copy engine itself.

**Next, cheapest first (each user-launched; each kernel runs for microseconds; a stuck step means a TDR reset after ~10 s, which the card recovered from every time so far):**
1. `sycl/tools/graph_probe.cpp`: eager kernels, then the same as a graph, replayed. Build it with the icx line of the other probes, then run:
   - a. `ONEAPI_DEVICE_SELECTOR=level_zero:1` → `graph_probe 0`
   - b. `ONEAPI_DEVICE_SELECTOR=level_zero:0;level_zero:1` → `graph_probe 1` (both cards in one context, as the engine has)
   - c. b with `SYCL_UR_USE_LEVEL_ZERO_V2=0`
   - d. b on the B580: `graph_probe 0`, as a control

   If b hangs and c passes, it is the v2 adapter.
   **Executed 2026-10-15 22:4x: a, b, c, d all PASSED, 100 replays of a 2,400-node graph each, sentinel verified per replay, no TDR events.** The source is now `sycl/tools/graph_probe.cpp` (probe logs kept at `C:\strata-deps\gp_[a-d].log`). Two corrections to the suspect list from the run itself: both Arcs report driver `1.17.39894+4` in every arrangement (a, b, c, d), and the `1.6.34728` runtime belongs to the UHD 770 iGPU (extra probe at position 2), not the A770 - the "old driver" half of the prime suspect is void. The v2-adapter half is not confirmed by plain graphs, but plain graphs carry no waits or flags: b passing does not clear v2 for graphs that embed host-mapped flag waits, so the engine run under `SYCL_UR_USE_LEVEL_ZERO_V2=0` with graphs on stays a cheap second datapoint.
2. If the probe confirms it: the engine run `start-2c-nomtp.bat` with `SYCL_UR_USE_LEVEL_ZERO_V2=0` and **without** `STRATA_VERIFY_EAGER` (graphs on). If that decodes several windows, add `--mtp` back.
3. If every probe passes: the trouble is graph *content*, not graphs as such. **This is the branch the probe landed in.** Run `start-2c-b580.bat` next (section 10), and consider an eager commit (`record_commit` out of `capture_commit`, run directly under `STRATA_VERIFY_EAGER`). Then a fully graph-free A770 stage is possible as a test, except for MTP.

**Session pause (2026-10-15 ~22:55): the PC is being restarted to update the GPU driver.** On resume: driver version numbers will have changed, so re-run the cheap `graph_probe` series first (it doubles as the driver fingerprint), re-check the two-card startup lines (`OS budget of 15.13 GiB`, the `less this process's X GiB (DXGI)` shrink figures) against the new driver, then proceed with branch 3: `start-2c-b580.bat`, then eager-commit / graph-free A770 stage work. Open configs stand as-is: base two-card (`strata-iq2_xs-2card.json`, mtp+graphs), nomtp variants (diagnostic), `start-2c-b580.bat`. Branch `windows-sycl` is committed through `de805a2`; everything diagnostic stays git-excluded.

**Resumed after the reboot (2026-10-16): fingerprint re-taken, same verdict.** Display drivers now 32.0.101.9150 (2026-10-07) on both Arcs (UHD 770 stays on 32.0.101.7092); the Level Zero driver string is unchanged (`1.17.39894+4` on both Arcs); probes a, b, d pass again, 100 replays each, no TDR. New tooling gotcha: `icx -fsycl` links `libmmd.dll` (Intel's math library) into any exe by default, and it only lives in `compiler\latest\bin` - a terminal without setvars shows "graph_probe.exe - System Error: libmmd.dll was not found". Fixed by adding `-Xlinker /nodefaultlib:libmmd` to the probe build line (verified: dependency gone, probe still passes). Also: `C:\Program Files (x86)\Intel\oneAPI\2026.1\bin\libmmd.dll` is a 0-byte leftover - harmless unless that dir is on PATH (probe2j.bat includes it); the good copy is `compiler\2026.1\bin\libmmd.dll`; patching it hit a file lock from the crashed probe run, retry when free. Next run stays branch 3: `start-2c-b580.bat`.

**Probe runs are now wrapper-driven.** A second dialog appeared for a bare-terminal run: "graph_probe.exe - System Error: **ur_loader.dll** was not found" - the SYCL runtime's own loader (`compiler\latest\bin\ur_loader.dll`), which no link flag can remove; the probe needs a setvars'd environment. `C:\strata-deps\run_probe.bat` now provides it: modes `a`, `b`, `c`, `d` run the whole matrix with the right selectors and flags, raw mode `<pos> <replays>` also accepted. The probe source (repo copy `sycl/tools/graph_probe.cpp` updated) gained a main-level try/catch that prints the exception and exits 5 instead of dying at -1; two bat-routed matrix runs (b, d) passed 100/100 with it. One earlier bat attempt died right after "create dedicated queue" with exit -1 and no event - likely a transient while the crashed probe's System Error dialog was still holding the L0 loader; it did not reproduce once that dialog was closed.

## 13. The B580 control run passed: the fault follows the A770 side

`start-2c-b580.bat` (23:16 run, log `strata-iq2_xs-2card-b580.log`): same engine, graphs ON, all 48 layers and both stages on the B580 (`layer split: layers 0-17 (CUDA0), 18-47 (CUDA0)`; no A770 line appears at all - the engine never initialized it), expert cache 3468 slots / 4.67 GiB, 2014 MiB free. Six verify windows captured clean, then a full request served: `prompt 3280 tokens ... in 34072 ms (96.3 tok/s), 150 generated in 14376 ms (10.4 tok/s)`, draft acceptance 3/5, decode cache hit 70.2%. No error, no TDR.

The differential is now sharp, all with the same engine build:
- stage/verify work on the A770 (layers 18-47, CUDA1): **always dies** after the first clean window (runs 1-5).
- the identical machinery on the B580 (every stage on CUDA0, A770 untouched): **passes end to end**.

Not yet separated: (i) the A770 device itself (driver/runtime mishandling the engine's graph content on that card) versus (ii) the two-device UR/L0 orchestration (the passing run removed both variables at once - different card AND no cross-device path). `STRATA_VRAM_FREE_MIB` is per-device-index for cards whose driver reports no free memory (`sycl/src/core/expert_cache.cpp:70`), so the single-card A770 control needs `"13000"` alone.

Two controls are staged for the user to launch (both git-excluded):
- **Control C (run next): `start-a770only.bat`** - `strata-iq2_xs-a770only.json`, selector `level_zero:1`, no split, every layer and stage on the A770 alone, graphs ON, no `--mtp`, `STRATA_VRAM_FREE_MIB=13000`. One variable from the passing b580 run: the card. If it dies, the A770 runtime mishandles the engine's graph content (eager-commit / graph-free stage work is the fix path). If it passes, the fault is the cross-device orchestration, not the card.
- **Control B: `start-2c-nomtp-v2off.bat`** - `strata-iq2_xs-2card-nomtp-v2off.json`, the base nomtp split on the A770 with graphs ON (no EAGER) and `SYCL_UR_USE_LEVEL_ZERO_V2=0`, the cheap remaining datapoint for the v2-adapter half of the suspect list.

**Control C result (23:20 run, `strata-iq2_xs-a770only.log`): PASSED.** The A770 alone ran all 48 layers + stage (expert cache 4362 slots / 5.88 GiB, PCIe probe 8.9 GB/s, pcie_frac 0.25), captured four window graphs (4/1/2-token, then a 6-token after the first request), and served two complete requests: `prompt 3330 ... 88.6 tok/s, 287 generated ... 14.0 tok/s`, second `prompt 3336 (3325 reused) ... 20.7 tok/s` decode, cache hits 90.8% then 99.9%. No error, no TDR. **The A770 handles the engine's full graph content fine on its own; the fault is the cross-device orchestration**, not the card. Notable counter-evidence to VRAM pressure: this run logged `0 MiB of VRAM free with everything loaded - LOW` (suggests `--vram-reserve-mib 2048`) yet worked, while the two-card runs had 2077 MiB free and still died.

Looping lines during the run were benign: `get_memory_info: ext_intel_free_memory is not supported.` printed per free-memory query (the xe path on the A770 rejects that L0 extension; B580 supports it, which is why the b580 log had none), and each triggers its `CUDA0 reports no free memory; STRATA_VRAM_FREE_MIB ...` companion line.

New suspect region for the fix work: the per-window cross-device hand-off machinery (`one hand-off per window`, layers 17->18 transfer), the device-1 allocations/event pools in verify staging under a two-device selector, and any UR v2 involvement; `start-2c-nomtp-v2off.bat` (Control B) is still unrun and is the next cheap datapoint. The machinery lives in `sycl/src/core/verify.cpp` (stage hand-off buffers: lines 508, 842, 2414, 3861) and `sycl/src/prefill/prefill.cpp` (two pinned hand-off chunk buffers used in turn: lines 731, 962, 2241, 4424).

## 14. Control B: v2-off also dies - deadlock is in the cross-device wait itself

`start-2c-nomtp-v2off.bat` (23:24 run, log `strata-iq2_xs-2card-nomtp-v2off.log`): startup identical to the healthy two-card shape (B580 5614 slots, A770 5070 slots / 6.92 GiB, 2085 MiB free, CUDA1 PCIe probe 17.3 GB/s pcie_frac 0.48). The first request then stalled in **prefill**: the #29 watchdog fired at `reading the prompt (batched): finishing the chunk from token 2048`; stall report: 0 layers served, 0/0 pool jobs, 11/11 workers parked, verify window never started (`GPU rang 0`, flags all 0), host idle 90 s. The engine wrote a thread-stack dump to `strata-stall-18136.dmp` (repo root, git-excluded) before stopping.

Dump analysis (no debugger on this PC; parser written at `C:\strata-deps\mdstacks.py`, full output `C:\strata-deps\mdstacks_out.txt`, 74 modules / 35 threads / 41 regions):
- The host thread (tid 20340) is the only thread with GPU-path frames: `ntdll+0xa4a04` (syscall wait stub) <- `ur_adapter_level_zero.dll+0x160710/+0x199610/+0x5e89c/+0x36da3` <- `sycl9.dll+0x1b483e/+0x2c3cc9/+0x1be133` <- `ur_loader.dll` <- `strata.exe`. It is parked in a SYCL wait that the **v1** Level Zero adapter never completes - consistent with the prefill hand-off event (`prefill.cpp` chunk machinery, the pinned buffers "used two chunks ago" at line 4424) never being signaled by the CUDA1 side.
- Exactly 11 threads show the uniform CRT worker pattern (`ntdll+0x556e3` <- `KERNELBASE+0x6cb7b` <- strata trampoline <- `KERNELBASE+0x2931b` <- `ucrtbase+0x29333`) - the expert pool, all parked, consistent with the stall report. ~20 more threads sleep in ordinary `ntdll+0x358be/0x5af88` waits (runtime housekeeping).
- The module list proves both adapters exist (`ur_adapter_level_zero.dll` and `ur_adapter_level_zero_v2.dll`); with `SYCL_UR_USE_LEVEL_ZERO_V2=0` the deadlock moved from verify window 2 to the prefill hand-off but did not go away.

Net: **the v2 adapter is not the culprit**. Cross-device signaling between the two Arcs deadlocks under either adapter generation, at whichever point the engine first does a device-0 -> device-1 event wait (verify window 2 with v2, prefill chunk with v1). Prime suspect now: the hand-off event/wait objects themselves across two Level Zero adapters on Windows (same-driver two-card setup), including the pinned host chunk buffers and the `ze_event_pool` usage per stage; the passing single-card runs never create a cross-device wait.

**Reproduction confirmed (23:45 second run, same config): byte-identical failure.** Same stall stage (`finishing the chunk from token 2048`), same idle counts (host idle, all workers parked), new dump `strata-stall-19756.dmp` with the same thread fingerprint as run 1: host thread parked at `ntdll+0xa4a04` over the identical v1-adapter frame sequence (`ur_adapter_level_zero.dll+0x160710/+0x199610/+0x5e89c/+0x36da3` under `sycl9.dll+0x1b483e/+0x2c3cc9/+0x1be133` via `ur_loader.dll`), 11 workers in the same CRT wait. No TDR events for either stall - this is a pure deadlock, not a GPU hang. The `0xc0000409` application-error events at both run ends are the watchdog's own fast-fail exit after writing the dump (`ucrtbase.dll+0x7f6fe`), not an independent fault. The point of failure is deterministic at the first prefill hand-off crossing (chunk 2048 = first chunk that crosses to CUDA1); run 2 also mapped more host memory before stalling (6255 vs 1678 MiB resident, 3.98M vs 1.41M page faults) without changing the outcome.

## 15. Review of sections 12-14 (2026-10-16): per-card choices made once per process

**Section 12's graph theory is withdrawn.** The probes and Controls B/C rule it out. The differential in section 13 is the useful result: the same machinery passes on either card alone and fails only with both cards.

**What every earlier two-card test had in common: identical cards** (2x B60, 2x B70). This pair mixes generations: a Battlemage B580 and an Alchemist A770. The SYCL port has several workarounds per generation, and some are decided in a function-level `static` with the device of whichever card asks first. In a split that is the B580 (stage 0), and its choice then applies to the A770 too:

| Where | Choice | What the A770 got in the split |
|---|---|---|
| `prefill/gemm.dp.cpp` `no_stage` | stage GEMM operands past 4 GiB (the A770 oneMKL bug, docs/INTEL.md: non-finite prompts, "routed id out of range", segfaults) | **no staging** |
| `core/verify.cpp` `wmode` | drain before every window (#2194, the A770 window that never starts) | **no drain**, unless `STRATA_WINDOW_SYNC=1` was set (it was in the 22:37 run only) |
| `kernels/cuda/fused_gr.dp.cpp` `gr_large_grf`, `gr_more_groups` | large GRF / more work-groups | Battlemage defaults (speed only) |
| `prefill/kernels.dp.cpp` `gdn_xe_on` | the GDN Xe variant | Battlemage default (speed only, same bits) |

Checked and fine:
- `spin_max` and the copy engine are per device.
- `doorbell_plain_payload` reads sysfs, which gives the same answer for both cards on Windows.
- The `generate.cpp:2256` borrow rule applies only to single-card runs.
- dpct's current device is per thread.
- The hand-off buffers are host USM, with no P2P.

**Fixed in this commit:** `strata::intel_gpu_gen_of(device)` (sycl_queue.hpp) caches the generation per device, and the four places above ask it with their own queue's device. The `STRATA_*` env overrides still apply to every card. Runs on one card, or on two cards of one generation, make exactly the choices they made before. Not built here (no oneAPI in this container).

Whether this is *the* fault is not known.
- The GEMM staging matters only where an operand lies past 4 GiB into its allocation. In a split the prompt path keeps its own buffers (`max_context <= 32768`), so it depends on where the dense weights and the cache slots sit on the A770 (6.9 GiB cache).
- If the A770 stage computes garbage, the hangs follow from it: garbage routing makes doorbell plans the CPU pool cannot serve. That is the documented failure family of the 4 GiB bug.

**Two confounds to keep in mind when comparing runs:**
1. **Prompt length chooses the cross-card mechanism.**
   - A prompt of one chunk (≤ 2048 tokens here, e.g. the 1,960-token runs) runs the stages one after the other. The A770 helps the B580's stage as a stream-only peer, with queue barriers on events of the other card (`prefill.cpp` `bind_stage_helper`; the helper share is `STRATA_PREFILL_HELP_FRAC`).
   - A longer prompt (Control B) runs stage 1 on a `std::async` thread **concurrently** with stage 0's next chunk (`prefill.cpp:4457`). That is the path that stalled.

   So the v2 runs and the v1 run (Control B) did not differ only in the adapter. The v1 result does not clear or convict v2.
2. **`std::async` on MSVC runs on a reused thread pool.** On Linux, libstdc++ starts a new thread each time. `thread_local` state (dpct's device stack, `gemm_stage`'s scratch) persists across tasks on Windows. No harm found, but it is a Windows-only difference on exactly the stalled path.

**Runs, in order (user-launched; `start-2c-nomtp.bat` base, graphs on, no `STRATA_VERIFY_EAGER`, no `SYCL_UR_USE_LEVEL_ZERO_V2` override):**
1. **Rebuilt engine, same ~3.3K-token prompt as Control B.** If it decodes several windows, the per-card choices were the fault; then add `--mtp` back.
2. **If 1 stalls:** add `--prefill 4096` and `STRATA_PREFILL_HELP_FRAC=0`. The prompt is then one chunk, stage 1 runs after stage 0, and no queue waits on the other card's events.
   - It passes: the concurrent stage thread or the cross-card waits are the fault.
   - It stalls the same way: the fault is in the A770 stage's own work while both cards are initialized.
3. **For any stall dump:** say which thread holds the SYCL frames. The main thread waiting on `next_run_` would sit in an ordinary wait, and the stage-1 task thread would sit in the SYCL wait. That tells whether the B580 hand-off memcpy or the A770 stage is stuck.

Latent, not on the engine's paths as far as traced: `kernels/cuda/qsa.dp.cpp` `step_scratch()` is one device buffer per process, allocated on the first caller's card. It serves the legacy direct-launch `kv_append` / `qsa_index` / `topk_512`, not the captured ones.

**Run 1 result (00:18, `strata-iq2_xs-2card-nomtp-clean.log`, rebuilt engine at `a774ac5`): stalled, with a new signature.** Startup healthy as ever (CUDA1 stage 5063 slots / 6.91 GiB, 2077 MiB free). The request captured two 4-token windows clean, then `verify: timed out at layer 18; its GPU waits were released but the GPU did not finish within 5 s (#267)`, and the system log shows a TDR (`Display #4101`) at 00:24:53 - unlike Control B's pure deadlock, this time the GPU itself hung. So the per-device fixes did not by themselves make the split run; the failure moved back to the original Alchemist family (the verify window stuck at the CUDA1 stage's first layer) but **with the drain now enabled for the A770 by construction** (`wmode` asks each stage's own device). No stall dump for this run (the #267 timeout path exits without one). Run 2 is staged: `start-2c-nomtp-p4096.bat` (`--prefill 4096`, `STRATA_PREFILL_HELP_FRAC=0`, everything else identical).

**Run 3 result (00:38, `strata-iq2_xs-2card-nomtp-p4096-eager.log`): a third signature, same family.** Run 2 + `STRATA_VERIFY_EAGER=1`. The single-chunk prefill crossed the split fine (no stall, no watchdog), then the first verify window threw `level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)` out of `Verifier::run` (caught at the `verify.cpp:2457` handler, `std::exit(1)`), with a TDR (`#4101` at 00:39:43). Pre-fix, the 22:37 eager run showed the same error 40 at window 2 - the `no_stage` fix did not remove it. So post-fix the split fails in **every** verify mode: captured graph hangs the A770 at layer 18 (runs 1-2, TDR), eager throws resource exhaustion (run 3, TDR), all while the identical card+mode passes alone (Control C) and plain dual-card graph replays pass (probe d). Prefill hand-off with a single chunk is NOT implicated (run 3 cleared it). New candidate: the A770 stage runs with its VRAM essentially full (`0.00 GiB free`, 8.1+ GiB of this process's allocations on it per DXGI) - OUT_OF_RESOURCES + TDR under a second active device context suggests a WDDM-level resource pool per card per process. Run 4 is staged to test the pressure hypothesis: `start-2c-nomtp-p4096-lowfree.bat` = run 2 config with `STRATA_VRAM_FREE_MIB=0,7000` (A770 cache shrinks ~1.5 GiB, several GiB slack). Passes: pressure is the trigger and the split sizing must account for the dual-device overhead. Stalls/throws the same: it is not a capacity question, and the suspect is the concurrent second device context itself.

**Run 4 attempt 1 (00:45, `0,7000`): the engine refused the split (not the fault - sizing did its job).** `CUDA1 ... 4.27 GiB free` then `room for experts 0.00 GiB` / `CUDA1 expert cache: no room`. Two findings from it: (1) the sizing floor for a CUDA1 stage is above 4.27 GiB minus the reserve, so the slack control must stay near the working figure - the config is now `0,11000` (expected room ~3.6 GiB, still several GiB of real slack on the card); (2) **a new engine bug for the fix list**: on the `no room` path `strata.exe` does not exit cleanly - it crashes with an access violation `0xc0000005` in `ur_loader.dll` (Application Error #1000 at 00:45:46). Worth tracing whatever device/queue teardown the "no room" path performs through the UR loader.

**Run 4 attempt 2 (00:48, `0,11000`, `strata-iq2_xs-2card-nomtp-p4096-lowfree.log`): stalled identically - VRAM pressure ruled out.** The A770 stage took only 2656 slots / 3.63 GiB (vs 4086 / 5.58 in run 2), leaving ~8 GiB of real slack on the card, and the end was the same: two windows captured, `timed out at layer 18`, TDR `#4101` at 00:50:14. The full post-fix elimination list: card (C), graph support (probes), adapter generation (v1/v2), the per-device statics (fixed, rebuilt - did not fix), concurrent prefill stage thread + cross-card prefill waits (run 2), prefill hand-off in general (run 3 crossed it in eager), eager vs captured verify (both fail), VRAM pressure on the A770 (run 4). What survives: **the CUDA1-stage verify execution in a two-device process** (captured: hang at layer 18 + TDR; eager: error 40 + TDR). **Run 5 attempt (00:53, swapped selector): the swap is not reachable through env knobs on this PC - use an engine knob instead.** `ONEAPI_DEVICE_SELECTOR=level_zero:1;level_zero:0` does NOT reorder (docs confirm: the selector filters by index, order comes from the adapter); the engine still got `GPU 0: B580`, so `STRATA_VRAM_FREE_MIB=11000,0` starved the A770 at index 1 - `no room` refusal + the known `ur_loader.dll` crash (second reproduction of that defect). Probe `C:\strata-deps\sycl_dev_list.cpp` (built with the libmmd-free recipe) mapped the enumeration: baseline UR order is [B580, A770, UHD770] per adapter, and the Arc pair appears TWICE in the raw list (v1 and v2 adapters both expose them - matches the stall dumps loading both adapter DLLs). `ZE_AFFINITY_MASK` indexes the **native ze order, which is reversed vs the UR indices**: mask `0` keeps [A770, UHD770], mask `1` keeps [B580]. So: UR index 0 = B580 while ze device 0 = A770. Any code path that mixes the two orders on this PC would target the wrong card - worth checking in the fix work. To actually run the role swap, the engine needs a small knob (stage 0's device is hardwired to the first visible one; `--split-device` only assigns the later stages) - or physically swap the PCIe slots. Run 5 therefore goes to the fix agent's plate along with: role-swap result still OPEN (role vs card not yet separated).

## 16. Review of runs 1-5 (2026-10-16): one SYCL context per card

**The surviving symptom points at memory the A770 reads but does not own.** After runs 1-4, only the CUDA1-stage verify work in a two-device process still fails. Captured: hang at the stage's first layer + TDR. Eager: error 40 + TDR. The same work passes on either card alone.

**Cause found in the vendored dpct (`sycl/include/dpct/device.hpp`):**
- `device_ext` builds **its own one-device `sycl::context`** (`_ctx(*this)`), so every card's queues live in a separate context.
- Host USM belongs to the context it was allocated in. The engine allocates buffers that both cards read through one card's queue (the current device at allocation time, normally CUDA0):
  - the verify stages' hand-off rows (`generate.cpp`, `sycl::malloc_host(hb, dpct::get_in_order_queue())`; the A770 stage reads them as its first layer's input);
  - the expert mirror chunks the copy kernels read (`gguf_expert_source.cpp`; the A770 stage's `pcie_frac 0.48` reads them);
  - the prefill hand-off.
- SYCL does not guarantee such a pointer on another context's device.
  - Linux evidently lets the other card read it, so the Linux two-card runs worked.
  - Under WDDM, residency is per device. That fits an A770 that hangs on its first layer's reads.
  - A `memcpy` from such memory still works (the driver treats it as plain host memory). That is why the prefill hand-offs, which are copies, mostly crossed fine, and the kernels that read the memory in place did not.
- **The graph probe could not catch this:** it built one context for both cards (`sycl::context{devs}`), unlike the engine.

Checked and ruled out:
- No engine code mixes the UR and native Level Zero device orders (no `zeDeviceGet`; `copy_queue.cpp` takes native handles from the SYCL device).
- The only queue built outside dpct (`copy_queue.cpp`) uses dpct's context.

**Fix in this commit:**
- `dev_mgr` builds **one context for all GPUs of a platform** and gives it to each `device_ext`.
- Default: on for Windows, off elsewhere, so Linux is unchanged. `STRATA_SYCL_SHARED_CONTEXT=1` or `=0` overrides.
- With one GPU visible (the single-card configs) the context holds the same one device as before.
- If the runtime refuses the shared context, the devices keep their own as before.
- Not built here (no oneAPI in this container).

**Run 6 (user-launched): the run-2 config** (`start-2c-nomtp-p4096.bat`: graphs on, `--prefill 4096`, `STRATA_PREFILL_HELP_FRAC=0`) **on the rebuilt engine.**
- Watch Task Manager as before, but expect one change: with a shared context the host USM buffers can be mapped for both cards, so **"shared GPU memory" may now rise on the A770 too**. That is pinned system RAM made visible to the card, not a VRAM overcommit.
- Judge it against **dedicated** memory and Windows RAM. If shared memory rises well past ~0.2 GB and you are unsure, stop and record the figures; that is information too.

Reading the result:
- **Decodes several windows:** the context was the fault. Then the default config: drop `--prefill 4096` and `STRATA_PREFILL_HELP_FRAC=0`, then `--mtp`, then speeds for `--layer-split 18 / 14 / 22`.
- **Same failure:** re-run with `STRATA_SYCL_SHARED_CONTEXT=0`. If the log is identical, the fix did not take effect; check the build. If it differs, compare.
- The role-swap knob (section 15, run 5) is only worth building if run 6 fails.

**Still open, separate:** the `ur_loader.dll` access violation on the "no room" exit path (two reproductions). It is likely teardown order: queues and contexts destroyed after the UR loader unloads, at static destruction. Not looked at yet.

**Section 16 probes (01:0x-01:2x, `C:\strata-deps\dpct_ctx_probe.cpp` and `C:\strata-deps\cross_usm_probe.cpp`, built with the libmmd-free recipe). Run 6 (01:09, rebuilt engine at `97dbfd8`) was identical because the shared-context fix is a NO-OP on this PC - and the probes show why, plus reproduce the fault family in two minutes:**
1. Under UR on Windows **every GPU is its own SYCL platform** (raw platforms 0/1/2 all named "Intel(R) oneAPI Unified Runtime over Level-Zero V2", one device each - the WDDM per-device driver-handle behavior). `dev_mgr`'s platform-equality grouping never fires; the probe confirms the built engine gives `A770 ctx != B580 ctx`. `STRATA_SYCL_SHARED_CONTEXT` therefore has no observable effect here - worth a log line or comment so the next investigator is not misled.
2. **T1: `sycl::context{both devices}` is refused** - "Can't add devices across platforms to a single context". A one-context fix cannot work under Windows UR (the fallback branch is what runs = old behavior). `SYCL_EXT_ONEAPI_USM_IMPORT` is undefined in the 2026.1 headers, so `import_mem` is out too.
3. **T3 reproduces the engine's failure family with no model**: `sycl::malloc_host` through the B580 queue, a trivial kernel on the A770 (own context) reads it in place - the kernel COMPLETES but **4095 of 4096 values are wrong**. Silent cross-context data corruption (the second card reads through its own mapping). In the engine that garbage flows into doorbell/spin kernels - which is how it becomes hang + TDR (captured) or error 40 (eager).
4. **T5 shows the fix direction**: same test, but the A770 first `memcpy`s the foreign host buffer into its OWN device buffer, then reads in place - **0 of 4096 wrong**. Cross-context COPIES work (matching the memcpy observation); in-place foreign-context host reads silently corrupt.

Real-fix candidates (your plate): make the cross-card shared buffers per-consumer - a staging copy on the reading card (T5 pattern: verify hand-off rows, expert mirror reads, prefill hand-off at kernel level), or allocate each shared host buffer through the READING card's queue (per-stage `malloc_host` mirrors). Cross-platform shared contexts are not achievable. Run 7 only after a real fix lands.

## 17. Review of the section 16 probes (2026-10-16): no shared context; one buffer per card instead

The probes are decisive and correct the earlier fix.
- Under UR on Windows every GPU is its own platform, and a context cannot span platforms (T1). `97dbfd8` was a no-op here and is **reverted** in this commit: `sycl/include/dpct/device.hpp` is back to the vendored file, and `STRATA_SYCL_SHARED_CONTEXT` is gone.
- T3 / T5 settle the mechanism:
  - a kernel reading another context's host USM in place gets wrong data with no error;
  - a `memcpy` from it is right;
  - (by the same logic) a card's kernels on its own context's host USM are right, and so is the CPU on any host memory.

**Every place a card's kernel touched another context's host USM in a B580 + A770 split, and what it does now:**

| Path | Before | Now (only when the two cards' contexts differ) |
|---|---|---|
| Verify hand-off rows (`copy_from_mapped` kernels: stage k writes, stage k+1 reads, one shared `malloc_host` buffer) | one side always foreign | one buffer per side, each allocated through its own card; the host `memcpy`s stage k's rows into stage k+1's buffer after stage k's window synced (`Verifier::forward_handoff`, before `next_->run` / `run_slot_rows`) |
| The pool's PCIe share on a later stage (copy kernels read mirror / arena blobs allocated through CUDA0) | A770 kernels read CUDA0-context host memory | that stage's `pcie_num` is 0: the CPU computes its misses (`STRATA_SPLIT_PCIE_FOREIGN=1` keeps the reads, for testing) |
| Prefill stage helper (the next card computes a share through mapped host rows) | both cards read in place | not bound across contexts: each stage reads its prompts alone |
| `--pipeline-windows`, `--batch-groups` > 1 (the caller drives the hand-offs) | same shared buffers | refused with a message for now |

Not affected, checked:
- The prefill hand-off is a `memcpy`.
- The CUDA0 device-plan mirror table is CUDA0's only (`in_mirror_table`).
- Cache fills and adaptive swaps are copies.
- Each verifier's flags, plans and pool rows are allocated under its own device.

**"Contexts differ"** is `strata::cross_context(dev, owner)` (sycl_queue.hpp). It is true only on Windows, for two devices whose dpct contexts differ; `STRATA_SPLIT_CROSS_CONTEXT=1` / `=0` overrides. So:
- Linux keeps its in-place hand-offs and PCIe reads.
- One card, or `--split-device 0`, is unchanged.

Not built here (no oneAPI in this container).

**Cost:** the A770 stage's missed experts go to the CPU instead of PCIe, so expect lower decode than a split with PCIe reads. If the split now runs, the follow-up is to allocate each stage's share of the mirror / arena through that stage's own card and turn its PCIe share back on.

**Run 7 (user-launched):** rebuild, then `start-2c-nomtp-p4096.bat` unchanged.
- Expect the new lines at start:
  - "CUDA0 and CUDA1 are in different SYCL contexts; the hand-off rows are copied ...";
  - "CUDA1 ... reads no experts over PCIe";
  - "stage 0 reads its prompts alone ...".
- **If it decodes several windows:** drop `--prefill 4096` / `STRATA_PREFILL_HELP_FRAC=0` (the helper is off across contexts anyway), then `--mtp`, then speeds.
- **If it still fails:** `STRATA_SPLIT_CROSS_CONTEXT=0` must reproduce the old failure exactly (proof the fix took effect). Then look for another in-place cross-context access; the T3 probe is the quick check for any suspect buffer.

**Run 7 result (01:26, `strata-iq2_xs-2card-nomtp-p4096.log`, rebuilt at `f4fc1e3`): PASSED - the first full request across the B580 + A770 split.** All three new lines printed at start (`stage 0 / stage 1 read their prompts alone`, `CUDA1 ... reads no experts over PCIe (the CPU computes its misses)`, `CUDA0 and CUDA1 are in different SYCL contexts; the hand-off rows are copied between their own buffers on the host`). The request captured six windows (4/4, 1/1, 3/3 tokens, all "upload no error, sync no error") and completed: **prompt 3384 tokens at 182.3 tok/s, 44 generated at 10.5 tok/s (drafts accepted 2 of 2), decode cache hit rate 80.2%, no TDR, no stall dump, no application errors.** Compare: single-card prompt was 88.6 tok/s (A770 alone) / 96.3 (B580 `--split-device 0`) - the split prefill is ~2x either card alone; decode 10.5 vs 14.0 (A770 alone) matches the expected cost of the A770 stage's expert misses going to the CPU. **The cross-context mechanism (section 16) is confirmed as THE fault, and `f4fc1e3` is the cure.** Next per this section's ladder: drop `--prefill 4096` / `STRATA_PREFILL_HELP_FRAC=0` (clean config, multi-chunk prompt through the copied hand-offs), add `--mtp`, then the speed ladder for splits 18/14/22; the decode recovery is the per-stage mirror/arena allocation so the PCIe share can come back on.

**Run 8 (clean config, `start-2c-nomtp-clean.bat`, log `strata-iq2_xs-2card-nomtp-clean.log`): PASSED - multi-chunk prefill through the copied hand-offs works.** Same four cross-context lines; six windows captured; **prompt 3402 tokens at 122.6 tok/s, 32 generated at 13.9 tok/s (drafts 2 of 2), cache hit 82.1%, no TDR.** Note the shape: with `--prefill auto` (multi-chunk, stages running their prompts alone since helpers are not bound across contexts) prompt is 122.6 vs 182.3 with the single 4096-chunk, and decode recovered to 13.9 (vs the A770-alone 14.0). Run 9 staged: `start-2c-mtp-clean.bat` = the clean config + `--mtp C:\strata-data\mtp\rt` + `STRATA_MTP_PREFILL_SYNC=1` (the original mtp wiring) - the first mtp attempt on the fixed engine.

**Run 9 (01:41, `start-2c-mtp-clean.bat`, log `strata-iq2_xs-2card-mtp-clean.log`): FAILED in the mtp warmup - the only path the fix has not reached.** Startup was fully healthy (all four cross-context lines, draft layer 839 MiB loaded, draft head over 106299 tokens 137.9 MiB, 2077 MiB free), then: `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` printed between the draft head line and `strata serve: mtp prefill: no error`, and one second later TDR #4101 and the process fast-failed (`0xc0000409` at `ucrtbase.dll+0x7f6fe`, the usual abort path; no new stall dump). So: every non-mtp split path now runs (runs 7-8), and the death has moved exactly to the mtp prefill sync - the phase warmup spends the draft head. Suspect, by the T3 pattern: the draft head buffer (and the hidden rows the draft layer consumes) crossed contexts - allocated through CUDA0's in-order queue while the stage-1 (A770) kernels write its rows, or read by the draft graph on the other card. Worth checking in `mtp` warmup code which queue allocates the draft head and the draft-layer hidden-row buffers, and which card's kernels touch them; the same per-card-buffer pattern as `set_handoff_copy` should apply. Isolation staged for the user: run 9 with `STRATA_MTP_PREFILL_SYNC=0` (`start-2c-mtp-clean-nosync.bat`); a single-card control is `--split-device 0` with `--mtp` (mtp ran on one card before all this, so a pass there confirms the fault is the cross-context mtp buffer, not mtp itself).
