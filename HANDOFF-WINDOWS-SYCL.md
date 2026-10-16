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
     - Without a single DXGI adapter of the device's name (two identical cards, or no DXGI), the variable is not used for that card, and the log says so. The pinned arena's LUID lookup needs a unique name now too, instead of taking the first match.
     - Anything but numbers, spaces and commas disables the variable with one message. `0;13000` used to hang the parser and fill RAM.
     - The variable is Windows-only now; Linux keeps its DRM-fdinfo fallback.
     - The split search uses `device_free_bytes()` only when the driver reports 0, so Linux splits size as before.
     - **To check on the first run:** the lines "CUDA1 reports no free memory; STRATA_VRAM_FREE_MIB 13000 MiB less this process's X GiB on it (DXGI)", one per read. X must grow once the A770's cache is allocated (to about the cache size in the log, ~9.5 GiB). If it stays near 0, DXGI does not count Level Zero allocations: stop, and do not trust the figure.
3. **`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` is required on the A770.** Without it `sycl::malloc_device` past 4 GiB returns aliased pages: writes land before 4 GiB, reads past it come back wrong - deterministic across runs (test matrix: `C:\strata-deps\device1_probe*.cpp`, 13 GiB allocations, host-written tags). With the env var plus JIT kernels the matrix reads back clean 2/2. The engine's arena alias guard (`STRATA_ARENA_ALIAS_CHECK`) detects the aliasing but cannot fix it. Set in the 2-card config/bat.
4. **AOT for dg2 is a dead end past 4 GiB addressing** (finding, not committed). A dual-target AOT build (`-device bmg-g21,dg2`) compiles ("Build succeeded for : dg2") and its images launch, but kernels AOT-targeted at dg2 cannot address past 4 GiB of any allocation - host-written tags read correctly to +4095 MiB and everything past +4096 MiB reads wrong - even with relaxed allocation limits. Only usable with arenas under 4 GiB; the JIT tree is the regime for the A770 (the `STRATA_SYCL_JIT_STATELESS` comment above is the same 4 GiB surface-offset trap, approached from the other side).

Tree and files: `build-sycl-jit` = the `build-sycl-aot` recipe with `-DSTRATA_SYCL_AOT=` empty, `-DSTRATA_SYCL_PARITY=OFF`, `-DSTRATA_SYCL_JIT_STATELESS=ON` (default). Startup probe: `C:\strata-deps\probe2j.bat` (bounded, load + verify only). Git-excluded locals: `strata-iq2_xs-2card.json`, `start-strata-2card.bat`, their logs.

Unchanged open work for whoever continues: measure prompt/decode t/s for `--layer-split 18` (then 14 and 22), the "too few cache slots to borrow" prompt path, fix-list items 3 (tail-slack OOB read) and 4 (`fd_pread` overlapped I/O), item 6's single-card config cleanup, then the prompt-speed steps of section 7.

## 10. Status of the two-card launch (2026-10-16, after `d5b2875`)

Section 9's startup success no longer holds on `d5b2875`: the rebuilt engine exits at ~20 s with `layer split, CUDA1: 0.00 GiB free of 15.56, room for experts 0.00 GiB` then `layer split, CUDA1 expert cache: no room` (B580 stages normally, 5607 slots). **Do not launch `start-strata-2card.bat` until section 11 is fixed.** The build itself is clean - `pinned.dp.cpp` (DXGI code) and `expert_cache.cpp` compile without errors (only the usual dpct unused-function warnings); the failure is runtime, and the config/env are correct as listed.

## 11. Regression in `d5b2875`: the A770 answers twice in DXGI

- `stated_free_bytes()` (expert_cache.cpp, the `_WIN32` block) drops `STRATA_VRAM_FREE_MIB` when `dxgi_local_memory()` fails, and `with_adapter_named()` (pinned.dp.cpp) requires exactly ONE adapter whose `Description` equals the SYCL device name.
- **On this PC DXGI enumerates the A770 twice**: adapter 0 `Intel(R) Arc(TM) A770 Graphics` LUID 0:81183457 and adapter 3 the same name, LUID 0:130438 - same DeviceId 0x56A0, same dedicated 15.88 GiB, same WDDM accounting. WMI shows one PCI device (`VEN_8086&DEV_56A0`): the Intel driver registers two WDDM render adapters for the one physical card. So `matches == 2`, not 0: the guard prints "no single adapter named..." and the stated figure is discarded - free 0, zero slots, engine refuses the split.
- Measured with two probes (sources + compile lines at `C:\strata-deps\dxgi_probe.cpp`, `C:\strata-deps\dxgi_l0_probe.cpp`; run the latter with `ONEAPI_DEVICE_SELECTOR=level_zero:1`): after a 2 GiB `sycl::malloc_device` + fill on the A770, **both** duplicates show **usage 2.01 GiB** (budget 15.13 GiB): the shrink mechanism is sound - Level Zero allocations ARE counted, and the duplicates report the same card identically. Only the adapter selection is wrong.
- Same guard also backs `current_device_luid` / `sliced_pin_limit` (pinned.dp.cpp): on the A770 they silently fall back (`sliced_pin_limit` to RAM/2 - 8 GiB).
- Direction for the fix (not applied here): among same-name matches, treat duplicates as the same card when their accounts agree (equal VendorId/DeviceId/dedicated memory - here byte-for-byte) and use the first; only genuinely differing same-name adapters (two identical physical cards) stay ambiguous - say so in the comment, as section 7 item 5 already asked. With the selection fixed, the first run must show the "STRATA_VRAM_FREE_MIB 13000 MiB less this process's X GiB" lines, X growing to ~the cache size - section 9's first-run check.
