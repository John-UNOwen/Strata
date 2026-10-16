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
| `4857f1d`, `2f205be` | "no free-memory report" fallbacks in `expert_cache.cpp` / `remote_experts.cpp` (Windows only). With `--expert-cache-device1 auto`, `remote_experts` shrinks the tier to fit, as the CUDA engine does; an explicit slot count that does not fit still refuses |
| review fix | `/MAP` gated on `WIN32`; the `remote_experts` fallback gated on `_WIN32` and the shrink gated on `auto`, so the Linux and explicit-count paths match upstream again |

## 6. Fix list (do these first)

1. ~~`LINKER:/MAP` outside `if(WIN32)` broke the Linux link.~~ Fixed: it is inside `if(WIN32)` now.
2. **The free-memory fallbacks (`4857f1d`, `2f205be`) assume the card is empty** (`total - 1 GiB`, `total - 2 GiB`). On this PC that is how VRAM gets overcommitted: the desktop sits on the A770, and VicGPT's server may still be running. Windows Level Zero **does** report free memory here (the log says "6.80 GiB free"), so these paths may never be needed.
   - Prefer: drop them.
   - Or: require an explicit per-card env figure. The WSL experiment used `STRATA_WSL_FREE_MIB="11000,14000"` for this.
   - This matters most for the two-card split (step 2): on the A770, which drives the desktop (~0.6-1 GB used), the fallback would assume 14 of 16 GiB free.
   - Still open. Both fallbacks now run on Windows only.
3. **The `+4096` tail slack hides a real out-of-bounds read.** Host code (the reference gemv, the pool kernels) reads past the end of these buffers. Linux has the same bug; it is just masked there. Find the reader and bound it, or at least file it upstream.
4. **`fd_pread` serializes.** `CreateFileA` is called without `FILE_FLAG_OVERLAPPED`, so Windows serializes I/O on the handle, and the "several threads pread one handle" design (mirror pool, prompt stagers) runs one read at a time.
   - Open with `FILE_FLAG_OVERLAPPED` and wait per call (event in the OVERLAPPED, or `GetOverlappedResult`), or open one handle per thread.
   - Check the effect on prompt speed.
   - Fix the comment above `fd_open` too (it says several threads can read one handle at once), and clamp `n` to `DWORD` per call (a read over 4 GiB would be cut short).
5. **AOT `bmg-g21` only.** The A770 (Alchemist/DG2) can't run this binary. A two-card split needs a JIT build (`-DSTRATA_SYCL_AOT=` empty), or a `bmg-g21` + `dg2` AOT target set if the toolchain allows it.
6. **Config:**
   - `--max-context 65536` in `strata-iq2_xs-win.json` is untested (the last run used 32768, with ~2.3 GB free after load). Go back to 32768 until measured.
   - Remove `STRATA_SYCL_RESIDENT_EXPERTS=0`: nothing in the repo reads it.
   - `STRATA_NO_NORM_ROPE=1` turns off the fused norm+rope (`sycl/src/kernels/cuda/native_rope.dp.cpp`). If the B580 needs it for correct output, that is a parity failure: run `rope_parity` (check 6) on the B580 and record the result.
7. **`start-strata.bat`:**
   - The comment's stop signal is wrong. It is shared GPU memory above ~0.2 GB, or dedicated near ~11.3 GB; not "shared past ~11.3 GB".
   - It runs `python` from PATH. Pin a venv, or check the required packages at startup.

## 7. Next development steps (in order)

1. **Fix list items 2 and 6** (safety and correctness), then commit. Item 1 is done.
2. **Two-card split, A770 + B580.** This is the main lever: it gives ~26 GB of VRAM instead of ~11, so many more experts on the GPU.
   - Build with JIT (item 5) and set `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` with `--layer-split auto`.
   - Under WSL+OpenCL the auto split judged the A770 much faster (64 vs 20 compute units). It gave the B580 only layers 0-1 and left 15,781 experts host-served.
   - Try explicit `--layer-split N` values. Watch the log lines "layer split, CUDA1 ... room for experts", "experts of the later stages are not in VRAM", and the hit rate.
   - Expect the A770 path to be slower per expert: it has no ahead-of-time kernels and no coopmat.
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
