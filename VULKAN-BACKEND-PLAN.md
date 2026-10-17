# Vulkan backend for Strata on Intel Arc (Windows): work plan (2026-10-17)

Goal: a Vulkan engine for the B580 + A770 PC that keeps Strata's memory design (hot-expert cache in VRAM, CPU pool
for misses, layer split across the two cards) and is correct where the SYCL / Level Zero engine is not: more than
26 layers on the A770.

Status: plan only. Nothing here is built. Each phase ends at a gate; stop and report at a failed gate instead of
working around it.

## Reviewed against the repo and this PC (2026-10-17)

Branch `vulkan-backend` cut from `main` at `3364fdf`. What the review found, with the file each fact came from:

- **Vulkan already works on both cards.** `vulkaninfo --summary`: A770 = GPU0 (`apiVersion 1.4.359`, `driverVersion
  101.9150`, `conformanceVersion 1.4.2.0`), B580 = GPU2 (same), UHD 770 = GPU1 (integrated, `1.4.323 / 101.7092`, the
  one to skip). The loader `C:\Windows\System32\vulkan-1.dll` is there. So P1 is half answered before any code exists:
  both Arcs enumerate in one instance, and API 1.4 is available where this plan targets 1.3.
- **One driver package behind both APIs.** `Win32_VideoController` reports B580 and A770 both at `32.0.101.9150`
  (2026-10-07). The Level Zero front end of that same package reported `1.17.39894+4` in the graph probes
  (`HANDOFF-WINDOWS-SYCL.md` section 12). So the bet is not "a different driver", it is "the same driver's other front
  end". That promotes P2 from a nice-to-have to the decisive probe: the known SYCL failure is an allocation that comes
  back **aliased past 4 GiB** (section 9.3, `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS`), and P2 asks exactly that
  question through Vulkan. If P2 passes, the >26-layer fault is not a general driver fault.
- **No Vulkan build toolchain exists on this PC.** No `C:\VulkanSDK`, no `VULKAN_SDK`, no `vulkan.h` in the Windows Kits
  or Visual Studio trees, no `glslc`, no `glslangValidator`, no `dxc` on PATH. `dxc.exe` does exist at
  `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe`. 72 GB free on C:. This is a gate of its own,
  below as "Phase 0 preconditions".
- **`third_party/ggml` does not carry ggml-vulkan, or ggml.** That directory holds `ggml-common.h` (134 KiB), `LICENSE`
  and `VERSION.txt` (commit `3cf0325`, 2026-09-20) and nothing else. Phase 3's starting point is therefore a download of
  `ggml/src/ggml-vulkan` (MIT, ask first, pin the commit), not something already in the tree.
- **The llama.cpp evidence, with citable files.** VicGPT's auto-fit run,
  `VicGPT/logs/vulkan-flashnext-auto_2026-10-16_235849.log.stderr` (321 lines), records `Vulkan0 (A770): 36 layers (22
  overflowing), 13802 MiB used, 1623 MiB free`, `offloaded 49/49 layers to GPU`, `Vulkan0 model buffer 13190 MiB`,
  `Vulkan2 (B580) model buffer 9272 MiB`, and two `CPU_Mapped` totals of 37,034 and 27,466 MiB, which is the 64.5 GiB
  mapped on 32 GB of RAM. Its output artifact is `VicGPT/benchmarks/vulkan-flashnext-auto_2026-10-17_000222.md`, a
  coherent digest. Launcher: `VicGPT/scripts/test_vulkan_flashnext.ps1`. The 45-49 / 5.3-5.6 tok/s figures and the
  "~330 s of model time against Strata's ~67 s" are recorded in the KB note
  `llamacpp-vulkan-flashnext-test-2026-10-17`, but the log that note cites
  (`VicGPT/logs/vulkan-flashnext-auto_2026-10-17_000222.log`) is **not in the VicGPT repo under that name**: what is on
  disk is `...2026-10-16_235708.log` (5 lines, a start that was stopped) and `...2026-10-16_235849.log.stderr`. Re-run
  the test and keep its report before treating the two speeds as measured.
- **Counts corrected.** `include/strata/kernels/*.hpp` is 51 files, not 52. `native_head.hpp` is at
  `include/strata/core/native_head.hpp`. The doorbell this plan names is `sycl/include/strata/sycl_doorbell.hpp`, a
  SYCL-only file; Vulkan needs its own. `metal/src/engine` is 14 files / 5,227 lines, `stubs.cpp` is 367 lines holding
  198 `strata::kernels::` stubs, and the shader trees total 5,837 lines. `metal/src/gpu/runtime.cpp` is exactly 640
  lines, and `metal/include/metal_compat/` is the three headers named below. `sycl/src/kernels/*_parity.cpp`: 38 oracles.

## Why, and what this plan does not claim

- **SYCL engine today** (k22 split, B580 layers 0-21, A770 22-47): prompt 260-320 tok/s, decode 18-25 tok/s, clean.
  More layers on the A770 measured up to ~2x faster in short runs (split11: 621 / 851 prompt at 4K / 16K, 35.5
  decode, commit `2ccc945`), but those runs produce NaN: 37 A770 layers fail, 32 fail with images, 26 are clean
  (`OPTIMIZATIONS-2026-10-16.md`, "Where things stand").
- **llama.cpp Vulkan on the same model and PC** (VicGPT's auto-fit run, cited file by file in the review above): a
  clean, coherent digest with **36 layers on the A770** and 49/49 offloaded across the two cards, but prompt 45-49 tok/s
  and decode 5.3-5.6 tok/s (KB-recorded; the log that record cites is not on disk under that name, see the review).
  Its slowness is llama.cpp's memory layout: no hot-expert cache, 22 layers' experts on the CPU, 64.5 GiB mapped on 32
  GB of RAM so decode pages from the SSD. It is not evidence about Vulkan kernels.
- **So the bet is:** Vulkan's driver path may not have the >26-layer fault, and Strata's layout is what makes this
  model fast. Both halves are n=1 or inferred. Phase 0 tests the first half for about a week of work; Phase 2 tests
  whether Vulkan kernels are fast enough. Do not start Phase 3 before both gates pass.
- **The cheaper alternative stays open:** the same test suggests the >26-layer fault is on the SYCL / Level Zero side.
  Finding it there (the KB note `references/split11-nan-story-2026-10-16.md`, `bench/results/handoff-mtp-k22-2026-10-16.md`)
  could give the split11 speed without a new backend. Whoever owns this plan should know that path exists.

## Constraints (read before any work)

- **Machine rules, `HANDOFF-WINDOWS-SYCL.md` section 1.** The user launches every model load and every GPU test, in
  a normal terminal, watching Task Manager. The agent builds, writes probes and launchers, and reads logs.
  - Stop signals: shared GPU memory above ~0.2 GB on either Arc; B580 dedicated near ~10.5-11.3 GB; Windows RAM nearly
    full; any screen flicker (a TDR).
  - Vulkan host-visible buffers show up as "shared GPU memory" even with VRAM to spare (seen in the llama.cpp test).
    A spill check must compare each card's **dedicated** use against its own limit; read the per-card counters by
    DXGI LUID, as `strata::core::dxgi_local_memory` already does for SYCL (`sycl/src/core/pinned.dp.cpp`).
  - Ask before any download or install (the Vulkan SDK included): name it, its source and its size.
- **Repo rules, `AGENTS.md`.** One change per pull request. A new backend is opt-in: its own CMake project under
  `vulkan/`, like `metal/` and `sycl/`, never part of the CUDA, HIP or SYCL builds. Those builds stay byte-identical.
  Measured numbers only, each with what it was measured on.
- **Evidence rules on this PC.** A single clean run proves nothing (shown twice). Every gate needs N >= 3 identical
  runs, a restart between runs, and `tools/strata_bench.py`'s pollution gate on every reply. NaN output decodes
  faster than real output: a speed number from a polluted run is void.
- **Branch:** `vulkan-backend` from `main`. Commit per phase step; the plan's checkboxes below are the progress record.

## Architecture: copy the Metal port's shape

The Metal port (`metal/`, `docs/MAC.md`) is the template. It builds the shared host engine from `src/` (the CUDA
host code: `src/core/*`, `src/program/generate.cpp`) **unchanged**, with three pieces of its own:

| Metal piece | What it does | Vulkan equivalent to write |
|---|---|---|
| `metal/include/metal_compat/cuda_runtime.h` | Maps every CUDA runtime name the host code uses (`cudaStream_t`, `cudaMalloc`, `cudaGraphLaunch`, ...) onto `strata::gpu`. Goes first on the include path. Only names the host code really uses; anything else fails to compile, on purpose | `vulkan/include/vulkan_compat/cuda_runtime.h`, same inventory, same rule |
| `metal/src/gpu/runtime.cpp` (640 lines) | The `strata::gpu` runtime: devices, memory, streams, events, graphs | `vulkan/src/gpu/runtime.cpp` over Vulkan 1.3 |
| `metal/src/engine/*` (14 files, 5,227 lines) + kernels (5,837 lines of MSL) | The kernel library: the entry points the host code calls, implemented for Metal. A verify window is encoded as one dependency graph in one command buffer (`window.cpp`) | `vulkan/src/engine/*` + compute shaders (GLSL, compiled to SPIR-V with `glslc`) |
| `metal/tools/gen_stubs.py` + `stubs.cpp` (367 lines, 198 `strata::kernels::` stubs) | Every entry point not ported gets a stub that stops the engine with its name, generated from the failed link | Same tool, adapted to MSVC's link errors ("unresolved external symbol"), so the engine links on day one |

Why the CUDA host tree and not the SYCL tree: `sycl/` is a dpct migration with SYCL types throughout its host code;
`src/` is the upstream host code, already proven portable twice (HIP through `hip_compat`, Metal through
`metal_compat`). Read `metal/include/metal_compat/cuda_runtime.h` first: its comment lists the inventory method.

How Strata's mechanisms map to Vulkan:

| Strata mechanism | Vulkan mapping | Note |
|---|---|---|
| CUDA graphs: a verify window, the commit, the session steps recorded once and replayed | A command buffer recorded once and resubmitted; push constants or a small uniform buffer for per-window values | Graph *update* is not needed: the engine already keeps per-window values in device buffers so graphs replay unchanged (see `include/strata/kernels/qsa.hpp`, the step-state note) |
| Raw device pointers everywhere (slot tables, `ptr2` arrays of expert addresses, `cache_base + slot_off`) | `VK_KHR_buffer_device_address` (core in 1.2) + `GL_EXT_buffer_reference` in shaders: 64-bit pointers in shaders | Also removes the 4 GiB binding limit (`maxStorageBufferRange` is usually 4 GiB): the 6-7 GiB expert cache is addressed by pointer, not by one binding |
| Mapped host memory the GPU polls (doorbells, `sycl_doorbell.hpp`), CPU pool results the GPU reads | Host-visible, host-coherent memory; shader polls with atomic loads under the Vulkan memory model (`VK_KHR_vulkan_memory_model`, core 1.2); keep Strata's bounded spins | A dispatch that spins too long trips the Windows TDR (`TdrDelay=10` on this PC). Probe it (Phase 0, P5) |
| The pinned expert arena / RAM mirror read over PCIe | `VK_EXT_external_memory_host`: import an existing host allocation as device-visible memory. **Import the same host buffer into both cards' devices**: one copy readable in place by both | This is the structural difference from SYCL, where a kernel reading another context's host memory got wrong data (probe T3). Probe it (Phase 0, P3) |
| Streams and events, multi-stream joins | One `VkQueue` per stream (or command buffers on one queue); events = timeline semaphore values (core 1.2) | Cross-**card** waits: no shared semaphores between two devices without external-semaphore interop; do them on the host, as the SYCL split already does (`Verifier::forward_handoff`) |
| `dp4a`, sub-group shuffles, XMX `joint_matrix` | `VK_KHR_shader_integer_dot_product`, subgroup operations + `VK_EXT_subgroup_size_control` (Arc: 8/16/32), `VK_KHR_cooperative_matrix` for XMX | XMX is a prompt-speed item for Phase 5, not needed for correctness |
| oneMKL / cuBLAS GEMMs in the prompt path | Own GEMM shaders. Starting point: llama.cpp's `ggml-vulkan` shaders (MIT, `third_party/ggml` already carries ggml), including its IQ2_XS dequantization and `mul_mm` | Keep the license header; note the source commit |

## Phase 0 preconditions: the toolchain (its own gate, needs your call)

Nothing in this plan can be compiled today. Verified absent: the Vulkan SDK, `VULKAN_SDK`, `vulkan.h`, `glslc`,
`glslangValidator`. Verified present: the loader `vulkan-1.dll`, `vulkaninfo.exe`, `dxc.exe` in the Windows Kits, and
the oneAPI/MSVC toolchain the SYCL engine already builds with. Three ways in, in the order I would take them:

| Route | What it installs | Cost | What it costs us later |
|---|---|---|---|
| **A. Vulkan SDK** (recommended, and what this plan assumes) | Khronos SDK via `winget install KhronosGroup.VulkanSDK`; headers, `vulkan-1.lib`, `glslc`, validation layers, `vulkaninfo` | about 250-300 MB download, about 1 GB on disk (72 GB free) | none; Phases 1-3 want the same tools, and the validation layer is what makes a bad descriptor visible instead of a mystery |
| **B. Clone and build** | Khronos `Vulkan-Headers` (about 4 MB) vendored under `third_party/`, plus `glslang`, `SPIRV-Tools`, `SPIRV-Headers` built from source with the installed compiler | four clones, a 5-10 minute build, no installer | no validation layer; glslang must be rebuilt if the toolchain moves |
| **C. `dxc` from the Windows Kits + cloned headers** | only `Vulkan-Headers` cloned; shaders written in HLSL and compiled with `dxc -T cs_6_0 -spirv` | smallest download | HLSL has no clean way to hold a 64-bit device address in a shader, which is exactly what P2 and the engine's `ptr2` tables need; the later phases would then be written in a language the port does not use |

Route C is not acceptable for P2. If no install is wanted, B is the fallback, and the shaders stay GLSL.

Whichever route is chosen, record it here as done before the probe is built, because the probe's CMake is the same
build skeleton Phase 1 needs (`find_package(Vulkan)`, `glslc` for `vulkan/shaders/*.comp`, SPIR-V embedded as byte
arrays).

## Phase 0: capability probe (gate, no engine code)

One standalone program, `vulkan/probe/vk_caps_probe.cpp`, built with the Vulkan SDK, runs on both cards in one
process, prints a table, exits. Each test is bounded (a few seconds). The user runs it.

- [ ] **P1. Devices and limits.** Both Arcs enumerate in one `VkInstance` (the UHD 770 is skipped). Print per card:
  driver version, `maxMemoryAllocationSize`, `maxStorageBufferRange`, `maxBufferSize` (`VK_KHR_maintenance4`),
  `maxMemoryAllocationSize`, `bufferDeviceAddressAlignment`, `minImportedHostPointerAlignment`
  (`VK_EXT_external_memory_host`), the memory heaps and types (device-local, host-visible, host-coherent, host-cached)
  with each heap's size, subgroup sizes, and the extensions in the mapping table above.
  - Also print, per card, `VK_EXT_memory_budget` if present (budget and current usage per heap). The spill rule of this
    PC is about **dedicated** memory (`HANDOFF-WINDOWS-SYCL.md` section 1), and Vulkan host-visible buffers read as
    shared memory, so the probe needs a per-card dedicated number of its own. `VK_EXT_memory_budget` gives it from
    inside the API; DXGI by LUID is the cross-check the SYCL engine already uses (`sycl/src/core/pinned.dp.cpp`,
    `dxgi_local_memory`), remembering section 11: the A770 answers twice in DXGI on this PC, so match by LUID/UUID and
    say which one was read.
- [ ] **P2. A 7 GiB device allocation on the A770,** read and written past 4 GiB through buffer device address:
  write a tag every 64 MiB from a shader, read it back from the host through a copy, and from a second shader.
  All tags right = pass. This is the Vulkan twin of the SYCL A770 4 GiB problem (`HANDOFF-WINDOWS-SYCL.md` section 9.3).
  If one allocation is refused, try 2 x 3.5 GiB and record the limit.
- [ ] **P2b. The same buffer bound past 4 GiB the ordinary way.** A shader reads and writes through a buffer *view*
  created at an offset of 5 GiB (`vkCmdBindDescriptorSets` with `offset`), with no buffer device address, and a second
  one reads at `offset + 64 MiB`. Record `maxStorageBufferRange` next to what actually worked. This is the question the
  SYCL engine paid for: borrowed prompt operands past 4 GiB needed `staged_gemm` on the A770. If P2b works, Phase 3's
  "addressed by pointer, so no staging copy" claim is proven on this driver by the cheaper mechanism too. If it fails
  while P2 passes, only the pointer path is usable and every borrowed operand must come through an address.
- [ ] **P3. One host buffer, both cards.** `VirtualAlloc` 1 GiB, fill a pattern, import it with
  `VK_EXT_external_memory_host` into the B580's device **and** the A770's device; a shader on each reads it in place
  and writes a checksum. Then each card writes its own half and the other card reads it after a host-side wait.
  Correct on both = the cross-card hand-off and the shared mirror need no copies. Wrong data = same rule as SYCL:
  per-card buffers plus host copies.
- [ ] **P4. Replay cost.** Record one command buffer with 2,400 small dispatches (the size of a verify window graph,
  `HANDOFF-WINDOWS-SYCL.md` section 12), submit it 100 times, report ms per submit on each card.
- [ ] **P5. Host-polled spin.** A shader on each card spins on a host-coherent flag (bounded, 2,000,000 reads, as
  `STRATA_SPIN_MAX` on this PC); the host raises the flag after 1, 10 and 100 ms. Report the latency the shader
  saw, and that no TDR fired at 100 ms.
- [ ] **P6. Cooperative matrix and integer dot product.** List the supported cooperative matrix shapes and types per
  card (fp16 and int8), and whether 4x8-bit packed integer dot product is accelerated.

**Gate 0:** P1, P2, P2b, P4, P5 pass on both cards. P3 decides the cross-card design (shared or copied), P6 decides
Phase 5's scope. Run the probe **three times with a full exit between runs** and require the same verdict three times:
the fault this plan is chasing was intermittent on the SYCL side, and a capability probe that passes once is exactly the
single clean run this PC has taught us not to trust. Write the three runs to `bench/results/vulkan-caps-<date>.md`. If
P2 fails on the A770 with no workaround, stop: the expert cache cannot live there. If P2 passes but P2b fails, write it
down: it means Phase 3 must route every borrowed operand through an address, which is a design decision, not a detail.

## Phase 1: skeleton that links

- [ ] `vulkan/CMakeLists.txt`: its own project like `metal/` (Windows and Linux, `find_package(Vulkan)`, `glslc` for
  `vulkan/shaders/*.comp` -> SPIR-V embedded as byte arrays). Builds `strata` from `src/` with `vulkan_compat` first
  on the include path. The top-level `CMakeLists.txt` gets no new option in this phase; build with
  `cmake -S vulkan -B build-vulkan`.
- [ ] `vulkan/include/vulkan_compat/cuda_runtime.h`: the CUDA names the host code uses. Get the inventory by building
  with an empty header and reading the compile errors (`metal_compat`'s comment describes the method), then
  mapping each name. Also any other CUDA header the host code includes (compare `metal/include/metal_compat/`:
  `cuda_runtime.h`, `immintrin.h`, `cpuid.h`).
- [ ] `vulkan/src/gpu/runtime.cpp`: devices (one `VkDevice` per Arc), allocation (a sub-allocator over large
  `VkDeviceMemory` blocks; buffer device address on every buffer), streams, events (timeline semaphores), graphs
  (stream capture records into a command buffer; launch resubmits it), host-mapped memory (P3's result decides
  `cudaHostAlloc(..., Portable | Mapped)`), `cudaMemGetInfo` from the DXGI budget minus this process's usage.
- [ ] `vulkan/tools/gen_stubs.py`: Metal's tool, adapted to MSVC / lld-link undefined-symbol output and to
  `undname` or clang demangling.
- **Exit:** `strata.exe --help` and `--families` run on Windows; a model load reaches the first stub and stops with
  its name. No GPU kernel exists yet.

## Phase 2: one card, decode only, correct (gate)

Port the kernel entry points the decode path hits for this model (Qwen3.8-Flash-Next IQ2_XS native pack), on the
**A770 alone** first (`level_zero` is not involved; select the device in the runtime). Port in the order a token
needs them, replacing one stub at a time:

1. embedding gather (`NativeEmbed`) and the residual / norm / elementwise kernels (`include/strata/kernels/elementwise.hpp`, `native_gr_norm.hpp`, `fused_gr.hpp`);
2. the router top-10 (`native_router.hpp`, `router_top10.hpp`);
3. IQ2_XS and the model's other quant types in the matrix-vector kernels (`native_mmvq.hpp`, `iq_kernels.hpp`): the dense projections, the shared expert, the routed experts reading cache slots by pointer;
4. GDN decode (`native_gdn.hpp`, `native_gdn_preprocess.hpp`, `gdn.hpp`);
5. QSA decode attention, its indexer, scoring and selection, the K/V append (`native_qsa*.hpp`, `qsa_decode_attn.hpp`, `qsa_select.hpp`, `kv_q8.hpp`);
6. the head and the sampler (`native_head`, `sampler.hpp`);
7. the verify-window kernels and doorbells (`verify_kernels.hpp`), so `--spec 4` drafts work.

For each entry point, a parity test against the CPU reference or the SYCL parity oracle
(`sycl/src/kernels/*_parity.cpp`; the tolerances there are the ones to meet). Metal's `metal/src/engine/` shows which
entry points a working engine needed: its 248 stubs are the ones it did not.

- **Gate 2, correctness:** greedy (temperature 0) 64 tokens on a fixed ~2K prompt are token-identical to the SYCL
  engine on the A770 alone, N = 3.
- **Gate 2, speed:** decode within 1.5x of the SYCL A770-alone figure (14-21 tok/s, README) on the same prompt and
  cache size, N = 3, pollution gate clean. Slower than that with no identified cause = stop and report: the port's
  premise needs the kernels to be competitive.
- **Missing piece the first draft did not name:** the pollution gate lives in `tools/strata_bench.py`, which starts the
  engine through `serve/server.py --engine strata --config <cfg>`. For Gate 2 to be measurable at all, the Vulkan build
  must produce a `strata.exe` that `--serve` speaks the same loop, or a small standalone driver that emits the same
  `strata serve: prompt ... drafts accepted ...` lines the gate and `tools/arms_logscan.py` parse. Decide that at the
  end of Phase 1, not at Phase 2: it is a host-side job, it is small, and without it nothing in Phase 2 can be called
  clean.

## Phase 3: prompt path, one card

- [ ] Prompt GEMMs for IQ2_XS chunks (start from ggml-vulkan's `mul_mm` and IQ2_XS dequant shaders), QSA prompt
  attention, GDN prompt, K/V writes, the expert streaming of a chunk (copies from the arena / mirror into staging,
  as `src/prefill/` does).
- [ ] The 4 GiB question again for prompt buffers borrowed from the cache: addressed by pointer, so no staging copy
  should be needed. Verify it with a prompt whose borrowed operands lie past 4 GiB (the SYCL engine needed
  `staged_gemm` for this on the A770).
- **Exit:** a 16K prompt gives the same first 64 greedy tokens as SYCL A770-alone; prompt speed recorded beside the
  SYCL figure (90-130 tok/s, README).

## Phase 4: two cards, the deciding test

- [ ] The layer split through the shared host code's multi-GPU path (`cudaSetDevice` per stage, the stage hand-off,
  per-stage caches). Cross-card buffers follow P3: imported once into both devices if P3 passed, else per-card
  buffers with host copies (the `f4fc1e3` pattern).
- [ ] The CPU pool for both stages' misses; the A770 stage's PCIe reads from the mirror (blocked on SYCL by the
  context rule; with P3 passing they can stay on).
- [ ] Run the split ladder with real traffic, text and images, pollution gate on, N >= 3 each:
  k22 (26 A770 layers), split16 (32), split11 (37).
- **Gate 4 (the reason for this plan):** split11 and split16 clean on Vulkan, N >= 3, with text and with images.
  - Clean and faster: record speed against SYCL k22 (260-320 / 18-25) and the split11 ceiling (621-851 / 35.5). This is
    the result to take to the user.
  - Clean but **not** faster than SYCL k22: then the port bought correctness, not speed. Say so plainly, and the cheaper
    alternative (finding the fault inside SYCL, `OPTIMIZATIONS-2026-10-16.md` "Fault search") becomes the main line
    again rather than a footnote.
  - Same NaN at the same layer counts: the fault is not the driver path. Stop, report, and hand the evidence to the
    SYCL NaN hunt.
  - **Measure against a same-hour baseline.** Two-card numbers on this PC drift 5-7% across an evening with no config
    change (`OPTIMIZATIONS-2026-10-16.md`, "Baseline round"). A Vulkan run at 21:00 compared with a SYCL run at 19:30 is
    not a comparison: run the SYCL arm in the same block, interleaved, or the verdict is noise.

## Phase 5: speed (only after Gate 4)

- XMX via `VK_KHR_cooperative_matrix` for the prompt GEMMs and prompt attention (P6 decides the shapes).
- Expert mirror for the later stage's misses (the open item in `OPTIMIZATIONS-2026-10-16.md`, Speed item 1) with
  the import from P3.
- Prompt chunk size arms (`--prefill auto:16384` at 64K), the expert profile from VicGPT traffic.
- Launcher `start-strata-vulkan-split11.bat` and a config; the SYCL launchers stay the default until Vulkan wins
  Gate 4 and a VicGPT A/B.

## First steps on this branch, in order

1. The toolchain decision above, recorded as done. Nothing builds before it, and it is yours to make or to refuse.
2. `feat: vulkan probe build`: `vulkan/CMakeLists.txt` (probe target only, `find_package(Vulkan)`, `glslc` rules),
   `vulkan/tools/embed_spv.py` (SPIR-V to byte arrays), `start-vulkan-probe.bat` (build, run, write
   `bench/results/vulkan-caps-<run>.log`). No engine code. The CUDA, HIP and SYCL builds and the top-level
   `CMakeLists.txt` stay untouched.
3. `feat: vulkan capability probe`: `vulkan/probe/vk_caps_probe.cpp`, one test per function, each switchable
   (`--only p2b,p3`), every allocation bounded and freed, `--limit-gib` and `--host-mib` so the first pass can start
   small on a machine you are watching, and a printed per-card dedicated-memory line from P1.
4. `feat: vulkan probe shaders`: `vulkan/probe/shaders/*.comp` (tag write, tag read, checksum, half write, no-op
   dispatch, bounded spin).
5. You run `start-vulkan-probe.bat` three times, with a full exit between runs. Seconds each, no model load, and I do
   not run it. I read the logs, write `bench/results/vulkan-caps-<date>.md`, and either tick Phase 0 here or stop the
   plan at the gate that failed and report which one.



## Rough sizes (for planning, not commitments)

| Phase | Work | Comparable |
|---|---|---|
| 0 | One probe program, ~600-900 lines | the SYCL probes in `C:\strata-deps\` |
| 1 | Compat header + runtime + build, ~1,500-2,500 lines | Metal: 640-line runtime + compat header |
| 2 | Decode kernels + parity tests | Metal's decode share of its 5,837 lines of shaders |
| 3 | Prompt kernels | the rest of that, plus GEMMs (ggml-vulkan as a base) |
| 4 | Mostly host glue and testing | the SYCL split work of 2026-10-15/16 |

## Files to read first

- `HANDOFF-WINDOWS-SYCL.md` (sections 1, 9, 16, 17): the PC, its rules, and how the cross-context fault was found.
- `OPTIMIZATIONS-2026-10-16.md`: the current baseline and backlog.
- `docs/INTEL_WINDOWS.md`: the Windows runbook (configs, env, launchers).
- `docs/MAC.md`, `metal/CMakeLists.txt`, `metal/include/metal_compat/cuda_runtime.h`, `metal/src/gpu/runtime.cpp`,
  `metal/src/engine/window.cpp`, `metal/tools/gen_stubs.py`: the port template.
- `include/strata/kernels/*.hpp` (51 headers) plus `include/strata/core/native_head.hpp`: the kernel entry points to implement.
- `sycl/src/kernels/*_parity.cpp`: parity oracles and tolerances.
- The VicGPT test behind this plan, on this PC: `VicGPT/scripts/test_vulkan_flashnext.ps1`,
  `VicGPT/logs/vulkan-flashnext-auto_2026-10-16_235849.log.stderr` (the fit: 36 layers on the A770, 64.5 GiB mapped),
  `VicGPT/benchmarks/vulkan-flashnext-auto_2026-10-17_000222.md` (the coherent digest), the KB note
  `llamacpp-vulkan-flashnext-test-2026-10-17`, and `VicGPT/benchmarks/sycl-vs-vulkan_2026-10-15_082834/` for the earlier
  llama.cpp Vulkan runs (those are qwen35 27B Q4_K, not this model).
