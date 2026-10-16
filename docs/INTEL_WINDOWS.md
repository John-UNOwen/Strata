# Strata on Intel Arc, native Windows

The SYCL engine (`sycl/`, see [INTEL.md](INTEL.md)) builds and runs natively on Windows with Intel oneAPI 2026.1, on
Arc **Alchemist** cards (A750 / A770) and **Battlemage** cards (B580, Arc Pro B-series). Same server, same APIs, same
web app as the CUDA engine. Written for and measured on an Arc B580 (12 GB) + Arc A770 (16 GB) PC, 2026-10
(engine 0.1.44.1, IQ2_XS).

## What works today

- **Single card**: B580 alone or A770 alone, full 64K-128K context serving.
- **Two Arc cards, mixed generations**: a layer split (default: B580 layers 0-10 with all its experts in VRAM, A770
  layers 11-47 with a 9.6 GiB expert cache, the rest streamed from RAM through the CPU). Measured IQ2_XS:
  **~35.5 tokens/s decode**, prompt reads 621 tokens/s (4K) / 851 tokens/s (16K), 64K context.
- Built-in drafts (speculation windows, `--spec 4`) work on all of these. The optional MTP drafter is **parked on
  the two-card Windows path** (see caveats).

## Build

```
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
cmake -S sycl -B build-sycl-jit -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DSTRATA_GGML_DIR=<llama.cpp src>
cmake --build build-sycl-jit --target strata
```

The **JIT tree** (no `-DSTRATA_SYCL_AOT`) is the regime for two cards and for any A770 work: AOT-compiled Alchemist
kernels silently read wrong data past 4 GiB inside one allocation (driver/IGC 32-bit surface offsets; the JIT path
compiles with `-ze-opt-greater-than-4GB-buffer-required` and is clean). An AOT `-DSTRATA_SYCL_AOT=bmg-g21` tree is an
alternative launch build for B580-only setups. Prepend `%MKLROOT%\bin` to PATH or strata.exe dies with a missing-DLL
exit (setvars.bat omits it).

## Run

Daily launchers in the repo root (port 8081 for the Arc configs):

| Bat | Setup |
| --- | --- |
| `start-strata-split11.bat` | the default: B580 + A770, 64K context, no MTP |
| `start-strata-split11-ctx32k.bat` | same split at 32K context (roomier VRAM) |
| `start-strata-split16.bat` | a deeper split variant (B580 layers 0-15) |

The config JSONs next to them carry the engine env. The pieces the two-card run needs:

```
"ONEAPI_DEVICE_SELECTOR": "level_zero:0;level_zero:1",
"SYCL_CACHE_PERSISTENT": "1",
"UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS": "1",
"STRATA_VRAM_FREE_MIB": "0,13000",
"STRATA_NO_NORM_ROPE": "1",
"STRATA_STAGE_TRIM": "1",
"STRATA_MAX_CPU_EXPERTS": "4096"
```

`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` is required on the A770 (its driver refuses big allocations otherwise).
`STRATA_VRAM_FREE_MIB` sizes the per-card headroom the expert cache may use (the A770's driver reports 0 free).

## Caveats, learned the hard way

- **Keep VRAM headroom.** Watch shared GPU memory and each card's dedicated use; a screen flicker (Windows TDR) means
  a card reset - stop, lower context or cache. Do not run two engines at once.
- **Big arenas on the A770 are the open frontier.** Its expert-cache arena is one allocation over 4 GiB. The JIT path
  copes, and 32K-context runs stage no GEMM operand past 4 GiB at all; the 64K path (8192-token prompt chunks) has a
  known standing issue class here, tracked with its fix window in
  [HANDOFF-WINDOWS-SYCL.md](../HANDOFF-WINDOWS-SYCL.md) and the bench handoffs. For today: split11 at 64K is the
  measured daily setup.
- **MTP (the optional drafter) is parked on Windows two-card runs.** In every clean measurement (single card, both
  cards) MTP drafts matched or lost to the built-in drafts, and the engine's own draft auto-disable turned them off
  and ran faster on the A770. On the two-card split the MTP path also carries an unresolved NaN hunt documented in
  `bench/results/handoff-mtp-k22-2026-10-16.md`. The built-in drafts (`--spec 4`) are stable and accepted 29-86% of
  drafts depending on the prompt.
- **Cross-context memory on Windows is strict**: each Arc is its own SYCL platform; a kernel may not read another
  context's buffers in place. The engine stages across cards through host memory (per-card hand-off buffers,
  per-queue gemm scratch); don't point two cards at one shared buffer in your own experiments.
- Windows engine logs go to the file named by the config's `"log"` field.

## Debug instruments (for the open items above)

`STRATA_DBG_NAN=1` (state scans, window-head scan, commit-row scan, hand-off row scans), `STRATA_DBG_ALLOC=1`
(device allocation map at prompt end). On this PC a single clean run proves nothing: gate every conclusion behind
N>=3 identical runs.
