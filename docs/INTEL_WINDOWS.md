# Strata on Intel Arc, native Windows

The SYCL engine (`sycl/`, see [INTEL.md](INTEL.md)) builds and runs natively on Windows with Intel oneAPI 2026.1, on
Arc **Alchemist** cards (A750 / A770) and **Battlemage** cards (B580, Arc Pro B-series). Same server, same APIs, same
web app as the CUDA engine. Written for and measured on an Arc B580 (12 GB) + Arc A770 (16 GB) PC, 2026-10
(engine 0.1.44.1, IQ2_XS).

## What works today

- **Single card**: B580 alone or A770 alone, full 64K-128K context serving.
- **Two Arc cards, mixed generations, the 22/26 layer split** (`--layer-split 22`, called k22): B580 layers 0-21,
  A770 layers 22-47. Works for text and for pictures (the vision encoder runs on the CPU). Benchmark-certified
  IQ2_XS at 64K context (pollution-gated, 7 sessions 2026-10-16): **21.5 tokens/s decode**, prompt reads **365
  tokens/s (4K) / 608 (16K)** across the 6 clean sessions, drafts accepting up to ~86% on good prompts. The first
  sessions of an evening read 22.2 / 377 / 628 and the numbers drift down about 5% after hours of benchmarking.
  128K context also starts and serves.
- **Deeper A770-heavy splits are the open frontier**: a 32-layer A770 side (split16) ran text-only clean but NaNs as
  soon as pictures enter; the 37-layer side (split11) NaNs text at 64K and at 32K. The frontier is not purely the
  layer count either: in the gated ladder k22 itself produced storm sessions at 32K context and at 16384-token
  prompt chunks. Treat "clean" as per-shape with a rate: at the 64K auto shape 4 of 17 sessions stormed (about 1 in 4),
  32K context is now 3 for 3 (23:17 and 23:36 both lit on the first reply after a 4K prefill: 24 tokens of `!` from the
  first character, in a single-token window with no drafts, after a prefill at 539-541 tok/s where clean sessions read
  425-445), 16384-token chunks 0 for 1, and 9 of 22 sessions overall. A storm usually lights inside short chat decode,
  once inside a long agent turn, and it can start mid-reply. Re-check any new shape with the
  benchmark gate (N>=3) and treat a stormed session as void.
- Built-in drafts (speculation windows, `--spec 4`) work on all of these. The optional MTP drafter is **parked on
  the two-card Windows path** (see caveats).

## Run

Daily launchers in the repo root (port 8081 for the Arc configs):

| Bat | Setup |
| --- | --- |
| `start-2card-k22-ctx64k.bat` | the daily default: k22 split, 64K, text |
| `start-2card-k22-ctx128k.bat` | the same split at 128K context |
| `start-strata-k22-vision.bat` | k22 with pictures on the CPU encoder (the clean image fallback) |
| `start-strata-split11.bat`, `start-strata-split11-ctx32k.bat`, `start-strata-split16*.bat` | split-search probes, currently the unstable class |

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

The config JSONs next to the launchers carry the engine env. The pieces the two-card run needs:

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
- **The open frontier tracks how many layers the Alchemist card holds.** Up to 26 layers on the A770 (k22) is
  reliable for text and pictures; from 32 layers the split starts producing all-NaN windows (text-only at 37, text
  then pictures at 32). The earlier "one big arena past 4 GiB" reading did not survive the split search: clean k22
  runs stage GEMM operands past 4 GiB, and split16 NaNs on requests that stage nothing past it. Per-op staging is
  neither necessary nor sufficient as the explanation. The fault shape is tracked in
  [HANDOFF-WINDOWS-SYCL.md](../HANDOFF-WINDOWS-SYCL.md) and `bench/results/`. Once a session has served a
  non-finite window, every later request on that server stays garbage (fast, fake speeds) until a restart.
- **Debug scans can hide the fault.** In 2 of 2 split11 runs the synchronous `STRATA_DBG_NAN` scans changed timing
  enough that the run looked clean. Gate fault conclusions on normal launchers without debug env (N>=3).
- **`--prefill auto:N` snaps back to 8192** for any N below it; to actually run 4096-token chunks, ask for the
  fixed `--prefill 4096`.
- **MTP (the optional drafter) is parked on Windows two-card runs.** In every clean measurement (single card, both
  cards) MTP drafts matched or lost to the built-in drafts, and the engine's own draft auto-disable turned them off
  and ran faster on the A770. On the two-card split the MTP path also carries an unresolved NaN hunt documented in
  `bench/results/handoff-mtp-k22-2026-10-16.md`. The built-in drafts (`--spec 4`) are stable and accepted 29-86% of
  drafts depending on the prompt.
- **Cross-context memory on Windows is strict**: each Arc is its own SYCL platform, and the runtime refuses a context
  spanning two platforms (`sycl::context{b580, a770}` throws "Can't add devices across platforms"), so the engine must
  keep one context per card. A kernel may not read the other context's buffers in place (it returns wrong data with no
  error); staging through host memory works. That is why cross-card hand-offs cost a PCIe round trip, and why one card
  cannot help the other's prompt read without the same staging. On Linux both cards share one platform, one context
  holds both, and the engine keeps in-place hand-offs.
- Windows engine logs go to the file named by the config's `"log"` field.

## Debug instruments (for the open items above)

`STRATA_DBG_NAN=1` (state scans, window-head scan, commit-row scan, hand-off row scans), `STRATA_DBG_ALLOC=1`
(device allocation map at prompt end). On this PC a single clean run proves nothing: gate every conclusion behind
N>=3 identical runs.

The serve log carries its own pollution signature, independent of the reply text: a junk reply accepts every suffix
draft (`drafts accepted 48 of 48`) and runs about 1.3-2x the session's normal decode speed, because the model is
emitting one repeated token. `python tools/arms_logscan.py` reads those `strata serve: prompt ...` lines and prints
the suspicious rows per session; it flagged all seven known-storm sessions of 2026-10-16 and none of the thirteen clean ones.
