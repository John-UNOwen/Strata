# MTP buffer inventory (Phase 0, 2026-10-16)

Method: read-only audit of the MTP code paths on `mtp-cross-card` (HEAD `20c693c`, drafter WIP included). No GPU run. Context column = which SYCL context the allocation belongs to: on Windows each GPU is its own platform, so "default" means CUDA0 (B580) unless an `OnDevice` guard is active at the call.

Legend: OK = reader and allocator contexts match. FLAG = candidate cross-context in-place access, must be resolved in Phase 2. CHECK = needs one grep or a 1-line experiment to settle.

## Drafter (MtpDrafter), device_ = last stage card = A770 (mtp.cpp:268 "the drafter lives there")

| Buffer | Alloc site | Context | Written by | Read by | Verdict |
|---|---|---|---|---|---|
| dense_, experts_, state_arena_, arena_, head_logits_, dhead_, dvocab_, dense4_, pf_dev_, cparams_, cring_, cscratch_, dinv_ | mtp.cpp:224/308/355/392/467/617-634/731-760/1328 (`malloc_device_guarded` / `malloc_device`, `dev_queue()`) | A770 | drafter copies (A770) | drafter kernels (A770) | OK after WIP `dev_queue()` (mtp.cpp:60-63) |
| mapped staging: h/m_tok_, step_, pos_, row_, out_, prob_, force_, cparams_, chist_ | mtp.cpp:426-432/621 `mapped(..., dev_queue())` -> `host_malloc_polled` (uncached host, device pointer = host pointer) | A770 | host (CPU, always safe) + drafter kernels | host + drafter kernels | OK after WIP |
| cs_ (private stream), side_ (shared-expert branch) | mtp.cpp:504/523 `get_current_device().create_queue` during `load()` | A770 if `load` is called with current device already switched | CHECK: confirm every call site that uses `side_` (mtp.cpp:1020-1042) has `OnDevice(device_)` active; note mtp.hpp:241-242 initializes `side_ = &dpct::get_in_order_queue()` before load - safe only if never used pre-load | | CHECK |
| weights shared between drafter copies | mtp.cpp:277-287 (`shared->dense_/experts_`) | A770 (rejects mismatched device_) | - | - | OK |

## Cross-card data the drafter CONSUMES

| Buffer | Alloc site | Context | Read by | Verdict |
|---|---|---|---|---|
| verify window payload rows `R_` / `final_R(t)` / `final_R_all()` (mtp `bind(window_R)`, `set_source_R`, `draft_first(..., ver.final_R_all(), ...)`, pipeline `mtp.prefill(v.final_R(0), ...)` at generate.cpp:10254, 11329-11393, 11795, 12585) | verify.cpp:530-552 `mapped()` = `host_malloc_polled`/`malloc_host` with `dpct::get_in_order_queue()` at call time | per-device **if** `Verifier::init` runs under `OnDevice(device_)` (verify.cpp:328 shows the guard; confirm it encloses the 530-552 block) | drafter kernels on A770 (copy_from_mapped into its own Rin_) | CHECK (likely OK: the non-mtp split decode already reads these rows on A770 fine) |
| **prompt path chunk rows `R_rows` in `sp.on_chunk` -> `mtp.prefill(R_rows, ...)` (generate.cpp:7915-7920)** | PromptPath `sp.init` per stage (generate.cpp:6958 uses `st.stream` per stage; single-GPU form 6965 uses `main_cs`); rows staged in the prompt path's own buffers | **multi_gpu forces the token path (`batched = !multi_gpu`), so warmup AND every prompt chunk run `mtp.prefill` - exactly where the engine dies (DEVICE_LOST right after the window captures, then the literal status line `mtp prefill: no error` in run 9, run 10, `strata-iq2_xs-2card-mtp-k22.log`, `...-k22-spin.log` on the WIP build)** | drafter copy kernel on A770 | **FLAG #1 - prime suspect. Resolve: print `sycl::get_pointer_device(R_rows)`-style, or better: in `mtp.prefill`, stage R_rows through the drafter's own mapped buffer via a HOST memcpy (never a foreign-context kernel read), then copy host -> Rin_ on A770** |
| `nxt.data()` / `cur` (prompt token ids), positions | plain host vectors (heap) | none | host only | OK |
| drafter KV ring catch-up inputs (main layers' K/V cells) | session per-stage structures | per stage | drafter (A770) | CHECK: confirm the catch-up only reads the last stage's cells (generate.cpp:7692 comment says the draft K/V lives on the last stage; verify the loop ranges) |

## Cross-card data the drafter PRODUCES

| Buffer | Context | Read by | Verdict |
|---|---|---|---|
| m_out_/m_prob_ (drafts + probabilities), chain_tok()/chain_prob() | A770 (WIP) | host polls them, host hands draft ints to verifier input staging | OK |
| verifier input staging (h_tok_/h_step_/h_pos_ per verifier, filled from host) | per verifier's own device (verify.cpp:530 under init guard) | own stage kernels | OK |

## Shared session buffers touched during mtp paths

| Buffer | Alloc site | Context | Read by | Verdict |
|---|---|---|---|---|
| **SessionLoopScratch y_miss** (pool staging "read by the token graph's handoff kernel through its device pointer") | session.cpp:762 `host_malloc_polled(dpct::get_in_order_queue())`, init at generate.cpp:5907-5915 in main (current device = CUDA0) | **B580** | handoff kernels - of which stage(s)? | **FLAG #2 - if any stage-1 (A770) kernel touches y_miss it is cross-context. Resolve: grep the kernel submissions that receive `y_miss` (session.cpp:817-876 `session_loop`) for the device context active; Phase 3 probe #1 can round-trip it on both cards** |
| host_res (residency table), d_res copy | plain host vector + device copies via memcpy (generate.cpp:5928-6011) | host + per-memcpy device | kernels read d_res (device copy), host reads host_res | OK (copies only) |
| hand-off row buffers hand[b] / hand_in[b] | generate.cpp per-side allocations after `f4fc1e3` | writer card / reader card | own-side kernels | OK (fixed, runs 7-8/11 pass) |
| mapped GGUF file views: native pack expert blobs, 322 MiB IQ4_XS token embeddings, PLE table (3200), speed projection cvec tables | mmap file views, no allocating queue | kernels on BOTH stages (stage-1 reads embeddings; pack blobs during fills) | **FLAG #3 (class): unproven. Mapped file memory is not USM; T3 did not cover it. Phase 3 probe #0: map a view, have the A770 kernel read it while the B580 owns the session - byte-exact test. If it corrupts, the embedding/PLE reads for stage 1 need per-card copies** |

## Where the death happens, pinned from logs
Startup order: draft head load (137.9 MiB) -> 2078 MiB free -> captures fine -> **first mtp prefill call** (`mtp.prefill` via `sp.on_chunk`, multi_gpu token path) -> DEVICE_LOST (20) + TDR within seconds. Both WIP k22 runs and both pre-WIP runs die identically, so moving the drafter's allocations to `dev_queue()` was necessary but not sufficient: the remaining cross-context in-place read is in the prefill path's INPUTS - FLAG #1 first, then FLAG #2, then FLAG #3.

## Phase 2 fix list derived from this inventory
1. `mtp.prefill` / `prefill` row inputs: host-staged copy of `R_rows` into the drafter's own mapped buffer (host memcpy in, then `copy_from_mapped` on A770). This matches the `set_handoff_copy` pattern and the T5 probe (host-to-host copies are CPU work: always safe).
2. Give `SessionLoopScratch::init` the right device (per-stage scratch, or init under `OnDevice` for each stage that uses it), if the CHECK settles as cross-stage.
3. `STRATA_MTP_CHECK_CTX=1` assert: before each submit, `sycl::get_pointer_device(ptr)` vs queue device for every USM pointer (mapped file views will report host device - keep them on the Phase 3 probe #0 list instead).
4. Never `dpct::get_in_order_queue()` inside `MtpDrafter` without `OnDevice(device_)` active (mtp.hpp:241 default still points at the default queue until load).

## Open CHECKs for Phase 1 (quick)
- Confirm `Verifier::init` line 530 sits inside the `OnDevice(device_)` scope that starts near verify.cpp:328.
- Confirm `side_` usage sites (mtp.cpp:1020-1042) run under `OnDevice(device_)`.
- Confirm the KV catch-up loop ranges (generate.cpp:7692 area).
