# MTP across both Arcs on Windows SYCL: workflow (2026-10-16)

> **PARKED / SUPERSEDED (2026-10-16).** Cross-card MTP is parked: see the knowledge base decision
> `projects/strata-windows-sycl/decisions/park-cross-card-mtp-unless-it-beats-18-24.md`. The split11 NaN is
> settled as two faults (staging-scratch race, fixed in 9953a5b; >4 GiB A770 layout at 64K, open):
> `references/split11-nan-story-2026-10-16.md`. Kept as history.


Goal: speculative decoding (MTP drafter + verifier) with the k22 split (B580 layers 0-21,
A770 layers 22-47) and no DEVICE_LOST. Reference: MTP gave Qwen3.8-27B +47% decode on this box.
Background: `HANDOFF-WINDOWS-SYCL.md`, `bench/results/STATUS-2026-10-16.md`.

## Known facts
- Windows UR gives each GPU its own platform, so each card has its own SYCL context.
- A kernel that reads another context's host USM in place gets **wrong data, no error**
  (the `cross_usm_probe` showed it; commit `f4fc1e3` fixed the layer hand-off with per-card
  buffers and host copies via `set_handoff_copy`). A shared context (`97dbfd8`) was a no-op
  on Windows and was reverted.
- Two-card MTP fails with DEVICE_LOST (error 20) plus a TDR (event 4101). Leading theory: the
  drafter/verifier path still reads a buffer that belongs to the other card's context.
- B580-only MTP spilled 1.2 GB into shared memory (12.3 GB dedicated). **Do not rerun it.**
- Uncommitted: `mtp.hpp` `dev_queue()`, `mtp.cpp` changes, `generate.cpp` logging catch and
  terminate handler.

## Safety rules
- The user runs every GPU test, in a normal terminal, after `setvars.bat`. The agent builds,
  writes probes and launchers, and reads logs.
- Each test is time-bounded (a few tokens, then exit). Hard stops: any TDR, shared GPU memory
  > ~0.2 GB, B580 dedicated near ~11.3 GB.
- No full two-card MTP run until phases 1-3 pass.

## Phase 0: Inventory (code only, no GPU)
Make a table of every buffer the MTP path touches:

| buffer | allocated by (file:line) | which context/queue | written by | read by (kernel, card) |
|---|---|---|---|---|

Cover at least:
- drafter hidden-state input (last hidden from layer 47 = A770)
- drafter weights and KV
- draft tokens / logits
- verifier batch inputs (draft rows entering layer 0 = B580)
- accept/reject results
- position / RoPE / KV-rollback indices
- any `malloc_host`, `malloc_shared` or mapped memory (`session.cpp:751`, `verify.cpp:242`)

Flag every row where the reading card's context differs from the allocating context. That
list is the fix list. Deliverable: `bench/results/mtp-buffer-inventory.md`.

## Phase 1: Decide where the drafter lives
The drafter consumes the final hidden state (A770 side) and its tokens feed layer 0 (B580).
Prefer **the drafter on the A770**: its input is already local, it has more free VRAM, and
it has the fast PCIe link. Only the small draft-token array (k ints) and positions then cross
to the B580. Write down the choice and its VRAM cost, measured from the B580 dedicated figure.

## Phase 2: Explicit copies for every flagged row
- Each card reads only memory allocated in its own context: a device buffer, or a host buffer
  from that card's own `malloc_host`.
- Cross-card data goes through a per-card staging pair: src card `memcpy` to its host stage,
  host `memcpy` to the dst card's host stage, dst card `memcpy` in. Same pattern as
  `set_handoff_copy`; reuse it rather than writing a second one.
- Order with events or `wait()` on the right queue. Use `dev_queue()` from `mtp.hpp` and never
  `dpct::get_in_order_queue()` (the default queue belongs to one card).
- Add a debug assert (`STRATA_MTP_CHECK_CTX=1`): before each kernel submit, check that the
  pointer's `sycl::get_pointer_device` / context matches the queue. Off by default.

## Phase 3: Probes (agent builds, user runs, each < 1 min)
1. `probe/mtp_xfer`: allocate the real staging buffers and round-trip a known pattern
   A770 -> B580 -> A770 with the real sizes. Must match byte for byte.
2. Drafter alone on the A770, one draft step from a saved hidden state. Compare its logits
   to the single-card reference (tolerance documented).
3. Verifier with a fixed draft (no drafter): k=1, then k=2, k=4. Accepted tokens must equal
   plain greedy decode on the same prompt.
Stop at the first failure and fix it before going on.

## Phase 4: Full two-card MTP, short
- Greedy (temperature 0), 64 tokens, same prompt as plain decode. Output must be
  **token-identical** to MTP-off. That is the correctness gate.
- Then 512 tokens. Log draft acceptance, decode t/s, VRAM peaks per card.
- Then a VicGPT map chunk (8-13k prompt). Record prompt t/s too: on Qwen, MTP cost -37% prompt.

## Phase 5: Tune
- Draft length k: 1, 2, 3 (Qwen accepted ~0.74 at k=1-2).
- Measure the copy overhead per step from the instrumentation (`OPTIMIZATION-PLAN.md` item 1).
  If it is large, batch the drafter's outputs into one copy per step.
- Decide whether MTP stays on for VicGPT. Its map calls are prompt-heavy (~16:1 prompt:output),
  where Qwen's MTP was a wash.

## Deliverables
- Buffer inventory, probes, and code with explicit copies on branch `windows-sycl` (one commit per phase).
- `bench/results/STATUS-<date>-mtp.md`: per-phase pass/fail, acceptance, t/s, VRAM peaks.
- A launcher `start-strata-mtp.bat`, separate from the default; the default stays MTP-off
  until phase 4 passes.

## Not in scope
- B580-only MTP (memory spill). Revisit only at 32K context, with a measured VRAM budget.
- Shared-context approaches (shown to be a no-op on Windows UR).
