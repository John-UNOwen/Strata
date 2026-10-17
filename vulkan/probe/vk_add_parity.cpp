// vulkan/probe/vk_add_parity.cpp - strata_vk_parity: the parity run for the first real Vulkan engine kernel.
//
// Phase 1's probes proved the pieces (vk_caps_probe asked the driver what it can do; strata_vk_smoke drove the
// runtime's own mechanisms and dispatched nothing).  This is the first run that DISPATCHES an engine kernel, and it
// calls it the way the engine does - through the real declaration in include/strata/kernels/elementwise.hpp:52 - so
// what is measured is the whole Phase 2 route and not a private copy of it:
//   vulkan/shaders/elementwise_add.comp -> glslc -> SPIR-V -> strata::gpu::register_pipeline("add_inplace")
//   -> strata::kernels::add_inplace (vulkan/src/kernels/elementwise.cpp) -> the four uint64 words of the argument blob
//   -> push constants + storage bindings 0 and 1 -> vkCmdDispatch(grid_for(n), 1, 1) -> device_synchronize
//   -> the bytes read back through strata::gpu::host_view, compared with a CPU that computed them independently.
//
// The expectation is computed element by element on the host, never from what the GPU wrote, and the values are chosen
// so the comparison is exact rather than a tolerance: every dst element is a multiple of 1/4 in [-64, 63.75] and every
// src element a multiple of 1/8 in [0, 15.875], so their sum is a multiple of 1/8 with magnitude under 80 - well inside
// what 24 mantissa bits represent exactly.  A parity test that has to allow slop cannot tell a wrong kernel from a
// noisy one.
//
// WHAT CASE B IS FOR, and what it can and cannot show.  VULKAN-BACKEND-PLAN.md's layout rule for this port is
// "bindings under 4095 MiB, offsets free, including past 4 GiB", and Phase 0 P2b measured the driver honouring that
// (a 512 MiB binding at offset 4096 MiB read and wrote correct tags on both cards).  This port's own sub-allocator
// clamps every VkDeviceMemory block to min(maxStorageBufferRange, maxBufferSize, maxMemoryAllocationSize) = 4095 MiB
// (block_limit), so an offset INSIDE one of its blocks can only reach ~4091 MiB with a 4 MiB span: past-4-GiB offsets
// are reachable through a block the probe builds itself (what vk_caps_probe did in P2b), not through the runtime's
// allocator.  What IS reachable, and what case B measures, is a device OPERAND whose address lies past 4 GiB - either
// a buffer the sub-allocator placed past a 4200 MiB floor, or a pointer arithmetic'd past 4 GiB inside one big
// allocation, which is the shape the engine actually uses (cache_base + slot_off, borrowed prompt operands).  Both
// forms are run, each one's real offset printed in MiB, and the binding offsets are printed from record_dispatch
// itself (STRATA_VULKAN_TRACE_DISPATCH=1, which this file turns on) rather than asserted from the pointers.
//
// Nothing here polls or waits on the GPU: every synchronisation is a submit followed by a timeline-semaphore wait
// (device_synchronize), because Phase 0 P5 measured that this driver never makes a host write visible to a dispatch
// already running.  Every allocation is freed on every path, and cudaMemGetInfo is read before and after so a run
// cannot leak the 4 GiB it asked for.  Exit 0 only if every case passed.
#include "strata/gpu/runtime.hpp"

#include "strata/kernels/elementwise.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <vector>

// The runtime and the kernel entry points live in strata::gpu and strata::kernels; this probe calls them by their
// short names, so the prefix is opened once here instead of on every line.
using namespace strata;

namespace {

constexpr uint64_t kMiB = 1024ull * 1024ull;

// Case A's element count, as specified: 1048576 floats = exactly 4 MiB per buffer.
constexpr int64_t kN = 1048576;
constexpr size_t kBufBytes = (size_t) kN * 4u;
constexpr uint64_t kFloorBytes = 4200ull * kMiB;     // case B's floor: ~4.1 GiB, under this PC's 5 GiB ceiling

int g_fail = 0;

void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

void step(const char* what) { std::printf("\n== %s\n", what); }

void bad(const char* what) { std::printf("  FAIL %s\n", what); ++g_fail; }

// The values the host writes and the value the host expects.  Both sides are the host's own arithmetic; the GPU's
// output is only ever compared against them.
float dst_at(int64_t i) { return (float) ((int) (i % 512) - 256) / 4.0f; }
float src_at(int64_t i) { return (float) ((int) (i % 128)) / 8.0f; }
float sum_at(int64_t i) { return dst_at(i) + src_at(i); }

struct Buf {
    void* dev = nullptr;      // the device pointer the engine would be handed
    float* cpu = nullptr;     // strata::gpu::host_view(dev): the CPU's view of the same bytes
    uint64_t addr = 0;
    size_t bytes = 0;
};

Buf alloc_dev(size_t bytes, const char* what) {
    Buf b;
    if (gpu::malloc_device_raw(&b.dev, bytes) != gpu::kSuccess) {
        say("  FAIL malloc_device_raw(%s, %llu MiB) refused\n", what, (unsigned long long) (bytes / kMiB));
        ++g_fail;
        b.dev = nullptr;
        return b;
    }
    b.bytes = bytes;
    b.addr = (uint64_t) (uintptr_t) b.dev;
    b.cpu = (float*) gpu::host_view(b.dev);
    if (b.cpu == nullptr) {
        say("  FAIL host_view(%s): this runtime handed out no CPU view of that allocation, so the test cannot write "
            "or read it\n", what);
        ++g_fail;
    }
    say("  %-6s  %llu MiB  device address 0x%llx\n", what, (unsigned long long) (bytes / kMiB),
        (unsigned long long) b.addr);
    return b;
}

void release(Buf& b) {
    if (b.dev != nullptr) {
        const gpu::Error e = gpu::free_device(b.dev);
        // A free that failed is a failure of this run, not a note: an allocation this probe still owns is VRAM the
        // next process does not get.
        if (e != gpu::kSuccess) { say("  FAIL free_device: %s\n", gpu::error_name(e)); ++g_fail; }
        b.dev = nullptr;
    }
    b.cpu = nullptr;
    b.addr = 0;
    b.bytes = 0;
}

void release_dev(void*& p) {
    if (p != nullptr) {
        const gpu::Error e = gpu::free_device(p);
        if (e != gpu::kSuccess) { say("  FAIL free_device: %s\n", gpu::error_name(e)); ++g_fail; }
        p = nullptr;
    }
}

// Peak device usage this process caused, read the way the engine reads it (cudaMemGetInfo -> strata::gpu::mem_get_info,
// which is VK_EXT_memory_budget's heap budget minus what is already committed).
uint64_t g_peak_used_mib = 0;
void mem_line(const char* when) {
    size_t free_b = 0, total_b = 0;
    if (gpu::mem_get_info(&free_b, &total_b) != gpu::kSuccess) {
        say("  mem_get_info(%s): refused\n", when);
        return;
    }
    const uint64_t used = (total_b > free_b ? total_b - free_b : 0) / kMiB;
    if (used > g_peak_used_mib) g_peak_used_mib = used;
    say("  device usage %s: %llu MiB committed of a %llu MiB budget  (free %llu MiB)\n", when,
        (unsigned long long) used, (unsigned long long) (total_b / kMiB), (unsigned long long) (free_b / kMiB));
}

// One add_inplace call plus the comparison, printed as one line.  `guard_from` / `guard_to`, when guard_to is non-zero,
// are the elements that had to STAY as they were: the tail of the last workgroup past n, which is what the SPIR-V's
// `if (long(i) < pc.n)` guard exists for.  src is checked for not having been written at all - a kernel that added
// into the wrong buffer would otherwise still look like a passing add.
int run_case(const char* name, Buf& dst, Buf& src, int64_t n, int64_t guard_from, int64_t guard_to) {
    if (dst.dev == nullptr || dst.cpu == nullptr || src.dev == nullptr || src.cpu == nullptr) { bad(name); return -1; }
    for (int64_t i = 0; i < n; ++i) { dst.cpu[i] = dst_at(i); src.cpu[i] = src_at(i); }
    for (int64_t i = guard_from; i < guard_to; ++i) { dst.cpu[i] = 0.0f; src.cpu[i] = 0.0f; }
    std::fflush(stdout);

    // The engine's own entry point, called exactly as src/core/expert_source.cpp calls it.
    strata::kernels::add_inplace((float*) dst.dev, (const float*) src.dev, n, nullptr);
    if (gpu::device_synchronize() != gpu::kSuccess) { bad(name); return -1; }

    int64_t mismatches = 0, guard_broken = 0;
    int64_t first_bad = -1;
    for (int64_t i = 0; i < n; ++i) {
        const float got = dst.cpu[i];
        const float want = sum_at(i);
        if (std::memcmp(&got, &want, sizeof want) != 0) {
            if (mismatches == 0) first_bad = i;
            ++mismatches;
        }
        const float want_src = src_at(i);
        if (std::memcmp(&src.cpu[i], &want_src, sizeof(float)) != 0) ++guard_broken;   // src must not be written
    }
    for (int64_t i = guard_from; i < guard_to; ++i) {
        if (dst.cpu[i] != 0.0f || src.cpu[i] != 0.0f) ++guard_broken;
    }

    say("  %s: n %lld, %lld mismatch(es), %lld out-of-contract write(s) (src, or the tail past n)\n", name,
        (long long) n, (long long) mismatches, (long long) guard_broken);
    if (first_bad >= 0) {
        say("      first mismatch at i %lld: got %.9g, the host expected %.9g\n", (long long) first_bad,
            (double) dst.cpu[first_bad], (double) sum_at(first_bad));
    }
    if (mismatches != 0 || guard_broken != 0) { bad(name); return -1; }
    say("  %s: PASS\n", name);
    return 0;
}

void set_trace_env() {
#if defined(_MSC_VER)
    _putenv_s("STRATA_VULKAN_TRACE_DISPATCH", "1");
#else
    setenv("STRATA_VULKAN_TRACE_DISPATCH", "1", 1);
#endif
}

}  // namespace

int main(int argc, char** argv) {
    // The binding offsets record_dispatch actually wrote are what this probe has to report, so it asks for them
    // instead of inferring them from the pointers it was handed.
    set_trace_env();

    uint64_t floor_bytes = kFloorBytes;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--floor-mib=", 12) == 0) floor_bytes = std::strtoull(argv[i] + 12, nullptr, 10) * kMiB;
    }

    step("the device this run is on");
    int ndev = 0;
    if (gpu::device_count(&ndev) != gpu::kSuccess || ndev == 0) {
        say("  FAIL no Vulkan device this port selected\n");
        return 2;
    }
    int cur = -1;
    gpu::get_device(&cur);
    say("  devices selected: %d, this run's current device: %d\n", ndev, cur);
    for (int i = 0; i < ndev; ++i) {
        gpu::DeviceProp p;
        if (gpu::get_device_properties(&p, i) != gpu::kSuccess) continue;
        say("  [%d]%s %s  totalGlobalMem %llu MiB  maxSharedMemoryPerOptin %d B\n", i, i == cur ? " *" : "  ",
            p.name, (unsigned long long) (p.totalGlobalMem / kMiB), p.maxSharedMemoryPerOptin);
    }
    gpu::Caps c = gpu::caps();
    say("  caps: unified_memory=%d host_mapped_waits=%d (Phase 0 P5: no host-polled doorbell on this driver)\n",
        (int) c.unified_memory, (int) c.host_mapped_waits);

    mem_line("before");

    // ------------------------------------------------------------------ case A ----------------------------------
    step("case A: add_inplace over n = 1048576 in two 4 MiB device buffers");
    Buf a_dst = alloc_dev(kBufBytes, "dst");
    Buf a_src = alloc_dev(kBufBytes, "src");
    mem_line("after case A's allocations");
    run_case("case A", a_dst, a_src, kN, 0, 0);
    release(a_dst);
    release(a_src);

    // ------------------------------------------------------------------ case A2: the tail guard --------------------
    // n is deliberately not a multiple of the 256-invocation workgroup: grid_for rounds the workgroup count up, so the
    // last workgroup runs invocations past n, and those must write nothing.  The buffers are 1 MiB and the last 100
    // elements are the guard region.
    step("case A2: the same kernel with n not a multiple of 256 (the SPIR-V's i < n guard)");
    const int64_t tail_n = 262144 - 100;                 // 262044; ceil(262044 / 256) = 1024 workgroups = 262144 threads
    Buf t_dst = alloc_dev(1 * 1024 * 1024, "dst");
    Buf t_src = alloc_dev(1 * 1024 * 1024, "src");
    run_case("case A2", t_dst, t_src, tail_n, tail_n, 262144);
    release(t_dst);
    release(t_src);
    mem_line("after case A2");

    // ------------------------------------------------------------------ case B -------------------------------------
    step("case B: the same add with dst past 4 GiB of device address space");
    say("  floor: %llu MiB device local, asked for in one allocation\n", (unsigned long long) (floor_bytes / kMiB));

    void* floor = nullptr;
    bool floor_one = (gpu::malloc_device_raw(&floor, floor_bytes) == gpu::kSuccess);
    std::vector<void*> floor_parts;
    uint64_t floor_base = floor_one ? (uint64_t) (uintptr_t) floor : 0;
    if (floor_one) {
        say("  floor: one allocation at device address 0x%llx covering %llu MiB\n",
            (unsigned long long) floor_base, (unsigned long long) (floor_bytes / kMiB));
    } else {
        // The one-allocation form is what the task asks for, and this port refuses it when the blocks that would serve
        // it do not come out contiguous in device-address space (sub_alloc's contiguity check, runtime.cpp).  The same
        // address distance is then produced with two allocations each under the card's 4095 MiB block clamp.  Said out
        // loud, not smoothed over.
        say("  one %llu MiB allocation was REFUSED (see the strata vulkan line above); falling back to two of %llu MiB\n",
            (unsigned long long) (floor_bytes / kMiB), (unsigned long long) ((floor_bytes / 2) / kMiB));
        bool ok = true;
        for (int i = 0; i < 2; ++i) {
            void* p = nullptr;
            if (gpu::malloc_device_raw(&p, floor_bytes / 2) != gpu::kSuccess) { ok = false; break; }
            floor_parts.push_back(p);
            say("  floor[%d] %llu MiB at device address 0x%llx\n", i, (unsigned long long) ((floor_bytes / 2) / kMiB),
                (unsigned long long) (uintptr_t) p);
        }
        if (!ok) {
            bad("case B floor");
            for (void*& p : floor_parts) release_dev(p);
            floor_parts.clear();          // they are freed; the cleanup below must not free them a second time
        } else if (!floor_parts.empty()) {
            floor = floor_parts[0];
            floor_base = (uint64_t) (uintptr_t) floor_parts[0];   // the distance in case B1 is measured from here
        }
    }
    mem_line("after the floor");

    if (floor != nullptr) {
        // The 4 MiB buffers, allocated AFTER the floor: the sub-allocator places them in blocks after it, which is what
        // puts dst's device address past 4 GiB of the device's address space.
        Buf b_dst = alloc_dev(kBufBytes, "dst");
        Buf b_src = alloc_dev(kBufBytes, "src");
        mem_line("after case B's buffers");
        if (b_dst.cpu != nullptr && b_src.cpu != nullptr) {
            const uint64_t past = b_dst.addr - floor_base;
            say("  case B1 (dst allocated after the floor): dst is %llu MiB past the floor's base address "
                "(%s 4 GiB)\n", (unsigned long long) (past / kMiB), past > 4ull * 1024 * kMiB ? "past" : "short of");
            // And what the record_dispatch lines above actually say about it: dst's binding offset is 0 MiB, because
            // this runtime gives every allocation its own block and every block is clamped to the card's 4095 MiB.  A
            // binding at an offset PAST 4 GiB inside one block therefore cannot be produced through this sub-allocator
            // at all; Phase 0 produced one on a raw 5120 MiB VkBuffer (P2b), which is exactly the shape this port
            // refuses to create.  The consequence for a tensor bigger than 4095 MiB is several allocations with the
            // offsets carried by the caller, not one buffer whose interior reaches past 4 GiB.
            say("  case B1: the device ADDRESS reached past 4 GiB and the add over it was correct; the binding OFFSET "
                "stayed at 0 MiB inside dst's own block, by design\n");
            run_case("case B1", b_dst, b_src, kN, 0, 0);
        } else {
            bad("case B1");
        }
        release(b_dst);
        release(b_src);

        // B2 is the shape the engine actually uses: one big allocation, and a pointer arithmetic'd past 4 GiB inside
        // it.  Only possible with the single floor (a pointer past 4 GiB inside one allocation).
        if (floor_one) {
            const uint64_t inside = floor_bytes - 8ull * kMiB;      // 4192 MiB: past 4096 MiB, 8 MiB from the end
            void* b2_dst = (void*) (uintptr_t) (((char*) floor) + inside);
            say("  case B2 (dst is a pointer INSIDE the one allocation): 0x%llx = the floor + %llu MiB, i.e. %llu MiB "
                "past its base\n", (unsigned long long) (uintptr_t) b2_dst, (unsigned long long) (inside / kMiB),
                (unsigned long long) (inside / kMiB));
            Buf b2_d;
            b2_d.dev = b2_dst;
            b2_d.addr = (uint64_t) (uintptr_t) b2_dst;
            b2_d.bytes = kBufBytes;
            b2_d.cpu = (float*) gpu::host_view(b2_dst);
            Buf b2_s = alloc_dev(kBufBytes, "src");
            if (b2_d.cpu == nullptr) {
                bad("case B2 (host_view of the past-4-GiB pointer returned null)");
            } else {
                for (int64_t i = 0; i < kN; ++i) { b2_d.cpu[i] = dst_at(i); b2_s.cpu[i] = src_at(i); }
                strata::kernels::add_inplace((float*) b2_dst, (const float*) b2_s.dev, kN, nullptr);
                if (gpu::device_synchronize() != gpu::kSuccess) {
                    bad("case B2");
                } else {
                    int64_t mismatches = 0, first_bad = -1;
                    for (int64_t i = 0; i < kN; ++i) {
                        const float want_b2 = sum_at(i);
                        if (std::memcmp(&b2_d.cpu[i], &want_b2, sizeof(float)) != 0) {
                            if (first_bad < 0) first_bad = i;
                            ++mismatches;
                        }
                    }
                    say("  case B2: n %lld, %lld mismatch(es), dst at %llu MiB inside the allocation, its binding "
                        "offset printed above\n", (long long) kN, (long long) mismatches,
                        (unsigned long long) (inside / kMiB));
                    if (first_bad >= 0)
                        say("      first mismatch at i %lld: got %.9g, the host expected %.9g\n",
                            (long long) first_bad, (double) b2_d.cpu[first_bad], (double) sum_at(first_bad));
                    if (mismatches != 0) bad("case B2"); else say("  case B2: PASS\n");
                }
            }
            release(b2_s);
            // b2_d is a window into `floor`; it is freed with the floor, never separately.
        }
    }

    // On the fallback route `floor` IS floor_parts[0], so it is freed once, through the vector.
    if (floor_one) release_dev(floor);
    for (void*& p : floor_parts) release_dev(p);
    mem_line("after every free");
    say("\npeak device usage this run caused: %llu MiB\n", (unsigned long long) g_peak_used_mib);

    if (g_fail != 0) {
        say("\nstrata_vk_parity: %d failure(s)\n", g_fail);
        return 1;
    }
    say("\nstrata_vk_parity OK: 0 failure(s)\n");
    return 0;
}
