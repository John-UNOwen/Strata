// Phase 0 capability probe for VULKAN-BACKEND-PLAN.md.
// No model, no inference, no server: allocations, tiny shaders, timing, limits.
// The user runs it (start-vulkan-probe.bat); the agent reads the log.
//   P1   devices, limits, memory types and budgets, the mapping table
//   P2   one big device-local allocation, written and read past 4 GiB through a buffer device address
//   P2b  the same buffer bound the ordinary way (descriptor offset and range past 4 GiB)
//   P3   one host allocation imported by both cards, read in place, each card writes its own half
//   P4   the cost of a 2,400 dispatch command buffer replayed 100 times
//   P5   a flag the host raises while a shader polls host coherent memory
//   P6   cooperative matrix shapes and packed 8 bit dot product

#ifdef _WIN32
// windows.h arrives through vulkan/vulkan.h, so the guards have to come first: the min and max
// macros otherwise break std::min and std::max below.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "shaders/tag_write.h"
#include "shaders/tag_read.h"
#include "shaders/span_write.h"
#include "shaders/span_read.h"
#include "shaders/checksum.h"
#include "shaders/write_half.h"
#include "shaders/noop.h"
#include "shaders/spin.h"

namespace {

const size_t KiB = 1024ull;
const size_t MiB = 1024ull * KiB;
const size_t GiB = 1024ull * MiB;

const uint64_t kMagic = 0xa5a5a5a5a5a5a5a5ull;   // write_half marks bytes the GPU wrote

const char* rname(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "INITIALIZATION_FAILED";
        case VK_ERROR_MEMORY_MAP_FAILED: return "MEMORY_MAP_FAILED";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "FEATURE_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "EXTENSION_NOT_PRESENT";
        case VK_ERROR_TOO_MANY_OBJECTS: return "TOO_MANY_OBJECTS";
        case VK_ERROR_DEVICE_LOST: return "DEVICE_LOST (TDR or driver reset)";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "FRAGMENTATION";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        default: return "other";
    }
}

const char* comp_name(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "float32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "float64";
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "float16";
        case VK_COMPONENT_TYPE_SINT8_KHR: return "int8";
        case VK_COMPONENT_TYPE_SINT16_KHR: return "int16";
        case VK_COMPONENT_TYPE_SINT32_KHR: return "int32";
        case VK_COMPONENT_TYPE_SINT64_KHR: return "int64";
        case VK_COMPONENT_TYPE_UINT8_KHR: return "uint8";
        case VK_COMPONENT_TYPE_UINT16_KHR: return "uint16";
        case VK_COMPONENT_TYPE_UINT32_KHR: return "uint32";
        case VK_COMPONENT_TYPE_UINT64_KHR: return "uint64";
        case VK_COMPONENT_TYPE_BFLOAT16_KHR: return "bfloat16";
        default: return "other";
    }
}

uint64_t mix(uint64_t x) {   // the tag pattern: host and shader must agree
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27; x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

void host_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    (void)ms;
#endif
}

// Push constants shared by the tag and checksum shaders.
// stride: bytes between samples. count: samples. a, b: mode dependent, said per shader.
struct PC { uint32_t stride; uint32_t count; uint32_t a; uint32_t b; };

struct Opts {
    double limit_gib = 7.0;       // P2 and P2b big allocation
    double alt_gib = 3.5;         // fallback if the first is refused
    double stride_mib = 64;       // distance between verified tags
    double view_off_mib = 5120;   // P2b: the region starts 5 GiB in
    double host_mib = 1024;       // P3 shared host buffer
    uint32_t spin_max = 2000000;  // P5 bounded spin, mirrors STRATA_SPIN_MAX
    uint32_t dispatches = 2400;   // P4, the size of a verify window graph
    uint32_t replays = 100;
    bool discrete_only = true;
    bool spans = false;          // P2 wide span walks: they fault the device on this driver
    std::string card_filter;     // only cards whose name contains this text
    std::vector<std::string> only;
    bool want(const char* n) const {
        if (only.empty()) return true;
        for (const std::string& s : only) if (s == n) return true;
        return false;
    }
};

struct Card {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    std::string name;
    VkPhysicalDeviceProperties props{};
    // Resolved by hand: the loader exports these under their core names, and the SDK import
    // library does not carry the KHR spellings.
    PFN_vkGetBufferDeviceAddress get_address = nullptr;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_memory = nullptr;
    bool lost = false;   // DEVICE_LOST is sticky for the device; the rest of the run is void after it
    std::vector<const char*> exts;
    bool ok_ext(const char* e) const {
        for (const char* x : exts) if (std::strcmp(x, e) == 0) return true;
        return false;
    }
};

struct Buf {
    VkBuffer b = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct Pipe {
    VkShaderModule mod = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

struct CmdPool {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    bool init(Card& c) {
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = c.qfam;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(c.dev, &pi, nullptr, &pool) != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        return vkAllocateCommandBuffers(c.dev, &ai, &cb) == VK_SUCCESS;
    }
    void destroy(Card& c) {
        if (pool) vkDestroyCommandPool(c.dev, pool, nullptr);
        pool = VK_NULL_HANDLE; cb = VK_NULL_HANDLE;
    }
};

bool ext_available(VkPhysicalDevice pd, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> av(n ? n : 1);
    if (n) vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, av.data());
    for (const auto& x : av) if (std::strcmp(x.extensionName, name) == 0) return true;
    return false;
}

bool find_mem(const Card& c, uint32_t bits, VkMemoryPropertyFlags want, uint32_t& out) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(c.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { out = i; return true; }
    return false;
}

VkResult make_buf(Card& c, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, Buf& out) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateBuffer(c.dev, &bi, nullptr, &out.b);
    if (r != VK_SUCCESS) return r;
    out.size = size;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(c.dev, out.b, &req);
    uint32_t mt = 0;
    if (!find_mem(c, req.memoryTypeBits, props, mt)) return VK_ERROR_FEATURE_NOT_PRESENT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    // The address flag is required by any buffer created with SHADER_DEVICE_ADDRESS_BIT. Without
    // it vkGetBufferDeviceAddress returns a value that is not a pointer and the shader faults.
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    ai.pNext = &flags;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    r = vkAllocateMemory(c.dev, &ai, nullptr, &out.m);
    if (r != VK_SUCCESS) return r;
    return vkBindBufferMemory(c.dev, out.b, out.m, 0);
}

void free_buf(Card& c, Buf& b) {
    if (b.m) { vkFreeMemory(c.dev, b.m, nullptr); b.m = VK_NULL_HANDLE; }
    if (b.b) { vkDestroyBuffer(c.dev, b.b, nullptr); b.b = VK_NULL_HANDLE; }
}

bool make_pipe(Card& c, const std::vector<uint8_t>& spv, size_t nbufs, Pipe& p) {
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = spv.size();
    mi.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    if (vkCreateShaderModule(c.dev, &mi, nullptr, &p.mod) != VK_SUCCESS) return false;

    std::vector<VkDescriptorSetLayoutBinding> bs(nbufs);
    for (size_t i = 0; i < nbufs; ++i) {
        bs[i].binding = (uint32_t)i;
        bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bs[i].descriptorCount = 1;
        bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)bs.size();
    li.pBindings = bs.data();
    if (vkCreateDescriptorSetLayout(c.dev, &li, nullptr, &p.dsl) != VK_SUCCESS) return false;

    const uint32_t pc_size = (uint32_t)sizeof(PC);
    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &p.dsl;
    if (pc_size <= c.props.limits.maxPushConstantsSize) { pi.pushConstantRangeCount = 1; pi.pPushConstantRanges = &pc; }
    if (vkCreatePipelineLayout(c.dev, &pi, nullptr, &p.layout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = p.mod;
    ci.stage.pName = "main";
    ci.layout = p.layout;
    if (vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nbufs};
    VkDescriptorPoolCreateInfo pi2{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi2.maxSets = 1;
    pi2.poolSizeCount = 1;
    pi2.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(c.dev, &pi2, nullptr, &p.pool) != VK_SUCCESS) return false;

    VkDescriptorSetAllocateInfo sa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sa.descriptorPool = p.pool;
    sa.descriptorSetCount = 1;
    sa.pSetLayouts = &p.dsl;
    return vkAllocateDescriptorSets(c.dev, &sa, &p.set) == VK_SUCCESS;
}

void free_pipe(Card& c, Pipe& p) {
    if (p.pool) vkDestroyDescriptorPool(c.dev, p.pool, nullptr);
    if (p.pipe) vkDestroyPipeline(c.dev, p.pipe, nullptr);
    if (p.layout) vkDestroyPipelineLayout(c.dev, p.layout, nullptr);
    if (p.dsl) vkDestroyDescriptorSetLayout(c.dev, p.dsl, nullptr);
    if (p.mod) vkDestroyShaderModule(c.dev, p.mod, nullptr);
    p = Pipe{};
}

void bind(Card& c, Pipe& p, const std::vector<VkDescriptorBufferInfo>& infos) {
    std::vector<VkWriteDescriptorSet> w(infos.size());
    for (size_t i = 0; i < infos.size(); ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = p.set;
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(c.dev, (uint32_t)w.size(), w.data(), 0, nullptr);
}

// Which Vulkan call last failed, so a verdict names the step instead of guessing.
const char* g_last_call = "";

bool begin_cmd(CmdPool& cp) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    g_last_call = "vkBeginCommandBuffer";
    return vkBeginCommandBuffer(cp.cb, &bi) == VK_SUCCESS;
}

VkResult end_submit_wait(Card& c, CmdPool& cp) {
    g_last_call = "vkEndCommandBuffer";
    if (vkEndCommandBuffer(cp.cb) != VK_SUCCESS) return VK_ERROR_INITIALIZATION_FAILED;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cp.cb;
    g_last_call = "vkQueueSubmit";
    VkResult r = vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) { if (r == VK_ERROR_DEVICE_LOST) c.lost = true; return r; }
    g_last_call = "vkQueueWaitIdle";
    r = vkQueueWaitIdle(c.queue);
    if (r == VK_ERROR_DEVICE_LOST) c.lost = true;
    return r;
}

// One pipeline, one dispatch, with push constants. `groups` workgroups of 64.
VkResult run_once(Card& c, CmdPool& cp, Pipe& p, const PC& pc, uint32_t groups) {
    if (!begin_cmd(cp)) return VK_ERROR_INITIALIZATION_FAILED;
    vkCmdBindPipeline(cp.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(cp.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0, nullptr);
    vkCmdPushConstants(cp.cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
    vkCmdDispatch(cp.cb, groups, 1, 1);
    return end_submit_wait(c, cp);
}

VkResult copy_out(Card& c, CmdPool& cp, VkBuffer src, Buf& stage, void** mapped, size_t bytes) {
    if (!begin_cmd(cp)) return VK_ERROR_INITIALIZATION_FAILED;
    VkBufferCopy cpy{0, 0, bytes};
    vkCmdCopyBuffer(cp.cb, src, stage.b, 1, &cpy);
    VkResult r = end_submit_wait(c, cp);
    if (r != VK_SUCCESS) return r;
    return vkMapMemory(c.dev, stage.m, 0, bytes, 0, mapped);
}

// Zero a device buffer from the host through a staging buffer.
VkResult zero_buf(Card& c, Buf& dst, size_t bytes) {
    Buf z;
    VkResult r = make_buf(c, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, z);
    if (r != VK_SUCCESS) return r;
    void* m = nullptr;
    r = vkMapMemory(c.dev, z.m, 0, bytes, 0, &m);
    if (r == VK_SUCCESS) { std::memset(m, 0, bytes); vkUnmapMemory(c.dev, z.m); }
    CmdPool cp; cp.init(c);
    if (r == VK_SUCCESS && begin_cmd(cp)) {
        VkBufferCopy cpy{0, 0, bytes};
        vkCmdCopyBuffer(cp.cb, z.b, dst.b, 1, &cpy);
        r = end_submit_wait(c, cp);
    }
    cp.destroy(c);
    free_buf(c, z);
    return r;
}

// Write u64 words into a device buffer from the host.
VkResult stage_words(Card& c, Buf& dst, const std::vector<uint64_t>& words) {
    const size_t bytes = words.size() * 8;
    Buf z;
    VkResult r = make_buf(c, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, z);
    if (r != VK_SUCCESS) return r;
    void* m = nullptr;
    r = vkMapMemory(c.dev, z.m, 0, bytes, 0, &m);
    if (r == VK_SUCCESS) { std::memcpy(m, words.data(), bytes); vkUnmapMemory(c.dev, z.m); }
    CmdPool cp; cp.init(c);
    if (r == VK_SUCCESS && begin_cmd(cp)) {
        VkBufferCopy cpy{0, 0, bytes};
        vkCmdCopyBuffer(cp.cb, z.b, dst.b, 1, &cpy);
        r = end_submit_wait(c, cp);
    }
    cp.destroy(c);
    free_buf(c, z);
    return r;
}

std::vector<uint8_t> spv_bytes(const unsigned char* src, unsigned int n) {
    return std::vector<uint8_t>(src, src + n);
}

// ------------------------------------------------------------------ P1

void print_table(const Card& c) {
    static const char* kWanted[] = {
        "VK_KHR_buffer_device_address", "VK_KHR_timeline_semaphore", "VK_KHR_maintenance4",
        "VK_KHR_maintenance5", "VK_KHR_vulkan_memory_model", "VK_KHR_external_memory",
        "VK_KHR_external_memory_capabilities", "VK_KHR_external_semaphore", "VK_EXT_external_memory_host",
        "VK_KHR_shader_integer_dot_product", "VK_EXT_subgroup_size_control", "VK_KHR_cooperative_matrix",
        "VK_KHR_16bit_storage", "VK_KHR_shader_float16_int8", "VK_KHR_8bit_storage",
        "VK_EXT_descriptor_indexing", "VK_EXT_memory_budget", "VK_EXT_host_query_reset",
        "VK_KHR_pipeline_executable_properties", "VK_EXT_shader_64bit_indexing"};
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(c.phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> av(n ? n : 1);
    if (n) vkEnumerateDeviceExtensionProperties(c.phys, nullptr, &n, av.data());
    printf("    the plan's mapping table:\n");
    for (const char* w : kWanted) {
        bool has = false;
        for (const auto& x : av) if (std::strcmp(x.extensionName, w) == 0) { has = true; break; }
        printf("      %-38s %s\n", w, has ? "present" : "absent");
    }
}

void test_p1(const std::vector<Card>& cards) {
    for (const Card& c : cards) {
        const VkPhysicalDeviceProperties& p = c.props;
        printf("  %s\n", c.name.c_str());
        printf("    api %u.%u.%u  vendor %04x  device %04x\n",
               VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion),
               p.vendorID, p.deviceID);
        VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceMaintenance4Properties m4{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
        VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
        VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR};
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &drv; drv.pNext = &m4; m4.pNext = &m3; m3.pNext = &host; host.pNext = &idp; idp.pNext = &sg;
        vkGetPhysicalDeviceProperties2(c.phys, &p2);
        printf("    driver \"%s\" id %u  \"%s\"\n", drv.driverName, drv.driverID, drv.driverInfo);
        printf("    limits: maxMemoryAllocationCount %u  maxStorageBufferRange %llu MiB  maxBufferSize %llu MiB  maxMemoryAllocationSize %llu MiB\n",
               p.limits.maxMemoryAllocationCount, (unsigned long long)(p.limits.maxStorageBufferRange / MiB),
               (unsigned long long)(m4.maxBufferSize / MiB), (unsigned long long)(m3.maxMemoryAllocationSize / MiB));
        printf("    alignment: minStorageBufferOffsetAlignment %llu  bufferImageGranularity %llu  nonCoherentAtomSize %llu\n",
               (unsigned long long)p.limits.minStorageBufferOffsetAlignment,
               (unsigned long long)p.limits.bufferImageGranularity,
               (unsigned long long)p.limits.nonCoherentAtomSize);
        printf("    subgroup: maxComputeWorkGroupInvocations %u  subgroupSize %u  maxComputeWorkGroupCount %u\n",
               p.limits.maxComputeWorkGroupInvocations, sg.subgroupSize, p.limits.maxComputeWorkGroupCount[0]);
        printf("    external_memory_host: minImportedHostPointerAlignment %llu KiB\n",
               (unsigned long long)(host.minImportedHostPointerAlignment / KiB));

        VkPhysicalDeviceFeatures2 feats2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR idpf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR};
        VkPhysicalDeviceScalarBlockLayoutFeatures sbl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES};
        feats2.pNext = &bda; bda.pNext = &cmf; cmf.pNext = &idpf; idpf.pNext = &sbl;
        vkGetPhysicalDeviceFeatures2(c.phys, &feats2);
        printf("    features: bufferDeviceAddress %s  cooperativeMatrix %s  integerDotProduct %s  scalarBlockLayout %s  robustBufferAccess %s\n",
               bda.bufferDeviceAddress ? "yes" : "no",
               cmf.cooperativeMatrix ? "yes" : "no", idpf.shaderIntegerDotProduct ? "yes" : "no",
               sbl.scalarBlockLayout ? "yes" : "no", feats2.features.robustBufferAccess ? "yes" : "no");
        if (p.apiVersion >= VK_MAKE_VERSION(1, 4, 0))
            printf("    note: api 1.4, so maxStorageBufferRange may be lifted by shader64BitIndexing. P2b measures the behaviour.\n");

        VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bud{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        if (c.ok_ext("VK_EXT_memory_budget")) mp2.pNext = &bud;
        vkGetPhysicalDeviceMemoryProperties2(c.phys, &mp2);
        const VkPhysicalDeviceMemoryProperties& mp = mp2.memoryProperties;
        for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
            printf("    heap %u: %llu MiB %s", i, (unsigned long long)(mp.memoryHeaps[i].size / MiB),
                   (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "device-local" : "not device local");
            if (mp2.pNext)
                printf(", budget %llu MiB, used now %llu MiB\n",
                       (unsigned long long)(bud.heapBudget[i] / MiB), (unsigned long long)(bud.heapUsage[i] / MiB));
            else printf(", no VK_EXT_memory_budget\n");
        }
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            printf("    type %2u heap %u: %s%s%s%s%s\n", i, mp.memoryTypes[i].heapIndex,
                   (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEV_LOCAL " : "",
                   (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "",
                   (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "",
                   (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED " : "",
                   (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? "LAZY" : "");
        }
        print_table(c);
    }
}

// ------------------------------------------------ P2 and P2b: past 4 GiB on the A770

struct Tags { uint32_t ok = 0, bad = 0; };

Tags check_tags(Card& c, Buf& res, size_t count) {
    Tags t;
    CmdPool cp; cp.init(c);
    Buf stage;
    make_buf(c, count * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage);
    void* m = nullptr;
    if (copy_out(c, cp, res.b, stage, &m, count * 16) == VK_SUCCESS) {
        const uint64_t* rows = (const uint64_t*)m;
        for (uint32_t i = 0; i < count; ++i) {
            if (rows[i * 2] == 1ull && rows[i * 2 + 1] == mix(i)) { ++t.ok; continue; }
            if (t.bad < 4)
                printf("      tag %u wrong: flag %llu value %llu expected %llu\n", i,
                       (unsigned long long)rows[i * 2], (unsigned long long)rows[i * 2 + 1],
                       (unsigned long long)mix(i));
            ++t.bad;
        }
        vkUnmapMemory(c.dev, stage.m);
    } else {
        t.bad = (uint32_t)count;
    }
    cp.destroy(c);
    free_buf(c, stage);
    return t;
}

bool test_p2(Card& c, const Opts& o) {
    VkDeviceSize want = (VkDeviceSize)(o.limit_gib * (double)GiB);
    Buf big;
    // SHADER_DEVICE_ADDRESS_BIT is what makes vkGetBufferDeviceAddress legal. Without it the
    // query returns garbage, the shader writes through a bad pointer and the device is lost.
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    // Control first. If a small buffer cannot be reached through its own device address, the
    // big test says nothing about 4 GiB and everything about the address path.
    {
        Buf small, a2, cres;
        const uint32_t cstride = 256u * 1024u, ccount = 8;
        if (make_buf(c, 4ull * MiB, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, small) == VK_SUCCESS &&
            make_buf(c, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a2) == VK_SUCCESS &&
            make_buf(c, ccount * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, cres) == VK_SUCCESS) {
            zero_buf(c, cres, ccount * 16);
            VkBufferDeviceAddressInfo si{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            si.buffer = small.b;
            const uint64_t sa = c.get_address ? c.get_address(c.dev, &si) : 0;
            stage_words(c, a2, {sa});
            Pipe pw, pr;
            const bool built = make_pipe(c, spv_bytes(tag_write_spv, tag_write_spv_len), 2, pw) &&
                               make_pipe(c, spv_bytes(tag_read_spv, tag_read_spv_len), 2, pr);
            if (built) {
                bind(c, pw, {{a2.b, 0, VK_WHOLE_SIZE}, {cres.b, 0, VK_WHOLE_SIZE}});
                bind(c, pr, {{a2.b, 0, VK_WHOLE_SIZE}, {cres.b, 0, VK_WHOLE_SIZE}});
                CmdPool cp; cp.init(c);
                PC pc{cstride, ccount, 0, 0};
                VkResult cw = run_once(c, cp, pw, pc, (ccount + 63) / 64);
                VkResult cr = cw == VK_SUCCESS ? run_once(c, cp, pr, pc, (ccount + 63) / 64) : VK_ERROR_INITIALIZATION_FAILED;
                uint32_t cbad = ccount;
                if (cr == VK_SUCCESS) cbad = check_tags(c, cres, ccount).bad;
                printf("    control: a 4 MiB buffer at address %llu  write %-14s read %-14s  tags %s\n",
                       (unsigned long long)sa, rname(cw), rname(cr), cbad == 0 ? "all correct" : (std::to_string(cbad) + " wrong").c_str());
                if (cw == VK_ERROR_DEVICE_LOST || cr == VK_ERROR_DEVICE_LOST) c.lost = true;
                cp.destroy(c);
                free_pipe(c, pw); free_pipe(c, pr);
            }
        }
        free_buf(c, small); free_buf(c, a2); free_buf(c, cres);
    }
    if (c.lost) { printf("    the device was already lost by the control, the big test cannot run\n"); return false; }

    VkResult r = make_buf(c, want, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, big);
    if (r != VK_SUCCESS) {
        printf("    %llu MiB device local allocation: %s\n", (unsigned long long)(want / MiB), rname(r));
        want = (VkDeviceSize)(o.alt_gib * (double)GiB);
        r = make_buf(c, want, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, big);
        if (r != VK_SUCCESS) { printf("    fallback %llu MiB also failed: %s\n", (unsigned long long)(want / MiB), rname(r)); return false; }
        printf("    fallback %llu MiB allocated: %s\n", (unsigned long long)(want / MiB), rname(r));
    } else {
        printf("    %llu MiB device local allocation: ok\n", (unsigned long long)(want / MiB));
    }

    const uint32_t stride = (uint32_t)(o.stride_mib * MiB);
    const uint32_t count = (uint32_t)std::min<uint64_t>(want / stride, 256);
    if ((uint64_t)count * stride > want) { printf("    buffer too small for the tag walk\n"); free_buf(c, big); return false; }
    printf("    walking %u tags %u MiB apart, the last one at %llu MiB, %llu MiB from the end\n", count, stride / (uint32_t)MiB,
           (unsigned long long)((count - 1) * (uint64_t)stride / MiB), (unsigned long long)((want - (uint64_t)count * stride) / MiB));

    Buf addr, res, stage;
    make_buf(c, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, addr);
    make_buf(c, (VkDeviceSize)count * 16,
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, res);
    make_buf(c, (VkDeviceSize)count * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage);
    zero_buf(c, res, (size_t)count * 16);

    VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    dai.buffer = big.b;
    const uint64_t address = c.get_address ? c.get_address(c.dev, &dai) : 0;
    printf("    buffer device address query: %s, returned %llu (0 means it failed)\n",
           c.get_address ? "resolved" : "NOT resolved", (unsigned long long)address);
    if (address == 0) { free_buf(c, big); free_buf(c, addr); free_buf(c, res); free_buf(c, stage); return false; }
    stage_words(c, addr, {address});

    Pipe pw, pr;
    const bool built = make_pipe(c, spv_bytes(tag_write_spv, tag_write_spv_len), 2, pw) &&
                       make_pipe(c, spv_bytes(tag_read_spv, tag_read_spv_len), 2, pr);
    if (!built) {
        printf("    tag pipelines failed\n");
        free_pipe(c, pw); free_pipe(c, pr);
        free_buf(c, big); free_buf(c, addr); free_buf(c, res); free_buf(c, stage);
        return false;
    }
    bind(c, pw, {{addr.b, 0, VK_WHOLE_SIZE}, {res.b, 0, VK_WHOLE_SIZE}});
    bind(c, pr, {{addr.b, 0, VK_WHOLE_SIZE}, {res.b, 0, VK_WHOLE_SIZE}});

    CmdPool cp; cp.init(c);
    // One tag per step at a rising byte offset, ascending so the first fault still leaves the
    // whole ladder below it on the record.
    struct Walk { uint64_t base; };
    const uint64_t steps[] = {0, 1, 8, 64, 256, 512, 1024, 2048, 3072, 3968, 4096, 4608, 5056};
    bool low_ok = false, past_ok = false;
    uint64_t max_ok = 0;
    bool reached = false;
    for (uint64_t mib : steps) {
        if (mib * MiB >= want) break;
        if (c.lost) { printf("    offset %llu MiB: skipped, the device is already lost\n", (unsigned long long)mib); continue; }
        zero_buf(c, res, (size_t)count * 16);
        PC pc{stride, 1, (uint32_t)(mib * MiB & 0xffffffffull), (uint32_t)((mib * MiB) >> 32)};
        VkResult rw = run_once(c, cp, pw, pc, 1);
        VkResult rr = rw == VK_SUCCESS ? run_once(c, cp, pr, pc, 1) : VK_ERROR_INITIALIZATION_FAILED;
        bool good = false;
        if (rr == VK_SUCCESS) good = (check_tags(c, res, 1).bad == 0);
        printf("    tag written through the address at %5llu MiB: write %-16s read %-16s tag %s\n",
               (unsigned long long)mib, rname(rw), rname(rr), good ? "correct" : "wrong");
        if (rw == VK_ERROR_DEVICE_LOST || rr == VK_ERROR_DEVICE_LOST) { c.lost = true; reached = true; break; }
        if (good) {
            max_ok = mib;
            if (mib < 4096) low_ok = true; else past_ok = true;
        }
    }
    // Second phase: how far apart the touches may be inside one dispatch. The ladder above
    // touches one place per dispatch, so it cannot show a span limit. Opt in with --spans: on
    // this driver a wide span faults the device, so the gate runs leave it out.
    struct Span { uint32_t tags; uint32_t stride_mib; };
    const Span spans[] = {{8, 1}, {64, 1}, {16, 8}, {40, 64}, {80, 64}};
    if (o.spans) for (const Span& s : spans) {
        if (c.lost) { printf("    %u tags %u MiB apart: skipped, the device is already lost\n", s.tags, s.stride_mib); continue; }
        if ((uint64_t)s.tags * s.stride_mib * MiB > want) continue;
        zero_buf(c, res, (size_t)count * 16);
        PC pc{s.stride_mib * (uint32_t)MiB, s.tags, 0, 0};
        VkResult rw = run_once(c, cp, pw, pc, (s.tags + 63) / 64);
        VkResult rr = rw == VK_SUCCESS ? run_once(c, cp, pr, pc, (s.tags + 63) / 64) : VK_ERROR_INITIALIZATION_FAILED;
        bool good = false;
        if (rr == VK_SUCCESS) good = (check_tags(c, res, s.tags).bad == 0);
        printf("    %3u tags written %4u MiB apart over %5llu MiB in one dispatch: %s %s\n", s.tags, s.stride_mib,
               (unsigned long long)((s.tags - 1) * (uint64_t)s.stride_mib / 1), rname(rr), good ? "all tags correct" : "not correct");
        if (rw == VK_ERROR_DEVICE_LOST || rr == VK_ERROR_DEVICE_LOST) { c.lost = true; break; }
    }
    printf("    P2 verdict: the address path reaches at least %llu MiB into the buffer; under 4 GiB %s, past 4 GiB %s%s\n",
           (unsigned long long)max_ok, low_ok ? "works" : "does not work", past_ok ? "works" : "does not work",
           reached ? " (the walk stopped at a device fault)" : "");
    cp.destroy(c);
    free_pipe(c, pw); free_pipe(c, pr);
    free_buf(c, big); free_buf(c, addr); free_buf(c, res); free_buf(c, stage);
    return past_ok;
}

// The question the SYCL engine pays staged_gemm for: can a buffer bigger than the
// device's own range limit be addressed the ordinary way, from an offset past 4 GiB.
bool test_p2b(Card& c, const Opts& o) {
    const uint32_t stride = (uint32_t)(o.stride_mib * MiB);
    const uint32_t count = 8;
    const VkDeviceSize off = (VkDeviceSize)(o.view_off_mib * MiB);
    // The buffer only has to reach past the region the test binds, so its size follows
    // --view-off-mib. That keeps the allocation inside the agreed per-card ceiling.
    const VkDeviceSize size = off + (VkDeviceSize)count * stride + 256ull * MiB;
    Buf res;
    make_buf(c, (VkDeviceSize)count * 16,
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, res);
    zero_buf(c, res, (size_t)count * 16);

    const VkDeviceSize half = (VkDeviceSize)count * stride;   // the region one pass touches
    struct Variant { const char* name; VkDeviceSize alloc; VkDeviceSize off; VkDeviceSize range; bool decides; };
    const Variant vars[] = {
        {"1 GiB buffer whole (control)", 1ull * GiB, 0, VK_WHOLE_SIZE, false},
        {"1 GiB buffer, offset 512 MiB", 1ull * GiB, 512ull * MiB, half, false},
        {"big buffer whole, VK_WHOLE_SIZE", size, 0, VK_WHOLE_SIZE, false},
        {"big buffer, explicit range", size, 0, size, false},
        {"big buffer, offset 1 GiB", size, 1ull * GiB, half, false},
        {"big buffer, offset under 4 GiB", size, 3968ull * MiB, half, false},
        {"big buffer, offset past 4 GiB", size, off, half, true},
    };
    bool offset_ok = false;
    for (const Variant& v : vars) {
        const VkDeviceSize eff = (v.range == VK_WHOLE_SIZE) ? (v.alloc - v.off) : v.range;
        const bool over = (eff > c.props.limits.maxStorageBufferRange);
        const bool s64 = ext_available(c.phys, "VK_EXT_shader_64bit_indexing");
        const std::string range_txt = v.range == VK_WHOLE_SIZE ? std::string("WHOLE_SIZE")
                                                               : std::to_string(v.range / MiB) + " MiB";
        if (over && !s64) {
            printf("    P2b %-34s alloc %5llu MiB offset %5llu MiB range %-11s  SKIPPED: the range passes maxStorageBufferRange "
                   "%llu MiB and VK_EXT_shader_64bit_indexing is absent, so this binding is invalid by spec\n",
                   v.name, (unsigned long long)(v.alloc / MiB), (unsigned long long)(v.off / MiB), range_txt.c_str(),
                   (unsigned long long)(c.props.limits.maxStorageBufferRange / MiB));
            continue;
        }
        Buf big;
        if (make_buf(c, v.alloc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, big) != VK_SUCCESS) {
            printf("    P2b %-34s alloc %5llu MiB: the allocation failed at %s\n", v.name,
                   (unsigned long long)(v.alloc / MiB), g_last_call);
            continue;
        }
        Pipe pw, pr;
        const bool built = make_pipe(c, spv_bytes(span_write_spv, span_write_spv_len), 2, pw) &&
                           make_pipe(c, spv_bytes(span_read_spv, span_read_spv_len), 2, pr);
        if (!built) { printf("    variant %s: pipelines failed\n", v.name); free_pipe(c, pw); free_pipe(c, pr); free_buf(c, big); continue; }
        bind(c, pw, {{big.b, v.off, v.range}, {res.b, 0, VK_WHOLE_SIZE}});
        bind(c, pr, {{big.b, v.off, v.range}, {res.b, 0, VK_WHOLE_SIZE}});
        CmdPool cp; cp.init(c);
        PC pc{stride, count, 0, 0};
        VkResult rw = run_once(c, cp, pw, pc, (count + 63) / 64);
        VkResult rr = rw == VK_SUCCESS ? run_once(c, cp, pr, pc, (count + 63) / 64) : VK_ERROR_INITIALIZATION_FAILED;
        uint32_t bad = count;
        if (rr == VK_SUCCESS) { Tags t = check_tags(c, res, count); bad = t.bad; }
        printf("    P2b %-34s alloc %5llu MiB offset %5llu MiB range %-11s  write %-14s read %-14s at %s  tags %s\n",
               v.name, (unsigned long long)(v.alloc / MiB), (unsigned long long)(v.off / MiB), range_txt.c_str(),
               rname(rw), rname(rr), g_last_call, bad == 0 ? "all correct" : (std::to_string(bad) + " wrong").c_str());
        if (v.decides && rr == VK_SUCCESS && bad == 0) offset_ok = true;
        zero_buf(c, res, (size_t)count * 16);
        cp.destroy(c);
        free_pipe(c, pw); free_pipe(c, pr);
        free_buf(c, big);
    }
    printf("    P2b verdict: %s\n", offset_ok ? "a region past 4 GiB works the ordinary way, no staging copy is needed"
                                              : "it does not work the ordinary way, the staged layout stays necessary");
    free_buf(c, res);
    return offset_ok;
}

// ------------------------------------------- P3: one host allocation, both cards, in place

bool test_p3(std::vector<Card>& cards, const Opts& o) {
#ifdef _WIN32
    if (cards.size() < 2) { printf("    P3 needs two cards\n"); return false; }
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &host;
    vkGetPhysicalDeviceProperties2(cards[0].phys, &p2);
    if (host.minImportedHostPointerAlignment == 0) { printf("    P3: VK_EXT_external_memory_host not reported\n"); return false; }
    const size_t align = (size_t)host.minImportedHostPointerAlignment;
    const size_t bytes = ((size_t)(o.host_mib * MiB) + align - 1) & ~(align - 1);
    const uint64_t samples = 4096;
    const uint64_t step = (bytes / 8) / samples;
    auto expect_host = [&](uint64_t i) { return mix(i * step); };
    auto expect_gpu = [&](uint64_t i) { return mix(i) ^ kMagic; };
    printf("    host buffer %llu MiB, import alignment %llu KiB, %llu samples %llu bytes apart\n",
           (unsigned long long)(bytes / MiB), (unsigned long long)(align / KiB), samples, step * 8);

    uint8_t* mem = (uint8_t*)VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!mem) { printf("    VirtualAlloc of %llu MiB failed\n", (unsigned long long)(bytes / MiB)); return false; }
    for (uint64_t i = 0; i < samples; ++i) *(uint64_t*)(mem + i * step * 8) = expect_host(i);

    struct Side { Card* c; Buf buf, res; Pipe rd, wr; bool ready = false; };
    std::vector<Side> sides(cards.size());
    bool ok = true;
    for (size_t d = 0; d < cards.size() && ok; ++d) {
        Card& c = cards[d];
        Side& s = sides[d]; s.c = &c;
        if (!c.ok_ext("VK_EXT_external_memory_host")) { printf("    %s lacks VK_EXT_external_memory_host\n", c.name.c_str()); ok = false; break; }
        VkPhysicalDeviceExternalBufferInfo ebi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
        ebi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        ebi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkExternalBufferProperties ebp{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        vkGetPhysicalDeviceExternalBufferProperties(c.phys, &ebi, &ebp);
        printf("    %s: host allocation handle type %s, features 0x%x (importable = %s)\n", c.name.c_str(),
               (ebp.externalMemoryProperties.compatibleHandleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT) ? "compatible" : "NOT compatible",
               ebp.externalMemoryProperties.externalMemoryFeatures,
               (ebp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) ? "yes" : "no");

        VkExternalMemoryBufferCreateInfo extb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        extb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.pNext = &extb;
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(c.dev, &bi, nullptr, &s.buf.b) != VK_SUCCESS) { printf("    buffer create failed\n"); ok = false; break; }
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(c.dev, s.buf.b, &req);
        // For a host pointer import the driver names the allowed memory types; picking any other
        // one is invalid and the bind then fails.
        VkMemoryHostPointerPropertiesEXT hpp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        if (!c.get_host_memory ||
            c.get_host_memory(c.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, mem, &hpp) != VK_SUCCESS) {
            printf("    %s: vkGetMemoryHostPointerPropertiesEXT failed\n", c.name.c_str());
            ok = false; break;
        }
        printf("    %s: host pointer allowed memoryTypeBits 0x%x (buffer requirements 0x%x)\n",
               c.name.c_str(), hpp.memoryTypeBits, req.memoryTypeBits);
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(c.phys, &mp);
        VkResult r = VK_ERROR_FEATURE_NOT_PRESENT;
        for (uint32_t t = 0; t < mp.memoryTypeCount && r != VK_SUCCESS; ++t) {
            if (!((req.memoryTypeBits & hpp.memoryTypeBits) & (1u << t))) continue;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
            imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
            imp.pHostPointer = mem;
            ai.pNext = &imp;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = t;
            r = vkAllocateMemory(c.dev, &ai, nullptr, &s.buf.m);
            if (r == VK_SUCCESS) {
                printf("    %s imported the host allocation on memory type %u\n", c.name.c_str(), t);
                s.ready = true;
            }
        }
        if (!s.ready) { printf("    %s could not import the host allocation: %s\n", c.name.c_str(), rname(r)); ok = false; break; }
        if (vkBindBufferMemory(c.dev, s.buf.b, s.buf.m, 0) != VK_SUCCESS) { printf("    bind of imported memory failed\n"); ok = false; break; }
        make_buf(c, samples * 16,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, s.res);
        zero_buf(c, s.res, (size_t)samples * 16);
        s.ready = make_pipe(c, spv_bytes(checksum_spv, checksum_spv_len), 2, s.rd) &&
                  make_pipe(c, spv_bytes(write_half_spv, write_half_spv_len), 2, s.wr);
        if (!s.ready) { printf("    %s: P3 pipelines failed\n", c.name.c_str()); ok = false; break; }
        bind(c, s.rd, {{s.buf.b, 0, VK_WHOLE_SIZE}, {s.res.b, 0, VK_WHOLE_SIZE}});
        bind(c, s.wr, {{s.buf.b, 0, VK_WHOLE_SIZE}, {s.res.b, 0, VK_WHOLE_SIZE}});
    }

    if (ok) {
        const uint64_t half = samples / 2;
        // each card reads what the host wrote, in place
        for (size_t d = 0; d < cards.size() && ok; ++d) {
            Card& c = cards[d]; CmdPool cp; cp.init(c);
            PC pc{(uint32_t)(step * 8), (uint32_t)samples, 0, 0};
            VkResult r = run_once(c, cp, sides[d].rd, pc, (uint32_t)((samples + 63) / 64));
            uint64_t bad = samples;
            if (r == VK_SUCCESS) {
                Buf stage;
                make_buf(c, samples * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage);
                void* m = nullptr;
                if (copy_out(c, cp, sides[d].res.b, stage, &m, samples * 16) == VK_SUCCESS) {
                    const uint64_t* rows = (const uint64_t*)m;
                    bad = 0;
                    for (uint64_t i = 0; i < samples; ++i) if (rows[i * 2 + 1] != expect_host(i)) ++bad;
                    vkUnmapMemory(c.dev, stage.m);
                }
                free_buf(c, stage);
            }
            printf("    %s read the host buffer in place: %llu of %llu samples correct\n", c.name.c_str(),
                   (unsigned long long)(samples - bad), samples);
            if (bad) ok = false;
            cp.destroy(c);
        }
        // each card writes its own half, host waits between, then the host view is checked
        for (size_t d = 0; d < cards.size() && ok; ++d) {
            Card& c = cards[d]; CmdPool cp; cp.init(c);
            PC pc{(uint32_t)(step * 8), (uint32_t)samples, (uint32_t)(d * half), (uint32_t)((d + 1) * half)};
            VkResult r = run_once(c, cp, sides[d].wr, pc, (uint32_t)((samples + 63) / 64));
            if (r != VK_SUCCESS) { printf("    %s write failed: %s\n", c.name.c_str(), rname(r)); ok = false; }
            cp.destroy(c);
            host_sleep_ms(50);
        }
        uint64_t seen = 0;
        for (uint64_t i = 0; i < samples; ++i) if (*(uint64_t*)(mem + i * step * 8) == expect_gpu(i)) ++seen;
        printf("    host view after the GPU writes: %llu of %llu samples carry what the GPUs wrote\n", seen, samples);
        if (seen != samples) ok = false;
    }
    for (size_t d = 0; d < cards.size(); ++d) {
        Card& c = cards[d];
        free_pipe(c, sides[d].rd); free_pipe(c, sides[d].wr);
        free_buf(c, sides[d].buf); free_buf(c, sides[d].res);
    }
    VirtualFree(mem, 0, MEM_RELEASE);
    return ok;
#else
    (void)cards; (void)o;
    printf("    P3 in this probe is Windows only\n");
    return false;
#endif
}

// --------------------------------- P4: the cost of a window sized command buffer

bool test_p4(Card& c, const Opts& o) {
    Buf sink_buf;
    make_buf(c, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sink_buf);
    Pipe p;
    if (!make_pipe(c, spv_bytes(noop_spv, noop_spv_len), 1, p)) { printf("    pipeline failed\n"); free_buf(c, sink_buf); return false; }
    const std::vector<VkDescriptorBufferInfo> infos{VkDescriptorBufferInfo{sink_buf.b, 0, VK_WHOLE_SIZE}};
    bind(c, p, infos);
    CmdPool cp; cp.init(c);
    std::vector<double> ms;
    bool ok = true;
    for (uint32_t rep = 0; rep < o.replays && ok; ++rep) {
        if (!begin_cmd(cp)) { ok = false; break; }
        vkCmdBindPipeline(cp.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        vkCmdBindDescriptorSets(cp.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0, nullptr);
        for (uint32_t i = 0; i < o.dispatches; ++i) {
            PC pc{i, 0, 0, 0};
            vkCmdPushConstants(cp.cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
            vkCmdDispatch(cp.cb, 1, 1, 1);
        }
        auto t0 = std::chrono::steady_clock::now();
        VkResult r = end_submit_wait(c, cp);
        auto t1 = std::chrono::steady_clock::now();
        if (r != VK_SUCCESS) { printf("    submit %u: %s\n", rep, rname(r)); ok = false; break; }
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    if (!ms.empty()) {
        double mn = ms[0], mx = ms[0], sum = 0;
        for (double v : ms) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
        printf("    %u dispatches in one command buffer, %u submits: mean %.3f ms, min %.3f, max %.3f, %.2f us per dispatch\n",
               o.dispatches, (unsigned)ms.size(), sum / ms.size(), mn, mx, (sum / ms.size()) * 1000.0 / o.dispatches);
    }
    cp.destroy(c);
    free_pipe(c, p); free_buf(c, sink_buf);
    return ok;
}

// ------------------------------- P5: a flag the host raises while a shader polls it

bool test_p5(Card& c, const Opts& o, VkMemoryPropertyFlags props, const char* flavor, bool atomic_write) {
    struct SpinPC { uint32_t max_spins; uint32_t a, b, d; };   // same size as PC
    Buf flag, res;
    printf("    flag memory flavor: %s\n", flavor);
    if (make_buf(c, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, props, flag) != VK_SUCCESS) {
        printf("    no buffer of that flavor for the flag\n"); return false;
    }
    make_buf(c, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, res);
    Pipe p;
    if (!make_pipe(c, spv_bytes(spin_spv, spin_spv_len), 2, p)) { printf("    pipeline failed\n"); free_buf(c, flag); free_buf(c, res); return false; }
    bind(c, p, {{flag.b, 0, VK_WHOLE_SIZE}, {res.b, 0, VK_WHOLE_SIZE}});
    CmdPool cp; cp.init(c);

    // Mapped once and left mapped: mapping and unmapping on every write is not what an engine
    // doorbell does, and it hides whether the write itself reaches the GPU.
    void* flag_map = nullptr;
    if (vkMapMemory(c.dev, flag.m, 0, VK_WHOLE_SIZE, 0, &flag_map) != VK_SUCCESS) {
        printf("    the flag memory cannot be mapped\n");
        cp.destroy(c); free_pipe(c, p); free_buf(c, flag); free_buf(c, res); return false;
    }
    printf("    host write method: %s\n", atomic_write ? "lock prefixed store plus a flush call" : "plain volatile store");

    auto set_flag = [&](uint32_t v) {
        if (!flag_map) return;
        if (atomic_write) {
            // A lock prefixed store: this is what drains x86 write combining buffers.
            std::atomic_store((std::atomic<uint32_t>*)flag_map, v);
            VkMappedMemoryRange rng{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            rng.memory = flag.m;
            rng.size = 4;
            vkFlushMappedMemoryRanges(c.dev, 1, &rng);
        } else {
            *(volatile uint32_t*)flag_map = v;
        }
    };
    auto spins_written = [&]() -> uint32_t {
        Buf stage;
        make_buf(c, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage);
        uint32_t v = 0;
        void* m = nullptr;
        if (copy_out(c, cp, res.b, stage, &m, 256) == VK_SUCCESS) { v = *(const uint32_t*)m; vkUnmapMemory(c.dev, stage.m); }
        free_buf(c, stage);
        return v;
    };

    auto launch = [&](uint32_t budget) -> VkResult {
        if (!begin_cmd(cp)) return VK_ERROR_INITIALIZATION_FAILED;
        vkCmdBindPipeline(cp.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        vkCmdBindDescriptorSets(cp.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0, nullptr);
        const SpinPC pc{budget, 0, 0, 0};
        vkCmdPushConstants(cp.cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vkCmdDispatch(cp.cb, 1, 1, 1);
        return vkEndCommandBuffer(cp.cb);
    };

    double base = 0, full = 0; uint32_t sp = 0;
    set_flag(1);
    {   // the floor: the flag is already raised, so the loop exits at once
        if (launch(o.spin_max) != VK_SUCCESS) { cp.destroy(c); free_pipe(c, p); free_buf(c, flag); free_buf(c, res); return false; }
        auto t0 = std::chrono::steady_clock::now();
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cp.cb;
        if (vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS || vkQueueWaitIdle(c.queue) != VK_SUCCESS) {
            printf("    floor run failed\n"); cp.destroy(c); free_pipe(c, p); free_buf(c, flag); free_buf(c, res); return false;
        }
        base = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        sp = spins_written();
        printf("    flag preset: %.3f ms, spins %u (this is the submit and readback floor)\n", base, sp);
    }
    set_flag(0);
    {   // the ceiling: nobody raises the flag, the spin runs out
        if (launch(o.spin_max) != VK_SUCCESS) { cp.destroy(c); free_pipe(c, p); free_buf(c, flag); free_buf(c, res); return false; }
        auto t0 = std::chrono::steady_clock::now();
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cp.cb;
        if (vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS || vkQueueWaitIdle(c.queue) != VK_SUCCESS) {
            printf("    ceiling run failed\n"); cp.destroy(c); free_pipe(c, p); free_buf(c, flag); free_buf(c, res); return false;
        }
        full = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        sp = spins_written();
        printf("    no flag ever raised: %.3f ms, spins counted %u of %u\n", full, sp, o.spin_max);
    }
    if (full < base * 2.0 || sp == 0) {
        printf("    P5: the loop did not actually spin (the compiler removed it or the flag was not re read)\n");
        cp.destroy(c); free_pipe(c, p); free_buf(c, flag); free_buf(c, res); return false;
    }
    const double rate = (double)sp / (full - base);   // spins per ms of real polling
    printf("    polling rate: %.1f spins per ms\n", rate);

    // The delay runs need a polling window that outlasts the host delay, otherwise the loop
    // ends before the host has written anything and the measurement says nothing.
    const int poll_ms = 60;
    const uint32_t poll_budget = (uint32_t)(rate * (double)poll_ms);
    printf("    delay runs poll for %u spins, about %d ms of window\n", poll_budget, poll_ms);

    bool ok = true;
    for (int delay : {1, 10, 40}) {
        set_flag(0);
        if (launch(poll_budget) != VK_SUCCESS) { ok = false; break; }
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cp.cb;
        if (vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) { printf("    submit failed\n"); ok = false; break; }
        host_sleep_ms(delay);
        set_flag(1);
        if (vkQueueWaitIdle(c.queue) != VK_SUCCESS) { printf("    wait after the flag failed (device lost?)\n"); ok = false; break; }
        const uint32_t seen = spins_written();
        const double seen_ms = (double)seen / rate;
        printf("    host raised the flag after %d ms: the shader had spun %.1f ms (%u spins) when it saw it  %s\n",
               delay, seen_ms, seen,
               seen >= poll_budget ? "the flag was never seen, the poll ran out" : "seen");
        if (seen >= poll_budget) { printf("    P5: the host write was not visible to the polling shader at all\n"); ok = false; }
        else if (seen_ms > delay * 2.0 + 5.0) { printf("    P5: the flag arrived too late for a doorbell\n"); ok = false; }
    }
    vkUnmapMemory(c.dev, flag.m);
    cp.destroy(c);
    free_pipe(c, p); free_buf(c, flag); free_buf(c, res);
    return ok;
}

// ------------------------------- P6: cooperative matrix shapes and packed dot product

// The cooperative matrix query is resolved by name for the same reason as the address query.
PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR g_get_coop = nullptr;

void test_p6(const std::vector<Card>& cards) {
    for (const Card& c : cards) {
        printf("  %s\n", c.name.c_str());
        uint32_t n = 0;
        if (!g_get_coop) {
            printf("    no cooperative matrix entry point in this loader\n");
        } else if (g_get_coop(c.phys, &n, nullptr) != VK_SUCCESS || n == 0) {
            printf("    no cooperative matrix shapes\n");
        } else {
            std::vector<VkCooperativeMatrixPropertiesKHR> v(n);
            for (auto& x : v) x.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            g_get_coop(c.phys, &n, v.data());
            printf("    cooperative matrix shapes: %u\n", n);
            for (uint32_t i = 0; i < n && i < 24; ++i)
                printf("      M %u N %u K %u  A %s B %s C %s result %s  saturating %s\n", v[i].MSize, v[i].NSize, v[i].KSize,
                       comp_name(v[i].AType), comp_name(v[i].BType), comp_name(v[i].CType), comp_name(v[i].ResultType),
                       v[i].saturatingAccumulation ? "yes" : "no");
        }
        VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &idp;
        vkGetPhysicalDeviceProperties2(c.phys, &p2);
        printf("    packed 4x8 bit dot product: unsigned %s  signed %s  mixed %s\n",
               idp.integerDotProduct4x8BitPackedUnsignedAccelerated ? "yes" : "no",
               idp.integerDotProduct4x8BitPackedSignedAccelerated ? "yes" : "no",
               idp.integerDotProduct4x8BitPackedMixedSignednessAccelerated ? "yes" : "no");
        printf("    8 bit dot product: unsigned %s  signed %s   16 bit: unsigned %s  signed %s\n",
               idp.integerDotProduct8BitUnsignedAccelerated ? "yes" : "no",
               idp.integerDotProduct8BitSignedAccelerated ? "yes" : "no",
               idp.integerDotProduct16BitUnsignedAccelerated ? "yes" : "no",
               idp.integerDotProduct16BitSignedAccelerated ? "yes" : "no");
    }
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&](double d) { return (i + 1 < argc) ? std::atof(argv[++i]) : d; };
        if (a == "--limit-gib") o.limit_gib = nxt(7.0);
        else if (a == "--alt-gib") o.alt_gib = nxt(3.5);
        else if (a == "--stride-mib") o.stride_mib = nxt(64);
        else if (a == "--view-off-mib") o.view_off_mib = nxt(5120);
        else if (a == "--host-mib") o.host_mib = nxt(1024);
        else if (a == "--spin-max") o.spin_max = (uint32_t)nxt(2000000);
        else if (a == "--dispatches") o.dispatches = (uint32_t)nxt(2400);
        else if (a == "--replays") o.replays = (uint32_t)nxt(100);
        else if (a == "--include-igpu") o.discrete_only = false;
        else if (a == "--spans") o.spans = true;
        else if (a == "--card" && i + 1 < argc) o.card_filter = argv[++i];
        else if (a == "--only") {
            if (i + 1 < argc) {
                std::string s = argv[++i]; size_t p = 0;
                while (p < s.size()) {
                    size_t q = s.find(',', p);
                    o.only.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
                    if (q == std::string::npos) break;
                    p = q + 1;
                }
            }
        } else if (a == "--help") {
            printf("vk_caps_probe [--only p1,p2,p2b,p3,p4,p5,p6] [--limit-gib 7] [--alt-gib 3.5] [--stride-mib 64]\n"
                   "              [--view-off-mib 5120] [--host-mib 1024] [--spin-max 2000000] [--dispatches 2400]\n"
                   "              [--replays 100] [--card A770] [--include-igpu]\n");
            return 0;
        }
    }
    printf("strata vulkan capability probe\n");
    printf("args: big %g GiB (fallback %g GiB), tag stride %g MiB, view offset %g MiB, host %g MiB, spin max %u, "
           "dispatches %u, replays %u\n", o.limit_gib, o.alt_gib, o.stride_mib, o.view_off_mib, o.host_mib,
           o.spin_max, o.dispatches, o.replays);

    std::vector<const char*> iext = {"VK_KHR_get_physical_device_properties2", "VK_KHR_portability_enumeration"};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vk_caps_probe";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)iext.size();
    ici.ppEnabledExtensionNames = iext.data();
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ici, nullptr, &inst);
    if (r != VK_SUCCESS) { printf("vkCreateInstance: %s\n", rname(r)); return 2; }

    g_get_coop = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> phys(n ? n : 1);
    if (n) vkEnumeratePhysicalDevices(inst, &n, phys.data());
    printf("physical devices: %u\n", n);

    std::vector<Card> cards;
    for (auto pd : phys) {
        Card c; c.phys = pd;
        vkGetPhysicalDeviceProperties(pd, &c.props);
        c.name = c.props.deviceName;
        if (o.discrete_only && c.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { printf("  skipping %s (integrated)\n", c.name.c_str()); continue; }
        if (!o.card_filter.empty() && c.name.find(o.card_filter) == std::string::npos) { printf("  skipping %s (filtered by --card)\n", c.name.c_str()); continue; }
        uint32_t en = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
        std::vector<VkExtensionProperties> av(en ? en : 1);
        if (en) vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, av.data());
        static const char* kWant[] = {"VK_EXT_external_memory_host", "VK_EXT_memory_budget", "VK_KHR_driver_properties",
                                      "VK_KHR_buffer_device_address", "VK_EXT_subgroup_size_control"};
        std::vector<const char*> use;
        for (const char* w : kWant) for (const auto& x : av) if (std::strcmp(x.extensionName, w) == 0) { use.push_back(w); break; }
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn ? qn : 1);
        if (qn) vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs.data());
        bool have_q = false;
        for (uint32_t i = 0; i < qn; ++i) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qfam = i; have_q = true; break; }
        if (!have_q) { printf("  skipping %s (no compute queue)\n", c.name.c_str()); continue; }
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = c.qfam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        // The probe's SPIR-V asks for Int64, scalar block layout and buffer device address, so
        // all three features have to be enabled on the device or pipeline creation fails.
        VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        feats.features.shaderInt64 = VK_TRUE;
        VkPhysicalDeviceScalarBlockLayoutFeatures sbl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES};
        sbl.scalarBlockLayout = VK_TRUE;
        VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        bda.bufferDeviceAddress = VK_TRUE;
        feats.pNext = &sbl;
        sbl.pNext = &bda;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.pNext = &feats;
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = (uint32_t)use.size();
        di.ppEnabledExtensionNames = use.data();
        r = vkCreateDevice(pd, &di, nullptr, &c.dev);
        if (r != VK_SUCCESS) { printf("  vkCreateDevice %s: %s\n", c.name.c_str(), rname(r)); continue; }
        vkGetDeviceQueue(c.dev, c.qfam, 0, &c.queue);
        c.get_address = (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(c.dev, "vkGetBufferDeviceAddress");
        if (!c.get_address)
            c.get_address = (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(c.dev, "vkGetBufferDeviceAddressKHR");
        c.get_host_memory = (PFN_vkGetMemoryHostPointerPropertiesEXT)
            vkGetInstanceProcAddr(inst, "vkGetMemoryHostPointerPropertiesEXT");
        for (const char* u : use) c.exts.push_back(u);
        cards.push_back(c);
    }
    if (cards.empty()) { printf("no usable device\n"); vkDestroyInstance(inst, nullptr); return 2; }
    printf("cards in this run: %zu\n\n", cards.size());

    struct V { std::string test, who, result; bool pass; };
    std::vector<V> verdicts;

    if (o.want("p1")) {
        printf("P1 devices, limits, memory, the mapping table\n");
        test_p1(cards);
        verdicts.push_back({"p1", "all", "printed", true});
    }
    for (Card& c : cards) {
        if (c.lost) { printf("  %s: the device was lost earlier in this run, the rest of its tests are skipped\n", c.name.c_str()); continue; }
        if (o.want("p2")) {
            printf("\nP2 big device allocation, pointer access past 4 GiB: %s\n", c.name.c_str());
            const bool ok = test_p2(c, o);
            verdicts.push_back({"p2", c.name, ok ? "pass" : "FAIL", ok});
        }
        if (o.want("p2b")) {
            printf("\nP2b the same buffer bound the ordinary way past 4 GiB: %s\n", c.name.c_str());
            const bool ok = test_p2b(c, o);
            verdicts.push_back({"p2b", c.name, ok ? "pass" : "FAIL", ok});
        }
        if (o.want("p4")) {
            printf("\nP4 dispatch cost of a window sized command buffer: %s\n", c.name.c_str());
            const bool ok = test_p4(c, o);
            verdicts.push_back({"p4", c.name, ok ? "measured" : "FAIL", ok});
        }
        if (o.want("p5")) {
            // Three flavors of flag memory: uncached system pages, cached system pages, and
            // device local memory the host can also write. A doorbell needs the fastest one.
            const struct { VkMemoryPropertyFlags f; const char* n; bool atomic; } flavors[] = {
                {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "host coherent", false},
                {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "host coherent", true},
                {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                 "host coherent cached", true},
                {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 "device local host coherent", true},
            };
            for (const auto& fl : flavors) {
                printf("\nP5 host raised flag, shader polling: %s (%s flag)\n", c.name.c_str(), fl.n);
                const bool ok = test_p5(c, o, fl.f, fl.n, fl.atomic);
                verdicts.push_back({"p5", c.name + " " + fl.n, ok ? "measured" : "FAIL", ok});
            }
        }
    }
    if (o.want("p3")) {
        printf("\nP3 one host allocation, both cards, in place\n");
        const bool ok = test_p3(cards, o);
        verdicts.push_back({"p3", "both cards", ok ? "pass" : "FAIL or partial", ok});
    }
    if (o.want("p6")) { printf("\nP6 cooperative matrix and packed dot product\n"); test_p6(cards); verdicts.push_back({"p6", "all", "printed", true}); }

    printf("\n== verdicts ==\n");
    int fails = 0;
    for (const auto& v : verdicts) {
        printf("  %-4s %-34s %s\n", v.test.c_str(), v.who.c_str(), v.result.c_str());
        if (!v.pass) ++fails;
    }
    printf("failed: %d\n", fails);
    for (Card& c : cards) vkDestroyDevice(c.dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return fails ? 1 : 0;
}
