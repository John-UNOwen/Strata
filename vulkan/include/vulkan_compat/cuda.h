// vulkan/include/vulkan_compat/cuda.h - CUDA's *driver* API names the host tree uses, on Vulkan.
//
// Justified by iteration 0, which could not open this file at all:
//   src\core\vmm.cpp(4,10): error C1083: Cannot open include file: 'cuda.h': No such file or directory
//   src\core\expert_cache.cpp(12,10): error C1083: Cannot open include file: 'cuda.h': No such file or directory
// and by what those two files then ask for: the virtual-memory-management subset (CU_MEM_*, cuMem*, cuDevice*), which
// they reach through cudaGetDriverEntryPoint by name (src/core/vmm.cpp:49-52, src/core/expert_cache.cpp:252-260),
// because they must work without a link against libcuda.
//
// The signatures below are CUDA's, spelled exactly, because vmm.cpp:16-22 takes them with `decltype(&cuMemMap)` and
// expert_cache.cpp:220-230 with `CUresult (CUDAAPI* map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
// unsigned long long)`: a different spelling would not be the same type.  The implementations are this port's own
// (vulkan/src/gpu/runtime.cpp), reached through strata::gpu::driver_entry_point (vulkan_compat/cuda_runtime.h's
// cudaGetDriverEntryPoint).  What they do on Vulkan: the arena is one address range per region, and each segment is
// its own VkDeviceMemory block bound into one buffer with vkBindBufferMemory2 and released when the segment is
// unmapped - the sub-allocator Phase 1 names, and the thing Phase 0 P2 measured (a 64 MiB allocation whose returned
// buffer address crosses 4 GiB reads back correct, so the Level Zero fault this plan is chasing is not a driver fault).
#pragma once

#include <cstddef>
#include <cstdint>

// CUDA's calling-convention marker for its driver entry points.  It is nothing on this compiler: the port's
// implementations are ordinary C++ functions with external linkage.
#define CUDAAPI

typedef int CUresult;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_PERMITTED 201
#define CUDA_ERROR_NOT_SUPPORTED 801

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;      // CUDA's device address type; this port's device address is the buffer's

typedef enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 122,   // what vmm.cpp:55 asks
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
} CUdevice_attribute;

typedef enum CUmemAllocationType { CU_MEM_ALLOCATION_TYPE_INVALID = 0, CU_MEM_ALLOCATION_TYPE_PINNED = 1 }
    CUmemAllocationType;
typedef enum CUmemAllocationGranularity_flags { CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
                                                CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1 }
    CUmemAllocationGranularity_flags;
typedef enum CUmemLocationType { CU_MEM_LOCATION_TYPE_INVALID = 0, CU_MEM_LOCATION_TYPE_DEVICE = 1 }
    CUmemLocationType;
typedef enum CUmemAccessFlags { CU_MEM_ACCESS_FLAGS_PROT_NONE = 0, CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3 }
    CUmemAccessFlags;

typedef struct CUmemLocation_st { CUmemLocationType type; int32_t id; } CUmemLocation;   // id = the device ordinal
typedef unsigned long long CUmemGenericAllocationHandle;                                  // a chunk this port can find
                                                                                        // its VkDeviceMemory through
typedef struct CUmemAllocationProp_st {
    CUmemAllocationType type;
    CUmemLocation location;
    int32_t rotation;
    void* customPool;
    void* win32HandleMetaData;
    struct { unsigned char compressionType; unsigned char gpuAccessGuide; } allocFlags;   // CUDA 12's tail; unused here
} CUmemAllocationProp;
typedef struct CUmemAccessDesc_st { CUmemLocation location; CUmemAccessFlags flags; } CUmemAccessDesc;

// The set the two files above name, with CUDA 12's signatures.
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetAttribute(int* value, CUdevice_attribute attribute, CUdevice device);
CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop,
                                       CUmemAllocationGranularity_flags option);
CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
                             unsigned long long flags);
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop,
                     unsigned long long flags);
CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
                  unsigned long long flags);
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);
