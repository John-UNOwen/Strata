# Phase 0 of VULKAN-BACKEND-PLAN.md: the capability probe and nothing else.
#
# This file is deliberately NOT add_subdirectory'd from the top level CMakeLists.txt.
# The CUDA, HIP and SYCL builds stay byte identical while the Vulkan bet is being tested.
# start-vulkan-probe.bat configures this directory on its own, into build-vulkan-probe/.
#
# Needs the Khronos Vulkan SDK (find_package(Vulkan)) and glslc on PATH.
# See the plan's "Phase 0 preconditions" for the routes and their sizes.

cmake_minimum_required(VERSION 3.22)
project(strata_vulkan_probe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
  message(FATAL_ERROR
    "No Vulkan SDK found (find_package(Vulkan) failed). Install KhronosGroup.VulkanSDK, or set "
    "VULKAN_SDK to an unpacked SDK. glslc must be on PATH.")
endif()

find_program(GLSLC NAMES glslc)
if(NOT GLSLC)
  message(FATAL_ERROR "glslc not found on PATH. It ships with the Vulkan SDK.")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# probe/shaders/<name>.comp -> build/spv/<name>.spv -> generated/shaders/<name>.h
set(PROBE_SHADERS tag_write tag_read span_write span_read checksum write_half noop spin)
set(SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spv)
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${SPV_DIR} ${GEN_DIR}/shaders)

set(GENERATED_HEADERS "")
foreach(name ${PROBE_SHADERS})
  set(spv ${SPV_DIR}/${name}.spv)
  set(hdr ${GEN_DIR}/shaders/${name}.h)
  add_custom_command(
    OUTPUT ${spv}
    COMMAND ${GLSLC} -O --target-env=vulkan1.3 -fshader-stage=comp
            ${CMAKE_CURRENT_SOURCE_DIR}/probe/shaders/${name}.comp -o ${spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/probe/shaders/${name}.comp
    VERBATIM)
  add_custom_command(
    OUTPUT ${hdr}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${spv} ${hdr} ${name}
    DEPENDS ${spv} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py
    VERBATIM)
  list(APPEND GENERATED_HEADERS ${hdr})
endforeach()

add_custom_target(vulkan_probe_shaders DEPENDS ${GENERATED_HEADERS})

add_executable(vk_caps_probe probe/vk_caps_probe.cpp)
add_dependencies(vk_caps_probe vulkan_probe_shaders)
target_include_directories(vk_caps_probe PRIVATE ${GEN_DIR})
target_link_libraries(vk_caps_probe PRIVATE Vulkan::Vulkan)

if(MSVC)
  target_compile_options(vk_caps_probe PRIVATE /W3)
else()
  target_compile_options(vk_caps_probe PRIVATE -Wall)
endif()
