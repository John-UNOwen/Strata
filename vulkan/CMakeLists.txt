# Phase 0 (the capability probe) and the Phase 1 build skeleton (the `strata` engine target over ../src)
# of VULKAN-BACKEND-PLAN.md.
#
# This file is deliberately NOT add_subdirectory'd from the top level CMakeLists.txt.
# The CUDA, HIP and SYCL builds stay byte identical while the Vulkan bet is being tested.
# start-vulkan-probe.bat configures this directory on its own, into build-vulkan-probe/. The probe target and its
# shader chain are unchanged by Phase 1 and they are standalone: with STRATA_VULKAN_ENGINE=OFF this file stops after
# the probe (the return() in the ggml block) and asks nothing of llama.cpp - no ggml subproject, no C language, no
# engine target - which is what the Phase 0 version of this file (git show HEAD:vulkan/CMakeLists.txt) did.
# build-vulkan-probe's cache already holds STRATA_VULKAN_ENGINE=OFF, so its default build stays the probe; when a
# pinned llama.cpp checkout happens to be available, OFF means "engine configured but EXCLUDE_FROM_ALL" (bottom of
# this file) and `--target strata` still works.  A brand-new probe build dir has to ask for that state explicitly
# (-DSTRATA_VULKAN_ENGINE=OFF), because the Phase 1 gate is the plain configure of this directory with no flags and
# needs strata in the default build.
#   cmake -S vulkan -B build-vulkan            # probe + engine, engine in the default build
#   cmake -S vulkan -B <dir> -DSTRATA_VULKAN_ENGINE=OFF   # the Phase 0 probe, on its own
#
# Needs the Khronos Vulkan SDK (find_package(Vulkan)) and glslc on PATH.
# See the plan's "Phase 0 preconditions" for the routes and their sizes.

cmake_minimum_required(VERSION 3.22)
project(strata_vulkan_probe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Not QUIET on purpose: the `Found Vulkan: <path> (found ... version ...)` line find_package prints is the evidence
# the Phase 1 gates quote for the SDK this build was configured against, and QUIET suppresses it (find_package_handle_
# standard_args honours <PackageName>_FIND_QUIETLY). The SDK is named by VULKAN_SDK, as start-vulkan-probe.bat does.
find_package(Vulkan)
if(NOT Vulkan_FOUND)
  message(FATAL_ERROR
    "No Vulkan SDK found (find_package(Vulkan) failed). Install KhronosGroup.VulkanSDK, or set "
    "VULKAN_SDK to an unpacked SDK. glslc must be on PATH.")
endif()

find_program(GLSLC NAMES glslc)
if(NOT GLSLC)
  # find_program(GLSLC) searches PATH, and PATH is the one thing a fresh shell of this PC does not have right: the
  # machine sets VULKAN_SDK in the interactive profile but never puts C:\VulkanSDK\<version>\Bin on the PATH of a new
  # process.  The evidence is the configure of `cmake -S vulkan -B build-vulkan` with VULKAN_SDK set and nothing else
  # (no glslc on PATH, `Get-Command glslc` empty) - line 22 reports
  #   -- Found Vulkan: C:/VulkanSDK/1.4.363.0/Lib/vulkan-1.lib (found version "1.4.363") found components: glslc glslangValidator
  # and the next line dies with "glslc not found on PATH".  So the SDK find_package just located already knows where
  # its own glslc is (FindVulkan sets Vulkan_GLSLC_EXECUTABLE for the glslc component it reported finding); use that
  # and the SDK's Bin, instead of assuming something else exported PATH first.
  set(GLSLC_FROM_PATH "")
  set(GLSLC_CANDIDATES "")
  if(Vulkan_GLSLC_EXECUTABLE)
    list(APPEND GLSLC_CANDIDATES ${Vulkan_GLSLC_EXECUTABLE})
  endif()
  foreach(GLSLC_SDK_ROOT IN ITEMS ${Vulkan_INCLUDE_DIR} $ENV{VULKAN_SDK})
    if(GLSLC_SDK_ROOT)
      get_filename_component(GLSLC_SDK_ROOT "${GLSLC_SDK_ROOT}/.." ABSOLUTE)
      foreach(GLSLC_NAME IN ITEMS glslc glslc.exe)
        foreach(GLSLC_SUBDIR IN ITEMS Bin bin)
          list(APPEND GLSLC_CANDIDATES "${GLSLC_SDK_ROOT}/${GLSLC_SUBDIR}/${GLSLC_NAME}")
        endforeach()
      endforeach()
    endif()
  endforeach()
  foreach(GLSLC_CANDIDATE IN LISTS GLSLC_CANDIDATES)
    if(EXISTS "${GLSLC_CANDIDATE}")
      set(GLSLC "${GLSLC_CANDIDATE}")
      set(GLSLC_FROM_PATH "from the Vulkan SDK above")
      break()
    endif()
  endforeach()
  if(GLSLC_FROM_PATH)
    message(STATUS "strata vulkan: glslc is not on PATH; building the probe shaders with ${GLSLC} ${GLSLC_FROM_PATH}")
  endif()
endif()
if(NOT GLSLC)
  message(FATAL_ERROR "glslc not found: not on PATH, and not in the Vulkan SDK (Vulkan_GLSLC_EXECUTABLE is empty and "
                      "VULKAN_SDK='$ENV{VULKAN_SDK}' has no Bin/glslc). It ships with the Vulkan SDK.")
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

# vulkan/shaders/<name>.comp -> build/spv/engine_<name>.spv -> generated/vulkan_shaders/<name>.h is the ENGINE's
# shader chain, and it is configured in the engine half below (after the STRATA_VULKAN_ENGINE / pinned-checkout gate),
# not here: the Phase 0 probe's default build stays the probe's own shaders, so the probe route needs nothing from
# Phase 1 - no llama.cpp checkout, and not the engine's .comp files either.  What that chain is, and where a Phase 2
# kernel plugs into it, is documented where it is configured.

add_executable(vk_caps_probe probe/vk_caps_probe.cpp)
add_dependencies(vk_caps_probe vulkan_probe_shaders)
target_include_directories(vk_caps_probe PRIVATE ${GEN_DIR})
target_link_libraries(vk_caps_probe PRIVATE Vulkan::Vulkan)

if(MSVC)
  target_compile_options(vk_caps_probe PRIVATE /W3)
else()
  target_compile_options(vk_caps_probe PRIVATE -Wall)
endif()

# ---- Phase 1: the engine target -------------------------------------------------------------------------------
#
# `strata` is the shared host engine: ../src, UNCHANGED, built the way metal/ builds it
# (metal/CMakeLists.txt: the compat dir goes first on the include path so the CUDA host code gets its CUDA
# runtime names on a foreign device - include/strata/hip_compat does the same for HIP). ../src is the upstream
# CUDA host tree: 10 host dirs, artifact core kernels ngram plan platform prefill program qwen35 spec.
#
# STRATA_VULKAN_ENGINE is what puts the engine in the DEFAULT build or keeps it out of it.  It is ON by default,
# because the Phase 1 gate is the plain configure of this directory - `cmake -S vulkan -B build-vulkan`, no -D flags -
# and the generated project has to contain strata next to vk_caps_probe.  It was OFF with a return() before, and that
# is why the plain configure produced only vk_caps_probe.vcxproj / vulkan_probe_shaders.vcxproj / ALL_BUILD.vcxproj
# (measured: `-- strata vulkan: STRATA_VULKAN_ENGINE=OFF: probe only`, and no strata.vcxproj in the build dir).
# It is still a knob, and OFF has two honest outcomes.  With a llama.cpp checkout at the pinned commit available the
# engine targets are configured and marked EXCLUDE_FROM_ALL at the bottom of this file, so start-vulkan-probe.bat -
# which configures THIS directory into build-vulkan-probe with STRATA_VULKAN_ENGINE=OFF in its cache and builds every
# default target - keeps building the probe alone, and `--target strata` still works in that dir.  With no such
# checkout, OFF is the Phase 0 route in full: this file returns after the probe and the engine is not configured at
# all, because the probe needs nothing from llama.cpp and must not be made to depend on one.  Both routes:
#   cmake -S vulkan -B build-vulkan            # probe + engine, engine in the default build
#   cmake --build build-vulkan --config Release
#   cmake -S vulkan -B build-vulkan-probe -DCMAKE_BUILD_TYPE=Release   # probe only (cached OFF)
#   cmake -S vulkan -B <new> -DCMAKE_BUILD_TYPE=Release -DSTRATA_VULKAN_ENGINE=OFF   # the same with no llama.cpp at all
#
# The layout mirrors metal/CMakeLists.txt for one reason that matters to the error loop: ../src holds seven files
# with their own main() (program/generate.cpp, core/{device,load,overlap,concurrent}_main.cpp, plan/plan_main.cpp,
# qwen35/q36_main.cpp) plus the *_test.cpp harnesses.  Compiled as one executable they collide (LNK2005 / LNK1169),
# and that noise hides the link errors this phase wants to read.  So: every host TU is compiled into a static
# library, and the executable is the engine's own main (src/program/generate.cpp), the same one metal links.
#
# No output directory is set, on purpose: that is what keeps this multi-config safe. The Visual Studio
# generators place every target in a <Config>\ subdirectory of the build dir, so the exe lands at
# build-vulkan\Release\strata.exe; a single-config generator puts it in the build dir itself. Hardcoding
# either path is the mistake that breaks the other generator (start-vulkan-probe.bat checks both).
option(STRATA_VULKAN_ENGINE "Build the strata engine over ../src (Phase 1); ON = the engine is in the default build, OFF = EXCLUDE_FROM_ALL when a pinned llama.cpp checkout is available, and no engine at all (the Phase 0 probe alone) when it is not" ON)

get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(S ${STRATA_ROOT}/src)

# ggml: the native expert formats' type traits and ggml-cpu, from a llama.cpp checkout at the commit this checkout
# pins (third_party/ggml/VERSION.txt), exactly as metal/CMakeLists.txt and the top-level project use it.  Nothing is
# downloaded here: the flag names the checkout, and this PC already has the pinned commit at
# build-sycl-jit/_deps/strata_llamacpp-src from the SYCL build.
#
# Two rules this block had to acquire, both from the regression its first version introduced:
#
#   1. It belongs to the ENGINE half, not to the file.  The Phase 0 probe defined above (vk_caps_probe and its shader
#      chain) needs nothing from llama.cpp - the Phase 0 version of this file (git show HEAD:vulkan/CMakeLists.txt,
#      65 lines) contains no ggml, no STRATA_GGML_DIR and no add_subdirectory at all - yet the discovery loop, the
#      FATAL_ERROR and the add_subdirectory further down ran whatever STRATA_VULKAN_ENGINE said, so the probe route
#      (start-vulkan-probe.bat, whose build-vulkan-probe cache holds STRATA_VULKAN_ENGINE=OFF) could not configure
#      without a checkout.  Measured before this block was gated:
#        cmake -S vulkan -B <temp> -DCMAKE_BUILD_TYPE=Release -DSTRATA_VULKAN_ENGINE=OFF -DSTRATA_GGML_DIR=C:/definitely/not/here
#          CMake Error at CMakeLists.txt:201 (message): STRATA_GGML_DIR=... is not a llama.cpp checkout   [exit 1]
#      The requirement is now conditional on STRATA_VULKAN_ENGINE, and an engine-off configure with no usable checkout
#      stops after the probe (the return() at the end of this block) instead of failing.
#
#   2. The pinned commit is CHECKED, not asserted.  The first version took the first candidate that merely had a
#      ggml/CMakeLists.txt and then printed "(pinned commit: third_party/ggml/VERSION.txt)", an assurance it had not
#      earned: ../llama.cpp at any arbitrary commit would have been taken and the engine would have been built
#      against it without a word.  Every checkout used from here is compared, hash against hash, with
#      third_party/ggml/VERSION.txt as the reference - the same value the top-level project pins FetchContent's
#      GIT_TAG to (CMakeLists.txt:1309 GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d).  Which candidate happens to
#      be first in a list no longer decides what the engine is built against.
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout at the pinned commit (third_party/ggml/VERSION.txt)")
option(STRATA_GGML_ALLOW_UNPINNED "Build ggml from an explicitly named llama.cpp checkout even when its commit is not the pinned one. OFF by default; with ON the configure says so in words instead of claiming the pin." OFF)

# The reference commit, read from the file that is the pin - not copied here as a second copy that can rot.
# The file's one line is "<commit> <date>" (measured: 73 bytes, "3cf03257f219afbe7334045ff7c6a06ac68c627d Sun Sep 20
# 16:08:11 2026 +0800"), so the commit is what precedes the first space and the whole line is kept for the log.
set(STRATA_GGML_VERSION_FILE "${STRATA_ROOT}/third_party/ggml/VERSION.txt")
set(STRATA_GGML_VERSION_LINE "")
set(STRATA_GGML_PINNED "")
if(EXISTS "${STRATA_GGML_VERSION_FILE}")
  file(READ "${STRATA_GGML_VERSION_FILE}" STRATA_GGML_VERSION_LINE)
  string(REGEX REPLACE "\r?\n.*" "" STRATA_GGML_VERSION_LINE "${STRATA_GGML_VERSION_LINE}")   # first line only
  string(STRIP "${STRATA_GGML_VERSION_LINE}" STRATA_GGML_VERSION_LINE)
  string(REGEX REPLACE "[ \t].*" "" STRATA_GGML_PINNED "${STRATA_GGML_VERSION_LINE}")         # drop the date
  string(TOLOWER "${STRATA_GGML_PINNED}" STRATA_GGML_PINNED)
endif()

# A git commit id is 40 hex characters.  The length is checked as a length, not as a regex: CMake's regex engine has
# no {n} interval quantifier, and that is not obvious - measured with cmake -P, a known-good 40-character id answers
# FALSE to `MATCHES "^[0-9a-f]{40}$"` (and therefore TRUE to `NOT ... MATCHES`), which is the shape of bug this would
# otherwise have been: every checkout, the pinned one included, rejected as having an unreadable commit.
function(strata_vulkan_is_commit VALUE OUT)
  set(_sv_is FALSE)
  if(VALUE MATCHES "^[0-9a-fA-F]+$")
    string(LENGTH "${VALUE}" _sv_len)
    if(_sv_len EQUAL 40)
      set(_sv_is TRUE)
    endif()
  endif()
  set(${OUT} "${_sv_is}" PARENT_SCOPE)
endfunction()

# What commit a checkout is at, read the way git itself reads it, and without running git.  Running git for this is
# not available on this PC, which is why the first version did not check at all: the FetchContent checkout is its own
# repository root, owned by BUILTIN/Administrators while a shell token here is UAC-filtered, so
#   git -C build-sycl-jit/_deps/strata_llamacpp-src rev-parse HEAD
#     fatal: detected dubious ownership in repository at '.../build-sycl-jit/_deps/strata_llamacpp-src'   [exit 128]
# (the machine's global safe.directory lists C:/new_tdx and the Strata repo, not that nested repo, and adding an
# exception outside vulkan/ is not this file's to make).  .git/HEAD is the file git reads to answer the same question,
# it needs no ownership exception, and for the checkout that matters here it is the answer directly: measured, 41
# bytes holding 3cf03257f219afbe7334045ff7c6a06ac68c627d, because a FetchContent checkout is detached and git writes
# the commit itself into HEAD.  A normal clone instead writes "ref: refs/heads/<branch>", resolved from
# .git/refs/<ref> and then from .git/packed-refs, which is also what git does.
function(strata_vulkan_checkout_commit CHECKOUT OUT_COMMIT)
  set(_sv_gitdir "")
  if(IS_DIRECTORY "${CHECKOUT}/.git")
    set(_sv_gitdir "${CHECKOUT}/.git")
  elseif(EXISTS "${CHECKOUT}/.git")
    # linked worktree / submodule: .git is a one-line file, "gitdir: <path>"
    file(READ "${CHECKOUT}/.git" _sv_link)
    string(REGEX REPLACE "^gitdir:[ \t]*" "" _sv_link "${_sv_link}")
    string(STRIP "${_sv_link}" _sv_link)
    get_filename_component(_sv_gitdir "${_sv_link}" ABSOLUTE BASE_DIR "${CHECKOUT}")
  endif()
  if(_sv_gitdir AND EXISTS "${_sv_gitdir}/HEAD")
    file(READ "${_sv_gitdir}/HEAD" _sv_head)
    string(STRIP "${_sv_head}" _sv_head)
    strata_vulkan_is_commit("${_sv_head}" _sv_head_is_commit)
    if(_sv_head_is_commit)
      set(${OUT_COMMIT} "${_sv_head}" PARENT_SCOPE)          # detached checkout: HEAD *is* the commit
      return()
    endif()
    if(_sv_head MATCHES "^ref:[ \t]*(.+)$")
      set(_sv_ref "${CMAKE_MATCH_1}")                        # a normal clone: HEAD names a ref, "refs/heads/<branch>"
      # The loose ref file is <git dir>/<ref> - the "refs/" in the path is the one HEAD already named, and writing a
      # second one here is the mistake that reads a real clone as having no readable HEAD (measured on a fixture with
      # .git/HEAD = "ref: refs/heads/master" and .git/refs/heads/master present: the wrong path came back empty).
      if(EXISTS "${_sv_gitdir}/${_sv_ref}")
        file(READ "${_sv_gitdir}/${_sv_ref}" _sv_from_ref)
        string(STRIP "${_sv_from_ref}" _sv_from_ref)
        strata_vulkan_is_commit("${_sv_from_ref}" _sv_ref_is_commit)
        if(_sv_ref_is_commit)
          string(TOLOWER "${_sv_from_ref}" _sv_from_ref)
          set(${OUT_COMMIT} "${_sv_from_ref}" PARENT_SCOPE)
          return()
        endif()
      endif()
      if(EXISTS "${_sv_gitdir}/packed-refs")
        # packed-refs lines are "<commit> <ref>".  ^ is the start of the whole string in CMake's regex, so the
        # alternation with \n is what makes this match a line rather than only the first one.
        file(READ "${_sv_gitdir}/packed-refs" _sv_packed)
        string(REGEX MATCH "(^|\n)([0-9a-fA-F]+)[ \t]+${_sv_ref}[ \t]*(\r|\n|$)" _sv_hit "${_sv_packed}")
        if(_sv_hit)
          string(TOLOWER "${CMAKE_MATCH_2}" _sv_from_ref)
          strata_vulkan_is_commit("${_sv_from_ref}" _sv_packed_is_commit)
          if(_sv_packed_is_commit)
            set(${OUT_COMMIT} "${_sv_from_ref}" PARENT_SCOPE)
            return()
          endif()
        endif()
      endif()
    endif()
  endif()
  set(${OUT_COMMIT} "" PARENT_SCOPE)                         # no readable .git: the commit is unknown, not assumed
endfunction()

# Is this a checkout this file may build ggml from: a ggml tree, at the pinned commit.  Sets <OUT>_OK, <OUT>_COMMIT
# and <OUT>_WHY in the caller's scope; WHY is the sentence the configure log shows when it is not, so a rejection is
# never silent and a match is never assumed.
function(strata_vulkan_check_ggml CHECKOUT PINNED OUT)
  set(_sv_ok FALSE)
  set(_sv_commit "")
  set(_sv_why "")
  strata_vulkan_is_commit("${PINNED}" _sv_have_pin)
  if(NOT CHECKOUT)
    set(_sv_why "no checkout was given")
  elseif(NOT EXISTS "${CHECKOUT}/ggml/CMakeLists.txt")
    set(_sv_why "has no ggml/CMakeLists.txt")
  elseif(NOT _sv_have_pin)
    set(_sv_why "cannot be checked: ${STRATA_GGML_VERSION_FILE} does not name a commit to check against")
  else()
    strata_vulkan_checkout_commit("${CHECKOUT}" _sv_commit)
    if(NOT _sv_commit)
      set(_sv_why "has no readable .git/HEAD, so its commit cannot be checked against the pinned one")
    elseif(_sv_commit STREQUAL PINNED)
      set(_sv_ok TRUE)
    else()
      set(_sv_why "is at ${_sv_commit}, not the pinned ${PINNED}")
    endif()
  endif()
  set(${OUT}_OK "${_sv_ok}" PARENT_SCOPE)
  set(${OUT}_COMMIT "${_sv_commit}" PARENT_SCOPE)
  set(${OUT}_WHY "${_sv_why}" PARENT_SCOPE)
endfunction()

set(STRATA_GGML_RESOLVED FALSE)   # a checkout was found that this file is willing to build ggml from
set(STRATA_GGML_MATCHED FALSE)    # ... and it was verified to BE the pinned commit
set(STRATA_GGML_COMMIT "")
set(STRATA_GGML_REJECTED "")

# An explicit -DSTRATA_GGML_DIR is a request, so it is checked exactly like a discovered one, and it is never
# silently replaced by a discovered one when it fails: the gate below reports it and says what to change.
if(STRATA_GGML_DIR)
  get_filename_component(STRATA_GGML_DIR "${STRATA_GGML_DIR}" ABSOLUTE)
  strata_vulkan_check_ggml("${STRATA_GGML_DIR}" "${STRATA_GGML_PINNED}" STRATA_GGML_NAMED)
  if(STRATA_GGML_NAMED_OK)
    set(STRATA_GGML_RESOLVED TRUE)
    set(STRATA_GGML_MATCHED TRUE)
    set(STRATA_GGML_COMMIT "${STRATA_GGML_NAMED_COMMIT}")
  elseif(EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt" AND STRATA_GGML_ALLOW_UNPINNED)
    # The only way an unpinned checkout is used at all: named, by an option whose name says what it costs, and the
    # configure then says in words that the pin is not matched - instead of printing the pin it never checked.
    set(STRATA_GGML_RESOLVED TRUE)
    set(STRATA_GGML_COMMIT "${STRATA_GGML_NAMED_COMMIT}")
    if(STRATA_GGML_COMMIT)
      set(STRATA_GGML_UNPINNED_NOTE "commit ${STRATA_GGML_COMMIT}, pinned is ${STRATA_GGML_PINNED}")
    else()
      set(STRATA_GGML_UNPINNED_NOTE "commit unknown, pinned is ${STRATA_GGML_PINNED}")
    endif()
  else()
    list(APPEND STRATA_GGML_REJECTED "STRATA_GGML_DIR='${STRATA_GGML_DIR}' ${STRATA_GGML_NAMED_WHY}")
  endif()
endif()

if(NOT STRATA_GGML_DIR)
  # Nothing is downloaded here, and no -D is needed either: the flag stays as the override, and with no flag the
  # checkout is looked for where the pinned commit already sits on this PC.  Required, because the plain configure the
  # gate runs has no -DSTRATA_GGML_DIR and stopped here before this block existed:
  #   CMake Error at CMakeLists.txt:113 (message):
  #     STRATA_GGML_DIR='' is not a llama.cpp checkout: point it at one at the pinned commit
  # build-sycl-jit/_deps/strata_llamacpp-src is the FetchContent checkout the SYCL build already fetched at that
  # commit; the siblings are the same FetchContent name under the other SYCL build dirs, and ../llama.cpp is the
  # plain local clone's usual place.  What changed is what a candidate has to satisfy: a ggml/CMakeLists.txt is not
  # enough, its commit has to match, so what gets picked is the checkout that IS the pinned commit rather than the
  # first directory in this list that happens to hold a ggml tree (measured before: the list's order was the only
  # thing keeping ../llama.cpp at an arbitrary commit out of the engine build).  A candidate that is a checkout but
  # not the pinned one is named in the log and skipped.  The winner is written back to the cache so the build dir
  # shows which checkout the engine was built from.
  foreach(GGML_CANDIDATE IN ITEMS
              "$ENV{STRATA_GGML_DIR}"
              "${STRATA_ROOT}/build-sycl-jit/_deps/strata_llamacpp-src"
              "${STRATA_ROOT}/build-sycl-aot/_deps/strata_llamacpp-src"
              "${STRATA_ROOT}/build-sycl-win/_deps/strata_llamacpp-src"
              "${STRATA_ROOT}/../llama.cpp")
    if(GGML_CANDIDATE)
      get_filename_component(GGML_CANDIDATE "${GGML_CANDIDATE}" ABSOLUTE)
      strata_vulkan_check_ggml("${GGML_CANDIDATE}" "${STRATA_GGML_PINNED}" STRATA_GGML_CAND)
      if(STRATA_GGML_CAND_OK)
        set(STRATA_GGML_DIR "${GGML_CANDIDATE}" CACHE PATH "llama.cpp checkout at the pinned commit" FORCE)
        set(STRATA_GGML_RESOLVED TRUE)
        set(STRATA_GGML_MATCHED TRUE)
        set(STRATA_GGML_COMMIT "${STRATA_GGML_CAND_COMMIT}")
        break()
      elseif(EXISTS "${GGML_CANDIDATE}/ggml/CMakeLists.txt")
        list(APPEND STRATA_GGML_REJECTED "candidate ${GGML_CANDIDATE} ${STRATA_GGML_CAND_WHY}")
      endif()
    endif()
  endforeach()
endif()

# The gate, and the two outcomes.  Before this, the FATAL_ERROR below ran on every configure of this directory,
# including the probe-only one, which is what made Phase 0 stop being standalone.
if(NOT STRATA_GGML_RESOLVED)
  if(STRATA_VULKAN_ENGINE)
    set(_sv_reject "")
    foreach(_sv_line IN LISTS STRATA_GGML_REJECTED)
      string(APPEND _sv_reject "\n  ${_sv_line}")
    endforeach()
    message(FATAL_ERROR
      "No llama.cpp checkout at the pinned commit ${STRATA_GGML_PINNED} (read from ${STRATA_GGML_VERSION_FILE}: "
      "${STRATA_GGML_VERSION_LINE}); STRATA_VULKAN_ENGINE=ON needs one to build ggml from.${_sv_reject} "
      "Point -DSTRATA_GGML_DIR at a checkout at that commit (git clone https://github.com/ggml-org/llama.cpp.git "
      "llama.cpp && git -C llama.cpp checkout ${STRATA_GGML_PINNED}), clear the stale STRATA_GGML_DIR from this "
      "build dir's cache, or build the Phase 0 probe alone with -DSTRATA_VULKAN_ENGINE=OFF.")
  endif()
  # The Phase 0 route, restored: the probe and its shader chain are configured above and need nothing from
  # llama.cpp, so an engine-off configure stops here rather than requiring one.  Nothing from here down - the C
  # language, the engine's shader chain, ggml's ASM request and its CMP0194 landmine, the ../src libraries and the
  # engine executables - is part of the probe, and none of it runs.
  set(_sv_pin "at the pinned commit ${STRATA_GGML_PINNED}")
  if(NOT STRATA_GGML_PINNED)
    set(_sv_pin "and no pin to check a checkout against (${STRATA_GGML_VERSION_FILE} does not exist here)")
  endif()
  message(STATUS "strata vulkan: STRATA_VULKAN_ENGINE=OFF and no llama.cpp checkout ${_sv_pin}: configuring the "
                 "Phase 0 probe alone; the engine targets below are not configured (no ggml subproject, no engine "
                 "target, the default build stays vk_caps_probe)")
  foreach(_sv_line IN LISTS STRATA_GGML_REJECTED)
    message(STATUS "strata vulkan:   ${_sv_line}")
  endforeach()
  return()
endif()

# The line that used to assert the pin without checking it now reports the commit this file actually read, and says
# plainly when it is not the pinned one.
if(STRATA_GGML_MATCHED)
  message(STATUS "strata vulkan: ggml from ${STRATA_GGML_DIR}, verified at commit ${STRATA_GGML_COMMIT} "
                 "(pinned: ${STRATA_GGML_VERSION_FILE}: ${STRATA_GGML_VERSION_LINE})")
else()
  message(STATUS "strata vulkan: ggml from ${STRATA_GGML_DIR}, ${STRATA_GGML_UNPINNED_NOTE} - NOT verified as the "
                 "pinned commit, accepted because STRATA_GGML_ALLOW_UNPINNED=ON")
endif()
enable_language(C)

# vulkan/shaders/<name>.comp -> build/spv/engine_<name>.spv -> generated/vulkan_shaders/<name>.h
# The engine's own shader list, separate from the probe's because the probe's shaders are instruments (spin.comp is
# the one that measured Phase 0 P5) and this list is what the runtime builds pipelines from.  Phase 1 ships exactly
# one entry here - noop.comp, whose body has no loop and no host-flag poll, used only to prove the
# glslc -> SPIR-V -> VkShaderModule -> VkComputePipeline path on the real cards.  A Phase 2 kernel is a new .comp in
# this directory plus a new name in this list; vulkan/src/gpu/runtime.cpp's strata::gpu::register_pipeline is the call
# that turns its words into a dispatchable pipeline under a name launch_kernel() can be called with.
#
# It lives here, in the engine half, because it is the engine's: strata_gpu_vulkan add_dependencies() on it below, so
# configuring it in the probe-only route would drag Phase 1's .comp files into the Phase 0 default build.
set(STRATA_VULKAN_SHADERS noop)
file(MAKE_DIRECTORY ${GEN_DIR}/vulkan_shaders)
set(ENGINE_SHADER_HEADERS "")
foreach(name ${STRATA_VULKAN_SHADERS})
  set(engine_spv ${SPV_DIR}/engine_${name}.spv)
  set(engine_hdr ${GEN_DIR}/vulkan_shaders/${name}.h)
  add_custom_command(
    OUTPUT ${engine_spv}
    COMMAND ${GLSLC} -O --target-env=vulkan1.3 -fshader-stage=comp
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${name}.comp -o ${engine_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${name}.comp
    VERBATIM)
  add_custom_command(
    OUTPUT ${engine_hdr}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${engine_spv} ${engine_hdr} ${name}
    DEPENDS ${engine_spv} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py
    VERBATIM)
  list(APPEND ENGINE_SHADER_HEADERS ${engine_hdr})
endforeach()
add_custom_target(vulkan_engine_shaders DEPENDS ${ENGINE_SHADER_HEADERS})

# The ASM language ggml asks for, and CMake 4.1's policy CMP0194.  This block is what makes the plain configure of
# this directory reproducible under the mandated PowerShell prefix; the measured chain is:
#
#   1. ggml requests the ASM language unconditionally - the pinned checkout's ggml/CMakeLists.txt:3 is
#      `project("ggml" C CXX ASM)`, and the only assembler sources in that tree are ggml-metal's macOS metallib embed
#      (.s) and ggml-hexagon's HTP kernels, neither of which is enabled on Windows.  So ASM is requested and never used.
#   2. That checkout's own cmake_minimum_required(VERSION 3.14...3.28) leaves CMP0194 ("MSVC is not an assembler for
#      language ASM", introduced in CMake 4.1; this PC runs cmake 4.3.1-msvc1) UNSET, so
#      Modules/CMakeDetermineASMCompiler.cmake:232-234 runs `message(AUTHOR_WARNING ...)` whenever the ASM compiler it
#      picked has the MSVC id - which is cl.exe, since there is no other candidate on Windows.  AUTHOR_WARNING goes to
#      stderr.
#   3. winenv.ps1:3 leaves `$ErrorActionPreference='Stop'`, and under it PowerShell turns a native command's stderr
#      write into a terminating NativeCommandError.  cmake dies mid-configure and the build dir is left configured
#      halfway: measured, a scratch dir ended with CMakeFiles/generated/ggml/spv and Test-Path CMakeCache.txt = False,
#      pipeline exit 1:
#        CMake Warning (dev) at .../cmake-4.3/Modules/CMakeDetermineASMCompiler.cmake:234 (message):
#          Policy CMP194 is not set: MSVC is not an assembler for language ASM. ...
#        Call Stack: .../build-sycl-jit/_deps/strata_llamacpp-src/ggml/CMakeLists.txt:3 (project)
#      The warning is benign in itself (with Continue the same configure exits 0); it is the shell state that makes a
#      stderr write fatal, and changing that state per call is not something the gate authorises.  So the fix has to
#      stop the stderr write, inside this directory.
#
# Setting the policy in THIS scope is not enough, measured in a two-project reproduction of the same shape (outer =
# 3.22/CXX + add_subdirectory, inner = 3.14...3.28 + project(C CXX ASM)): `cmake_policy(SET CMP0194 OLD)` or `NEW`
# before the add_subdirectory leaves the child's warning exactly where it was (stderr 7 lines) - the subproject's own
# cmake_minimum_required re-establishes the policy state for its directory scope.  What does work: enable the language
# here, where the policy can be stated.  Once ASM is loaded in this scope, project("ggml" ... ASM) reuses the cached
# CMAKE_ASM_COMPILER, CMakeDetermineASMCompiler never runs in a scope where CMP0194 is unset, and stderr stays empty
# (same reproduction: exit 0, stderr 0 lines).
#
# CMP0194 NEW is the behaviour CMake 4.1+ prefers and it is available here: MSVC does have a real assembler, ml64.exe,
# in the same bin directory as the cl.exe this build already uses.  NEW without it fails ("No CMAKE_ASM_COMPILER could
# be found", measured exit 1), because NEW drops cl from the assembler candidates - which is why ggml's request cannot
# just be answered with NEW.  Where ml64 is not installed, OLD is stated instead: OLD is what CMake 4.3 already does
# when the policy is unset ("successfully enable ASM even if cl is the only available candidate"), so that branch
# changes no behaviour, it only stops the warning.  Either way nothing here compiles an assembler source, so the
# choice cannot move the engine build.
if(MSVC AND NOT CMAKE_VERSION VERSION_LESS 4.1)
  get_filename_component(STRATA_VULKAN_MASM_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
  find_program(STRATA_VULKAN_MASM NAMES ml64 ml64.exe HINTS "${STRATA_VULKAN_MASM_DIR}")
  if(STRATA_VULKAN_MASM)
    set(CMAKE_ASM_COMPILER "${STRATA_VULKAN_MASM}" CACHE FILEPATH "ASM compiler: the MSVC macro assembler next to the C compiler" FORCE)
    cmake_policy(SET CMP0194 NEW)
    message(STATUS "strata vulkan: CMP0194 NEW, ASM (which ggml requests and nothing here uses) is ${STRATA_VULKAN_MASM}")
  else()
    cmake_policy(SET CMP0194 OLD)
    message(STATUS "strata vulkan: CMP0194 OLD (no ml64 next to ${CMAKE_C_COMPILER}): the behaviour CMake 4.3 already "
                   "used with the policy unset, stated so the developer warning does not reach stderr")
  endif()
  enable_language(ASM)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

# The glob is over the whole host tree: the .cpp files are what strata compiles, and the tree count is what the
# configure log prints, so it can be checked against the tree as it stands. STRATA_ROOT is
# get_filename_component("${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE), so these are the plan's
# file(GLOB_RECURSE ... ${CMAKE_CURRENT_SOURCE_DIR}/../src/*) and .../../src/*.cpp, normalised so the generated
# project shows absolute paths. The .cu files under src/ are the CUDA kernels this backend replaces with GLSL ->
# SPIR-V, so they are counted and not compiled - with the host-side exceptions named below.
file(GLOB_RECURSE STRATA_VULKAN_TREE CONFIGURE_DEPENDS ${STRATA_ROOT}/src/*)
file(GLOB_RECURSE STRATA_VULKAN_HOST_SOURCES CONFIGURE_DEPENDS ${STRATA_ROOT}/src/*.cpp)
list(LENGTH STRATA_VULKAN_TREE STRATA_VULKAN_TREE_COUNT)
list(LENGTH STRATA_VULKAN_HOST_SOURCES STRATA_VULKAN_CPP_COUNT)

# Which .cu files are host code that happens to live in a .cu, and which are not.  All 64 .cu under ../src were counted
# for device syntax (<<< launches / __global__ / __device__ / __constant__ / __shared__ / threadIdx / blockIdx /
# __syncthreads / cudaLaunchKernel / shuffle intrinsics); five have none: pinned.cu (959 lines, 0, 0),
# prefill/ggml_cuda_host.cu (148, 0, 0), prefill/gemm_bf16_parity.cu (156, 0, 0), prefill/mmq_kq_lib.cu (61, 0, 0),
# prefill/mmq_nvfp4_w4a8.cu (25, 0, 0).  Only pinned.cu is compiled as C++ here, because the other four are CUDA
# host code in the other sense - iteration 5's evidence, once they were tried:
#   src\prefill\ggml_cuda_host.cu(4,10): error C1083: Cannot open include file: 'common.cuh': No such file or directory
#   src\prefill\mmq_kq_lib.cu(8,10): error C1083: Cannot open include file: 'common.cuh': No such file or directory
#   src\prefill\mmq_nvfp4_w4a8.cu(10,10): error C1083: Cannot open include file: 'common.cuh': No such file or directory
#   src\prefill\gemm_bf16_parity.cu(13,10): error C1083: Cannot open include file: 'cublas_v2.h': No such file or directory
# `common.cuh` is ggml's CUDA backend internal and `cublas_v2.h` is cuBLAS: those four belong with the CUDA half this
# port replaces (its prompt GEMM is its own path), not with the portable host tree a compat header can serve.
# pinned.cu, by contrast, has 0 device constructs and its Linux headers are already behind #ifdef _WIN32 - metal
# compiles it the same way (metal/CMakeLists.txt:92 in the host list, :101 LANGUAGE CXX).
set(STRATA_VULKAN_HOST_CU ${S}/core/pinned.cu)
set_source_files_properties(${STRATA_VULKAN_HOST_CU} PROPERTIES LANGUAGE CXX)
list(LENGTH STRATA_VULKAN_HOST_CU STRATA_VULKAN_HOST_CU_COUNT)
file(GLOB_RECURSE STRATA_VULKAN_CU CONFIGURE_DEPENDS ${STRATA_ROOT}/src/*.cu)
list(LENGTH STRATA_VULKAN_CU STRATA_VULKAN_CU_COUNT)
math(EXPR STRATA_VULKAN_KERNEL_CU_COUNT "${STRATA_VULKAN_CU_COUNT} - ${STRATA_VULKAN_HOST_CU_COUNT}")

# One host TU is left out, and the reason is the upstream build's own gate.  src/core/pinned_shared_test.cpp opens
# with "// Linux selftest for the optional MAP_SHARED backing of PinnedArena" and includes <unistd.h> at :9 with no
# platform guard (iteration 0: `src\core\pinned_shared_test.cpp(9,10): error C1083: Cannot open include file:
# 'unistd.h'`).  CMakeLists.txt:1096 adds it only under `if(CMAKE_SYSTEM_NAME STREQUAL "Linux")`.  Mapping mkstemp /
# close / /tmp onto Windows to make a Linux-backed-arena selftest "compile" would be a fake, not a compat header.
# src/ngram/ple_reader_test.cpp and src/kernels/cpu/pool_affinity_linux_test.cpp sit behind the same upstream gate
# (:1102, :1681) but compile as they are on this platform (the latter's body is inside `#if defined(__linux__)`), so
# they stay in the tree.
list(REMOVE_ITEM STRATA_VULKAN_HOST_SOURCES ${S}/core/pinned_shared_test.cpp)

# The engine's own main is the executable's source, not a library member.
set(STRATA_VULKAN_EXE ${S}/program/generate.cpp)
list(REMOVE_ITEM STRATA_VULKAN_HOST_SOURCES ${STRATA_VULKAN_EXE})

# ../src also holds the harnesses: 96 .cpp files define their own main() (the *_test.cpp / *_parity.cpp / *_bench.cpp
# selftests and the {device,load,overlap,concurrent,plan,q36}_main tools), each one built as its own executable
# upstream (CMakeLists.txt:117-122 is that pattern, and :1092-1097, :1681-1682).  Left in the same link as
# program/generate.cpp they collide - iteration 4's evidence:
#   strata_vulkan_host.lib(session_save_reclaim_test.obj) : error LNK2005: main already defined in generate.obj
#   draft_policy_test.obj : warning LNK4006: main already defined in suffix_drafter_test.obj; second definition ignored
# (102 of those warnings, from the CRT's own reference to main pulling object files out of the library).  So they are
# compiled - into their own library, which the build builds but the engine does not link - exactly as upstream
# compiles them into their own executables.  Which file is which is read from the files themselves, not from a list
# that rots: a main() at file scope.
set(STRATA_VULKAN_ENGINE_SOURCES "")
set(STRATA_VULKAN_HARNESS_SOURCES "")
foreach(strata_host_src IN LISTS STRATA_VULKAN_HOST_SOURCES)
  file(READ "${strata_host_src}" strata_host_src_text)
  string(REPLACE "\r\n" "\n" strata_host_src_text "${strata_host_src_text}")   # CMake's regex has no \r
  string(REGEX MATCHALL "(^|\n)[ \t]*int[ \t]+main[ \t]*\\(" strata_host_main "${strata_host_src_text}")
  if(strata_host_main)
    list(APPEND STRATA_VULKAN_HARNESS_SOURCES "${strata_host_src}")
  else()
    list(APPEND STRATA_VULKAN_ENGINE_SOURCES "${strata_host_src}")
  endif()
endforeach()
list(LENGTH STRATA_VULKAN_ENGINE_SOURCES STRATA_VULKAN_ENGINE_COUNT)
list(LENGTH STRATA_VULKAN_HARNESS_SOURCES STRATA_VULKAN_HARNESS_COUNT)
message(STATUS "strata vulkan: ../src has ${STRATA_VULKAN_TREE_COUNT} files, of which ${STRATA_VULKAN_CPP_COUNT} are "
               ".cpp; compiled here: ${STRATA_VULKAN_ENGINE_COUNT} "
               "engine .cpp + ${STRATA_VULKAN_HARNESS_COUNT} harness .cpp (their own main, upstream's own executables) "
               "+ ${STRATA_VULKAN_HOST_CU_COUNT} host .cu as C++; not compiled: the ${STRATA_VULKAN_KERNEL_CU_COUNT} .cu "
               "with device code (their launches are what the SPIR-V kernels replace) and src/core/"
               "pinned_shared_test.cpp (Linux-only upstream at CMakeLists.txt:1096)")

# src/core/device.cu is the one host-shaped .cu that cannot be compiled here: its poison path is CUDA-shaped
# (`#if !defined(STRATA_GPU_METAL)` at :22, `cudaFuncGetAttributes(&a, poison_kernel)` at :242,
# `poison_kernel<<<(unsigned) chunk, threads>>>((float*) base_ + b * threads, n - b * threads)` at :342) and cl.exe
# has no <<< at all - only nvcc / hipcc / clang's CUDA mode parse that.  The Vulkan port supplies the same symbols as
# its own file (src/core/device_vk.cpp) instead of pretending the launch compiles.
# S4: two more files in this library.
#   src/engine/missing.cpp   - hand-written, permanent: prints the qualified name of an entry point that has no Vulkan
#                              implementation yet and exits 3.
#   src/engine/stubs.cpp     - GENERATED by vulkan/tools/gen_stubs.py from a failed link of `strata` (see the file's own
#                              header).  It is regenerated, never edited: every symbol link.exe reported unresolved gets
#                              a self-naming stub, so the link closes without anything being implemented, and a run that
#                              reaches one stops at the entry point instead of continuing with a fake result.  Porting a
#                              kernel means making it stop being unresolved - re-run the generator from a fresh link log
#                              and its stub disappears.  It is a real file here (not in ../src) exactly as metal/ does it
#                              with metal/src/engine/stubs.cpp.
#
# STRATA_VULKAN_STUBS is the switch between the two states Phase 1 reads, so that the S2 state is RE-DERIVABLE instead
# of remembered in a report:
#   ON  (the default): strata links, because every unresolved kernel entry point is answered by a stub.
#   OFF: src/engine/stubs.cpp is not in the library.  The ../src host TUs compile exactly as they do with it, and the
#        link stops at the kernel entry points - the state the compat-header gate is quoted from ("host TUs compile,
#        kernel symbols unresolved on purpose").  Route:
#          cmake -S vulkan -B build-vulkan -DSTRATA_VULKAN_STUBS=OFF
#          cmake --build build-vulkan --config Release      # ends in LNK2019/LNK2001 + LNK1120, and nothing else
option(STRATA_VULKAN_STUBS "Answer the unresolved kernel entry points with the generated stubs (ON = strata links; OFF = the S2 state, kernel symbols left unresolved)" ON)
set(STRATA_VULKAN_STUB_SOURCES src/engine/stubs.cpp)
if(NOT STRATA_VULKAN_STUBS)
  set(STRATA_VULKAN_STUB_SOURCES "")
  message(STATUS "strata vulkan: STRATA_VULKAN_STUBS=OFF: src/engine/stubs.cpp is not built; the kernel entry points "
                 "stay unresolved (the link fails by design)")
endif()

add_library(strata_gpu_vulkan STATIC src/gpu/runtime.cpp src/core/device_vk.cpp src/engine/missing.cpp
                                     ${STRATA_VULKAN_STUB_SOURCES})
target_include_directories(strata_gpu_vulkan BEFORE PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include/vulkan_compat   # FIRST: the CUDA runtime names, on Vulkan
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STRATA_ROOT}/include
  ${STRATA_ROOT}/src/core
  ${STRATA_ROOT}/third_party/ggml)
target_link_libraries(strata_gpu_vulkan PUBLIC Vulkan::Vulkan)
# The generated engine shaders, so runtime.cpp's `#if __has_include("vulkan_shaders/noop.h")` finds the file the same
# build produced rather than silently compiling without it.
target_include_directories(strata_gpu_vulkan PUBLIC ${GEN_DIR})
add_dependencies(strata_gpu_vulkan vulkan_engine_shaders)
target_compile_features(strata_gpu_vulkan PUBLIC cxx_std_20)   # ../src is built with C++20 by the top-level project
target_compile_definitions(strata_gpu_vulkan PUBLIC STRATA_NATIVE_EXPERTS=1 STRATA_VERSION="0.1.44.1-vulkan")

# The CPU kernel files carry their own ISA contract and refuse to build without it.  Iteration 0 hit TWO of them
# (build-vulkan\logs\iter-00.log has exactly two `error C1189` lines, quoted there verbatim):
#   src\kernels\cpu\iq_avx1.cpp(36,1): error C1189: #error:  "iq_avx1.cpp must be compiled with AVX enabled (see the
#       per-source flags in CMakeLists.txt)"
#   src\kernels\cpu\kq_avx1.cpp(17,1): error C1189: #error:  "kq_avx1.cpp must be compiled with AVX enabled (see the
#       per-source flags in CMakeLists.txt)"
# q2_avx1.cpp produced NO error in iteration 0 (it compiled; its only diagnostic is a C4996 at :179): its guard at
# :26-28 is the opposite kind - `#if defined(__AVX2__) || defined(__FMA__) || defined(__F16C__) #error` - so it fires
# only when those ARE on.  It still needs the flag for a different reason: upstream compiles it /arch:AVX
# (CMakeLists.txt:1331-1336), and these are those same flags on the same files.
set_source_files_properties(${S}/kernels/cpu/iq_avx512.cpp ${S}/kernels/cpu/nvfp4_avx512.cpp
                            PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
set_source_files_properties(${S}/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
set_source_files_properties(${S}/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
set_source_files_properties(${S}/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
set_source_files_properties(${S}/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
set_source_files_properties(${S}/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
set_source_files_properties(${S}/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")

add_library(strata_vulkan_host STATIC ${STRATA_VULKAN_ENGINE_SOURCES} ${STRATA_VULKAN_HOST_CU})
target_include_directories(strata_vulkan_host PUBLIC
  ${S}/core ${STRATA_ROOT}/third_party/ggml
  ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src ${STRATA_GGML_DIR}/ggml/src/ggml-cpu)
target_link_libraries(strata_vulkan_host PUBLIC strata_gpu_vulkan ggml-cpu ggml-base)

# The harness TUs: compiled, and linked to nothing.  Upstream gives each of these its own executable
# (CMakeLists.txt:117-122), which is why 100 of them defining main() is not a problem there.  Collected in one archive
# here it is LIB's complaint - iteration 6 and 7's 101 `warning LNK4006: main already defined in
# suffix_drafter_test.obj; second definition ignored` - so the archive is told to ignore that one warning number:
# these mains are what upstream links separately, and none of them is in the engine's link (the exe links only
# strata_vulkan_host, and iteration 7 has no LNK2005 in it at all).
add_library(strata_vulkan_harness STATIC ${STRATA_VULKAN_HARNESS_SOURCES})
target_include_directories(strata_vulkan_harness PUBLIC
  ${S}/core ${STRATA_ROOT}/third_party/ggml
  ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src ${STRATA_GGML_DIR}/ggml/src/ggml-cpu)
target_link_libraries(strata_vulkan_harness PUBLIC strata_vulkan_host)
if(MSVC)
  set_target_properties(strata_vulkan_harness PROPERTIES STATIC_LIBRARY_OPTIONS "/ignore:4006")
endif()

add_executable(strata ${STRATA_VULKAN_EXE})

# vulkan_compat first on strata's own include path, stated on the target that runs the engine rather than only
# inherited through the library chain.  The order is the whole point: `#include <cuda_runtime.h>` in ../src has four
# possible answers on this checkout (measured with Get-ChildItem -Recurse -Filter cuda_runtime.h):
#   include\strata\hip_compat\cuda_runtime.h          <- the HIP port's
#   include\strata\platform\hip_compat\cuda_runtime.h <- the HIP port's, per-platform half
#   metal\include\metal_compat\cuda_runtime.h         <- the Metal port's
#   vulkan\include\vulkan_compat\cuda_runtime.h       <- this one
# and the first directory on the include path picks which one the engine compiles against.
target_include_directories(strata PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/vulkan_compat   # FIRST: the CUDA runtime names, on Vulkan
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STRATA_ROOT}/include
  ${S}/core
  ${STRATA_ROOT}/third_party/ggml)
target_link_libraries(strata PRIVATE strata_vulkan_host Vulkan::Vulkan)

# Flags for a long error loop: /utf-8 so the sources' non-ASCII comments are read as UTF-8 by cl.exe instead of
# mangled into the token stream, /EHsc so a real exception path is semantics and not a C4530 note per file,
# /FC so an error line names the file it came from, /MP because the loop rebuilds 145 translation units and the
# Visual Studio generator otherwise serialises them, /W3 like the probe. Not /W4 /WX yet: Phase 1 is reading
# errors, not grading warnings.
if(MSVC)
  target_compile_options(strata_gpu_vulkan PRIVATE /utf-8 /EHsc /W3 /FC /MP)
  target_compile_options(strata_vulkan_host PRIVATE /utf-8 /EHsc /W3 /FC /MP)
  target_compile_options(strata_vulkan_harness PRIVATE /utf-8 /EHsc /W3 /FC /MP)
  target_compile_options(strata PRIVATE /utf-8 /EHsc /W3 /FC)
else()
  target_compile_options(strata_gpu_vulkan PRIVATE -Wall)
  target_compile_options(strata_vulkan_host PRIVATE -Wall)
  target_compile_options(strata_vulkan_harness PRIVATE -Wall)
  target_compile_options(strata PRIVATE -Wall)
endif()

# S4's verification probe (probe/stub_probe.cpp, next to vk_caps_probe.cpp): it calls the first kernel entry point the
# generator had to stub and must stop inside the stub with that symbol's name and exit 3.  It is the only way to see
# the stub path fire on a CPU this engine refuses to run on (generate.cpp:2899 gates the expert path on AVX512-VNNI /
# AVX512-VBMI before any device call), and it touches no device, no model and no kernel.
add_executable(stub_probe probe/stub_probe.cpp)
target_link_libraries(stub_probe PRIVATE strata_vulkan_host)
if(MSVC)
  target_compile_options(stub_probe PRIVATE /utf-8 /EHsc /W3 /FC)
else()
  target_compile_options(stub_probe PRIVATE -Wall)
endif()
# With the stubs switched off there is no stub to reach, so this probe cannot link - it is out of the default build in
# that state, and its absence is stated here rather than showing up as an unexplained second link failure.
if(NOT STRATA_VULKAN_STUBS)
  set_target_properties(stub_probe PROPERTIES EXCLUDE_FROM_ALL TRUE)
  message(STATUS "strata vulkan: stub_probe is EXCLUDE_FROM_ALL (it calls a generated stub; with no stubs it cannot link)")
endif()

# strata-device: ../src/core/device_main.cpp, which upstream builds as its own executable (the pattern at
# CMakeLists.txt:117-122; the file defines main(), which is why the source scan above put it in the harness library).
# Linked here as the executable upstream would link it: device_info() for the card, the memory plan measured against
# cudaMemGetInfo, and --selftest's DeviceArena path - allocate, honour the alignment, write the arena's poison through
# strata::gpu::host_view, and refuse an over-allocation rather than wrap.
add_executable(strata-device ${S}/core/device_main.cpp)
target_include_directories(strata-device PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/vulkan_compat
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STRATA_ROOT}/include
  ${S}/core
  ${STRATA_ROOT}/third_party/ggml)
target_link_libraries(strata-device PRIVATE strata_vulkan_host Vulkan::Vulkan)

# strata_vk_smoke: this port's own runtime smoke, living in vulkan/ so that it is this backend's test rather than a
# ../src harness.  It calls no engine kernel and dispatches nothing: it opens the runtime, prints what the selection
# predicate kept and dropped, allocates through the sub-allocator, registers nothing, captures one empty graph,
# replays it, records and waits on a timeline-semaphore event, imports a host allocation, and reads cudaMemGetInfo
# before and after its own allocations.  It is the run the S3 report's self-check dump is quoted from.
add_executable(strata_vk_smoke probe/vk_runtime_smoke.cpp)
target_include_directories(strata_vk_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/vulkan_compat
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STRATA_ROOT}/include
  ${S}/core
  ${STRATA_ROOT}/third_party/ggml)
target_link_libraries(strata_vk_smoke PRIVATE strata_gpu_vulkan Vulkan::Vulkan)

if(MSVC)
  target_compile_options(strata-device PRIVATE /utf-8 /EHsc /W3 /FC)
  target_compile_options(strata_vk_smoke PRIVATE /utf-8 /EHsc /W3 /FC)
else()
  target_compile_options(strata-device PRIVATE -Wall)
  target_compile_options(strata_vk_smoke PRIVATE -Wall)
endif()

# STRATA_VULKAN_ENGINE=OFF: the targets are configured - they exist, `--target strata` still works - but they are out
# of the default build.  That is what keeps start-vulkan-probe.bat (build-vulkan-probe, cached OFF, builds every
# default target) building the probe and its shaders and nothing else, while the plain configure of this directory
# puts strata in the project and in ALL_BUILD.
if(NOT STRATA_VULKAN_ENGINE)
  message(STATUS "strata vulkan: STRATA_VULKAN_ENGINE=OFF, pinned checkout available: the engine targets and the "
                 "engine's shader target are configured but EXCLUDE_FROM_ALL (the default build stays the Phase 0 "
                 "probe and its own shaders; --target strata still works in this build dir)")
  foreach(strata_engine_target IN ITEMS strata strata_gpu_vulkan strata_vulkan_host strata_vulkan_harness stub_probe
                                        strata-device strata_vk_smoke vulkan_engine_shaders)
    set_target_properties(${strata_engine_target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endforeach()
endif()
