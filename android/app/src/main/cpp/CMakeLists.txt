cmake_minimum_required(VERSION 3.22)
project(rpcbridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp tree: same fork as iOS (StayLameBro/backburner-llama.cpp).
# iOS builds it as llama.xcframework with GGML_METAL + GGML_RPC; here we build
# the same sources for Android with Vulkan (or OpenCL) + RPC. The tail server
# (tools/split-prefill/tail-server.h) and ggml-rpc server come from that tree.
if(NOT DEFINED LLAMA_DIR)
  set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../llama.cpp")
endif()
if(NOT EXISTS "${LLAMA_DIR}/include/llama.h")
  message(FATAL_ERROR "llama.cpp not found at ${LLAMA_DIR} (-DLLAMA_DIR=...). Clone with --recursive.")
endif()

option(BACKBURNER_VULKAN "Use Vulkan GPU backend on Android" ON)
option(BACKBURNER_OPENCL "Use OpenCL GPU backend on Android (fallback)" OFF)

# phone-attn CPU kernel: scripts/sme/sme_attn.c. Same compile as iOS
# (-mcpu=apple-a18 -DSME_BENCH_MAIN -DSME_BENCH_NO_MAIN), but portable:
# NDK clang detects SME2 at runtime via sme2_available(); non-SME SoCs take
# the plain path, exactly like iOS with SME2 off.
add_library(sme_attn STATIC
  ../../../../../scripts/sme/sme_attn.c
)
target_compile_definitions(sme_attn PRIVATE SME_BENCH_MAIN SME_BENCH_NO_MAIN)
target_compile_options(sme_attn PRIVATE -O3)
# Best-effort SME2 codegen on arm64; harmless if the SoC lacks it (runtime gate).
if(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
  include(CheckCCompilerFlag)
  check_c_compiler_flag("-march=armv9-a+sme2" HAS_SME2_FLAG)
  if(HAS_SME2_FLAG)
    target_compile_options(sme_attn PRIVATE -march=armv9-a+sme2)
  endif()
endif()

add_library(rpcbridge SHARED
  rpcbridge.cc
  vulkan_engine.cc
  nnapi_page_engine.cc
)

target_include_directories(rpcbridge PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  "${LLAMA_DIR}/include"
  "${LLAMA_DIR}/ggml/include"
  "${LLAMA_DIR}/common"
  "${LLAMA_DIR}/tools/split-prefill"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../phone-attn"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../scripts/sme"
)

target_link_libraries(rpcbridge
  sme_attn
  android
  log
)

find_library(VULKAN_LIB vulkan)
if(VULKAN_LIB AND BACKBURNER_VULKAN)
  target_link_libraries(rpcbridge ${VULKAN_LIB})
  target_compile_definitions(rpcbridge PRIVATE BACKBURNER_HAS_VULKAN=1)
endif()

# NOTE: link the llama.cpp static libs (llama, ggml, ggml-base, ggml-cpu,
# ggml-vulkan|ggml-opencl, ggml-rpc) from your llama.cpp Android build here,
# e.g. via -DLLAMA_BUILD_DIR=... or add_subdirectory(${LLAMA_DIR} ...).
# Kept out of the default glob so this module configures even before the
# llama.cpp Android build exists; wire it in build-android.sh.
if(DEFINED LLAMA_BUILD_DIR)
  target_link_directories(rpcbridge PRIVATE "${LLAMA_BUILD_DIR}")
endif()
