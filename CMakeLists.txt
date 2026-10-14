cmake_minimum_required(VERSION 3.24)

project(flashgrad VERSION 0.1.0 LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(FLASHGRAD_BUILD_TESTS    "Build unit tests"                                ON)
option(FLASHGRAD_BUILD_EXAMPLES "Build example executables"                       ON)
option(FLASHGRAD_SYNC_KERNELS   "Synchronize after every kernel launch (debug)"   OFF)
option(FLASHGRAD_WERROR         "Treat host compiler warnings as errors"          ON)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# RTX 3090 = sm_86. Override with -DCMAKE_CUDA_ARCHITECTURES=... for other GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Shared compile settings (applied to every flashgrad target)
# ---------------------------------------------------------------------------
add_library(flashgrad_options INTERFACE)
target_compile_options(flashgrad_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${FLASHGRAD_WERROR}>>:-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  # Keep source-line info in device code so Nsight Compute can map SASS to source.
  # (-G in Debug already includes full device debug info.)
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)
if(FLASHGRAD_SYNC_KERNELS)
  target_compile_definitions(flashgrad_options INTERFACE FLASHGRAD_SYNC_KERNELS=1)
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(flashgrad
  src/core/allocator.cpp
  src/core/device.cpp
  src/core/dtype.cpp
  src/core/error.cpp
  src/core/factory.cpp
  src/core/print.cpp
  src/core/random.cpp
  src/core/shape.cpp
  src/core/storage.cpp
  src/core/tensor.cpp
  src/core/tensor_impl.cpp
  src/cpu/ops.cpp
  src/cuda/memory.cpp
  src/cuda/runtime.cpp
  src/cuda/smoke.cu
)
add_library(flashgrad::flashgrad ALIAS flashgrad)

target_include_directories(flashgrad
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src  # internal headers: core/, cpu/, cuda/
)
target_link_libraries(flashgrad
  PUBLIC  CUDA::cudart
  PRIVATE flashgrad_options
)
set_target_properties(flashgrad PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_definitions(flashgrad PUBLIC
  FLASHGRAD_VERSION_STRING="${PROJECT_VERSION}"
)

# ---------------------------------------------------------------------------
# Subprojects
# ---------------------------------------------------------------------------
if(FLASHGRAD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(FLASHGRAD_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
