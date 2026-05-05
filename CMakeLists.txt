cmake_minimum_required(VERSION 3.22)
project(tensorbit-run VERSION 1.0.0 LANGUAGES C CXX)

# --- Options ---
option(TENSORBIT_BUILD_TESTS "Build test suite" ON)
option(TENSORBIT_BACKEND_SIMD "Enable SIMD-accelerated kernels" ON)
option(TENSORBIT_BACKEND_CUDA "Enable CUDA GPU backend" OFF)
option(TENSORBIT_BACKEND_METAL "Enable Metal GPU backend (Apple)" OFF)
option(TENSORBIT_BACKEND_VULKAN "Enable Vulkan GPU backend" OFF)
option(TENSORBIT_BACKEND_NPU "Enable NPU backend interface" OFF)

# --- Compiler settings ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC warnings
if(MSVC)
    add_compile_options(/W4 /wd4100 /wd4189 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

# --- Platform detection ---
if(UNIX AND NOT APPLE)
    add_definitions(-DTB_PLATFORM_LINUX=1)
elseif(APPLE)
    add_definitions(-DTB_PLATFORM_APPLE=1)
elseif(WIN32)
    add_definitions(-DTB_PLATFORM_WINDOWS=1)
endif()

# --- Backend defines ---
if(TENSORBIT_BACKEND_SIMD)
    add_definitions(-DTB_HAS_SIMD=1)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        add_compile_options(-mavx2 -mfma)
        add_definitions(-DTB_SIMD_AVX2=1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        add_definitions(-DTB_SIMD_NEON=1)
    endif()
endif()

if(TENSORBIT_BACKEND_CUDA)
    add_definitions(-DTB_HAS_CUDA=1)
    list(APPEND TBRUN_C_SOURCES src/backend/cuda/registry.c)
    # Enable CUDA language for kernels.cu compilation
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;90")
    endif()
    find_package(CUDAToolkit 12.0 REQUIRED)
    # Add CUDA kernels as a separate static lib
    add_library(tensorbit-run-cuda STATIC src/backend/cuda/kernels.cu)
    target_include_directories(tensorbit-run-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(tensorbit-run-cuda PUBLIC CUDA::cudart tensorbit-run-c)
endif()

if(TENSORBIT_BACKEND_METAL)
    add_definitions(-DTB_HAS_METAL=1)
endif()

if(TENSORBIT_BACKEND_VULKAN)
    add_definitions(-DTB_HAS_VULKAN=1)
endif()

if(TENSORBIT_BACKEND_NPU)
    add_definitions(-DTB_HAS_NPU=1)
endif()

# --- C Core Library (pure C11, no STL, no exceptions, no RTTI) ---
set(TBRUN_C_SOURCES
    src/model.c
    src/model_esp32.c
    src/ops/linear.c
    src/ops/norm.c
    src/ops/activation.c
    src/ops/softmax.c
    src/ops/attention.c
    src/ops/rope.c
    src/ops/embedding.c
    src/ops/residual.c
    src/ops/conv2d.c
)

set(TBRUN_CPU_SOURCES
    src/backend/cpu/linear.c
    src/backend/cpu/norm.c
    src/backend/cpu/activation.c
    src/backend/cpu/softmax.c
    src/backend/cpu/attention.c
    src/backend/cpu/rope.c
    src/backend/cpu/embedding.c
    src/backend/cpu/residual.c
    src/backend/cpu/conv2d.c
    src/backend/cpu/registry.c
)

add_library(tensorbit-run-c STATIC
    ${TBRUN_C_SOURCES}
    ${TBRUN_CPU_SOURCES}
)
target_include_directories(tensorbit-run-c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(UNIX)
    target_link_libraries(tensorbit-run-c PUBLIC m)
endif()
# No C++-specific flags for C library

# --- C++ Wrapper Library ---
set(TBRUN_CPP_SOURCES
    src/model.cpp
)

add_library(tensorbit-run-cpp STATIC
    ${TBRUN_CPP_SOURCES}
)
target_include_directories(tensorbit-run-cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(tensorbit-run-cpp PUBLIC tensorbit-run-c)

# --- Executable ---
add_executable(tb-run src/main.cpp)
target_link_libraries(tb-run PRIVATE tensorbit-run-cpp)
if(TENSORBIT_BACKEND_CUDA)
    target_link_libraries(tb-run PRIVATE tensorbit-run-cuda)
endif()
target_compile_definitions(tb-run PRIVATE $<$<CONFIG:Debug>:TB_DEBUG>)

# --- Tests ---
if(TENSORBIT_BUILD_TESTS)
    enable_testing()

    add_executable(test_linear tests/test_linear.c)
    target_link_libraries(test_linear PRIVATE tensorbit-run-c)

    add_executable(test_norm tests/test_norm.c)
    target_link_libraries(test_norm PRIVATE tensorbit-run-c)

    add_executable(test_softmax tests/test_softmax.c)
    target_link_libraries(test_softmax PRIVATE tensorbit-run-c)

    add_executable(test_attention tests/test_attention.c)
    target_link_libraries(test_attention PRIVATE tensorbit-run-c)

    add_executable(test_tbm tests/test_tbm.c)
    target_link_libraries(test_tbm PRIVATE tensorbit-run-c)

    add_executable(test_integration tests/test_integration.cpp)
    target_link_libraries(test_integration PRIVATE tensorbit-run-cpp)

    add_test(NAME test_linear COMMAND test_linear)
    add_test(NAME test_norm COMMAND test_norm)
    add_test(NAME test_softmax COMMAND test_softmax)
    add_test(NAME test_attention COMMAND test_attention)
    add_test(NAME test_tbm COMMAND test_tbm)
    add_test(NAME test_integration COMMAND test_integration)
endif()

# --- Install ---
install(TARGETS tb-run DESTINATION bin)
install(TARGETS tensorbit-run-c tensorbit-run-cpp DESTINATION lib)
install(DIRECTORY include/tensorbit DESTINATION include)
