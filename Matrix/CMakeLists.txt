cmake_minimum_required(VERSION 3.16)
project(HermitianMatrixInversion LANGUAGES CXX HIP)

# ==============================================================================
# Configuration
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU Architecture 
# AI100/MI100 = gfx908, MI250X = gfx90a, MI300 = gfx942
set(GPU_ARCH "gfx908" CACHE STRING "GPU architecture (gfx908 for AMD AI100)")

# ==============================================================================
# Find ROCm Packages
# ==============================================================================

# Set ROCm path if not found automatically
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

# ==============================================================================
# Main Executable (Basic version)
# ==============================================================================

add_executable(matrix_invert matrix_invert.cpp)
set_source_files_properties(matrix_invert.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(matrix_invert PRIVATE
    hip::host
    hip::device
    roc::rocblas
    roc::rocsolver
)

set_target_properties(matrix_invert PROPERTIES
    HIP_ARCHITECTURES "${GPU_ARCH}"
)

target_compile_options(matrix_invert PRIVATE
    -O3
    -ffast-math
    -funroll-loops
    -ffp-contract=fast
    -Wall
    -Wextra
    -DNDEBUG
)

# ==============================================================================
# Advanced Executable (Custom Kernels + Batched)
# ==============================================================================

add_executable(matrix_invert_advanced matrix_invert_advanced.cpp)
set_source_files_properties(matrix_invert_advanced.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(matrix_invert_advanced PRIVATE
    hip::host
    hip::device
    roc::rocblas
    roc::rocsolver
)

set_target_properties(matrix_invert_advanced PROPERTIES
    HIP_ARCHITECTURES "${GPU_ARCH}"
)

target_compile_options(matrix_invert_advanced PRIVATE
    -O3
    -ffast-math
    -funroll-loops
    -ffp-contract=fast
    -Wall
    -Wextra
    -DNDEBUG
)

# ==============================================================================
# Profiling Targets
# ==============================================================================

# Basic profiling - standard version
add_custom_target(profile
    COMMAND rocprof --stats --basenames on $<TARGET_FILE:matrix_invert>
    DEPENDS matrix_invert
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running basic matrix inversion with rocprof..."
)

# Basic profiling - advanced version
add_custom_target(profile_advanced
    COMMAND rocprof --stats --basenames on $<TARGET_FILE:matrix_invert_advanced>
    DEPENDS matrix_invert_advanced
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running advanced matrix inversion with rocprof..."
)

# Detailed profiling with hardware counters
add_custom_target(profile_detailed
    COMMAND rocprof --timestamp on -i ${CMAKE_SOURCE_DIR}/rocprof_counters.txt $<TARGET_FILE:matrix_invert_advanced>
    DEPENDS matrix_invert_advanced
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running detailed profiling with hardware counters..."
)

# ==============================================================================
# Install Target
# ==============================================================================

install(TARGETS matrix_invert matrix_invert_advanced DESTINATION bin)

# ==============================================================================
# Info
# ==============================================================================

message(STATUS "")
message(STATUS "=== Hermitian Matrix Inversion (AMD AI100, Debian) ===")
message(STATUS "  GPU Architecture: ${GPU_ARCH}")
message(STATUS "  ROCm Path: ${ROCM_PATH}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Build with:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DGPU_ARCH=gfx908")
message(STATUS "  make -j$(nproc)")
message(STATUS "")
message(STATUS "Run profiling with:")
message(STATUS "  make profile")
message(STATUS "  make profile_detailed")
message(STATUS "")

