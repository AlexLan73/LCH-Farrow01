cmake_minimum_required(VERSION 3.16)
project(MatrixInversionProfiler LANGUAGES CXX HIP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROCm packages
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

# Add main executable
add_executable(matrix_invert matrix_invert.cpp)

# Link libraries
target_link_libraries(matrix_invert PRIVATE
    hip::host
    rocblas::rocblas
    rocsolver::rocsolver
)

# Set HIP architecture (for MI100/MI200 use gfx908/gfx90a)
# For AI100, use appropriate CDNA architecture
set_target_properties(matrix_invert PROPERTIES
    HIP_ARCHITECTURES "gfx908"
)

# Compiler options for optimization
target_compile_options(matrix_invert PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
)

# Add rocprof profiling options
add_custom_target(profile
    COMMAND rocprof --stats --basenames on $<TARGET_FILE:matrix_invert>
    DEPENDS matrix_invert
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running matrix inversion with rocprof..."
)

add_custom_target(profile_detailed
    COMMAND rocprof --timestamp on -i rocprof_counters.txt $<TARGET_FILE:matrix_invert>
    DEPENDS matrix_invert
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running detailed profiling with rocprof..."
)
