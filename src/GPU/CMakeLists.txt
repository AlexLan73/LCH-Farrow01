# ============================================================================
# GPU Module CMakeLists (SIMPLIFIED - HEADER ONLY)
# src/GPU/CMakeLists.txt
# ============================================================================
# НАЗНАЧЕНИЕ: GPU модуль (только заголовки - header-only)
# ============================================================================

message(STATUS "")
message(STATUS "🔧 Processing: src/GPU/")
message(STATUS "")

# ============================================================================
# GPU МОДУЛЬ - HEADER ONLY (без cpp файлов)
# ============================================================================

# Создаем interface library (только заголовки, нет реализации)
add_library(lfm_gpu INTERFACE)

message(STATUS "✅ Created interface library: lfm_gpu (header-only)")

# ============================================================================
# ПОДКЛЮЧЕНИЕ INCLUDE ДИРЕКТОРИЙ
# ============================================================================

target_include_directories(lfm_gpu INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/GPU
    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL
    ${CMAKE_SOURCE_DIR}/include/interface
)

# ============================================================================
# ЛИНКОВКА ЗАВИСИМОСТЕЙ
# ============================================================================

target_link_libraries(lfm_gpu INTERFACE
    lfm_opencl_manager
    OpenCL::OpenCL
)

if(CLFFT_FOUND)
    target_link_libraries(lfm_gpu INTERFACE "${CLFFT_LIB}")
    target_include_directories(lfm_gpu INTERFACE "${CLFFT_INCLUDE_DIR}")
endif()

# ============================================================================
# COMPILE DEFINITIONS
# ============================================================================

target_compile_definitions(lfm_gpu INTERFACE
    GPU_MODULE_ENABLED=1
    OPENCL_ENABLED=1
    CL_TARGET_OPENCL_VERSION=300
)

message(STATUS "✅ GPU module configured (header-only)")
message(STATUS "")