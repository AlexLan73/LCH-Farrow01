# ============================================================================
# GPU Module CMakeLists (STATIC LIBRARY - includes cpp files)
# src/GPU/CMakeLists.txt
# ============================================================================
# НАЗНАЧЕНИЕ: GPU модуль (GeneratorGPU, AntennaFFT)
# ============================================================================

message(STATUS "")
message(STATUS "🔧 Processing: src/GPU/")
message(STATUS "")

# ============================================================================
# GPU МОДУЛЬ - STATIC LIBRARY (с cpp файлами)
# ============================================================================

# Исходные файлы
set(GPU_SOURCES
    generator_gpu_new.cpp
    antenna_fft_proc_max.cpp
    fractional_delay_processor.cpp
)

# Создаем статическую библиотеку
add_library(lfm_gpu STATIC ${GPU_SOURCES})

message(STATUS "✅ Created library: lfm_gpu (STATIC)")
message(STATUS "   Sources: ${GPU_SOURCES}")

# ============================================================================
# ПОДКЛЮЧЕНИЕ INCLUDE ДИРЕКТОРИЙ
# ============================================================================

target_include_directories(lfm_gpu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/GPU
    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL
    ${CMAKE_SOURCE_DIR}/include/interface
    ${CMAKE_SOURCE_DIR}/clFFT/include        # Локальный clFFT с inline функциями
    ${OpenCL_INCLUDE_DIRS}
)

# ============================================================================
# ЛИНКОВКА ЗАВИСИМОСТЕЙ
# ============================================================================

target_link_libraries(lfm_gpu PUBLIC
    lfm_opencl_manager
    OpenCL::OpenCL
)

if(CLFFT_FOUND)
    target_link_libraries(lfm_gpu PUBLIC "${CLFFT_LIB}")
    target_include_directories(lfm_gpu PUBLIC "${CLFFT_INCLUDE_DIR}")
endif()

# ============================================================================
# COMPILE DEFINITIONS
# ============================================================================

target_compile_definitions(lfm_gpu PUBLIC
    GPU_MODULE_ENABLED=1
    OPENCL_ENABLED=1
    CL_TARGET_OPENCL_VERSION=300
)

message(STATUS "✅ GPU module configured (STATIC library)")
message(STATUS "")
