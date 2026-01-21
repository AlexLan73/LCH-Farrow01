# ============================================================================
# Main CMakeLists for src/ (SIMPLIFIED - HEADER ONLY APPROACH)
# src/CMakeLists.txt
# ============================================================================
# НАЗНАЧЕНИЕ: Главная сборка (header-only модули)
# ============================================================================

message(STATUS "")
message(STATUS "🔧 Processing: src/ (Main build)")
message(STATUS "")

# ============================================================================
# ПОДМОДУЛИ (header-only)
# ============================================================================

# OpenCL Manager Library
add_subdirectory(ManagerOpenCL)

# GPU Module (header-only)
add_subdirectory(GPU)

# Tests Module (header-only)
add_subdirectory(Test)

# ============================================================================
# ИСХОДНЫЕ ФАЙЛЫ ГЛАВНОГО ПРИЛОЖЕНИЯ
# ============================================================================

set(MAIN_SOURCES
    main.cpp
)

# ============================================================================
# СОЗДАНИЕ ИСПОЛНЯЕМОГО ФАЙЛА
# ============================================================================

add_executable(LCH-Farrow1 ${MAIN_SOURCES})

message(STATUS "🔨 Creating main executable: LCH-Farrow1")

# ============================================================================
# ЛИНКОВКА БИБЛИОТЕК И МОДУЛЕЙ
# ============================================================================

# ВАЖНО: Для статических библиотек порядок критичен!
# Сначала наши библиотеки, потом внешние зависимости (clFFT, OpenCL)
target_link_libraries(LCH-Farrow1 PRIVATE
    # Наши библиотеки (порядок: от высокоуровневых к низкоуровневым)
    lfm_tests               # Tests Module (STATIC) - использует lfm_gpu
    lfm_gpu                 # GPU Module (STATIC) - использует lfm_opencl_manager
    lfm_opencl_manager      # OpenCL Manager (STATIC) - использует OpenCL/clFFT
)

# clFFT ДОЛЖЕН быть ПОСЛЕ наших библиотек (статическая линковка)
if(CLFFT_FOUND)
    target_link_libraries(LCH-Farrow1 PRIVATE "${CLFFT_LIB}")
endif()

# OpenCL ДОЛЖЕН быть последним
target_link_libraries(LCH-Farrow1 PRIVATE OpenCL::OpenCL)

# nlohmann_json
if(NLOHMANN_JSON_FOUND)
    target_link_libraries(LCH-Farrow1 PRIVATE nlohmann_json::nlohmann_json)
endif()

message(STATUS "✅ Linked library: lfm_opencl_manager")
message(STATUS "✅ Linked: OpenCL")
if(CLFFT_FOUND)
    message(STATUS "✅ Linked: clFFT")
endif()

# ============================================================================
# ПОДКЛЮЧЕНИЕ INCLUDE ДИРЕКТОРИЙ
# ============================================================================

target_include_directories(LCH-Farrow1 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL
    ${CMAKE_SOURCE_DIR}/include/GPU
    ${CMAKE_SOURCE_DIR}/include/Test
    ${CMAKE_SOURCE_DIR}/include/interface
    ${CMAKE_SOURCE_DIR}/include/radar
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCL_INCLUDE_DIRS}
)

# ============================================================================
# COMPILER FLAGS
# ============================================================================

if(MSVC)
    target_compile_options(LCH-Farrow1 PRIVATE /arch:AVX2 /Gy)
else()
    target_compile_options(LCH-Farrow1 PRIVATE -march=native)
endif()

# ============================================================================
# COMPILE DEFINITIONS
# ============================================================================

target_compile_definitions(LCH-Farrow1 PRIVATE
    OPENCL_ENABLED=1
    CLFFT_VERSION=2
    CL_TARGET_OPENCL_VERSION=300
)

# ============================================================================
# SET PROPERTIES
# ============================================================================

set_target_properties(LCH-Farrow1 PROPERTIES
    VERSION 1.0.0
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

message(STATUS "✅ Main executable configured: LCH-Farrow1")
message(STATUS "   Output: build/linux-nvidia/LCH-Farrow1")
message(STATUS "")