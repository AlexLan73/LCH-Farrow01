# ============================================================================
# Test Module CMakeLists (SIMPLIFIED - HEADER ONLY)
# src/Test/CMakeLists.txt
# ============================================================================
# НАЗНАЧЕНИЕ: Test модуль (только заголовки - header-only)
# ============================================================================

message(STATUS "")
message(STATUS "🔧 Processing: src/Test/")
message(STATUS "")

# ============================================================================
# TEST МОДУЛЬ - HEADER ONLY (без cpp файлов)
# ============================================================================

# Создаем interface library (только заголовки, нет реализации)
add_library(lfm_tests INTERFACE)

message(STATUS "✅ Created interface library: lfm_tests (header-only)")

# ============================================================================
# ПОДКЛЮЧЕНИЕ INCLUDE ДИРЕКТОРИЙ
# ============================================================================

target_include_directories(lfm_tests INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/Test
    ${CMAKE_SOURCE_DIR}/include/GPU
    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL
    ${CMAKE_SOURCE_DIR}/include/interface
    ${CMAKE_SOURCE_DIR}/include/radar
    ${CMAKE_SOURCE_DIR}/src
)

# ============================================================================
# ЛИНКОВКА ЗАВИСИМОСТЕЙ
# ============================================================================

target_link_libraries(lfm_tests INTERFACE
    lfm_gpu
    lfm_opencl_manager
    OpenCL::OpenCL
)

if(CLFFT_FOUND)
    target_link_libraries(lfm_tests INTERFACE "${CLFFT_LIB}")
    target_include_directories(lfm_tests INTERFACE "${CLFFT_INCLUDE_DIR}")
endif()

if(NLOHMANN_JSON_FOUND)
    target_link_libraries(lfm_tests INTERFACE nlohmann_json::nlohmann_json)
endif()

# ============================================================================
# COMPILE DEFINITIONS
# ============================================================================

target_compile_definitions(lfm_tests INTERFACE
    TEST_MODULE_ENABLED=1
    OPENCL_ENABLED=1
    GPU_MODULE_ENABLED=1
    CL_TARGET_OPENCL_VERSION=300
)

message(STATUS "✅ Test module configured (header-only)")
message(STATUS "")