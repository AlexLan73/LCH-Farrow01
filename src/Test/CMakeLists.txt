# ============================================================================
# Test Module CMakeLists (STATIC LIBRARY - includes cpp files)
# src/Test/CMakeLists.txt
# ============================================================================
# НАЗНАЧЕНИЕ: Test модуль (тесты антенны и сигналов)
# ============================================================================

message(STATUS "")
message(STATUS "🔧 Processing: src/Test/")
message(STATUS "")

# ============================================================================
# TEST МОДУЛЬ - STATIC LIBRARY (с cpp файлами)
# ============================================================================

# Исходные файлы
set(TEST_SOURCES
    test_antenna_fft_proc_max.cpp
    test_signal_sinusoids.cpp
    test_fractional_delay_processor.cpp
)

# Создаем статическую библиотеку
add_library(lfm_tests STATIC ${TEST_SOURCES})

message(STATUS "✅ Created library: lfm_tests (STATIC)")
message(STATUS "   Sources: ${TEST_SOURCES}")

# ============================================================================
# ПОДКЛЮЧЕНИЕ INCLUDE ДИРЕКТОРИЙ
# ============================================================================

target_include_directories(lfm_tests PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/Test
    ${CMAKE_SOURCE_DIR}/include/GPU
    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL
    ${CMAKE_SOURCE_DIR}/include/interface
    ${CMAKE_SOURCE_DIR}/include/radar
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCL_INCLUDE_DIRS}
)

# ============================================================================
# ЛИНКОВКА ЗАВИСИМОСТЕЙ
# ============================================================================

target_link_libraries(lfm_tests PUBLIC
    lfm_gpu
    lfm_opencl_manager
    OpenCL::OpenCL
)

if(CLFFT_FOUND)
    target_link_libraries(lfm_tests PUBLIC "${CLFFT_LIB}")
    target_include_directories(lfm_tests PUBLIC "${CLFFT_INCLUDE_DIR}")
endif()

if(NLOHMANN_JSON_FOUND)
    target_link_libraries(lfm_tests PUBLIC nlohmann_json::nlohmann_json)
endif()

# ============================================================================
# COMPILE DEFINITIONS
# ============================================================================

target_compile_definitions(lfm_tests PUBLIC
    TEST_MODULE_ENABLED=1
    OPENCL_ENABLED=1
    GPU_MODULE_ENABLED=1
    CL_TARGET_OPENCL_VERSION=300
)

message(STATUS "✅ Test module configured (STATIC library)")

# ============================================================================
# ОТДЕЛЬНЫЙ EXECUTABLE ДЛЯ ТЕСТОВ FractionalDelayProcessor
# ============================================================================

add_executable(test_fractional_delay test_fractional_delay_processor.cpp)

target_include_directories(test_fractional_delay PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/Test
    ${CMAKE_SOURCE_DIR}/include/GPU
    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL
    ${CMAKE_SOURCE_DIR}/include/interface
    ${CMAKE_SOURCE_DIR}/include/radar
    ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(test_fractional_delay PRIVATE
    lfm_gpu
    lfm_opencl_manager
    OpenCL::OpenCL
)

if(CLFFT_FOUND)
    target_link_libraries(test_fractional_delay PRIVATE "${CLFFT_LIB}")
endif()

if(NLOHMANN_JSON_FOUND)
    target_link_libraries(test_fractional_delay PRIVATE nlohmann_json::nlohmann_json)
endif()

target_compile_definitions(test_fractional_delay PRIVATE
    TEST_MODULE_ENABLED=1
    OPENCL_ENABLED=1
    GPU_MODULE_ENABLED=1
    CL_TARGET_OPENCL_VERSION=300
)

message(STATUS "✅ Created executable: test_fractional_delay")
message(STATUS "")
