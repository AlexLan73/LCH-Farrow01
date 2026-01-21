# ============================================================================
# ManagerOpenCL Library CMakeLists
# src/ManagerOpenCL/CMakeLists.txt
# ============================================================================
# НАЗНАЧЕНИЕ: Создание отдельной библиотеки для управления OpenCL
# КРИТИЧНО: Эта библиотека используется GPU модулем и main приложением
# ============================================================================

message(STATUS "")
message(STATUS "📚 Processing: src/ManagerOpenCL/ (Creating library)")
message(STATUS "")

# ============================================================================
# ИСХОДНЫЕ ФАЙЛЫ БИБЛИОТЕКИ
# ============================================================================

set(OPENCL_MANAGER_SOURCES
    opencl_manager.cpp
    gpu_memory_manager.cpp
)

# ============================================================================
# ЗАГОЛОВОЧНЫЕ ФАЙЛЫ (для зависимостей)
# ============================================================================

set(OPENCL_MANAGER_HEADERS
    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL/opencl_manager.h
    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL/gpu_memory_manager.hpp
)

# ============================================================================
# СОЗДАНИЕ БИБЛИОТЕКИ
# ============================================================================

add_library(lfm_opencl_manager STATIC
    ${OPENCL_MANAGER_SOURCES}
    ${OPENCL_MANAGER_HEADERS}
)

message(STATUS "✅ Created library: lfm_opencl_manager (STATIC)")

# ============================================================================
# ПОДКЛЮЧЕНИЕ INCLUDE ДИРЕКТОРИЙ
# ============================================================================

target_include_directories(lfm_opencl_manager PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL
    ${CMAKE_SOURCE_DIR}/include/interface
)

target_include_directories(lfm_opencl_manager PRIVATE
    ${OpenCL_INCLUDE_DIRS}
)

# ============================================================================
# ЛИНКОВКА ЗАВИСИМОСТЕЙ
# ============================================================================

# OpenCL
target_link_libraries(lfm_opencl_manager PUBLIC
    OpenCL::OpenCL
)

# clFFT
if(CLFFT_FOUND)
    target_link_libraries(lfm_opencl_manager PUBLIC
        "${CLFFT_LIB}"
    )
    target_include_directories(lfm_opencl_manager PUBLIC
        "${CLFFT_INCLUDE_DIR}"
    )
    target_compile_definitions(lfm_opencl_manager PUBLIC CLFFT_FOUND=1)
else()
    target_compile_definitions(lfm_opencl_manager PUBLIC CLFFT_FOUND=0)
endif()

# ============================================================================
# COMPILER FLAGS
# ============================================================================

if(MSVC)
    target_compile_options(lfm_opencl_manager PRIVATE /arch:AVX2 /Gy)
else()
    target_compile_options(lfm_opencl_manager PRIVATE -march=native)
endif()

# ============================================================================
# COMPILE DEFINITIONS
# ============================================================================

target_compile_definitions(lfm_opencl_manager PUBLIC
    OPENCL_ENABLED=1
    CLFFT_VERSION=2
    CL_TARGET_OPENCL_VERSION=300
)

# ============================================================================
# SET PROPERTIES
# ============================================================================

set_target_properties(lfm_opencl_manager PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

message(STATUS "✅ ManagerOpenCL library configured")
message(STATUS "   Public include: ${CMAKE_SOURCE_DIR}/include")
message(STATUS "   OpenCL: Linked")
if(CLFFT_FOUND)
    message(STATUS "   clFFT: Linked")
endif()
message(STATUS "")
