# ============================================================================
# Debug Configuration Module
# cmake/debug-config.cmake
# ============================================================================
# НАЗНАЧЕНИЕ: Функции для отладки и вывода информации
# Помогает диагностировать проблемы с конфигурацией
# ============================================================================

# ============================================================================
# ФУНКЦИЯ: Логирование конфигурации
# ============================================================================
function(log_build_configuration)
    message(STATUS "")
    message(STATUS "╔══════════════════════════════════════════════════════════╗")
    message(STATUS "║          BUILD CONFIGURATION DEBUG INFO                 ║")
    message(STATUS "╚══════════════════════════════════════════════════════════╝")
    message(STATUS "")
    
    # Платформа
    message(STATUS "📍 Platform Information:")
    message(STATUS "   CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
    message(STATUS "   CMAKE_SYSTEM_VERSION: ${CMAKE_SYSTEM_VERSION}")
    message(STATUS "   CMAKE_HOST_SYSTEM: ${CMAKE_HOST_SYSTEM}")
    message(STATUS "   PLATFORM_NAME: ${PLATFORM_NAME}")
    message(STATUS "")
    
    # Компилятор
    message(STATUS "🔧 Compiler Information:")
    message(STATUS "   CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
    message(STATUS "   CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "   CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "   CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
    message(STATUS "")
    
    # Директории
    message(STATUS "📁 Directory Paths:")
    message(STATUS "   CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
    message(STATUS "   CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
    message(STATUS "   CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "")
    
    # Build type
    message(STATUS "🏗️  Build Information:")
    message(STATUS "   CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
    message(STATUS "   CMAKE_CONFIGURATION_TYPES: ${CMAKE_CONFIGURATION_TYPES}")
    message(STATUS "")
    
    # GPU Configuration
    message(STATUS "🎮 GPU Configuration:")
    message(STATUS "   ENABLE_CUDA: ${ENABLE_CUDA}")
    message(STATUS "   CUDA_ENABLED: ${CUDA_ENABLED}")
    if(CUDA_ENABLED)
        message(STATUS "   CUDA_VERSION: ${CUDA_VERSION}")
        message(STATUS "   CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "   CUDA_ARCH: ${CUDA_ARCH}")
    endif()
    message(STATUS "")
    
    message(STATUS "   ENABLE_OPENCL: ${ENABLE_OPENCL}")
    message(STATUS "   OPENCL_ENABLED: ${OPENCL_ENABLED}")
    if(OPENCL_ENABLED)
        message(STATUS "   OpenCL_VERSION_STRING: ${OpenCL_VERSION_STRING}")
        message(STATUS "   OpenCL_INCLUDE_DIRS: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "   OpenCL_LIBRARIES: ${OpenCL_LIBRARIES}")
    endif()
    message(STATUS "")
    
    # FFT Configuration
    message(STATUS "📊 FFT Configuration:")
    message(STATUS "   CLFFT_FOUND: ${CLFFT_FOUND}")
    if(CLFFT_FOUND)
        message(STATUS "   CLFFT_LIB: ${CLFFT_LIB}")
        message(STATUS "   CLFFT_INCLUDE_DIR: ${CLFFT_INCLUDE_DIR}")
        if(CLFFT_DLL)
            message(STATUS "   CLFFT_DLL: ${CLFFT_DLL}")
        endif()
    endif()
    message(STATUS "")
    
    # JSON Configuration
    message(STATUS "📋 JSON Configuration:")
    message(STATUS "   NLOHMANN_JSON_FOUND: ${NLOHMANN_JSON_FOUND}")
    message(STATUS "")
    
    message(STATUS "╚══════════════════════════════════════════════════════════╝")
    message(STATUS "")
endfunction()

# ============================================================================
# ВЫЗОВ ФУНКЦИИ ЛОГИРОВАНИЯ
# ============================================================================
# Раскомментируйте для отладки:
# log_build_configuration()
