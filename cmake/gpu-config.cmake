# ============================================================================
# GPU Configuration Module
# cmake/gpu-config.cmake
# ============================================================================
# НАЗНАЧЕНИЕ: Выбор GPU платформы (CUDA или OpenCL)
# Значения берутся из CMakePresets.json
# ============================================================================

message(STATUS "")
message(STATUS "🔍 GPU Configuration:")
message(STATUS "")

# ============================================================================
# ПОЛУЧЕНИЕ НАСТРОЕК ИЗ PRESETS
# ============================================================================
# Если пришли из CMakePresets - используем их
# Иначе - устанавливаем по умолчанию для платформы

if(NOT DEFINED ENABLE_CUDA)
    if(IS_WINDOWS)
        set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA support (Windows default)")
    else()
        set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support (Linux default)")
    endif()
endif()

if(NOT DEFINED ENABLE_OPENCL)
    if(IS_WINDOWS)
        set(ENABLE_OPENCL OFF CACHE BOOL "Enable OpenCL support (Windows default)")
    else()
        set(ENABLE_OPENCL ON CACHE BOOL "Enable OpenCL support (Linux default)")
    endif()
endif()

# GPU тип (для информации)
if(NOT DEFINED TYPE_GPU)
    set(TYPE_GPU "auto-detect" CACHE STRING "GPU type for information")
endif()

# CUDA архитектура
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "auto" CACHE STRING "CUDA architecture")
endif()

# ============================================================================
# ВЫВОД ТЕКУЩИХ НАСТРОЕК
# ============================================================================
message(STATUS "  ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "  ENABLE_OPENCL: ${ENABLE_OPENCL}")
message(STATUS "  TYPE_GPU: ${TYPE_GPU}")
message(STATUS "  CUDA_ARCH: ${CUDA_ARCH}")
message(STATUS "")

# ============================================================================
# ПРОВЕРКА: Хотя бы одна платформа должна быть включена
# ============================================================================
if(NOT ENABLE_CUDA AND NOT ENABLE_OPENCL)
    message(WARNING "⚠️  ⚠️  ⚠️  ВНИМАНИЕ: Ни CUDA, ни OpenCL не включены!")
    message(WARNING "   Проект будет скомпилирован БЕЗ GPU поддержки")
    message(WARNING "   Измените CMakePresets.json или передайте -DENABLE_OPENCL=ON")
endif()

# ============================================================================
# ПЕРЕМЕННЫЕ ОТЛАДКИ (по умолчанию OFF)
# ============================================================================
set(VERBOSE_GPU_CONFIG OFF CACHE BOOL "Enable verbose GPU configuration output")

if(VERBOSE_GPU_CONFIG)
    message(STATUS "DEBUG GPU CONFIG:")
    message(STATUS "  ENABLE_CUDA: ${ENABLE_CUDA}")
    message(STATUS "  ENABLE_OPENCL: ${ENABLE_OPENCL}")
    message(STATUS "  IS_WINDOWS: ${IS_WINDOWS}")
    message(STATUS "  IS_LINUX: ${IS_LINUX}")
endif()
