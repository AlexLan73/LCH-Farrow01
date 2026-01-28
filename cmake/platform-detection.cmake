# ============================================================================
# Platform Detection Module
# cmake/platform-detection.cmake
# ============================================================================
# НАЗНАЧЕНИЕ: Определяет ОС и устанавливает переменные платформы
# Используется ВЕЗДЕ для условных компиляций
# ============================================================================

if(WIN32)
    set(IS_WINDOWS TRUE)
    set(IS_LINUX FALSE)
    set(PLATFORM_NAME "Windows")
    message(STATUS "✅ Platform detected: WINDOWS (WIN32)")
    
    if(MSVC)
        message(STATUS "   Compiler: MSVC (Visual Studio)")
        set(COMPILER_TYPE "MSVC")
    endif()
    
elseif(UNIX AND NOT APPLE)
    set(IS_LINUX TRUE)
    set(IS_WINDOWS FALSE)
    set(PLATFORM_NAME "Linux")
    message(STATUS "✅ Platform detected: LINUX (UNIX)")
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(COMPILER_TYPE "GCC")
        message(STATUS "   Compiler: GCC ${CMAKE_CXX_COMPILER_VERSION}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(COMPILER_TYPE "Clang")
        message(STATUS "   Compiler: Clang ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    
elseif(APPLE)
    set(IS_LINUX TRUE)
    set(IS_WINDOWS FALSE)
    set(PLATFORM_NAME "macOS")
    message(STATUS "✅ Platform detected: macOS (APPLE)")
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(COMPILER_TYPE "AppleClang")
        message(STATUS "   Compiler: Apple Clang ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    
else()
    message(FATAL_ERROR "❌ Unknown platform! Cannot proceed.")
endif()

# ============================================================================
# SET C++ STANDARD
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# BUILD TYPE HANDLING
# ============================================================================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "ℹ️  Build type not specified, using: Release")
endif()

if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "ℹ️  Multi-config generator detected (Debug/Release switchable)")
else()
    message(STATUS "ℹ️  Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ============================================================================
# EXPORT COMPILE COMMANDS (для IDE и analysis tools)
# ============================================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
message(STATUS "✅ compile_commands.json will be generated in build directory")
