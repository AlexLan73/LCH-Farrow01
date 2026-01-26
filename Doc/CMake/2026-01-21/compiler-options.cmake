# ============================================================================
# Compiler Options and Optimization Module
# cmake/compiler-options.cmake
# ============================================================================
# НАЗНАЧЕНИЕ: Установка флагов компилятора и оптимизаций
# Применяется ГЛОБАЛЬНО для всех целей
# ============================================================================

message(STATUS "")
message(STATUS "⚙️  Compiler Configuration:")
message(STATUS "")

# ============================================================================
# ОБЩИЕ НАСТРОЙКИ
# ============================================================================
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ Standard: C${CMAKE_CXX_STANDARD}")

# Export compile commands for IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# MSVC (Windows - Visual Studio)
# ============================================================================
if(MSVC)
    message(STATUS "  Platform: MSVC (Visual Studio)")
    
    # Флаги для Debug
    set(CMAKE_CXX_FLAGS_DEBUG 
        "/MDd /Zi /Od /RTC1 /D_DEBUG /W4 /EHsc /permissive- /Zc:inline"
    )
    
    # Флаги для Release (с оптимизацией)
    set(CMAKE_CXX_FLAGS_RELEASE 
        "/MD /O2 /Oi /arch:AVX2 /DNDEBUG /W4 /EHsc /permissive- /Zc:inline"
    )
    
    # Флаги для RelWithDebInfo
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO 
        "/MD /O2 /Zi /DNDEBUG /W4 /EHsc /permissive- /Zc:inline"
    )
    
    # Дополнительные компилятивные опции
    add_compile_options(
        /W4              # Warning level 4
        /EHsc            # Exception handling
        /permissive-     # Standards conformance
        /Zc:inline       # Inline functions
    )
    
    # Определения
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTD_NO_WARNINGS
    )
    
    message(STATUS "  Optimization: /O2 /arch:AVX2")
    message(STATUS "  Warning Level: /W4")
    
# ============================================================================
# GCC / Clang (Linux / macOS)
# ============================================================================
else()
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "  Platform: GCC")
        set(COMPILER_TYPE "GCC")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "  Platform: Clang")
        set(COMPILER_TYPE "Clang")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        message(STATUS "  Platform: Apple Clang")
        set(COMPILER_TYPE "AppleClang")
    endif()
    
    # Флаги для Debug
    set(CMAKE_CXX_FLAGS_DEBUG 
        "-g -ggdb3 -O0 -Wall -Wextra -Wpedantic -fno-omit-frame-pointer"
    )
    
    # Флаги для Release (с оптимизацией)
    set(CMAKE_CXX_FLAGS_RELEASE 
        "-O3 -march=native -mtune=native -Wall -DNDEBUG -ffast-math -funroll-loops"
    )
    
    # Флаги для RelWithDebInfo
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO 
        "-O2 -g -march=native -Wall -DNDEBUG"
    )
    
    # Дополнительные компилятивные опции
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -ffast-math
        -funroll-loops
    )
    
    message(STATUS "  Optimization: -O3 -march=native")
    message(STATUS "  Warning Level: -Wall -Wextra -Wpedantic")
    
endif()

# ============================================================================
# ГЛОБАЛЬНЫЕ ОПЦИИ
# ============================================================================

# Позиционно-независимый код (PIC) для shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "  PIC (Position Independent Code): ON")
message(STATUS "")
