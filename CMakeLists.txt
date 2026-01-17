cmake_minimum_required(VERSION 3.20)

project(LCH-Farrow1 VERSION 1.0.0 LANGUAGES CXX)
# VS CODE INTELLISENSE FIX
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# ОПРЕДЕЛЕНИЕ ПЛАТФОРМЫ
# ============================================================================
if(WIN32)
    set(IS_WINDOWS TRUE)
    set(IS_LINUX FALSE)
    set(PLATFORM_NAME Windows)
    message(STATUS "✅ WINDOWS detected")
elseif(UNIX AND NOT APPLE)
    set(IS_LINUX TRUE)
    set(IS_WINDOWS FALSE)
    set(PLATFORM_NAME Linux)
    message(STATUS "✅ LINUX detected")
elseif(APPLE)
    set(IS_LINUX TRUE)
    set(IS_WINDOWS FALSE)
    set(PLATFORM_NAME macOS)
    message(STATUS "✅ macOS detected")
else()
    message(FATAL_ERROR "❌ Unknown platform!")
endif()

# ============================================================================
# GPU SUPPORT OPTIONS
# ============================================================================
if(UNIX AND NOT APPLE)
    # На Linux: CUDA выключена по умолчанию, OpenCL включена
    option(ENABLE_CUDA "Enable CUDA support" OFF)
    option(ENABLE_OPENCL "Enable OpenCL support" ON)
else()
    # На Windows: CUDA включена, OpenCL выключена
    option(ENABLE_CUDA "Enable CUDA support" ON)
    option(ENABLE_OPENCL "Enable OpenCL support" OFF)
endif()

# GPU Type Detection
option(TYPE_GPU "GPU Type, e.g., RTX 2080ti, RTX 3060, A100, V100" "auto-detect")
option(CUDA_ARCH "CUDA architecture (default: auto-detect)" "auto")

message(STATUS "GPU Support Options:")
message(STATUS "  ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "  ENABLE_OPENCL: ${ENABLE_OPENCL}")
message(STATUS "  TYPE_GPU: ${TYPE_GPU}")
message(STATUS "  CUDA_ARCH: ${CUDA_ARCH}")

# ============================================================================
# CUDA SUPPORT
# ============================================================================
if(ENABLE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "✅ CUDA found!")
        message(STATUS "   Version: ${CUDA_VERSION}")
        message(STATUS "   Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "   Include: ${CUDA_INCLUDE_DIRS}")
        
        if(CUDA_VERSION VERSION_LESS 2.0)
            message(WARNING "⚠️  CUDA version ${CUDA_VERSION} < 2.0, recommended 2.0+")
        endif()
        
        # Auto-detect GPU architecture if not specified
        if(NOT CUDA_ARCH STREQUAL "auto")
            message(STATUS "GPU Architecture from CMakePresets: ${CUDA_ARCH}")
        else()
            message(STATUS "Auto-detecting CUDA GPU architecture...")
            execute_process(
                COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/extras/demo_suite/deviceQuery
                OUTPUT_VARIABLE CUDA_DEVICE_QUERY
                ERROR_QUIET
            )
            if(NOT CUDA_DEVICE_QUERY)
                message(STATUS "deviceQuery not available, using common architectures: 70, 75, 80, 89")
                set(CUDA_ARCH "70;75;80;89")
            else()
                message(STATUS "GPU Info:\n${CUDA_DEVICE_QUERY}")
            endif()
        endif()
        
        message(STATUS "Target CUDA Arch: ${CUDA_ARCH}")
        set(CUDA_ENABLED TRUE)
    else()
        message(STATUS "❌ CUDA not found")
        message(STATUS "   Install: https://developer.nvidia.com/cuda-toolkit")
        message(STATUS "   Or disable: -DENABLE_CUDA=OFF")
        set(CUDA_ENABLED FALSE)
    endif()
else()
    message(STATUS "⏭️  CUDA disabled (ENABLE_CUDA=OFF)")
    set(CUDA_ENABLED FALSE)
endif()

# ============================================================================
# OpenCL SUPPORT
# ============================================================================
if(ENABLE_OPENCL)
    message(STATUS "🔍 Searching for OpenCL...")
    
    find_package(OpenCL QUIET)
    
    if(OpenCL_FOUND)
        message(STATUS "✅ OpenCL found!")
        message(STATUS "   Version: ${OpenCL_VERSION_STRING}")
        message(STATUS "   Include: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "   Libraries: ${OpenCL_LIBRARIES}")
        
        if(OpenCL_VERSION_STRING VERSION_LESS 2.0)
            message(WARNING "⚠️  OpenCL version ${OpenCL_VERSION_STRING} < 2.0 recommended")
        endif()
        
        set(OPENCL_ENABLED TRUE)
    else()
        message(STATUS "❌ OpenCL not found on this system")
        message(STATUS "   Ubuntu: sudo apt install opencl-headers ocl-icd-opencl-dev")
        message(STATUS "   Fedora: sudo dnf install opencl-headers ocl-icd-devel")
        message(STATUS "   macOS: brew install opencl-headers")
        message(STATUS "   AMD GPU: add AMD GPU driver package")
        set(OPENCL_ENABLED FALSE)
    endif()
else()
    message(STATUS "⏭️  OpenCL disabled (ENABLE_OPENCL=OFF)")
    set(OPENCL_ENABLED FALSE)
endif()

# ============================================================================
# clFFT SUPPORT (только если OpenCL включена)
# ============================================================================
if(OPENCL_ENABLED)
    # Проверка локального clFFT в проекте
    if(EXISTS "${CMAKE_SOURCE_DIR}/clFFT/include")
        set(CLFFT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/clFFT/include")
        message(STATUS "✅ Using local clFFT include: ${CLFFT_INCLUDE_DIR}")
    endif()
    
    if(EXISTS "${CMAKE_SOURCE_DIR}/clFFT/lib/x64")
        if(MSVC)
            set(CLFFT_LIBRARY "${CMAKE_SOURCE_DIR}/clFFT/lib/x64/clFFT.lib")
        else()
            set(CLFFT_LIBRARY "${CMAKE_SOURCE_DIR}/clFFT/lib/x64/libclFFT.a")
        endif()
        
        if(EXISTS "${CLFFT_LIBRARY}")
            message(STATUS "✅ Using local clFFT library: ${CLFFT_LIBRARY}")
        else()
            unset(CLFFT_LIBRARY)
        endif()
    endif()
    
    # Поиск в системных путях (если локальная не найдена)
    if(NOT CLFFT_INCLUDE_DIR)
        find_path(CLFFT_INCLUDE_DIR NAMES clFFT.h
            PATHS /usr/local/include /usr/include
            PATH_SUFFIXES clFFT
        )
    endif()
    
    if(NOT CLFFT_LIBRARY)
        find_library(CLFFT_LIBRARY NAMES clFFT
            PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
        )
    endif()
    
    # Проверка, найдена ли clFFT
    if(CLFFT_INCLUDE_DIR AND CLFFT_LIBRARY)
        message(STATUS "✅ clFFT found!")
        message(STATUS "   Include: ${CLFFT_INCLUDE_DIR}")
        message(STATUS "   Library: ${CLFFT_LIBRARY}")
        set(CLFFT_FOUND TRUE)
    else()
        message(WARNING "⚠️  clFFT not found")
        message(STATUS "   Ubuntu: sudo apt install libclfft-dev")
        message(STATUS "   Using CPU FFT fallback")
        set(CLFFT_FOUND FALSE)
        
        # Опционально: скачать через FetchContent
        option(USE_BUNDLED_CLFFT "Download and build clFFT if not found" ON)
        if(USE_BUNDLED_CLFFT)
            message(STATUS "📥 Attempting to download clFFT via FetchContent...")
            include(FetchContent)
            FetchContent_Declare(clFFT
                GIT_REPOSITORY https://github.com/clMathLibraries/clFFT.git
                GIT_TAG master
            )
            FetchContent_GetProperties(clFFT)
            if(NOT clFFT_POPULATED)
                FetchContent_Populate(clFFT)
            endif()
            
            if(EXISTS "${clFFT_SOURCE_DIR}/CMakeLists.txt")
                add_subdirectory(${clFFT_SOURCE_DIR} ${clFFT_BINARY_DIR} EXCLUDE_FROM_ALL)
                
                if(TARGET clFFT)
                    set(CLFFT_LIBRARY clFFT)
                    set(CLFFT_INCLUDE_DIR "${clFFT_SOURCE_DIR}/include")
                    set(CLFFT_FOUND TRUE)
                    message(STATUS "✅ clFFT downloaded and configured as target")
                elseif(TARGET clFFT::clFFT)
                    set(CLFFT_LIBRARY clFFT::clFFT)
                    set(CLFFT_INCLUDE_DIR "${clFFT_SOURCE_DIR}/include")
                    set(CLFFT_FOUND TRUE)
                    message(STATUS "✅ clFFT downloaded (using clFFT::clFFT target)")
                else()
                    message(WARNING "⚠️  clFFT downloaded but target not found")
                    set(CLFFT_FOUND FALSE)
                endif()
            else()
                message(WARNING "⚠️  clFFT CMakeLists.txt not found in ${clFFT_SOURCE_DIR}")
                set(CLFFT_FOUND FALSE)
            endif()
        endif()
    endif()
else()
    set(CLFFT_FOUND FALSE)
endif()

# ============================================================================
# C++ STANDARD & COMPILER FLAGS
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "C++ Standard: C${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Build type handling
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build Type: Multi-config (Debug/Release) - use --config with cmake --build")
else()
    message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
endif()

# Compiler-specific flags
if(MSVC)
    message(STATUS "Compiler: MSVC (Visual Studio ${MSVC_VERSION})")
    
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Od /RTC1 /D_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Oi /arch:AVX2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /O2 /Zi /DNDEBUG")
    
    add_compile_options(/W4 /EHsc /permissive- /Zc:inline)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTD_NO_WARNINGS)
    
    message(STATUS "Optimization: /O2 /arch:AVX2")
else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Compiler: GCC")
        set(COMPILER_NAME "GCC")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Compiler: Clang")
        set(COMPILER_NAME "Clang")
    else()
        message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
        set(COMPILER_NAME "${CMAKE_CXX_COMPILER_ID}")
    endif()
    
    set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb3 -O0 -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -Wall")
    
    add_compile_options(-Wall -Wextra -Wpedantic -ffast-math -funroll-loops)
    
    message(STATUS "Compiler: ${COMPILER_NAME}")
    message(STATUS "Optimization: -O3 -march=native")
endif()

# ============================================================================
# SOURCE FILES
# ============================================================================
message(STATUS "")
message(STATUS "Source dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build dir: ${CMAKE_BINARY_DIR}")

set(SOURCES
    src/main.cpp
#    src/mylib.cpp
#    src/application.cpp
#    src/signalgenerator.cpp
#    src/gpuprocessor.cpp
#    src/validator.cpp
#    src/reporter.cpp
)

#set(HEADERS include/mylib.h )

if(CUDA_ENABLED)
    list(APPEND SOURCES src/gpukernel.cu)
    list(APPEND HEADERS include/gpukernel.h)
    message(STATUS "✅ CUDA sources added")
endif()

if(OPENCL_ENABLED)
    list(APPEND SOURCES
#        src/signalbuffer.cpp
#        src/filterbank.cpp
#        src/processingpipeline.cpp
#        src/profilingengine.cpp
#        src/lagrangematrix.cpp
#        src/lfmsignalgenerator.cpp
#        src/gpubackend/openclbackend.cpp
#        src/gpubackend/gpufactory.cpp
#        src/fractionaldelaycpu.cpp
#        src/resultcomparator.cpp
#        src/gpuprofiling.cpp
#        src/generator/generatorgpu.cpp
        src/generator/generator_gpu_new.cpp
        src/GPU/opencl_manager.cpp  
        src/GPU/gpu_memory_manager.cpp
        include/GPU/opencl_core.cpp
        include/GPU/kernel_program.cpp
        include/GPU/opencl_compute_engine.cpp
        include/GPU/command_queue_pool.cpp   
    )
    list(APPEND HEADERS
#        include/signalbuffer.h
#        include/filterbank.h
#        include/processingpipeline.h
#        include/profilingengine.h
#        include/lagrangematrix.h
#        include/lfmsignalgenerator.h
#        include/gpubackend/igpubackend.h
#        include/gpubackend/openclbackend.h
#        include/gpubackend/gpufactory.h
#        include/fractionaldelaycpu.h
#        include/resultcomparator.h
#        include/gpuprofiling.h
#        include/generator/generatorgpu.h
        include/interface/lfm_parameters.h
        include/interface/DelayParameter.h
        include/interface/combined_delay_param.h        
        include/generator/generator_gpu_new.h        
        include/GPU/opencl_manager.h     
        include/GPU/gpu_memory_manager.hpp    
        include/GPU/opencl_core.hpp
        include/GPU/kernel_program.hpp
        include/GPU/opencl_compute_engine.hpp
        include/GPU/command_queue_pool.hpp   
        include/Test/example_usage.hpp

    )
    message(STATUS "✅ OpenCL sources added")
endif()

# ============================================================================
# CREATE EXECUTABLE
# ============================================================================
if(CUDA_ENABLED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    add_executable(${PROJECT_NAME} ${SOURCES})
    
    if(NOT CUDA_ARCH STREQUAL "auto")
        set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
    endif()
    
    message(STATUS "✅ Created executable with CUDA support")
else()
    add_executable(${PROJECT_NAME} ${SOURCES})
    message(STATUS "✅ Created executable")
endif()

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================================================
# OPENCL & clFFT LINKING
# ============================================================================
if(OPENCL_ENABLED)
    # Основная линковка OpenCL
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
    
    target_include_directories(${PROJECT_NAME} PRIVATE 
        ${OpenCL_INCLUDE_DIRS}
    )
    
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
        OPENCL_ENABLED=1
    )
    
    message(STATUS "✅ OpenCL libraries linked")
    
    # Добавление clFFT (если найдена)
    if(CLFFT_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${CLFFT_LIBRARY})
        target_include_directories(${PROJECT_NAME} PRIVATE ${CLFFT_INCLUDE_DIR})
        
        target_compile_definitions(${PROJECT_NAME} PRIVATE 
            CLFFT_FOUND=1
        )
        
        message(STATUS "✅ clFFT library linked")
        message(STATUS "   Library: ${CLFFT_LIBRARY}")
        message(STATUS "   Include: ${CLFFT_INCLUDE_DIR}")
    else()
        target_compile_definitions(${PROJECT_NAME} PRIVATE 
            CLFFT_FOUND=0
        )
        message(WARNING "⚠️  clFFT not linked (using CPU FFT fallback)")
    endif()
    
    # Путь к директории с kernel файлами
    set(OPENCL_KERNEL_DIR "${CMAKE_SOURCE_DIR}/kernels")
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
        OPENCL_KERNEL_DIR="${OPENCL_KERNEL_DIR}"
    )
    
    message(STATUS "✅ OpenCL kernel directory: ${OPENCL_KERNEL_DIR}")
    
else()
    # OpenCL отключена
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
        OPENCL_ENABLED=0
        CLFFT_FOUND=0
    )
    message(STATUS "⏭️  OpenCL support disabled in this build")
endif()

# ============================================================================
# CUDA LINKING
# ============================================================================
if(CUDA_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ENABLED=1)
    message(STATUS "✅ CUDA libraries linked")
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ENABLED=0)
endif()

# ============================================================================
# STANDARD LIBRARY LINKING
# ============================================================================
if(NOT MSVC)
    target_link_libraries(${PROJECT_NAME} PRIVATE m)
endif()

# ============================================================================
# COMPILER OPTIONS & OPTIMIZATIONS
# ============================================================================
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2 /Gy /fp:precise)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -march=native)
endif()

# ============================================================================
# LOCAL clFFT HEADERS (if exist)
# ============================================================================
if(EXISTS "${CMAKE_SOURCE_DIR}/clFFT/include")
    message(STATUS "ℹ️  Local clFFT include found: ${CMAKE_SOURCE_DIR}/clFFT/include")
    target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/clFFT/include")
endif()

# ============================================================================
# COPY DLLs ON WINDOWS
# ============================================================================
if(MSVC)
    if(EXISTS "${CMAKE_SOURCE_DIR}/clFFT/lib/x64/clFFT.dll")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/clFFT/lib/x64/clFFT.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        )
    endif()
    
    if(EXISTS "${CMAKE_SOURCE_DIR}/clFFT/lib/x64/StatTimer.dll")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/clFFT/lib/x64/StatTimer.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        )
    endif()
endif()

# ============================================================================
# BUILD SUMMARY
# ============================================================================
message(STATUS "")
message(STATUS "╔════════════════════════════════════════╗")
message(STATUS "║          BUILD CONFIGURATION           ║")
message(STATUS "╠════════════════════════════════════════╣")
message(STATUS "║ Platform: ${PLATFORM_NAME}")
message(STATUS "║ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "║ C++ Standard: C${CMAKE_CXX_STANDARD}")
message(STATUS "║ Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "╠════════════════════════════════════════╣")
message(STATUS "║ CUDA Support: ${CUDA_ENABLED}")
message(STATUS "║ OpenCL Support: ${OPENCL_ENABLED}")
message(STATUS "║ clFFT Support: ${CLFFT_FOUND}")
message(STATUS "╚════════════════════════════════════════╝")
message(STATUS "")

# ============================================================================
# BUILD INSTRUCTIONS
# ============================================================================
if(IS_WINDOWS)
    message(STATUS "📝 BUILD INSTRUCTIONS (Windows - MSVC):")
    message(STATUS "")
    message(STATUS "  With CUDA:")
    message(STATUS "    cmake -B build -G \"Visual Studio 17 2022\" -DENABLE_CUDA=ON")
    message(STATUS "    cmake --build build --config Release")
    message(STATUS "")
    message(STATUS "  Without CUDA:")
    message(STATUS "    cmake -B build -G \"Visual Studio 17 2022\" -DENABLE_CUDA=OFF")
    message(STATUS "    cmake --build build --config Release")
    message(STATUS "")
else()
    message(STATUS "📝 BUILD INSTRUCTIONS (Linux/macOS - Ninja):")
    message(STATUS "")
    message(STATUS "  With CUDA (if available):")
    message(STATUS "    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON")
    message(STATUS "    ninja -C build")
    message(STATUS "")
    message(STATUS "  With OpenCL:")
    message(STATUS "    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENCL=ON")
    message(STATUS "    ninja -C build")
    message(STATUS "")
    message(STATUS "  CPU only:")
    message(STATUS "    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DENABLE_OPENCL=OFF")
    message(STATUS "    ninja -C build")
    message(STATUS "")
endif()
