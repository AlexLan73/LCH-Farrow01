# ============================================================================
# Dependencies Configuration Module
# cmake/dependencies.cmake
# ============================================================================
# НАЗНАЧЕНИЕ: Поиск и конфигурация ВСЕХ библиотек
# КРИТИЧНО: Этот файл НЕ должен изменяться между сборками!
# Пути к локальным библиотекам берутся из CMakePresets.json
# ============================================================================

message(STATUS "")
message(STATUS "📚 Searching for dependencies...")
message(STATUS "")

# ============================================================================
# CUDA SUPPORT
# ============================================================================
set(CUDA_ENABLED FALSE)

if(ENABLE_CUDA)
    message(STATUS "🔍 Searching for CUDA...")
    
    find_package(CUDA QUIET)
    
    if(CUDA_FOUND)
        set(CUDA_ENABLED TRUE)
        message(STATUS "✅ CUDA found!")
        message(STATUS "   Version: ${CUDA_VERSION}")
        message(STATUS "   Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "   Include: ${CUDA_INCLUDE_DIRS}")
        message(STATUS "   Libraries: ${CUDA_LIBRARIES}")
        
        if(CUDA_VERSION VERSION_LESS 11.0)
            message(WARNING "⚠️  CUDA version ${CUDA_VERSION} < 11.0, recommended 11.0+")
        endif()
        
        # Auto-detect GPU architecture if not specified
        if(NOT CUDA_ARCH STREQUAL "auto")
            message(STATUS "   GPU Architecture (from preset): ${CUDA_ARCH}")
        else()
            message(STATUS "   GPU Architecture: auto (will use CMake defaults)")
        endif()
        
    else()
        message(WARNING "❌ CUDA not found!")
        message(STATUS "   Install: https://developer.nvidia.com/cuda-toolkit")
        message(STATUS "   Or disable: -DENABLE_CUDA=OFF")
    endif()
    
else()
    message(STATUS "⏭️  CUDA disabled (ENABLE_CUDA=OFF)")
endif()

# ============================================================================
# OpenCL SUPPORT
# ============================================================================
set(OPENCL_ENABLED FALSE)

if(ENABLE_OPENCL)
    message(STATUS "🔍 Searching for OpenCL...")
    
    find_package(OpenCL QUIET)
    
    if(OpenCL_FOUND)
        set(OPENCL_ENABLED TRUE)
        message(STATUS "✅ OpenCL found!")
        message(STATUS "   Version: ${OpenCL_VERSION_STRING}")
        message(STATUS "   Include: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "   Libraries: ${OpenCL_LIBRARIES}")
        
        if(OpenCL_VERSION_STRING VERSION_LESS 2.0)
            message(WARNING "⚠️  OpenCL version ${OpenCL_VERSION_STRING} < 2.0 recommended")
        endif()
        
    else()
        message(WARNING "❌ OpenCL not found!")
        message(STATUS "   Ubuntu: sudo apt install opencl-headers ocl-icd-opencl-dev")
        message(STATUS "   Fedora: sudo dnf install opencl-headers ocl-icd-devel")
        message(STATUS "   macOS: brew install opencl-headers")
        message(STATUS "   ROCm (AMD): install AMD GPU driver package")
        message(STATUS "   Or disable: -DENABLE_OPENCL=OFF")
    endif()
    
else()
    message(STATUS "⏭️  OpenCL disabled (ENABLE_OPENCL=OFF)")
endif()

# ============================================================================
# clFFT LIBRARY (только если OpenCL включена)
# ============================================================================
set(CLFFT_FOUND FALSE)

if(OPENCL_ENABLED)
    message(STATUS "")
    message(STATUS "🔍 Searching for clFFT...")
    
    set(CLFFT_LOCAL_DIR "${CMAKE_SOURCE_DIR}/clFFT")
    
    if(IS_WINDOWS)
        # ============================================================
        # WINDOWS: ищем локальный clFFT
        # ============================================================
        message(STATUS "   [Windows mode] Looking for local clFFT...")
        
        if(EXISTS "${CLFFT_LOCAL_DIR}/include/clFFT.h")
            set(CLFFT_INCLUDE_DIR "${CLFFT_LOCAL_DIR}/include")
            message(STATUS "   ✅ clFFT headers found: ${CLFFT_INCLUDE_DIR}")
            
            # Определяем lib путь для Windows (x64)
            if(CMAKE_BUILD_TYPE MATCHES "Debug")
                set(CLFFT_LIB_PATHS
                    "${CLFFT_LOCAL_DIR}/lib/x64/Debug"
                    "${CLFFT_LOCAL_DIR}/lib/x64"
                    "${CLFFT_LOCAL_DIR}/lib"
                )
            else()
                set(CLFFT_LIB_PATHS
                    "${CLFFT_LOCAL_DIR}/lib/x64/Release"
                    "${CLFFT_LOCAL_DIR}/lib/x64"
                    "${CLFFT_LOCAL_DIR}/lib"
                )
            endif()
            
            # Ищем clFFT.lib
            find_library(CLFFT_LIB
                NAMES clFFT.lib clFFT
                PATHS ${CLFFT_LIB_PATHS}
                NO_DEFAULT_PATH
            )
            
            if(CLFFT_LIB)
                set(CLFFT_FOUND TRUE)
                message(STATUS "   ✅ clFFT.lib found: ${CLFFT_LIB}")
                
                # Ищем DLL для runtime
                get_filename_component(CLFFT_LIBDIR "${CLFFT_LIB}" DIRECTORY)
                find_file(CLFFT_DLL
                    NAMES clFFT.dll
                    PATHS "${CLFFT_LIBDIR}" "${CLFFT_LOCAL_DIR}/bin"
                    NO_DEFAULT_PATH
                )
                
                if(CLFFT_DLL)
                    message(STATUS "   ✅ clFFT.dll found: ${CLFFT_DLL}")
                else()
                    message(WARNING "   ⚠️  clFFT.dll not found (may cause runtime errors)")
                endif()
                
            else()
                message(WARNING "   ❌ clFFT.lib NOT found")
                message(STATUS "      Searched in:")
                foreach(path ${CLFFT_LIB_PATHS})
                    message(STATUS "      - ${path}")
                endforeach()
            endif()
            
        else()
            message(WARNING "   ❌ clFFT headers NOT found in ${CLFFT_LOCAL_DIR}/include")
            message(STATUS "      Expected: ${CLFFT_LOCAL_DIR}/include/clFFT.h")
        endif()
        
    else()
        # ============================================================
        # LINUX/MACOS: ищем системные пакеты clFFT
        # ============================================================
        message(STATUS "   [Linux/macOS mode] Looking for system clFFT...")
        
        # Попытка 1: find_package
        find_package(clFFT QUIET)
        
        # Попытка 2: pkg-config
        if(NOT CLFFT_FOUND)
            find_package(PkgConfig QUIET)
            if(PKG_CONFIG_FOUND)
                pkg_check_modules(CLFFT QUIET clFFT)
                if(CLFFT_FOUND)
                    set(CLFFT_LIB "${CLFFT_LIBRARIES}")
                    set(CLFFT_INCLUDE_DIR "${CLFFT_INCLUDE_DIRS}")
                    message(STATUS "   ✅ clFFT found via pkg-config")
                endif()
            endif()
        endif()
        
        # Попытка 3: ручной поиск
        if(NOT CLFFT_FOUND)
            find_library(CLFFT_LIB
                NAMES clFFT
                PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu 
                      /lib/x86_64-linux-gnu /opt/AMD/clFFT/lib64
            )
            
            find_path(CLFFT_INCLUDE_DIR
                NAMES clFFT.h
                PATHS /usr/local/include /usr/include /opt/AMD/clFFT/include
            )
            
            if(CLFFT_LIB AND CLFFT_INCLUDE_DIR)
                set(CLFFT_FOUND TRUE)
                message(STATUS "   ✅ clFFT found (manual search)")
                message(STATUS "      Library: ${CLFFT_LIB}")
                message(STATUS "      Include: ${CLFFT_INCLUDE_DIR}")
            endif()
        endif()
        
        if(NOT CLFFT_FOUND)
            message(WARNING "   ❌ clFFT NOT found!")
            message(STATUS "      Ubuntu/Debian: sudo apt install libclfft-dev")
            message(STATUS "      Fedora: sudo dnf install clFFT-devel")
            message(STATUS "      macOS: brew install clfft")
        endif()
        
    endif()
    
else()
    message(STATUS "⏭️  clFFT skipped (OpenCL disabled)")
endif()

# ============================================================================
# nlohmann_json LIBRARY
# ============================================================================
set(NLOHMANN_JSON_FOUND FALSE)

if(IS_LINUX)
    message(STATUS "")
    message(STATUS "🔍 Searching for nlohmann_json...")
    
    find_package(nlohmann_json QUIET)
    
    if(nlohmann_json_FOUND OR TARGET nlohmann_json::nlohmann_json)
        set(NLOHMANN_JSON_FOUND TRUE)
        message(STATUS "✅ nlohmann_json found!")
        
    else()
        message(WARNING "❌ nlohmann_json NOT found!")
        message(STATUS "   Ubuntu: sudo apt install nlohmann-json3-dev")
        message(STATUS "   Fedora: sudo dnf install nlohmann-json-devel")
        message(STATUS "   Or: header-only version from GitHub")
    endif()
    
endif()

# ============================================================================
# SUMMARY
# ============================================================================
message(STATUS "")
message(STATUS "📦 Dependencies Summary:")
message(STATUS "   CUDA_ENABLED: ${CUDA_ENABLED}")
message(STATUS "   OPENCL_ENABLED: ${OPENCL_ENABLED}")
message(STATUS "   CLFFT_FOUND: ${CLFFT_FOUND}")
message(STATUS "   NLOHMANN_JSON_FOUND: ${NLOHMANN_JSON_FOUND}")
message(STATUS "")
