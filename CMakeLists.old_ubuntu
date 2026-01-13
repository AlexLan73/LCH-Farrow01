
cmake_minimum_required(VERSION 3.20)


project(LCH-Farrow VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# ЧАСТЬ 0: ДЕТЕКЦИЯ GPU И CUDA
# ============================================================================

# Параметры для выбора GPU
# Для Ubuntu по умолчанию используем OpenCL
if(UNIX AND NOT APPLE)
    option(ENABLE_CUDA "Enable CUDA support" OFF)
    option(ENABLE_OPENCL "Enable OpenCL support" ON)
else()
    option(ENABLE_CUDA "Enable CUDA support" ON)
    option(ENABLE_OPENCL "Enable OpenCL support" OFF)
endif()

################################################################################
# GPU Type Detection
################################################################################
# Сначала объявляем переменную ДО проверки CUDA_ARCH

# ВАЖНО: Объявляем как option() чтобы CMakePresets мог передать значение
option(TYPE_GPU "GPU Type (e.g., RTX 2080ti, RTX 3060, A100, V100)" "")
# ВАЖНО: Объявляем как option() чтобы CMakePresets мог передать значение

message(STATUS "DEBUG: TYPE_GPU from CMakePresets = '${TYPE_GPU}'")
message(STATUS "DEBUG: CUDA_ARCH = '${CUDA_ARCH}'")

mark_as_advanced(TYPE_GPU)

# ВАЖНО: Проверяем СНАЧАЛА значение из CMakePresets (уже загруженное)
# Если оно пустое - только тогда auto-detect по CUDA_ARCH
if(NOT TYPE_GPU OR TYPE_GPU STREQUAL "")
    # Auto-detect по CUDA_ARCH если TYPE_GPU не задан в CMakePresets
    if(CUDA_ARCH MATCHES "75")
        set(TYPE_GPU "RTX 2080 / RTX 2080ti")
    elseif(CUDA_ARCH MATCHES " 80|88|89")
        set(TYPE_GPU "A100 / RTX 30690")
    elseif(CUDA_ARCH MATCHES "70")
        set(TYPE_GPU "V100 / Tesla P100")
    else()
        set(TYPE_GPU "Unknown / Auto-detect")
    endif()
endif()

# message(STATUS "🎮 GPU TYPE: ${TYPE_GPU}")


option(CUDA_ARCH "CUDA architecture (default: auto-detect)" "")

# Опция: использовать встроенную (bundled) сборку clFFT если системная не найдена
option(USE_BUNDLED_CLFFT "Download and build clFFT if not found" ON)

# ============================================================================
# ЧАСТЬ 0: ДЕТЕКЦИЯ GPU И CUDA
# ============================================================================

# Параметры для выбора GPU
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_OPENCL "Enable OpenCL support" OFF)
option(CUDA_ARCH "CUDA architecture (default: auto-detect)" "")

# ✅ ДОБАВИТЬ ЭТО - переменная TYPE_GPU
set(TYPE_GPU "" CACHE STRING "GPU Type (e.g., RTX 2080ti, V100, A100)")

# ✅ Автоматически определить GPU по архитектуре
if(TYPE_GPU STREQUAL "")
    if(CUDA_ARCH MATCHES "75")
        set(TYPE_GPU "RTX 2080/2080ti")
    elseif(CUDA_ARCH MATCHES "80")
        set(TYPE_GPU "A100 / RTX 3090")
    elseif(CUDA_ARCH MATCHES "70")
        set(TYPE_GPU "V100")
    else()
        set(TYPE_GPU "Unknown/Auto-detect")
    endif()
endif()

message(STATUS "🎮 GPU TYPE: ${TYPE_GPU}")




message(STATUS "")
message(STATUS "🎮 GPU SUPPORT OPTIONS:")
message(STATUS "   ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "   ENABLE_OPENCL: ${ENABLE_OPENCL}")
message(STATUS "")

# Попытка найти CUDA
if(ENABLE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "✅ CUDA НАЙДЕНА!")
        message(STATUS "   CUDA Version: ${CUDA_VERSION}")
        message(STATUS "   CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "   CUDA Include: ${CUDA_INCLUDE_DIRS}")
        
        # Определить GPU архитектуру
        if(CUDA_ARCH STREQUAL "")
            # Попробовать определить автоматически
            message(STATUS "🔍 Автоматическое определение GPU архитектуры...")
            execute_process(
                COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/extras/demo_suite/deviceQuery
                OUTPUT_VARIABLE CUDA_DEVICE_QUERY
                ERROR_QUIET
            )
            
            # Попробовать альтернативный способ
            if(NOT CUDA_DEVICE_QUERY)
                message(STATUS "⚠️  deviceQuery не найден, используем стандартные архитектуры")
                set(CUDA_ARCH "70;75;80")  # V100, RTX 2080, A100
            else()
                message(STATUS "📊 GPU Info:")
                message(STATUS "${CUDA_DEVICE_QUERY}")
            endif()
        endif()
        
        message(STATUS "   Target CUDA Arch: ${CUDA_ARCH}")
        set(CUDA_ENABLED TRUE)
    else()
        message(STATUS "⚠️  CUDA НЕ НАЙДЕНА")
        message(STATUS "   Установите CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit")
        message(STATUS "   Или отключите CUDA: -DENABLE_CUDA=OFF")
        set(CUDA_ENABLED FALSE)
    endif()
else()
    message(STATUS "⏭️  CUDA отключена (ENABLE_CUDA=OFF)")
    set(CUDA_ENABLED FALSE)
endif()

# Попытка найти OpenCL
if(ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
    if(OpenCL_FOUND)
        message(STATUS "✅ OpenCL НАЙДЕНА!")
        message(STATUS "   OpenCL Version: ${OpenCL_VERSION_STRING}")
        message(STATUS "   OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "   OpenCL Libraries: ${OpenCL_LIBRARIES}")
        
        # Проверка версии OpenCL (требуется >= 2.0)
        if(OpenCL_VERSION_STRING VERSION_LESS "2.0")
            message(WARNING "⚠️  OpenCL версия ${OpenCL_VERSION_STRING} < 2.0, могут быть проблемы")
        endif()
        
        # Предпочитаем локальную поставку clFFT если она есть в репозитории
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

        # Поиск clFFT в системных путях только если не найдена локальная
        if(NOT CLFFT_INCLUDE_DIR)
            find_path(CLFFT_INCLUDE_DIR
                NAMES clFFT.h
                PATHS /usr/local/include /usr/include
                PATH_SUFFIXES clFFT
            )
        endif()

        if(NOT CLFFT_LIBRARY)
            find_library(CLFFT_LIBRARY
                NAMES clFFT
                PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
            )
        endif()
        
        if(CLFFT_INCLUDE_DIR AND CLFFT_LIBRARY)
            message(STATUS "✅ clFFT НАЙДЕНА!")
            message(STATUS "   clFFT Library: ${CLFFT_LIBRARY}")
            message(STATUS "   clFFT Include: ${CLFFT_INCLUDE_DIR}")
            set(CLFFT_FOUND TRUE)
        else()
            message(WARNING "⚠️  clFFT НЕ НАЙДЕНА!")
            # Попробуем автоматически скачать и собрать clFFT с помощью FetchContent
            if(USE_BUNDLED_CLFFT)
                message(STATUS "⏬ Попробуем скачать и собрать clFFT (USE_BUNDLED_CLFFT=ON)")
                include(FetchContent)
                FetchContent_Declare(
                    clFFT
                    GIT_REPOSITORY https://github.com/clMathLibraries/clFFT.git
                    GIT_TAG master
                )
                FetchContent_GetProperties(clFFT)
                if(NOT clFFT_POPULATED)
                    FetchContent_Populate(clFFT)
                endif()

                # Попытаемся добавить поддиректорию clFFT как внешний проект
                if(EXISTS "${clFFT_SOURCE_DIR}/CMakeLists.txt")
                    add_subdirectory(${clFFT_SOURCE_DIR} ${clFFT_BINARY_DIR} EXCLUDE_FROM_ALL)
                    # После добавления цель может называться clFFT или clFFT::clFFT
                    if(TARGET clFFT)
                        set(CLFFT_LIBRARY clFFT)
                        set(CLFFT_INCLUDE_DIR "${clFFT_SOURCE_DIR}/inc")
                        set(CLFFT_FOUND TRUE)
                        message(STATUS "✅ clFFT собран и доступен как target 'clFFT'")
                    elseif(TARGET clFFT::clFFT)
                        set(CLFFT_LIBRARY clFFT::clFFT)
                        set(CLFFT_INCLUDE_DIR "${clFFT_SOURCE_DIR}/inc")
                        set(CLFFT_FOUND TRUE)
                        message(STATUS "✅ clFFT собран и доступен как target 'clFFT::clFFT'")
                    else()
                        message(WARNING "Не удалось создать цель clFFT из источников; оставляем CLFFT_FOUND=FALSE")
                        set(CLFFT_FOUND FALSE)
                    endif()
                else()
                    message(WARNING "Исходники clFFT не скачаны или отсутствуют: ${clFFT_SOURCE_DIR}")
                    set(CLFFT_FOUND FALSE)
                endif()
            else()
                message(STATUS "   Установите: sudo apt install libclfft-dev")
                message(STATUS "   FFT будет работать через CPU (временно)")
                set(CLFFT_FOUND FALSE)
            endif()
        endif()
        
        set(OPENCL_ENABLED TRUE)
    else()
        message(FATAL_ERROR "❌ OpenCL НЕ НАЙДЕНА! Установите OpenCL SDK")
    endif()
else()
    message(STATUS "⏭️  OpenCL отключена (ENABLE_OPENCL=OFF)")
    set(OPENCL_ENABLED FALSE)
    set(CLFFT_FOUND FALSE)
endif()

# ============================================================================
# ЧАСТЬ 1: ОПРЕДЕЛЕНИЕ ОПЕРАЦИОННОЙ СИСТЕМЫ
# ============================================================================

if(WIN32)
    set(IS_WINDOWS TRUE)
    set(IS_LINUX FALSE)
    set(PLATFORM_NAME "Windows")
    message(STATUS "🪟 Операционная система: WINDOWS")
    
elseif(UNIX AND NOT APPLE)
    set(IS_LINUX TRUE)
    set(IS_WINDOWS FALSE)
    set(PLATFORM_NAME "Linux")
    message(STATUS "🐧 Операционная система: LINUX")
    
elseif(APPLE)
    set(IS_LINUX TRUE)
    set(IS_WINDOWS FALSE)
    set(PLATFORM_NAME "macOS")
    message(STATUS "🍎 Операционная система: macOS")
    
else()
    message(FATAL_ERROR "❌ Неизвестная операционная система!")
endif()

# ============================================================================
# ЧАСТЬ 2: ВЕРСИЯ И СТАНДАРТЫ C++
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "C++ Standard: C++17")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ============================================================================
# ЧАСТЬ 3: ПАРАМЕТРЫ СБОРКИ
# ============================================================================

# Для multi-config генераторов (Visual Studio) CMAKE_BUILD_TYPE не используется
# CLion автоматически передаёт конфигурацию через --config Debug/Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build Type: Multi-config (Debug/Release) - используйте --config")
else()
    message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
endif()

# ============================================================================
# ЧАСТЬ 4: КОМПИЛЯТОР И ФЛАГИ ОПТИМИЗАЦИИ
# ============================================================================

if(MSVC)
    # ===== WINDOWS (MSVC компилятор) =====
    message(STATUS "🔧 Конфигурация: MSVC (Visual Studio)")
    
    # Флаги для Debug режима (отладка)
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Od /RTC1 /DEBUG")
    # Флаги для Release режима (оптимизация)
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Oi /arch:AVX2 /DNDEBUG")
    # Для RelWithDebInfo (Release с отладочной информацией)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /O2 /Zi /DNDEBUG")
    
    add_compile_options(
        /W4
        /EHsc
        /permissive-
        /Zc:inline
    )
    
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_WARNINGS
    )
    
    message(STATUS "Compiler: MSVC ${MSVC_VERSION}")
    message(STATUS "Optimization: /O2 /arch:AVX2")
    
else()
    # ===== LINUX / WSL / macOS (GCC / Clang) =====
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "🔧 Конфигурация: GCC")
        set(COMPILER_NAME "GCC")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "🔧 Конфигурация: Clang")
        set(COMPILER_NAME "Clang")
    else()
        message(STATUS "🔧 Конфигурация: ${CMAKE_CXX_COMPILER_ID}")
        set(COMPILER_NAME "${CMAKE_CXX_COMPILER_ID}")
    endif()
    
    set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb3 -O0 -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -Wall")
    
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -ffast-math
        -funroll-loops
    )
    
    message(STATUS "Compiler: ${COMPILER_NAME}")
    message(STATUS "Optimization: -O3 -march=native")
    
endif()

# ============================================================================
# ЧАСТЬ 5: СТРУКТУРА ПРОЕКТА
# ============================================================================

message(STATUS "")
message(STATUS "📁 Структура проекта:")
message(STATUS "   Source dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "   Build dir: ${CMAKE_BINARY_DIR}")
message(STATUS "")

set(SOURCES
    src/main.cpp
    src/mylib.cpp
    # OOP refactor sources
    src/application.cpp
    src/signal_generator.cpp
    src/gpu_processor.cpp
    src/validator.cpp
    src/reporter.cpp
)

set(HEADERS
    include/mylib.h
)

# Добавить CUDA файлы если включена
if(CUDA_ENABLED)
    list(APPEND SOURCES
        src/gpu_kernel.cu
    )
    list(APPEND HEADERS
        include/gpu_kernel.h
    )
    message(STATUS "✅ CUDA источники добавлены")
endif()

# Добавить OpenCL файлы если включена
if(OPENCL_ENABLED)
    list(APPEND SOURCES
        src/signal_buffer.cpp
        src/filter_bank.cpp
        src/processing_pipeline.cpp
        src/profiling_engine.cpp
        src/lagrange_matrix.cpp
        src/lfm_signal_generator.cpp
        src/gpu_backend/opencl_backend.cpp
        src/gpu_backend/gpu_factory.cpp
        src/fractional_delay_cpu.cpp
        src/result_comparator.cpp
        src/gpu_profiling.cpp
        src/generator/generator_gpu.cpp
    )
    list(APPEND HEADERS
        include/signal_buffer.h
        include/filter_bank.h
        include/processing_pipeline.h
        include/profiling_engine.h
        include/lagrange_matrix.h
        include/lfm_signal_generator.h
        include/gpu_backend/igpu_backend.h
        include/gpu_backend/opencl_backend.h
        include/gpu_backend/gpu_factory.h
        include/fractional_delay_cpu.h
        include/result_comparator.h
        include/gpu_profiling.h
        include/generator/generator_gpu.h
    )
    message(STATUS "✅ OpenCL источники добавлены")
endif()

# ============================================================================
# ЧАСТЬ 6: СОЗДАНИЕ ИСПОЛНЯЕМОГО ФАЙЛА
# ============================================================================

if(CUDA_ENABLED)
    # Использовать CUDA для компиляции
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    add_executable(${PROJECT_NAME} ${SOURCES})
    
    # Установить CUDA архитектуру
    if(NOT CUDA_ARCH STREQUAL "")
        set_target_properties(${PROJECT_NAME} PROPERTIES
            CUDA_ARCHITECTURES ${CUDA_ARCH}
        )
    endif()
    
    message(STATUS "✅ Используется CUDA компилятор для GPU кода")
else()
    add_executable(${PROJECT_NAME} ${SOURCES})
endif()

# Подключить include директорию
target_include_directories(${PROJECT_NAME} 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# Если в проекте добавлены локальные копии clFFT/OpenCL headers, подключим их
if(EXISTS "${CMAKE_SOURCE_DIR}/clFFT/include")
    message(STATUS "✅ Local clFFT include found: ${CMAKE_SOURCE_DIR}/clFFT/include")
    target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/clFFT/include")
endif()

# Локальная clFFT уже обработана выше (CLFFT_INCLUDE_DIR / CLFFT_LIBRARY)
# ============================================================================
# ЧАСТЬ 7: ЛИНКОВКА И ЗАВИСИМОСТИ
# ============================================================================

if(NOT MSVC)
    target_link_libraries(${PROJECT_NAME} PRIVATE m)
endif()

# Линковка CUDA
if(CUDA_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ENABLED=1)
    message(STATUS "✅ CUDA библиотеки добавлены в линковку")
endif()

# Линковка OpenCL
if(OPENCL_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
    target_include_directories(${PROJECT_NAME} PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCL_INCLUDE_DIRS}
    )
    
    # Добавить clFFT если найдена
    if(CLFFT_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${CLFFT_LIBRARY})
        target_include_directories(${PROJECT_NAME} PRIVATE ${CLFFT_INCLUDE_DIR})
        target_compile_definitions(${PROJECT_NAME} PRIVATE 
            OPENCL_ENABLED=1
            CLFFT_FOUND=1
        )
        message(STATUS "✅ clFFT библиотека добавлена в линковку")
    else()
        target_compile_definitions(${PROJECT_NAME} PRIVATE 
            OPENCL_ENABLED=1
            CLFFT_FOUND=0
        )
    endif()
    
    # Добавить директорию с kernel файлами
    set(OPENCL_KERNEL_DIR "${CMAKE_SOURCE_DIR}/kernels")
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
        OPENCL_KERNEL_DIR="${OPENCL_KERNEL_DIR}"
    )
    
    message(STATUS "✅ OpenCL библиотеки добавлены в линковку")
    message(STATUS "   OpenCL Kernel Dir: ${OPENCL_KERNEL_DIR}")
endif()

# ============================================================================
# ЧАСТЬ 8: ОПЦИИ КОМПИЛЯЦИИ ДЛЯ ЦЕЛИ
# ============================================================================

if(MSVC)
    target_compile_options(${PROJECT_NAME} 
        PRIVATE 
        /arch:AVX2
        /Gy
        /fp:precise
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -march=native)
endif()

# На Windows скопируем локальные DLL (если они есть) рядом с exe, чтобы не было ошибок загрузки
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
# ЧАСТЬ 9: ВЫВОД ИНФОРМАЦИИ О СБОРКЕ
# ============================================================================

message(STATUS "")
message(STATUS "📊 Информация о сборке:")
message(STATUS "   Platform: ${PLATFORM_NAME}")
message(STATUS "   Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "   C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "   Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "   CUDA: ${CUDA_ENABLED}")
message(STATUS "   OpenCL: ${OPENCL_ENABLED}")
message(STATUS "")

# ============================================================================
# ЧАСТЬ 10: ИНСТРУКЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ
# ============================================================================

if(IS_WINDOWS)
    message(STATUS "✨ Команды для сборки на Windows:")
    message(STATUS "")
    message(STATUS "   С CUDA:")
    message(STATUS "   cmake -B build -G \"Visual Studio 17 2022\" -DENABLE_CUDA=ON")
    message(STATUS "   cmake --build build --config Release")
    message(STATUS "")
    message(STATUS "   Без CUDA:")
    message(STATUS "   cmake -B build -G \"Visual Studio 17 2022\" -DENABLE_CUDA=OFF")
    message(STATUS "   cmake --build build --config Release")
    message(STATUS "")
else()
    message(STATUS "✨ Команды для сборки на Linux/macOS:")
    message(STATUS "")
    message(STATUS "   С CUDA и Ninja:")
    message(STATUS "   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON")
    message(STATUS "   ninja -C build")
    message(STATUS "")
    message(STATUS "   С OpenCL:")
    message(STATUS "   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENCL=ON")
    message(STATUS "   ninja -C build")
    message(STATUS "")
    message(STATUS "   Без GPU:")
    message(STATUS "   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF")
    message(STATUS "   ninja -C build")
    message(STATUS "")
endif()

# ============================================================================
# КОНЕЦ КОНФИГУРАЦИИ
# ============================================================================

message(STATUS "✅ CMake конфигурация успешна!")
