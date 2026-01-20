# CMakeLists.txt для FractionalDelayProcessor

cmake_minimum_required(VERSION 3.16)
project(FractionalDelayProcessor VERSION 2.0.0 LANGUAGES CXX)

# ════════════════════════════════════════════════════════════════════════════
# Опции компиляции
# ════════════════════════════════════════════════════════════════════════════

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# ════════════════════════════════════════════════════════════════════════════
# Зависимости
# ════════════════════════════════════════════════════════════════════════════

# OpenCL
find_package(OpenCL REQUIRED)

# clFFT (опционально, для FFT обработки)
# find_package(clFFT REQUIRED)

# ════════════════════════════════════════════════════════════════════════════
# Включаемые пути (УБЕДИТЕСЬ ЧТО ОНИ ЕСТЬ В ВАШЕМ ПРОЕКТЕ)
# ════════════════════════════════════════════════════════════════════════════

set(GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../GPU")
set(GENERATOR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../generator")
set(INTERFACE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../interface")
set(RADAR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

include_directories(
    ${GPU_INCLUDE_DIR}
    ${GENERATOR_INCLUDE_DIR}
    ${INTERFACE_INCLUDE_DIR}
    ${RADAR_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
)

# ════════════════════════════════════════════════════════════════════════════
# БИБЛИОТЕКА: FractionalDelayProcessor
# ════════════════════════════════════════════════════════════════════════════

# Файлы источников (из архитектуры GPU)
set(GPU_SOURCES
    ../GPU/opencl_core.cpp
    ../GPU/opencl_compute_engine.cpp
    ../GPU/gpu_memory_buffer.cpp
    ../GPU/command_queue_pool.cpp
    ../GPU/kernel_program.cpp
    # ... добавить другие файлы GPU архитектуры
)

# Файлы источников (генератор сигналов)
set(GENERATOR_SOURCES
    ../generator/generator_gpu_new.cpp
    # ... добавить другие файлы генератора
)

# Файлы нашего процессора дробной задержки
set(PROCESSOR_SOURCES
    fractional_delay_processor.cpp
)

# Создать библиотеку
add_library(fractional_delay_processor_lib
    ${PROCESSOR_SOURCES}
    ${GPU_SOURCES}
    ${GENERATOR_SOURCES}
)

# Линковать с OpenCL
target_link_libraries(fractional_delay_processor_lib
    PUBLIC ${OpenCL_LIBRARIES}
    # PUBLIC clFFT::clFFT  # если используем FFT
)

# ════════════════════════════════════════════════════════════════════════════
# ИСПОЛНЯЕМЫЙ ФАЙЛ: Пример использования
# ════════════════════════════════════════════════════════════════════════════

add_executable(fractional_delay_example
    fractional_delay_example.cpp
)

target_link_libraries(fractional_delay_example
    fractional_delay_processor_lib
    ${OpenCL_LIBRARIES}
)

# ════════════════════════════════════════════════════════════════════════════
# Опции компиляции для DEBUG / RELEASE
# ════════════════════════════════════════════════════════════════════════════

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    message(STATUS "Building in DEBUG mode")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
    message(STATUS "Building in RELEASE mode")
endif()

# ════════════════════════════════════════════════════════════════════════════
# Информация о сборке
# ════════════════════════════════════════════════════════════════════════════

message(STATUS "")
message(STATUS "════════════════════════════════════════════════════════════")
message(STATUS "FractionalDelayProcessor Build Configuration")
message(STATUS "════════════════════════════════════════════════════════════")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
message(STATUS "════════════════════════════════════════════════════════════")
message(STATUS "")

# ════════════════════════════════════════════════════════════════════════════
# Использование:
# ════════════════════════════════════════════════════════════════════════════

# Создание build директории:
# $ mkdir build && cd build
#
# Debug сборка:
# $ cmake -DCMAKE_BUILD_TYPE=Debug ..
# $ make
#
# Release сборка:
# $ cmake -DCMAKE_BUILD_TYPE=Release ..
# $ make
#
# Запуск примера:
# $ ./fractional_delay_example
