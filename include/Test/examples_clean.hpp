// ════════════════════════════════════════════════════════════════════════════
// examples_clean_api.hpp - примеры ПРАВИЛЬНОГО использования
// ════════════════════════════════════════════════════════════════════════════

#pragma once

#include "GPU/gpu_memory_manager.hpp"
#include "GPU/opencl_manager.h"
#include "generator/generator_gpu.h"

#include <memory>
#include <iostream>

namespace examples {

// ════════════════════════════════════════════════════════════════════════════
// ИНИЦИАЛИЗАЦИЯ (один раз в main)
// ════════════════════════════════════════════════════════════════════════════

inline void InitializeGPU() {
    // 1. Инициализировать OpenCL
    gpu::OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
    std::cout << gpu::OpenCLManager::GetInstance().GetDeviceInfo();

    // 2. Инициализировать менеджер памяти
    gpu::GPUMemoryManager::Initialize();
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 1: Создать НОВЫЙ буфер и работать с ним
// ════════════════════════════════════════════════════════════════════════════

inline void Example1_CreateNewBuffer() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ПРИМЕР 1: Создание нового GPU буфера\n"
              << std::string(70, '=') << "\n\n";

    try {
        // OK ВСЕ просто! Только num_elements и тип
        auto buffer = gpu::GPUMemoryManager::CreateBuffer(
            1024,  // элементов
            gpu::MemoryType::GPU_READ_WRITE
        );

        // Показать информацию
        buffer->PrintStats();

        // Подготовить данные на CPU
        std::vector<std::complex<float>> test_data(1024);
        for (size_t i = 0; i < 1024; ++i) {
            test_data[i] = std::complex<float>(i, i * 2);
        }

        // Записать на GPU
        buffer->WriteToGPU(test_data);

        // Прочитать обратно
        auto readback = buffer->ReadFromGPU();

        // Проверить
        std::cout << "\nOK First 5 elements:\n";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << "  [" << i << "] = " << readback[i].real()
                      << " + " << readback[i].imag() << "j\n";
        }

        gpu::GPUMemoryManager::PrintStatistics();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 2: Читать данные от GeneratorGPU (ГЛАВНЫЙ USE CASE)
// ════════════════════════════════════════════════════════════════════════════

inline void Example2_ReadFromGenerator() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ПРИМЕР 2: Чтение данных от GeneratorGPU (ГЛАВНЫЙ СЛУЧАЙ)\n"
              << std::string(70, '=') << "\n\n";

    try {
        // 1. Создать генератор (имеет свои GPU буферы)
        LFMParameters params;
        params.f_start = 0.4e6;
        params.f_stop = 0.5e6;
        params.sample_rate = 12e6;
        params.num_beams = 256;
//        params.duration = 1e-3;  // 1 ms
        params.count_points = 1024*8;  // 
        auto xx = params.IsValid();

        auto gen_gpu = std::make_shared<radar::GeneratorGPU>(params);

        // 2. Генератор создаёт сигнал и возвращает cl_mem
        cl_mem signal_gpu = gen_gpu->signal_base();

        // 3. OK НОВЫЙ СПОСОБ: обернуть буфер через менеджер
        //    БЕЗ передачи context, queue - всё берётся из менеджера!
        auto reader = gpu::GPUMemoryManager::WrapExternalBuffer(
            signal_gpu,
            gen_gpu->GetTotalSize(),
            gpu::MemoryType::GPU_WRITE_ONLY
        );

        // 4. Показать что получили
        reader->PrintStats();

        // 5. Прочитать данные (читаем ИЗ signal_gpu генератора!)
        std::cout << "\nRead from GeneratorGPU signal...\n";
        auto partial = reader->ReadPartial(10);

        std::cout << "OK First 10 samples from GeneratorGPU:\n";
        for (size_t i = 0; i < partial.size(); ++i) {
            std::cout << "  [" << i << "] = " << partial[i].real()
                      << " + " << partial[i].imag() << "j\n";
        }

        // 6. reader уничтожится, но signal_gpu остаётся живым (управляется gen_gpu)
        // OK Это правильно! Non-owning буфер.

        gpu::GPUMemoryManager::PrintStatistics();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 3: Множество буферов через менеджер
// ════════════════════════════════════════════════════════════════════════════

inline void Example3_MultipleBuffers() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ПРИМЕР 3: Работа с несколькими буферами\n"
              << std::string(70, '=') << "\n\n";

    try {
        std::vector<std::unique_ptr<gpu::GPUMemoryBuffer>> buffers;

        // Создать 3 буфера разного размера
        size_t sizes[] = {512, 1024, 2048};
        for (size_t size : sizes) {
            auto buf = gpu::GPUMemoryManager::CreateBuffer(
                size,
                gpu::MemoryType::GPU_READ_WRITE
            );
            buffers.push_back(std::move(buf));
            std::cout << "Created buffer with " << size << " elements\n";
        }

        std::cout << "\n";
        for (size_t i = 0; i < buffers.size(); ++i) {
            std::cout << "Buffer " << i << ":\n";
            buffers[i]->PrintStats();
        }

        gpu::GPUMemoryManager::PrintStatistics();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ПОЛНЫЙ ПРИМЕР: main()
// ════════════════════════════════════════════════════════════════════════════

void RunAllExamples() {
    try {
        // Инициализация
        InitializeGPU();

        // Примеры
        Example1_CreateNewBuffer();
        Example2_ReadFromGenerator();
        Example3_MultipleBuffers();

        std::cout << "\n" << std::string(70, '=') << "\n"
                  << "OK ALL EXAMPLES COMPLETED SUCCESSFULLY\n"
                  << std::string(70, '=') << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR FATAL: " << e.what() << "\n";
    }
}

} // namespace examples

// ════════════════════════════════════════════════════════════════════════════
// ТИПИЧНОЕ ИСПОЛЬЗОВАНИЕ В ВАШЕМ КОДЕ:
// ════════════════════════════════════════════════════════════════════════════

/*

// main.cpp или initialization
int main() {
    // 1. ОДИН РАЗ инициализировать
    gpu::OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
    gpu::GPUMemoryManager::Initialize();

    // 2. Использовать везде
    auto gen = std::make_shared<GeneratorGPU>(params);
    cl_mem signal = gen->signal_base();

    // 3. ПРОСТО обернуть буфер (никаких context/queue!)
    auto reader = gpu::GPUMemoryManager::WrapExternalBuffer(
        signal,
        num_elements,
        gpu::MemoryType::GPU_WRITE_ONLY
    );

    // 4. Работать
    auto data = reader->ReadFromGPU();
    // ...

    return 0;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧТО ИЗМЕНИЛОСЬ:
// ════════════════════════════════════════════════════════════════════════════
//
// БЫЛО:
//   auto buffer = std::make_unique<gpu::GPUMemoryBuffer>(
//       gen_gpu_->GetContext(),      // <- передаём context
//       gen_gpu_->GetQueue(),        // <- передаём queue
//       signal_gpu,                  // <- внешний буфер
//       num_elements,                // <- передаём количество
//       gpu::MemoryType::GPU_WRITE_ONLY  // <- передаём тип
//   );
//
// СТАЛО:
//   auto buffer = gpu::GPUMemoryManager::WrapExternalBuffer(
//       signal_gpu,                  // <- только нужное!
//       num_elements,
//       gpu::MemoryType::GPU_WRITE_ONLY
//   );
//
// OK Context и queue берутся из синглтонов автоматически
// OK Одна инициализация - везде доступно
// OK Код читается понятнее
// OK Нет дублирования параметров
// OK OOP на уровне seniors

*/
