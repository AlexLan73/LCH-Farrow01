// ════════════════════════════════════════════════════════════════════════════
// test_unified_manager.hpp - Тест унифицированного OpenCLManager
// ════════════════════════════════════════════════════════════════════════════
//
// Тестирует:
// 1. Создание буферов через OpenCLManager
// 2. Обертку внешних буферов от GeneratorGPU
// 3. Валидацию context (защита от ошибки -34)
// 4. Реестр буферов для переиспользования
//
// ════════════════════════════════════════════════════════════════════════════

#pragma once

#include "GPU/opencl_manager.h"
#include "GPU/gpu_memory_manager.hpp"
#include "generator/generator_gpu.h"
#include "interface/lfm_parameters.h"

#include <memory>
#include <iostream>
#include <vector>
#include <iomanip>

namespace test_unified {

// ════════════════════════════════════════════════════════════════════════════
// ИНИЦИАЛИЗАЦИЯ (один раз в main)
// ════════════════════════════════════════════════════════════════════════════

inline void InitializeGPU() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ИНИЦИАЛИЗАЦИЯ OpenCLManager\n"
              << std::string(70, '=') << "\n\n";

    // 1. Инициализировать OpenCL (ОДИН РАЗ!)
    gpu::OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
    std::cout << gpu::OpenCLManager::GetInstance().GetDeviceInfo();
    
    std::cout << "\n✅ OpenCLManager инициализирован\n";
    std::cout << "✅ Теперь доступны методы управления памятью:\n";
    std::cout << "   - CreateBuffer()\n";
    std::cout << "   - WrapExternalBuffer()\n";
    std::cout << "   - RegisterBuffer() / GetBuffer()\n";
}

// ════════════════════════════════════════════════════════════════════════════
// ТЕСТ 1: Создание нового буфера через OpenCLManager
// ════════════════════════════════════════════════════════════════════════════

inline void Test1_CreateBuffer() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ТЕСТ 1: Создание нового GPU буфера через OpenCLManager\n"
              << std::string(70, '=') << "\n\n";

    try {
        auto& manager = gpu::OpenCLManager::GetInstance();

        // Создать буфер через OpenCLManager
        auto buffer = manager.CreateBuffer(
            1024,  // элементов
            gpu::MemoryType::GPU_READ_WRITE
        );

        std::cout << "✅ Буфер создан через OpenCLManager::CreateBuffer()\n";
        buffer->PrintStats();

        // Подготовить тестовые данные
        std::vector<std::complex<float>> test_data(1024);
        for (size_t i = 0; i < 1024; ++i) {
            test_data[i] = std::complex<float>(i, i * 2);
        }

        // Записать на GPU
        buffer->WriteToGPU(test_data);
        std::cout << "✅ Данные записаны на GPU\n";

        // Прочитать обратно
        auto readback = buffer->ReadFromGPU();
        std::cout << "✅ Данные прочитаны с GPU\n";

        // Проверить первые 5 элементов
        std::cout << "\nПервые 5 элементов:\n";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << "  [" << i << "] = " << readback[i].real()
                      << " + " << readback[i].imag() << "j\n";
        }

        manager.PrintMemoryStatistics();

    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << "\n";
        throw;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ТЕСТ 2: Обертка внешнего буфера от GeneratorGPU (ГЛАВНЫЙ ТЕСТ!)
// ════════════════════════════════════════════════════════════════════════════

inline void Test2_WrapGeneratorBuffer() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ТЕСТ 2: Обертка буфера от GeneratorGPU\n"
              << "        (проверка валидации context - защита от ошибки -34)\n"
              << std::string(70, '=') << "\n\n";

    try {
        // 1. Создать генератор (имеет свои GPU буферы)
        LFMParameters params;
        params.f_start = 0.4e6f;
        params.f_stop = 0.5e6f;
        params.sample_rate = 12e6f;
        params.num_beams = 256;
        params.count_points = 1024 * 8;

        if (!params.IsValid()) {
            throw std::runtime_error("Invalid LFMParameters");
        }

        std::cout << "Создание GeneratorGPU...\n";
        auto gen_gpu = std::make_shared<radar::GeneratorGPU>(params);
        std::cout << "✅ GeneratorGPU создан\n";

        // 2. Генератор создаёт сигнал и возвращает cl_mem
        std::cout << "\nГенерация базового сигнала...\n";
        cl_mem signal_gpu = gen_gpu->signal_base();
        std::cout << "✅ Сигнал сгенерирован на GPU\n";

        // 3. НОВЫЙ СПОСОБ: обернуть буфер через OpenCLManager
        //    С АВТОМАТИЧЕСКОЙ ВАЛИДАЦИЕЙ CONTEXT!
        std::cout << "\nОбертка буфера через OpenCLManager::WrapExternalBuffer()...\n";
        auto& manager = gpu::OpenCLManager::GetInstance();
        
        auto reader = manager.WrapExternalBuffer(
            signal_gpu,
            gen_gpu->GetTotalSize(),
            gpu::MemoryType::GPU_WRITE_ONLY
        );

        std::cout << "✅ Буфер обернут (context проверен автоматически)\n";
        reader->PrintStats();

        // 4. Прочитать данные (читаем ИЗ signal_gpu генератора!)
        std::cout << "\nЧтение данных из буфера генератора...\n";
        auto partial = reader->ReadPartial(10);

        std::cout << "✅ Первые 10 отсчётов из GeneratorGPU:\n";
        for (size_t i = 0; i < partial.size(); ++i) {
            std::cout << "  [" << i << "] = " << std::fixed << std::setprecision(4)
                      << partial[i].real() << " + " << partial[i].imag() << "j\n";
        }

        // 5. reader уничтожится, но signal_gpu остаётся живым (управляется gen_gpu)
        //    Это правильно! Non-owning буфер.

        manager.PrintMemoryStatistics();

    } catch (const std::runtime_error& e) {
        std::cerr << "\n❌ ОШИБКА: " << e.what() << "\n";
        
        // Проверить, это ли ошибка валидации context
        if (e.what() && std::string(e.what()).find("different context") != std::string::npos) {
            std::cerr << "\n⚠️  ВНИМАНИЕ: Обнаружено несовпадение context!\n";
            std::cerr << "   Это означает, что GeneratorGPU создает свой context,\n";
            std::cerr << "   а не использует OpenCLManager.\n";
            std::cerr << "   Решение: GeneratorGPU должен использовать context из OpenCLManager.\n";
        }
        throw;
    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << "\n";
        throw;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ТЕСТ 3: Реестр буферов для переиспользования
// ════════════════════════════════════════════════════════════════════════════

inline void Test3_BufferRegistry() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ТЕСТ 3: Реестр буферов для переиспользования\n"
              << std::string(70, '=') << "\n\n";

    try {
        auto& manager = gpu::OpenCLManager::GetInstance();

        // 1. Создать буфер и зарегистрировать
        std::cout << "Создание и регистрация буфера 'signal_base'...\n";
        auto signal1 = manager.CreateBuffer(1024, gpu::MemoryType::GPU_READ_WRITE);
        
        // Заполнить тестовыми данными
        std::vector<std::complex<float>> test_data(1024);
        for (size_t i = 0; i < 1024; ++i) {
            test_data[i] = std::complex<float>(i, i * 2);
        }
        signal1->WriteToGPU(test_data);

        // Зарегистрировать
        manager.RegisterBuffer("signal_base", 
            std::shared_ptr<gpu::GPUMemoryBuffer>(signal1.release()));
        std::cout << "✅ Буфер зарегистрирован как 'signal_base'\n";

        // 2. Получить зарегистрированный буфер
        std::cout << "\nПолучение зарегистрированного буфера...\n";
        auto cached = manager.GetBuffer("signal_base");
        
        if (cached) {
            std::cout << "✅ Буфер получен из реестра\n";
            auto data = cached->ReadPartial(5);
            std::cout << "Первые 5 элементов:\n";
            for (size_t i = 0; i < 5; ++i) {
                std::cout << "  [" << i << "] = " << data[i].real()
                          << " + " << data[i].imag() << "j\n";
            }
        } else {
            std::cout << "❌ Буфер не найден или истек\n";
        }

        // 3. Использовать GetOrCreateBuffer
        std::cout << "\nИспользование GetOrCreateBuffer('temp_buffer')...\n";
        auto temp = manager.GetOrCreateBuffer("temp_buffer", 512, gpu::MemoryType::GPU_READ_WRITE);
        std::cout << "✅ Буфер создан/получен\n";
        temp->PrintStats();

        // Попробовать получить еще раз (должен вернуть существующий)
        auto temp2 = manager.GetOrCreateBuffer("temp_buffer", 512, gpu::MemoryType::GPU_READ_WRITE);
        std::cout << "✅ Тот же буфер получен повторно\n";

        manager.PrintMemoryStatistics();

    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << "\n";
        throw;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ТЕСТ 4: Множественные буферы
// ════════════════════════════════════════════════════════════════════════════

inline void Test4_MultipleBuffers() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ТЕСТ 4: Работа с несколькими буферами\n"
              << std::string(70, '=') << "\n\n";

    try {
        auto& manager = gpu::OpenCLManager::GetInstance();

        std::vector<std::unique_ptr<gpu::GPUMemoryBuffer>> buffers;

        // Создать 3 буфера разного размера
        size_t sizes[] = {512, 1024, 2048};
        for (size_t size : sizes) {
            auto buf = manager.CreateBuffer(
                size,
                gpu::MemoryType::GPU_READ_WRITE
            );
            buffers.push_back(std::move(buf));
            std::cout << "✅ Создан буфер с " << size << " элементами\n";
        }

        std::cout << "\nИнформация о буферах:\n";
        for (size_t i = 0; i < buffers.size(); ++i) {
            std::cout << "\nБуфер " << i << ":\n";
            buffers[i]->PrintStats();
        }

        manager.PrintMemoryStatistics();

    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << "\n";
        throw;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ГЛАВНАЯ ФУНКЦИЯ: Запустить все тесты
// ════════════════════════════════════════════════════════════════════════════

inline void RunAllTests() {
    try {
        // Инициализация
        InitializeGPU();

        // Тесты
        Test1_CreateBuffer();
        Test2_WrapGeneratorBuffer();  // ГЛАВНЫЙ ТЕСТ!
        Test3_BufferRegistry();
        Test4_MultipleBuffers();

        std::cout << "\n" << std::string(70, '=') << "\n"
                  << "✅ ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ\n"
                  << std::string(70, '=') << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n" << std::string(70, '=') << "\n"
                  << "❌ КРИТИЧЕСКАЯ ОШИБКА: " << e.what() << "\n"
                  << std::string(70, '=') << "\n\n";
        throw;
    }
}

} // namespace test_unified

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР ИСПОЛЬЗОВАНИЯ В main():
// ════════════════════════════════════════════════════════════════════════════

/*
#include "Doc/MemRAII/test_unified_manager.hpp"

int main() {
    try {
        test_unified::RunAllTests();
        return 0;
    } catch (...) {
        return 1;
    }
}
*/

