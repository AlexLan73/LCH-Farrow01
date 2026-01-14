// ════════════════════════════════════════════════════════════════════════════
// examples_usage.hpp - Примеры использования OpenCLManager с кэшированием
// ════════════════════════════════════════════════════════════════════════════
//
// Демонстрирует:
// 1. Использование реестра буферов для переиспользования
// 2. Очистку expired буферов
// 3. Кэширование kernels (автоматически через GeneratorGPU)
// 4. Оптимизацию для долгоживущих программ
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
#include <thread>
#include <chrono>

namespace examples_usage {

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 1: Переиспользование буферов в цикле (для долгоживущих программ)
// ════════════════════════════════════════════════════════════════════════════

inline void Example1_BufferReuseInLoop() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ПРИМЕР 1: Переиспользование буферов в цикле\n"
              << "         (оптимизация для долгоживущих программ)\n"
              << std::string(70, '=') << "\n\n";

    try {
        auto& manager = gpu::OpenCLManager::GetInstance();

        // Создать буфер ОДИН РАЗ и зарегистрировать
        std::cout << "Создание и регистрация рабочего буфера...\n";
        auto work_buffer = manager.GetOrCreateBuffer(
            "work_buffer",
            1024,
            gpu::MemoryType::GPU_READ_WRITE
        );
        std::cout << "✅ Буфер создан и зарегистрирован\n";

        // Симуляция долгоживущего цикла (1000 итераций)
        std::cout << "\nЗапуск цикла (1000 итераций)...\n";
        for (int i = 0; i < 1000; ++i) {
            // Получить тот же буфер (не создаем новый!)
            auto buffer = manager.GetBuffer("work_buffer");
            
            if (!buffer) {
                // Если буфер истек, создать новый
                buffer = manager.GetOrCreateBuffer("work_buffer", 1024, gpu::MemoryType::GPU_READ_WRITE);
            }

            // Работа с буфером
            std::vector<std::complex<float>> data(1024);
            for (size_t j = 0; j < 1024; ++j) {
                data[j] = std::complex<float>(i + j, (i + j) * 2);
            }
            buffer->WriteToGPU(data);
            
            // Периодически очищать expired буферы (каждые 100 итераций)
            if (i % 100 == 0) {
                manager.CleanupExpiredBuffers();
            }
        }

        std::cout << "✅ Цикл завершен. Буфер переиспользован 1000 раз!\n";
        manager.PrintMemoryStatistics();

    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << "\n";
        throw;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 2: Кэширование kernels через GeneratorGPU
// ════════════════════════════════════════════════════════════════════════════

inline void Example2_KernelCaching() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ПРИМЕР 2: Кэширование kernels через GeneratorGPU\n"
              << "         (kernels компилируются один раз, переиспользуются)\n"
              << std::string(70, '=') << "\n\n";

    try {
        auto& manager = gpu::OpenCLManager::GetInstance();

        // Параметры для генератора
        LFMParameters params;
        params.f_start = 0.4e6f;
        params.f_stop = 0.5e6f;
        params.sample_rate = 12e6f;
        params.num_beams = 256;
        params.count_points = 1024 * 8;

        std::cout << "Создание первого GeneratorGPU...\n";
        auto gen1 = std::make_shared<radar::GeneratorGPU>(params);
        std::cout << "✅ GeneratorGPU #1 создан (kernels скомпилированы)\n";

        // Получить статистику кэша
        std::cout << "\nСтатистика кэша kernels:\n";
        std::cout << manager.GetKernelCacheStatistics();

        std::cout << "\nСоздание второго GeneratorGPU (с теми же параметрами)...\n";
        auto gen2 = std::make_shared<radar::GeneratorGPU>(params);
        std::cout << "✅ GeneratorGPU #2 создан (kernels из кэша!)\n";

        // Статистика должна показать cache hit
        std::cout << "\nСтатистика кэша kernels после второго генератора:\n";
        std::cout << manager.GetKernelCacheStatistics();

        // Использовать генераторы
        std::cout << "\nГенерация сигналов...\n";
        cl_mem signal1 = gen1->signal_base();
        cl_mem signal2 = gen2->signal_base();
        std::cout << "✅ Сигналы сгенерированы\n";

        // Обернуть для чтения
        auto reader1 = manager.WrapExternalBuffer(
            signal1,
            gen1->GetTotalSize(),
            gpu::MemoryType::GPU_WRITE_ONLY
        );
        auto reader2 = manager.WrapExternalBuffer(
            signal2,
            gen2->GetTotalSize(),
            gpu::MemoryType::GPU_WRITE_ONLY
        );

        // Прочитать данные
        auto data1 = reader1->ReadPartial(10);
        auto data2 = reader2->ReadPartial(10);

        std::cout << "\nПервые 5 элементов из генератора #1:\n";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << "  [" << i << "] = " << std::fixed << std::setprecision(4)
                      << data1[i].real() << " + " << data1[i].imag() << "j\n";
        }

        std::cout << "\nПервые 5 элементов из генератора #2:\n";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << "  [" << i << "] = " << std::fixed << std::setprecision(4)
                      << data2[i].real() << " + " << data2[i].imag() << "j\n";
        }

        // Финальная статистика
        std::cout << "\nФинальная статистика:\n";
        std::cout << manager.GetCacheStatistics();
        std::cout << manager.GetKernelCacheStatistics();
        manager.PrintMemoryStatistics();

    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << "\n";
        throw;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 3: Долгоживущая программа с периодической очисткой
// ════════════════════════════════════════════════════════════════════════════

inline void Example3_LongRunningProgram() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ПРИМЕР 3: Симуляция долгоживущей программы\n"
              << "         (с периодической очисткой expired буферов)\n"
              << std::string(70, '=') << "\n\n";

    try {
        auto& manager = gpu::OpenCLManager::GetInstance();

        // Создать несколько буферов для разных задач
        std::vector<std::string> buffer_names = {
            "signal_base",
            "signal_delayed",
            "work_buffer_1",
            "work_buffer_2"
        };

        std::cout << "Создание рабочих буферов...\n";
        for (const auto& name : buffer_names) {
            auto buffer = manager.GetOrCreateBuffer(name, 1024, gpu::MemoryType::GPU_READ_WRITE);
            std::cout << "  ✅ Создан буфер: " << name << "\n";
        }

        // Симуляция долгоживущего цикла
        const int total_iterations = 100;
        const int cleanup_interval = 20;  // Очистка каждые 20 итераций

        std::cout << "\nЗапуск долгоживущего цикла (" << total_iterations << " итераций)...\n";
        std::cout << "Очистка expired буферов каждые " << cleanup_interval << " итераций\n\n";

        for (int i = 0; i < total_iterations; ++i) {
            // Работа с буферами
            for (const auto& name : buffer_names) {
                auto buffer = manager.GetBuffer(name);
                if (buffer) {
                    // Работа с буфером
                    std::vector<std::complex<float>> data(1024);
                    buffer->WriteToGPU(data);
                }
            }

            // Периодическая очистка
            if (i > 0 && i % cleanup_interval == 0) {
                std::cout << "  [Итерация " << i << "] Очистка expired буферов...\n";
                manager.CleanupExpiredBuffers();
                manager.PrintMemoryStatistics();
            }
        }

        std::cout << "\n✅ Долгоживущий цикл завершен\n";
        std::cout << "\nФинальная статистика:\n";
        manager.PrintMemoryStatistics();
        std::cout << manager.GetCacheStatistics();
        std::cout << manager.GetKernelCacheStatistics();

    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << "\n";
        throw;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 4: Оптимизация для множественных расчетов
// ════════════════════════════════════════════════════════════════════════════

inline void Example4_MultipleCalculations() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ПРИМЕР 4: Множественные расчеты с переиспользованием\n"
              << std::string(70, '=') << "\n\n";

    try {
        auto& manager = gpu::OpenCLManager::GetInstance();

        // Параметры
        LFMParameters params;
        params.f_start = 0.4e6f;
        params.f_stop = 0.5e6f;
        params.sample_rate = 12e6f;
        params.num_beams = 256;
        params.count_points = 1024 * 8;

        // Расчет 1: создать генератор и сохранить буфер
        std::cout << "Расчет 1: Создание генератора и генерация сигнала...\n";
        auto gen1 = std::make_shared<radar::GeneratorGPU>(params);
        cl_mem signal1 = gen1->signal_base();
        
        // Сохранить буфер для последующего использования
        auto reader1 = manager.WrapExternalBuffer(
            signal1,
            gen1->GetTotalSize(),
            gpu::MemoryType::GPU_WRITE_ONLY
        );
        manager.RegisterBuffer("calculation_1_result", 
            std::shared_ptr<gpu::GPUMemoryBuffer>(reader1.release()));
        std::cout << "✅ Результат расчета 1 сохранен в реестре\n";

        // Расчет 2: использовать тот же генератор (kernels из кэша!)
        std::cout << "\nРасчет 2: Создание второго генератора (kernels из кэша)...\n";
        auto gen2 = std::make_shared<radar::GeneratorGPU>(params);
        cl_mem signal2 = gen2->signal_base();
        std::cout << "✅ Генератор #2 создан (kernels переиспользованы из кэша)\n";

        // Расчет 3: получить результат расчета 1 из реестра
        std::cout << "\nРасчет 3: Получение результата расчета 1 из реестра...\n";
        auto cached_result = manager.GetBuffer("calculation_1_result");
        if (cached_result) {
            std::cout << "✅ Результат расчета 1 получен из реестра\n";
            auto data = cached_result->ReadPartial(10);
            std::cout << "Первые 5 элементов:\n";
            for (size_t i = 0; i < 5; ++i) {
                std::cout << "  [" << i << "] = " << std::fixed << std::setprecision(4)
                          << data[i].real() << " + " << data[i].imag() << "j\n";
            }
        } else {
            std::cout << "⚠️  Результат расчета 1 не найден (возможно истек)\n";
        }

        // Статистика
        std::cout << "\nСтатистика кэширования:\n";
        std::cout << manager.GetCacheStatistics();
        std::cout << manager.GetKernelCacheStatistics();
        manager.PrintMemoryStatistics();

    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << "\n";
        throw;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 5: Работа с большим количеством kernels
// ════════════════════════════════════════════════════════════════════════════

inline void Example5_ManyKernels() {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "ПРИМЕР 5: Работа с большим количеством kernels\n"
              << "         (демонстрация масштабирования и очистки кэша)\n"
              << std::string(70, '=') << "\n\n";

    try {
        auto& manager = gpu::OpenCLManager::GetInstance();

        // Создать несколько программ с разными kernels
        std::vector<std::pair<std::string, std::string>> kernel_groups = {
            {"signal_group", R"(
                __kernel void generate(__global float2* out) { 
                    int id = get_global_id(0);
                    out[id] = (float2)(1.0f, 0.0f);
                }
                __kernel void modulate(__global float2* data) {
                    int id = get_global_id(0);
                    data[id].x *= 2.0f;
                }
                __kernel void filter(__global float2* data) {
                    int id = get_global_id(0);
                    data[id].y *= 0.5f;
                }
            )"},
            {"math_group", R"(
                __kernel void multiply(__global float* a, __global float* b, __global float* out) {
                    int id = get_global_id(0);
                    out[id] = a[id] * b[id];
                }
                __kernel void add(__global float* a, __global float* b, __global float* out) {
                    int id = get_global_id(0);
                    out[id] = a[id] + b[id];
                }
                __kernel void subtract(__global float* a, __global float* b, __global float* out) {
                    int id = get_global_id(0);
                    out[id] = a[id] - b[id];
                }
            )"},
            {"transform_group", R"(
                __kernel void fft(__global float2* data) {
                    int id = get_global_id(0);
                    // Simplified FFT placeholder
                    float temp = data[id].x;
                    data[id].x = data[id].y;
                    data[id].y = temp;
                }
                __kernel void ifft(__global float2* data) {
                    int id = get_global_id(0);
                    // Simplified IFFT placeholder
                    float temp = data[id].x;
                    data[id].x = data[id].y;
                    data[id].y = temp;
                }
            )"}
        };

        std::vector<cl_program> programs;
        std::vector<std::vector<cl_kernel>> all_kernels;

        // Создать все programs и kernels
        std::cout << "Создание kernel groups...\n";
        for (const auto& [group_name, source] : kernel_groups) {
            std::cout << "  Группа: " << group_name << "\n";
            cl_program program = manager.GetOrCompileProgram(source);
            programs.push_back(program);

            std::vector<cl_kernel> kernels;
            if (group_name == "signal_group") {
                kernels.push_back(manager.GetOrCreateKernel(program, "generate"));
                kernels.push_back(manager.GetOrCreateKernel(program, "modulate"));
                kernels.push_back(manager.GetOrCreateKernel(program, "filter"));
            } else if (group_name == "math_group") {
                kernels.push_back(manager.GetOrCreateKernel(program, "multiply"));
                kernels.push_back(manager.GetOrCreateKernel(program, "add"));
                kernels.push_back(manager.GetOrCreateKernel(program, "subtract"));
            } else if (group_name == "transform_group") {
                kernels.push_back(manager.GetOrCreateKernel(program, "fft"));
                kernels.push_back(manager.GetOrCreateKernel(program, "ifft"));
            }
            all_kernels.push_back(kernels);
            std::cout << "    ✅ Создано " << kernels.size() << " kernels\n";
        }

        // Статистика после создания
        std::cout << "\nСтатистика после создания всех kernels:\n";
        std::cout << manager.GetKernelCacheStatistics();
        size_t initial_size = manager.GetKernelCacheSize();
        std::cout << "  Всего kernels в кэше: " << initial_size << "\n";

        // Повторное использование kernels (cache hits)
        std::cout << "\nПовторное использование kernels (cache hits)...\n";
        for (size_t i = 0; i < programs.size(); ++i) {
            cl_program program = programs[i];
            if (i == 0) {
                // Повторно получить kernels из signal_group
                cl_kernel k1 = manager.GetOrCreateKernel(program, "generate");
                cl_kernel k2 = manager.GetOrCreateKernel(program, "modulate");
                cl_kernel k3 = manager.GetOrCreateKernel(program, "filter");
                std::cout << "  ✅ Повторно получены kernels из signal_group (из кэша!)\n";
            }
        }

        std::cout << "\nСтатистика после повторного использования:\n";
        std::cout << manager.GetKernelCacheStatistics();

        // Очистка kernels конкретной группы
        std::cout << "\nОчистка kernels группы 'signal_group'...\n";
        manager.ClearKernelsForProgram(programs[0]);
        size_t after_clear = manager.GetKernelCacheSize();
        std::cout << "  Kernels в кэше после очистки: " << after_clear << "\n";
        std::cout << "  Удалено: " << (initial_size - after_clear) << " kernels\n";

        // Полная очистка
        std::cout << "\nПолная очистка kernel cache...\n";
        manager.ClearKernelCache();
        size_t final_size = manager.GetKernelCacheSize();
        std::cout << "  Kernels в кэше после полной очистки: " << final_size << "\n";

        // Демонстрация автоматического пересоздания
        std::cout << "\nАвтоматическое пересоздание kernels...\n";
        cl_program first_program = programs[0];
        cl_kernel regenerated = manager.GetOrCreateKernel(first_program, "generate");
        std::cout << "  ✅ Kernel 'generate' пересоздан автоматически\n";
        std::cout << "  Размер кэша: " << manager.GetKernelCacheSize() << "\n";

        // Финальная статистика
        std::cout << "\nФинальная статистика:\n";
        std::cout << manager.GetKernelCacheStatistics();
        std::cout << manager.GetCacheStatistics();

        std::cout << "\n✅ Пример работы с большим количеством kernels завершен\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << "\n";
        throw;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ГЛАВНАЯ ФУНКЦИЯ: Запустить все примеры
// ════════════════════════════════════════════════════════════════════════════

inline void RunAllExamples() {
    try {
        // Инициализация
        std::cout << "\n" << std::string(70, '=') << "\n"
                  << "ИНИЦИАЛИЗАЦИЯ OpenCLManager\n"
                  << std::string(70, '=') << "\n\n";

        gpu::OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
        std::cout << gpu::OpenCLManager::GetInstance().GetDeviceInfo();

        // Примеры
        Example1_BufferReuseInLoop();
        Example2_KernelCaching();
        Example3_LongRunningProgram();
        Example4_MultipleCalculations();
        Example5_ManyKernels();

        std::cout << "\n" << std::string(70, '=') << "\n"
                  << "✅ ВСЕ ПРИМЕРЫ УСПЕШНО ЗАВЕРШЕНЫ\n"
                  << std::string(70, '=') << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n" << std::string(70, '=') << "\n"
                  << "❌ КРИТИЧЕСКАЯ ОШИБКА: " << e.what() << "\n"
                  << std::string(70, '=') << "\n\n";
        throw;
    }
}

} // namespace examples_usage

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР ИСПОЛЬЗОВАНИЯ В main():
// ════════════════════════════════════════════════════════════════════════════

/*
#include "Doc/MemRAII/examples_usage.hpp"

int main() {
    try {
        examples_usage::RunAllExamples();
        return 0;
    } catch (...) {
        return 1;
    }
}
*/

