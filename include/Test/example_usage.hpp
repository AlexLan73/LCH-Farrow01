
#include <iostream>
#include <vector>
#include <complex>

// Новая архитектура
#include "ManagerOpenCL/opencl_core.hpp"
#include "ManagerOpenCL/command_queue_pool.hpp"
#include "ManagerOpenCL/opencl_compute_engine.hpp"

// Генератор
#include "GPU/generator_gpu_new.h"
#include "interface/lfm_parameters.h"
#include "interface/DelayParameter.h"

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 1: Базовый генератор ЛЧМ сигнала
// ════════════════════════════════════════════════════════════════════════════

void example_basic_lfm() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ПРИМЕР 1: Базовый ЛЧМ сигнал" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    try {
        // ✅ ШАГ 1: Инициализация архитектуры (один раз в main)
        std::cout << "Step 1: Initializing OpenCL infrastructure..." << std::endl;
        
        gpu::OpenCLCore::Initialize(gpu::DeviceType::GPU);
        gpu::CommandQueuePool::Initialize(4);  // 4 command queues
        gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
        
        std::cout << "✅ OpenCL infrastructure ready\n" << std::endl;

        // ✅ ШАГ 2: Создать параметры сигнала
        std::cout << "Step 2: Creating LFM parameters..." << std::endl;
        
        LFMParameters params;
        params.f_start = 1.00e6f;         // 100 kHz
        params.f_stop = 2.5000e6f;          // 500 kHz
        params.sample_rate = 12.0e6f;    // 12 MHz sampling rate
        params.num_beams = 256;          // 256 beams
        params.count_points = 1024 * 16;  // 8192 samples per beam
        params.angle_step_deg = 0.5f;    // 0.5° step
        params.SetAngle();               // Auto-set angle range
        
        std::cout << "Parameters:" << std::endl;
        std::cout << "  f_start = " << params.f_start / 1e6 << " MHz" << std::endl;
        std::cout << "  f_stop = " << params.f_stop / 1e6 << " MHz" << std::endl;
        std::cout << "  sample_rate = " << params.sample_rate / 1e6 << " MHz" << std::endl;
        std::cout << "  num_beams = " << params.num_beams << std::endl;
        std::cout << "  count_points = " << params.count_points << std::endl;
        std::cout << "  duration = " << params.duration * 1e6 << " µs\n" << std::endl;

        // ✅ ШАГ 3: Создать генератор
        std::cout << "Step 3: Creating GeneratorGPU..." << std::endl;
        
        radar::GeneratorGPU gen(params);
        
        std::cout << "✅ GeneratorGPU created\n" << std::endl;

        // ✅ ШАГ 4: Генерировать базовый сигнал
        std::cout << "Step 4: Generating signal_base()..." << std::endl;
        
        cl_mem signal_gpu = gen.signal_base();
        size_t total_size = gen.GetTotalSize();
        size_t memory_size = gen.GetMemorySizeBytes();
        
        std::cout << "Signal allocated on GPU:" << std::endl;
        std::cout << "  Total elements = " << total_size << std::endl;
        std::cout << "  Memory size = " << (memory_size / (1024*1024)) << " MB\n" << std::endl;

        // ✅ ШАГ 5: Синхронизировать GPU
        std::cout << "Step 5: Syncing GPU..." << std::endl;
        
        gen.ClearGPU();
        
        std::cout << "✅ GPU synced\n" << std::endl;

        // ✅ ШАГ 6: Прочитать результаты (опционально)
        std::cout << "Step 6: Reading results from GPU..." << std::endl;
        
        auto& engine = gpu::OpenCLComputeEngine::GetInstance();
        
        // Прочитать все данные (БОЛЬШОЙ ОБЪЁМ!)
        //std::vector<std::complex<float>> result = 
//             engine. ReadBufferFromGPU(signal_gpu, total_size);
             //engine.Rea(signal_gpu, total_size);
        
        // Прочитать только первый луч (256 samples)
//        size_t num_samples = gen.GetNumSamples();
        size_t num_samples = 10;
        std::vector<std::complex<float>> first_beam(num_samples);
        
        auto beam0 = gen.GetSignalAsVector(0);
         
        // TODO: Реализовать ReadPartial в OpenCLComputeEngine
        for (int i = 0; i < num_samples; i++) {
             first_beam[i] = beam0[i];
         }
        
        std::cout << "✅ First beam data (would be read from GPU)" << std::endl;
        std::cout << "  Beam 0, first 5 samples: (would show values)\n" << std::endl;

        std::cout << "✅ EXAMPLE 1 COMPLETED SUCCESSFULLY\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR: " << e.what() << std::endl;
        return;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 2: ЛЧМ сигнал с дробной задержкой
// ════════════════════════════════════════════════════════════════════════════

void example_delayed_lfm() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ПРИМЕР 2: ЛЧМ сигнал с дробной задержкой" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    try {
        // ✅ Инициализация (если ещё не инициализирована)
        if (!gpu::OpenCLComputeEngine::IsInitialized()) {
            gpu::OpenCLCore::Initialize(gpu::DeviceType::GPU);
            gpu::CommandQueuePool::Initialize(4);
            gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
        }

        // ✅ Параметры
        LFMParameters params;
        params.f_start = 0.400e6f;         // 100 kHz
        params.f_stop = 0.500e6f;          // 500 kHz
        params.sample_rate = 12.0e6f;
        params.num_beams = 256;
        params.count_points = 1024 * 8;
        params.angle_step_deg = 0.5f;    // 0.5° step
        params.SetAngle();

        std::cout << "Creating GeneratorGPU with delay parameters..." << std::endl;
        radar::GeneratorGPU gen(params);

        // ✅ Создать параметры задержки
        std::cout << "Creating delay parameters..." << std::endl;
        
        std::vector<DelayParameter> delays(params.num_beams);
        
        // Задержки меняются линейно от -64° до +64°
        float angle_start = params.angle_start_deg;
        float angle_stop = params.angle_stop_deg;
        float angle_range = angle_stop - angle_start;
        
        for (size_t i = 0; i < params.num_beams; i++) {
            delays[i].beam_index = i;
            delays[i].delay_degrees = angle_start + 
                (angle_range * static_cast<float>(i) / (params.num_beams - 1));
        }
        
        std::cout << "Delay parameters:" << std::endl;
        std::cout << "  Beam 0:   " << delays[0].delay_degrees << "°" << std::endl;
        std::cout << "  Beam 128: " << delays[128].delay_degrees << "°" << std::endl;
        std::cout << "  Beam 255: " << delays[255].delay_degrees << "°\n" << std::endl;

        // ✅ Генерировать сигнал с задержкой
        std::cout << "Generating signal_valedation()..." << std::endl;
        
        cl_mem signal_delayed_gpu = gen.signal_valedation(
            delays.data(),
            delays.size()
        );
        
        std::cout << "✅ Signal with delays generated on GPU" << std::endl;
        std::cout << "  Memory size = " << (gen.GetMemorySizeBytes() / (1024*1024)) 
                  << " MB\n" << std::endl;

        // ✅ Синхронизация
//        gen.ClearGPU();

        std::cout << "✅ EXAMPLE 2 COMPLETED SUCCESSFULLY\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR: " << e.what() << std::endl;
        return;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 3: Несколько генераторов одновременно (асинхронность)
// ════════════════════════════════════════════════════════════════════════════

void example_multiple_generators() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ПРИМЕР 3: Несколько генераторов (асинхронность)" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    try {
        // ✅ Инициализация
        if (!gpu::OpenCLComputeEngine::IsInitialized()) {
            gpu::OpenCLCore::Initialize(gpu::DeviceType::GPU);
            gpu::CommandQueuePool::Initialize(4);  // ← 4 очереди для асинхронности!
            gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
        }

        // ✅ Создать несколько генераторов с разными параметрами
        std::cout << "Creating multiple generators..." << std::endl;
        
        std::vector<radar::GeneratorGPU> generators;
        
        // Генератор 1: 100-500 MHz
        LFMParameters params1;
        params1.f_start = 100.0f;
        params1.f_stop = 500.0f;
        params1.sample_rate = 12.0e6f;
        params1.num_beams = 256;
        params1.count_points = 1024 * 8;
        generators.emplace_back(params1);
        std::cout << "✓ Generator 1 created (100-500 MHz)" << std::endl;
        
        // Генератор 2: 1-5 GHz
        LFMParameters params2;
        params2.f_start = 1.0e9f;
        params2.f_stop = 5.0e9f;
        params2.sample_rate = 12.0e9f;
        params2.num_beams = 128;
        params2.count_points = 1024 * 4;
        generators.emplace_back(params2);
        std::cout << "✓ Generator 2 created (1-5 GHz)" << std::endl;

        // ✅ Генерировать сигналы ПАРАЛЛЕЛЬНО (в разных очередях!)
        std::cout << "\nGenerating signals ASYNCHRONOUSLY..." << std::endl;
        
        std::vector<cl_mem> signals;
        for (size_t i = 0; i < generators.size(); i++) {
            cl_mem sig = generators[i].signal_base();
            signals.push_back(sig);
            std::cout << "✓ Signal " << (i+1) << " generated (in queue " 
                      << (i % 4) << ")" << std::endl;
        }

        // ✅ Синхронизация всех
        std::cout << "\nWaiting for all operations to complete..." << std::endl;
        gpu::CommandQueuePool::FinishAll();
        
        std::cout << "✅ All signals completed\n" << std::endl;

        // Статистика
        auto& engine = gpu::OpenCLComputeEngine::GetInstance();
        std::cout << engine.GetStatistics();

        std::cout << "✅ EXAMPLE 3 COMPLETED SUCCESSFULLY\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR: " << e.what() << std::endl;
        return;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN - УДАЛЕНА из заголовочного файла (должна быть только в .cpp)
// Используйте функции example_basic_lfm(), example_delayed_lfm() и т.д.
// из вашего main() в main.cpp
// ════════════════════════════════════════════════════════════════════════════

/*
// Пример main() - используйте в main.cpp:
void run_all_examples() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        GeneratorGPU Examples (NEW ARCHITECTURE)                  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n" << std::endl;

    // Запустить примеры
    example_basic_lfm();
    example_delayed_lfm();
    example_multiple_generators();

    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    ALL EXAMPLES COMPLETED                        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n" << std::endl;
}
*/
