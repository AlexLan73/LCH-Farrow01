#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "fractional_delay_processor.hpp"
#include "opencl_core.hpp"
#include "command_queue_pool.hpp"
#include "opencl_compute_engine.hpp"
#include "generator_gpu_new.hpp"

using namespace radar;
using namespace gpu;

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

void PrintHeader(const std::string& text) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  " << text << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void PrintStep(int step, const std::string& text) {
    std::cout << "\n[Step " << step << "] " << text << "\n";
    std::cout << std::string(40, '-') << "\n";
}

// ============================================================================
// ГЛАВНАЯ ПРОГРАММА
// ============================================================================

int main() {
    try {
        PrintHeader("🚀 FRACTIONAL DELAY PROCESSOR - ПОЛНЫЙ ПРИМЕР");
        
        // ====================================================================
        // ЭТАП 1: Инициализация OpenCL Core
        // ====================================================================
        PrintStep(1, "Инициализация OpenCL Core");
        
        OpenCLCore::Initialize(DeviceType::GPU);
        std::cout << "✅ OpenCLCore инициализирован\n";
        
        // ====================================================================
        // ЭТАП 2: Инициализация Command Queue Pool
        // ====================================================================
        PrintStep(2, "Инициализация Command Queue Pool");
        
        CommandQueuePool::Initialize();
        std::cout << "✅ CommandQueuePool инициализирован\n";
        
        // ====================================================================
        // ЭТАП 3: Инициализация OpenCL Compute Engine
        // ====================================================================
        PrintStep(3, "Инициализация OpenCL Compute Engine");
        
        OpenCLComputeEngine::Initialize(DeviceType::GPU);
        auto& engine = OpenCLComputeEngine::GetInstance();
        std::cout << "✅ OpenCLComputeEngine инициализирован\n";
        
        // ====================================================================
        // ЭТАП 4: Конфигурация параметров
        // ====================================================================
        PrintStep(4, "Конфигурация параметров");
        
        // Конфигурация процессора дробной задержки
        auto config = FractionalDelayConfig::Diagnostic();
        config.num_beams = 64;        // Количество антенн/лучей
        config.num_samples = 1024;    // Количество отсчётов на луч
        config.verbose = true;        // Подробный вывод
        
        std::cout << "Configuration:\n";
        std::cout << "  - Beams:     " << config.num_beams << "\n";
        std::cout << "  - Samples:   " << config.num_samples << "\n";
        std::cout << "  - Verbose:   " << (config.verbose ? "Yes" : "No") << "\n";
        
        // Параметры LFM сигнала
        LFMParameters lfm;
        lfm.num_beams = config.num_beams;
        lfm.count_points = config.num_samples;
        lfm.f_start = 1.0e9f;         // 1 GHz
        lfm.f_stop = 2.0e9f;          // 2 GHz
        lfm.sample_rate = 4.0e9f;     // 4 GHz
        lfm.amplitude = 1.0f;
        
        std::cout << "\nLFM Parameters:\n";
        std::cout << "  - Beams:       " << lfm.num_beams << "\n";
        std::cout << "  - Points:      " << lfm.count_points << "\n";
        std::cout << "  - F_start:     " << (lfm.f_start / 1e9) << " GHz\n";
        std::cout << "  - F_stop:      " << (lfm.f_stop / 1e9) << " GHz\n";
        
        // ====================================================================
        // ЭТАП 5: Создание процессора дробной задержки
        // ====================================================================
        PrintStep(5, "Создание FractionalDelayProcessor");
        
        FractionalDelayProcessor processor(config, lfm);
        std::cout << "✅ Процессор создан и инициализирован\n";
        
        // ====================================================================
        // ЭТАП 6: Генерирование входных данных на GPU
        // ====================================================================
        PrintStep(6, "Генерирование LFM сигналов на GPU");
        
        // Используем GeneratorGPU::signal_base() для создания сигналов
        GeneratorGPU generator(lfm);
        auto gpu_buffer = generator.signal_base();  // Генерирует на GPU
        
        std::cout << "✅ LFM сигналы сгенерированы на GPU\n";
        std::cout << "   Размер: " << lfm.num_beams << " x " 
                  << lfm.count_points << " = " 
                  << (lfm.num_beams * lfm.count_points) << " точек\n";
        
        // ====================================================================
        // ЭТАП 7: Обработка с ОДНОЙ задержкой
        // ====================================================================
        PrintStep(7, "Обработка с дробной задержкой");
        
        // Применить задержку к лучу 0: 2.5 градуса
        DelayParameter delay;
        delay.beam_index = 0;
        delay.delay_degrees = 2.5f;
        
        std::cout << "Параметры:\n";
        std::cout << "  - Beam index: " << delay.beam_index << "\n";
        std::cout << "  - Delay:      " << delay.delay_degrees << "°\n";
        
        auto result = processor.ProcessWithFractionalDelay(delay);
        
        if (result.success) {
            std::cout << "\n✅ Обработка успешна!\n";
            std::cout << "\nПрофилирование:\n";
            std::cout << "  - GPU kernel time:    " << std::fixed 
                      << std::setprecision(3) << result.gpu_execution_time_ms 
                      << " ms\n";
            std::cout << "  - GPU readback time:  " << result.gpu_readback_time_ms 
                      << " ms\n";
            std::cout << "  - Total time:         " << result.total_time_ms 
                      << " ms\n";
            std::cout << "  - Beams processed:    " << result.beams_processed << "\n";
        } else {
            std::cout << "❌ Ошибка: " << result.error_message << "\n";
            return 1;
        }
        
        // ====================================================================
        // ЭТАП 8: Проверка результатов (данные на CPU!)
        // ====================================================================
        PrintStep(8, "Проверка результатов на CPU");
        
        std::cout << "Результаты находятся на CPU: result.output_data\n";
        std::cout << "Размер: " << result.output_data.size() << " комплексных чисел\n";
        
        if (result.output_data.size() > 0) {
            std::cout << "\nПервые 5 отсчётов луча 0:\n";
            auto beam0 = result.GetBeam(0, 5);  // Получить первые 5 отсчётов
            
            for (size_t i = 0; i < beam0.size(); i++) {
                std::cout << "  [" << i << "] = " 
                          << beam0[i].real << " + " 
                          << beam0[i].imag << "j\n";
            }
        }
        
        // ====================================================================
        // ЭТАП 9: Batch обработка (несколько задержек)
        // ====================================================================
        PrintStep(9, "Batch обработка - несколько задержек");
        
        std::vector<DelayParameter> delays;
        delays.push_back({0, 0.5f});
        delays.push_back({1, 1.5f});
        delays.push_back({2, 2.5f});
        
        std::cout << "Обработка " << delays.size() << " различных задержек...\n";
        
        auto batch_results = processor.ProcessBatch(delays);
        
        std::cout << "\nРезультаты Batch обработки:\n";
        for (size_t i = 0; i < batch_results.size(); i++) {
            auto& r = batch_results[i];
            std::cout << "\n  Задержка [" << i << "]:\n";
            std::cout << "    Success:       " << (r.success ? "Yes" : "No") << "\n";
            std::cout << "    GPU time:      " << std::fixed 
                      << std::setprecision(3) << r.gpu_execution_time_ms 
                      << " ms\n";
            std::cout << "    Output size:   " << r.output_data.size() << "\n";
        }
        
        // ====================================================================
        // ИТОГОВАЯ ИНФОРМАЦИЯ
        // ====================================================================
        PrintHeader("📊 ИТОГОВАЯ ИНФОРМАЦИЯ");
        
        processor.PrintInfo();
        
        std::cout << "\n✅ ПОЛНЫЙ ПРИМЕР ЗАВЕРШЁН УСПЕШНО!\n";
        
        // ====================================================================
        // ОСОБЕННОСТИ РЕАЛИЗАЦИИ
        // ====================================================================
        PrintHeader("🎯 КЛЮЧЕВЫЕ ОСОБЕННОСТИ");
        
        std::cout << "✅ ОДИН вектор на ВХОД:\n";
        std::cout << "   - Все num_beams x num_samples комплексных чисел\n";
        std::cout << "   - Передаются одновременно на GPU\n\n";
        
        std::cout << "✅ ОДИН вектор на ВЫХОД:\n";
        std::cout << "   - Результаты обработки на CPU\n";
        std::cout << "   - Размер: num_beams * num_samples комплексных чисел\n";
        std::cout << "   - Доступны через: result.output_data\n\n";
        
        std::cout << "✅ GPU БУФЕРЫ ОСТАЮТСЯ НА GPU:\n";
        std::cout << "   - buffer_input_: переиспользуется для новых данных\n";
        std::cout << "   - buffer_output_: переиспользуется для результатов\n";
        std::cout << "   - Оптимизация памяти и скорости\n\n";
        
        std::cout << "✅ ВСТРОЕННЫЙ KERNEL:\n";
        std::cout << "   - Lagrange интерполяция 4-го порядка\n";
        std::cout << "   - Поддержка целой и дробной части задержки\n";
        std::cout << "   - Оптимизирован для GPU\n\n";
        
        std::cout << "✅ ПРОФИЛИРОВАНИЕ:\n";
        std::cout << "   - GPU execution time: время работы kernel'а\n";
        std::cout << "   - GPU readback time: время передачи на CPU\n";
        std::cout << "   - Total time: общее время обработки\n\n";
        
        std::cout << std::string(70, '=') << "\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ОШИБКА: " << e.what() << "\n";
        return 1;
    }
}
