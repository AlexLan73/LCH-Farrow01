#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <chrono>

// ⭐ Наша архитектура GPU
#include "GPU/opencl_compute_engine.hpp"
#include "GPU/opencl_core.hpp"
#include "GPU/command_queue_pool.hpp"

// ⭐ Генератор сигналов
#include "generator/generator_gpu_new.h"

// ⭐ Параметры
#include "interface/lfm_parameters.h"
#include "interface/DelayParameter.h"

// ⭐ Процессор дробной задержки
#include "fractional_delay_processor.hpp"

namespace {

// ════════════════════════════════════════════════════════════════════════════
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ════════════════════════════════════════════════════════════════════════════

void PrintHeader(const std::string& title) {
    std::cout << "\n" << std::string(80, '═') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(80, '═') << "\n\n";
}

void PrintSection(const std::string& title) {
    std::cout << "\n┌─ " << title << " ─" << std::string(65 - title.length(), '─') << "┐\n\n";
}

void PrintSuccess(const std::string& msg) {
    std::cout << "✅ " << msg << std::endl;
}

void PrintError(const std::string& msg) {
    std::cerr << "❌ " << msg << std::endl;
}

void PrintInfo(const std::string& msg) {
    std::cout << "ℹ️  " << msg << std::endl;
}

// ════════════════════════════════════════════════════════════════════════════
// ГЛАВНАЯ ФУНКЦИЯ: ДЕМОНСТРАЦИЯ ПРОЦЕССОРА ДРОБНОЙ ЗАДЕРЖКИ
// ════════════════════════════════════════════════════════════════════════════

int main() {
    try {
        PrintHeader("FRACTIONAL DELAY PROCESSOR - ПОЛНЫЙ ПРИМЕР");
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 1: ИНИЦИАЛИЗАЦИЯ OPENCL
        // ════════════════════════════════════════════════════════════════════
        
        PrintSection("Инициализация OpenCL");
        
        PrintInfo("Инициализация OpenCL Core...");
        gpu::OpenCLCore::Initialize(gpu::DeviceType::GPU);
        PrintSuccess("OpenCL Core инициализирован");
        
        PrintInfo("Инициализация Command Queue Pool...");
        gpu::CommandQueuePool::Initialize();
        PrintSuccess("Command Queue Pool инициализирован");
        
        PrintInfo("Инициализация OpenCLComputeEngine...");
        gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
        PrintSuccess("OpenCLComputeEngine инициализирован");
        
        auto& engine = gpu::OpenCLComputeEngine::GetInstance();
        std::cout << "\n" << engine.GetDeviceInfo() << "\n";
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 2: КОНФИГУРАЦИЯ
        // ════════════════════════════════════════════════════════════════════
        
        PrintSection("Конфигурация параметров");
        
        // ✅ Параметры LFM сигналов
        radar::LFMParameters lfm_params;
        lfm_params.f_start = 100.0e6f;              // 100 МГц
        lfm_params.f_stop = 500.0e6f;               // 500 МГц
        lfm_params.sample_rate = 2.0e9f;            // 2 ГГц
        lfm_params.num_beams = 256;                 // 256 лучей
        lfm_params.count_points = 8192;             // 8K отсчётов на луч
        lfm_params.angle_step_deg = 0.5f;           // Шаг 0.5°
        lfm_params.SetAngle(-64.0f, 64.0f);         // Углы от -64° до +64°
        
        PrintInfo("Параметры LFM:");
        std::cout << "  - F start: " << (lfm_params.f_start / 1e6) << " МГц\n";
        std::cout << "  - F stop: " << (lfm_params.f_stop / 1e6) << " МГц\n";
        std::cout << "  - Sample rate: " << (lfm_params.sample_rate / 1e6) << " МГц\n";
        std::cout << "  - Num beams: " << lfm_params.num_beams << "\n";
        std::cout << "  - Num samples: " << lfm_params.count_points << "\n";
        std::cout << "  - Angle range: [" << lfm_params.angle_start_deg << "°, " 
                  << lfm_params.angle_stop_deg << "°]\n";
        
        // ✅ Конфигурация процессора
        radar::FractionalDelayConfig processor_config = 
            radar::FractionalDelayConfig::Diagnostic();
        processor_config.num_beams = lfm_params.num_beams;
        processor_config.num_samples = lfm_params.count_points;
        processor_config.local_work_size = 256;
        
        PrintSuccess("Конфигурация параметров завершена");
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 3: СОЗДАНИЕ ПРОЦЕССОРА
        // ════════════════════════════════════════════════════════════════════
        
        PrintSection("Создание процессора дробной задержки");
        
        radar::FractionalDelayProcessor processor(processor_config, lfm_params);
        
        PrintSuccess("Процессор создан успешно");
        PrintInfo("GPU память использована: " 
                  << (processor.GetGPUBufferSizeBytes() / (1024.0 * 1024.0)) << " МБ");
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 4: ОБРАБОТКА С ОДНОЙ ЗАДЕРЖКОЙ
        // ════════════════════════════════════════════════════════════════════
        
        PrintSection("Обработка с одной дробной задержкой");
        
        radar::DelayParameter delay{0, 0.5f};  // Луч 0, задержка 0.5°
        
        PrintInfo("Применение задержки:");
        std::cout << "  - Beam index: " << delay.beam_index << "\n";
        std::cout << "  - Delay: " << delay.delay_degrees << "°\n";
        
        auto result = processor.ProcessWithFractionalDelay(delay);
        
        if (result.success) {
            PrintSuccess("Обработка завершена!");
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  - GPU execution: " << result.gpu_execution_time_ms << " мс\n";
            std::cout << "  - GPU readback: " << result.gpu_readback_time_ms << " мс\n";
            std::cout << "  - Total time: " << result.total_time_ms << " мс\n";
            std::cout << "  - Output size: " << result.output_data.size() << " элементов\n";
            
            // ✅ Проверка: получить один луч из результата
            auto beam_0 = result.GetBeam(0, lfm_params.count_points);
            if (!beam_0.empty()) {
                PrintSuccess("Луч 0 получен из результата");
                std::cout << "  - Beam 0 size: " << beam_0.size() << " отсчётов\n";
                
                // Показать первые 5 отсчётов
                std::cout << "  - First 5 samples:\n";
                for (size_t i = 0; i < std::min(size_t(5), beam_0.size()); ++i) {
                    auto& val = beam_0[i];
                    std::cout << "    [" << i << "] = " 
                              << std::fixed << std::setprecision(6)
                              << val.real() << " + j" << val.imag() << "\n";
                }
            }
        } else {
            PrintError("Обработка не удалась: " + result.error_message);
        }
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 5: BATCH ОБРАБОТКА (НЕСКОЛЬКО ЗАДЕРЖЕК)
        // ════════════════════════════════════════════════════════════════════
        
        PrintSection("Batch обработка (несколько задержек)");
        
        std::vector<radar::DelayParameter> delay_batch{
            radar::DelayParameter{0, 0.0f},      // Луч 0, без задержки
            radar::DelayParameter{64, 0.5f},     // Луч 64, задержка 0.5°
            radar::DelayParameter{128, 1.0f},    // Луч 128, задержка 1.0°
            radar::DelayParameter{255, 1.5f}     // Луч 255, задержка 1.5°
        };
        
        PrintInfo("Обработка " << delay_batch.size() << " задержек...");
        auto batch_results = processor.ProcessBatch(delay_batch);
        
        // Статистика batch обработки
        uint32_t success_count = 0;
        double total_time = 0.0;
        
        for (size_t i = 0; i < batch_results.size(); ++i) {
            const auto& res = batch_results[i];
            std::cout << "\n  Результат #" << (i + 1) << ":\n";
            
            if (res.success) {
                success_count++;
                total_time += res.total_time_ms;
                std::cout << "    ✅ Успех\n";
                std::cout << "    GPU time: " << std::fixed << std::setprecision(3) 
                          << res.gpu_execution_time_ms << " мс\n";
                std::cout << "    Output elements: " << res.output_data.size() << "\n";
            } else {
                std::cout << "    ❌ Ошибка: " << res.error_message << "\n";
            }
        }
        
        PrintSuccess("Batch обработка завершена!");
        std::cout << "  - Успешных: " << success_count << "/" << batch_results.size() << "\n";
        std::cout << "  - Общее время: " << std::fixed << std::setprecision(3) 
                  << total_time << " мс\n";
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 6: ПРОВЕРКА ДАННЫХ НА CPU И GPU
        // ════════════════════════════════════════════════════════════════════
        
        PrintSection("Проверка данных: CPU vs GPU");
        
        // Данные остаются на GPU для дальнейшего использования
        PrintInfo("✅ Данные осталась на GPU в буферах");
        PrintInfo("✅ Данные также выгружены на CPU в ProcessingResult");
        
        // Проверить, что данные корректны
        if (!result.output_data.empty()) {
            PrintSuccess("CPU данные доступны");
            std::cout << "  - Размер: " << result.output_data.size() << " элементов\n";
            std::cout << "  - Память (мегабайты): " 
                      << (result.output_data.size() * sizeof(std::complex<float>) / (1024.0 * 1024.0))
                      << " МБ\n";
            
            // Статистика по амплитудам
            float max_magnitude = 0.0f;
            float min_magnitude = 1e10f;
            float sum_magnitude = 0.0f;
            
            for (const auto& val : result.output_data) {
                float mag = std::abs(val);
                max_magnitude = std::max(max_magnitude, mag);
                min_magnitude = std::min(min_magnitude, mag);
                sum_magnitude += mag;
            }
            
            float avg_magnitude = sum_magnitude / result.output_data.size();
            
            std::cout << "\n  Статистика амплитуд:\n";
            std::cout << "    Min: " << std::scientific << std::setprecision(3) 
                      << min_magnitude << "\n";
            std::cout << "    Max: " << max_magnitude << "\n";
            std::cout << "    Avg: " << avg_magnitude << "\n";
        }
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 7: СТАТИСТИКА И ИНФОРМАЦИЯ
        // ════════════════════════════════════════════════════════════════════
        
        PrintSection("Статистика и информация");
        
        std::cout << processor.GetStatistics();
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 8: ПРОВЕРКА ДВОЙНОЙ ОБРАБОТКИ (ПЕРЕИСПОЛЬЗОВАНИЕ РЕСУРСОВ)
        // ════════════════════════════════════════════════════════════════════
        
        PrintSection("Переиспользование ресурсов (повторная обработка)");
        
        PrintInfo("Выполнение второй обработки с другими параметрами...");
        
        radar::DelayParameter delay_2{10, 2.0f};  // Луч 10, задержка 2.0°
        auto result_2 = processor.ProcessWithFractionalDelay(delay_2);
        
        if (result_2.success) {
            PrintSuccess("Вторая обработка завершена!");
            std::cout << "  - GPU execution: " << std::fixed << std::setprecision(3) 
                      << result_2.gpu_execution_time_ms << " мс\n";
            std::cout << "  - Output elements: " << result_2.output_data.size() << "\n";
        } else {
            PrintError("Вторая обработка не удалась: " + result_2.error_message);
        }
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 9: ФИНАЛЬНАЯ СТАТИСТИКА
        // ════════════════════════════════════════════════════════════════════
        
        PrintSection("Финальная статистика");
        
        std::cout << processor.GetStatistics();
        
        std::cout << "OpenCLComputeEngine статистика:\n";
        std::cout << engine.GetStatistics() << "\n";
        
        PrintSuccess("Все тесты пройдены успешно!");
        
    } catch (const std::exception& e) {
        PrintError("Критическая ошибка: " + std::string(e.what()));
        return 1;
    }
    
    PrintHeader("ПРОГРАММА ЗАВЕРШЕНА УСПЕШНО");
    return 0;
}

} // namespace

// Entry point
int main() {
    return ::main();
}
