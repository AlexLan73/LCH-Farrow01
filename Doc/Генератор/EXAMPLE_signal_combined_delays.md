📚 ПОЛНЫЙ ПРИМЕР: signal_combined_delays()
🎯 Функция подписи
cpp
cl_mem GeneratorGPU::signal_combined_delays(
    const CombinedDelayParam* combined_delays,
    size_t num_delay_params
);
Что она делает:

Берёт массив параметров задержек (delay_degrees + delay_time_ns для каждого beam)

Загружает их на GPU

Генерирует сигнал с применённой комбинированной задержкой (угол + время)

Возвращает cl_mem буфер на GPU с результатом

📖 ПРИМЕР 1: Базовое использование (без задержек)
cpp
#include <iostream>
#include <vector>
#include <complex>
#include "interface/lfm_parameters.h"
#include "GPU/generator_gpu_new.h"

using namespace radar;
using namespace gpu;

int main() {
    try {
        // ✅ ШАГ 1: Инициализировать GPU
        OpenCLCore::Initialize(DeviceType::GPU);
        CommandQueuePool::Initialize(4);
        OpenCLComputeEngine::Initialize(DeviceType::GPU);

        // ✅ ШАГ 2: Создать параметры сигнала
        LFMParameters params;
        params.fstart = 100.0f;        // 100 МГц
        params.fstop = 500.0f;         // 500 МГц
        params.samplerate = 12.0e6f;   // 12 МГц sampling rate
        params.numbeams = 256;         // 256 лучей
        params.duration = 1.0e-6f;     // 1 микросекунда
        params.countpoints = static_cast<size_t>(
            params.duration * params.samplerate
        );

        // ✅ ШАГ 3: Создать генератор
        GeneratorGPU gen(params);

        // ✅ ШАГ 4: Создать массив с нулевыми задержками
        std::vector<CombinedDelayParam> delays(params.numbeams);
        for (size_t i = 0; i < params.numbeams; i++) {
            delays[i].delay_degrees = 0.0f;   // Нет углов
            delays[i].delay_time_ns = 0.0f;   // Нет временных задержек
        }

        // ✅ ШАГ 5: Сгенерировать сигнал с задержками
        std::cout << "Generating signal with combined delays..." << std::endl;
        cl_mem gpu_signal = gen.signal_combined_delays(
            delays.data(),
            delays.size()
        );

        std::cout << "Signal generated on GPU!" << std::endl;

        // ✅ ШАГ 6: Получить результат на хост (для проверки)
        auto signal_data = gen.GetSignalAsVector(0);  // Луч 0
        std::cout << "Beam 0 first 10 samples:" << std::endl;
        for (size_t i = 0; i < 10 && i < signal_data.size(); i++) {
            std::cout << "  [" << i << "] = " 
                      << signal_data[i].real() << " + j" 
                      << signal_data[i].imag() << std::endl;
        }

        // ✅ ШАГ 7: Очистить GPU
        gen.ClearGPU();

        std::cout << "Done!" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
📖 ПРИМЕР 2: С разными углами (steering)
cpp
// ... инициализация (см. пример 1) ...

GeneratorGPU gen(params);

// Создать задержки с разными углами
std::vector<CombinedDelayParam> delays(params.numbeams);

// Для каждого луча: разный угол, без временных задержек
for (size_t i = 0; i < params.numbeams; i++) {
    float angle_deg = -60.0f + (i * 0.5f);  // От -60° до +60°
    delays[i].delay_degrees = angle_deg;
    delays[i].delay_time_ns = 0.0f;
}

// Генерировать сигнал
cl_mem gpu_signal = gen.signal_combined_delays(
    delays.data(),
    delays.size()
);

std::cout << "Signal with steering angles generated!" << std::endl;

// Получить луч 128 (середина)
auto beam_middle = gen.GetSignalAsVector(128);
📖 ПРИМЕР 3: С разными временами (time delays)
cpp
// ... инициализация ...

GeneratorGPU gen(params);

// Создать задержки с разными временами
std::vector<CombinedDelayParam> delays(params.numbeams);

// Для каждого луча: линейная временная задержка
for (size_t i = 0; i < params.numbeams; i++) {
    delays[i].delay_degrees = 0.0f;           // Нет углов
    delays[i].delay_time_ns = 10.0f * i;      // 10 нс * i (0, 10, 20, ..., 2550 нс)
}

// Генерировать сигнал
cl_mem gpu_signal = gen.signal_combined_delays(
    delays.data(),
    delays.size()
);

std::cout << "Signal with time delays generated!" << std::endl;
📖 ПРИМЕР 4: ПОЛНЫЙ (с углами И временем)
cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "interface/lfm_parameters.h"
#include "GPU/generator_gpu_new.h"

using namespace radar;
using namespace gpu;

int main() {
    try {
        // Инициализация GPU
        OpenCLCore::Initialize(DeviceType::GPU);
        CommandQueuePool::Initialize(4);
        OpenCLComputeEngine::Initialize(DeviceType::GPU);

        // Параметры сигнала
        LFMParameters params;
        params.fstart = 100.0f;
        params.fstop = 500.0f;
        params.samplerate = 12.0e6f;
        params.numbeams = 256;
        params.countpoints = 12000;  // 1 мс при 12 МГц

        GeneratorGPU gen(params);

        // Создать задержки: комбинированные (угол + время)
        std::vector<CombinedDelayParam> delays(params.numbeams);
        
        for (size_t i = 0; i < params.numbeams; i++) {
            // Угол: линейное сканирование от -90° до +90°
            float beam_index_norm = static_cast<float>(i) / (params.numbeams - 1);
            float angle = -90.0f + beam_index_norm * 180.0f;
            
            // Время: пропорционально индексу луча
            float delay_time = 5.0f * i;  // 5 нс * i
            
            delays[i].delay_degrees = angle;
            delays[i].delay_time_ns = delay_time;
        }

        // Генерировать сигнал с комбинированными задержками
        std::cout << "Generating beamformed signal with steering + time delays..." << std::endl;
        cl_mem gpu_signal = gen.signal_combined_delays(
            delays.data(),
            delays.size()
        );

        // Получить несколько лучей для анализа
        std::cout << "\nAnalyzing beams..." << std::endl;
        
        for (int beam_idx : {0, 64, 128, 192, 255}) {
            auto beam_signal = gen.GetSignalAsVector(beam_idx);
            
            // Найти максимум
            float max_amp = 0.0f;
            for (const auto& sample : beam_signal) {
                float amp = std::abs(sample);
                if (amp > max_amp) max_amp = amp;
            }
            
            std::cout << "Beam " << beam_idx 
                      << " - Angle: " << delays[beam_idx].delay_degrees
                      << "° - Time: " << delays[beam_idx].delay_time_ns
                      << " ns - Max amplitude: " << max_amp << std::endl;
        }

        // Статистика
        std::cout << "\nGPU Statistics:" << std::endl;
        auto& engine = OpenCLComputeEngine::GetInstance();
        std::cout << engine.GetStatistics();

        // Очистка
        gen.ClearGPU();
        OpenCLComputeEngine::Cleanup();
        CommandQueuePool::Cleanup();
        OpenCLCore::Cleanup();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
🔍 ДЕТАЛИ СТРУКТУРЫ CombinedDelayParam
cpp
struct CombinedDelayParam {
    float delay_degrees;   // Угловая задержка (steering angle)
    float delay_time_ns;   // Временная задержка в наносекундах
};
Применяется kernel'ом kernel_lfm_combined так:

text
// GPU kernel (OpenCL C)
__kernel void kernel_lfm_combined(
    __global float2 *output,
    __global const CombinedDelayParam *delays
) {
    // Для каждого луча применяется его задержка
    // phase_shift = 2π * (angle_to_freq * delay_degrees + 
    //                     freq_start * delay_time_ns)
}
✅ CHECKLIST: Что нужно сделать
 Инициализировать GPU (OpenCLCore::Initialize)

 Инициализировать пул очередей (CommandQueuePool::Initialize)

 Инициализировать engine (OpenCLComputeEngine::Initialize)

 Создать GeneratorGPU с параметрами

 Заполнить массив CombinedDelayParam

 Вызвать signal_combined_delays()

 Получить результат через GetSignalAsVector()

 Очистить GPU через ClearGPU()

 Очистить ресурсы

🚀 КОМПИЛЯЦИЯ
bash
g++ -std=c++17 -O3 example.cpp generator_gpu_new.cpp opencl_compute_engine.cpp \
    -o example -lOpenCL -I./include
Готово к использованию! 🎉

