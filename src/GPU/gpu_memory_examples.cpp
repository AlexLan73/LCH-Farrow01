// ═══════════════════════════════════════════════════════════════════════════
// ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ GPUMemoryBuffer
// ═══════════════════════════════════════════════════════════════════════════

#include "gpu_memory_buffer.hpp"
#include "generatorgpu.h"
#include <iostream>
#include <vector>

using namespace radar::gpu;

// ═════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 1: Полный трансфер GPU → CPU с RAII
// ═════════════════════════════════════════════════════════════════════════════

void Example1_FullTransfer(
    std::shared_ptr<GeneratorGPU>& gen_gpu,
    const cl_mem& signal_gpu
) {
    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "ПРИМЕР 1: Полный GPU → CPU трансфер с RAII\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";

    try {
        // 1. Создать GPUMemoryBuffer (автоматическое выделение памяти)
        auto buffer = std::make_unique<GPUMemoryBuffer>(
            gen_gpu->GetContext(),
            gen_gpu->GetQueue(),
            gen_gpu->GetNumSamples() * gen_gpu->GetNumBeams(),
            MemoryType::GPU_WRITE_ONLY  // Kernel пишет, CPU читает
        );

        // 2. Показать статистику памяти
        buffer->PrintStats();

        // 3. Читать данные GPU → CPU (с pinned memory оптимизацией)
        std::vector<std::complex<float>> cpu_data = buffer->ReadFromGPU();

        // 4. Обработать данные
        std::cout << "📊 First 10 samples (ray 0):\n";
        for (size_t i = 0; i < std::min(size_t(10), cpu_data.size()); ++i) {
            std::cout << "  [" << i << "] = " << cpu_data[i].real() 
                      << " + " << cpu_data[i].imag() << "j\n";
        }

        // 5. Автоматическое освобождение памяти при выходе из области видимости
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 2: Частичное чтение (первые N элементов)
// ═════════════════════════════════════════════════════════════════════════════

void Example2_PartialRead(
    std::shared_ptr<GeneratorGPU>& gen_gpu,
    const cl_mem& signal_gpu
) {
    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "ПРИМЕР 2: Частичный GPU → CPU трансфер (первые 10 элементов)\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";

    try {
        auto buffer = std::make_unique<GPUMemoryBuffer>(
            gen_gpu->GetContext(),
            gen_gpu->GetQueue(),
            gen_gpu->GetNumSamples() * gen_gpu->GetNumBeams(),
            MemoryType::GPU_WRITE_ONLY
        );

        // Читать только первые 10 элементов (быстрее!)
        std::vector<std::complex<float>> partial_data = buffer->ReadPartial(10);

        std::cout << "📊 Partial data (10 samples):\n";
        for (size_t i = 0; i < partial_data.size(); ++i) {
            std::cout << "  [" << i << "] = " << partial_data[i].real() 
                      << " + " << partial_data[i].imag() << "j\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 3: Двусторонний трансфер (CPU → GPU → CPU)
// ═════════════════════════════════════════════════════════════════════════════

void Example3_Bidirectional(
    std::shared_ptr<GeneratorGPU>& gen_gpu
) {
    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "ПРИМЕР 3: Двусторонний трансфер CPU ↔ GPU\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";

    try {
        // Использовать GPU_READ_WRITE для чтения и записи
        auto buffer = std::make_unique<GPUMemoryBuffer>(
            gen_gpu->GetContext(),
            gen_gpu->GetQueue(),
            gen_gpu->GetNumSamples() * gen_gpu->GetNumBeams(),
            MemoryType::GPU_READ_WRITE
        );

        // 1. Подготовить тестовые данные на CPU
        std::vector<std::complex<float>> test_data(buffer->GetNumElements());
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = std::complex<float>(
                static_cast<float>(i),
                static_cast<float>(i * 2)
            );
        }

        // 2. Записать на GPU
        buffer->WriteToGPU(test_data);

        // 3. Проверить, что данные на GPU "dirty"
        std::cout << "GPU Dirty flag: " << (buffer->IsGPUDirty() ? "Yes" : "No") << "\n";

        // 4. Прочитать обратно с GPU
        std::vector<std::complex<float>> readback = buffer->ReadFromGPU();

        // 5. Сравнить
        std::cout << "\n📊 Data verification (first 5 elements):\n";
        bool all_match = true;
        for (size_t i = 0; i < std::min(size_t(5), readback.size()); ++i) {
            bool match = (test_data[i] == readback[i]);
            std::cout << "  [" << i << "] Original: " << test_data[i]
                      << " Read: " << readback[i]
                      << " " << (match ? "✓" : "✗") << "\n";
            if (!match) all_match = false;
        }

        std::cout << "\n" << (all_match ? "✅ All data matches!" : "❌ Data mismatch!") << "\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 4: Pool буферов для множественных операций
// ═════════════════════════════════════════════════════════════════════════════

void Example4_BufferPool(
    std::shared_ptr<GeneratorGPU>& gen_gpu
) {
    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "ПРИМЕР 4: Pool буферов для нескольких операций\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";

    try {
        size_t num_buffers = 3;
        std::vector<std::unique_ptr<GPUMemoryBuffer>> buffer_pool;

        // Создать pool буферов
        for (size_t i = 0; i < num_buffers; ++i) {
            buffer_pool.push_back(
                std::make_unique<GPUMemoryBuffer>(
                    gen_gpu->GetContext(),
                    gen_gpu->GetQueue(),
                    gen_gpu->GetNumSamples() * gen_gpu->GetNumBeams(),
                    MemoryType::GPU_READ_WRITE
                )
            );
            std::cout << "Created buffer " << i + 1 << "/" << num_buffers << "\n";
        }

        // Использовать буферы
        for (size_t i = 0; i < buffer_pool.size(); ++i) {
            std::cout << "\nBuffer " << i << " info:\n";
            buffer_pool[i]->PrintStats();
        }

        std::cout << "\n✅ Total GPU memory: " 
                  << (buffer_pool.size() * 
                      buffer_pool[0]->GetTotalBytes() / (1024.0 * 1024.0))
                  << " MB\n";

        // Буферы автоматически удаляются при выходе из области видимости

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 5: Использование с функцией gpu_to_cpu (вместо вашей текущей версии)
// ═════════════════════════════════════════════════════════════════════════════

void Example5_ReplacementForGpuToCpu(
    std::shared_ptr<GeneratorGPU>& gen_gpu,
    const cl_mem& signal_gpu
) {
    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "ПРИМЕР 5: Замена для вашей gpu_to_cpu функции\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";

    try {
        // НОВЫЙ ПОДХОД с GPUMemoryBuffer (вместо старой функции)
        auto buffer = std::make_unique<GPUMemoryBuffer>(
            gen_gpu->GetContext(),
            gen_gpu->GetQueue(),
            gen_gpu->GetNumSamples() * gen_gpu->GetNumBeams(),
            MemoryType::GPU_WRITE_ONLY
        );

        // Частично читаем (как в вашей старой функции)
        size_t read_samples = std::min(size_t(10), gen_gpu->GetNumSamples());
        std::vector<std::complex<float>> cpu_data = buffer->ReadPartial(read_samples);

        std::cout << "📤 Трансфер данных GPU → CPU (первый луч, первые " 
                  << read_samples << " отсчётов signal_base):\n";

        for (size_t i = 0; i < cpu_data.size(); ++i) {
            std::cout << "  [" << i << "] = " << cpu_data[i].real() 
                      << " + " << cpu_data[i].imag() << "j\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка при чтении из GPU (код: " << e.what() << ")\n";
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ГЛАВНАЯ ФУНКЦИЯ - запустить все примеры
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    try {
        // Инициализация GPU (предполагаем, что это уже сделано)
        auto gen_gpu = std::make_shared<GeneratorGPU>(
            LFMParameters{...}  // Ваши параметры
        );

        cl_mem signal_gpu = gen_gpu->signal_base();

        std::cout << "🚀 GPU Memory Transfer Examples\n\n";

        // Запустить примеры
        Example1_FullTransfer(gen_gpu, signal_gpu);
        Example2_PartialRead(gen_gpu, signal_gpu);
        Example3_Bidirectional(gen_gpu);
        Example4_BufferPool(gen_gpu);
        Example5_ReplacementForGpuToCpu(gen_gpu, signal_gpu);

        std::cout << "\n✅ All examples completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
