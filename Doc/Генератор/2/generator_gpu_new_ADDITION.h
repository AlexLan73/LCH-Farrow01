#pragma once

#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

/**
 * @file generator_gpu_new_fixed.h
 * @brief GeneratorGPU с добавленным методом GetSignalAsVector()
 * 
 * Этот файл содержит ДОПОЛНЕНИЕ к generator_gpu_new.h
 * Добавляем публичный метод для удобного чтения результатов с GPU
 */

namespace radar {

// ════════════════════════════════════════════════════════════════════════════
// ДОПОЛНЕНИЕ К КЛАССУ GeneratorGPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class GeneratorGPU
 * @brief ДОПОЛНЕНИЕ: Методы для чтения результатов с GPU
 * 
 * Добавляем эти методы в класс GeneratorGPU (после существующих методов)
 */

// ════════════════════════════════════════════════════════════════════════════
// ВСТАВИТЬ ЭТО В КОНЕЦ ПУБЛИЧНОЙ ЧАСТИ КЛАССА GeneratorGPU:
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить сигнал конкретного луча как вектор комплексных чисел
 * @param beam_index Индекс луча (0 до num_beams-1)
 * @return std::vector<std::complex<float>> вектор сигнала
 * 
 * ИСПОЛЬЗОВАНИЕ:
 * ```cpp
 * GeneratorGPU gen(params);
 * cl_mem gpu_signal = gen.signal_base();
 * gen.ClearGPU();  // Синхронизировать!
 * 
 * auto beam0_data = gen.GetSignalAsVector(0);  // Луч 0
 * auto beam1_data = gen.GetSignalAsVector(1);  // Луч 1
 * ```
 * 
 * ПРЕИМУЩЕСТВА:
 * - Автоматическая синхронизация GPU
 * - Правильное извлечение нужного луча
 * - Проверка индекса
 * - Подробное логирование
 */
// std::vector<std::complex<float>> GetSignalAsVector(int beam_index = 0);

// ════════════════════════════════════════════════════════════════════════════
// РЕАЛИЗАЦИЯ (вставить в generator_gpu_new.cpp)
// ════════════════════════════════════════════════════════════════════════════

/*

std::vector<std::complex<float>> GeneratorGPU::GetSignalAsVector(int beam_index) {
    // ✅ ШАГИ:
    // 1. Проверить индекс
    // 2. Синхронизировать GPU
    // 3. Получить engine и OpenCLCore
    // 4. Обернуть raw cl_mem в GPUMemoryBuffer (NON-OWNING!)
    // 5. Прочитать все данные
    // 6. Извлечь нужный луч
    // 7. Логировать результат
    // 8. Вернуть вектор
    
    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 1: Проверить индекс
    // ════════════════════════════════════════════════════════════════════════
    
    if (beam_index < 0 || beam_index >= (int)num_beams_) {
        std::cerr << "❌ GeneratorGPU::GetSignalAsVector() - Invalid beam index: " 
                  << beam_index << " (expected 0 to " << (num_beams_ - 1) << ")" << std::endl;
        return {};  // Пустой вектор при ошибке
    }
    
    std::cout << "[GPU] Reading beam " << beam_index << " from GPU..." << std::endl;
    
    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 2: Синхронизировать GPU перед чтением
    // ════════════════════════════════════════════════════════════════════════
    
    ClearGPU();  // Ждём завершения всех операций
    
    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 3: Получить engine и OpenCLCore
    // ════════════════════════════════════════════════════════════════════════
    
    auto& engine = gpu::OpenCLComputeEngine::GetInstance();
    auto& core = gpu::OpenCLCore::GetInstance();
    
    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 4: Обернуть raw cl_mem в GPUMemoryBuffer (NON-OWNING!)
    // ════════════════════════════════════════════════════════════════════════
    // 
    // ВАЖНО: Используем NON-OWNING конструктор (второй параметр - external buffer)!
    // Это значит GPUMemoryBuffer НЕ удалит cl_mem при своём разрушении.
    // Удаление сделает GeneratorGPU в своём деструкторе.
    // 
    // Конструктор:
    // GPUMemoryBuffer(context, queue, external_buffer, num_elements, type)
    
    try {
        gpu::GPUMemoryBuffer buffer(
            core.GetContext(),                          // контекст OpenCL
            gpu::CommandQueuePool::GetNextQueue(),      // очередь для операции
            buffer_signal_base_,                        // raw cl_mem (НЕ удалится!)
            total_size_,                                // всего элементов (num_beams * num_samples)
            gpu::MemoryType::GPU_READ_ONLY              // тип: только чтение
        );
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 5: Прочитать все данные
        // ════════════════════════════════════════════════════════════════════
        
        std::cout << "[READ] Reading " << total_size_ << " samples from GPU..." << std::endl;
        auto all_data = buffer.ReadFromGPU();
        
        if (all_data.empty()) {
            std::cerr << "❌ Failed to read data from GPU!" << std::endl;
            return {};
        }
        
        std::cout << "[READ] ✅ Successfully read " << all_data.size() << " samples" << std::endl;
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 6: Извлечь нужный луч
        // ════════════════════════════════════════════════════════════════════
        // 
        // Структура в GPU памяти:
        // [Beam 0: samples 0..N-1] [Beam 1: samples N..2N-1] ... [Beam K-1: samples ...]
        // 
        // beam_index = 0: start = 0*N, end = 1*N
        // beam_index = 1: start = 1*N, end = 2*N
        // beam_index = K-1: start = (K-1)*N, end = K*N
        
        size_t beam_start = beam_index * num_samples_;
        size_t beam_end = beam_start + num_samples_;
        
        std::cout << "[EXTRACT] Extracting beam " << beam_index 
                  << " (samples " << beam_start << ".." << (beam_end-1) << ")" << std::endl;
        
        std::vector<std::complex<float>> result(
            all_data.begin() + beam_start,
            all_data.begin() + beam_end
        );
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 7: Логировать результат
        // ════════════════════════════════════════════════════════════════════
        
        std::cout << "✅ Beam " << beam_index << " read successfully" << std::endl;
        std::cout << "   Size: " << result.size() << " samples" << std::endl;
        
        if (!result.empty()) {
            float amp = std::abs(result[0]);
            float phase = std::arg(result[0]);
            std::cout << "   First sample: " << result[0].real() << " + j" 
                      << result[0].imag() << std::endl;
            std::cout << "   Amplitude: " << amp << ", Phase: " << phase << " rad" << std::endl;
        }
        
        // ════════════════════════════════════════════════════════════════════
        // ШАГ 8: Вернуть вектор
        // ════════════════════════════════════════════════════════════════════
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in GetSignalAsVector(): " << e.what() << std::endl;
        return {};
    }
}

*/

// ════════════════════════════════════════════════════════════════════════════
// АЛЬТЕРНАТИВНЫЙ МЕТОД: Для частичного чтения (ОПЦИОНАЛЬНО)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить несколько сэмплов конкретного луча
 * @param beam_index Индекс луча
 * @param num_samples Количество сэмплов для чтения
 * @return Вектор комплексных чисел
 * 
 * ЕСЛИ У ВАС БОЛЬШОЙ БУФЕР И НУЖНЫ ТОЛЬКО НЕСКОЛЬКО СЭМПЛОВ:
 * 
 * ```cpp
 * // Прочитать только первые 32 сэмпла луча 0
 * auto first_samples = gen.GetSignalAsVectorPartial(0, 32);
 * ```
 */
// std::vector<std::complex<float>> GetSignalAsVectorPartial(int beam_index, size_t num_samples);

/*

std::vector<std::complex<float>> GeneratorGPU::GetSignalAsVectorPartial(
    int beam_index, 
    size_t num_samples
) {
    // Такой же как GetSignalAsVector(), но используем ReadPartial():
    
    if (beam_index < 0 || beam_index >= (int)num_beams_) {
        return {};
    }
    
    if (num_samples > num_samples_) {
        num_samples = num_samples_;
    }
    
    ClearGPU();
    
    auto& core = gpu::OpenCLCore::GetInstance();
    
    gpu::GPUMemoryBuffer buffer(
        core.GetContext(),
        gpu::CommandQueuePool::GetNextQueue(),
        buffer_signal_base_,
        total_size_,
        gpu::MemoryType::GPU_READ_ONLY
    );
    
    // РАЗЛИЧИЕ: используем ReadPartial() вместо ReadFromGPU()
    auto all_data = buffer.ReadPartial(total_size_);  // Сначала читаем всё
    
    size_t beam_start = beam_index * num_samples_;
    size_t beam_end = beam_start + num_samples;  // ← num_samples, не num_samples_!
    
    std::vector<std::complex<float>> result(
        all_data.begin() + beam_start,
        all_data.begin() + beam_end
    );
    
    return result;
}

*/

} // namespace radar

// ════════════════════════════════════════════════════════════════════════════
// ИНСТРУКЦИЯ ПО ДОБАВЛЕНИЮ
// ════════════════════════════════════════════════════════════════════════════

/**
 * 1. ОТКРЫТЬ: generator_gpu_new.h
 * 
 * 2. НАЙТИ: Конец публичной части класса GeneratorGPU (перед private:)
 * 
 * 3. ДОБАВИТЬ эту строку:
 * 
 *    std::vector<std::complex<float>> GetSignalAsVector(int beam_index = 0);
 * 
 * 4. ОТКРЫТЬ: generator_gpu_new.cpp
 * 
 * 5. ДОБАВИТЬ в конец файла код из секции "РЕАЛИЗАЦИЯ" выше (между /* и */)
 * 
 * 6. СКОМПИЛИРОВАТЬ:
 * 
 *    cmake ..
 *    cmake --build .
 * 
 * 7. ИСПОЛЬЗОВАТЬ:
 * 
 *    GeneratorGPU gen(params);
 *    cl_mem signal = gen.signal_base();
 *    gen.ClearGPU();
 *    
 *    auto beam0 = gen.GetSignalAsVector(0);
 *    auto beam1 = gen.GetSignalAsVector(1);
 *    
 *    std::cout << "Beam 0 size: " << beam0.size() << std::endl;
 *    std::cout << "First sample: " << beam0[0] << std::endl;
 */

#endif // GENERATOR_GPU_NEW_FIXED_H
