#ifndef FRACTIONAL_DELAY_PROCESSOR_HPP
#define FRACTIONAL_DELAY_PROCESSOR_HPP

#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <CL/cl.h>

// Forward declarations
namespace gpu {
    class OpenCLComputeEngine;
}

namespace radar {

// ============================================================================
// ТИПЫ ДАННЫХ
// ============================================================================

/// Комплексное число для LFM сигналов
struct Complex {
    float real;
    float imag;
    
    Complex() : real(0.0f), imag(0.0f) {}
    Complex(float r, float i) : real(r), imag(i) {}
};

/// Вектор комплексных чисел
using ComplexVector = std::vector<Complex>;

// ============================================================================
// КОНФИГУРАЦИЯ
// ============================================================================

/// Конфигурация обработки дробной задержки
struct FractionalDelayConfig {
    uint32_t num_beams;              ///< Количество лучей/антенн (1..512)
    uint32_t num_samples;            ///< Количество отсчётов (16+)
    uint32_t local_work_size;        ///< Размер workgroup (1..1024)
    bool verbose;                    ///< Диагностический вывод
    
    /// Стандартная конфигурация (256 лучей, 8K отсчётов)
    static FractionalDelayConfig Standard() {
        return {256, 8192, 256, false};
    }
    
    /// Конфигурация для производительности
    static FractionalDelayConfig Performance() {
        return {512, 131072, 512, false};
    }
    
    /// Конфигурация для диагностики
    static FractionalDelayConfig Diagnostic() {
        return {64, 1024, 64, true};
    }
};

// ============================================================================
// ПАРАМЕТРЫ ЗАДЕРЖКИ
// ============================================================================

/// Параметр задержки для одного луча
struct DelayParameter {
    uint32_t beam_index;        ///< Номер луча (0..num_beams-1)
    float delay_degrees;        ///< Задержка в градусах
};

// ============================================================================
// РЕЗУЛЬТАТ ОБРАБОТКИ
// ============================================================================

/// Результат обработки дробной задержки
struct ProcessingResult {
    /// УСПЕХ ОПЕРАЦИИ
    bool success;
    std::string error_message;
    
    /// GPU ПРОФИЛИРОВАНИЕ (в миллисекундах)
    double gpu_execution_time_ms;    ///< Время выполнения kernel'а
    double gpu_readback_time_ms;     ///< Время передачи данных с GPU
    double total_time_ms;            ///< Общее время
    
    /// РЕЗУЛЬТАТЫ
    uint32_t beams_processed;        ///< Количество обработанных лучей
    ComplexVector output_data;       ///< ✅ ВЕКТОР НА CPU (num_beams * num_samples)
    
    /// Получить один луч из результатов
    /**
     * @param beam_index - номер луча
     * @param num_samples - количество отсчётов в луче
     * @return Вектор отсчётов луча
     */
    ComplexVector GetBeam(uint32_t beam_index, uint32_t num_samples) const {
        if (beam_index * num_samples >= output_data.size()) {
            throw std::out_of_range("Invalid beam index");
        }
        auto start = output_data.begin() + beam_index * num_samples;
        auto end = start + num_samples;
        return ComplexVector(start, end);
    }
};

// ============================================================================
// ПАРАМЕТРЫ LFM
// ============================================================================

/// Параметры LFM сигнала
struct LFMParameters {
    uint32_t num_beams;          ///< Количество лучей
    uint32_t count_points;       ///< Количество отсчётов
    float f_start;               ///< Начальная частота (Hz)
    float f_stop;                ///< Конечная частота (Hz)
    float sample_rate;           ///< Частота дискретизации (Hz)
    float amplitude;             ///< Амплитуда сигнала
};

// ============================================================================
// ГЛАВНЫЙ КЛАСС - FractionalDelayProcessor
// ============================================================================

/**
 * @class FractionalDelayProcessor
 * @brief Процессор дробной задержки для LFM сигналов на GPU
 * 
 * Использует OpenCL kernel для применения дробной задержки к LFM сигналам.
 * 
 * АРХИТЕКТУРА:
 * - ОДИН вектор на ВХОД (все антенны, все точки)
 * - ОДИН вектор на ВЫХОД (результат обработки на CPU)
 * - GPU буферы переиспользуются для оптимизации
 * 
 * EXAMPLE:
 * @code
 * // Инициализировать OpenCL
 * gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
 * 
 * // Создать конфигурацию и параметры
 * auto config = FractionalDelayConfig::Standard();
 * LFMParameters lfm;
 * lfm.num_beams = 256;
 * lfm.count_points = 8192;
 * 
 * // Создать процессор
 * FractionalDelayProcessor processor(config, lfm);
 * 
 * // Обработать с задержкой
 * DelayParameter delay{0, 0.5f};
 * auto result = processor.ProcessWithFractionalDelay(delay);
 * 
 * if (result.success) {
 *     std::cout << "GPU time: " << result.gpu_execution_time_ms << "ms\n";
 *     auto beam = result.GetBeam(0, lfm.count_points);
 * }
 * @endcode
 */
class FractionalDelayProcessor {
public:
    // ========================================================================
    // КОНСТРУКТОР / ДЕСТРУКТОР
    // ========================================================================
    
    /**
     * @brief Конструктор с инициализацией
     * @param config - конфигурация обработки
     * @param lfm_params - параметры LFM сигнала
     * @throws std::runtime_error если OpenCL не инициализирован
     */
    FractionalDelayProcessor(
        const FractionalDelayConfig& config,
        const LFMParameters& lfm_params
    );
    
    /// Деструктор с освобождением GPU ресурсов
    ~FractionalDelayProcessor();
    
    // Move семантика (оптимальная эффективность)
    FractionalDelayProcessor(FractionalDelayProcessor&& other) noexcept;
    FractionalDelayProcessor& operator=(FractionalDelayProcessor&& other) noexcept;
    
    // Удаляем копирование (нельзя копировать GPU буферы)
    FractionalDelayProcessor(const FractionalDelayProcessor&) = delete;
    FractionalDelayProcessor& operator=(const FractionalDelayProcessor&) = delete;
    
    // ========================================================================
    // ОСНОВНАЯ ФУНКЦИОНАЛЬНОСТЬ
    // ========================================================================
    
    /**
     * @brief Обработка сигнала с ОДНОЙ задержкой
     * 
     * Применяет дробную задержку ко всем лучам/антеннам.
     * 
     * @param delay - параметр задержки
     * @return ProcessingResult с результатами на CPU
     * 
     * ГАРАНТИРУЕТ:
     * - Данные остаются на GPU (buffer_input_, buffer_output_)
     * - Результаты выгружены на CPU
     * - Профилирование GPU времени
     * - Exception-safe
     */
    ProcessingResult ProcessWithFractionalDelay(const DelayParameter& delay);
    
    /**
     * @brief Batch обработка (несколько задержек)
     * 
     * Применяет несколько различных задержек с переиспользованием буферов.
     * 
     * @param delays - вектор параметров задержек
     * @return Вектор результатов (по одному на каждую задержку)
     */
    std::vector<ProcessingResult> ProcessBatch(
        const std::vector<DelayParameter>& delays
    );
    
    // ========================================================================
    // ДИАГНОСТИКА И СТАТИСТИКА
    // ========================================================================
    
    /// Вывести информацию о процессоре
    void PrintInfo() const;
    
    /// Получить статистику использования
    std::string GetStatistics() const;
    
private:
    // ========================================================================
    // ПРИВАТНЫЕ ДАННЫЕ
    // ========================================================================
    
    // Конфигурация
    FractionalDelayConfig config_;
    LFMParameters lfm_params_;
    gpu::OpenCLComputeEngine* engine_;
    
    // OpenCL объекты
    cl_kernel kernel_;
    std::unique_ptr<gpu::IMemoryBuffer> buffer_input_;   ///< GPU буфер входа
    std::unique_ptr<gpu::IMemoryBuffer> buffer_output_;  ///< GPU буфер выхода
    
    // Статистика
    uint64_t total_samples_processed_;
    double total_gpu_time_;
    uint32_t total_calls_;
    
    // ========================================================================
    // ПРИВАТНЫЕ МЕТОДЫ
    // ========================================================================
    
    /// Инициализировать процессор
    void Initialize();
    
    /// Загрузить OpenCL kernel
    void LoadKernels();
    
    /// Создать GPU буферы (ОДИН вектор на вход/выход!)
    void CreateBuffers();
    
    /// Синхронизировать GPU
    void SyncGPU();
    
    /// Получить исходный код kernel'а
    static std::string GetKernelSource();
};

} // namespace radar

#endif // FRACTIONAL_DELAY_PROCESSOR_HPP
