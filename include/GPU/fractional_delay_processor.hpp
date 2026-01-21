/**
 * @file fractional_delay_processor.hpp
 * @brief Процессор дробной задержки для LFM сигналов на GPU (OpenCL)
 * 
 * @details Использует интерполяцию Лагранжа с матрицей 48×5 коэффициентов
 *          для точного вычисления дробной задержки LFM радарных сигналов.
 * 
 * АРХИТЕКТУРА (GRASP & GoF паттерны):
 * - Information Expert: Класс владеет всеми данными для обработки
 * - Creator: Factory методы для создания конфигураций
 * - Low Coupling: Минимальные зависимости от внешних классов
 * - High Cohesion: Все методы связаны с задержкой сигналов
 * - Strategy: Разные стратегии задержки (градусы/отсчёты)
 * 
 * ФОРМАТ ДАННЫХ:
 * - Входной буфер: [beam0_sample0, beam0_sample1, ..., beam0_sampleN,
 *                   beam1_sample0, beam1_sample1, ..., beam1_sampleN, ...]
 * - Формула индекса: buffer[beam_idx * num_samples + sample_idx]
 * - IN-PLACE обработка: результат записывается в тот же буфер
 * 
 * @author LCH-Farrow01 Project
 * @version 2.0
 * @date 2026-01-21
 */

#ifndef FRACTIONAL_DELAY_PROCESSOR_HPP
#define FRACTIONAL_DELAY_PROCESSOR_HPP

#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <complex>
#include <array>
#include <CL/cl.h>

// Forward declarations
namespace gpu {
    class OpenCLComputeEngine;
    class GPUMemoryBuffer;
}

namespace radar {

// ============================================================================
// КОНСТАНТЫ
// ============================================================================

/// Количество строк в матрице Лагранжа (48 дробных значений: 0.00, 0.02, ..., 0.98)
constexpr uint32_t LAGRANGE_ROWS = 48;

/// Количество коэффициентов в каждой строке (5-точечная интерполяция)
constexpr uint32_t LAGRANGE_COLS = 5;

/// Максимальное количество лучей
constexpr uint32_t MAX_BEAMS = 256;

/// Максимальное количество отсчётов на луч
constexpr uint32_t MAX_SAMPLES = 1310720;  // ~1.3M

// ============================================================================
// ТИПЫ ДАННЫХ
// ============================================================================

/// Комплексное число для LFM сигналов (совместимо с OpenCL float2)
struct alignas(8) Complex {
    float real;
    float imag;
    
    Complex() : real(0.0f), imag(0.0f) {}
    Complex(float r, float i) : real(r), imag(i) {}
    Complex(const std::complex<float>& c) : real(c.real()), imag(c.imag()) {}
    
    operator std::complex<float>() const { return std::complex<float>(real, imag); }
};

/// Вектор комплексных чисел
using ComplexVector = std::vector<Complex>;

// ============================================================================
// ПАРАМЕТРЫ ЗАДЕРЖКИ
// ============================================================================

/**
 * @struct DelayParams
 * @brief Параметры задержки для одного луча
 * 
 * Задержка разбивается на:
 * - Целую часть (delay_integer): смещение в отсчётах
 * - Дробную часть (lagrange_row): строка матрицы Лагранжа (0-47)
 *   Row 0 = 0.00, Row 1 = 0.02, ..., Row 47 = 0.98 (шаг ~0.0208)
 */
struct DelayParams {
    int32_t  delay_integer;    ///< Целая часть задержки (отсчёты), может быть отрицательной
    uint32_t lagrange_row;     ///< Строка матрицы Лагранжа [0..47] для дробной части
    
    /// Конструктор по умолчанию (нулевая задержка)
    DelayParams() : delay_integer(0), lagrange_row(0) {}
    
    /// Конструктор с параметрами
    DelayParams(int32_t delay_int, uint32_t lag_row) 
        : delay_integer(delay_int), lagrange_row(lag_row % LAGRANGE_ROWS) {}
    
    /// Создать из общей задержки в отсчётах (float)
    static DelayParams FromSamples(float delay_samples) {
        DelayParams p;
        p.delay_integer = static_cast<int32_t>(std::floor(delay_samples));
        float frac = delay_samples - p.delay_integer;  // [0, 1)
        p.lagrange_row = static_cast<uint32_t>(frac * LAGRANGE_ROWS) % LAGRANGE_ROWS;
        return p;
    }
    
    /// Получить общую задержку в отсчётах
    float GetTotalDelaySamples() const {
        return static_cast<float>(delay_integer) + 
               static_cast<float>(lagrange_row) / static_cast<float>(LAGRANGE_ROWS);
    }
};

// ============================================================================
// КОНФИГУРАЦИЯ
// ============================================================================

/**
 * @struct FractionalDelayConfig
 * @brief Конфигурация процессора дробной задержки
 */
struct FractionalDelayConfig {
    uint32_t num_beams;              ///< Количество лучей/антенн [1..256]
    uint32_t num_samples;            ///< Количество отсчётов на луч [16..1310720]
    uint32_t local_work_size;        ///< Размер workgroup для OpenCL [64..512]
    bool     verbose;                ///< Подробный вывод
    bool     enable_profiling;       ///< Включить GPU профилирование
    
    /// Стандартная конфигурация (64 луча, 8K отсчётов)
    static FractionalDelayConfig Standard() {
        return {64, 8192, 256, false, true};
    }
    
    /// Конфигурация для производительности (256 лучей, 1M отсчётов)
    static FractionalDelayConfig Performance() {
        return {256, 1048576, 512, false, true};
    }
    
    /// Конфигурация для диагностики
    static FractionalDelayConfig Diagnostic() {
        return {16, 1024, 64, true, true};
    }
    
    /// Валидация параметров
    bool IsValid() const {
        return num_beams >= 1 && num_beams <= MAX_BEAMS &&
               num_samples >= 16 && num_samples <= MAX_SAMPLES &&
               local_work_size >= 32 && local_work_size <= 1024;
    }
};

// ============================================================================
// РЕЗУЛЬТАТ ПРОФИЛИРОВАНИЯ
// ============================================================================

/**
 * @struct FDPProfilingResults
 * @brief Результаты GPU профилирования
 */
struct FDPProfilingResults {
    double upload_time_ms;       ///< Время загрузки матрицы Лагранжа
    double kernel_time_ms;       ///< Время выполнения kernel'а
    double total_time_ms;        ///< Общее время обработки
    
    uint64_t samples_processed;  ///< Обработано отсчётов
    uint32_t beams_processed;    ///< Обработано лучей
    
    /// Получить пропускную способность (отсчётов/сек)
    double GetThroughput() const {
        return (total_time_ms > 0) ? (samples_processed * 1000.0 / total_time_ms) : 0.0;
    }
};

// ============================================================================
// МАТРИЦА ЛАГРАНЖА
// ============================================================================

/**
 * @struct LagrangeMatrix
 * @brief Матрица коэффициентов Лагранжа 48×5
 * 
 * Загружается из lagrange_matrix.json
 * Каждая строка соответствует дробной части задержки:
 * Row 0: frac = 0.00, Row 1: frac ≈ 0.02, ..., Row 47: frac ≈ 0.98
 */
struct LagrangeMatrix {
    std::array<std::array<float, LAGRANGE_COLS>, LAGRANGE_ROWS> coefficients;
    
    /// Загрузить из JSON файла
    static LagrangeMatrix LoadFromJSON(const std::string& filepath);
    
    /// Получить коэффициенты для строки
    const std::array<float, LAGRANGE_COLS>& GetRow(uint32_t row) const {
        return coefficients[row % LAGRANGE_ROWS];
    }
    
    /// Проверка загрузки
    bool IsValid() const {
        // Проверяем сумма первой строки ≈ 1.0 (свойство интерполяции)
        float sum = 0.0f;
        for (uint32_t i = 0; i < LAGRANGE_COLS; ++i) {
            sum += coefficients[0][i];
        }
        return std::abs(sum - 1.0f) < 0.01f;
    }
};

// ============================================================================
// ГЛАВНЫЙ КЛАСС - FractionalDelayProcessor
// ============================================================================

/**
 * @class FractionalDelayProcessor
 * @brief Процессор дробной задержки для LFM сигналов на GPU
 * 
 * Использует OpenCL для параллельной обработки до 256 лучей одновременно.
 * 
 * ОСОБЕННОСТИ:
 * - IN-PLACE обработка (один буфер на вход/выход)
 * - Матрица Лагранжа 48×5 из JSON
 * - 5-точечная интерполяция
 * - GPU профилирование через OpenCL events
 * - Поддержка batch обработки
 * 
 * EXAMPLE:
 * @code
 * // Инициализировать OpenCL
 * gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
 * 
 * // Загрузить матрицу Лагранжа
 * auto lagrange = LagrangeMatrix::LoadFromJSON("lagrange_matrix.json");
 * 
 * // Создать конфигурацию
 * auto config = FractionalDelayConfig::Standard();
 * config.num_beams = 64;
 * config.num_samples = 8192;
 * 
 * // Создать процессор
 * FractionalDelayProcessor processor(config, lagrange);
 * 
 * // Настроить задержки для каждого луча
 * std::vector<DelayParams> delays(64);
 * for (int i = 0; i < 64; ++i) {
 *     delays[i] = DelayParams::FromSamples(i * 0.5f);  // 0, 0.5, 1.0, ...
 * }
 * 
 * // Получить данные от GeneratorGPU
 * auto gpu_buffer = generator.signal_base();
 * 
 * // Обработать IN-PLACE
 * processor.Process(gpu_buffer, delays);
 * 
 * // Получить профилирование
 * auto prof = processor.GetLastProfiling();
 * std::cout << "Kernel time: " << prof.kernel_time_ms << " ms\n";
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
     * @param lagrange_matrix - матрица коэффициентов Лагранжа
     * @throws std::runtime_error если OpenCL не инициализирован
     */
    FractionalDelayProcessor(
        const FractionalDelayConfig& config,
        const LagrangeMatrix& lagrange_matrix
    );
    
    /// Деструктор с освобождением GPU ресурсов
    ~FractionalDelayProcessor();
    
    // Move семантика
    FractionalDelayProcessor(FractionalDelayProcessor&& other) noexcept;
    FractionalDelayProcessor& operator=(FractionalDelayProcessor&& other) noexcept;
    
    // Удаляем копирование (нельзя копировать GPU буферы)
    FractionalDelayProcessor(const FractionalDelayProcessor&) = delete;
    FractionalDelayProcessor& operator=(const FractionalDelayProcessor&) = delete;
    
    // ========================================================================
    // ОСНОВНАЯ ФУНКЦИОНАЛЬНОСТЬ
    // ========================================================================
    
    /**
     * @brief Обработка IN-PLACE с индивидуальными задержками для каждого луча
     * 
     * @param gpu_buffer - cl_mem буфер с данными (num_beams × num_samples complex)
     * @param delays - вектор параметров задержки для каждого луча
     * 
     * @note Результат записывается в тот же буфер!
     * @throws std::runtime_error при ошибке GPU
     */
    void Process(cl_mem gpu_buffer, const std::vector<DelayParams>& delays);
    
    /**
     * @brief Обработка IN-PLACE с одинаковой задержкой для всех лучей
     * 
     * @param gpu_buffer - cl_mem буфер с данными
     * @param delay - задержка (применяется ко всем лучам)
     */
    void Process(cl_mem gpu_buffer, const DelayParams& delay);
    
    /**
     * @brief Обработка с задержкой в отсчётах (float)
     * 
     * @param gpu_buffer - cl_mem буфер с данными
     * @param delay_samples - задержка в отсчётах (может быть дробной)
     */
    void ProcessWithDelay(cl_mem gpu_buffer, float delay_samples);
    
    /**
     * @brief Batch обработка - несколько буферов последовательно
     * 
     * @param buffers - вектор cl_mem буферов
     * @param all_delays - вектор векторов задержек для каждого буфера
     */
    void ProcessBatch(
        const std::vector<cl_mem>& buffers,
        const std::vector<std::vector<DelayParams>>& all_delays
    );
    
    // ========================================================================
    // УТИЛИТЫ И ДИАГНОСТИКА
    // ========================================================================
    
    /// Получить последние результаты профилирования
    const FDPProfilingResults& GetLastProfiling() const { return last_profiling_; }
    
    /// Получить строку статистики профилирования
    std::string GetProfilingStats() const;
    
    /// Вывести информацию о процессоре
    void PrintInfo() const;
    
    /// Получить конфигурацию
    const FractionalDelayConfig& GetConfig() const { return config_; }
    
    /// Обновить конфигурацию (пересоздаёт буферы)
    void UpdateConfig(const FractionalDelayConfig& new_config);
    
    /// Синхронизировать GPU (дождаться завершения всех операций)
    void SyncGPU();
    
private:
    // ========================================================================
    // ПРИВАТНЫЕ ДАННЫЕ
    // ========================================================================
    
    // Конфигурация
    FractionalDelayConfig config_;
    LagrangeMatrix lagrange_matrix_;
    gpu::OpenCLComputeEngine* engine_;
    
    // OpenCL объекты
    cl_context context_;
    cl_command_queue queue_;
    cl_device_id device_;
    cl_kernel kernel_;
    cl_program program_;
    
    // GPU буферы
    std::unique_ptr<gpu::GPUMemoryBuffer> buffer_lagrange_;  ///< Матрица Лагранжа 48×5
    std::unique_ptr<gpu::GPUMemoryBuffer> buffer_delays_;    ///< Параметры задержек для лучей
    std::unique_ptr<gpu::GPUMemoryBuffer> buffer_temp_;      ///< Временный буфер для IN-PLACE
    
    // Статистика
    FDPProfilingResults last_profiling_;
    uint64_t total_samples_processed_;
    uint32_t total_calls_;
    
    // ========================================================================
    // ПРИВАТНЫЕ МЕТОДЫ
    // ========================================================================
    
    /// Инициализировать процессор
    void Initialize();
    
    /// Загрузить и скомпилировать OpenCL kernel
    void LoadKernel();
    
    /// Создать GPU буферы
    void CreateBuffers();
    
    /// Загрузить матрицу Лагранжа на GPU
    void UploadLagrangeMatrix();
    
    /// Получить исходный код kernel'а
    static std::string GetKernelSource();
    
    /// Профилирование события OpenCL
    double ProfileEvent(cl_event event, const std::string& name);
};

} // namespace radar

#endif // FRACTIONAL_DELAY_PROCESSOR_HPP
