#pragma once

/**
 * @file fractional_delay_processor.hpp
 * @brief Процессор дробной задержки для антенных лучей
 * 
 * Этот класс реализует обработку сигналов с дробной задержкой на GPU.
 * Используется паттерн Factory (создание буферов) и Strategy (выбор стратегии памяти).
 * 
 * Основные возможности:
 * - Генерация сигналов базовых лучей (GeneratorGPU)
 * - Применение дробной задержки через OpenCL kernel
 * - Автоматическое управление GPU памятью
 * - Чтение результатов с GPU на CPU
 * - Полная поддержка движка OpenCLComputeEngine
 * 
 * @author GPU Computing Team
 * @date 2026-01-20
 */

#include <memory>
#include <vector>
#include <complex>
#include <string>
#include <stdexcept>

// GPU компоненты (наш движок)
#include "GPU/opencl_compute_engine.hpp"
#include "GPU/kernel_program.hpp"
#include "GPU/command_queue_pool.hpp"
#include "GPU/gpu_memory_buffer.hpp"
#include "GPU/i_memory_buffer.hpp"

// Генератор сигналов
#include "generator/generator_gpu_new.h"

// Параметры
#include "interface/lfm_parameters.h"
#include "interface/DelayParameter.h"

namespace radar {

// Type aliases для удобства
using ComplexFloat = std::complex<float>;
using ComplexVector = std::vector<ComplexFloat>;
using ComplexBuffer = gpu::GPUMemoryBuffer;

// ════════════════════════════════════════════════════════════════════════════
// Structure: FractionalDelayConfig - конфигурация процессора
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct FractionalDelayConfig
 * @brief Конфигурация для процессора дробной задержки
 */
struct FractionalDelayConfig {
    /// Количество лучей (антенн)
    uint32_t num_beams = 256;
    
    /// Количество отсчётов на луч
    uint32_t num_samples = 8192;
    
    /// Размер локальной работы для GPU kernel
    uint32_t local_work_size = 256;
    
    /// Выводить ли диагностику
    bool verbose = true;
    
    /// Тип памяти для результатов
    gpu::MemoryType result_memory_type = gpu::MemoryType::GPU_READ_WRITE;
    
    /**
     * @brief Валидация конфигурации
     */
    bool IsValid() const {
        return num_beams > 0 && num_beams <= 512 &&
               num_samples >= 16 &&
               local_work_size > 0 && local_work_size <= 1024;
    }
    
    /**
     * @brief Предустановка: стандартная конфигурация
     */
    static FractionalDelayConfig Standard() {
        return FractionalDelayConfig{256, 8192, 256, false, gpu::MemoryType::GPU_READ_WRITE};
    }
    
    /**
     * @brief Предустановка: максимальная производительность
     */
    static FractionalDelayConfig Performance() {
        return FractionalDelayConfig{512, 1300000, 512, false, gpu::MemoryType::GPU_READ_ONLY};
    }
    
    /**
     * @brief Предустановка: диагностика (с выводом информации)
     */
    static FractionalDelayConfig Diagnostic() {
        return FractionalDelayConfig{256, 8192, 256, true, gpu::MemoryType::GPU_READ_WRITE};
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Structure: ProcessingResult - результаты обработки
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct ProcessingResult
 * @brief Результаты обработки с дробной задержкой
 */
struct ProcessingResult {
    /// Статус обработки (true = успех)
    bool success = false;
    
    /// Сообщение об ошибке (если есть)
    std::string error_message;
    
    /// Время выполнения на GPU (мс)
    double gpu_execution_time_ms = 0.0;
    
    /// Время чтения с GPU (мс)
    double gpu_readback_time_ms = 0.0;
    
    /// Общее время обработки (мс)
    double total_time_ms = 0.0;
    
    /// Количество обработанных лучей
    uint32_t beams_processed = 0;
    
    /// Выходные данные (num_beams x num_samples)
    ComplexVector output_data;
    
    /**
     * @brief Получить луч из результата
     * @param beam_index Индекс луча (0..num_beams-1)
     * @param num_samples Количество отсчётов на луч
     * @return Вектор одного луча
     */
    ComplexVector GetBeam(uint32_t beam_index, uint32_t num_samples) const {
        if (output_data.empty()) {
            return ComplexVector();
        }
        
        size_t start = beam_index * num_samples;
        size_t end = start + num_samples;
        
        if (end > output_data.size()) {
            return ComplexVector();
        }
        
        return ComplexVector(output_data.begin() + start, output_data.begin() + end);
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Class: FractionalDelayProcessor - главный класс
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class FractionalDelayProcessor
 * @brief Основной класс для обработки сигналов с дробной задержкой
 * 
 * Инкапсулирует:
 * - Управление GPU памятью через OpenCLComputeEngine
 * - Генерацию базовых сигналов (GeneratorGPU::signal_base)
 * - Применение дробной задержки через kernel
 * - Чтение результатов на CPU
 * 
 * Паттерны:
 * - Facade Pattern - упрощённый интерфейс к сложной архитектуре
 * - RAII - автоматическое управление ресурсами
 * - Strategy - выбор стратегии памяти (через OpenCLComputeEngine)
 * 
 * Использование:
 * @code
 * // 1. Инициализация OpenCL
 * gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
 * 
 * // 2. Создание параметров
 * LFMParameters lfm_params;
 * lfm_params.num_beams = 256;
 * lfm_params.count_points = 8192;
 * 
 * // 3. Создание конфигурации
 * FractionalDelayConfig config = FractionalDelayConfig::Standard();
 * config.num_beams = lfm_params.num_beams;
 * config.num_samples = lfm_params.count_points;
 * 
 * // 4. Создание процессора
 * FractionalDelayProcessor processor(config, lfm_params);
 * 
 * // 5. Обработка с дробной задержкой
 * DelayParameter delay{0, 0.5f}; // Луч 0 с задержкой 0.5°
 * auto result = processor.ProcessWithFractionalDelay(delay);
 * 
 * // 6. Проверка результата
 * if (result.success) {
 *     std::cout << "✅ Обработка успешна!" << std::endl;
 *     std::cout << "GPU время: " << result.gpu_execution_time_ms << " мс" << std::endl;
 *     auto beam_data = result.GetBeam(0, config.num_samples);
 * } else {
 *     std::cerr << "❌ " << result.error_message << std::endl;
 * }
 * @endcode
 */
class FractionalDelayProcessor {
public:
    
    // ═════════════════════════════════════════════════════════════════════════
    // Конструктор / Деструктор
    // ═════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать процессор дробной задержки
     * @param config Конфигурация
     * @param lfm_params Параметры LFM сигналов
     * @throws std::runtime_error если OpenCLComputeEngine не инициализирован
     * @throws std::invalid_argument если конфигурация невалидна
     */
    FractionalDelayProcessor(
        const FractionalDelayConfig& config,
        const LFMParameters& lfm_params
    );
    
    /**
     * @brief Деструктор (автоматическое освобождение ресурсов)
     */
    ~FractionalDelayProcessor();
    
    // Запрет копирования (но разрешить move)
    FractionalDelayProcessor(const FractionalDelayProcessor&) = delete;
    FractionalDelayProcessor& operator=(const FractionalDelayProcessor&) = delete;
    
    // Move конструктор и оператор присваивания
    FractionalDelayProcessor(FractionalDelayProcessor&& other) noexcept;
    FractionalDelayProcessor& operator=(FractionalDelayProcessor&& other) noexcept;
    
    // ═════════════════════════════════════════════════════════════════════════
    // Основные методы обработки
    // ═════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Обработать сигнал с дробной задержкой
     * @param delay_param Параметр задержки (луч и угол)
     * @return Результаты обработки с данными на CPU
     * 
     * Этапы обработки:
     * 1. Генерация базового сигнала через GeneratorGPU::signal_base()
     * 2. Создание буферов для результатов на GPU
     * 3. Выполнение kernel дробной задержки
     * 4. Чтение результатов с GPU на CPU
     * 5. Профилирование времени выполнения
     */
    ProcessingResult ProcessWithFractionalDelay(
        const DelayParameter& delay_param
    );
    
    /**
     * @brief Обработать несколько лучей с разными задержками (batch)
     * @param delays Вектор параметров задержек
     * @return Вектор результатов для каждого луча
     */
    std::vector<ProcessingResult> ProcessBatch(
        const std::vector<DelayParameter>& delays
    );
    
    // ═════════════════════════════════════════════════════════════════════════
    // Методы диагностики и информации
    // ═════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить конфигурацию
     */
    const FractionalDelayConfig& GetConfig() const { return config_; }
    
    /**
     * @brief Получить параметры LFM
     */
    const LFMParameters& GetLFMParameters() const { return lfm_params_; }
    
    /**
     * @brief Вывести информацию о процессоре
     */
    void PrintInfo() const;
    
    /**
     * @brief Получить статистику использования GPU
     */
    std::string GetStatistics() const;
    
    /**
     * @brief Проверить инициализацию
     */
    bool IsInitialized() const { return initialized_; }
    
    /**
     * @brief Получить размер GPU буфера в байтах
     */
    size_t GetGPUBufferSizeBytes() const;
    
private:
    
    // ═════════════════════════════════════════════════════════════════════════
    // Приватные методы
    // ═════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализация: загрузка kernel'а и создание буферов
     */
    void Initialize();
    
    /**
     * @brief Загрузить OpenCL kernel для дробной задержки
     */
    void LoadKernels();
    
    /**
     * @brief Создать GPU буферы для обработки
     */
    void CreateBuffers();
    
    /**
     * @brief Очистить GPU ресурсы перед обработкой
     */
    void SyncGPU();
    
    /**
     * @brief Получить исходный код kernel'а дробной задержки
     */
    std::string GetKernelSource() const;
    
    // ═════════════════════════════════════════════════════════════════════════
    // Члены класса
    // ═════════════════════════════════════════════════════════════════════════
    
    /// Конфигурация
    FractionalDelayConfig config_;
    
    /// Параметры LFM сигналов
    LFMParameters lfm_params_;
    
    /// Флаг инициализации
    bool initialized_;
    
    /// Указатель на OpenCLComputeEngine (не владеем)
    gpu::OpenCLComputeEngine* engine_;
    
    /// Генератор сигналов
    std::unique_ptr<GeneratorGPU> signal_generator_;
    
    /// OpenCL программа (ядро дробной задержки)
    std::shared_ptr<gpu::KernelProgram> kernel_program_;
    
    /// OpenCL kernel
    cl_kernel kernel_fractional_delay_;
    
    /// Буфер для входных данных (базовый сигнал с GPU)
    std::unique_ptr<gpu::IMemoryBuffer> buffer_input_;
    
    /// Буфер для выходных данных (результаты задержки)
    std::unique_ptr<gpu::IMemoryBuffer> buffer_output_;
    
    /// Статистика обработки
    struct {
        uint64_t total_processed = 0;
        uint64_t total_gpu_time_ms = 0;
        uint64_t total_readback_time_ms = 0;
    } stats_;
};

} // namespace radar
