#pragma once

#include <CL/cl.h>
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

// Forward declarations
namespace gpu {
    class OpenCLComputeEngine;
    class KernelProgram;
    class GPUMemoryBuffer;
}

struct LFMParameters;
struct DelayParameter;

namespace radar {

/**
 * @class GeneratorGPU
 * @brief GPU генератор ЛЧМ сигналов (ПЕРЕДЕЛАННЫЙ под новую архитектуру)
 * 
 * АРХИТЕКТУРА:
 * ├─ OpenCLCore       (единый контекст OpenCL) 
 * ├─ CommandQueuePool (пул асинхронных очередей 4+)
 * ├─ KernelProgram    (компилированные программы с кэшем)
 * ├─ GPUMemoryBuffer  (обёртка над GPU памятью)
 * └─ OpenCLComputeEngine (главный фасад)
 * 
 * Два основных kernel'а:
 * 1. kernel_lfm_basic() → базовый ЛЧМ сигнал (без задержек)
 * 2. kernel_lfm_delayed() → ЛЧМ сигнал с дробной задержкой
 * 
 * ИСПОЛЬЗОВАНИЕ:
 * ```cpp
 * // Инициализация (один раз в main)
 * gpu::OpenCLCore::Initialize(gpu::DeviceType::GPU);
 * gpu::CommandQueuePool::Initialize(4);
 * gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
 * 
 * // Использование
 * LFMParameters params;
 * params.f_start = 100.0f;
 * params.f_stop = 500.0f;
 * params.sample_rate = 12.0e6f;
 * params.num_beams = 256;
 * params.count_points = 1024 * 8;
 * 
 * radar::GeneratorGPU gen(params);
 * 
 * // Генерация базового сигнала
 * cl_mem signal_gpu = gen.signal_base();
 * 
 * // Прочитать результаты (через engine)
 * auto& engine = gpu::OpenCLComputeEngine::GetInstance();
 * auto result = engine.ReadBufferFromGPU(signal_gpu, total_size);
 * ```
 */
class GeneratorGPU {
public:
    // ════════════════════════════════════════════════════════════════
    // CONSTRUCTOR / DESTRUCTOR
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief Конструктор с параметрами ЛЧМ
     * @param params Параметры сигнала (частоты, sample_rate, num_beams, duration/count_points)
     * @throws std::runtime_error если параметры невалидны или OpenCLComputeEngine не инициализирован
     */
    explicit GeneratorGPU(const LFMParameters& params);

    /**
     * @brief Деструктор
     * ✅ Ресурсы управляются OpenCLComputeEngine, не нужно очищать вручную
     */
    ~GeneratorGPU();

    // DELETE COPY, ALLOW MOVE
    GeneratorGPU(const GeneratorGPU&) = delete;
    GeneratorGPU& operator=(const GeneratorGPU&) = delete;

    GeneratorGPU(GeneratorGPU&&) noexcept;
    GeneratorGPU& operator=(GeneratorGPU&&) noexcept;

    // ════════════════════════════════════════════════════════════════
    // PUBLIC API - ОСНОВНЫЕ ФУНКЦИИ
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief Сформировать БАЗОВЫЙ ЛЧМ сигнал на GPU
     * 
     * Параллельно на GPU генерирует ЛЧМ сигнал для всех лучей.
     * Использует kernel_lfm_basic() из OpenCL программы.
     * 
     * ВХОДНЫЕ ПАРАМЕТРЫ:
     * - LFMParameters (из конструктора):
     *   ├─ f_start, f_stop     (начальная/конечная частота)
     *   ├─ sample_rate         (частота дискретизации)
     *   ├─ num_beams           (количество лучей)
     *   └─ duration/count_points (длительность/количество отсчётов)
     * 
     * ВЫХОДНЫЕ ДАННЫЕ:
     * @return cl_mem GPU адрес памяти с базовыми сигналами
     * 
     * СТРУКТУРА В ПАМЯТИ:
     * [ray0_sample0, ray0_sample1, ..., ray0_sampleN,
     *  ray1_sample0, ray1_sample1, ..., ray1_sampleN,
     *  ...
     *  rayM_sample0, rayM_sample1, ..., rayM_sampleN]
     * 
     * РАЗМЕР: num_beams × num_samples × sizeof(complex<float>) байт
     * 
     * @throws std::runtime_error если OpenCL операция не удалась
     */
    cl_mem signal_base();

    /**
     * @brief Сформировать ЛЧМ сигнал с ДРОБНОЙ ЗАДЕРЖКОЙ на GPU
     * 
     * Параллельно на GPU генерирует ЛЧМ сигналы с заданными дробными
     * задержками по лучам. Использует kernel_lfm_delayed() из OpenCL программы.
     * 
     * ВХОДНЫЕ ПАРАМЕТРЫ:
     * - LFMParameters (из конструктора) - основные параметры сигнала
     * - m_delay[] массив DelayParameter (размер = num_beams):
     *   ├─ m_delay[0]   = {beam_index: 0,   delay_degrees: 0.5}
     *   ├─ m_delay[1]   = {beam_index: 1,   delay_degrees: 1.5}
     *   └─ m_delay[255] = {beam_index: 255, delay_degrees: 64.5}
     * - num_delay_params количество элементов в m_delay[] (обычно = num_beams)
     * 
     * ВЫХОДНЫЕ ДАННЫЕ:
     * @return cl_mem GPU адрес памяти с задержанными сигналами
     * 
     * СТРУКТУРА В ПАМЯТИ: Как signal_base(), но с применённой задержкой
     * РАЗМЕР: num_beams × num_samples × sizeof(complex<float>) байт
     * 
     * ПАРАМЕТРЫ:
     * @param m_delay Массив параметров задержки (должен быть на CPU!)
     * @param num_delay_params Количество элементов (должно быть = num_beams)
     * 
     * @throws std::runtime_error если OpenCL операция не удалась
     * @throws std::invalid_argument если параметры невалидны
     */
    cl_mem signal_valedation(
        const DelayParameter* m_delay,
        size_t num_delay_params
    );

    /**
     * @brief Очистить GPU память (синхронизировать очереди)
     * 
     * Вызывает Finish() на всех command queues.
     * Используйте перед чтением результатов с GPU.
     */
    void ClearGPU();

    // ════════════════════════════════════════════════════════════════
    // GETTERS - ИНФОРМАЦИЯ О СИГНАЛЕ
    // ════════════════════════════════════════════════════════════════

    /// Получить количество лучей
    size_t GetNumBeams() const noexcept { return num_beams_; }

    /// Получить количество отсчётов на луч
    size_t GetNumSamples() const noexcept { return num_samples_; }

    /// Получить общее количество элементов (лучи × отсчёты)
    size_t GetTotalSize() const noexcept { return total_size_; }

    /// Получить размер данных в байтах (для выделения памяти на CPU)
    size_t GetMemorySizeBytes() const noexcept {
        return total_size_ * sizeof(std::complex<float>);
    }

    /// Получить параметры ЛЧМ сигнала (const ссылка)
    const LFMParameters& GetParameters() const noexcept { return params_; }

    /// Получить начальный угол (градусы)
    float GetAngleStart() const noexcept { return params_.angle_start_deg; }

    /// Получить конечный угол (градусы)
    float GetAngleStop() const noexcept { return params_.angle_stop_deg; }

    /// Получить шаг по углу (градусы)
    float GetAngleStep() const noexcept { return params_.angle_step_deg; }

    /// Установить углы
    void SetParametersAngle(float angle_start = 0.0f, float angle_stop = 0.0f);

private:
    // ════════════════════════════════════════════════════════════════
    // PRIVATE MEMBERS - СОСТОЯНИЕ ГЕНЕРАТОРА
    // ════════════════════════════════════════════════════════════════

    /// ✅ Указатель на главный фасад (НЕ создаём свой контекст!)
    gpu::OpenCLComputeEngine* engine_;

    /// Параметры ЛЧМ сигнала
    LFMParameters params_;

    /// Размеры данных (кэш для быстрого доступа)
    size_t num_samples_;   // Количество отсчётов на луч
    size_t num_beams_;     // Количество лучей
    size_t total_size_;    // num_beams * num_samples

    /// Кэшированные программы и kernels
    std::shared_ptr<gpu::KernelProgram> kernel_program_;
    cl_kernel kernel_lfm_basic_;      // kernel_lfm_basic
    cl_kernel kernel_lfm_delayed_;    // kernel_lfm_delayed

    /// Буферы результатов (кэш)
    cl_mem buffer_signal_base_;     // Результат signal_base()
    cl_mem buffer_signal_delayed_;  // Результат signal_valedation()

    // ════════════════════════════════════════════════════════════════
    // PRIVATE METHODS - ИНИЦИАЛИЗАЦИЯ И УТИЛИТЫ
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief Инициализировать (получить контекст из engine)
     * ✅ Больше не создаём свой контекст - берём из OpenCLComputeEngine!
     */
    void Initialize();

    /**
     * @brief Загрузить и скомпилировать kernels
     * ✅ Использует engine->LoadProgram() с кэшем
     */
    void LoadKernels();

    /**
     * @brief Получить исходный код OpenCL kernels
     * @return Строка с исходником OpenCL C кода
     */
    std::string GetKernelSource() const;

    /**
     * @brief Выполнить kernel в GPU
     * @param kernel Compiled kernel
     * @param output_buffer GPU адрес выходного буфера
     * @param delay_buffer (опционально) GPU адрес буфера задержек
     */
    void ExecuteKernel(
        cl_kernel kernel,
        cl_mem output_buffer,
        cl_mem delay_buffer = nullptr
    );
};

} // namespace radar
