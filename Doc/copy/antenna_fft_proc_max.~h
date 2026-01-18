#pragma once

#include "interface/antenna_fft_params.h"
#include "GPU/opencl_compute_engine.hpp"
#include "GPU/opencl_core.hpp"
#include "GPU/command_queue_pool.hpp"
#include "GPU/gpu_memory_buffer.hpp"
#include <CL/cl.h>
#include <clFFT.h>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <unordered_map>
#include <chrono>
#include <mutex>

namespace antenna_fft {

/**
 * @class AntennaFFTProcMax
 * @brief Высокопроизводительный класс для FFT обработки с поиском максимальных амплитуд
 * 
 * Использует clFFT с callback'ами для максимальной производительности.
 * Все операции поиска максимумов выполняются ТОЛЬКО на GPU.
 * 
 * АРХИТЕКТУРА:
 * - Использует OpenCLComputeEngine для доступа к контексту и очередям
 * - Кэширует clFFT план для переиспользования
 * - Pre-callback: подготовка данных (перенос + padding)
 * - Post-callback: fftshift + вычисление magnitude/phase
 * - Reduction kernel: поиск топ-N максимумов на GPU
 * - Детальное профилирование всех операций
 * 
 * ИСПОЛЬЗОВАНИЕ:
 * ```cpp
 * // Инициализация OpenCL (один раз)
 * gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
 * 
 * // Создание процессора
 * AntennaFFTParams params(5, 1000, 512, 3); // 5 лучей, 1000 точек, 512 выходных точек, 3 максимума
 * AntennaFFTProcMax processor(params);
 * 
 * // Обработка данных
 * cl_mem input_signal = ...; // Входной сигнал на GPU
 * AntennaFFTResult result = processor.Process(input_signal);
 * 
 * // Вывод результатов
 * processor.PrintResults(result);
 * processor.SaveResultsToFile(result, "Reports/result.md");
 * ```
 */
class AntennaFFTProcMax {
public:
    /**
     * @brief Конструктор
     * @param params Параметры обработки (beam_count, count_points, out_count_points_fft, max_peaks_count)
     * @throws std::runtime_error если OpenCLComputeEngine не инициализирован или параметры невалидны
     */
    explicit AntennaFFTProcMax(const AntennaFFTParams& params);
    
    /**
     * @brief Деструктор
     * Освобождает clFFT план и буферы
     */
    ~AntennaFFTProcMax();
    
    // DELETE COPY, ALLOW MOVE
    AntennaFFTProcMax(const AntennaFFTProcMax&) = delete;
    AntennaFFTProcMax& operator=(const AntennaFFTProcMax&) = delete;
    AntennaFFTProcMax(AntennaFFTProcMax&&) noexcept;
    AntennaFFTProcMax& operator=(AntennaFFTProcMax&&) noexcept;
    
    /**
     * @brief Основной метод обработки FFT
     * @param input_signal GPU буфер с входными комплексными данными (beam_count * count_points элементов)
     * @return AntennaFFTResult с результатами для всех лучей
     * @throws std::runtime_error если обработка не удалась
     */
    AntennaFFTResult Process(cl_mem input_signal);
    
    /**
     * @brief Обработка FFT с входными данными из CPU
     * @param input_data Вектор комплексных чисел (beam_count * count_points элементов)
     * @return AntennaFFTResult с результатами для всех лучей
     */
    AntennaFFTResult Process(const std::vector<std::complex<float>>& input_data);
    
    /**
     * @brief Вывести результаты в консоль (таблица)
     * @param result Результаты обработки
     */
    void PrintResults(const AntennaFFTResult& result) const;
    
    /**
     * @brief Сохранить результаты в файл (таблица + JSON)
     * @param result Результаты обработки
     * @param filepath Путь к файлу (будет создан в Reports/ если путь относительный)
     */
    void SaveResultsToFile(const AntennaFFTResult& result, const std::string& filepath);
    
    /**
     * @brief Получить статистику профилирования последней операции
     * @return Строка с детальной статистикой
     */
    std::string GetProfilingStats() const;
    
    /**
     * @brief Получить вычисленный размер nFFT
     */
    size_t GetNFFT() const { return nFFT_; }
    
    /**
     * @brief Получить вычисленный размер nFFT (алиас для GetNFFT)
     */
    size_t GetNFFTSize() const { return nFFT_; }
    
    /**
     * @brief Получить последние результаты профилирования
     */
    const FFTProfilingResults& GetLastProfilingResults() const;
    
    /**
     * @brief Обновить параметры (пересоздаст план FFT если нужно)
     * @param params Новые параметры
     */
    void UpdateParams(const AntennaFFTParams& params);

private:
    // ═══════════════════════════════════════════════════════════════
    // Внутренние методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Вычислить nFFT из count_points
     * Проверяет кратность 2^n, дополняет до ближайшего большего, умножает на 2
     */
    size_t CalculateNFFT(size_t count_points) const;
    
    /**
     * @brief Проверить является ли число степенью двойки
     */
    bool IsPowerOf2(size_t n) const;
    
    /**
     * @brief Найти ближайшую большую степень двойки
     */
    size_t NextPowerOf2(size_t n) const;
    
    /**
     * @brief Создать или переиспользовать clFFT план
     */
    void CreateOrReuseFFTPlan();
    
    /**
     * @brief Освободить clFFT план
     */
    void ReleaseFFTPlan();

    /**
     * @brief Создать pre-callback строку для clFFT
     */
    std::string GetPreCallbackSource() const;

    /**
     * @brief Создать post-callback строку для clFFT
     */
    std::string GetPostCallbackSource() const;

    /**
     * @brief Создать reduction kernel для поиска максимумов
     */
    void CreateMaxReductionKernel();
    
    /**
     * @brief Выполнить поиск максимумов на GPU
     * @param fft_output Буфер с результатами FFT
     * @param beam_idx Индекс луча
     * @return Вектор FFTMaxResult с найденными максимумами
     */
    std::vector<std::vector<FFTMaxResult>> FindMaximaAllBeamsOnGPU();
    
    /**
     * @brief Профилировать событие OpenCL
     * @param event Событие OpenCL
     * @param operation_name Название операции
     * @return Время выполнения в миллисекундах
     */
    double ProfileEvent(cl_event event, const std::string& operation_name);
    
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    AntennaFFTParams params_;              // Параметры обработки
    size_t nFFT_;                          // Вычисленный размер FFT
    
    // OpenCL ресурсы
    gpu::OpenCLComputeEngine* engine_;     // Указатель на engine (не владеем)
    cl_context context_;                   // OpenCL контекст
    cl_command_queue queue_;               // Command queue
    cl_device_id device_;                 // OpenCL устройство
    
    // clFFT ресурсы
    clfftPlanHandle plan_handle_;         // Handle плана FFT
    bool plan_created_;                    // Флаг создания плана
    
    // Буферы GPU (persistent для переиспользования)
    std::unique_ptr<gpu::GPUMemoryBuffer> buffer_input_;      // Входной буфер (может быть внешним)
    std::unique_ptr<gpu::GPUMemoryBuffer> buffer_fft_input_;  // Буфер для FFT (nFFT * beam_count)
    std::unique_ptr<gpu::GPUMemoryBuffer> buffer_fft_output_; // Буфер для результатов FFT
    std::unique_ptr<gpu::GPUMemoryBuffer> buffer_magnitude_;  // Буфер для magnitude (после post-callback)
    std::unique_ptr<gpu::GPUMemoryBuffer> buffer_maxima_;     // Буфер для максимумов (после reduction)
    
    
    // Userdata буферы для callback'ов
    cl_mem pre_callback_userdata_;         // Userdata для pre-callback
    cl_mem post_callback_userdata_;        // Userdata для post-callback

    // Reduction kernel
    std::shared_ptr<gpu::KernelProgram> reduction_program_;  // Программа для reduction
    cl_kernel reduction_kernel_;           // Kernel для поиска максимумов
    
    // Профилирование
    struct ProfilingData {
        double upload_time_ms;
        double pre_callback_time_ms;
        double fft_time_ms;
        double post_callback_time_ms;
        double reduction_time_ms;
        double download_time_ms;
        double total_time_ms;
    };
    ProfilingData last_profiling_;
    
    // Кэш для планов FFT (ключ: hash параметров)
    struct PlanCacheKey {
        size_t beam_count;
        size_t count_points;
        size_t nFFT;
        size_t out_count_points_fft;
        size_t max_peaks_count;
        
        bool operator==(const PlanCacheKey& other) const {
            return beam_count == other.beam_count &&
                   count_points == other.count_points &&
                   nFFT == other.nFFT &&
                   out_count_points_fft == other.out_count_points_fft &&
                   max_peaks_count == other.max_peaks_count;
        }
    };
    
    struct PlanCacheKeyHash {
        size_t operator()(const PlanCacheKey& key) const {
            return std::hash<size_t>()(key.beam_count) ^
                   (std::hash<size_t>()(key.count_points) << 1) ^
                   (std::hash<size_t>()(key.nFFT) << 2) ^
                   (std::hash<size_t>()(key.out_count_points_fft) << 3) ^
                   (std::hash<size_t>()(key.max_peaks_count) << 4);
        }
    };
    
    static std::unordered_map<PlanCacheKey, clfftPlanHandle, PlanCacheKeyHash> plan_cache_;
    static std::mutex plan_cache_mutex_;
};

} // namespace antenna_fft

