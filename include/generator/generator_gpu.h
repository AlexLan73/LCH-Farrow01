#pragma once

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <complex>
#include <stdexcept>
#include <string>
#include <interface/lfm_parameters.h>
#include <interface/DelayParameter.h>
// Для совместимости с LFMParameters

namespace radar {


// ═════════════════════════════════════════════════════════════════════
// GPU GENERATOR CLASS
// ═════════════════════════════════════════════════════════════════════

/**
 * @brief Генератор ЛЧМ сигналов на GPU (OpenCL)
 * 
 * Формирует базовые сигналы ЛЧМ и сигналы с дробной задержкой
 * напрямую на GPU памяти для минимизации времени работы.
 * 
 * ПАРАЛЛЕЛЬНАЯ работа:
 * - signal_base() → базовый ЛЧМ сигнал на GPU
 * - signal_valedation() → ЛЧМ сигнал с задержками из m_delay[]
 * 
 * Возвращаемые значения - ссылки на GPU память (cl_mem)
 */
class GeneratorGPU {
private:
    // OpenCL контекст и команды
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    
    // OpenCL kernels
    cl_kernel kernel_lfm_basic_;         // Базовый ЛЧМ
    cl_kernel kernel_lfm_delayed_;       // ЛЧМ с задержками
    
    // Параметры сигнала (из конструктора)
    LFMParameters params_;
    
    // Размеры данных
    size_t num_samples_;
    size_t num_beams_;
    size_t total_size_;  // num_beams * num_samples
    
    // PRIVATE METHODS
    
    /**
     * @brief Инициализация OpenCL платформы и устройства
     */
    void InitializeOpenCL();
    
    /**
     * @brief Компиляция OpenCL kernels
     */
    void CompileKernels();
    
    /**
     * @brief Получить исходный код kernels из файла или встроенный
     */
    std::string GetKernelSource() const;
    
public:
    // CONSTRUCTOR / DESTRUCTOR
    
    /**
     * @brief Конструктор с параметрами ЛЧМ
     * @param params Параметры сигнала (частоты, sample_rate, num_beams, duration)
     * @throws std::runtime_error если OpenCL инициализация не удалась
     */
    explicit GeneratorGPU(const LFMParameters& params);
    
    /**
     * @brief Деструктор - освобождение GPU ресурсов
     */
    ~GeneratorGPU();
    
    // DELETE COPY, ALLOW MOVE
    GeneratorGPU(const GeneratorGPU&) = delete;
    GeneratorGPU& operator=(const GeneratorGPU&) = delete;
    GeneratorGPU(GeneratorGPU&&) noexcept;
    GeneratorGPU& operator=(GeneratorGPU&&) noexcept;
    
    // ═════════════════════════════════════════════════════════════════
    // PUBLIC API
    // ═════════════════════════════════════════════════════════════════
    
    /**
     * @brief Сформировать БАЗОВЫЙ ЛЧМ сигнал на GPU
     * 
     * Параллельно на GPU генерирует ЛЧМ сигнал для всех лучей.
     * Сигнал записывается в GPU памяти.
     * 
     * ВХОДНЫЕ ПАРАМЕТРЫ:
     * - LFMParameters (из конструктора):
     *   ├─ f_start, f_stop (начальная/конечная частота)
     *   ├─ sample_rate (частота дискретизации)
     *   ├─ num_beams (количество лучей)
     *   └─ duration / count_points (длительность/количество отсчётов)
     * 
     * ВЫХОДНЫЕ ПАРАМЕТРЫ:
     * @return cl_mem адрес GPU памяти с базовыми сигналами
     *         Структура: [ray0_all_samples][ray1_all_samples]...[rayn_all_samples]
     *         Размер: num_beams * num_samples * sizeof(complex<float>)
     * 
     * @throws std::runtime_error если OpenCL операция не удалась
     */
    cl_mem signal_base();
    
    /**
     * @brief Сформировать ЛЧМ сигнал с ДРОБНОЙ ЗАДЕРЖКОЙ на GPU
     * 
     * Параллельно на GPU генерирует ЛЧМ сигналы с заданными задержками
     * по лучам. Все вычисления происходят на GPU.
     * 
     * ВХОДНЫЕ ПАРАМЕТРЫ:
     * - LFMParameters (из конструктора) - основные параметры сигнала
     * - m_delay[] массив DelayParameter:
     *   ├─ m_delay[0] = {beam_index: 0, delay_degrees: 0.5}
     *   ├─ m_delay[1] = {beam_index: 1, delay_degrees: 1.5}
     *   └─ m_delay[255] = {beam_index: 255, delay_degrees: ...}
     * - num_delay_params = количество элементов в m_delay[] (обычно = num_beams)
     * 
     * ВЫХОДНЫЕ ПАРАМЕТРЫ:
     * @return cl_mem адрес GPU памяти с сигналами по лучам с задержками
     *         Структура: [ray0_delayed_samples][ray1_delayed_samples]...
     *         Размер: num_beams * num_samples * sizeof(complex<float>)
     * 
     * @param m_delay Массив параметров задержки (beam_id, delay_degrees)
     * @param num_delay_params Количество элементов в m_delay (обычно num_beams)
     * 
     * @throws std::runtime_error если OpenCL операция не удалась
     * @throws std::invalid_argument если размеры параметров неверны
     */
    cl_mem signal_valedation(
        const DelayParameter* m_delay,
        size_t num_delay_params
    );
    
    /**
     * @brief Очистить GPU память
     * 
     * Освобождает все временные буферы на GPU.
     * Основные буферы результатов сохраняют адреса, но память может быть переиспользована.
     */
    void ClearGPU();
    
    // ═════════════════════════════════════════════════════════════════
    // GETTERS
    // ═════════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество лучей
     */
    size_t GetNumBeams() const noexcept { return num_beams_; }
    
    /**
     * @brief Получить количество отсчётов на луч
     */
    size_t GetNumSamples() const noexcept { return num_samples_; }
    
    /**
     * @brief Получить общее количество элементов (лучи × отсчёты)
     */
    size_t GetTotalSize() const noexcept { return total_size_; }
    
    /**
     * @brief Получить размер данных в байтах
     */
    size_t GetMemorySizeBytes() const noexcept {
        return total_size_ * sizeof(std::complex<float>);
    }
    
    /**
     * @brief Получить OpenCL контекст
     */
    cl_context GetContext() const noexcept { return context_; }
    
    /**
     * @brief Получить OpenCL очередь команд
     */
    cl_command_queue GetQueue() const noexcept { return queue_; }
    
    /**
     * @brief Получить OpenCL устройство
     */
    cl_device_id GetDevice() const noexcept { return device_; }
    
    /**
     * @brief Получить параметры ЛЧМ сигнала
     */
  
    const LFMParameters& GetParameters() const noexcept { return params_; }

    /**
     * @brief Установить углы 
     *  angle_start_deg - начальный 
     *  angle_stop_deg - конечный
     *  - если значения = 0.0f то углы считаются по формуле стар = - кол-во лучей/2,  стоп = кол-во лучей/2
     */
    void SetParametersAngle(float angle_start = 0.0f, float angle_stop = 0.0f)  { params_.SetAngle(angle_start, angle_stop);  }

    /**
     * @brief Значение начального угла
     */
    float GetAngleStart()  { return params_.angle_start_deg;  }

    /**
     * @brief Значение конечного угла
     */
    float GetAngleStop()  { return params_.angle_stop_deg;  }

    /**
     * доделать step Angl
     */
};

} // namespace radar
