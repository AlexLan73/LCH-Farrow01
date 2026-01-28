#include "GPU/generator_gpu_new.h"

// Включаем компоненты новой архитектуры
#include "ManagerOpenCL/opencl_compute_engine.hpp"
#include "ManagerOpenCL/kernel_program.hpp"
#include "ManagerOpenCL/command_queue_pool.hpp"
#include "ManagerOpenCL/opencl_core.hpp"

// Параметры сигнала
#include "interface/lfm_parameters.h"
#include "interface/DelayParameter.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <CL/cl.h>
#include <algorithm>

// Структура для передачи параметров синусоид в OpenCL (должна совпадать с OpenCL структурой)
struct RaySinusoidParams {
    cl_uint ray_index;     // Номер луча
    cl_uint num_sinusoids; // Количество синусоид для этого луча
    struct SinusoidParam {
        float amplitude;    // Амплитуда
        float period;       // Период в точках
        float phase_deg;    // Фаза в градусах
    } sinusoids[10];        // Максимум 10 синусоид на луч
};

namespace radar
{

  // ════════════════════════════════════════════════════════════════════════════
  // CONSTRUCTOR / DESTRUCTOR
  // ════════════════════════════════════════════════════════════════════════════

  GeneratorGPU::GeneratorGPU(const LFMParameters &params)
      : engine_(nullptr),
        params_(params),
        num_samples_(0),
        num_beams_(params.num_beams),
        total_size_(0),
        kernel_program_(nullptr),
        kernel_lfm_basic_(nullptr),
        kernel_lfm_delayed_(nullptr),
        kernel_lfm_combined_(nullptr),
        kernel_sinusoid_combined_(nullptr),
        buffer_signal_base_(nullptr),
        buffer_signal_delayed_(nullptr),
        buffer_signal_combined_(nullptr),
        buffer_signal_sinusoid_(nullptr)
  {

    // ✅ Валидировать параметры
    if (!params_.IsValid())
    {
      throw std::invalid_argument(
          "[GeneratorGPU] LFMParameters invalid: "
          "check f_start, f_stop, sample_rate, num_beams, duration/count_points");
    }

    // ✅ Получить engine (ДОЛЖЕН быть инициализирован!)
    try
    {
      engine_ = &ManagerOpenCL::OpenCLComputeEngine::GetInstance();
    }
    catch (const std::exception &e)
    {
      throw std::runtime_error(
          "[GeneratorGPU] OpenCLComputeEngine not initialized.\n"
          "Call ManagerOpenCL::OpenCLCore::Initialize() → CommandQueuePool::Initialize() → "
          "OpenCLComputeEngine::Initialize() before creating GeneratorGPU");
    }

    // ✅ Инициализировать (получить контекст из engine)
    try
    {
      Initialize();
      LoadKernels();
    }
    catch (const std::exception &e)
    {
      throw std::runtime_error(
          std::string("[GeneratorGPU] Initialization failed: ") + e.what());
    }

    std::cout << "[GeneratorGPU] ✅ Created successfully" << std::endl;
    std::cout << "  - Beams: " << num_beams_ << std::endl;
    std::cout << "  - Samples per beam: " << num_samples_ << std::endl;
    std::cout << "  - Total size: " << total_size_ << " elements ("
              << (GetMemorySizeBytes() / (1024 * 1024)) << " MB)" << std::endl;
  }

  GeneratorGPU::~GeneratorGPU()
  {
    // ✅ ВАЖНО: Ресурсы управляются OpenCLComputeEngine
    // Не вызываем clRelease* - engine сам управляет жизненным циклом
    // Просто обнуляем указатели

    kernel_lfm_basic_ = nullptr;
    kernel_lfm_delayed_ = nullptr;
    kernel_program_ = nullptr;
    buffer_signal_base_.reset(); // Освободит unique_ptr (буфер будет освобожден автоматически)
    buffer_signal_delayed_.reset();
    engine_ = nullptr;
    kernel_lfm_combined_ = nullptr;
    buffer_signal_combined_.reset();
    kernel_sinusoid_combined_ = nullptr;
    buffer_signal_sinusoid_.reset();

    std::cout << "[GeneratorGPU] ✅ Destroyed" << std::endl;
  }

  // Move семантика
  GeneratorGPU::GeneratorGPU(GeneratorGPU &&other) noexcept
      : engine_(other.engine_),
        params_(other.params_),
        num_samples_(other.num_samples_),
        num_beams_(other.num_beams_),
        total_size_(other.total_size_),
        kernel_program_(std::move(other.kernel_program_)),
        kernel_lfm_basic_(other.kernel_lfm_basic_),
        kernel_lfm_delayed_(other.kernel_lfm_delayed_),
        buffer_signal_base_(std::move(other.buffer_signal_base_)),
        buffer_signal_delayed_(std::move(other.buffer_signal_delayed_)),
        kernel_lfm_combined_(other.kernel_lfm_combined_),
        buffer_signal_combined_(std::move(other.buffer_signal_combined_)),
        kernel_sinusoid_combined_(other.kernel_sinusoid_combined_),
        buffer_signal_sinusoid_(std::move(other.buffer_signal_sinusoid_))
  {

    other.engine_ = nullptr;
    other.kernel_lfm_basic_ = nullptr;
    other.kernel_lfm_delayed_ = nullptr;
    other.buffer_signal_base_.reset();
    other.buffer_signal_delayed_.reset();
    other.kernel_lfm_combined_ = nullptr;
    other.buffer_signal_combined_.reset();
    other.kernel_sinusoid_combined_ = nullptr;
    other.buffer_signal_sinusoid_.reset();
  }

  GeneratorGPU &GeneratorGPU::operator=(GeneratorGPU &&other) noexcept
  {
    if (this != &other)
    {
      // Очистить текущие ресурсы
      kernel_lfm_basic_ = nullptr;
      kernel_lfm_delayed_ = nullptr;
      kernel_program_ = nullptr;
      buffer_signal_base_.reset();
      buffer_signal_delayed_.reset();
      kernel_lfm_combined_ = nullptr;
      buffer_signal_combined_.reset();
      kernel_sinusoid_combined_ = nullptr;
      buffer_signal_sinusoid_.reset();

      // Переместить от other
      engine_ = other.engine_;
      params_ = other.params_;
      num_samples_ = other.num_samples_;
      num_beams_ = other.num_beams_;
      total_size_ = other.total_size_;
      kernel_program_ = std::move(other.kernel_program_);
      kernel_lfm_basic_ = other.kernel_lfm_basic_;
      kernel_lfm_delayed_ = other.kernel_lfm_delayed_;
      buffer_signal_base_ = std::move(other.buffer_signal_base_);
      buffer_signal_delayed_ = std::move(other.buffer_signal_delayed_);
      kernel_lfm_combined_ = other.kernel_lfm_combined_;
      buffer_signal_combined_ = std::move(other.buffer_signal_combined_);
      kernel_sinusoid_combined_ = other.kernel_sinusoid_combined_;
      buffer_signal_sinusoid_ = std::move(other.buffer_signal_sinusoid_);

      // Обнулить в other
      other.engine_ = nullptr;
      other.kernel_lfm_basic_ = nullptr;
      other.kernel_lfm_delayed_ = nullptr;
      other.buffer_signal_base_.reset();
      other.buffer_signal_delayed_.reset();
      other.kernel_lfm_combined_ = nullptr;
      other.buffer_signal_combined_.reset();
      other.kernel_sinusoid_combined_ = nullptr;
      other.buffer_signal_sinusoid_.reset();
    }
    return *this;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // PRIVATE METHODS
  // ════════════════════════════════════════════════════════════════════════════

  void GeneratorGPU::Initialize()
  {
    // ✅ Рассчитать num_samples из duration или count_points
    if (params_.count_points > 0)
    {
      num_samples_ = params_.count_points;
      params_.duration = static_cast<float>(num_samples_) / params_.sample_rate;
    }
    else if (params_.duration > 0.0f)
    {
      num_samples_ = static_cast<size_t>(params_.duration * params_.sample_rate);
      params_.count_points = num_samples_;
    }
    else
    {
      throw std::invalid_argument(
          "[GeneratorGPU] Either count_points or duration must be > 0");
    }

    // ✅ Рассчитать общий размер
    total_size_ = num_beams_ * num_samples_;

    std::cout << "[GeneratorGPU] Initialized:" << std::endl;
    std::cout << "  - Duration: " << params_.duration * 1e6 << " us" << std::endl;
    std::cout << "  - Num samples: " << num_samples_ << std::endl;
    std::cout << "  - Total size: " << total_size_ << std::endl;
  }

  void GeneratorGPU::LoadKernels()
  {
    // ✅ Получить исходный код
    std::string source = GetKernelSource();

    std::cout << "[GeneratorGPU] Loading kernels from GPU engine..." << std::endl;

    // ✅ Получить или скомпилировать программу (с кэшем!)
    kernel_program_ = engine_->LoadProgram(source);
    if (!kernel_program_)
    {
      throw std::runtime_error("[GeneratorGPU] Failed to load kernel program");
    }

    // ✅ Получить или создать kernels
    kernel_lfm_basic_ = engine_->GetKernel(kernel_program_, "kernel_lfm_basic");
    if (!kernel_lfm_basic_)
    {
      throw std::runtime_error("[GeneratorGPU] Failed to create kernel_lfm_basic");
    }

    kernel_lfm_delayed_ = engine_->GetKernel(kernel_program_, "kernel_lfm_delayed");
    if (!kernel_lfm_delayed_)
    {
      throw std::runtime_error("[GeneratorGPU] Failed to create kernel_lfm_delayed");
    }

    kernel_lfm_combined_ = engine_->GetKernel(kernel_program_, "kernel_lfm_combined");
    if (!kernel_lfm_combined_)
    {
      throw std::runtime_error("Failed to create kernel_lfm_combined");
    }

    kernel_sinusoid_combined_ = engine_->GetKernel(kernel_program_, "kernel_sinusoid_combined");
    if (!kernel_sinusoid_combined_)
    {
      throw std::runtime_error("[GeneratorGPU] Failed to create kernel_sinusoid_combined");
    }

    std::cout << "[GeneratorGPU] ✅ Kernels loaded successfully" << std::endl;
  }

  std::string GeneratorGPU::GetKernelSource() const
  {
    // ✅ Встроенный OpenCL C код с правильной структурой
    return R"(
// ═════════════════════════════════════════════════════════════════════════
// СТРУКТУРЫ (должны быть в начале!)
// ═════════════════════════════════════════════════════════════════════════

typedef struct {
    uint beam_index;
    float delay_degrees;
} DelayParam;

typedef struct {
    float delay_degrees;
    float delay_time_ns;
} CombinedDelayParam;

typedef struct {
    float amplitude;    // Амплитуда
    float period;       // Период в точках
    float phase_deg;    // Фаза в градусах
} SinusoidParam;

// Структура для передачи параметров синусоидов для каждого луча
typedef struct {
    uint ray_index;     // Номер луча
    uint num_sinusoids; // Количество синусоид для этого луча
    SinusoidParam sinusoids[10]; // Максимум 10 синусоид на луч (достаточно для большинства случаев)
} RaySinusoidParams;

// ═════════════════════════════════════════════════════════════════════════
// KERNEL 1: БАЗОВЫЙ ЛЧМ СИГНАЛ (БЕЗ ЗАДЕРЖЕК)
// ═════════════════════════════════════════════════════════════════════════

__kernel void kernel_lfm_basic(
    __global float2 *output,      // [ray0][ray1]...[rayn] - выходные сигналы
    float f_start,                // Начальная частота (Гц)
    float f_stop,                 // Конечная частота (Гц)
    float sample_rate,            // Частота дискретизации (Гц)
    float duration,               // Длительность сигнала (сек)
    uint num_samples,             // Количество отсчётов на луч
    uint num_beams               // Количество лучей
) {
    uint gid = get_global_id(0);  // Глобальный индекс потока
    
    // Проверка границ
    if (gid >= (uint)num_samples * num_beams) return;
    
    // Распределить работу: каждый поток обрабатывает один элемент
    uint ray_id = gid / num_samples;     // Номер луча (0...255)
    uint sample_id = gid % num_samples;  // Номер отсчёта (0...N)
    
    if (ray_id >= num_beams || sample_id >= num_samples) return;
    
    // ✅ Время для этого отсчёта (одинаково для всех лучей)
    float t = (float)sample_id / sample_rate;
    
    // ✅ Параметры ЛЧМ (Linear Frequency Modulation)
    float chirp_rate = (f_stop - f_start) / duration;
    
    // ✅ Фаза: φ(t) = 2π(f_start * t + 0.5 * chirp_rate * t²)
    float phase = 2.0f * 3.14159265f * (
        f_start * t + 0.5f * chirp_rate * t * t
    );
    
    // ✅ Комплексный сигнал: exp(iφ) = cos(φ) + i*sin(φ)
    float real = cos(phase);
    float imag = sin(phase);
    
    // ✅ Записать результат в GPU память
    uint out_idx = ray_id * num_samples + sample_id;
    output[out_idx] = (float2)(real, imag);
}

// ═════════════════════════════════════════════════════════════════════════
// KERNEL 2: ЛЧМ СИГНАЛ С ДРОБНОЙ ЗАДЕРЖКОЙ
// ═════════════════════════════════════════════════════════════════════════

__kernel void kernel_lfm_delayed(
    __global float2 *output,           // Выходные сигналы с задержкой
    __global const DelayParam *delays, // ✅ __global вместо __constant!
    float f_start,                     // Начальная частота (Гц)
    float f_stop,                      // Конечная частота (Гц)
    float sample_rate,                 // Частота дискретизации (Гц)
    float duration,                    // Длительность сигнала (сек)
    float speed_of_light,              // Скорость света (м/с)
    uint num_samples,                  // Количество отсчётов на луч
    uint num_beams,                   // Количество лучей
    uint num_delays                    // Количество параметров задержки
) {
    uint gid = get_global_id(0);
    
    if (gid >= (uint)num_samples * num_beams) return;
    
    uint ray_id = gid / num_samples;
    uint sample_id = gid % num_samples;
    
    if (ray_id >= num_beams || sample_id >= num_samples) return;
    
    // ✅ Получить задержку для этого луча
    float delay_degrees = delays[ray_id].delay_degrees;
    
    // ✅ Конвертировать градусы в секунды задержки
    // delay_rad = delay_degrees * π / 180
    // delay_time = delay_rad * wavelength / speed_of_light
    float f_center = (f_start + f_stop) / 2.0f;
    float wavelength = speed_of_light / f_center;
    float delay_rad = delay_degrees * 3.14159265f / 180.0f;
    float delay_time = delay_rad * wavelength / speed_of_light;
    float delay_samples = delay_time * sample_rate;
    
    // ✅ Вычислить эффективный индекс с задержкой
    int delayed_sample_int = (int)sample_id - (int)delay_samples;
    
    float real, imag;
    
    if (delayed_sample_int < 0) {
        // До начала сигнала - нули
        real = 0.0f;
        imag = 0.0f;
    } else {
        // ✅ Время для задержанного отсчёта
        float t = (float)delayed_sample_int / sample_rate;
        
        // ✅ Параметры ЛЧМ
        float chirp_rate = (f_stop - f_start) / duration;
        
        // ✅ Фаза
        float phase = 2.0f * 3.14159265f * (
            f_start * t + 0.5f * chirp_rate * t * t
        );
        
        real = cos(phase);
        imag = sin(phase);
    }
    
    // ✅ Записать результат
    uint out_idx = ray_id * num_samples + sample_id;
    output[out_idx] = (float2)(real, imag);
}
// ═════════════════════════════════════════════════════════════════════════════════════════
// KERNEL 3: ЛЧМ СИГНАЛ С ДРОБНОЙ ЗАДЕРЖКОЙ ПО КОМБИНИРОВАННОМУ ПАРАМЕТРУ ВРЕМЕНИ И УГЛУ
// ═════════════════════════════════════════════════════════════════════════════════════════

__kernel void kernel_lfm_combined(
    __global float2 *output,
    __global const CombinedDelayParam *combined,
    float f_start, float f_stop, float sample_rate,
    float duration, float speed_of_light,
    uint num_samples, uint num_beams, uint num_delays
) {
    uint gid = get_global_id(0);
    if (gid >= (uint)num_samples * num_beams) return;
    
    uint ray_id = gid / num_samples;
    uint sample_id = gid % num_samples;
    
    float delay_degrees = combined[ray_id].delay_degrees;
    float delay_time_ns = combined[ray_id].delay_time_ns;
    
    float f_center = (f_start + f_stop) / 2.0f;
    float wavelength = speed_of_light / f_center;
    float delay_rad = delay_degrees * 3.14159265f / 180.0f;
    float delay_angle_sec = delay_rad * wavelength / speed_of_light;
    float delay_time_sec = delay_time_ns * 1e-9f;
    float total_delay_sec = delay_angle_sec + delay_time_sec;
    
    // ✅ ГЛАВНОЕ ИСПРАВЛЕНИЕ: ИСПОЛЬЗУЕМ FLOAT ВМЕСТО INT!
    float total_delay_samples = total_delay_sec * sample_rate;
    float delayed_sample_float = (float)sample_id - total_delay_samples;
    
    if (delayed_sample_float < 0.0f) {
        output[ray_id * num_samples + sample_id] = (float2)(0.0f, 0.0f);
        return;
    }
    
    int sample_int = (int)delayed_sample_float;
    float sample_frac = delayed_sample_float - (float)sample_int;
    
    if (sample_int >= (int)num_samples - 1) {
        output[ray_id * num_samples + sample_id] = (float2)(0.0f, 0.0f);
    }
    else if (sample_frac < 1e-6f) {
        float t = (float)sample_int / sample_rate;
        float chirp_rate = (f_stop - f_start) / duration;
        float phase = 2.0f * 3.14159265f * (f_start * t + 0.5f * chirp_rate * t * t);
        output[ray_id * num_samples + sample_id] = (float2)(cos(phase), sin(phase));
    }
    else {
        // ✅ ИНТЕРПОЛЯЦИЯ между двумя соседними отсчётами
        float t1 = (float)sample_int / sample_rate;
        float chirp_rate = (f_stop - f_start) / duration;
        float phase1 = 2.0f * 3.14159265f * (f_start * t1 + 0.5f * chirp_rate * t1 * t1);
        float real1 = cos(phase1), imag1 = sin(phase1);
        
        float t2 = (float)(sample_int + 1) / sample_rate;
        float phase2 = 2.0f * 3.14159265f * (f_start * t2 + 0.5f * chirp_rate * t2 * t2);
        float real2 = cos(phase2), imag2 = sin(phase2);
        
        float real = real1 * (1.0f - sample_frac) + real2 * sample_frac;
        float imag = imag1 * (1.0f - sample_frac) + imag2 * sample_frac;
        output[ray_id * num_samples + sample_id] = (float2)(real, imag);
    }
}

// ═════════════════════════════════════════════════════════════════════════
// KERNEL 4: ГЕНЕРАЦИЯ СУММЫ СИНУСОИД НА GPU
// ═════════════════════════════════════════════════════════════════════════

__kernel void kernel_sinusoid_combined(
    __global float2 *output,           // Выходные комплексные сигналы
    __global const RaySinusoidParams *ray_params, // Параметры синусоидов для каждого луча
    uint num_ray_params,               // Количество описанных лучей в ray_params
    uint num_samples,                  // Количество отсчётов на луч
    uint num_beams                    // Количество лучей (из параметров)
) {
    uint gid = get_global_id(0);
    
    if (gid >= (uint)num_samples * num_beams) return;
    
    uint ray_id = gid / num_samples;
    uint sample_id = gid % num_samples;
    
    if (ray_id >= num_beams || sample_id >= num_samples) return;
    
    float real_sum = 0.0f;
    float imag_sum = 0.0f;
    
    // Найти параметры для текущего луча
    bool ray_found = false;
    for (uint i = 0; i < num_ray_params; i++) {
        if (ray_params[i].ray_index == ray_id) {
            ray_found = true;
            uint num_sinusoids = ray_params[i].num_sinusoids;
            
            for (uint j = 0; j < num_sinusoids; j++) {
                SinusoidParam sin_param = ray_params[i].sinusoids[j];
                
                // Вычислить фазу для текущего отсчёта
                float phase_rad = 2.0f * 3.14159265f * (float)sample_id / sin_param.period;
                float phase_deg_rad = sin_param.phase_deg * 3.14159265f / 180.0f;
                float total_phase = phase_rad + phase_deg_rad;
                
                // Добавить к сумме
                real_sum += sin_param.amplitude * cos(total_phase);
                imag_sum += sin_param.amplitude * sin(total_phase);
            }
            break;
        }
    }
    
    // Если для луча нет параметров - использовать значения по умолчанию
    if (!ray_found) {
        float amp = 1.0f;
        float period = (float)(num_samples / 2); // Период = половина количества точек
        float phase_deg = 0.0f;
        
        float phase_rad = 2.0f * 3.14159265f * (float)sample_id / period;
        float phase_deg_rad = phase_deg * 3.14159265f / 180.0f;
        float total_phase = phase_rad + phase_deg_rad;
        
        real_sum = amp * cos(total_phase);
        imag_sum = amp * sin(total_phase);
    }
    
    // Записать результат
    uint out_idx = ray_id * num_samples + sample_id;
    output[out_idx] = (float2)(real_sum, imag_sum);
}

// ═════════════════════════════════════════════════════════════════════════

)";
  }

  void GeneratorGPU::ExecuteKernel(
      cl_kernel kernel,
      cl_mem output_buffer,
      cl_mem delay_buffer)
  {

    if (!kernel || !output_buffer)
    {
      throw std::invalid_argument("[GeneratorGPU] Invalid kernel or output_buffer");
    }

    // ✅ Используем CommandQueuePool для получения очереди
    cl_command_queue queue = ManagerOpenCL::CommandQueuePool::GetNextQueue();

    cl_int err = CL_SUCCESS;

    // ✅ Установить аргументы kernel в зависимости от типа
    if (delay_buffer)
    {
      // kernel_lfm_delayed с параметрами задержки

      // arg 0: output buffer
      err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_buffer);
      if (err != CL_SUCCESS)
        throw std::runtime_error("clSetKernelArg 0 failed");

      // arg 1: delay buffer
      err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &delay_buffer);
      if (err != CL_SUCCESS)
        throw std::runtime_error("clSetKernelArg 1 failed");

      // arg 2-8: scalar parameters
      err = clSetKernelArg(kernel, 2, sizeof(float), &params_.f_start);
      err = clSetKernelArg(kernel, 3, sizeof(float), &params_.f_stop);
      err = clSetKernelArg(kernel, 4, sizeof(float), &params_.sample_rate);
      err = clSetKernelArg(kernel, 5, sizeof(float), &params_.duration);

      float speed_of_light = 3.0e8f;
      err = clSetKernelArg(kernel, 6, sizeof(float), &speed_of_light);

      cl_uint num_samples = static_cast<cl_uint>(num_samples_);
      cl_uint num_beams = static_cast<cl_uint>(num_beams_);
      cl_uint num_delays = num_beams;

      err = clSetKernelArg(kernel, 7, sizeof(cl_uint), &num_samples);
      err = clSetKernelArg(kernel, 8, sizeof(cl_uint), &num_beams);
      err = clSetKernelArg(kernel, 9, sizeof(cl_uint), &num_delays);
    }
    else
    {
      // kernel_lfm_basic без параметров задержки

      // arg 0: output buffer
      err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_buffer);
      if (err != CL_SUCCESS)
        throw std::runtime_error("clSetKernelArg 0 failed");

      // arg 1-6: scalar parameters
      err = clSetKernelArg(kernel, 1, sizeof(float), &params_.f_start);
      err = clSetKernelArg(kernel, 2, sizeof(float), &params_.f_stop);
      err = clSetKernelArg(kernel, 3, sizeof(float), &params_.sample_rate);
      err = clSetKernelArg(kernel, 4, sizeof(float), &params_.duration);

      cl_uint num_samples = static_cast<cl_uint>(num_samples_);
      cl_uint num_beams = static_cast<cl_uint>(num_beams_);

      err = clSetKernelArg(kernel, 5, sizeof(cl_uint), &num_samples);
      err = clSetKernelArg(kernel, 6, sizeof(cl_uint), &num_beams);
    }

    // ✅ Выполнить kernel
    size_t global_work_size = total_size_;
    size_t local_work_size = 256; // Оптимально для GPU

    err = clEnqueueNDRangeKernel(
        queue,
        kernel,
        1, // одномерная сетка
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr, nullptr);

    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "[GeneratorGPU] clEnqueueNDRangeKernel failed with error " +
          std::to_string(err));
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // PUBLIC METHODS - API
  // ════════════════════════════════════════════════════════════════════════════

  cl_mem GeneratorGPU::signal_base()
  {
    if (!engine_)
    {
      throw std::runtime_error("[GeneratorGPU] OpenCLComputeEngine not initialized");
    }

    if (!kernel_lfm_basic_)
    {
      throw std::runtime_error("[GeneratorGPU] kernel_lfm_basic not loaded");
    }

    std::cout << "[GeneratorGPU] Generating signal_base()..." << std::endl;

    // ✅ Создать GPU буфер через engine
    auto output = engine_->CreateBuffer(total_size_, ManagerOpenCL::MemoryType::GPU_WRITE_ONLY);

    try
    {
      // ✅ Выполнить kernel
      ExecuteKernel(kernel_lfm_basic_, output->Get());

      // ✅ Сохранить unique_ptr в cache (ВАЖНО: буфер не будет освобожден!)
      buffer_signal_base_ = std::move(output);

      std::cout << "[GeneratorGPU] ✅ signal_base() completed" << std::endl;

      // ✅ Вернуть cl_mem из сохраненного буфера
      return buffer_signal_base_->Get();
    }
    catch (const std::exception &e)
    {
      throw std::runtime_error(
          std::string("[GeneratorGPU] signal_base() failed: ") + e.what());
    }
  }

  cl_mem GeneratorGPU::signal_valedation(
      const DelayParameter *m_delay,
      size_t num_delay_params)
  {

    if (!engine_)
    {
      throw std::runtime_error("[GeneratorGPU] OpenCLComputeEngine not initialized");
    }

    if (!kernel_lfm_delayed_)
    {
      throw std::runtime_error("[GeneratorGPU] kernel_lfm_delayed not loaded");
    }

    if (!m_delay)
    {
      throw std::invalid_argument("[GeneratorGPU] m_delay array is nullptr");
    }

    if (num_delay_params != num_beams_)
    {
      throw std::invalid_argument(
          "[GeneratorGPU] num_delay_params (" + std::to_string(num_delay_params) +
          ") must equal num_beams (" + std::to_string(num_beams_) + ")");
    }

    std::cout << "[GeneratorGPU] Generating signal_valedation() with "
              << num_delay_params << " delay parameters..." << std::endl;

    try
    {
      // ✅ Создать GPU буфер для параметров задержки
      auto delay_gpu_buffer = engine_->CreateBufferWithData(
          std::vector<std::complex<float>>(
              reinterpret_cast<const std::complex<float> *>(m_delay),
              reinterpret_cast<const std::complex<float> *>(m_delay) + num_delay_params),
          ManagerOpenCL::MemoryType::GPU_READ_ONLY);

      // ✅ Создать GPU буфер для выходных данных
      auto output = engine_->CreateBuffer(total_size_, ManagerOpenCL::MemoryType::GPU_WRITE_ONLY);

      // ✅ Выполнить kernel
      ExecuteKernel(kernel_lfm_delayed_, output->Get(), delay_gpu_buffer->Get());

      // ✅ Сохранить unique_ptr в cache (ВАЖНО: буфер не будет освобожден!)
      buffer_signal_delayed_ = std::move(output);

      std::cout << "[GeneratorGPU] ✅ signal_valedation() completed" << std::endl;

      return buffer_signal_delayed_->Get();
    }
    catch (const std::exception &e)
    {
      throw std::runtime_error(
          std::string("[GeneratorGPU] signal_valedation() failed: ") + e.what());
    }
  }

  cl_mem GeneratorGPU::signal_combined_delays(
      const CombinedDelayParam* combined_delays,
      size_t num_delay_params) {

      if (!engine_) {
          throw std::runtime_error("GeneratorGPU: Engine not initialized");
      }
      if (!kernel_lfm_combined_) {
          throw std::runtime_error("GeneratorGPU: kernel_lfm_combined not loaded");
      }
      if (!combined_delays) {
          throw std::invalid_argument("GeneratorGPU: combined_delays is null");
      }
      if (num_delay_params != num_beams_) {
          throw std::invalid_argument(
              "GeneratorGPU: num_delay_params (" + std::to_string(num_delay_params) +
              ") must equal num_beams (" + std::to_string(num_beams_) + ")"
          );
      }



      try {
          // ✅ Шаг 1: Подготовить хостовый вектор параметров
          std::vector<CombinedDelayParam> combined_host(
              combined_delays,
              combined_delays + num_delay_params
          );

          // ✅ Шаг 2: Загрузить на GPU через типобезопасный API
          auto combined_gpu_buffer = engine_->CreateTypedBufferWithData(
              combined_host,
              ManagerOpenCL::MemoryType::GPU_READ_ONLY
          );

          // ✅ Шаг 3: Создать выходной буфер
          auto output = engine_->CreateBuffer(
              total_size_,
              ManagerOpenCL::MemoryType::GPU_WRITE_ONLY
          );

          // ✅ Шаг 4: Выполнить kernel
          ExecuteKernel(
              kernel_lfm_combined_,
              output->Get(),
              combined_gpu_buffer->Get()
          );

          // ✅ Шаг 5: Кэшировать результат и вернуть
          buffer_signal_combined_ = std::move(output);

          std::cout << "GeneratorGPU: signal_combined_delays completed." << std::endl;

          return buffer_signal_combined_->Get();

      } catch (const std::exception& e) {
          throw std::runtime_error(
              std::string("GeneratorGPU: signal_combined_delays failed: ") + e.what()
          );
      }
  }

  cl_mem GeneratorGPU::signal_sinusoids(
      const SinusoidGenParams& params,
      const RaySinusoidMap& map_ray)
  {
      if (!engine_) {
          throw std::runtime_error("[GeneratorGPU] OpenCLComputeEngine not initialized");
      }

      if (!kernel_sinusoid_combined_) {
          throw std::runtime_error("[GeneratorGPU] kernel_sinusoid_combined not loaded");
      }

      // Валидация параметров
      if (params.num_rays == 0 || params.count_points == 0) {
          throw std::invalid_argument(
              "[GeneratorGPU] signal_sinusoids: num_rays and count_points must be > 0");
      }

      std::cout << "[GeneratorGPU] Generating signal_sinusoids() with "
                << params.num_rays << " rays, " << params.count_points 
                << " points per ray..." << std::endl;

      try {
          // ════════════════════════════════════════════════════════════════
          // ШАГ 1: Преобразовать map в массив RaySinusoidParams
          // ════════════════════════════════════════════════════════════════
          
          std::vector<RaySinusoidParams> ray_params_array;

          if (map_ray.empty()) {
              // Дефолтные параметры для всех лучей
              std::cout << "[GeneratorGPU] Map is empty, using default parameters for all rays" << std::endl;
              for (size_t i = 0; i < params.num_rays; ++i) {
                  RaySinusoidParams rp;
                  rp.ray_index = static_cast<cl_uint>(i);
                  rp.num_sinusoids = 1;
                  rp.sinusoids[0].amplitude = 1.0f;
                  rp.sinusoids[0].period = static_cast<float>(params.count_points / 2); // Целая часть
                  rp.sinusoids[0].phase_deg = 0.0f;
                  ray_params_array.push_back(rp);
              }
          } else {
              // Только описанные лучи
              std::cout << "[GeneratorGPU] Map contains " << map_ray.size() << " ray(s)" << std::endl;
              for (const auto& pair : map_ray) {
                  if (pair.first < 0 || pair.first >= static_cast<int>(params.num_rays)) {
                      std::cerr << "⚠️  Warning: ray_index " << pair.first 
                                << " is out of range [0, " << (params.num_rays - 1) << "], skipping" << std::endl;
                      continue;
                  }

                  RaySinusoidParams rp;
                  rp.ray_index = static_cast<cl_uint>(pair.first);
                  rp.num_sinusoids = static_cast<cl_uint>(std::min(pair.second.size(), size_t(10)));
                  
                  if (pair.second.size() > 10) {
                      std::cerr << "⚠️  Warning: ray " << pair.first 
                                << " has " << pair.second.size() 
                                << " sinusoids, only first 10 will be used" << std::endl;
                  }

                  for (size_t j = 0; j < rp.num_sinusoids; ++j) {
                      rp.sinusoids[j].amplitude = pair.second[j].amplitude;
                      rp.sinusoids[j].period = pair.second[j].period;
                      rp.sinusoids[j].phase_deg = pair.second[j].phase_deg;
                  }
                  ray_params_array.push_back(rp);
              }
          }

          if (ray_params_array.empty()) {
              throw std::runtime_error(
                  "[GeneratorGPU] signal_sinusoids: No valid ray parameters after processing map");
          }

          std::cout << "[GeneratorGPU] Prepared " << ray_params_array.size() 
                    << " ray parameter(s) for GPU" << std::endl;

          // ════════════════════════════════════════════════════════════════
          // ШАГ 2: Создать буфер параметров на GPU
          // ════════════════════════════════════════════════════════════════
          
          auto params_buffer = engine_->CreateTypedBufferWithData(
              ray_params_array,
              ManagerOpenCL::MemoryType::GPU_READ_ONLY
          );

          // ════════════════════════════════════════════════════════════════
          // ШАГ 3: Создать выходной буфер
          // ════════════════════════════════════════════════════════════════
          
          size_t total_size = params.num_rays * params.count_points;
          auto output = engine_->CreateBuffer(
              total_size,
              ManagerOpenCL::MemoryType::GPU_WRITE_ONLY
          );

          // ════════════════════════════════════════════════════════════════
          // ШАГ 4: Установить аргументы kernel
          // ════════════════════════════════════════════════════════════════
          
          cl_command_queue queue = ManagerOpenCL::CommandQueuePool::GetNextQueue();
          cl_int err = CL_SUCCESS;

          cl_mem output_mem = output->Get();
          cl_mem params_mem = params_buffer->Get();
          cl_uint num_ray_params = static_cast<cl_uint>(ray_params_array.size());
          cl_uint num_samples = static_cast<cl_uint>(params.count_points);
          cl_uint num_beams = static_cast<cl_uint>(params.num_rays);

          err = clSetKernelArg(kernel_sinusoid_combined_, 0, sizeof(cl_mem), &output_mem);
          if (err != CL_SUCCESS) {
              throw std::runtime_error("clSetKernelArg 0 failed: " + std::to_string(err));
          }

          err = clSetKernelArg(kernel_sinusoid_combined_, 1, sizeof(cl_mem), &params_mem);
          if (err != CL_SUCCESS) {
              throw std::runtime_error("clSetKernelArg 1 failed: " + std::to_string(err));
          }

          err = clSetKernelArg(kernel_sinusoid_combined_, 2, sizeof(cl_uint), &num_ray_params);
          if (err != CL_SUCCESS) {
              throw std::runtime_error("clSetKernelArg 2 failed: " + std::to_string(err));
          }

          err = clSetKernelArg(kernel_sinusoid_combined_, 3, sizeof(cl_uint), &num_samples);
          if (err != CL_SUCCESS) {
              throw std::runtime_error("clSetKernelArg 3 failed: " + std::to_string(err));
          }

          err = clSetKernelArg(kernel_sinusoid_combined_, 4, sizeof(cl_uint), &num_beams);
          if (err != CL_SUCCESS) {
              throw std::runtime_error("clSetKernelArg 4 failed: " + std::to_string(err));
          }

          // ════════════════════════════════════════════════════════════════
          // ШАГ 5: Выполнить kernel
          // ════════════════════════════════════════════════════════════════

          size_t global_work_size = total_size;
          size_t* p_local_work_size = nullptr; // Let OpenCL choose optimal local work size

          std::cout << "[GeneratorGPU] Executing kernel_sinusoid_combined (grid: "
                    << global_work_size << ", block: auto)" << std::endl;

          err = clEnqueueNDRangeKernel(
              queue,
              kernel_sinusoid_combined_,
              1, // одномерная сетка
              nullptr,
              &global_work_size,
              p_local_work_size,
              0, nullptr, nullptr
          );

          if (err != CL_SUCCESS) {
              throw std::runtime_error(
                  "[GeneratorGPU] clEnqueueNDRangeKernel failed with error " + 
                  std::to_string(err));
          }

          // ════════════════════════════════════════════════════════════════
          // ШАГ 6: Синхронизация - дождаться завершения kernel
          // ВАЖНО: Без этого данные могут быть не готовы для FFT!
          // ════════════════════════════════════════════════════════════════
          err = clFinish(queue);
          if (err != CL_SUCCESS) {
              throw std::runtime_error(
                  "[GeneratorGPU] clFinish failed with error " + std::to_string(err));
          }

          // ════════════════════════════════════════════════════════════════
          // ШАГ 7: Сохранить результат и вернуть
          // ════════════════════════════════════════════════════════════════
          
          buffer_signal_sinusoid_ = std::move(output);

          std::cout << "[GeneratorGPU] ✅ signal_sinusoids() completed" << std::endl;

          return buffer_signal_sinusoid_->Get();

      } catch (const std::exception& e) {
          throw std::runtime_error(
              std::string("[GeneratorGPU] signal_sinusoids() failed: ") + e.what()
          );
      }
  }

  void GeneratorGPU::ClearGPU()
  {
    if (!engine_)
    {
      throw std::runtime_error("[GeneratorGPU] OpenCLComputeEngine not initialized");
    }

    std::cout << "[GeneratorGPU] Syncing GPU..." << std::endl;

    // ✅ Дождаться завершения всех операций
    engine_->Finish();

    std::cout << "[GeneratorGPU] ✅ GPU synced" << std::endl;
  }

  void GeneratorGPU::SetParametersAngle(float angle_start, float angle_stop)
  {
    params_.SetAngle(angle_start, angle_stop);
    std::cout << "[GeneratorGPU] Angle set: [" << params_.angle_start_deg
              << "°, " << params_.angle_stop_deg << "°]" << std::endl;
  }

  std::vector<std::complex<float>> GeneratorGPU::GetSignalAsVector(int beam_index)
  {
    // ✅ Проверка индекса
    if (beam_index < 0 || beam_index >= (int)num_beams_)
    {
      std::cerr << "❌ GetSignalAsVector: Invalid beam_index " << beam_index
                << " (valid range: 0-" << (num_beams_ - 1) << ")" << std::endl;
      return {};
    }

    // ✅ Проверка, что хотя бы один буфер создан
    ManagerOpenCL::GPUMemoryBuffer* active_buffer = nullptr;
    if (buffer_signal_sinusoid_ && buffer_signal_sinusoid_->Get())
    {
      active_buffer = buffer_signal_sinusoid_.get();
    }
    else if (buffer_signal_combined_ && buffer_signal_combined_->Get())
    {
      active_buffer = buffer_signal_combined_.get();
    }
    else if (buffer_signal_delayed_ && buffer_signal_delayed_->Get())
    {
      active_buffer = buffer_signal_delayed_.get();
    }
    else if (buffer_signal_base_ && buffer_signal_base_->Get())
    {
      active_buffer = buffer_signal_base_.get();
    }
    else
    {
      std::cerr << "❌ GetSignalAsVector: No valid buffer found. "
                << "Call signal_base(), signal_valedation(), signal_combined_delays(), or signal_sinusoids() first!" << std::endl;
      return {};
    }

    // ✅ Синхронизация GPU перед чтением
    ClearGPU();

    std::vector<std::complex<float>> result(num_samples_);

    try
    {
      // ✅ Получить валидную очередь
      cl_command_queue queue = ManagerOpenCL::CommandQueuePool::GetNextQueue();
      if (!queue)
      {
        std::cerr << "❌ GetSignalAsVector: Invalid command queue" << std::endl;
        return {};
      }

      // ✅ Вычислить смещение и размер
      size_t offset_bytes = beam_index * num_samples_ * sizeof(std::complex<float>);
      size_t size_bytes = num_samples_ * sizeof(std::complex<float>);

      // ✅ Проверка границ
      size_t total_buffer_size = total_size_ * sizeof(std::complex<float>);
      if (offset_bytes + size_bytes > total_buffer_size)
      {
        std::cerr << "❌ GetSignalAsVector: Offset+Size exceeds buffer size. "
                  << "offset=" << offset_bytes << " size=" << size_bytes
                  << " total=" << total_buffer_size << std::endl;
        return {};
      }

      // ✅ Вызов clEnqueueReadBuffer с правильными параметрами
      cl_int err = clEnqueueReadBuffer(
          queue,                      // command_queue
          active_buffer->Get(),       // buffer (получаем cl_mem из активного буфера)
          CL_TRUE,                    // blocking_read (CL_TRUE = ждём завершения)
          offset_bytes,               // offset в байтах
          size_bytes,                 // размер в байтах
          result.data(),              // указатель на host память
          0,                          // num_events_in_wait_list (0 = нет зависимостей)
          nullptr,                    // event_wait_list (nullptr = нет событий)
          nullptr                     // event (nullptr = не возвращаем событие)
      );

      if (err != CL_SUCCESS)
      {
        std::cerr << "❌ clEnqueueReadBuffer error: " << err << std::endl;
        std::cerr << "   beam_index=" << beam_index
                  << " offset_bytes=" << offset_bytes
                  << " size_bytes=" << size_bytes << std::endl;
        return {};
      }

      return result;
    }
    catch (const std::exception &e)
    {
      std::cerr << "❌ Exception in GetSignalAsVector: " << e.what() << std::endl;
      return {};
    }
    catch (...)
    {
      std::cerr << "❌ Unknown exception in GetSignalAsVector" << std::endl;
      return {};
    }
  }

  std::vector<std::complex<float>> GeneratorGPU::GetSignalAsVectorPartial(int beam_index, size_t num_samples)
  {
    // Такой же как GetSignalAsVector(), но используем ReadPartial():

    if (beam_index < 0 || beam_index >= (int)num_beams_)
    {
      return {};
    }

    if (num_samples > num_samples_)
    {
      num_samples = num_samples_;
    }

    ClearGPU();

    auto &core = ManagerOpenCL::OpenCLCore::GetInstance();

    // ✅ Проверка, что хотя бы один буфер создан
    ManagerOpenCL::GPUMemoryBuffer* active_buffer = nullptr;
    if (buffer_signal_sinusoid_ && buffer_signal_sinusoid_->Get())
    {
      active_buffer = buffer_signal_sinusoid_.get();
    }
    else if (buffer_signal_combined_ && buffer_signal_combined_->Get())
    {
      active_buffer = buffer_signal_combined_.get();
    }
    else if (buffer_signal_delayed_ && buffer_signal_delayed_->Get())
    {
      active_buffer = buffer_signal_delayed_.get();
    }
    else if (buffer_signal_base_ && buffer_signal_base_->Get())
    {
      active_buffer = buffer_signal_base_.get();
    }
    else
    {
      std::cerr << "❌ GetSignalAsVectorPartial: No valid buffer found" << std::endl;
      return {};
    }

    ManagerOpenCL::GPUMemoryBuffer buffer(
        core.GetContext(),
        ManagerOpenCL::CommandQueuePool::GetNextQueue(),
        active_buffer->Get(), // Получаем cl_mem из активного буфера
        total_size_,
        ManagerOpenCL::MemoryType::GPU_READ_ONLY);

    // РАЗЛИЧИЕ: используем ReadPartial() вместо ReadFromGPU()
    auto all_data = buffer.ReadPartial(total_size_); // Сначала читаем всё

    size_t beam_start = beam_index * num_samples_;
    size_t beam_end = beam_start + num_samples; // ← num_samples, не num_samples_!

    std::vector<std::complex<float>> result(
        all_data.begin() + beam_start,
        all_data.begin() + beam_end);

    return result;
  }

  std::vector<std::complex<float>> GeneratorGPU::GetSignalAsVectorAll()
  {
    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 1: Синхронизировать GPU перед чтением
    // ════════════════════════════════════════════════════════════════════════

    ClearGPU(); // Ждём завершения всех операций

    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 2: Получить engine и OpenCLCore
    // ════════════════════════════════════════════════════════════════════════

    auto &core = ManagerOpenCL::OpenCLCore::GetInstance();

    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 3: Обернуть raw cl_mem в GPUMemoryBuffer (NON-OWNING!)
    // ════════════════════════════════════════════════════════════════════════
    //
    // ВАЖНО: Используем NON-OWNING конструктор (второй параметр - external buffer)!
    // Это значит GPUMemoryBuffer НЕ удалит cl_mem при своём разрушении.
    // Удаление сделает GeneratorGPU в своём деструкторе.
    //
    // Конструктор:
    // GPUMemoryBuffer(context, queue, external_buffer, num_elements, type)

    // ✅ Проверка, что хотя бы один буфер создан
    ManagerOpenCL::GPUMemoryBuffer* active_buffer = nullptr;
    if (buffer_signal_sinusoid_ && buffer_signal_sinusoid_->Get())
    {
      active_buffer = buffer_signal_sinusoid_.get();
    }
    else if (buffer_signal_combined_ && buffer_signal_combined_->Get())
    {
      active_buffer = buffer_signal_combined_.get();
    }
    else if (buffer_signal_delayed_ && buffer_signal_delayed_->Get())
    {
      active_buffer = buffer_signal_delayed_.get();
    }
    else if (buffer_signal_base_ && buffer_signal_base_->Get())
    {
      active_buffer = buffer_signal_base_.get();
    }
    else
    {
      std::cerr << "❌ GetSignalAsVectorAll: No valid buffer found" << std::endl;
      return {};
    }

    try
    {
      ManagerOpenCL::GPUMemoryBuffer buffer(
          core.GetContext(),                     // контекст OpenCL
          ManagerOpenCL::CommandQueuePool::GetNextQueue(), // очередь для операции
          active_buffer->Get(),                  // cl_mem из активного буфера (НЕ удалится!)
          total_size_,                           // всего элементов (num_beams * num_samples)
          ManagerOpenCL::MemoryType::GPU_READ_ONLY         // тип: только чтение
      );

      // ════════════════════════════════════════════════════════════════════
      // ШАГ 5: Прочитать все данные
      // ════════════════════════════════════════════════════════════════════

      std::cout << "[READ] Reading " << total_size_ << " samples from GPU..." << std::endl;
      auto all_data = buffer.ReadFromGPU();

      if (all_data.empty())
      {
        std::cerr << "❌ Failed to read data from GPU!" << std::endl;
        return {};
      }

      std::cout << "[READ] ✅ Successfully read " << all_data.size() << " samples" << std::endl;

      return all_data;
    }
    catch (const std::exception &e)
    {
      std::cerr << "❌ Exception in GetSignalAsVector(): " << e.what() << std::endl;
      return {};
    }
  };



} // namespace radar

/**
 * 
 ```cpp
cl_mem GeneratorGPU::signal_combined_delays(
    const CombinedDelayParam* combined_delays,
    size_t num_delay_params) {
    
    if (!engine_) throw std::runtime_error("Engine not initialized");
    if (!kernel_lfm_combined_) throw std::runtime_error("kernel not loaded");
    if (!combined_delays) throw std::invalid_argument("combined_delays is null");
    if (num_delay_params != num_beams_) throw std::invalid_argument("num mismatch");
    
    auto combined_gpu_buffer = engine_->CreateBufferWithData(
        std::vector<CombinedDelayParam>(combined_delays, combined_delays + num_delay_params),
        ManagerOpenCL::MemoryType::GPU_READ_ONLY
    );
    
    auto output = engine_->CreateBuffer(total_size_, ManagerOpenCL::MemoryType::GPU_WRITE_ONLY);
    ExecuteKernel(kernel_lfm_combined_, output->Get(), combined_gpu_buffer->Get());
    
    buffer_signal_combined_ = std::move(output);
    return buffer_signal_combined_->Get();
}
```
* 
 */