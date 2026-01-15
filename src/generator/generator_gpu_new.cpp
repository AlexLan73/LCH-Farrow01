#include "generator/generator_gpu_new.h"

// Включаем компоненты новой архитектуры
#include "GPU/opencl_compute_engine.hpp"
#include "GPU/kernel_program.hpp"
#include "GPU/command_queue_pool.hpp"
#include "GPU/opencl_core.hpp"

// Параметры сигнала
#include "interface/lfm_parameters.h"
#include "interface/DelayParameter.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include <stdexcept>

namespace radar {

// ════════════════════════════════════════════════════════════════════════════
// CONSTRUCTOR / DESTRUCTOR
// ════════════════════════════════════════════════════════════════════════════

GeneratorGPU::GeneratorGPU(const LFMParameters& params)
    : engine_(nullptr),
      params_(params),
      num_samples_(0),
      num_beams_(params.num_beams),
      total_size_(0),
      kernel_program_(nullptr),
      kernel_lfm_basic_(nullptr),
      kernel_lfm_delayed_(nullptr),
      buffer_signal_base_(nullptr),
      buffer_signal_delayed_(nullptr) {

    // ✅ Валидировать параметры
    if (!params_.IsValid()) {
        throw std::invalid_argument(
            "[GeneratorGPU] LFMParameters invalid: "
            "check f_start, f_stop, sample_rate, num_beams, duration/count_points"
        );
    }

    // ✅ Получить engine (ДОЛЖЕН быть инициализирован!)
    try {
        engine_ = &gpu::OpenCLComputeEngine::GetInstance();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "[GeneratorGPU] OpenCLComputeEngine not initialized.\n"
            "Call gpu::OpenCLCore::Initialize() → CommandQueuePool::Initialize() → "
            "OpenCLComputeEngine::Initialize() before creating GeneratorGPU"
        );
    }

    // ✅ Инициализировать (получить контекст из engine)
    try {
        Initialize();
        LoadKernels();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("[GeneratorGPU] Initialization failed: ") + e.what()
        );
    }

    std::cout << "[GeneratorGPU] ✅ Created successfully" << std::endl;
    std::cout << "  - Beams: " << num_beams_ << std::endl;
    std::cout << "  - Samples per beam: " << num_samples_ << std::endl;
    std::cout << "  - Total size: " << total_size_ << " elements (" 
              << (GetMemorySizeBytes() / (1024*1024)) << " MB)" << std::endl;
}

GeneratorGPU::~GeneratorGPU() {
    // ✅ ВАЖНО: Ресурсы управляются OpenCLComputeEngine
    // Не вызываем clRelease* - engine сам управляет жизненным циклом
    // Просто обнуляем указатели
    
    kernel_lfm_basic_ = nullptr;
    kernel_lfm_delayed_ = nullptr;
    kernel_program_ = nullptr;
    buffer_signal_base_ = nullptr;
    buffer_signal_delayed_ = nullptr;
    engine_ = nullptr;

    std::cout << "[GeneratorGPU] ✅ Destroyed" << std::endl;
}

// Move семантика
GeneratorGPU::GeneratorGPU(GeneratorGPU&& other) noexcept
    : engine_(other.engine_),
      params_(other.params_),
      num_samples_(other.num_samples_),
      num_beams_(other.num_beams_),
      total_size_(other.total_size_),
      kernel_program_(std::move(other.kernel_program_)),
      kernel_lfm_basic_(other.kernel_lfm_basic_),
      kernel_lfm_delayed_(other.kernel_lfm_delayed_),
      buffer_signal_base_(other.buffer_signal_base_),
      buffer_signal_delayed_(other.buffer_signal_delayed_) {
    
    other.engine_ = nullptr;
    other.kernel_lfm_basic_ = nullptr;
    other.kernel_lfm_delayed_ = nullptr;
    other.buffer_signal_base_ = nullptr;
    other.buffer_signal_delayed_ = nullptr;
}

GeneratorGPU& GeneratorGPU::operator=(GeneratorGPU&& other) noexcept {
    if (this != &other) {
        // Очистить текущие ресурсы
        kernel_lfm_basic_ = nullptr;
        kernel_lfm_delayed_ = nullptr;
        kernel_program_ = nullptr;
        buffer_signal_base_ = nullptr;
        buffer_signal_delayed_ = nullptr;

        // Переместить от other
        engine_ = other.engine_;
        params_ = other.params_;
        num_samples_ = other.num_samples_;
        num_beams_ = other.num_beams_;
        total_size_ = other.total_size_;
        kernel_program_ = std::move(other.kernel_program_);
        kernel_lfm_basic_ = other.kernel_lfm_basic_;
        kernel_lfm_delayed_ = other.kernel_lfm_delayed_;
        buffer_signal_base_ = other.buffer_signal_base_;
        buffer_signal_delayed_ = other.buffer_signal_delayed_;

        // Обнулить в other
        other.engine_ = nullptr;
        other.kernel_lfm_basic_ = nullptr;
        other.kernel_lfm_delayed_ = nullptr;
        other.buffer_signal_base_ = nullptr;
        other.buffer_signal_delayed_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// PRIVATE METHODS
// ════════════════════════════════════════════════════════════════════════════

void GeneratorGPU::Initialize() {
    // ✅ Рассчитать num_samples из duration или count_points
    if (params_.count_points > 0) {
        num_samples_ = params_.count_points;
        params_.duration = static_cast<float>(num_samples_) / params_.sample_rate;
    } else if (params_.duration > 0.0f) {
        num_samples_ = static_cast<size_t>(params_.duration * params_.sample_rate);
        params_.count_points = num_samples_;
    } else {
        throw std::invalid_argument(
            "[GeneratorGPU] Either count_points or duration must be > 0"
        );
    }

    // ✅ Рассчитать общий размер
    total_size_ = num_beams_ * num_samples_;

    std::cout << "[GeneratorGPU] Initialized:" << std::endl;
    std::cout << "  - Duration: " << params_.duration * 1e6 << " us" << std::endl;
    std::cout << "  - Num samples: " << num_samples_ << std::endl;
    std::cout << "  - Total size: " << total_size_ << std::endl;
}

void GeneratorGPU::LoadKernels() {
    // ✅ Получить исходный код
    std::string source = GetKernelSource();

    std::cout << "[GeneratorGPU] Loading kernels from GPU engine..." << std::endl;

    // ✅ Получить или скомпилировать программу (с кэшем!)
    kernel_program_ = engine_->LoadProgram(source);
    if (!kernel_program_) {
        throw std::runtime_error("[GeneratorGPU] Failed to load kernel program");
    }

    // ✅ Получить или создать kernels
    kernel_lfm_basic_ = engine_->GetKernel(kernel_program_, "kernel_lfm_basic");
    if (!kernel_lfm_basic_) {
        throw std::runtime_error("[GeneratorGPU] Failed to create kernel_lfm_basic");
    }

    kernel_lfm_delayed_ = engine_->GetKernel(kernel_program_, "kernel_lfm_delayed");
    if (!kernel_lfm_delayed_) {
        throw std::runtime_error("[GeneratorGPU] Failed to create kernel_lfm_delayed");
    }

    std::cout << "[GeneratorGPU] ✅ Kernels loaded successfully" << std::endl;
}

std::string GeneratorGPU::GetKernelSource() const {
    // ✅ Встроенный OpenCL C код с правильной структурой
    return R"(
// ═════════════════════════════════════════════════════════════════════════
// СТРУКТУРЫ (должны быть в начале!)
// ═════════════════════════════════════════════════════════════════════════

typedef struct {
    uint beam_index;
    float delay_degrees;
} DelayParam;

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
)";
}

void GeneratorGPU::ExecuteKernel(
    cl_kernel kernel,
    cl_mem output_buffer,
    cl_mem delay_buffer) {
    
    if (!kernel || !output_buffer) {
        throw std::invalid_argument("[GeneratorGPU] Invalid kernel or output_buffer");
    }

    // ✅ Используем CommandQueuePool для получения очереди
    cl_command_queue queue = gpu::CommandQueuePool::GetNextQueue();
    
    cl_int err = CL_SUCCESS;

    // ✅ Установить аргументы kernel в зависимости от типа
    if (delay_buffer) {
        // kernel_lfm_delayed с параметрами задержки
        
        // arg 0: output buffer
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_buffer);
        if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 0 failed");
        
        // arg 1: delay buffer
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &delay_buffer);
        if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 1 failed");
        
        // arg 2-8: scalar parameters
        err = clSetKernelArg(kernel, 2, sizeof(float), &params_.f_start);
        err = clSetKernelArg(kernel, 3, sizeof(float), &params_.f_stop);
        err = clSetKernelArg(kernel, 4, sizeof(float), &params_.sample_rate);
        err = clSetKernelArg(kernel, 5, sizeof(float), &params_.duration);
        
        float speed_of_light = 3.0e8f;
        err = clSetKernelArg(kernel, 6, sizeof(float), &speed_of_light);
        
        uint num_samples = static_cast<uint>(num_samples_);
        uint num_beams = static_cast<uint>(num_beams_);
        uint num_delays = num_beams;
        
        err = clSetKernelArg(kernel, 7, sizeof(uint), &num_samples);
        err = clSetKernelArg(kernel, 8, sizeof(uint), &num_beams);
        err = clSetKernelArg(kernel, 9, sizeof(uint), &num_delays);
    } else {
        // kernel_lfm_basic без параметров задержки
        
        // arg 0: output buffer
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_buffer);
        if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 0 failed");
        
        // arg 1-6: scalar parameters
        err = clSetKernelArg(kernel, 1, sizeof(float), &params_.f_start);
        err = clSetKernelArg(kernel, 2, sizeof(float), &params_.f_stop);
        err = clSetKernelArg(kernel, 3, sizeof(float), &params_.sample_rate);
        err = clSetKernelArg(kernel, 4, sizeof(float), &params_.duration);
        
        uint num_samples = static_cast<uint>(num_samples_);
        uint num_beams = static_cast<uint>(num_beams_);
        
        err = clSetKernelArg(kernel, 5, sizeof(uint), &num_samples);
        err = clSetKernelArg(kernel, 6, sizeof(uint), &num_beams);
    }

    // ✅ Выполнить kernel
    size_t global_work_size = total_size_;
    size_t local_work_size = 256;  // Оптимально для GPU

    err = clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,  // одномерная сетка
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr, nullptr
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "[GeneratorGPU] clEnqueueNDRangeKernel failed with error " +
            std::to_string(err)
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PUBLIC METHODS - API
// ════════════════════════════════════════════════════════════════════════════

cl_mem GeneratorGPU::signal_base() {
    if (!engine_) {
        throw std::runtime_error("[GeneratorGPU] OpenCLComputeEngine not initialized");
    }

    if (!kernel_lfm_basic_) {
        throw std::runtime_error("[GeneratorGPU] kernel_lfm_basic not loaded");
    }

    std::cout << "[GeneratorGPU] Generating signal_base()..." << std::endl;

    // ✅ Создать GPU буфер через engine
    auto output = engine_->CreateBuffer(total_size_, gpu::MemoryType::GPU_WRITE_ONLY);

    try {
        // ✅ Выполнить kernel
        ExecuteKernel(kernel_lfm_basic_, output->Get());
        
        // ✅ Сохранить буфер в cache
        buffer_signal_base_ = output->Get();
        
        std::cout << "[GeneratorGPU] ✅ signal_base() completed" << std::endl;
        
        // ✅ Освободить unique_ptr (выход из области видимости), но GPU память остаётся!
        // ВАЖНО: GetSizeBytes используется для отслеживания памяти в engine
        return output->Get();
        
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("[GeneratorGPU] signal_base() failed: ") + e.what()
        );
    }
}

cl_mem GeneratorGPU::signal_valedation(
    const DelayParameter* m_delay,
    size_t num_delay_params) {
    
    if (!engine_) {
        throw std::runtime_error("[GeneratorGPU] OpenCLComputeEngine not initialized");
    }

    if (!kernel_lfm_delayed_) {
        throw std::runtime_error("[GeneratorGPU] kernel_lfm_delayed not loaded");
    }

    if (!m_delay) {
        throw std::invalid_argument("[GeneratorGPU] m_delay array is nullptr");
    }

    if (num_delay_params != num_beams_) {
        throw std::invalid_argument(
            "[GeneratorGPU] num_delay_params (" + std::to_string(num_delay_params) +
            ") must equal num_beams (" + std::to_string(num_beams_) + ")"
        );
    }

    std::cout << "[GeneratorGPU] Generating signal_valedation() with " 
              << num_delay_params << " delay parameters..." << std::endl;

    try {
        // ✅ Создать GPU буфер для параметров задержки
        auto delay_gpu_buffer = engine_->CreateBufferWithData(
            std::vector<std::complex<float>>(
                reinterpret_cast<const std::complex<float>*>(m_delay),
                reinterpret_cast<const std::complex<float>*>(m_delay) + num_delay_params
            ),
            gpu::MemoryType::GPU_READ_ONLY
        );

        // ✅ Создать GPU буфер для выходных данных
        auto output = engine_->CreateBuffer(total_size_, gpu::MemoryType::GPU_WRITE_ONLY);

        // ✅ Выполнить kernel
        ExecuteKernel(kernel_lfm_delayed_, output->Get(), delay_gpu_buffer->Get());
        
        // ✅ Сохранить буфер в cache
        buffer_signal_delayed_ = output->Get();
        
        std::cout << "[GeneratorGPU] ✅ signal_valedation() completed" << std::endl;
        
        return output->Get();
        
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("[GeneratorGPU] signal_valedation() failed: ") + e.what()
        );
    }
}

void GeneratorGPU::ClearGPU() {
    if (!engine_) {
        throw std::runtime_error("[GeneratorGPU] OpenCLComputeEngine not initialized");
    }

    std::cout << "[GeneratorGPU] Syncing GPU..." << std::endl;
    
    // ✅ Дождаться завершения всех операций
    engine_->Finish();
    
    std::cout << "[GeneratorGPU] ✅ GPU synced" << std::endl;
}

void GeneratorGPU::SetParametersAngle(float angle_start, float angle_stop) {
    params_.SetAngle(angle_start, angle_stop);
    std::cout << "[GeneratorGPU] Angle set: [" << params_.angle_start_deg 
              << "°, " << params_.angle_stop_deg << "°]" << std::endl;
}

} // namespace radar
