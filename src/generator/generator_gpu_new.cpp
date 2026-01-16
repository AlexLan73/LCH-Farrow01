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
#include <CL/cl.h>

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
    buffer_signal_base_.reset();  // Освободит unique_ptr (буфер будет освобожден автоматически)
    buffer_signal_delayed_.reset();
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
      buffer_signal_base_(std::move(other.buffer_signal_base_)),
      buffer_signal_delayed_(std::move(other.buffer_signal_delayed_)) {
    
    other.engine_ = nullptr;
    other.kernel_lfm_basic_ = nullptr;
    other.kernel_lfm_delayed_ = nullptr;
    other.buffer_signal_base_.reset();
    other.buffer_signal_delayed_.reset();
}

GeneratorGPU& GeneratorGPU::operator=(GeneratorGPU&& other) noexcept {
    if (this != &other) {
        // Очистить текущие ресурсы
        kernel_lfm_basic_ = nullptr;
        kernel_lfm_delayed_ = nullptr;
        kernel_program_ = nullptr;
        buffer_signal_base_.reset();
        buffer_signal_delayed_.reset();

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

        // Обнулить в other
        other.engine_ = nullptr;
        other.kernel_lfm_basic_ = nullptr;
        other.kernel_lfm_delayed_ = nullptr;
        other.buffer_signal_base_.reset();
        other.buffer_signal_delayed_.reset();
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
    cl_uint beam_index;
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
        
        cl_uint num_samples = static_cast<cl_uint>(num_samples_);
        cl_uint num_beams = static_cast<cl_uint>(num_beams_);
        cl_uint num_delays = num_beams;

        err = clSetKernelArg(kernel, 7, sizeof(cl_uint), &num_samples);
        err = clSetKernelArg(kernel, 8, sizeof(cl_uint), &num_beams);
        err = clSetKernelArg(kernel, 9, sizeof(cl_uint), &num_delays);
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
        
        cl_uint num_samples = static_cast<cl_uint>(num_samples_);
        cl_uint num_beams = static_cast<cl_uint>(num_beams_);

        err = clSetKernelArg(kernel, 5, sizeof(cl_uint), &num_samples);
        err = clSetKernelArg(kernel, 6, sizeof(cl_uint), &num_beams);
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
        
        // ✅ Сохранить unique_ptr в cache (ВАЖНО: буфер не будет освобожден!)
        buffer_signal_base_ = std::move(output);
        
        std::cout << "[GeneratorGPU] ✅ signal_base() completed" << std::endl;
        
        // ✅ Вернуть cl_mem из сохраненного буфера
        return buffer_signal_base_->Get();
        
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
        
        // ✅ Сохранить unique_ptr в cache (ВАЖНО: буфер не будет освобожден!)
        buffer_signal_delayed_ = std::move(output);
        
        std::cout << "[GeneratorGPU] ✅ signal_valedation() completed" << std::endl;
        
        return buffer_signal_delayed_->Get();
        
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

std::vector<std::complex<float>> GeneratorGPU::GetSignalAsVector(int beam_index) {
    // ✅ Проверка индекса
    if (beam_index < 0 || beam_index >= (int)num_beams_) {
        std::cerr << "❌ GetSignalAsVector: Invalid beam_index " << beam_index 
                  << " (valid range: 0-" << (num_beams_ - 1) << ")" << std::endl;
        return {};
    }
    
    // ✅ Проверка, что буфер создан
    if (!buffer_signal_base_ || !buffer_signal_base_->Get()) {
        std::cerr << "❌ GetSignalAsVector: buffer_signal_base_ is nullptr. "
                  << "Call signal_base() first!" << std::endl;
        return {};
    }
    
    // ✅ Синхронизация GPU перед чтением
    ClearGPU();
    
    std::vector<std::complex<float>> result(num_samples_);
    
    try {
        // ✅ Получить валидную очередь
        cl_command_queue queue = gpu::CommandQueuePool::GetNextQueue();
        if (!queue) {
            std::cerr << "❌ GetSignalAsVector: Invalid command queue" << std::endl;
            return {};
        }
        
        // ✅ Вычислить смещение и размер
        size_t offset_bytes = beam_index * num_samples_ * sizeof(std::complex<float>);
        size_t size_bytes = num_samples_ * sizeof(std::complex<float>);
        
        // ✅ Проверка границ
        size_t total_buffer_size = total_size_ * sizeof(std::complex<float>);
        if (offset_bytes + size_bytes > total_buffer_size) {
            std::cerr << "❌ GetSignalAsVector: Offset+Size exceeds buffer size. "
                      << "offset=" << offset_bytes << " size=" << size_bytes 
                      << " total=" << total_buffer_size << std::endl;
            return {};
        }
        
        // ✅ Вызов clEnqueueReadBuffer с правильными параметрами
        cl_int err = clEnqueueReadBuffer(
            queue,                           // command_queue
            buffer_signal_base_->Get(),      // buffer (получаем cl_mem из unique_ptr)
            CL_TRUE,                         // blocking_read (CL_TRUE = ждём завершения)
            offset_bytes,                    // offset в байтах
            size_bytes,                      // размер в байтах
            result.data(),                   // указатель на host память
            0,                               // num_events_in_wait_list (0 = нет зависимостей)
            nullptr,                         // event_wait_list (nullptr = нет событий)
            nullptr                          // event (nullptr = не возвращаем событие)
        );
        
        if (err != CL_SUCCESS) {
            std::cerr << "❌ clEnqueueReadBuffer error: " << err << std::endl;
            std::cerr << "   beam_index=" << beam_index 
                      << " offset_bytes=" << offset_bytes 
                      << " size_bytes=" << size_bytes << std::endl;
            return {};
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in GetSignalAsVector: " << e.what() << std::endl;
        return {};
    } catch (...) {
        std::cerr << "❌ Unknown exception in GetSignalAsVector" << std::endl;
        return {};
    }
}

  std::vector<std::complex<float>>  GeneratorGPU::GetSignalAsVectorPartial(int beam_index, size_t num_samples){
    // Такой же как GetSignalAsVector(), но используем ReadPartial():
    
    if (beam_index < 0 || beam_index >= (int)num_beams_) {
        return {};
    }
    
    if (num_samples > num_samples_) {
        num_samples = num_samples_;
    }
    
    ClearGPU();
    
    auto& core = gpu::OpenCLCore::GetInstance();
    
    if (!buffer_signal_base_ || !buffer_signal_base_->Get()) {
        std::cerr << "❌ GetSignalAsVectorPartial: buffer_signal_base_ is nullptr" << std::endl;
        return {};
    }
    
    gpu::GPUMemoryBuffer buffer(
        core.GetContext(),
        gpu::CommandQueuePool::GetNextQueue(),
        buffer_signal_base_->Get(),  // Получаем cl_mem из unique_ptr
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

  
  std::vector<std::complex<float>>  GeneratorGPU::GetSignalAsVectorAll(){
    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 1: Синхронизировать GPU перед чтением
    // ════════════════════════════════════════════════════════════════════════
    
    ClearGPU();  // Ждём завершения всех операций
    
    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 2: Получить engine и OpenCLCore
    // ════════════════════════════════════════════════════════════════════════
    
    auto& core = gpu::OpenCLCore::GetInstance();
    
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
    
    if (!buffer_signal_base_ || !buffer_signal_base_->Get()) {
        std::cerr << "❌ GetSignalAsVectorAll: buffer_signal_base_ is nullptr" << std::endl;
        return {};
    }
    
    try {
        gpu::GPUMemoryBuffer buffer(
            core.GetContext(),                          // контекст OpenCL
            gpu::CommandQueuePool::GetNextQueue(),      // очередь для операции
            buffer_signal_base_->Get(),                 // cl_mem из unique_ptr (НЕ удалится!)
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
        
        return all_data;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in GetSignalAsVector(): " << e.what() << std::endl;
        return {};
    }

  };

} // namespace radar
