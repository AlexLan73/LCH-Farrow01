//#include "generator_gpu.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

#include "generator/generator_gpu.h"
//#include "lfm_parameters.h"
namespace radar {

// ═════════════════════════════════════════════════════════════════════
// CONSTRUCTOR / DESTRUCTOR
// ═════════════════════════════════════════════════════════════════════

GeneratorGPU::GeneratorGPU(const LFMParameters& params)
    : platform_(nullptr),
      device_(nullptr),
      context_(nullptr),
      queue_(nullptr),
      program_(nullptr),
      kernel_lfm_basic_(nullptr),
      kernel_lfm_delayed_(nullptr),
      params_(params),
      num_samples_(params.GetNumSamples()),
      num_beams_(params.num_beams),
      total_size_(num_beams_ * num_samples_) {
    
    if (!params_.IsValid()) {
        throw std::invalid_argument("Invalid LFMParameters");
    }
    
    try {
        InitializeOpenCL();
        CompileKernels();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("GPU initialization failed: ") + e.what());
    }
}

GeneratorGPU::~GeneratorGPU() {
    // Освобождение OpenCL ресурсов
    if (kernel_lfm_basic_ != nullptr) clReleaseKernel(kernel_lfm_basic_);
    if (kernel_lfm_delayed_ != nullptr) clReleaseKernel(kernel_lfm_delayed_);
    if (program_ != nullptr) clReleaseProgram(program_);
    if (queue_ != nullptr) clReleaseCommandQueue(queue_);
    if (context_ != nullptr) clReleaseContext(context_);
}

// Move semantics
GeneratorGPU::GeneratorGPU(GeneratorGPU&& other) noexcept
    : platform_(other.platform_),
      device_(other.device_),
      context_(other.context_),
      queue_(other.queue_),
      program_(other.program_),
      kernel_lfm_basic_(other.kernel_lfm_basic_),
      kernel_lfm_delayed_(other.kernel_lfm_delayed_),
      params_(other.params_),
      num_samples_(other.num_samples_),
      num_beams_(other.num_beams_),
      total_size_(other.total_size_) {
    // Обнулить указатели в исходном объекте
    other.platform_ = nullptr;
    other.device_ = nullptr;
    other.context_ = nullptr;
    other.queue_ = nullptr;
    other.program_ = nullptr;
    other.kernel_lfm_basic_ = nullptr;
    other.kernel_lfm_delayed_ = nullptr;
}

GeneratorGPU& GeneratorGPU::operator=(GeneratorGPU&& other) noexcept {
    if (this != &other) {
        // Освободить текущие ресурсы
        if (queue_) clReleaseCommandQueue(queue_);
        if (program_) clReleaseProgram(program_);
        if (kernel_lfm_basic_) clReleaseKernel(kernel_lfm_basic_);
        if (kernel_lfm_delayed_) clReleaseKernel(kernel_lfm_delayed_);
        if (context_) clReleaseContext(context_);
        
        // Переместить ресурсы
        platform_ = other.platform_;
        device_ = other.device_;
        context_ = other.context_;
        queue_ = other.queue_;
        program_ = other.program_;
        kernel_lfm_basic_ = other.kernel_lfm_basic_;
        kernel_lfm_delayed_ = other.kernel_lfm_delayed_;
        // params_ не перемещаем, так как он const
        num_samples_ = other.num_samples_;
        num_beams_ = other.num_beams_;
        total_size_ = other.total_size_;
        
        // Обнулить указатели в исходном объекте
        other.platform_ = nullptr;
        other.device_ = nullptr;
        other.context_ = nullptr;
        other.queue_ = nullptr;
        other.program_ = nullptr;
        other.kernel_lfm_basic_ = nullptr;
        other.kernel_lfm_delayed_ = nullptr;
    }
    return *this;
}

// ═════════════════════════════════════════════════════════════════════
// PRIVATE METHODS
// ═════════════════════════════════════════════════════════════════════

void GeneratorGPU::InitializeOpenCL() {
    cl_int err = CL_SUCCESS;
    
    // 1. Получить платформу
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    
    if (num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    platform_ = platforms[0];  // Используем первую платформу
    
    // 2. Получить GPU устройство
    cl_uint num_devices = 0;
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    
    if (num_devices == 0) {
        throw std::runtime_error("No GPU devices found");
    }
    
    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    device_ = devices[0];  // Используем первое GPU устройство
    
    // 3. Создать контекст
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }
    
    // 4. Создать очередь команд
    // Используем clCreateCommandQueueWithProperties для OpenCL 2.0+
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL command queue");
    }
}

// ✅ ИСПРАВЛЕННЫЕ МЕТОДЫ для generator_gpu.cpp
// Копируйте эти методы в ваш файл

// ═══════════════════════════════════════════════════════════════════════════
// ИСПРАВЛЕННЫЙ МЕТОД CompileKernels() с детальным логированием ошибок
// ═══════════════════════════════════════════════════════════════════════════

void GeneratorGPU::CompileKernels() {
    cl_int err = CL_SUCCESS;
    std::string source = GetKernelSource();
    const char* source_str = source.c_str();
    size_t source_len = source.length();

    std::cout << "🔨 Creating OpenCL program from source (" << source_len << " chars)..." << std::endl;

    program_ = clCreateProgramWithSource(context_, 1, &source_str, &source_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL program (err: " + 
                                std::to_string(err) + ")");
    }

    // Компилировать kernel
    std::cout << "⚙️  Compiling OpenCL kernel code..." << std::endl;
    err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    
    // Получить BUILD LOG (важно: размер может быть 0 даже при ошибке)
    size_t log_size = 0;
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 
                         0, nullptr, &log_size);
    
    // Прочитать BUILD LOG
    std::string build_log;
    if (log_size > 1) {  // log_size включает null terminator
        std::vector<char> log_buffer(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 
                             log_size, log_buffer.data(), nullptr);
        build_log = std::string(log_buffer.begin(), log_buffer.end());
    }

    // Проверить результат компиляции
    if (err != CL_SUCCESS) {
        std::string error_msg = "❌ OpenCL kernel compilation FAILED\n";
        error_msg += "Error code: " + std::to_string(err);
        
        if (err == -11) {
            error_msg += " (CL_BUILD_PROGRAM_FAILURE - Kernel syntax error)\n";
        }
        
        if (!build_log.empty()) {
            error_msg += "\n📋 BUILD LOG:\n";
            error_msg += "═════════════════════════════════════════\n";
            error_msg += build_log;
            error_msg += "═════════════════════════════════════════\n";
        }
        
        throw std::runtime_error(error_msg);
    }

    // Если есть warnings (но err == CL_SUCCESS)
    if (!build_log.empty() && build_log.find("warning") != std::string::npos) {
        std::cout << "⚠️  Compilation warnings:\n" << build_log << std::endl;
    } else if (!build_log.empty()) {
        std::cout << "📋 Build log:\n" << build_log << std::endl;
    }

    // Создать kernels
    std::cout << "📦 Creating kernel objects..." << std::endl;
    
    kernel_lfm_basic_ = clCreateKernel(program_, "kernel_lfm_basic", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel_lfm_basic (err: " + 
                                std::to_string(err) + ")");
    }

    kernel_lfm_delayed_ = clCreateKernel(program_, "kernel_lfm_delayed", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel_lfm_delayed (err: " + 
                                std::to_string(err) + ")");
    }
    
    std::cout << "✅ OpenCL kernels compiled and created successfully!" << std::endl;
}

// ═══════════════════════════════════════════════════════════════════════════
// ИСПРАВЛЕННЫЙ МЕТОД GetKernelSource()
// СТРУКТУРА В НАЧАЛЕ + __global вместо __constant
// ═══════════════════════════════════════════════════════════════════════════

std::string GeneratorGPU::GetKernelSource() const {
    // Встроенный kernel код с ПРАВИЛЬНЫМ порядком определений
    return R"(

// ═════════════════════════════════════════════════════════════════════
// СТРУКТУРЫ (ОПРЕДЕЛИТЬ В НАЧАЛЕ!)
// ═════════════════════════════════════════════════════════════════════

typedef struct {
    uint beam_index;
    float delay_degrees;
} DelayParam;

// ═════════════════════════════════════════════════════════════════════
// KERNEL 1: БАЗОВЫЙ ЛЧМ СИГНАЛ
// ═════════════════════════════════════════════════════════════════════

__kernel void kernel_lfm_basic(
    __global float2 *output,
    float f_start,
    float f_stop,
    float sample_rate,
    float duration,
    uint num_samples,
    uint num_beams
) {
    uint gid = get_global_id(0);
    if (gid >= (uint)num_samples * num_beams) return;

    uint ray_id = gid / num_samples;
    uint sample_id = gid % num_samples;
    if (ray_id >= num_beams || sample_id >= num_samples) return;

    // Время для этого отсчёта
    float t = (float)sample_id / sample_rate;

    // Параметры ЛЧМ
    float chirp_rate = (f_stop - f_start) / duration;

    // Фаза: φ(t) = 2π(f_start * t + 0.5 * chirp_rate * t²)
    float phase = 2.0f * 3.14159265f * (
        f_start * t + 0.5f * chirp_rate * t * t
    );

    // Комплексный сигнал: cos(φ) + i*sin(φ)
    float real = cos(phase);
    float imag = sin(phase);

    // Записать в выход
    uint out_idx = ray_id * num_samples + sample_id;
    output[out_idx] = (float2)(real, imag);
}

// ═════════════════════════════════════════════════════════════════════
// KERNEL 2: ЛЧМ СИГНАЛ С ДРОБНОЙ ЗАДЕРЖКОЙ
// ═════════════════════════════════════════════════════════════════════

__kernel void kernel_lfm_delayed(
    __global float2 *output,
    __global DelayParam *m_delay,  // ← __global вместо __constant
    float f_start,
    float f_stop,
    float sample_rate,
    float duration,
    float speed_of_light,
    uint num_samples,
    uint num_beams,
    uint num_delays
) {
    uint gid = get_global_id(0);
    if (gid >= (uint)num_samples * num_beams) return;

    uint ray_id = gid / num_samples;
    uint sample_id = gid % num_samples;
    if (ray_id >= num_beams || sample_id >= num_samples) return;

    // Получить задержку для этого луча
    float delay_degrees = m_delay[ray_id].delay_degrees;

    // Конвертировать градусы в секунды задержки
    float f_center = (f_start + f_stop) / 2.0f;
    float wavelength = speed_of_light / f_center;
    float delay_rad = delay_degrees * 3.14159265f / 180.0f;
    float delay_time = delay_rad * wavelength / speed_of_light;
    float delay_samples = delay_time * sample_rate;

    // Вычислить эффективный индекс с задержкой
    int delayed_sample_int = (int)sample_id - (int)delay_samples;

    float real, imag;
    if (delayed_sample_int < 0) {
        // До начала сигнала - нули
        real = 0.0f;
        imag = 0.0f;
    } else {
        // Время для задержанного отсчёта
        float t = (float)delayed_sample_int / sample_rate;

        // Параметры ЛЧМ
        float chirp_rate = (f_stop - f_start) / duration;

        // Фаза
        float phase = 2.0f * 3.14159265f * (
            f_start * t + 0.5f * chirp_rate * t * t
        );

        real = cos(phase);
        imag = sin(phase);
    }

    // Записать в выход
    uint out_idx = ray_id * num_samples + sample_id;
    output[out_idx] = (float2)(real, imag);
}

)";
}

// ═════════════════════════════════════════════════════════════════════════════
// ИСПРАВЛЕННЫЙ МЕТОД signal_base() - использовать cl_float2
// ═════════════════════════════════════════════════════════════════════════════

cl_mem GeneratorGPU::signal_base() {
    cl_int err = CL_SUCCESS;

    // Создать буфер для вывода (использовать cl_float2, не std::complex)
    size_t buffer_size = total_size_ * sizeof(cl_float2);
    
    std::cout << "📦 Allocating GPU buffer for signal_base: " 
              << (buffer_size / (1024.0 * 1024.0)) << " MB" << std::endl;

    cl_mem output = clCreateBuffer(
        context_,
        CL_MEM_WRITE_ONLY,
        buffer_size,
        nullptr,
        &err
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to allocate GPU buffer for signal_base (err: " + 
                                std::to_string(err) + ")");
    }

    // Установить аргументы kernel
    clSetKernelArg(kernel_lfm_basic_, 0, sizeof(cl_mem), &output);
    clSetKernelArg(kernel_lfm_basic_, 1, sizeof(float), &params_.f_start);
    clSetKernelArg(kernel_lfm_basic_, 2, sizeof(float), &params_.f_stop);
    clSetKernelArg(kernel_lfm_basic_, 3, sizeof(float), &params_.sample_rate);
    clSetKernelArg(kernel_lfm_basic_, 4, sizeof(float), &params_.duration);

    uint num_samples = static_cast<uint>(num_samples_);
    uint num_beams = static_cast<uint>(num_beams_);

    clSetKernelArg(kernel_lfm_basic_, 5, sizeof(uint), &num_samples);
    clSetKernelArg(kernel_lfm_basic_, 6, sizeof(uint), &num_beams);

    // Запустить kernel
    size_t global_work_size = total_size_;
    size_t local_work_size = 256;

    std::cout << "⚙️  Executing kernel_lfm_basic (grid: " << global_work_size 
              << ", block: " << local_work_size << ")" << std::endl;

    err = clEnqueueNDRangeKernel(
        queue_,
        kernel_lfm_basic_,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr, nullptr
    );

    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("Failed to enqueue kernel_lfm_basic (err: " + 
                                std::to_string(err) + ")");
    }

    // Дождаться завершения
    clFinish(queue_);

    std::cout << "✅ signal_base() completed. GPU buffer: " 
              << (buffer_size / (1024 * 1024)) << " MB" << std::endl;

    return output;
}

// ═════════════════════════════════════════════════════════════════════════════
// ИСПРАВЛЕННЫЙ МЕТОД signal_valedation() - использовать cl_float2
// ═════════════════════════════════════════════════════════════════════════════

cl_mem GeneratorGPU::signal_valedation(
    const DelayParameter* m_delay,
    size_t num_delay_params
) {
    if (m_delay == nullptr) {
        throw std::invalid_argument("m_delay array is null");
    }

    if (num_delay_params != num_beams_) {
        throw std::invalid_argument(
            "num_delay_params must equal num_beams (" +
            std::to_string(num_beams_) + ")"
        );
    }

    cl_int err = CL_SUCCESS;

    // Создать буфер для вывода
    size_t buffer_size = total_size_ * sizeof(cl_float2);
    
    std::cout << "📦 Allocating GPU buffer for signal_valedation: " 
              << (buffer_size / (1024.0 * 1024.0)) << " MB" << std::endl;

    cl_mem output = clCreateBuffer(
        context_,
        CL_MEM_WRITE_ONLY,
        buffer_size,
        nullptr,
        &err
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to allocate GPU buffer for signal_valedation (err: " + 
                                std::to_string(err) + ")");
    }

    // Создать буфер для параметров задержки на GPU
    size_t delay_buffer_size = num_delay_params * sizeof(DelayParameter);
    
    std::cout << "📦 Allocating GPU buffer for delay parameters: " 
              << (delay_buffer_size / 1024.0) << " KB" << std::endl;

    cl_mem delay_buffer = clCreateBuffer(
        context_,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        delay_buffer_size,
        const_cast<DelayParameter*>(m_delay),
        &err
    );

    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("Failed to allocate GPU buffer for delay parameters (err: " + 
                                std::to_string(err) + ")");
    }

    // Установить аргументы kernel
    clSetKernelArg(kernel_lfm_delayed_, 0, sizeof(cl_mem), &output);
    clSetKernelArg(kernel_lfm_delayed_, 1, sizeof(cl_mem), &delay_buffer);
    clSetKernelArg(kernel_lfm_delayed_, 2, sizeof(float), &params_.f_start);
    clSetKernelArg(kernel_lfm_delayed_, 3, sizeof(float), &params_.f_stop);
    clSetKernelArg(kernel_lfm_delayed_, 4, sizeof(float), &params_.sample_rate);
    clSetKernelArg(kernel_lfm_delayed_, 5, sizeof(float), &params_.duration);

    float speed_of_light = 3.0e8f;
    clSetKernelArg(kernel_lfm_delayed_, 6, sizeof(float), &speed_of_light);

    uint num_samples = static_cast<uint>(num_samples_);
    uint num_beams = static_cast<uint>(num_beams_);
    uint num_delays = static_cast<uint>(num_delay_params);

    clSetKernelArg(kernel_lfm_delayed_, 7, sizeof(uint), &num_samples);
    clSetKernelArg(kernel_lfm_delayed_, 8, sizeof(uint), &num_beams);
    clSetKernelArg(kernel_lfm_delayed_, 9, sizeof(uint), &num_delays);

    // Запустить kernel
    size_t global_work_size = total_size_;
    size_t local_work_size = 256;

    std::cout << "⚙️  Executing kernel_lfm_delayed (grid: " << global_work_size 
              << ", block: " << local_work_size << ")" << std::endl;

    err = clEnqueueNDRangeKernel(
        queue_,
        kernel_lfm_delayed_,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr, nullptr
    );

    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        clReleaseMemObject(delay_buffer);
        throw std::runtime_error("Failed to enqueue kernel_lfm_delayed (err: " + 
                                std::to_string(err) + ")");
    }

    // Дождаться завершения
    clFinish(queue_);

    // Освободить буфер параметров (результат уже на GPU)
    clReleaseMemObject(delay_buffer);

    std::cout << "✅ signal_valedation() completed. GPU buffer: " 
              << (buffer_size / (1024 * 1024)) << " MB" << std::endl;

    return output;
}

void GeneratorGPU::ClearGPU() {
    // Очистка временных буферов (в данной реализации они освобождаются после use)
    // Основные буферы результатов остаются на GPU для дальнейшего использования
    clFinish(queue_);
}

} // namespace radar

/** 
void GeneratorGPU::CompileKernels() {
    cl_int err = CL_SUCCESS;
    
    // Получить исходный код kernels
    std::string source = GetKernelSource();
    const char* source_str = source.c_str();
    size_t source_len = source.length();
    
    // Создать программу
    program_ = clCreateProgramWithSource(context_, 1, &source_str, &source_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL program");
    }
    
    // Скомпилировать
    err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Получить лог ошибки
        size_t log_size = 0;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        throw std::runtime_error("Failed to build OpenCL program:\n" + log);
    }
    
    // Создать kernels
    kernel_lfm_basic_ = clCreateKernel(program_, "kernel_lfm_basic", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel_lfm_basic");
    }
    
    kernel_lfm_delayed_ = clCreateKernel(program_, "kernel_lfm_delayed", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel_lfm_delayed");
    }
}

std::string GeneratorGPU::GetKernelSource() const {
    // Встроенный kernel код (лучше чем грузить с файла)
    return R"(
        // ═════════════════════════════════════════════════════════════════
        // KERNEL 1: БАЗОВЫЙ ЛЧМ СИГНАЛ
        // ═════════════════════════════════════════════════════════════════
        
        __kernel void kernel_lfm_basic(
            __global float2 *output,        // Выход: [ray0_all][ray1_all]...
            float f_start,                  // Начальная частота (Гц)
            float f_stop,                   // Конечная частота (Гц)
            float sample_rate,              // Частота дискретизации
            float duration,                 // Длительность сигнала (сек)
            uint num_samples,               // Количество отсчётов на луч
            uint num_beams                  // Количество лучей
        ) {
            uint gid = get_global_id(0);    // Глобальный индекс потока
            
            if (gid >= (uint)num_samples * num_beams) return;
            
            // Распределить индексы: каждый поток обрабатывает один отсчёт одного луча
            uint ray_id = gid / num_samples;
            uint sample_id = gid % num_samples;
            
            if (ray_id >= num_beams || sample_id >= num_samples) return;
            
            // Время для этого отсчёта (одинаково для всех лучей)
            float t = (float)sample_id / sample_rate;
            
            // Параметры ЛЧМ
            float chirp_rate = (f_stop - f_start) / duration;
            
            // Фаза: φ(t) = 2π(f_start * t + 0.5 * chirp_rate * t²)
            float phase = 2.0f * 3.14159265f * (
                f_start * t + 0.5f * chirp_rate * t * t
            );
            
            // Комплексный сигнал: cos(φ) + i*sin(φ)
            float real = cos(phase);
            float imag = sin(phase);
            
            // Записать в выход
            uint out_idx = ray_id * num_samples + sample_id;
            output[out_idx] = (float2)(real, imag);
        }
        
        // ═════════════════════════════════════════════════════════════════
        // KERNEL 2: ЛЧМ СИГНАЛ С ДРОБНОЙ ЗАДЕРЖКОЙ
        // ═════════════════════════════════════════════════════════════════
        
        __kernel void kernel_lfm_delayed(
            __global float2 *output,                // Выход: [ray0_delayed][ray1_delayed]...
            __constant DelayParam *m_delay,         // Входные задержки {beam_id, delay_deg}
            float f_start,                          // Начальная частота (Гц)
            float f_stop,                           // Конечная частота (Гц)
            float sample_rate,                      // Частота дискретизации
            float duration,                         // Длительность сигнала (сек)
            float speed_of_light,                   // Скорость света (м/с)
            uint num_samples,                       // Количество отсчётов
            uint num_beams,                         // Количество лучей
            uint num_delays                         // Количество параметров задержки
        ) {
            uint gid = get_global_id(0);
            
            if (gid >= (uint)num_samples * num_beams) return;
            
            uint ray_id = gid / num_samples;
            uint sample_id = gid % num_samples;
            
            if (ray_id >= num_beams || sample_id >= num_samples) return;
            
            // Получить задержку для этого луча
            float delay_degrees = m_delay[ray_id].delay_degrees;
            
            // Конвертировать градусы в секунды задержки
            // delay_rad = delay_degrees * π / 180
            // delay_time = delay_rad * wavelength / speed_of_light
            float f_center = (f_start + f_stop) / 2.0f;
            float wavelength = speed_of_light / f_center;
            float delay_rad = delay_degrees * 3.14159265f / 180.0f;
            float delay_time = delay_rad * wavelength / speed_of_light;
            float delay_samples = delay_time * sample_rate;
            
            // Вычислить эффективный индекс с задержкой
            int delayed_sample_int = (int)sample_id - (int)delay_samples;
            
            float real, imag;
            
            if (delayed_sample_int < 0) {
                // До начала сигнала - нули
                real = 0.0f;
                imag = 0.0f;
            } else {
                // Время для задержанного отсчёта
                float t = (float)delayed_sample_int / sample_rate;
                
                // Параметры ЛЧМ
                float chirp_rate = (f_stop - f_start) / duration;
                
                // Фаза
                float phase = 2.0f * 3.14159265f * (
                    f_start * t + 0.5f * chirp_rate * t * t
                );
                
                real = cos(phase);
                imag = sin(phase);
            }
            
            // Записать в выход
            uint out_idx = ray_id * num_samples + sample_id;
            output[out_idx] = (float2)(real, imag);
        }
        
        // Структура для параметра задержки (должна совпадать с C++)
        typedef struct {
            uint beam_index;
            float delay_degrees;
        } DelayParam;
    )";
}

// ═════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═════════════════════════════════════════════════════════════════════

cl_mem GeneratorGPU::signal_base() {
    cl_int err = CL_SUCCESS;
    
    // Создать буфер для вывода
    size_t buffer_size = total_size_ * sizeof(std::complex<float>);
    cl_mem output = clCreateBuffer(
        context_,
        CL_MEM_WRITE_ONLY,
        buffer_size,
        nullptr,
        &err
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to allocate GPU buffer for signal_base");
    }
    
    // Установить аргументы kernel
    clSetKernelArg(kernel_lfm_basic_, 0, sizeof(cl_mem), &output);
    clSetKernelArg(kernel_lfm_basic_, 1, sizeof(float), &params_.f_start);
    clSetKernelArg(kernel_lfm_basic_, 2, sizeof(float), &params_.f_stop);
    clSetKernelArg(kernel_lfm_basic_, 3, sizeof(float), &params_.sample_rate);
    clSetKernelArg(kernel_lfm_basic_, 4, sizeof(float), &params_.duration);
    
    uint num_samples = static_cast<uint>(num_samples_);
    uint num_beams = static_cast<uint>(num_beams_);
    clSetKernelArg(kernel_lfm_basic_, 5, sizeof(uint), &num_samples);
    clSetKernelArg(kernel_lfm_basic_, 6, sizeof(uint), &num_beams);
    
    // Запустить kernel
    size_t global_work_size = total_size_;
    size_t local_work_size = 256;  // Оптимальный размер для GPU
    
    err = clEnqueueNDRangeKernel(
        queue_,
        kernel_lfm_basic_,
        1,          // одномерная сетка
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("Failed to enqueue kernel_lfm_basic");
    }
    
    // Дождаться завершения
    clFinish(queue_);
    
    std::cout << "✓ signal_base() завершен. GPU memory: " << buffer_size / (1024*1024) 
              << " MB" << std::endl;
    
    return output;  // Возвращаем адрес GPU памяти
}

cl_mem GeneratorGPU::signal_valedation(
    const DelayParameter* m_delay,
    size_t num_delay_params
) {
    if (m_delay == nullptr) {
        throw std::invalid_argument("m_delay array is null");
    }
    
    if (num_delay_params != num_beams_) {
        throw std::invalid_argument(
            "num_delay_params must equal num_beams (" +
            std::to_string(num_beams_) + ")"
        );
    }
    
    cl_int err = CL_SUCCESS;
    
    // Создать буфер для вывода
    size_t buffer_size = total_size_ * sizeof(std::complex<float>);
    cl_mem output = clCreateBuffer(
        context_,
        CL_MEM_WRITE_ONLY,
        buffer_size,
        nullptr,
        &err
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to allocate GPU buffer for signal_valedation");
    }
    
    // Создать буфер для параметров задержки на GPU
    size_t delay_buffer_size = num_delay_params * sizeof(DelayParameter);
    cl_mem delay_buffer = clCreateBuffer(
        context_,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        delay_buffer_size,
        const_cast<DelayParameter*>(m_delay),
        &err
    );
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("Failed to allocate GPU buffer for delay parameters");
    }
    
    // Установить аргументы kernel
    clSetKernelArg(kernel_lfm_delayed_, 0, sizeof(cl_mem), &output);
    clSetKernelArg(kernel_lfm_delayed_, 1, sizeof(cl_mem), &delay_buffer);
    clSetKernelArg(kernel_lfm_delayed_, 2, sizeof(float), &params_.f_start);
    clSetKernelArg(kernel_lfm_delayed_, 3, sizeof(float), &params_.f_stop);
    clSetKernelArg(kernel_lfm_delayed_, 4, sizeof(float), &params_.sample_rate);
    clSetKernelArg(kernel_lfm_delayed_, 5, sizeof(float), &params_.duration);
    
    float speed_of_light = 3.0e8f;
    clSetKernelArg(kernel_lfm_delayed_, 6, sizeof(float), &speed_of_light);
    
    uint num_samples = static_cast<uint>(num_samples_);
    uint num_beams = static_cast<uint>(num_beams_);
    uint num_delays = static_cast<uint>(num_delay_params);
    clSetKernelArg(kernel_lfm_delayed_, 7, sizeof(uint), &num_samples);
    clSetKernelArg(kernel_lfm_delayed_, 8, sizeof(uint), &num_beams);
    clSetKernelArg(kernel_lfm_delayed_, 9, sizeof(uint), &num_delays);
    
    // Запустить kernel
    size_t global_work_size = total_size_;
    size_t local_work_size = 256;
    
    err = clEnqueueNDRangeKernel(
        queue_,
        kernel_lfm_delayed_,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        clReleaseMemObject(delay_buffer);
        throw std::runtime_error("Failed to enqueue kernel_lfm_delayed");
    }
    
    // Дождаться завершения
    clFinish(queue_);
    
    // Освободить буфер параметров (результат уже на GPU)
    clReleaseMemObject(delay_buffer);
    
    std::cout << "✓ signal_valedation() завершен. GPU memory: " << buffer_size / (1024*1024) 
              << " MB" << std::endl;
    
    return output;  // Возвращаем адрес GPU памяти
}
  
  
 * 
*/