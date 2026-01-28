/**
 * @file fractional_delay_processor.cpp
 * @brief Реализация процессора дробной задержки для LFM сигналов на GPU
 * 
 * @details OpenCL kernel использует матрицу Лагранжа 48×5 для 5-точечной
 *          интерполяции. Формат данных: [beam * samples + sample].
 *          IN-PLACE обработка через двойную буферизацию.
 * 
 * @author LCH-Farrow01 Project
 * @version 2.0
 * @date 2026-01-21
 */

#include "GPU/fractional_delay_processor.hpp"
#include "ManagerOpenCL/opencl_compute_engine.hpp"
#include "ManagerOpenCL/opencl_core.hpp"
#include "ManagerOpenCL/command_queue_pool.hpp"
#include "ManagerOpenCL/gpu_memory_buffer.hpp"
#include "ManagerOpenCL/i_memory_buffer.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <nlohmann/json.hpp>

namespace radar {

// ============================================================================
// ЗАГРУЗКА МАТРИЦЫ ЛАГРАНЖА ИЗ JSON
// ============================================================================

LagrangeMatrix LagrangeMatrix::LoadFromJSON(const std::string& filepath) {
    LagrangeMatrix matrix;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    try {
        nlohmann::json json_data;
        file >> json_data;
        
        // Поддерживаемые форматы:
        // 1. {"lagrange_matrix": [[...], [...], ...]}
        // 2. {"data": [[...], [...], ...]}
        // 3. просто [[...], [...], ...]
        nlohmann::json coeffs;
        if (json_data.contains("lagrange_matrix")) {
            coeffs = json_data["lagrange_matrix"];
        } else if (json_data.contains("data")) {
            coeffs = json_data["data"];
        } else if (json_data.is_array()) {
            coeffs = json_data;
        } else {
            throw std::runtime_error("Unknown JSON format: expected 'lagrange_matrix', 'data' or array");
        }
        
        if (!coeffs.is_array() || coeffs.size() != LAGRANGE_ROWS) {
            throw std::runtime_error("Invalid matrix dimensions: expected " + 
                                    std::to_string(LAGRANGE_ROWS) + " rows, got " +
                                    std::to_string(coeffs.size()));
        }
        
        for (uint32_t row = 0; row < LAGRANGE_ROWS; ++row) {
            if (!coeffs[row].is_array() || coeffs[row].size() != LAGRANGE_COLS) {
                throw std::runtime_error("Invalid row " + std::to_string(row) + 
                                        ": expected " + std::to_string(LAGRANGE_COLS) + " columns");
            }
            
            for (uint32_t col = 0; col < LAGRANGE_COLS; ++col) {
                matrix.coefficients[row][col] = coeffs[row][col].get<float>();
            }
        }
        
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }
    
    if (!matrix.IsValid()) {
        throw std::runtime_error("Loaded matrix failed validation");
    }
    
    return matrix;
}

// ============================================================================
// ВСТРОЕННЫЙ KERNEL КОД
// ============================================================================

std::string FractionalDelayProcessor::GetKernelSource() {
    return R"CL(
// ============================================================================
// FRACTIONAL DELAY KERNEL v2.0
// ============================================================================
// 
// Дробная задержка LFM сигналов с использованием матрицы Лагранжа 48×5
// 
// ФОРМАТ ДАННЫХ:
//   buffer[beam_idx * num_samples + sample_idx]
//   Луч за лучом: beam0[sample0, sample1, ...], beam1[sample0, sample1, ...], ...
//
// IN-PLACE через двойную буферизацию:
//   1. Читаем из input_buffer, пишем в temp_buffer
//   2. Копируем temp_buffer → input_buffer
//
// ПАРАМЕТРЫ ЗАДЕРЖКИ (DelayParams):
//   - delay_integer: целая часть задержки в отсчётах (может быть < 0)
//   - lagrange_row: строка матрицы Лагранжа [0..47] для дробной части
//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Константы
#define LAGRANGE_ROWS 48
#define LAGRANGE_COLS 5

// ============================================================================
// Структуры данных (должны совпадать с C++)
// ============================================================================

/// Комплексное число
typedef struct {
    float real;
    float imag;
} Complex;

/// Параметры задержки для одного луча
typedef struct {
    int delay_integer;     // Целая часть задержки (отсчёты)
    uint lagrange_row;     // Строка матрицы Лагранжа [0..47]
} DelayParams;

// ============================================================================
// ОСНОВНОЙ KERNEL: Применение дробной задержки
// ============================================================================
/**
 * @brief Применить дробную задержку ко всем лучам
 * 
 * @param input_buffer    - Входной буфер (num_beams × num_samples complex)
 * @param output_buffer   - Выходной буфер (для IN-PLACE используется temp)
 * @param lagrange_matrix - Матрица Лагранжа [48][5] в row-major
 * @param delay_params    - Параметры задержки для каждого луча [num_beams]
 * @param num_beams       - Количество лучей
 * @param num_samples     - Количество отсчётов в каждом луче
 */
__kernel void fractional_delay_kernel(
    __global const Complex* input_buffer,
    __global Complex* output_buffer,
    __global const float* lagrange_matrix,   // [48][5] row-major
    __global const DelayParams* delay_params,
    const uint num_beams,
    const uint num_samples
) {
    // Глобальный индекс = beam_idx * num_samples + sample_idx
    uint gid = get_global_id(0);
    
    // Определить луч и позицию внутри луча
    uint beam_idx = gid / num_samples;
    uint sample_idx = gid % num_samples;
    
    // Проверка границ
    if (beam_idx >= num_beams) {
        return;
    }
    
    // Получить параметры задержки для этого луча
    DelayParams dp = delay_params[beam_idx];
    int delay_int = dp.delay_integer;
    uint lag_row = dp.lagrange_row;  // [0..47]
    
    // Загрузить коэффициенты Лагранжа для этой дробной части
    // lagrange_matrix[lag_row * 5 + col]
    float L0 = lagrange_matrix[lag_row * LAGRANGE_COLS + 0];
    float L1 = lagrange_matrix[lag_row * LAGRANGE_COLS + 1];
    float L2 = lagrange_matrix[lag_row * LAGRANGE_COLS + 2];
    float L3 = lagrange_matrix[lag_row * LAGRANGE_COLS + 3];
    float L4 = lagrange_matrix[lag_row * LAGRANGE_COLS + 4];
    
    // Вычислить исходный индекс с учётом целой задержки
    // Для 5-точечной интерполяции Лагранжа:
    // - При frac=0 (row=0): L1=1.0, значит center соответствует s1
    // - Окно: [center-1, center, center+1, center+2, center+3]
    int center = (int)sample_idx - delay_int;
    
    // Индексы 5 точек для интерполяции (сдвинуты на +1 относительно стандартного)
    // L0 → s0 = input[center-1]
    // L1 → s1 = input[center]    ← ЦЕНТР (L1=1.0 при frac=0)
    // L2 → s2 = input[center+1]
    // L3 → s3 = input[center+2]
    // L4 → s4 = input[center+3]
    int idx0 = center - 1;
    int idx1 = center;      // Центральная точка (L1=1.0 при frac=0)
    int idx2 = center + 1;
    int idx3 = center + 2;
    int idx4 = center + 3;
    
    // Смещение в буфере для этого луча
    uint beam_offset = beam_idx * num_samples;
    
    // Функция безопасного чтения (с граничным условием = 0)
    // Используем макрос для inline
    #define SAFE_READ(idx) \
        (((idx) >= 0 && (idx) < (int)num_samples) ? \
         input_buffer[beam_offset + (idx)] : (Complex){0.0f, 0.0f})
    
    // Читаем 5 точек
    Complex s0 = SAFE_READ(idx0);
    Complex s1 = SAFE_READ(idx1);
    Complex s2 = SAFE_READ(idx2);
    Complex s3 = SAFE_READ(idx3);
    Complex s4 = SAFE_READ(idx4);
    
    #undef SAFE_READ
    
    // 5-точечная интерполяция Лагранжа:
    // result = L0*s0 + L1*s1 + L2*s2 + L3*s3 + L4*s4
    Complex result;
    result.real = L0 * s0.real + L1 * s1.real + L2 * s2.real + 
                  L3 * s3.real + L4 * s4.real;
    result.imag = L0 * s0.imag + L1 * s1.imag + L2 * s2.imag + 
                  L3 * s3.imag + L4 * s4.imag;
    
    // Записать результат
    output_buffer[gid] = result;
}

// ============================================================================
// KERNEL: Копирование буфера (для IN-PLACE)
// ============================================================================
__kernel void copy_buffer_kernel(
    __global const Complex* src,
    __global Complex* dst,
    const uint total_elements
) {
    uint gid = get_global_id(0);
    if (gid < total_elements) {
        dst[gid] = src[gid];
    }
}

)CL";
}

// ============================================================================
// КОНСТРУКТОР
// ============================================================================

FractionalDelayProcessor::FractionalDelayProcessor(
    const FractionalDelayConfig& config,
    const LagrangeMatrix& lagrange_matrix
)
    : config_(config),
      lagrange_matrix_(lagrange_matrix),
      engine_(nullptr),
      context_(nullptr),
      queue_(nullptr),
      device_(nullptr),
      kernel_(nullptr),
      program_(nullptr),
      total_samples_processed_(0),
      total_calls_(0)
{
    // Валидация конфигурации
    if (!config_.IsValid()) {
        throw std::invalid_argument("FractionalDelayConfig: invalid parameters");
    }
    
    // Валидация матрицы
    if (!lagrange_matrix_.IsValid()) {
        throw std::invalid_argument("LagrangeMatrix: invalid matrix");
    }
    
    // Проверка инициализации OpenCL
    if (!ManagerOpenCL::OpenCLComputeEngine::IsInitialized()) {
        ManagerOpenCL::OpenCLComputeEngine::Initialize(ManagerOpenCL::DeviceType::GPU);
        
        if (!ManagerOpenCL::OpenCLComputeEngine::IsInitialized()) {
            throw std::runtime_error(
                "OpenCLComputeEngine not initialized. Call Initialize() first."
            );
        }
    }
    
    // Инициализация
    Initialize();
    
    if (config_.verbose) {
        PrintInfo();
    }
}

// ============================================================================
// ДЕСТРУКТОР
// ============================================================================

FractionalDelayProcessor::~FractionalDelayProcessor() {
    if (kernel_) {
        clReleaseKernel(kernel_);
    }
    if (program_) {
        clReleaseProgram(program_);
    }
    buffer_lagrange_.reset();
    buffer_delays_.reset();
    buffer_temp_.reset();
}

// ============================================================================
// MOVE СЕМАНТИКА
// ============================================================================

FractionalDelayProcessor::FractionalDelayProcessor(
    FractionalDelayProcessor&& other
) noexcept
    : config_(other.config_),
      lagrange_matrix_(std::move(other.lagrange_matrix_)),
      engine_(other.engine_),
      context_(other.context_),
      queue_(other.queue_),
      device_(other.device_),
      kernel_(other.kernel_),
      program_(other.program_),
      buffer_lagrange_(std::move(other.buffer_lagrange_)),
      buffer_delays_(std::move(other.buffer_delays_)),
      buffer_temp_(std::move(other.buffer_temp_)),
      last_profiling_(other.last_profiling_),
      total_samples_processed_(other.total_samples_processed_),
      total_calls_(other.total_calls_)
{
    other.kernel_ = nullptr;
    other.program_ = nullptr;
}

FractionalDelayProcessor& FractionalDelayProcessor::operator=(
    FractionalDelayProcessor&& other
) noexcept {
    if (this != &other) {
        if (kernel_) clReleaseKernel(kernel_);
        if (program_) clReleaseProgram(program_);
        
        config_ = other.config_;
        lagrange_matrix_ = std::move(other.lagrange_matrix_);
        engine_ = other.engine_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        kernel_ = other.kernel_;
        program_ = other.program_;
        buffer_lagrange_ = std::move(other.buffer_lagrange_);
        buffer_delays_ = std::move(other.buffer_delays_);
        buffer_temp_ = std::move(other.buffer_temp_);
        last_profiling_ = other.last_profiling_;
        total_samples_processed_ = other.total_samples_processed_;
        total_calls_ = other.total_calls_;
        
        other.kernel_ = nullptr;
        other.program_ = nullptr;
    }
    return *this;
}

// ============================================================================
// ИНИЦИАЛИЗАЦИЯ
// ============================================================================

void FractionalDelayProcessor::Initialize() {
    if (config_.verbose) {
        std::cout << "[FDP] Инициализация FractionalDelayProcessor...\n";
    }
    
    // Получить OpenCL объекты
    engine_ = &ManagerOpenCL::OpenCLComputeEngine::GetInstance();
    auto& core = ManagerOpenCL::OpenCLCore::GetInstance();
    context_ = core.GetContext();
    device_ = core.GetDevice();
    queue_ = ManagerOpenCL::CommandQueuePool::GetNextQueue();
    
    // Загрузить kernel
    LoadKernel();
    
    // Создать буферы
    CreateBuffers();
    
    // Загрузить матрицу Лагранжа на GPU
    UploadLagrangeMatrix();
    
    // Инициализировать профилирование
    last_profiling_ = {};
    
    if (config_.verbose) {
        std::cout << "[FDP] Инициализация завершена ✅\n";
    }
}

// ============================================================================
// ЗАГРУЗКА KERNEL
// ============================================================================

void FractionalDelayProcessor::LoadKernel() {
    if (config_.verbose) {
        std::cout << "[FDP] Загрузка kernel'а...\n";
    }
    
    std::string source = GetKernelSource();
    const char* src_ptr = source.c_str();
    size_t src_len = source.length();
    
    cl_int err;
    program_ = clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clCreateProgramWithSource failed: " + std::to_string(err));
    }
    
    err = clBuildProgram(program_, 1, &device_, "-cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Получить лог сборки
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        
        std::cerr << "[FDP] Kernel build error:\n" << log.data() << "\n";
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error("clBuildProgram failed");
    }
    
    kernel_ = clCreateKernel(program_, "fractional_delay_kernel", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error("clCreateKernel failed: " + std::to_string(err));
    }
    
    if (config_.verbose) {
        std::cout << "[FDP] Kernel загружен: fractional_delay_kernel ✅\n";
    }
}

// ============================================================================
// СОЗДАНИЕ БУФЕРОВ
// ============================================================================

void FractionalDelayProcessor::CreateBuffers() {
    if (config_.verbose) {
        std::cout << "[FDP] Создание GPU буферов...\n";
    }
    
    // Буфер для матрицы Лагранжа: 48 × 5 × sizeof(float) = 960 bytes
    size_t lagrange_size = LAGRANGE_ROWS * LAGRANGE_COLS * sizeof(float);
    
    // Буфер для параметров задержки: num_beams × sizeof(DelayParams)
    size_t delays_size = config_.num_beams * sizeof(DelayParams);
    
    // Временный буфер для IN-PLACE: num_beams × num_samples × sizeof(Complex)
    size_t temp_size = static_cast<size_t>(config_.num_beams) * config_.num_samples * sizeof(Complex);
    
    if (config_.verbose) {
        std::cout << "[FDP]   - Lagrange matrix: " << lagrange_size << " bytes\n";
        std::cout << "[FDP]   - Delay params:    " << delays_size << " bytes\n";
        std::cout << "[FDP]   - Temp buffer:     " << (temp_size / 1024.0 / 1024.0) << " MB\n";
    }
    
    // Создать буферы
    // Для простоты используем размер в complex элементах
    size_t lagrange_complex_size = (lagrange_size + sizeof(Complex) - 1) / sizeof(Complex);
    size_t delays_complex_size = (delays_size + sizeof(Complex) - 1) / sizeof(Complex);
    size_t temp_complex_size = static_cast<size_t>(config_.num_beams) * config_.num_samples;
    
    buffer_lagrange_ = engine_->CreateBuffer(lagrange_complex_size, ManagerOpenCL::MemoryType::GPU_READ_ONLY);
    buffer_delays_ = engine_->CreateBuffer(delays_complex_size, ManagerOpenCL::MemoryType::GPU_READ_ONLY);
    buffer_temp_ = engine_->CreateBuffer(temp_complex_size, ManagerOpenCL::MemoryType::GPU_READ_WRITE);
    
    if (config_.verbose) {
        std::cout << "[FDP] GPU буферы созданы ✅\n";
    }
}

// ============================================================================
// ЗАГРУЗКА МАТРИЦЫ ЛАГРАНЖА НА GPU
// ============================================================================

void FractionalDelayProcessor::UploadLagrangeMatrix() {
    if (config_.verbose) {
        std::cout << "[FDP] Загрузка матрицы Лагранжа на GPU...\n";
    }
    
    // Преобразовать матрицу в плоский массив (row-major)
    std::vector<float> flat_matrix(LAGRANGE_ROWS * LAGRANGE_COLS);
    for (uint32_t row = 0; row < LAGRANGE_ROWS; ++row) {
        for (uint32_t col = 0; col < LAGRANGE_COLS; ++col) {
            flat_matrix[row * LAGRANGE_COLS + col] = lagrange_matrix_.coefficients[row][col];
        }
    }
    
    // Загрузить на GPU
    cl_int err = clEnqueueWriteBuffer(
        queue_,
        buffer_lagrange_->Get(),
        CL_TRUE,  // Blocking
        0,
        flat_matrix.size() * sizeof(float),
        flat_matrix.data(),
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to upload Lagrange matrix: " + std::to_string(err));
    }
    
    if (config_.verbose) {
        std::cout << "[FDP] Матрица Лагранжа загружена ✅\n";
        std::cout << "[FDP]   Row 0: [" << flat_matrix[0] << ", " << flat_matrix[1] 
                  << ", " << flat_matrix[2] << ", " << flat_matrix[3] 
                  << ", " << flat_matrix[4] << "]\n";
    }
}

// ============================================================================
// СИНХРОНИЗАЦИЯ GPU
// ============================================================================

void FractionalDelayProcessor::SyncGPU() {
    cl_int err = clFinish(queue_);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("GPU sync failed: " + std::to_string(err));
    }
}

// ============================================================================
// ПРОФИЛИРОВАНИЕ СОБЫТИЯ
// ============================================================================

double FractionalDelayProcessor::ProfileEvent(cl_event event, const std::string& name) {
    if (!event || !config_.enable_profiling) return 0.0;
    
    cl_ulong start_time, end_time;
    cl_int err;
    
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
    if (err != CL_SUCCESS) return 0.0;
    
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
    if (err != CL_SUCCESS) return 0.0;
    
    double time_ms = (end_time - start_time) / 1e6;  // наносекунды → миллисекунды
    
    if (config_.verbose && !name.empty()) {
        std::cout << "[FDP] " << name << ": " << std::fixed << std::setprecision(4) 
                  << time_ms << " ms\n";
    }
    
    return time_ms;
}

// ============================================================================
// ОСНОВНАЯ ОБРАБОТКА - ИНДИВИДУАЛЬНЫЕ ЗАДЕРЖКИ
// ============================================================================

void FractionalDelayProcessor::Process(
    cl_mem gpu_buffer,
    const std::vector<DelayParams>& delays
) {
    if (delays.size() != config_.num_beams) {
        throw std::invalid_argument(
            "Delay count mismatch: expected " + std::to_string(config_.num_beams) +
            ", got " + std::to_string(delays.size())
        );
    }
    
    if (config_.verbose) {
        std::cout << "\n[FDP] Обработка " << config_.num_beams << " лучей × " 
                  << config_.num_samples << " отсчётов...\n";
    }
    
    cl_int err;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 1: Загрузить параметры задержек на GPU
    // ═══════════════════════════════════════════════════════════════════════════
    
    cl_event event_upload = nullptr;
    err = clEnqueueWriteBuffer(
        queue_,
        buffer_delays_->Get(),
        CL_FALSE,  // Non-blocking
        0,
        delays.size() * sizeof(DelayParams),
        delays.data(),
        0, nullptr,
        config_.enable_profiling ? &event_upload : nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to upload delay params: " + std::to_string(err));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 2: Установить аргументы kernel'а
    // ═══════════════════════════════════════════════════════════════════════════
    
    cl_mem temp_buf = buffer_temp_->Get();
    cl_mem lagrange_buf = buffer_lagrange_->Get();
    cl_mem delays_buf = buffer_delays_->Get();
    cl_uint num_beams = config_.num_beams;
    cl_uint num_samples = config_.num_samples;
    
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &gpu_buffer);      // input
    err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &temp_buf);       // output (temp)
    err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &lagrange_buf);   // lagrange matrix
    err |= clSetKernelArg(kernel_, 3, sizeof(cl_mem), &delays_buf);     // delay params
    err |= clSetKernelArg(kernel_, 4, sizeof(cl_uint), &num_beams);
    err |= clSetKernelArg(kernel_, 5, sizeof(cl_uint), &num_samples);
    
    if (err != CL_SUCCESS) {
        if (event_upload) clReleaseEvent(event_upload);
        throw std::runtime_error("Failed to set kernel args: " + std::to_string(err));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 3: Запустить kernel
    // ═══════════════════════════════════════════════════════════════════════════
    
    size_t total_work = static_cast<size_t>(num_beams) * num_samples;
    size_t local_size = config_.local_work_size;
    size_t global_size = ((total_work + local_size - 1) / local_size) * local_size;
    
    cl_event event_kernel = nullptr;
    cl_uint num_wait = event_upload ? 1 : 0;
    cl_event* wait_list = event_upload ? &event_upload : nullptr;
    
    err = clEnqueueNDRangeKernel(
        queue_,
        kernel_,
        1,                      // 1D
        nullptr,
        &global_size,
        &local_size,
        num_wait,
        wait_list,
        config_.enable_profiling ? &event_kernel : nullptr
    );
    
    if (err != CL_SUCCESS) {
        if (event_upload) clReleaseEvent(event_upload);
        throw std::runtime_error("clEnqueueNDRangeKernel failed: " + std::to_string(err));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 4: Копировать temp → input (IN-PLACE)
    // ═══════════════════════════════════════════════════════════════════════════
    
    cl_event event_copy = nullptr;
    cl_uint copy_wait = event_kernel ? 1 : 0;
    cl_event* copy_wait_list = event_kernel ? &event_kernel : nullptr;
    
    err = clEnqueueCopyBuffer(
        queue_,
        temp_buf,           // source
        gpu_buffer,         // destination
        0, 0,
        total_work * sizeof(Complex),
        copy_wait,
        copy_wait_list,
        config_.enable_profiling ? &event_copy : nullptr
    );
    
    if (err != CL_SUCCESS) {
        if (event_upload) clReleaseEvent(event_upload);
        if (event_kernel) clReleaseEvent(event_kernel);
        throw std::runtime_error("clEnqueueCopyBuffer failed: " + std::to_string(err));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 5: Дождаться завершения и профилировать
    // ═══════════════════════════════════════════════════════════════════════════
    
    clFinish(queue_);
    
    // Профилирование
    last_profiling_.upload_time_ms = ProfileEvent(event_upload, "Upload delays");
    last_profiling_.kernel_time_ms = ProfileEvent(event_kernel, "Kernel exec");
    double copy_time = ProfileEvent(event_copy, "Copy back");
    last_profiling_.total_time_ms = last_profiling_.upload_time_ms + 
                                     last_profiling_.kernel_time_ms + copy_time;
    last_profiling_.samples_processed = total_work;
    last_profiling_.beams_processed = num_beams;
    
    // Освободить события
    if (event_upload) clReleaseEvent(event_upload);
    if (event_kernel) clReleaseEvent(event_kernel);
    if (event_copy) clReleaseEvent(event_copy);
    
    // Статистика
    total_samples_processed_ += total_work;
    total_calls_++;
    
    if (config_.verbose) {
        std::cout << "[FDP] Обработка завершена ✅\n";
        std::cout << "[FDP]   Kernel time:  " << std::fixed << std::setprecision(4) 
                  << last_profiling_.kernel_time_ms << " ms\n";
        std::cout << "[FDP]   Total time:   " << last_profiling_.total_time_ms << " ms\n";
        std::cout << "[FDP]   Throughput:   " << std::setprecision(2) 
                  << last_profiling_.GetThroughput() / 1e6 << " Msamples/sec\n";
    }
}

// ============================================================================
// ОСНОВНАЯ ОБРАБОТКА - ОДИНАКОВАЯ ЗАДЕРЖКА
// ============================================================================

void FractionalDelayProcessor::Process(cl_mem gpu_buffer, const DelayParams& delay) {
    // Создать вектор одинаковых задержек
    std::vector<DelayParams> delays(config_.num_beams, delay);
    Process(gpu_buffer, delays);
}

// ============================================================================
// ОБРАБОТКА С ЗАДЕРЖКОЙ В ОТСЧЁТАХ
// ============================================================================

void FractionalDelayProcessor::ProcessWithDelay(cl_mem gpu_buffer, float delay_samples) {
    DelayParams dp = DelayParams::FromSamples(delay_samples);
    Process(gpu_buffer, dp);
}

// ============================================================================
// BATCH ОБРАБОТКА
// ============================================================================

void FractionalDelayProcessor::ProcessBatch(
    const std::vector<cl_mem>& buffers,
    const std::vector<std::vector<DelayParams>>& all_delays
) {
    if (buffers.size() != all_delays.size()) {
        throw std::invalid_argument("Buffers and delays count mismatch");
    }
    
    if (config_.verbose) {
        std::cout << "\n[FDP] Batch обработка " << buffers.size() << " буферов...\n";
    }
    
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (config_.verbose) {
            std::cout << "[FDP]   Buffer " << i << "...\n";
        }
        Process(buffers[i], all_delays[i]);
    }
    
    if (config_.verbose) {
        std::cout << "[FDP] Batch обработка завершена ✅\n";
    }
}

// ============================================================================
// ОБНОВЛЕНИЕ КОНФИГУРАЦИИ
// ============================================================================

void FractionalDelayProcessor::UpdateConfig(const FractionalDelayConfig& new_config) {
    if (!new_config.IsValid()) {
        throw std::invalid_argument("Invalid configuration");
    }
    
    bool need_rebuild = (config_.num_beams != new_config.num_beams ||
                         config_.num_samples != new_config.num_samples);
    
    config_ = new_config;
    
    if (need_rebuild) {
        buffer_delays_.reset();
        buffer_temp_.reset();
        CreateBuffers();
    }
}

// ============================================================================
// ДИАГНОСТИКА
// ============================================================================

void FractionalDelayProcessor::PrintInfo() const {
    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    std::cout << "  FractionalDelayProcessor v2.0\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
    std::cout << "Configuration:\n";
    std::cout << "  - Num beams:       " << config_.num_beams << "\n";
    std::cout << "  - Num samples:     " << config_.num_samples << "\n";
    std::cout << "  - Local work size: " << config_.local_work_size << "\n";
    std::cout << "  - Profiling:       " << (config_.enable_profiling ? "ON" : "OFF") << "\n";
    std::cout << "  - Verbose:         " << (config_.verbose ? "ON" : "OFF") << "\n";
    std::cout << "\n";
    std::cout << "Lagrange Matrix:\n";
    std::cout << "  - Rows:            " << LAGRANGE_ROWS << " (frac 0.00..0.98)\n";
    std::cout << "  - Cols:            " << LAGRANGE_COLS << " (5-point interp)\n";
    std::cout << "  - Valid:           " << (lagrange_matrix_.IsValid() ? "YES" : "NO") << "\n";
    std::cout << "\n";
    std::cout << "Memory:\n";
    size_t total_mem = LAGRANGE_ROWS * LAGRANGE_COLS * sizeof(float) +
                       config_.num_beams * sizeof(DelayParams) +
                       static_cast<size_t>(config_.num_beams) * config_.num_samples * sizeof(Complex);
    std::cout << "  - Total GPU:       " << (total_mem / 1024.0 / 1024.0) << " MB\n";
    std::cout << "\n";
    std::cout << "Statistics:\n";
    std::cout << "  - Total calls:     " << total_calls_ << "\n";
    std::cout << "  - Total samples:   " << total_samples_processed_ << "\n";
    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
}

std::string FractionalDelayProcessor::GetProfilingStats() const {
    std::ostringstream oss;
    oss << "\n";
    oss << "════════════════════════════════════════════════════════════════\n";
    oss << "  FractionalDelayProcessor Profiling\n";
    oss << "════════════════════════════════════════════════════════════════\n";
    oss << "  Upload time:     " << std::fixed << std::setprecision(4) 
        << last_profiling_.upload_time_ms << " ms\n";
    oss << "  Kernel time:     " << last_profiling_.kernel_time_ms << " ms\n";
    oss << "  Total time:      " << last_profiling_.total_time_ms << " ms\n";
    oss << "  Samples:         " << last_profiling_.samples_processed << "\n";
    oss << "  Beams:           " << last_profiling_.beams_processed << "\n";
    oss << "  Throughput:      " << std::setprecision(2) 
        << last_profiling_.GetThroughput() / 1e6 << " Msamples/sec\n";
    oss << "════════════════════════════════════════════════════════════════\n";
    return oss.str();
}

} // namespace radar
