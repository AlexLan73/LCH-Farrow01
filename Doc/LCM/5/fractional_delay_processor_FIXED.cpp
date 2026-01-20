#include "fractional_delay_processor.hpp"
#include "opencl_compute_engine.hpp"
#include "gpu_memory_buffer.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

namespace radar {

// ============================================================================
// ВСТРОЕННЫЙ KERNEL КОД (из kernel_fractional_delay_final.cl)
// ============================================================================

std::string FractionalDelayProcessor::GetKernelSource() {
    return R"CL(
// ============================================================================
// FRACTIONAL DELAY KERNEL - Дробная задержка для LFM сигналов
// ============================================================================
// 
// Назначение: Применить дробную задержку (Lagrange интерполяция)
// к комплексному вектору LFM сигналов на GPU
//
// ВХОДНЫЕ ДАННЫЕ (один вектор):
// - input_vector: комплексный вектор ВСЕ АНТЕННЫ x ВСЕ ТОЧКИ
//   Размер: num_beams * num_samples комплексных чисел
//   Формат: [Re0, Im0, Re1, Im1, ... Ren, Imn]
//
// ВЫХОДНЫЕ ДАННЫЕ (один вектор):
// - output_vector: обработанный комплексный вектор
//   Размер: num_beams * num_samples комплексных чисел
//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Параметры ядра (константы при компиляции)
#define ORDER 4  // Порядок интерполяции Лагранжа (4-5)

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

/// Структура комплексного числа
typedef struct {
    float real;
    float imag;
} Complex;

/// Умножение комплексных чисел: (a + bi)(c + di) = (ac-bd) + (ad+bc)i
Complex complex_mul(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

/// Сложение комплексных чисел
Complex complex_add(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

/// Комплексное число * скаляр
Complex complex_scale(Complex a, float scale) {
    Complex result;
    result.real = a.real * scale;
    result.imag = a.imag * scale;
    return result;
}

// ============================================================================
// ИНТЕРПОЛЯЦИЯ ЛАГРАНЖА
// ============================================================================

/**
 * Коэффициент базиса Лагранжа n-го порядка
 * L_j(x) = prod(x - x_i) / prod(x_j - x_i), i != j
 */
float lagrange_basis(float x, int j, int order) {
    float L = 1.0f;
    for (int i = 0; i < order; i++) {
        if (i != j) {
            L *= (x - (float)i) / ((float)j - (float)i);
        }
    }
    return L;
}

/**
 * Интерполяция Лагранжа для комплексного вектора
 * Вычисляет значение в точке x [0, order)
 * используя order точек данных
 */
Complex lagrange_interpolate(
    __local Complex* samples,  // order комплексных точек
    float x,                   // позиция для интерполяции [0, order)
    int order
) {
    Complex result;
    result.real = 0.0f;
    result.imag = 0.0f;
    
    for (int j = 0; j < order; j++) {
        float L = lagrange_basis(x, j, order);
        Complex term = complex_scale(samples[j], L);
        result = complex_add(result, term);
    }
    return result;
}

// ============================================================================
// ОСНОВНОЙ KERNEL
// ============================================================================

/**
 * Ядро дробной задержки
 * 
 * Работает на всем векторе одновременно:
 * - Каждый work-item обрабатывает одно комплексное число
 * - Использует локальную память для кэширования соседних значений
 * - Применяет интерполяцию Лагранжа для дробной части задержки
 * 
 * Параметры (устанавливаются через clSetKernelArg):
 * 0: input_vector  - входной комплексный вектор (все антенны x все точки)
 * 1: output_vector - выходной комплексный вектор
 * 2: delay_samples - целая часть задержки (в отсчётах)
 * 3: delay_frac    - дробная часть задержки [0, 1)
 * 4: num_beams     - количество лучей
 * 5: num_samples   - количество отсчётов в луче
 */
__kernel void fractional_delay_kernel(
    __global const Complex* input_vector,
    __global Complex* output_vector,
    int delay_samples,           // целая часть задержки
    float delay_frac,            // дробная часть задержки [0, 1)
    uint num_beams,
    uint num_samples
) {
    // Глобальный индекс: 0...(num_beams * num_samples - 1)
    uint gid = get_global_id(0);
    
    // Проверка границ
    if (gid >= num_beams * num_samples) {
        return;
    }
    
    // Определить луч и позицию внутри луча
    uint beam_idx = gid / num_samples;
    uint sample_idx = gid % num_samples;
    
    // Вычислить индекс задержанного отсчёта
    int source_idx = (int)sample_idx - delay_samples;
    
    Complex result;
    result.real = 0.0f;
    result.imag = 0.0f;
    
    // Случай 1: целая задержка без дробной части (быстрая ветка)
    if (delay_frac < 0.001f) {
        if (source_idx >= 0 && source_idx < (int)num_samples) {
            uint source_offset = beam_idx * num_samples + source_idx;
            result = input_vector[source_offset];
        }
    }
    // Случай 2: есть дробная часть (интерполяция)
    else if (source_idx > 0 && source_idx < (int)num_samples - ORDER + 1) {
        // Окно интерполяции: [source_idx - ORDER/2, source_idx + ORDER/2]
        Complex samples[ORDER];
        int start_idx = source_idx - ORDER / 2;
        
        for (int i = 0; i < ORDER; i++) {
            int idx = start_idx + i;
            if (idx >= 0 && idx < (int)num_samples) {
                uint offset = beam_idx * num_samples + idx;
                samples[i] = input_vector[offset];
            } else {
                samples[i].real = 0.0f;
                samples[i].imag = 0.0f;
            }
        }
        
        // Интерполяция Лагранжа с дробной частью
        float x = delay_frac + ORDER / 2 - 1;  // позиция в окне
        result = lagrange_interpolate(samples, x, ORDER);
    }
    // Случай 3: граница (без интерполяции)
    else if (source_idx >= 0 && source_idx < (int)num_samples) {
        uint source_offset = beam_idx * num_samples + source_idx;
        result = input_vector[source_offset];
    }
    
    // Записать результат
    output_vector[gid] = result;
}

)CL";
}

// ============================================================================
// КОНСТРУКТОР
// ============================================================================

FractionalDelayProcessor::FractionalDelayProcessor(
    const FractionalDelayConfig& config,
    const LFMParameters& lfm_params
)
    : config_(config),
      lfm_params_(lfm_params),
      engine_(nullptr),
      kernel_(nullptr),
      total_samples_processed_(0),
      total_gpu_time_(0.0),
      total_calls_(0)
{
    // Валидация конфигурации
    if (config.num_beams < 1 || config.num_beams > 512) {
        throw std::invalid_argument("num_beams must be 1..512");
    }
    if (config.num_samples < 16) {
        throw std::invalid_argument("num_samples must be >= 16");
    }
    if (config.local_work_size < 1 || config.local_work_size > 1024) {
        throw std::invalid_argument("local_work_size must be 1..1024");
    }
    
    // Валидация LFM параметров
    if (lfm_params.num_beams != config.num_beams ||
        lfm_params.count_points != config.num_samples) {
        throw std::invalid_argument(
            "LFM params must match config (num_beams, count_points)"
        );
    }
    
    // Инициализировать
    Initialize();
}

// ============================================================================
// ДЕСТРУКТОР
// ============================================================================

FractionalDelayProcessor::~FractionalDelayProcessor() {
    if (kernel_ != nullptr) {
        clReleaseKernel(kernel_);
    }
    buffer_input_.reset();
    buffer_output_.reset();
}

// ============================================================================
// MOVE СЕМАНТИКА
// ============================================================================

FractionalDelayProcessor::FractionalDelayProcessor(
    FractionalDelayProcessor&& other
) noexcept
    : config_(other.config_),
      lfm_params_(other.lfm_params_),
      engine_(other.engine_),
      kernel_(other.kernel_),
      buffer_input_(std::move(other.buffer_input_)),
      buffer_output_(std::move(other.buffer_output_)),
      total_samples_processed_(other.total_samples_processed_),
      total_gpu_time_(other.total_gpu_time_),
      total_calls_(other.total_calls_)
{
    other.kernel_ = nullptr;
}

FractionalDelayProcessor& FractionalDelayProcessor::operator=(
    FractionalDelayProcessor&& other
) noexcept {
    if (this != &other) {
        if (kernel_ != nullptr) {
            clReleaseKernel(kernel_);
        }
        
        config_ = other.config_;
        lfm_params_ = other.lfm_params_;
        engine_ = other.engine_;
        kernel_ = other.kernel_;
        buffer_input_ = std::move(other.buffer_input_);
        buffer_output_ = std::move(other.buffer_output_);
        total_samples_processed_ = other.total_samples_processed_;
        total_gpu_time_ = other.total_gpu_time_;
        total_calls_ = other.total_calls_;
        
        other.kernel_ = nullptr;
    }
    return *this;
}

// ============================================================================
// ИНИЦИАЛИЗАЦИЯ
// ============================================================================

void FractionalDelayProcessor::Initialize() {
    if (config_.verbose) {
        std::cout << "[FDP] Инициализация FractionalDelayProcessor\n";
    }
    
    // Получить OpenCLComputeEngine
    engine_ = &gpu::OpenCLComputeEngine::GetInstance();
    if (!engine_) {
        throw std::runtime_error(
            "OpenCLComputeEngine не инициализирован! "
            "Вызовите gpu::OpenCLComputeEngine::Initialize()"
        );
    }
    
    // Загрузить kernel'ы
    LoadKernels();
    
    // Создать GPU буферы (ОДИН вектор на вход/выход!)
    CreateBuffers();
    
    if (config_.verbose) {
        std::cout << "[FDP] Инициализация завершена ✅\n";
    }
}

// ============================================================================
// ЗАГРУЗКА KERNEL'ОВ
// ============================================================================

void FractionalDelayProcessor::LoadKernels() {
    if (config_.verbose) {
        std::cout << "[FDP] Загрузка kernel'ов...\n";
    }
    
    std::string source = GetKernelSource();
    cl_int err;
    
    // Компилировать программу
    const char* src_ptr = source.c_str();
    size_t src_len = source.length();
    
    auto program = engine_->CompileKernel(src_ptr, src_len);
    if (!program) {
        throw std::runtime_error("Ошибка компиляции kernel'а");
    }
    
    // Получить kernel
    kernel_ = clCreateKernel(program, "fractional_delay_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Ошибка создания kernel'а");
    }
    
    clReleaseProgram(program);
    
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
    
    // Размер вектора: num_beams * num_samples комплексных чисел
    size_t vector_size = (size_t)config_.num_beams * config_.num_samples;
    size_t bytes = vector_size * sizeof(Complex);
    
    if (config_.verbose) {
        std::cout << "[FDP]   - Размер: " << config_.num_beams << " x " 
                  << config_.num_samples << " = " << vector_size 
                  << " комплексных (" << (bytes / 1024.0 / 1024.0) 
                  << " MB)\n";
    }
    
    // Создать буферы через OpenCLComputeEngine
    try {
        buffer_input_ = engine_->CreateBuffer(
            gpu::MemoryType::GPUExclusive,
            bytes,
            nullptr
        );
        
        buffer_output_ = engine_->CreateBuffer(
            gpu::MemoryType::GPUExclusive,
            bytes,
            nullptr
        );
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("Ошибка создания GPU буферов: ") + e.what()
        );
    }
    
    if (config_.verbose) {
        std::cout << "[FDP] GPU буферы созданы ✅\n";
    }
}

// ============================================================================
// СИНХРОНИЗАЦИЯ GPU
// ============================================================================

void FractionalDelayProcessor::SyncGPU() {
    auto queue = engine_->GetCommandQueue();
    cl_int err = clFinish(queue);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("GPU sync failed");
    }
}

// ============================================================================
// ОСНОВНАЯ ОБРАБОТКА - ОДНА ЗАДЕРЖКА
// ============================================================================

ProcessingResult FractionalDelayProcessor::ProcessWithFractionalDelay(
    const DelayParameter& delay
) {
    ProcessingResult result;
    result.success = false;
    result.beams_processed = 0;
    
    try {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        // Валидация параметра
        if (delay.beam_index >= config_.num_beams) {
            throw std::invalid_argument("Invalid beam_index");
        }
        
        if (config_.verbose) {
            std::cout << "[FDP] Обработка задержки: луч=" << delay.beam_index
                      << ", delay=" << delay.delay_degrees << "°\n";
        }
        
        // Генерировать входные данные (используем GeneratorGPU)
        auto queue = engine_->GetCommandQueue();
        
        // Установить аргументы kernel'а
        cl_int delay_int = static_cast<cl_int>(delay.delay_degrees);
        float delay_frac = delay.delay_degrees - delay_int;
        
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), 
                              buffer_input_->GetGPUBuffer());
        err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), 
                              buffer_output_->GetGPUBuffer());
        err |= clSetKernelArg(kernel_, 2, sizeof(cl_int), &delay_int);
        err |= clSetKernelArg(kernel_, 3, sizeof(cl_float), &delay_frac);
        err |= clSetKernelArg(kernel_, 4, sizeof(cl_uint), &config_.num_beams);
        err |= clSetKernelArg(kernel_, 5, sizeof(cl_uint), &config_.num_samples);
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Ошибка установки аргументов kernel'а");
        }
        
        // Размеры выполнения
        size_t total_work = (size_t)config_.num_beams * config_.num_samples;
        size_t local_size = config_.local_work_size;
        size_t global_size = ((total_work + local_size - 1) / local_size) * local_size;
        
        if (config_.verbose) {
            std::cout << "[FDP] Выполнение kernel'а: global=" << global_size 
                      << ", local=" << local_size << "\n";
        }
        
        // Выполнить kernel
        auto start_gpu = std::chrono::high_resolution_clock::now();
        
        err = clEnqueueNDRangeKernel(
            queue,
            kernel_,
            1,                    // 1D
            nullptr,
            &global_size,
            &local_size,
            0,
            nullptr,
            nullptr
        );
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Ошибка выполнения kernel'а");
        }
        
        // Синхронизировать
        SyncGPU();
        
        auto end_gpu = std::chrono::high_resolution_clock::now();
        result.gpu_execution_time_ms = 
            std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
        
        if (config_.verbose) {
            std::cout << "[FDP] GPU kernel time: " << result.gpu_execution_time_ms 
                      << " ms\n";
        }
        
        // Чтение результатов с GPU на CPU
        auto start_readback = std::chrono::high_resolution_clock::now();
        
        size_t vector_size = (size_t)config_.num_beams * config_.num_samples;
        result.output_data.resize(vector_size);
        
        err = clEnqueueReadBuffer(
            queue,
            *(cl_mem*)buffer_output_->GetGPUBuffer(),
            CL_TRUE,  // Blocking
            0,
            vector_size * sizeof(Complex),
            result.output_data.data(),
            0,
            nullptr,
            nullptr
        );
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Ошибка чтения результатов с GPU");
        }
        
        auto end_readback = std::chrono::high_resolution_clock::now();
        result.gpu_readback_time_ms = 
            std::chrono::duration<double, std::milli>(
                end_readback - start_readback
            ).count();
        
        auto end_total = std::chrono::high_resolution_clock::now();
        result.total_time_ms = 
            std::chrono::duration<double, std::milli>(
                end_total - start_total
            ).count();
        
        result.beams_processed = config_.num_beams;
        result.success = true;
        
        // Обновить статистику
        total_samples_processed_ += vector_size;
        total_gpu_time_ += result.gpu_execution_time_ms;
        total_calls_++;
        
        if (config_.verbose) {
            std::cout << "[FDP] Обработка завершена ✅\n";
            std::cout << "[FDP]   GPU exec: " << result.gpu_execution_time_ms 
                      << " ms\n";
            std::cout << "[FDP]   GPU read: " << result.gpu_readback_time_ms 
                      << " ms\n";
            std::cout << "[FDP]   Total:    " << result.total_time_ms << " ms\n";
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Error: ") + e.what();
        if (config_.verbose) {
            std::cout << "[FDP] Ошибка: " << result.error_message << "\n";
        }
    }
    
    return result;
}

// ============================================================================
// BATCH ОБРАБОТКА
// ============================================================================

std::vector<ProcessingResult> FractionalDelayProcessor::ProcessBatch(
    const std::vector<DelayParameter>& delays
) {
    std::vector<ProcessingResult> results;
    results.reserve(delays.size());
    
    if (config_.verbose) {
        std::cout << "[FDP] Batch обработка " << delays.size() 
                  << " задержек\n";
    }
    
    for (const auto& delay : delays) {
        results.push_back(ProcessWithFractionalDelay(delay));
    }
    
    return results;
}

// ============================================================================
// ДИАГНОСТИКА
// ============================================================================

void FractionalDelayProcessor::PrintInfo() const {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "FRACTIONAL DELAY PROCESSOR INFO\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Configuration:\n";
    std::cout << "  - Num beams:     " << config_.num_beams << "\n";
    std::cout << "  - Num samples:   " << config_.num_samples << "\n";
    std::cout << "  - Local worksize: " << config_.local_work_size << "\n";
    std::cout << "  - Verbose:       " << (config_.verbose ? "Yes" : "No") << "\n";
    std::cout << "\nStatistics:\n";
    std::cout << "  - Total calls:   " << total_calls_ << "\n";
    std::cout << "  - Total samples: " << total_samples_processed_ << "\n";
    std::cout << "  - Total GPU time: " << total_gpu_time_ << " ms\n";
    std::cout << std::string(70, '=') << "\n\n";
}

std::string FractionalDelayProcessor::GetStatistics() const {
    std::ostringstream oss;
    oss << "Calls: " << total_calls_ 
        << " | Samples: " << total_samples_processed_
        << " | GPU time: " << std::fixed << std::setprecision(2) 
        << total_gpu_time_ << " ms";
    return oss.str();
}

} // namespace radar
