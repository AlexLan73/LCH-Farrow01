#include "fractional_delay_processor.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace radar {

// ════════════════════════════════════════════════════════════════════════════
// КОНСТРУКТОР / ДЕСТРУКТОР
// ════════════════════════════════════════════════════════════════════════════

FractionalDelayProcessor::FractionalDelayProcessor(
    const FractionalDelayConfig& config,
    const LFMParameters& lfm_params)
    : config_(config),
      lfm_params_(lfm_params),
      initialized_(false),
      engine_(nullptr),
      kernel_fractional_delay_(nullptr) {
    
    // ✅ Валидация конфигурации
    if (!config_.IsValid()) {
        throw std::invalid_argument(
            "[FractionalDelayProcessor] Invalid configuration: "
            "num_beams=" + std::to_string(config_.num_beams) +
            ", num_samples=" + std::to_string(config_.num_samples));
    }
    
    // ✅ Валидация параметров LFM
    if (!lfm_params_.IsValid()) {
        throw std::invalid_argument(
            "[FractionalDelayProcessor] Invalid LFMParameters: "
            "check f_start, f_stop, sample_rate, num_beams, count_points/duration");
    }
    
    // ✅ Проверить инициализацию OpenCL
    if (!gpu::OpenCLComputeEngine::IsInitialized()) {
        throw std::runtime_error(
            "[FractionalDelayProcessor] OpenCLComputeEngine not initialized.\n"
            "Call: gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU) first!");
    }
    
    // ✅ Получить engine
    engine_ = &gpu::OpenCLComputeEngine::GetInstance();
    
    // ✅ Инициализация
    try {
        Initialize();
        initialized_ = true;
        
        if (config_.verbose) {
            PrintInfo();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("[FractionalDelayProcessor] Initialization failed: ") + e.what());
    }
}

FractionalDelayProcessor::~FractionalDelayProcessor() {
    // ✅ ВАЖНО: Ресурсы управляются OpenCLComputeEngine
    // Не вызываем clRelease* вручную
    
    kernel_fractional_delay_ = nullptr;
    kernel_program_.reset();
    signal_generator_.reset();
    buffer_input_.reset();
    buffer_output_.reset();
    engine_ = nullptr;
    initialized_ = false;
    
    if (config_.verbose) {
        std::cout << "[FractionalDelayProcessor] ✅ Destroyed" << std::endl;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MOVE СЕМАНТИКА
// ════════════════════════════════════════════════════════════════════════════

FractionalDelayProcessor::FractionalDelayProcessor(
    FractionalDelayProcessor&& other) noexcept
    : config_(other.config_),
      lfm_params_(other.lfm_params_),
      initialized_(other.initialized_),
      engine_(other.engine_),
      signal_generator_(std::move(other.signal_generator_)),
      kernel_program_(std::move(other.kernel_program_)),
      kernel_fractional_delay_(other.kernel_fractional_delay_),
      buffer_input_(std::move(other.buffer_input_)),
      buffer_output_(std::move(other.buffer_output_)),
      stats_(other.stats_) {
    
    other.engine_ = nullptr;
    other.kernel_fractional_delay_ = nullptr;
    other.initialized_ = false;
}

FractionalDelayProcessor& FractionalDelayProcessor::operator=(
    FractionalDelayProcessor&& other) noexcept {
    
    if (this != &other) {
        // Очистить текущие ресурсы
        kernel_fractional_delay_ = nullptr;
        kernel_program_.reset();
        signal_generator_.reset();
        buffer_input_.reset();
        buffer_output_.reset();
        
        // Переместить от other
        config_ = other.config_;
        lfm_params_ = other.lfm_params_;
        initialized_ = other.initialized_;
        engine_ = other.engine_;
        signal_generator_ = std::move(other.signal_generator_);
        kernel_program_ = std::move(other.kernel_program_);
        kernel_fractional_delay_ = other.kernel_fractional_delay_;
        buffer_input_ = std::move(other.buffer_input_);
        buffer_output_ = std::move(other.buffer_output_);
        stats_ = other.stats_;
        
        // Обнулить в other
        other.engine_ = nullptr;
        other.kernel_fractional_delay_ = nullptr;
        other.initialized_ = false;
    }
    
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// ПРИВАТНЫЕ МЕТОДЫ: ИНИЦИАЛИЗАЦИЯ
// ════════════════════════════════════════════════════════════════════════════

void FractionalDelayProcessor::Initialize() {
    if (config_.verbose) {
        std::cout << "[FractionalDelayProcessor] Initializing..." << std::endl;
    }
    
    // ✅ Шаг 1: Создать генератор сигналов
    signal_generator_ = std::make_unique<GeneratorGPU>(lfm_params_);
    
    // ✅ Шаг 2: Загрузить kernel'ы
    LoadKernels();
    
    // ✅ Шаг 3: Создать буферы
    CreateBuffers();
    
    if (config_.verbose) {
        std::cout << "[FractionalDelayProcessor] ✅ Initialization complete" << std::endl;
        std::cout << " - Beams: " << config_.num_beams << std::endl;
        std::cout << " - Samples per beam: " << config_.num_samples << std::endl;
        std::cout << " - Total elements: " << (config_.num_beams * config_.num_samples) << std::endl;
        std::cout << " - GPU memory: " 
                  << (GetGPUBufferSizeBytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
    }
}

void FractionalDelayProcessor::LoadKernels() {
    if (config_.verbose) {
        std::cout << "[FractionalDelayProcessor] Loading kernels..." << std::endl;
    }
    
    // ✅ Получить исходный код kernel'а
    std::string kernel_source = GetKernelSource();
    
    // ✅ Загрузить программу через engine (с кэшем)
    kernel_program_ = engine_->LoadProgram(kernel_source);
    if (!kernel_program_) {
        throw std::runtime_error("[FractionalDelayProcessor] Failed to load kernel program");
    }
    
    // ✅ Получить kernel дробной задержки
    kernel_fractional_delay_ = engine_->GetKernel(
        kernel_program_, "kernel_fractional_delay_optimized");
    if (!kernel_fractional_delay_) {
        throw std::runtime_error(
            "[FractionalDelayProcessor] Failed to get kernel_fractional_delay_optimized");
    }
    
    if (config_.verbose) {
        std::cout << "[FractionalDelayProcessor] ✅ Kernels loaded" << std::endl;
    }
}

void FractionalDelayProcessor::CreateBuffers() {
    if (config_.verbose) {
        std::cout << "[FractionalDelayProcessor] Creating GPU buffers..." << std::endl;
    }
    
    size_t total_elements = config_.num_beams * config_.num_samples;
    
    try {
        // ✅ Входной буфер (базовый сигнал)
        buffer_input_ = engine_->CreateBuffer(
            total_elements,
            gpu::MemoryType::GPU_READ_WRITE);
        
        // ✅ Выходной буфер (результаты задержки)
        buffer_output_ = engine_->CreateBuffer(
            total_elements,
            config_.result_memory_type);
        
        if (config_.verbose) {
            std::cout << "[FractionalDelayProcessor] ✅ GPU buffers created" << std::endl;
            std::cout << " - Input buffer: " 
                      << (buffer_input_->GetSizeBytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << " - Output buffer: " 
                      << (buffer_output_->GetSizeBytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("[FractionalDelayProcessor] Buffer creation failed: ") + e.what());
    }
}

void FractionalDelayProcessor::SyncGPU() {
    if (!engine_) {
        throw std::runtime_error("[FractionalDelayProcessor] Engine not initialized");
    }
    engine_->Finish();
}

// ════════════════════════════════════════════════════════════════════════════
// ОСНОВНЫЕ МЕТОДЫ ОБРАБОТКИ
// ════════════════════════════════════════════════════════════════════════════

ProcessingResult FractionalDelayProcessor::ProcessWithFractionalDelay(
    const DelayParameter& delay_param) {
    
    ProcessingResult result;
    
    try {
        // ✅ Валидация входных параметров
        if (delay_param.beam_index >= config_.num_beams) {
            result.error_message = "Invalid beam_index: " + 
                                  std::to_string(delay_param.beam_index) +
                                  " (max: " + std::to_string(config_.num_beams - 1) + ")";
            result.success = false;
            return result;
        }
        
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 1: Синхронизировать GPU
        // ════════════════════════════════════════════════════════════════
        SyncGPU();
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 2: Генерировать базовый сигнал
        // ════════════════════════════════════════════════════════════════
        if (config_.verbose) {
            std::cout << "[ProcessWithFractionalDelay] Generating base signal..." << std::endl;
        }
        
        signal_generator_->signal_base();
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 3: Получить буфер входных данных из генератора
        // ════════════════════════════════════════════════════════════════
        auto gen_data = signal_generator_->GetSignalAsVectorAll();
        
        if (gen_data.empty()) {
            result.error_message = "Failed to get signal data from generator";
            result.success = false;
            return result;
        }
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 4: Загрузить данные в входной буфер
        // ════════════════════════════════════════════════════════════════
        auto gpu_start = std::chrono::high_resolution_clock::now();
        
        buffer_input_->Write(gen_data);
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 5: Установить аргументы kernel'а
        // ════════════════════════════════════════════════════════════════
        // Arguments for kernel:
        // 0: input buffer (global complex array)
        // 1: output buffer (global complex array)
        // 2: delay_radians (float)
        // 3: num_beams (uint)
        // 4: num_samples (uint)
        
        buffer_input_->SetAsKernelArg(kernel_fractional_delay_, 0);
        buffer_output_->SetAsKernelArg(kernel_fractional_delay_, 1);
        
        // Рассчитать задержку в радианах
        float delay_rad = (delay_param.delay_degrees * M_PI) / 180.0f;
        cl_int err = clSetKernelArg(kernel_fractional_delay_, 2, sizeof(float), &delay_rad);
        if (err != CL_SUCCESS) {
            result.error_message = "Failed to set delay_radians arg: " + std::to_string(err);
            result.success = false;
            return result;
        }
        
        cl_uint num_beams = config_.num_beams;
        err = clSetKernelArg(kernel_fractional_delay_, 3, sizeof(cl_uint), &num_beams);
        if (err != CL_SUCCESS) {
            result.error_message = "Failed to set num_beams arg: " + std::to_string(err);
            result.success = false;
            return result;
        }
        
        cl_uint num_samples = config_.num_samples;
        err = clSetKernelArg(kernel_fractional_delay_, 4, sizeof(cl_uint), &num_samples);
        if (err != CL_SUCCESS) {
            result.error_message = "Failed to set num_samples arg: " + std::to_string(err);
            result.success = false;
            return result;
        }
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 6: Выполнить kernel
        // ════════════════════════════════════════════════════════════════
        std::array<size_t, 3> global_work_size = {
            config_.num_beams,
            config_.num_samples,
            1
        };
        
        std::array<size_t, 3> local_work_size = {
            1,
            config_.local_work_size,
            1
        };
        
        if (config_.verbose) {
            std::cout << "[ProcessWithFractionalDelay] Executing kernel..." << std::endl;
            std::cout << " - Global work size: " << global_work_size[0] << " x " 
                      << global_work_size[1] << std::endl;
            std::cout << " - Local work size: " << local_work_size[0] << " x " 
                      << local_work_size[1] << std::endl;
            std::cout << " - Delay: " << delay_param.delay_degrees << "° = " 
                      << delay_rad << " rad" << std::endl;
        }
        
        engine_->ExecuteKernel(
            kernel_fractional_delay_,
            {buffer_input_.get(), buffer_output_.get()},
            global_work_size,
            local_work_size);
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 7: Синхронизировать и профилировать время GPU
        // ════════════════════════════════════════════════════════════════
        SyncGPU();
        
        auto gpu_end = std::chrono::high_resolution_clock::now();
        result.gpu_execution_time_ms = 
            std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 8: Прочитать результаты с GPU
        // ════════════════════════════════════════════════════════════════
        if (config_.verbose) {
            std::cout << "[ProcessWithFractionalDelay] Reading results from GPU..." << std::endl;
        }
        
        auto readback_start = std::chrono::high_resolution_clock::now();
        
        result.output_data = buffer_output_->Read();
        
        auto readback_end = std::chrono::high_resolution_clock::now();
        result.gpu_readback_time_ms = 
            std::chrono::duration<double, std::milli>(readback_end - readback_start).count();
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 9: Обновить статистику и возвращаемые значения
        // ════════════════════════════════════════════════════════════════
        auto cpu_end = std::chrono::high_resolution_clock::now();
        result.total_time_ms = 
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        
        result.beams_processed = config_.num_beams;
        result.success = true;
        
        // Обновить общую статистику
        stats_.total_processed++;
        stats_.total_gpu_time_ms += result.gpu_execution_time_ms;
        stats_.total_readback_time_ms += result.gpu_readback_time_ms;
        
        if (config_.verbose) {
            std::cout << "[ProcessWithFractionalDelay] ✅ Processing complete!" << std::endl;
            std::cout << " - GPU execution: " << std::fixed << std::setprecision(3) 
                      << result.gpu_execution_time_ms << " ms" << std::endl;
            std::cout << " - GPU readback: " << result.gpu_readback_time_ms << " ms" << std::endl;
            std::cout << " - Total time: " << result.total_time_ms << " ms" << std::endl;
            std::cout << " - Output size: " << result.output_data.size() << " elements (" 
                      << (result.output_data.size() * sizeof(ComplexFloat) / (1024.0 * 1024.0))
                      << " MB)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
        result.success = false;
    }
    
    return result;
}

std::vector<ProcessingResult> FractionalDelayProcessor::ProcessBatch(
    const std::vector<DelayParameter>& delays) {
    
    std::vector<ProcessingResult> results;
    
    if (config_.verbose) {
        std::cout << "[ProcessBatch] Processing " << delays.size() << " delay(s)..." << std::endl;
    }
    
    for (size_t i = 0; i < delays.size(); ++i) {
        if (config_.verbose) {
            std::cout << "[ProcessBatch] Item " << (i + 1) << "/" << delays.size() << std::endl;
        }
        
        auto result = ProcessWithFractionalDelay(delays[i]);
        results.push_back(result);
        
        if (!result.success) {
            std::cerr << "❌ Failed to process delay " << i << ": " 
                      << result.error_message << std::endl;
        }
    }
    
    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// МЕТОДЫ ДИАГНОСТИКИ
// ════════════════════════════════════════════════════════════════════════════

void FractionalDelayProcessor::PrintInfo() const {
    std::cout << "\n" << std::string(70, '═') << "\n";
    std::cout << "FractionalDelayProcessor Information\n";
    std::cout << std::string(70, '═') << "\n";
    
    std::cout << "Configuration:\n";
    std::cout << " - Beams: " << config_.num_beams << "\n";
    std::cout << " - Samples per beam: " << config_.num_samples << "\n";
    std::cout << " - Total elements: " << (config_.num_beams * config_.num_samples) << "\n";
    std::cout << " - Local work size: " << config_.local_work_size << "\n";
    
    std::cout << "\nLFM Parameters:\n";
    std::cout << " - F start: " << lfm_params_.f_start / 1e6 << " MHz\n";
    std::cout << " - F stop: " << lfm_params_.f_stop / 1e6 << " MHz\n";
    std::cout << " - Sample rate: " << lfm_params_.sample_rate / 1e6 << " MHz\n";
    std::cout << " - Duration: " << lfm_params_.duration * 1e6 << " µs\n";
    std::cout << " - Angle start: " << lfm_params_.angle_start_deg << "°\n";
    std::cout << " - Angle stop: " << lfm_params_.angle_stop_deg << "°\n";
    
    std::cout << "\nMemory Usage:\n";
    std::cout << " - GPU buffers: " << (GetGPUBufferSizeBytes() / (1024.0 * 1024.0)) << " MB\n";
    std::cout << " - Status: " << (initialized_ ? "✅ Initialized" : "❌ Not initialized") << "\n";
    
    std::cout << std::string(70, '═') << "\n\n";
}

std::string FractionalDelayProcessor::GetStatistics() const {
    std::ostringstream oss;
    
    oss << "\n" << std::string(60, '─') << "\n";
    oss << "FractionalDelayProcessor Statistics\n";
    oss << std::string(60, '─') << "\n";
    
    oss << std::left << std::setw(30) << "Total processed:" << stats_.total_processed << "\n";
    oss << std::left << std::setw(30) << "Total GPU time:" 
        << std::fixed << std::setprecision(2) << stats_.total_gpu_time_ms << " ms\n";
    oss << std::left << std::setw(30) << "Total readback time:" 
        << stats_.total_readback_time_ms << " ms\n";
    
    if (stats_.total_processed > 0) {
        oss << std::left << std::setw(30) << "Avg GPU time per call:" 
            << (stats_.total_gpu_time_ms / stats_.total_processed) << " ms\n";
    }
    
    oss << std::string(60, '─') << "\n\n";
    
    return oss.str();
}

size_t FractionalDelayProcessor::GetGPUBufferSizeBytes() const {
    size_t total = 0;
    if (buffer_input_) total += buffer_input_->GetSizeBytes();
    if (buffer_output_) total += buffer_output_->GetSizeBytes();
    return total;
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL SOURCE CODE
// ════════════════════════════════════════════════════════════════════════════

std::string FractionalDelayProcessor::GetKernelSource() const {
    return R"(
// ═════════════════════════════════════════════════════════════════════════
// kernel_fractional_delay_optimized
// 
// Применяет дробную задержку к комплексным сигналам
// 
// Аргументы:
//   input[]      - Входные комплексные отсчёты (float2 формат)
//   output[]     - Выходные комплексные отсчёты
//   delay_rad    - Задержка в радианах
//   num_beams    - Количество лучей
//   num_samples  - Количество отсчётов на луч
// ═════════════════════════════════════════════════════════════════════════

__kernel void kernel_fractional_delay_optimized(
    __global float2 *input,
    __global float2 *output,
    float delay_rad,
    uint num_beams,
    uint num_samples) {
    
    // Получить индексы потока
    uint beam_idx = get_global_id(0);    // Индекс луча
    uint sample_idx = get_global_id(1);  // Индекс отсчёта
    
    // ✅ Границы
    if (beam_idx >= num_beams || sample_idx >= num_samples) {
        return;
    }
    
    // ✅ Линейный индекс в буфере
    uint idx = beam_idx * num_samples + sample_idx;
    
    // ✅ Получить входное значение
    float2 input_val = input[idx];
    
    // ✅ Рассчитать фазовый сдвиг: exp(j * delay_rad * sample_idx)
    float phase = delay_rad * (float)sample_idx;
    
    // ✅ Вычислить cos(phase) и sin(phase)
    float cos_phase = cos(phase);
    float sin_phase = sin(phase);
    
    // ✅ Применить фазовый сдвиг: complex_mul(input, exp(j*phase))
    // complex_mul(a+jb, c+jd) = (ac-bd) + j(ad+bc)
    float2 output_val;
    output_val.x = input_val.x * cos_phase - input_val.y * sin_phase;
    output_val.y = input_val.x * sin_phase + input_val.y * cos_phase;
    
    // ✅ Записать результат
    output[idx] = output_val;
}
)";
}

} // namespace radar
