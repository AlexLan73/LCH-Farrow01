#include "fft/antenna_fft_proc_max.h"
#include "GPU/opencl_compute_engine.hpp"
#include "GPU/opencl_core.hpp"
#include "GPU/command_queue_pool.hpp"
#include "GPU/kernel_program.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <cstring>
#include <climits>

namespace antenna_fft {

// Структура для хранения максимумов (совпадает с kernel структурой)
struct MaxValue {
    cl_uint index;
    float magnitude;
    float phase;
    cl_uint pad;
};

constexpr size_t kMaxReductionPoints = 1024;

// ════════════════════════════════════════════════════════════════════════════
// Статические члены для кэша планов
// ════════════════════════════════════════════════════════════════════════════

std::unordered_map<AntennaFFTProcMax::PlanCacheKey, clfftPlanHandle, 
                   AntennaFFTProcMax::PlanCacheKeyHash> AntennaFFTProcMax::plan_cache_;
std::mutex AntennaFFTProcMax::plan_cache_mutex_;

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTProcMax::AntennaFFTProcMax(const AntennaFFTParams& params)
    : params_(params),
       nFFT_(0),
       engine_(nullptr),
       context_(nullptr),
       queue_(nullptr),
       device_(nullptr),
       plan_handle_(0),
       plan_created_(false),
       pre_callback_userdata_(nullptr),
       post_callback_userdata_(nullptr),
       reduction_kernel_(nullptr),
       padding_kernel_(nullptr),
       post_kernel_(nullptr),
       batch_config_(),
       batch_total_cpu_time_ms_(0.0),
       last_used_batch_mode_(false),
       batch_buffers_size_(0),
       batch_plan_handle_(0),
       batch_plan_beams_(0) {
    
    // Валидация параметров
    if (!params_.IsValid()) {
        throw std::invalid_argument("AntennaFFTParams: invalid parameters");
    }
    
    // Проверка инициализации OpenCLComputeEngine
    if (!gpu::OpenCLComputeEngine::IsInitialized()) {
        throw std::runtime_error("OpenCLComputeEngine not initialized. Call Initialize() first.");
    }
    
    engine_ = &gpu::OpenCLComputeEngine::GetInstance();
    
    // Получить контекст и устройство
    auto& core = gpu::OpenCLCore::GetInstance();
    context_ = core.GetContext();
    device_ = core.GetDevice();
    
    // Получить command queue
    queue_ = gpu::CommandQueuePool::GetNextQueue();
    
    // Вычислить nFFT
    nFFT_ = CalculateNFFT(params_.count_points);
    
    // Инициализировать clFFT
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftStatus status = clfftSetup(&fftSetup);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftSetup failed with status: " + std::to_string(status));
    }
    
    // Инициализировать профилирование
    last_profiling_ = {};
}

AntennaFFTProcMax::~AntennaFFTProcMax() {
    ReleaseFFTPlan();

    // Освободить batch FFT план
    if (batch_plan_handle_) {
        clfftDestroyPlan(&batch_plan_handle_);
    }

    if (pre_callback_userdata_) {
        clReleaseMemObject(pre_callback_userdata_);
    }
    if (post_callback_userdata_) {
        clReleaseMemObject(post_callback_userdata_);
    }
    if (reduction_kernel_) {
        clReleaseKernel(reduction_kernel_);
    }
    if (padding_kernel_) {
        clReleaseKernel(padding_kernel_);
    }
    if (post_kernel_) {
        clReleaseKernel(post_kernel_);
    }
}

AntennaFFTProcMax::AntennaFFTProcMax(AntennaFFTProcMax&& other) noexcept
    : params_(other.params_),
       nFFT_(other.nFFT_),
       engine_(other.engine_),
       context_(other.context_),
       queue_(other.queue_),
       device_(other.device_),
       plan_handle_(other.plan_handle_),
       plan_created_(other.plan_created_),
       buffer_input_(std::move(other.buffer_input_)),
       buffer_fft_input_(std::move(other.buffer_fft_input_)),
       buffer_fft_output_(std::move(other.buffer_fft_output_)),
       buffer_magnitude_(std::move(other.buffer_magnitude_)),
       buffer_maxima_(std::move(other.buffer_maxima_)),
       buffer_selected_complex_(std::move(other.buffer_selected_complex_)),
       buffer_selected_magnitude_(std::move(other.buffer_selected_magnitude_)),
       pre_callback_userdata_(other.pre_callback_userdata_),
       post_callback_userdata_(other.post_callback_userdata_),
       reduction_program_(std::move(other.reduction_program_)),
       reduction_kernel_(other.reduction_kernel_),
       padding_kernel_(other.padding_kernel_),
       post_kernel_(other.post_kernel_),
       last_profiling_(other.last_profiling_),
       batch_config_(other.batch_config_),
       batch_profiling_(std::move(other.batch_profiling_)),
       batch_total_cpu_time_ms_(other.batch_total_cpu_time_ms_),
       last_used_batch_mode_(other.last_used_batch_mode_) {

    other.plan_handle_ = 0;
    other.plan_created_ = false;
    other.pre_callback_userdata_ = nullptr;
    other.post_callback_userdata_ = nullptr;
    other.reduction_kernel_ = nullptr;
    other.padding_kernel_ = nullptr;
    other.post_kernel_ = nullptr;
}

AntennaFFTProcMax& AntennaFFTProcMax::operator=(AntennaFFTProcMax&& other) noexcept {
    if (this != &other) {
        ReleaseFFTPlan();

        if (pre_callback_userdata_) clReleaseMemObject(pre_callback_userdata_);
        if (post_callback_userdata_) clReleaseMemObject(post_callback_userdata_);
        if (reduction_kernel_) clReleaseKernel(reduction_kernel_);
        if (padding_kernel_) clReleaseKernel(padding_kernel_);
        if (post_kernel_) clReleaseKernel(post_kernel_);

        params_ = other.params_;
        nFFT_ = other.nFFT_;
        engine_ = other.engine_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        plan_handle_ = other.plan_handle_;
        plan_created_ = other.plan_created_;
        buffer_input_ = std::move(other.buffer_input_);
        buffer_fft_input_ = std::move(other.buffer_fft_input_);
        buffer_fft_output_ = std::move(other.buffer_fft_output_);
        buffer_magnitude_ = std::move(other.buffer_magnitude_);
        buffer_maxima_ = std::move(other.buffer_maxima_);
        buffer_selected_complex_ = std::move(other.buffer_selected_complex_);
        buffer_selected_magnitude_ = std::move(other.buffer_selected_magnitude_);
        pre_callback_userdata_ = other.pre_callback_userdata_;
        post_callback_userdata_ = other.post_callback_userdata_;
        reduction_program_ = std::move(other.reduction_program_);
        reduction_kernel_ = other.reduction_kernel_;
        padding_kernel_ = other.padding_kernel_;
        post_kernel_ = other.post_kernel_;
        last_profiling_ = other.last_profiling_;
        batch_config_ = other.batch_config_;
        batch_profiling_ = std::move(other.batch_profiling_);
        batch_total_cpu_time_ms_ = other.batch_total_cpu_time_ms_;
        last_used_batch_mode_ = other.last_used_batch_mode_;

        other.plan_handle_ = 0;
        other.plan_created_ = false;
        other.pre_callback_userdata_ = nullptr;
        other.post_callback_userdata_ = nullptr;
        other.reduction_kernel_ = nullptr;
        other.padding_kernel_ = nullptr;
        other.post_kernel_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Вычисление nFFT
// ════════════════════════════════════════════════════════════════════════════

size_t AntennaFFTProcMax::CalculateNFFT(size_t count_points) const {
    // Проверяем кратность 2^n
    if (!IsPowerOf2(count_points)) {
        // Дополняем до ближайшего большего числа кратного 2^n
        count_points = NextPowerOf2(count_points);
    }
    // Умножаем на 2
    return count_points * 2;
}

bool AntennaFFTProcMax::IsPowerOf2(size_t n) const {
    return n > 0 && (n & (n - 1)) == 0;
}

size_t AntennaFFTProcMax::NextPowerOf2(size_t n) const {
    if (n == 0) return 1;
    if (IsPowerOf2(n)) return n;
    
    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// ════════════════════════════════════════════════════════════════════════════
// Batch Processing: Оценка памяти и выбор стратегии
// ════════════════════════════════════════════════════════════════════════════

size_t AntennaFFTProcMax::EstimateRequiredMemory() const {
    // Входные данные: beam_count * count_points * sizeof(complex<float>)
    size_t input_size = params_.beam_count * params_.count_points * sizeof(std::complex<float>);
    
    // FFT буферы: beam_count * nFFT * sizeof(complex<float>) * 2 (input + output)
    size_t fft_buffers = params_.beam_count * nFFT_ * sizeof(std::complex<float>) * 2;
    
    // Pre-callback userdata: 32 bytes (params) + input_size
    size_t pre_userdata = 32 + input_size;
    
    // Post-processing буферы: beam_count * out_count_points_fft * (8 + 4) bytes
    size_t post_buffers = params_.beam_count * params_.out_count_points_fft * 
                         (sizeof(std::complex<float>) + sizeof(float));
    
    // Временные буферы clFFT (оценка ~nFFT * 8 bytes)
    size_t clfft_temp = nFFT_ * sizeof(std::complex<float>);
    
    return input_size + fft_buffers + pre_userdata + post_buffers + clfft_temp;
}

bool AntennaFFTProcMax::CheckAvailableMemory(size_t required_memory, double threshold) const {
    // Получить размер глобальной памяти GPU
    size_t global_memory = gpu::OpenCLCore::GetInstance().GetGlobalMemorySize();
    
    // Рассчитать доступную память с учётом порога
    size_t available_memory = static_cast<size_t>(global_memory * threshold);
    
    std::cout << "  ┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  MEMORY CHECK                                               │\n";
    std::cout << "  └─────────────────────────────────────────────────────────────┘\n";
    printf("  │  GPU Global Memory      │  %10zu MB  │\n", global_memory / (1024 * 1024));
    printf("  │  Threshold (%.0f%%)        │  %10zu MB  │\n", threshold * 100, available_memory / (1024 * 1024));
    printf("  │  Required Memory         │  %10zu MB  │\n", required_memory / (1024 * 1024));
    printf("  │  Status                  │  %s  │\n", 
           required_memory <= available_memory ? "    OK ✅    " : "  BATCH ⚠️  ");
    std::cout << "\n";
    
    return required_memory <= available_memory;
}

size_t AntennaFFTProcMax::CalculateBatchSize(size_t total_beams, double batch_percent) const {
    size_t batch_size = static_cast<size_t>(total_beams * batch_percent);
    
    // Минимум 1 луч
    if (batch_size < 1) batch_size = 1;
    
    // Не больше общего количества лучей
    if (batch_size > total_beams) batch_size = total_beams;
    
    return batch_size;
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessNew: Автоматический выбор стратегии обработки
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTResult AntennaFFTProcMax::ProcessNew(cl_mem input_signal) {
    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    std::cout << "  ProcessNew: Автоматический выбор стратегии\n";
    std::cout << "════════════════════════════════════════════════════════════════\n\n";
    
    // 1. Оценить требуемую память
    size_t required_memory = EstimateRequiredMemory();
    
    // 2. Проверить доступную память
    bool memory_ok = CheckAvailableMemory(required_memory, batch_config_.memory_usage_limit);
    
    // 3. Выбрать стратегию
    if (memory_ok) {
        std::cout << "  → Стратегия: SINGLE BATCH (полная обработка)\n";
        std::cout << "  → Вызываем Process()\n\n";
        last_used_batch_mode_ = false;
        return Process(input_signal);
    } else {
        std::cout << "  → Стратегия: MULTI-BATCH (batch processing)\n";
        std::cout << "  → Вызываем ProcessWithBatching()\n\n";
        last_used_batch_mode_ = true;
        return ProcessWithBatching(input_signal);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessWithBatching: Обработка с разбиением на батчи
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTResult AntennaFFTProcMax::ProcessWithBatching(cl_mem input_signal) {
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    // Рассчитать размер батча
    size_t batch_size = CalculateBatchSize(params_.beam_count, batch_config_.batch_size_ratio);
    
    // Рассчитать количество батчей
    size_t num_batches = (params_.beam_count + batch_size - 1) / batch_size;
    
    // Проверить: если в последнем батче остаётся 1-2 луча, добавить их в предыдущий
    size_t last_batch_beams = params_.beam_count - (num_batches - 1) * batch_size;
    if (num_batches > 1 && last_batch_beams <= 2) {
        num_batches--;  // Уменьшить количество батчей
        std::cout << "  ⚡ Оптимизация: " << last_batch_beams << " луч(а) добавлены в последний батч\n\n";
    }
    
    std::cout << "  ┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  BATCH PROCESSING                                           │\n";
    std::cout << "  └─────────────────────────────────────────────────────────────┘\n";
    printf("  │  Total beams             │  %10zu  │\n", params_.beam_count);
    printf("  │  Batch size (base)       │  %10zu  │\n", batch_size);
    printf("  │  Number of batches       │  %10zu  │\n", num_batches);
    printf("  │  Queue pool size         │  %10zu  │\n", gpu::CommandQueuePool::GetPoolSize());
    std::cout << "\n";
    
    // Очистить профилирование
    batch_profiling_.clear();
    batch_profiling_.reserve(num_batches);
    
    // Собрать все результаты
    std::vector<FFTResult> all_results;
    all_results.reserve(params_.beam_count);
    
    // Вектор для событий завершения батчей
    std::vector<cl_event> completion_events;
    completion_events.reserve(num_batches);
    
    // Структура для хранения информации о батчах (для профилирования)
    struct BatchInfo {
        size_t start_beam;
        size_t num_beams;
    };
    std::vector<BatchInfo> batch_infos;
    batch_infos.reserve(num_batches);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // СОЗДАТЬ БУФЕРЫ ОДИН РАЗ для максимального размера батча
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Найти максимальный размер батча (для последнего может быть больше)
    size_t max_batch_beams = (num_batches == 1) ? params_.beam_count : 
                            std::max(batch_size, params_.beam_count - (num_batches - 1) * batch_size);
    
    // Создать буферы если их нет или размер изменился
    if (!batch_fft_input_ || batch_buffers_size_ < max_batch_beams) {
        auto t_buf_start = std::chrono::high_resolution_clock::now();
        
        size_t fft_buf_size = max_batch_beams * nFFT_;
        // БЕЗ batch_input_buffer_ - читаем напрямую из input_signal!
        size_t output_buf_size = max_batch_beams * params_.out_count_points_fft;
        size_t mag_buf_elements = (output_buf_size * sizeof(float) + sizeof(std::complex<float>) - 1) / sizeof(std::complex<float>);
        
        batch_fft_input_ = engine_->CreateBuffer(fft_buf_size, gpu::MemoryType::GPU_READ_WRITE);
        batch_fft_output_ = engine_->CreateBuffer(fft_buf_size, gpu::MemoryType::GPU_READ_WRITE);
        // batch_input_buffer_ НЕ НУЖЕН - работаем напрямую с input_signal!
        batch_sel_complex_ = engine_->CreateBuffer(output_buf_size, gpu::MemoryType::GPU_READ_WRITE);
        batch_sel_magnitude_ = engine_->CreateBuffer(mag_buf_elements, gpu::MemoryType::GPU_READ_WRITE);
        batch_buffers_size_ = max_batch_beams;
        
        auto t_buf_end = std::chrono::high_resolution_clock::now();
        double buf_ms = std::chrono::duration<double, std::milli>(t_buf_end - t_buf_start).count();
        printf("  ⏱️  Created batch buffers (max %zu beams): %.2f ms\n\n", max_batch_beams, buf_ms);
    } else {
        std::cout << "  ♻️  Reusing cached batch buffers\n\n";
    }
    
    // Создать FFT план для батчей если нужно
    if (batch_plan_handle_ == 0 || batch_plan_beams_ != max_batch_beams) {
        if (batch_plan_handle_) {
            clfftDestroyPlan(&batch_plan_handle_);
        }
        
        auto t_plan_start = std::chrono::high_resolution_clock::now();
        
        size_t clLengths[1] = {nFFT_};
        clfftStatus status = clfftCreateDefaultPlan(&batch_plan_handle_, context_, CLFFT_1D, clLengths);
        if (status != CLFFT_SUCCESS) {
            throw std::runtime_error("ProcessWithBatching: clfftCreateDefaultPlan failed");
        }
        
        clfftSetPlanPrecision(batch_plan_handle_, CLFFT_SINGLE);
        clfftSetLayout(batch_plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        clfftSetResultLocation(batch_plan_handle_, CLFFT_OUTOFPLACE);
        clfftSetPlanBatchSize(batch_plan_handle_, max_batch_beams);
        
        size_t strides[1] = {1};
        clfftSetPlanInStride(batch_plan_handle_, CLFFT_1D, strides);
        clfftSetPlanOutStride(batch_plan_handle_, CLFFT_1D, strides);
        clfftSetPlanDistance(batch_plan_handle_, nFFT_, nFFT_);
        
        status = clfftBakePlan(batch_plan_handle_, 1, &queue_, nullptr, nullptr);
        if (status != CLFFT_SUCCESS) {
            clfftDestroyPlan(&batch_plan_handle_);
            batch_plan_handle_ = 0;
            throw std::runtime_error("ProcessWithBatching: clfftBakePlan failed");
        }
        
        batch_plan_beams_ = max_batch_beams;
        
        auto t_plan_end = std::chrono::high_resolution_clock::now();
        double plan_ms = std::chrono::duration<double, std::milli>(t_plan_end - t_plan_start).count();
        printf("  ⏱️  Created FFT plan (nFFT=%zu, batch=%zu): %.2f ms\n\n", nFFT_, max_batch_beams, plan_ms);
    } else {
        std::cout << "  ♻️  Reusing cached FFT plan\n\n";
    }
    
    // Убедиться что kernels созданы
    if (!post_kernel_) CreatePostKernel();
    if (!padding_kernel_) CreatePaddingKernel();
    
    // Обработать каждый батч
    size_t processed_beams = 0;
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        size_t start_beam = processed_beams;
        size_t beams_in_batch;
        
        if (batch_idx == num_batches - 1) {
            // Последний батч берёт все оставшиеся лучи
            beams_in_batch = params_.beam_count - processed_beams;
        } else {
            beams_in_batch = batch_size;
        }
        
        // Получить очередь из пула
        cl_command_queue batch_queue = gpu::CommandQueuePool::GetNextQueue();
        
        std::cout << "  [Batch " << batch_idx << "] Processing beams " 
                  << start_beam << "-" << (start_beam + beams_in_batch - 1)
                  << " (" << beams_in_batch << " beams, queue " 
                  << gpu::CommandQueuePool::GetCurrentQueueIndex() << ")\n";
        
        // Сохранить информацию о батче
        batch_infos.push_back({start_beam, beams_in_batch});
        
        // Событие завершения
        cl_event completion_event = nullptr;
        
        // Обработать батч
        auto batch_results = ProcessBatch(input_signal, start_beam, beams_in_batch, 
                                          batch_queue, &completion_event);
        
        // Сохранить результаты
        for (auto& result : batch_results) {
            all_results.push_back(std::move(result));
        }
        
        // Сохранить событие
        if (completion_event) {
            completion_events.push_back(completion_event);
        }
        
        processed_beams += beams_in_batch;
    }
    
    // Ждать завершения всех батчей
    if (!completion_events.empty()) {
        std::cout << "\n  Waiting for all batches to complete...\n";
        clWaitForEvents(static_cast<cl_uint>(completion_events.size()), completion_events.data());
        
        // Профилировать каждый батч
        for (size_t i = 0; i < completion_events.size(); ++i) {
            BatchProfilingData prof;
            prof.batch_index = i;
            prof.start_beam = batch_infos[i].start_beam;
            prof.num_beams = batch_infos[i].num_beams;
            prof.gpu_time_ms = ProfileEvent(completion_events[i], "Batch " + std::to_string(i));
            batch_profiling_.push_back(prof);
        }
        
        // Освободить события
        for (auto& event : completion_events) {
            clReleaseEvent(event);
        }
    }
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    batch_total_cpu_time_ms_ = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // Собрать результат
    AntennaFFTResult result(params_.beam_count, nFFT_, params_.task_id, params_.module_name);
    result.results = std::move(all_results);
    
    // Вывести профилирование
    std::cout << "\n  ┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  BATCH PROFILING                                            │\n";
    std::cout << "  └─────────────────────────────────────────────────────────────┘\n";
    
    double total_gpu_time = 0.0;
    for (const auto& prof : batch_profiling_) {
        printf("  │  Batch %zu (%3zu beams: %3zu-%3zu) │  %10.4f ms  │\n", 
               prof.batch_index, prof.num_beams, prof.start_beam, 
               prof.start_beam + prof.num_beams - 1, prof.gpu_time_ms);
        total_gpu_time += prof.gpu_time_ms;
    }
    
    std::cout << "  ├─────────────────────────────────────────────────────────────┤\n";
    printf("  │  Total GPU time          │  %10.4f ms  │\n", total_gpu_time);
    printf("  │  Total CPU time          │  %10.4f ms  │\n", batch_total_cpu_time_ms_);
    
    // Дополнительная статистика
    double avg_batch_time = batch_profiling_.empty() ? 0.0 : total_gpu_time / batch_profiling_.size();
    double beams_per_sec = batch_total_cpu_time_ms_ > 0 ? 
                           (params_.beam_count * 1000.0 / batch_total_cpu_time_ms_) : 0.0;
    
    std::cout << "  ├─────────────────────────────────────────────────────────────┤\n";
    printf("  │  Avg GPU time per batch  │  %10.4f ms  │\n", avg_batch_time);
    printf("  │  Total time              │  %10.2f sec │\n", batch_total_cpu_time_ms_ / 1000.0);
    printf("  │  Processing speed        │  %10.2f beams/sec │\n", beams_per_sec);
    std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";
    
    // Обновить профилирование для совместимости
    last_profiling_.total_time_ms = batch_total_cpu_time_ms_;
    last_profiling_.fft_time_ms = total_gpu_time;
    
    std::cout << "  ✅ Batch processing completed! (" << params_.beam_count << " beams in " 
              << std::fixed << std::setprecision(2) << (batch_total_cpu_time_ms_ / 1000.0) << " sec)\n\n";
    
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessBatch: Обработка одного батча лучей
// ════════════════════════════════════════════════════════════════════════════

std::vector<FFTResult> AntennaFFTProcMax::ProcessBatch(
    cl_mem input_signal,
    size_t start_beam,
    size_t num_beams,
    cl_command_queue batch_queue,
    cl_event* completion_event) {
    
    std::vector<FFTResult> results;
    results.reserve(num_beams);
    
    cl_int err;
    
    // Размер выходных данных
    size_t batch_output_size = num_beams * params_.out_count_points_fft;
    
    // ИСПОЛЬЗУЕМ КЭШИРОВАННЫЕ БУФЕРЫ (созданы в ProcessWithBatching)
    cl_mem fft_in = batch_fft_input_->Get();
    cl_mem fft_out = batch_fft_output_->Get();
    cl_mem sel_complex = batch_sel_complex_->Get();
    cl_mem sel_magnitude = batch_sel_magnitude_->Get();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 1: Padding kernel (БЕЗ КОПИРОВАНИЯ - работаем напрямую с input_signal!)
    // ═══════════════════════════════════════════════════════════════════════════
    
    cl_uint batch_beam_count = static_cast<cl_uint>(num_beams);
    cl_uint count_points = static_cast<cl_uint>(params_.count_points);
    cl_uint nfft = static_cast<cl_uint>(nFFT_);
    cl_uint beam_offset = static_cast<cl_uint>(start_beam);  // Смещение в лучах
    
    // Передаём исходный буфер и offset - БЕЗ КОПИРОВАНИЯ!
    err = clSetKernelArg(padding_kernel_, 0, sizeof(cl_mem), &input_signal);
    err |= clSetKernelArg(padding_kernel_, 1, sizeof(cl_mem), &fft_in);
    err |= clSetKernelArg(padding_kernel_, 2, sizeof(cl_uint), &batch_beam_count);
    err |= clSetKernelArg(padding_kernel_, 3, sizeof(cl_uint), &count_points);
    err |= clSetKernelArg(padding_kernel_, 4, sizeof(cl_uint), &nfft);
    err |= clSetKernelArg(padding_kernel_, 5, sizeof(cl_uint), &beam_offset);  // НОВЫЙ параметр!
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("ProcessBatch: set padding kernel args failed: " + std::to_string(err));
    }
    
    size_t padding_global_size = num_beams * nFFT_;
    cl_event event_padding = nullptr;
    err = clEnqueueNDRangeKernel(batch_queue, padding_kernel_, 1, nullptr, 
                                 &padding_global_size, nullptr, 0, nullptr, &event_padding);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("ProcessBatch: padding kernel failed: " + std::to_string(err));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 3: FFT (используем кэшированный план)
    // ═══════════════════════════════════════════════════════════════════════════
    
    cl_event event_fft = nullptr;
    
    clfftStatus status = clfftEnqueueTransform(
        batch_plan_handle_,
        CLFFT_FORWARD,
        1,
        &batch_queue,
        1, &event_padding,
        &event_fft,
        &fft_in,
        &fft_out,
        nullptr
    );
    
    if (status != CLFFT_SUCCESS) {
        clReleaseEvent(event_padding);
        throw std::runtime_error("ProcessBatch: clfftEnqueueTransform failed: " + std::to_string(status));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 4: Post-kernel (magnitude + select)
    // ═══════════════════════════════════════════════════════════════════════════
    
    cl_uint search_range = static_cast<cl_uint>(params_.out_count_points_fft);
    
    err = clSetKernelArg(post_kernel_, 0, sizeof(cl_mem), &fft_out);
    err |= clSetKernelArg(post_kernel_, 1, sizeof(cl_mem), &sel_complex);
    err |= clSetKernelArg(post_kernel_, 2, sizeof(cl_mem), &sel_magnitude);
    err |= clSetKernelArg(post_kernel_, 3, sizeof(cl_uint), &batch_beam_count);
    err |= clSetKernelArg(post_kernel_, 4, sizeof(cl_uint), &nfft);
    err |= clSetKernelArg(post_kernel_, 5, sizeof(cl_uint), &search_range);
    
    if (err != CL_SUCCESS) {
        clReleaseEvent(event_padding);
        clReleaseEvent(event_fft);
        throw std::runtime_error("ProcessBatch: set post kernel args failed: " + std::to_string(err));
    }
    
    size_t post_global_size = batch_output_size;
    cl_event event_post = nullptr;
    
    err = clEnqueueNDRangeKernel(batch_queue, post_kernel_, 1, nullptr, 
                                 &post_global_size, nullptr, 
                                 1, &event_fft,
                                 &event_post);
    
    if (err != CL_SUCCESS) {
        clReleaseEvent(event_padding);
        clReleaseEvent(event_fft);
        throw std::runtime_error("ProcessBatch: post kernel failed: " + std::to_string(err));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 5: Читаем результаты и находим максимумы
    // ═══════════════════════════════════════════════════════════════════════════
    
    clWaitForEvents(1, &event_post);
    
    // Читаем magnitude и complex
    std::vector<float> magnitudes(batch_output_size);
    std::vector<std::complex<float>> complexes(batch_output_size);
    
    err = clEnqueueReadBuffer(batch_queue, sel_magnitude, CL_TRUE, 0,
                              batch_output_size * sizeof(float), magnitudes.data(),
                              0, nullptr, nullptr);
    
    err |= clEnqueueReadBuffer(batch_queue, sel_complex, CL_TRUE, 0,
                               batch_output_size * sizeof(std::complex<float>), complexes.data(),
                               0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cerr << "ProcessBatch: read buffers failed: " << err << "\n";
    }
    
    // Найти максимумы для каждого луча в батче
    for (size_t beam = 0; beam < num_beams; ++beam) {
        FFTResult beam_result(params_.out_count_points_fft, params_.task_id, params_.module_name);
        
        // Найти топ-N максимумов для этого луча
        std::vector<std::pair<float, size_t>> mag_idx;
        mag_idx.reserve(params_.out_count_points_fft);
        
        for (size_t i = 0; i < params_.out_count_points_fft; ++i) {
            size_t idx = beam * params_.out_count_points_fft + i;
            mag_idx.push_back({magnitudes[idx], i});
        }
        
        // Сортировать по убыванию magnitude
        std::partial_sort(mag_idx.begin(), 
                         mag_idx.begin() + std::min(params_.max_peaks_count, mag_idx.size()),
                         mag_idx.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Взять топ-N
        for (size_t i = 0; i < std::min(params_.max_peaks_count, mag_idx.size()); ++i) {
            FFTMaxResult mv;
            mv.index_point = mag_idx[i].second;
            mv.amplitude = mag_idx[i].first;
            
            size_t cplx_idx = beam * params_.out_count_points_fft + mag_idx[i].second;
            float phase_rad = std::atan2(complexes[cplx_idx].imag(), complexes[cplx_idx].real());
            mv.phase = phase_rad * 180.0f / static_cast<float>(M_PI);
            
            beam_result.max_values.push_back(mv);
        }
        
        results.push_back(std::move(beam_result));
    }
    
    // Установить completion_event для этого батча
    if (completion_event) {
        *completion_event = event_post;
        // НЕ освобождаем event_post - он будет освобождён вызывающим кодом
    } else {
        clReleaseEvent(event_post);
    }
    
    // Освободить промежуточные события
    clReleaseEvent(event_padding);
    clReleaseEvent(event_fft);
    
    // FFT план НЕ освобождаем - он кэшируется для переиспользования!
    
    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// Основной метод обработки
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTResult AntennaFFTProcMax::Process(cl_mem input_signal) {
    
    // ═══════════════════════════════════════════════════════════════════════════
    // FFT с pre-callback + отдельный post-kernel
    // EVENT CHAIN для максимальной производительности!
    // ═══════════════════════════════════════════════════════════════════════════
    
    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    std::cout << "  FFT Pipeline (pre-callback + post-kernel + EVENT CHAIN)\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    
    // Создать план FFT с pre-callback (один раз)
    CreateFFTPlanWithPreCallbackOnly();
    
    // Создать буферы (один раз)
    size_t total_fft_size = params_.beam_count * nFFT_;
    
    if (!buffer_fft_input_) {
        buffer_fft_input_ = engine_->CreateBuffer(total_fft_size, gpu::MemoryType::GPU_READ_WRITE);
    }
    if (!buffer_fft_output_) {
        buffer_fft_output_ = engine_->CreateBuffer(total_fft_size, gpu::MemoryType::GPU_READ_WRITE);
    }
    
    cl_uint beam_count = static_cast<cl_uint>(params_.beam_count);
    cl_uint nfft = static_cast<cl_uint>(nFFT_);
    cl_uint search_range = static_cast<cl_uint>(params_.out_count_points_fft);
    
    if (!buffer_selected_complex_) {
        buffer_selected_complex_ = engine_->CreateBuffer(
            params_.beam_count * search_range, gpu::MemoryType::GPU_READ_WRITE);
    }
    if (!buffer_selected_magnitude_) {
        size_t float_elements = params_.beam_count * search_range;
        size_t complex_elements = (float_elements * sizeof(float) + sizeof(std::complex<float>) - 1) / sizeof(std::complex<float>);
        buffer_selected_magnitude_ = engine_->CreateBuffer(complex_elements, gpu::MemoryType::GPU_READ_WRITE);
    }
    
    if (!post_kernel_) {
        CreatePostKernel();
    }
    
    cl_int err;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 1: UPLOAD (non-blocking) → event_upload
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "\n[STEP 1] Upload данных (non-blocking)...\n";
    
    size_t pre_params_size = 32;
    size_t pre_input_size = params_.beam_count * params_.count_points * sizeof(std::complex<float>);
    
    cl_event event_upload = nullptr;
    err = clEnqueueCopyBuffer(
        queue_,
        input_signal,
        pre_callback_userdata_,
        0,
        pre_params_size,
        pre_input_size,
        0, nullptr,
        &event_upload  // OUTPUT EVENT
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clEnqueueCopyBuffer failed: " + std::to_string(err));
    }
    std::cout << "  → event_upload создан\n";
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 2: FFT (ждёт event_upload) → event_fft
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "\n[STEP 2] FFT с pre-callback (ждёт event_upload)...\n";
    
    cl_mem fft_input = buffer_fft_input_->Get();
    cl_mem fft_output = buffer_fft_output_->Get();
    cl_event event_fft = nullptr;
    
    clfftStatus status = clfftEnqueueTransform(
        plan_handle_,
        CLFFT_FORWARD,
        1,
        &queue_,
        1,                    // num_events_in_wait_list
        &event_upload,        // WAIT FOR event_upload!
        &event_fft,           // OUTPUT EVENT
        &fft_input,
        &fft_output,
        nullptr
    );
    
    if (status != CLFFT_SUCCESS) {
        clReleaseEvent(event_upload);
        throw std::runtime_error("clfftEnqueueTransform failed: " + std::to_string(status));
    }
    std::cout << "  → event_fft создан (ждёт event_upload)\n";
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 3: Post-kernel (ждёт event_fft) → event_post
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "\n[STEP 3] Post-kernel: Magnitude (ждёт event_fft)...\n";
    
    cl_mem selected_complex = buffer_selected_complex_->Get();
    cl_mem selected_magnitude = buffer_selected_magnitude_->Get();
    
    err = clSetKernelArg(post_kernel_, 0, sizeof(cl_mem), &fft_output);
    err |= clSetKernelArg(post_kernel_, 1, sizeof(cl_mem), &selected_complex);
    err |= clSetKernelArg(post_kernel_, 2, sizeof(cl_mem), &selected_magnitude);
    err |= clSetKernelArg(post_kernel_, 3, sizeof(cl_uint), &beam_count);
    err |= clSetKernelArg(post_kernel_, 4, sizeof(cl_uint), &nfft);
    err |= clSetKernelArg(post_kernel_, 5, sizeof(cl_uint), &search_range);
    
    if (err != CL_SUCCESS) {
        clReleaseEvent(event_upload);
        clReleaseEvent(event_fft);
        throw std::runtime_error("Failed to set post kernel args: " + std::to_string(err));
    }
    
    size_t post_global_size = params_.beam_count * search_range;
    cl_event event_post = nullptr;
    
    err = clEnqueueNDRangeKernel(
        queue_, 
        post_kernel_, 
        1, 
        nullptr, 
        &post_global_size, 
        nullptr, 
        1,                    // num_events_in_wait_list
        &event_fft,           // WAIT FOR event_fft!
        &event_post           // OUTPUT EVENT
    );
    
    if (err != CL_SUCCESS) {
        clReleaseEvent(event_upload);
        clReleaseEvent(event_fft);
        throw std::runtime_error("clEnqueueNDRangeKernel (post) failed: " + std::to_string(err));
    }
    std::cout << "  → event_post создан (ждёт event_fft)\n";
    
    // ═══════════════════════════════════════════════════════════════════════════
    // WAIT: Ждём только последний event (он тянет всю цепочку!)
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "\n[WAIT] Ждём event_post (тянет всю цепочку)...\n";
    clWaitForEvents(1, &event_post);
    std::cout << "  ✅ Все операции завершены!\n";
    
    // ═══════════════════════════════════════════════════════════════════════════
    // PROFILING: Собираем времена с каждого event
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "\n  ┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  PROFILING (EVENT CHAIN)                                    │\n";
    std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";
    
    last_profiling_.upload_time_ms = ProfileEvent(event_upload, "Upload");
    last_profiling_.fft_time_ms = ProfileEvent(event_fft, "FFT + pre-callback");
    last_profiling_.post_callback_time_ms = ProfileEvent(event_post, "Post-kernel");
    
    // Освобождаем events
    clReleaseEvent(event_upload);
    clReleaseEvent(event_fft);
    clReleaseEvent(event_post);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ПРОВЕРКА: FFT OUTPUT
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "\n  ┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  FFT OUTPUT (первые 15 точек спектра)                       │\n";
    std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";
    
    std::vector<std::complex<float>> fft_check(30);
    clEnqueueReadBuffer(queue_, fft_output, CL_TRUE, 0,
                       fft_check.size() * sizeof(std::complex<float>),
                       fft_check.data(), 0, nullptr, nullptr);
    
    float max_mag = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < fft_check.size(); ++i) {
        float m = std::sqrt(fft_check[i].real()*fft_check[i].real() + fft_check[i].imag()*fft_check[i].imag());
        if (m > max_mag) { max_mag = m; max_idx = i; }
    }
    
    std::cout << "  ┌────────┬──────────────────┬───────────────────────────────────┐\n";
    std::cout << "  │  Index │    Magnitude     │           Complex Value           │\n";
    std::cout << "  ├────────┼──────────────────┼───────────────────────────────────┤\n";
    
    for (size_t i = 0; i < 10; ++i) {
        float m = std::sqrt(fft_check[i].real()*fft_check[i].real() + fft_check[i].imag()*fft_check[i].imag());
        std::string marker = (i == max_idx) ? " ★ MAX" : "";
        printf("  │  %4zu  │  %12.4f    │  (%10.4f, %10.4f)  │%s\n", 
               i, m, fft_check[i].real(), fft_check[i].imag(), marker.c_str());
    }
    std::cout << "  └────────┴──────────────────┴───────────────────────────────────┘\n";
    std::cout << "\n  ✅ Max = " << max_mag << " at index " << max_idx << "\n";
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 4: Reduction - поиск максимумов
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "\n[STEP 4] Reduction: Поиск максимумов...\n";
    
    auto all_maxima = FindMaximaFromBuffers(selected_complex, selected_magnitude, search_range);
    
    AntennaFFTResult result(params_.beam_count, nFFT_, params_.task_id, params_.module_name);
    result.results.reserve(params_.beam_count);
    
    for (size_t beam_idx = 0; beam_idx < all_maxima.size(); ++beam_idx) {
        FFTResult beam_result(params_.out_count_points_fft, params_.task_id, params_.module_name);
        beam_result.max_values = std::move(all_maxima[beam_idx]);
        result.results.push_back(std::move(beam_result));
    }
    
    // Общее время GPU
    last_profiling_.total_time_ms = 
        last_profiling_.upload_time_ms + 
        last_profiling_.fft_time_ms + 
        last_profiling_.post_callback_time_ms;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ИТОГОВОЕ ПРОФИЛИРОВАНИЕ
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "\n  ┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  ИТОГО: GPU PROFILING                                       │\n";
    std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";
    
    printf("  ┌───────────────────────┬────────────────┐\n");
    printf("  │  Операция             │    Время (ms)  │\n");
    printf("  ├───────────────────────┼────────────────┤\n");
    printf("  │  Upload               │  %12.4f  │\n", last_profiling_.upload_time_ms);
    printf("  │  FFT + pre-callback   │  %12.4f  │\n", last_profiling_.fft_time_ms);
    printf("  │  Post-kernel          │  %12.4f  │\n", last_profiling_.post_callback_time_ms);
    printf("  ├───────────────────────┼────────────────┤\n");
    printf("  │  TOTAL GPU            │  %12.4f  │\n", last_profiling_.total_time_ms);
    printf("  └───────────────────────┴────────────────┘\n");
    
    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    std::cout << "  PIPELINE COMPLETE ✅\n";
    std::cout << "════════════════════════════════════════════════════════════════\n\n";
    
    return result;
}

AntennaFFTResult AntennaFFTProcMax::Process(const std::vector<std::complex<float>>& input_data) {
    // Создать буфер на GPU и загрузить данные
    size_t expected_size = params_.beam_count * params_.count_points;
    if (input_data.size() != expected_size) {
        throw std::invalid_argument("Input data size mismatch. Expected: " + 
                                   std::to_string(expected_size) + 
                                   ", got: " + std::to_string(input_data.size()));
    }
    
    auto buffer = engine_->CreateBufferWithData(input_data, gpu::MemoryType::GPU_READ_ONLY);
    return Process(buffer->Get());
}

// ════════════════════════════════════════════════════════════════════════════
// Управление clFFT планом
// ════════════════════════════════════════════════════════════════════════════

void AntennaFFTProcMax::CreateOrReuseFFTPlan() {
    // Проверить кэш
    PlanCacheKey key{params_.beam_count, params_.count_points, nFFT_, params_.out_count_points_fft, params_.max_peaks_count};
    
    {
        std::lock_guard<std::mutex> lock(plan_cache_mutex_);
        auto it = plan_cache_.find(key);
        if (it != plan_cache_.end()) {
            plan_handle_ = it->second;
            plan_created_ = true;
            return; // Переиспользовать существующий план
        }
    }
    
    // Создать новый план
    size_t clLengths[1] = {nFFT_};
    clfftStatus status = clfftCreateDefaultPlan(&plan_handle_, context_, CLFFT_1D, clLengths);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed with status: " + std::to_string(status));
    }
    
    // Настроить план
    clfftSetPlanPrecision(plan_handle_, CLFFT_SINGLE);
    clfftSetLayout(plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle_, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle_, params_.beam_count);
    
    size_t strides[1] = {1};
    size_t dist = nFFT_;
    clfftSetPlanInStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle_, dist, dist);
    
    // Регистрация callback'ов
    std::string pre_callback = GetPreCallbackSource();
    std::string post_callback = GetPostCallbackSource();

    // Создать userdata буферы для pre-callback
    struct PreCallbackUserData {
        cl_uint beam_count;
        cl_uint count_points;
        cl_uint nFFT;
        cl_uint padding;
    };

    PreCallbackUserData pre_cb_params = {
        static_cast<cl_uint>(params_.beam_count),
        static_cast<cl_uint>(params_.count_points),
        static_cast<cl_uint>(nFFT_),
        0
    };

    size_t pre_params_size = sizeof(PreCallbackUserData);
    size_t pre_input_size = params_.beam_count * params_.count_points * sizeof(std::complex<float>);
    size_t pre_userdata_size = pre_params_size + pre_input_size;

    if (pre_callback_userdata_) {
        clReleaseMemObject(pre_callback_userdata_);
    }

    cl_int err;
    pre_callback_userdata_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, pre_userdata_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create pre-callback userdata buffer: " + std::to_string(err));
    }

    // Записать параметры
    err = clEnqueueWriteBuffer(queue_, pre_callback_userdata_, CL_TRUE, 0, pre_params_size,
                               &pre_cb_params, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata_);
        throw std::runtime_error("Failed to write pre-callback params: " + std::to_string(err));
    }

    // Создать userdata буферы для post-callback
    // НОВЫЙ LAYOUT (без atomic locks!): params | complex_buffer | magnitude_buffer
    struct PostCallbackUserData {
        cl_uint beam_count;
        cl_uint nFFT;
        cl_uint out_count_points_fft;
        cl_uint max_peaks_count;
    };
    
    PostCallbackUserData post_cb_params = {
        static_cast<cl_uint>(params_.beam_count),
        static_cast<cl_uint>(nFFT_),
        static_cast<cl_uint>(params_.out_count_points_fft),
        static_cast<cl_uint>(params_.max_peaks_count)
    };
    
    size_t post_params_size = sizeof(PostCallbackUserData);
    size_t post_complex_size = params_.beam_count * params_.out_count_points_fft * sizeof(cl_float2);
    size_t post_magnitude_size = params_.beam_count * params_.out_count_points_fft * sizeof(float);
    size_t post_userdata_size = post_params_size + post_complex_size + post_magnitude_size;

    if (post_callback_userdata_) {
        clReleaseMemObject(post_callback_userdata_);
    }

    post_callback_userdata_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, post_userdata_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata_);
        throw std::runtime_error("Failed to create post-callback userdata buffer: " + std::to_string(err));
    }

    // Записать параметры
    err = clEnqueueWriteBuffer(queue_, post_callback_userdata_, CL_TRUE, 0, post_params_size,
                               &post_cb_params, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata_);
        clReleaseMemObject(post_callback_userdata_);
        throw std::runtime_error("Failed to write post-callback params: " + std::to_string(err));
    }

    // Зарегистрировать callback'ы
    status = clfftSetPlanCallback(plan_handle_, "prepareDataPre", pre_callback.c_str(), 0,
                                  PRECALLBACK, &pre_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata_);
        clReleaseMemObject(post_callback_userdata_);
        throw std::runtime_error("clfftSetPlanCallback (pre) failed: " + std::to_string(status));
    }

    status = clfftSetPlanCallback(plan_handle_, "processFFTPost", post_callback.c_str(), 0,
                                  POSTCALLBACK, &post_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        clReleaseMemObject(pre_callback_userdata_);
        clReleaseMemObject(post_callback_userdata_);
        throw std::runtime_error("clfftSetPlanCallback (post) failed: " + std::to_string(status));
    }
    
    // Скомпилировать план
    status = clfftBakePlan(plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftBakePlan failed with status: " + std::to_string(status));
    }
    
    plan_created_ = true;
    
    // Сохранить в кэш
    {
        std::lock_guard<std::mutex> lock(plan_cache_mutex_);
        plan_cache_[key] = plan_handle_;
    }
}

void AntennaFFTProcMax::ReleaseFFTPlan() {
    if (plan_created_ && plan_handle_ != 0) {
        // Не удаляем из кэша, так как может использоваться другими экземплярами
        // clfftDestroyPlan(&plan_handle_);
        plan_handle_ = 0;
        plan_created_ = false;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ОТЛАДОЧНЫЕ методы: FFT без callback'ов
// ════════════════════════════════════════════════════════════════════════════

void AntennaFFTProcMax::CreateFFTPlanNoCallbacks() {
    // Если план уже создан, освободить
    if (plan_created_) {
        return; // Уже есть план
    }
    
    // Создать новый план БЕЗ callback'ов
    size_t clLengths[1] = {nFFT_};
    clfftStatus status = clfftCreateDefaultPlan(&plan_handle_, context_, CLFFT_1D, clLengths);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed: " + std::to_string(status));
    }
    
    // Настроить план
    clfftSetPlanPrecision(plan_handle_, CLFFT_SINGLE);
    clfftSetLayout(plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle_, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle_, params_.beam_count);
    
    size_t strides[1] = {1};
    size_t dist = nFFT_;
    clfftSetPlanInStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle_, dist, dist);
    
    // НЕ регистрируем callback'и!
    
    // Скомпилировать план
    status = clfftBakePlan(plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftBakePlan failed: " + std::to_string(status));
    }
    
    plan_created_ = true;
    std::cout << "  Created FFT plan without callbacks (nFFT=" << nFFT_ << ", batch=" << params_.beam_count << ")\n";
}

void AntennaFFTProcMax::CreateFFTPlanWithPreCallbackOnly() {
    // Если план уже создан, используем его
    if (plan_created_) {
        return;
    }
    
    std::cout << "  Creating FFT plan with ONLY pre-callback...\n";
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 1. Создать userdata буфер для pre-callback (как в LOpenCl!)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Структура с padding до 32 байт (8 x uint)
    struct PreCallbackUserData {
        cl_uint beam_count;      // 0
        cl_uint count_points;    // 4
        cl_uint nFFT;            // 8
        cl_uint padding1;        // 12
        cl_uint padding2;        // 16
        cl_uint padding3;        // 20
        cl_uint padding4;        // 24
        cl_uint padding5;        // 28
    };  // = 32 байта
    
    PreCallbackUserData pre_cb_params = {
        static_cast<cl_uint>(params_.beam_count),
        static_cast<cl_uint>(params_.count_points),
        static_cast<cl_uint>(nFFT_),
        0, 0, 0, 0, 0
    };
    
    size_t pre_params_size = sizeof(PreCallbackUserData);  // 32 байта
    size_t pre_input_size = params_.beam_count * params_.count_points * sizeof(std::complex<float>);
    size_t pre_userdata_size = pre_params_size + pre_input_size;
    
    std::cout << "  PreCallbackUserData size = " << pre_params_size << " bytes\n";
    std::cout << "  Input data size = " << pre_input_size << " bytes\n";
    std::cout << "  Total userdata size = " << pre_userdata_size << " bytes\n";
    
    // Освободить старый буфер если есть
    if (pre_callback_userdata_) {
        clReleaseMemObject(pre_callback_userdata_);
    }
    
    cl_int err;
    pre_callback_userdata_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, pre_userdata_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create pre_callback_userdata: " + std::to_string(err));
    }
    
    // Записать параметры
    err = clEnqueueWriteBuffer(queue_, pre_callback_userdata_, CL_TRUE, 0, pre_params_size,
                               &pre_cb_params, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to write pre_callback params: " + std::to_string(err));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 2. Создать план FFT
    // ═══════════════════════════════════════════════════════════════════════════
    
    size_t clLengths[1] = {nFFT_};
    clfftStatus status = clfftCreateDefaultPlan(&plan_handle_, context_, CLFFT_1D, clLengths);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed: " + std::to_string(status));
    }
    
    clfftSetPlanPrecision(plan_handle_, CLFFT_SINGLE);
    clfftSetLayout(plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle_, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle_, params_.beam_count);
    
    size_t strides[1] = {1};
    size_t dist = nFFT_;
    clfftSetPlanInStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle_, dist, dist);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 3. Зарегистрировать ТОЛЬКО pre-callback
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Callback source с 32-байтной структурой (как в LOpenCl!)
    const char* pre_callback_source = 
        "typedef struct { "
        "    uint beam_count; "
        "    uint count_points; "
        "    uint nFFT; "
        "    uint padding1; "
        "    uint padding2; "
        "    uint padding3; "
        "    uint padding4; "
        "    uint padding5; "
        "} PreCallbackUserData; "
        "float2 prepareDataPre(__global void* input, uint inoffset, __global void* userdata) { "
        "    __global PreCallbackUserData* params = (__global PreCallbackUserData*)userdata; "
        "    __global float2* input_signal = (__global float2*)((__global char*)userdata + 32); " // Хардкод 32 байта!
        "    uint beam_count = params->beam_count; "
        "    uint count_points = params->count_points; "
        "    uint nFFT = params->nFFT; "
        "    uint beam_idx = inoffset / nFFT; "
        "    uint pos_in_fft = inoffset % nFFT; "
        "    if (beam_idx >= beam_count) { "
        "        return (float2)(0.0f, 0.0f); "
        "    } "
        "    if (pos_in_fft < count_points) { "
        "        uint input_idx = beam_idx * count_points + pos_in_fft; "
        "        return input_signal[input_idx]; "
        "    } else { "
        "        return (float2)(0.0f, 0.0f); "
        "    } "
        "}";
    
    status = clfftSetPlanCallback(plan_handle_, "prepareDataPre", pre_callback_source, 0,
                                  PRECALLBACK, &pre_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftSetPlanCallback (pre) failed: " + std::to_string(status));
    }
    
    std::cout << "  Pre-callback registered\n";
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 4. Скомпилировать план
    // ═══════════════════════════════════════════════════════════════════════════
    
    status = clfftBakePlan(plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftBakePlan failed: " + std::to_string(status));
    }
    
    plan_created_ = true;
    std::cout << "  ✅ FFT plan with pre-callback created (nFFT=" << nFFT_ << ", batch=" << params_.beam_count << ")\n";
}

void AntennaFFTProcMax::CreatePaddingKernel() {
    // Kernel для padding данных: count_points → nFFT
    // С поддержкой beam_offset для batch processing БЕЗ КОПИРОВАНИЯ!
    const char* kernel_source = R"CL(
        __kernel void padding_kernel(
            __global const float2* input,    // Входные данные: ПОЛНЫЙ буфер (все лучи)
            __global float2* output,         // Выходные данные: batch_beam_count * nFFT  
            uint batch_beam_count,           // Количество лучей в батче
            uint count_points,               // Точек на луч
            uint nFFT,                       // Размер FFT
            uint beam_offset                 // Смещение в лучах (для batch processing)
        ) {
            uint gid = get_global_id(0);
            
            // gid = local_beam_idx * nFFT + pos_in_fft
            uint local_beam_idx = gid / nFFT;
            uint pos_in_fft = gid % nFFT;
            
            if (local_beam_idx >= batch_beam_count) return;
            
            // Глобальный индекс луча = local + offset
            uint global_beam_idx = local_beam_idx + beam_offset;
            
            if (pos_in_fft < count_points) {
                // Читаем из глобального индекса, пишем в локальный
                uint src_idx = global_beam_idx * count_points + pos_in_fft;
                output[gid] = input[src_idx];
            } else {
                // Zero-padding
                output[gid] = (float2)(0.0f, 0.0f);
            }
        }
    )CL";
    
    cl_int err;
    const char* sources[] = {kernel_source};
    size_t lengths[] = {strlen(kernel_source)};
    
    cl_program program = clCreateProgramWithSource(context_, 1, sources, lengths, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create padding program: " + std::to_string(err));
    }
    
    err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Получить лог сборки
        size_t log_size;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Padding kernel build error:\n" << log.data() << "\n";
        clReleaseProgram(program);
        throw std::runtime_error("Failed to build padding program");
    }
    
    padding_kernel_ = clCreateKernel(program, "padding_kernel", &err);
    clReleaseProgram(program);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create padding kernel: " + std::to_string(err));
    }
    
    std::cout << "  Created padding_kernel\n";
}

void AntennaFFTProcMax::CreatePostKernel() {
    // Kernel для post-processing: вычисление magnitude и выбор search_range точек
    const char* kernel_source = R"CL(
        __kernel void post_kernel(
            __global const float2* fft_output,     // FFT результат: beam_count * nFFT
            __global float2* selected_complex,     // Выбранные точки: beam_count * search_range
            __global float* selected_magnitude,    // Magnitude: beam_count * search_range
            uint beam_count,
            uint nFFT,
            uint search_range
        ) {
            uint gid = get_global_id(0);
            
            // gid = beam_idx * search_range + pos
            uint beam_idx = gid / search_range;
            uint pos = gid % search_range;
            
            if (beam_idx >= beam_count || pos >= search_range) return;
            
            // Индекс в FFT output (берем первые search_range точек)
            uint fft_idx = beam_idx * nFFT + pos;
            
            float2 val = fft_output[fft_idx];
            float mag = sqrt(val.x * val.x + val.y * val.y);
            
            selected_complex[gid] = val;
            selected_magnitude[gid] = mag;
        }
    )CL";
    
    cl_int err;
    const char* sources[] = {kernel_source};
    size_t lengths[] = {strlen(kernel_source)};
    
    cl_program program = clCreateProgramWithSource(context_, 1, sources, lengths, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create post program: " + std::to_string(err));
    }
    
    err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Post kernel build error:\n" << log.data() << "\n";
        clReleaseProgram(program);
        throw std::runtime_error("Failed to build post program");
    }
    
    post_kernel_ = clCreateKernel(program, "post_kernel", &err);
    clReleaseProgram(program);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create post kernel: " + std::to_string(err));
    }
    
    std::cout << "  Created post_kernel\n";
}

std::vector<std::vector<FFTMaxResult>> AntennaFFTProcMax::FindMaximaFromBuffers(
    cl_mem selected_complex, cl_mem selected_magnitude, size_t search_range) {
    
    // Простая реализация: читаем magnitude на CPU и ищем максимумы
    std::vector<std::vector<FFTMaxResult>> result(params_.beam_count);
    
    size_t total_size = params_.beam_count * search_range;
    std::vector<float> magnitudes(total_size);
    std::vector<std::complex<float>> complexes(total_size);
    
    cl_int err;
    err = clEnqueueReadBuffer(queue_, selected_magnitude, CL_TRUE, 0,
                              total_size * sizeof(float), magnitudes.data(),
                              0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read selected_magnitude: " << err << "\n";
        return result;
    }
    
    err = clEnqueueReadBuffer(queue_, selected_complex, CL_TRUE, 0,
                              total_size * sizeof(std::complex<float>), complexes.data(),
                              0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read selected_complex: " << err << "\n";
        return result;
    }
    
    // Для каждого луча найти топ-N максимумов
    for (size_t beam = 0; beam < params_.beam_count; ++beam) {
        std::vector<std::pair<float, size_t>> mag_idx;
        mag_idx.reserve(search_range);
        
        for (size_t i = 0; i < search_range; ++i) {
            size_t idx = beam * search_range + i;
            mag_idx.push_back({magnitudes[idx], i});
        }
        
        // Сортировать по убыванию magnitude
        std::partial_sort(mag_idx.begin(), 
                         mag_idx.begin() + std::min(params_.max_peaks_count, mag_idx.size()),
                         mag_idx.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Взять топ-N
        for (size_t i = 0; i < std::min(params_.max_peaks_count, mag_idx.size()); ++i) {
            FFTMaxResult mv;
            mv.index_point = mag_idx[i].second;
            mv.amplitude = mag_idx[i].first;
            
            size_t cplx_idx = beam * search_range + mag_idx[i].second;
            float phase_rad = std::atan2(complexes[cplx_idx].imag(), complexes[cplx_idx].real());
            mv.phase = phase_rad * 180.0f / static_cast<float>(M_PI); // В градусы
            
            result[beam].push_back(mv);
        }
        
        std::cout << "  Beam " << beam << ": top magnitude = " << (result[beam].empty() ? 0.0f : result[beam][0].amplitude) 
                  << " at index " << (result[beam].empty() ? 0UL : result[beam][0].index_point) << "\n";
    }
    
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Callback функции
// ════════════════════════════════════════════════════════════════════════════

std::string AntennaFFTProcMax::GetPreCallbackSource() const {
    // Pre-callback: перенос данных из входного буфера в блоки nFFT с padding
    return R"(
        typedef struct {
            uint beam_count;
            uint count_points;
            uint nFFT;
            uint padding;
        } PreCallbackUserData;

        float2 prepareDataPre(__global void* input, uint inoffset, __global void* userdata) {
            __global PreCallbackUserData* params = (__global PreCallbackUserData*)userdata;
            __global float2* input_signal = (__global float2*)((__global char*)userdata + sizeof(PreCallbackUserData));

            uint beam_count = params->beam_count;
            uint count_points = params->count_points;
            uint nFFT = params->nFFT;

            // Вычислить индекс луча и позицию в блоке nFFT
            uint beam_idx = inoffset / nFFT;
            uint pos_in_fft = inoffset % nFFT;

            if (beam_idx >= beam_count) {
                return (float2)(0.0f, 0.0f);
            }

            // Если позиция в пределах count_points - копируем данные
            if (pos_in_fft < count_points) {
                uint input_idx = beam_idx * count_points + pos_in_fft;
                return input_signal[input_idx];
            } else {
                // Остальное - padding (нули)
                return (float2)(0.0f, 0.0f);
            }
        }
    )";
}

std::string AntennaFFTProcMax::GetPostCallbackSource() const {
    // Post-callback: ТОЛЬКО fftshift + magnitude + complex write
    // БЕЗ atomic locks! Поиск максимумов в отдельном kernel после FFT
    // Аналогично референсу: E:\C++\Cuda\OpenCLProd\LOpenCl\Custom\Native\clFFTCallbacks.h
    return R"(
        typedef struct {
            uint beam_count;
            uint nFFT;
            uint out_count_points_fft;
            uint max_peaks_count;
        } PostCallbackUserData;

        void processFFTPost(__global void* output, uint outoffset, __global void* userdata, float2 fftoutput) {
            __global PostCallbackUserData* params = (__global PostCallbackUserData*)userdata;

            uint beam_count = params->beam_count;
            uint nFFT = params->nFFT;
            uint out_count_points_fft = params->out_count_points_fft;

            // Вычислить индекс луча и позицию в FFT
            uint beam_idx = outoffset / nFFT;
            uint pos_in_fft = outoffset % nFFT;

            if (beam_idx >= beam_count) {
                return;
            }

            // Диапазоны для fftshift:
            // Диапазон 1 (отрицательные частоты): [nFFT - out_count_points_fft/2, nFFT - 1]
            // Диапазон 2 (положительные частоты): [0, out_count_points_fft/2 - 1]
            uint half_size = out_count_points_fft / 2;
            uint range1_start = nFFT - half_size;

            // Быстрая проверка - 99.9% потоков выходят здесь!
            bool in_range1 = (pos_in_fft >= range1_start);
            bool in_range2 = (pos_in_fft < half_size);

            if (!in_range1 && !in_range2) {
                return;  // Не в диапазоне fftshift - выходим быстро
            }

            // Вычислить индекс в выходном буфере (после fftshift)
            uint output_idx;
            if (in_range1) {
                // Отрицательные частоты → начало выходного буфера
                output_idx = pos_in_fft - range1_start;
            } else {
                // Положительные частоты → после отрицательных
                output_idx = half_size + pos_in_fft;
            }

            // Layout userdata: params | complex_buffer | magnitude_buffer
            __global float2* complex_buffer = (__global float2*)((__global char*)userdata + sizeof(PostCallbackUserData));
            __global float* magnitude_buffer = (__global float*)(complex_buffer + (beam_count * out_count_points_fft));

            uint base_idx = beam_idx * out_count_points_fft + output_idx;

            // Записать комплексный спектр (только для fftshift диапазона)
            complex_buffer[base_idx] = fftoutput;

            // Записать magnitude (без atomic - просто прямая запись)
            magnitude_buffer[base_idx] = length(fftoutput);
        }
    )";
}


void AntennaFFTProcMax::CreateMaxReductionKernel() {
    // Kernel для поиска top-N максимумов + вычисления phase
    // Запускается ПОСЛЕ FFT - один work-group на beam (parallel reduction)
    std::string reduction_kernel_source = R"(
        typedef struct {
            uint index;
            float magnitude;
            float phase;
            uint pad;
        } MaxValue;

        // Kernel для поиска top-N максимумов и вычисления phase
        // Один work-group обрабатывает один beam
        __kernel void findMaximaAndPhase(
            __global const float2* complex_buffer,  // Комплексный спектр после fftshift
            __global const float* magnitude_buffer, // Magnitude после fftshift
            __global MaxValue* maxima_buffer,       // Выходной буфер для top-N максимумов
            uint beam_count,
            uint out_count_points_fft,
            uint max_peaks_count
        ) {
            uint beam_idx = get_group_id(0);
            uint lid = get_local_id(0);
            uint local_size = get_local_size(0);
            
            if (beam_idx >= beam_count) return;
            
            // Local memory для top-N максимумов этого beam
            __local MaxValue local_max[8];  // max_peaks_count <= 8
            __local float local_mag[1024];  // Кэш magnitude для быстрого поиска
            __local uint local_idx[1024];   // Кэш индексов
            
            // Инициализация top-N (только первые потоки)
            if (lid < max_peaks_count) {
                local_max[lid].index = UINT_MAX;
                local_max[lid].magnitude = -1.0f;
                local_max[lid].phase = 0.0f;
                local_max[lid].pad = 0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Загрузить magnitude в local memory (каждый поток загружает несколько элементов)
            uint base_offset = beam_idx * out_count_points_fft;
            for (uint i = lid; i < out_count_points_fft; i += local_size) {
                local_mag[i] = magnitude_buffer[base_offset + i];
                local_idx[i] = i;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Поиск top-N (только первый поток - простой алгоритм)
            if (lid == 0) {
                for (uint k = 0; k < max_peaks_count; ++k) {
                    float max_mag = -1.0f;
                    uint max_idx = UINT_MAX;
                    
                    // Найти максимум среди оставшихся
                    for (uint i = 0; i < out_count_points_fft; ++i) {
                        if (local_mag[i] > max_mag) {
                            max_mag = local_mag[i];
                            max_idx = local_idx[i];
                        }
                    }
                    
                    if (max_idx != UINT_MAX && max_mag > 0.0f) {
                        // Вычислить phase
                        float2 cval = complex_buffer[base_offset + max_idx];
                        float phase_deg = atan2(cval.y, cval.x) * 57.2957795f;
                        
                        local_max[k].index = max_idx;
                        local_max[k].magnitude = max_mag;
                        local_max[k].phase = phase_deg;
                        
                        // Пометить как использованный
                        local_mag[max_idx] = -1.0f;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Записать результаты в глобальную память
            if (lid < max_peaks_count) {
                uint out_idx = beam_idx * max_peaks_count + lid;
                maxima_buffer[out_idx] = local_max[lid];
            }
        }
    )";

    reduction_program_ = engine_->LoadProgram(reduction_kernel_source);
    reduction_kernel_ = engine_->GetKernel(reduction_program_, "findMaximaAndPhase");
}

std::vector<std::vector<FFTMaxResult>> AntennaFFTProcMax::FindMaximaAllBeamsOnGPU(
    cl_event wait_event, cl_event* out_reduction_event, cl_event* out_read_event) {
    (void)wait_event;
    (void)out_reduction_event;
    (void)out_read_event;
    if (!post_callback_userdata_) {
        throw std::runtime_error("post_callback_userdata_ is not initialized");
    }

    // Layout: params | complex_buffer | magnitude_buffer
    size_t post_params_size = sizeof(cl_uint) * 4;
    size_t post_complex_size = params_.beam_count * params_.out_count_points_fft * sizeof(cl_float2);
    size_t post_magnitude_size = params_.beam_count * params_.out_count_points_fft * sizeof(float);
    size_t maxima_size = params_.beam_count * params_.max_peaks_count * sizeof(MaxValue);

    // ═══════════════════════════════════════════════════════════════════════════
    // ЭТАП 1: Создать kernel если его нет
    // ═══════════════════════════════════════════════════════════════════════════
    
    if (!reduction_kernel_) {
        CreateMaxReductionKernel();
    }
    
    // Создать буфер для результатов максимумов
    if (!buffer_maxima_) {
        const size_t maxima_elements = (maxima_size + sizeof(std::complex<float>) - 1) / sizeof(std::complex<float>);
        buffer_maxima_ = engine_->CreateBuffer(maxima_elements, gpu::MemoryType::GPU_READ_WRITE);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ЭТАП 2: Создать sub-buffers для complex и magnitude
    // ═══════════════════════════════════════════════════════════════════════════
    
    cl_int err;
    cl_buffer_region complex_region = {post_params_size, post_complex_size};
    cl_mem complex_sub_buffer = clCreateSubBuffer(
        post_callback_userdata_,
        CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION,
        &complex_region,
        &err
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create complex sub-buffer: " + std::to_string(err));
    }
    
    cl_buffer_region magnitude_region = {post_params_size + post_complex_size, post_magnitude_size};
    cl_mem magnitude_sub_buffer = clCreateSubBuffer(
        post_callback_userdata_,
        CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION,
        &magnitude_region,
        &err
    );
    if (err != CL_SUCCESS) {
        clReleaseMemObject(complex_sub_buffer);
        throw std::runtime_error("Failed to create magnitude sub-buffer: " + std::to_string(err));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ЭТАП 3: Запустить reduction kernel (findMaximaAndPhase)
    // ═══════════════════════════════════════════════════════════════════════════
    
    cl_uint beam_count = static_cast<cl_uint>(params_.beam_count);
    cl_uint out_count_points_fft = static_cast<cl_uint>(params_.out_count_points_fft);
    cl_uint max_peaks_count = static_cast<cl_uint>(params_.max_peaks_count);
    cl_mem maxima_mem = buffer_maxima_->Get();
    
    clSetKernelArg(reduction_kernel_, 0, sizeof(cl_mem), &complex_sub_buffer);
    clSetKernelArg(reduction_kernel_, 1, sizeof(cl_mem), &magnitude_sub_buffer);
    clSetKernelArg(reduction_kernel_, 2, sizeof(cl_mem), &maxima_mem);
    clSetKernelArg(reduction_kernel_, 3, sizeof(cl_uint), &beam_count);
    clSetKernelArg(reduction_kernel_, 4, sizeof(cl_uint), &out_count_points_fft);
    clSetKernelArg(reduction_kernel_, 5, sizeof(cl_uint), &max_peaks_count);
    
    // Один work-group на beam, local_size подбирается автоматически
    size_t global_work_size = params_.beam_count * 256;  // 256 потоков на beam
    size_t local_work_size = 256;
    
    // Ограничение: out_count_points_fft <= 1024 для local memory
    if (params_.out_count_points_fft > 1024) {
        local_work_size = 64;  // Уменьшить для больших размеров
        global_work_size = params_.beam_count * local_work_size;
    }
    
    cl_event reduction_event = nullptr;
    err = clEnqueueNDRangeKernel(
        queue_,
        reduction_kernel_,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr, &reduction_event
    );
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(complex_sub_buffer);
        clReleaseMemObject(magnitude_sub_buffer);
        throw std::runtime_error("Failed to enqueue reduction kernel: " + std::to_string(err));
    }
    
    // Профилирование
    last_profiling_.reduction_time_ms = ProfileEvent(reduction_event, "Reduction + Phase");
    clWaitForEvents(1, &reduction_event);
    clReleaseEvent(reduction_event);
    
    clReleaseMemObject(complex_sub_buffer);
    clReleaseMemObject(magnitude_sub_buffer);

    // ═══════════════════════════════════════════════════════════════════════════
    // ЭТАП 4: Прочитать результаты с GPU
    // ═══════════════════════════════════════════════════════════════════════════
    
    std::vector<MaxValue> maxima_result(params_.beam_count * params_.max_peaks_count);
    err = clEnqueueReadBuffer(
        queue_,
        buffer_maxima_->Get(),
        CL_TRUE,
        0,
        maxima_size,
        maxima_result.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read maxima from GPU: " + std::to_string(err));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ЭТАП 5: Преобразовать в FFTMaxResult
    // ═══════════════════════════════════════════════════════════════════════════
    
    std::vector<std::vector<FFTMaxResult>> all_results;
    all_results.resize(params_.beam_count);
    for (size_t beam_idx = 0; beam_idx < params_.beam_count; ++beam_idx) {
        auto& beam_out = all_results[beam_idx];
        beam_out.reserve(params_.max_peaks_count);
        for (size_t i = 0; i < params_.max_peaks_count; ++i) {
            const auto& mv = maxima_result[beam_idx * params_.max_peaks_count + i];
            if (mv.index != UINT_MAX && mv.magnitude > 0.0f) {
                FFTMaxResult max_result;
                max_result.index_point = mv.index;
                max_result.amplitude = mv.magnitude;
                max_result.phase = mv.phase;
                beam_out.push_back(max_result);
            }
        }
    }

    return all_results;
}

double AntennaFFTProcMax::ProfileEvent(cl_event event, const std::string& operation_name) {
    if (!event) return 0.0;
    (void)operation_name;
    
    cl_ulong start_time, end_time;
    cl_int err;
    
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
    if (err != CL_SUCCESS) return 0.0;
    
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
    if (err != CL_SUCCESS) return 0.0;
    
    double time_ms = (end_time - start_time) / 1e6; // наносекунды в миллисекунды
    return time_ms;
}

void AntennaFFTProcMax::PrintResults(const AntennaFFTResult& result) const {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  AntennaFFTProcMax Results\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "Task ID: " << result.task_id << "\n";
    std::cout << "Module: " << result.module_name << "\n";
    std::cout << "Total Beams: " << result.total_beams << "\n";
    std::cout << "nFFT: " << result.nFFT << "\n\n";
    
    for (size_t i = 0; i < result.results.size(); ++i) {
        std::cout << "Beam " << i << ":\n";
        std::cout << "  Max Values Found: " << result.results[i].max_values.size() << "\n";
        for (size_t j = 0; j < result.results[i].max_values.size(); ++j) {
            const auto& max_val = result.results[i].max_values[j];
            std::cout << "    [" << j << "] Index: " << max_val.index_point 
                      << ", Amplitude: " << std::fixed << std::setprecision(6) << max_val.amplitude
                      << ", Phase: " << max_val.phase << "°\n";
        }
        std::cout << "\n";
    }
}

void AntennaFFTProcMax::SaveResultsToFile(const AntennaFFTResult& result, const std::string& filepath) {
    std::string base_path = filepath;
    if (filepath.empty()) {
        base_path = "antenna_result.md";
    }
    if (base_path.find("/") != 0 && base_path.find(":\\") == std::string::npos) {
        // Относительный путь - добавить Reports/
        base_path = "Reports/" + base_path;
    }

    // Привести к базовому имени без расширения
    std::string base_no_ext = base_path;
    size_t dot_pos = base_no_ext.find_last_of('.');
    if (dot_pos != std::string::npos) {
        base_no_ext = base_no_ext.substr(0, dot_pos);
    }

    std::string md_path = base_no_ext + ".md";
    std::string json_path = base_no_ext + ".json";

    std::ofstream md_file(md_path);
    if (!md_file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + md_path);
    }

    // Получить timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    #ifdef _WIN32
    localtime_s(&tm_buf, &time_t);
    #else
    localtime_r(&time_t, &tm_buf);
    #endif
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

    // Записать таблицу
    md_file << "# AntennaFFTProcMax Results\n\n";
    md_file << "**Generated:** " << time_str << "\n\n";
    md_file << "**Task ID:** " << result.task_id << "\n";
    md_file << "**Module:** " << result.module_name << "\n";
    md_file << "**Total Beams:** " << result.total_beams << "\n";
    md_file << "**nFFT:** " << result.nFFT << "\n\n";

    md_file << "## Profiling (GPU events)\n\n";
    md_file << "Upload Time:        " << std::fixed << std::setprecision(3) << last_profiling_.upload_time_ms << " ms\n";
    md_file << "FFT Time:           " << std::fixed << std::setprecision(3) << last_profiling_.fft_time_ms << " ms\n";
    md_file << "Post-Callback Time: " << std::fixed << std::setprecision(3) << last_profiling_.post_callback_time_ms << " ms\n";
    md_file << "Reduction Time:     " << std::fixed << std::setprecision(3) << last_profiling_.reduction_time_ms << " ms\n";
    md_file << "Total Time:         " << std::fixed << std::setprecision(3) << last_profiling_.total_time_ms << " ms\n\n";

    md_file << "## Results by Beam\n\n";
    md_file << "| Beam | Index | Amplitude | Phase (deg) |\n";
    md_file << "|------|-------|-----------|-------------|\n";

    for (size_t i = 0; i < result.results.size(); ++i) {
        const auto& beam_result = result.results[i];
        if (beam_result.max_values.empty()) {
            md_file << "| " << i << " | - | - | - |\n";
        } else {
            for (size_t j = 0; j < beam_result.max_values.size(); ++j) {
                const auto& max_val = beam_result.max_values[j];
                md_file << "| " << i << " | " << max_val.index_point
                        << " | " << std::fixed << std::setprecision(6) << max_val.amplitude
                        << " | " << std::setprecision(2) << max_val.phase << " |\n";
            }
        }
    }

    md_file.close();

    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + json_path);
    }

    // Считать FFT комплексный вектор (fftshift) из post_callback_userdata_
    std::vector<std::complex<float>> fft_data;
    if (post_callback_userdata_) {
        size_t post_params_size = sizeof(cl_uint) * 4;
        size_t post_complex_size = params_.beam_count * params_.out_count_points_fft * sizeof(cl_float2);
        fft_data.resize(params_.beam_count * params_.out_count_points_fft);
        cl_int err = clEnqueueReadBuffer(
            queue_,
            post_callback_userdata_,
            CL_TRUE,
            post_params_size,
            post_complex_size,
            fft_data.data(),
            0, nullptr, nullptr
        );
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read complex FFT data from post_callback_userdata: " + std::to_string(err));
        }
    }

    json_file << "{\n";
    json_file << "  \"task_id\": \"" << result.task_id << "\",\n";
    json_file << "  \"module_name\": \"" << result.module_name << "\",\n";
    json_file << "  \"total_beams\": " << result.total_beams << ",\n";
    json_file << "  \"nFFT\": " << result.nFFT << ",\n";
    json_file << "  \"profiling_ms\": {\n";
    json_file << "    \"upload\": " << std::fixed << std::setprecision(3) << last_profiling_.upload_time_ms << ",\n";
    json_file << "    \"fft\": " << std::fixed << std::setprecision(3) << last_profiling_.fft_time_ms << ",\n";
    json_file << "    \"post_callback\": " << std::fixed << std::setprecision(3) << last_profiling_.post_callback_time_ms << ",\n";
    json_file << "    \"reduction\": " << std::fixed << std::setprecision(3) << last_profiling_.reduction_time_ms << ",\n";
    json_file << "    \"total\": " << std::fixed << std::setprecision(3) << last_profiling_.total_time_ms << "\n";
    json_file << "  },\n";
    json_file << "  \"results\": [\n";

    for (size_t i = 0; i < result.results.size(); ++i) {
        const auto& beam_result = result.results[i];
        json_file << "    {\n";
        json_file << "      \"beam_index\": " << i << ",\n";
        json_file << "      \"v_fft\": " << beam_result.v_fft << ",\n";
        json_file << "      \"max_values\": [\n";

        for (size_t j = 0; j < beam_result.max_values.size(); ++j) {
            const auto& max_val = beam_result.max_values[j];
            json_file << "        {\n";
            json_file << "          \"index_point\": " << max_val.index_point << ",\n";
            json_file << "          \"amplitude\": " << std::fixed << std::setprecision(6) << max_val.amplitude << ",\n";
            json_file << "          \"phase\": " << std::fixed << std::setprecision(2) << max_val.phase << "\n";
            json_file << "        }";
            if (j < beam_result.max_values.size() - 1) json_file << ",";
            json_file << "\n";
        }

        json_file << "      ],\n";
        json_file << "      \"fft_complex\": [\n";
        if (!fft_data.empty()) {
            size_t beam_offset = i * params_.out_count_points_fft;
            for (size_t k = 0; k < params_.out_count_points_fft; ++k) {
                size_t idx = beam_offset + k;
                if (idx < fft_data.size()) {
                    json_file << "        [" << std::fixed << std::setprecision(6)
                              << fft_data[idx].real() << ", " << fft_data[idx].imag() << "]";
                } else {
                    json_file << "        [0.0, 0.0]";
                }
                if (k + 1 < params_.out_count_points_fft) json_file << ",";
                json_file << "\n";
            }
        }
        json_file << "      ]\n";
        json_file << "    }";
        if (i < result.results.size() - 1) json_file << ",";
        json_file << "\n";
    }

    json_file << "  ]\n";
    json_file << "}\n";
    json_file.close();
}

std::string AntennaFFTProcMax::GetProfilingStats() const {
    std::ostringstream oss;
    oss << "\n═══════════════════════════════════════════════════════════\n";
    oss << "  Profiling Statistics\n";
    oss << "═══════════════════════════════════════════════════════════\n";
    oss << "Upload Time:        " << std::fixed << std::setprecision(3) 
        << last_profiling_.upload_time_ms << " ms\n";
    oss << "Pre-Callback Time:  " << last_profiling_.pre_callback_time_ms << " ms\n";
    oss << "FFT Time:           " << last_profiling_.fft_time_ms << " ms\n";
    oss << "Post-Callback Time: " << last_profiling_.post_callback_time_ms << " ms\n";
    oss << "Reduction Time:     " << last_profiling_.reduction_time_ms << " ms\n";
    oss << "Download Time:      " << last_profiling_.download_time_ms << " ms\n";
    oss << "Total Time:         " << last_profiling_.total_time_ms << " ms\n";
    return oss.str();
}

const FFTProfilingResults& AntennaFFTProcMax::GetLastProfilingResults() const {
    // Конвертируем внутреннюю структуру ProfilingData в FFTProfilingResults
    static FFTProfilingResults result;
    result.total_time_ms = last_profiling_.total_time_ms;
    result.upload_time_ms = last_profiling_.upload_time_ms;
    result.pre_callback_time_ms = last_profiling_.pre_callback_time_ms;
    result.fft_time_ms = last_profiling_.fft_time_ms;
    result.post_callback_time_ms = last_profiling_.post_callback_time_ms;
    result.reduction_time_ms = last_profiling_.reduction_time_ms;
    result.download_time_ms = last_profiling_.download_time_ms;
    return result;
}

void AntennaFFTProcMax::UpdateParams(const AntennaFFTParams& params) {
    if (!params.IsValid()) {
        throw std::invalid_argument("Invalid parameters");
    }
    
    bool need_rebuild = (params_.beam_count != params.beam_count) ||
                        (params_.count_points != params.count_points) ||
                        (params_.out_count_points_fft != params.out_count_points_fft) ||
                        (params_.max_peaks_count != params.max_peaks_count);
    
    params_ = params;
    nFFT_ = CalculateNFFT(params_.count_points);
    
    if (need_rebuild) {
        ReleaseFFTPlan();
        // Буферы будут пересозданы при следующем вызове Process()
        buffer_fft_input_.reset();
        buffer_fft_output_.reset();
        buffer_magnitude_.reset();
        buffer_maxima_.reset();
    }
}

} // namespace antenna_fft

