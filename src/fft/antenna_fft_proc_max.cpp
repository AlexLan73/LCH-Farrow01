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
#include <climits>  // Для UINT_MAX

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
       reduction_kernel_(nullptr) {
    
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

    if (pre_callback_userdata_) {
        clReleaseMemObject(pre_callback_userdata_);
    }
    if (post_callback_userdata_) {
        clReleaseMemObject(post_callback_userdata_);
    }
    if (reduction_kernel_) {
        clReleaseKernel(reduction_kernel_);
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
       pre_callback_userdata_(other.pre_callback_userdata_),
       post_callback_userdata_(other.post_callback_userdata_),
       reduction_program_(std::move(other.reduction_program_)),
       reduction_kernel_(other.reduction_kernel_),
       last_profiling_(other.last_profiling_) {

    other.plan_handle_ = 0;
    other.plan_created_ = false;
    other.pre_callback_userdata_ = nullptr;
    other.post_callback_userdata_ = nullptr;
    other.reduction_kernel_ = nullptr;
}

AntennaFFTProcMax& AntennaFFTProcMax::operator=(AntennaFFTProcMax&& other) noexcept {
    if (this != &other) {
        ReleaseFFTPlan();

        if (pre_callback_userdata_) clReleaseMemObject(pre_callback_userdata_);
        if (post_callback_userdata_) clReleaseMemObject(post_callback_userdata_);
        if (reduction_kernel_) clReleaseKernel(reduction_kernel_);

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
        pre_callback_userdata_ = other.pre_callback_userdata_;
        post_callback_userdata_ = other.post_callback_userdata_;
        reduction_program_ = std::move(other.reduction_program_);
        reduction_kernel_ = other.reduction_kernel_;
        last_profiling_ = other.last_profiling_;

        other.plan_handle_ = 0;
        other.plan_created_ = false;
        other.pre_callback_userdata_ = nullptr;
        other.post_callback_userdata_ = nullptr;
        other.reduction_kernel_ = nullptr;
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
// Основной метод обработки
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTResult AntennaFFTProcMax::Process(cl_mem input_signal) {
    
    // Создать или переиспользовать план FFT
    CreateOrReuseFFTPlan();
    
    // Создать буферы если нужно
    size_t total_fft_size = params_.beam_count * nFFT_;
    
    if (!buffer_fft_input_) {
        buffer_fft_input_ = engine_->CreateBuffer(total_fft_size, gpu::MemoryType::GPU_READ_WRITE);
    }
    if (!buffer_fft_output_) {
        buffer_fft_output_ = engine_->CreateBuffer(total_fft_size, gpu::MemoryType::GPU_READ_WRITE);
    }
    if (!buffer_magnitude_) {
        buffer_magnitude_ = engine_->CreateBuffer(params_.beam_count * params_.out_count_points_fft, 
                                                  gpu::MemoryType::GPU_READ_WRITE);
    }
    if (!buffer_maxima_) {
        const size_t maxima_bytes = params_.beam_count * params_.max_peaks_count * sizeof(MaxValue);
        const size_t maxima_elements = (maxima_bytes + sizeof(std::complex<float>) - 1) / sizeof(std::complex<float>);
        buffer_maxima_ = engine_->CreateBuffer(maxima_elements, gpu::MemoryType::GPU_READ_WRITE);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // EVENT-BASED PIPELINE: Минимизация синхронизации между этапами
    // Upload → FFT → Reduction → Read (один wait в конце!)
    // ═══════════════════════════════════════════════════════════════════════════
    
    cl_int err;
    cl_event upload_event = nullptr;
    cl_event fft_event = nullptr;
    cl_event reduction_event = nullptr;
    cl_event read_event = nullptr;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ЭТАП 1: Upload (async, возвращает event)
    // ═══════════════════════════════════════════════════════════════════════════
    size_t pre_params_size = sizeof(cl_uint) * 4; // beam_count, count_points, nFFT, padding
    size_t pre_input_size = params_.beam_count * params_.count_points * sizeof(std::complex<float>);

    err = clEnqueueCopyBuffer(
        queue_,
        input_signal,                    // Источник: входной сигнал
        pre_callback_userdata_,          // Назначение: userdata буфер
        0,                               // src_offset
        pre_params_size,                 // dst_offset (после параметров)
        pre_input_size,                  // size
        0, nullptr, &upload_event        // ← Возвращает event
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to copy input data to pre_callback_userdata: " + std::to_string(err));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ЭТАП 2: FFT (зависит от upload_event)
    // ═══════════════════════════════════════════════════════════════════════════
    cl_mem fft_input = buffer_fft_input_->Get();
    cl_mem fft_output = buffer_fft_output_->Get();
    
    clfftStatus status = clfftEnqueueTransform(
        plan_handle_,
        CLFFT_FORWARD,
        1,
        &queue_,
        1,                    // ← num_events_in_wait_list
        &upload_event,        // ← Зависимость от upload!
        &fft_event,
        &fft_input,
        &fft_output,
        nullptr
    );
    
    if (status != CLFFT_SUCCESS) {
        clReleaseEvent(upload_event);
        throw std::runtime_error("clfftEnqueueTransform failed with status: " + std::to_string(status));
    }

    last_profiling_.post_callback_time_ms = 0.0;

    // ═══════════════════════════════════════════════════════════════════════════
    // ЭТАП 3: Reduction + Read (зависит от fft_event, БЕЗ промежуточного wait!)
    // ═══════════════════════════════════════════════════════════════════════════
    auto all_maxima = FindMaximaAllBeamsOnGPU(fft_event, &reduction_event, &read_event);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ЕДИНСТВЕННОЕ ОЖИДАНИЕ - В САМОМ КОНЦЕ PIPELINE!
    // ═══════════════════════════════════════════════════════════════════════════
    if (read_event) {
        clWaitForEvents(1, &read_event);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Профилирование (после завершения всех операций)
    // ═══════════════════════════════════════════════════════════════════════════
    last_profiling_.upload_time_ms = ProfileEvent(upload_event, "Data Upload");
    last_profiling_.fft_time_ms = ProfileEvent(fft_event, "FFT Execution");
    last_profiling_.reduction_time_ms = ProfileEvent(reduction_event, "Reduction + Phase");
    
    // Освободить events
    if (upload_event) clReleaseEvent(upload_event);
    if (fft_event) clReleaseEvent(fft_event);
    if (reduction_event) clReleaseEvent(reduction_event);
    if (read_event) clReleaseEvent(read_event);

    // ═══════════════════════════════════════════════════════════════════════════
    // Формирование результатов (данные уже прочитаны)
    // ═══════════════════════════════════════════════════════════════════════════
    AntennaFFTResult result(params_.beam_count, nFFT_, params_.task_id, params_.module_name);
    result.results.reserve(params_.beam_count);

    for (size_t beam_idx = 0; beam_idx < all_maxima.size(); ++beam_idx) {
        FFTResult beam_result(params_.out_count_points_fft, params_.task_id, params_.module_name);
        beam_result.max_values = std::move(all_maxima[beam_idx]);
        result.results.push_back(std::move(beam_result));
    }
    
    // Общее время (сумма GPU этапов)
    last_profiling_.total_time_ms =
        last_profiling_.upload_time_ms +
        last_profiling_.fft_time_ms +
        last_profiling_.post_callback_time_ms +
        last_profiling_.reduction_time_ms;
    
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

// KEY CHANGES:
// 1. GetPostCallbackSource() - proper filtering to search_range only
// 2. CreateMaxReductionKernel() - complete phase calculation
// 3. FindMaximaAllBeamsOnGPU() - clear parameter passing
// 4. Struct PostCallbackUserData - renamed field for clarity
//
// USAGE: Copy these methods into your antenna_fft_proc_max.cpp file
// replacing the existing implementations
//

// ════════════════════════════════════════════════════════════════════════════
// METHOD 1: GetPostCallbackSource()
// ════════════════════════════════════════════════════════════════════════════

std::string AntennaFFTProcMax::GetPostCallbackSource() const {
    // User requirement:
    // - search_range parameter (e.g., 60)
    // - Search ONLY in first search_range/2 points AND last search_range/2 points
    // - Example: search_range=60 → search [0..29] and [nFFT-30..nFFT-1]
    
    return R"(
typedef struct {
    uint beam_count;
    uint nFFT;
    uint search_range;              // CHANGED: was out_count_points_fft
    uint max_peaks_count;
} PostCallbackUserData;

void processFFTPost(__global void* output, uint outoffset, __global void* userdata, float2 fftoutput) {
    __global PostCallbackUserData* params = (__global PostCallbackUserData*)userdata;
    
    uint beam_count = params->beam_count;
    uint nFFT = params->nFFT;
    uint search_range = params->search_range;        // e.g., 60
    uint half_search = search_range / 2;            // e.g., 30
    
    // Calculate beam index and position in FFT
    uint beam_idx = outoffset / nFFT;
    uint pos_in_fft = outoffset % nFFT;
    
    if (beam_idx >= beam_count) {
        return;
    }
    
    // ✅ FILTER: Check if position is in interesting range
    // Range 1: [0, half_search) - first half_search points
    // Range 2: [nFFT - half_search, nFFT) - last half_search points
    bool in_range1 = (pos_in_fft < half_search);
    bool in_range2 = (pos_in_fft >= nFFT - half_search);
    
    if (!in_range1 && !in_range2) {
        return;  // ✅ Fast return for 99.9% of threads!
    }
    
    // ✅ Calculate index in output buffer (0..search_range-1)
    uint output_idx;
    if (in_range1) {
        // First half_search points go to beginning
        output_idx = pos_in_fft;
    } else {
        // Last half_search points go after first half
        output_idx = half_search + (pos_in_fft - (nFFT - half_search));
    }
    
    // Layout userdata: params | complex_buffer | magnitude_buffer
    uint params_size = 16;  // sizeof(PostCallbackUserData)
    uint complex_offset = params_size;
    uint magnitude_offset = complex_offset + (beam_count * search_range * 8);  // 8 = sizeof(float2)
    
    __global float2* complex_buffer = (__global float2*)((__global char*)userdata + complex_offset);
    __global float* magnitude_buffer = (__global float*)((__global char*)userdata + magnitude_offset);
    
    // Calculate global index in buffer
    uint base_idx = beam_idx * search_range + output_idx;
    
    // ✅ Write complex spectrum (for future phase calculation)
    complex_buffer[base_idx] = fftoutput;
    
    // ✅ Write magnitude (for finding maxima)
    magnitude_buffer[base_idx] = length(fftoutput);
}
)";
}

// ════════════════════════════════════════════════════════════════════════════
// METHOD 2: CreateMaxReductionKernel()
// ════════════════════════════════════════════════════════════════════════════

void AntennaFFTProcMax::CreateMaxReductionKernel() {
    std::string reduction_kernel_source = R"(
typedef struct {
    uint index;
    float magnitude;
    float phase;
    uint pad;
} MaxValue;

// Find top-N maxima and calculate phase
// Kernel: one work-group per beam, parallel reduction on search_range points
__kernel void findMaximaAndPhase(
    __global const float2* complex_buffer,      // Complex spectrum (search_range points)
    __global const float* magnitude_buffer,     // Magnitude (search_range points)
    __global MaxValue* maxima_buffer,           // Output buffer for top-N
    uint beam_count,
    uint search_range,                          // Total points to search (e.g., 60)
    uint max_peaks_count                        // N = 5
) {
    uint beam_idx = get_group_id(0);
    uint tid = get_local_id(0);
    uint local_size = get_local_size(0);
    
    if (beam_idx >= beam_count) return;
    
    // Local memory for top-N maxima
    __local MaxValue local_max[8];
    __local float local_mag[256];
    __local uint local_idx[256];
    
    // ========================================================================
    // PHASE 1: Initialize top-N
    // ========================================================================
    if (tid < max_peaks_count) {
        local_max[tid].index = UINT_MAX;
        local_max[tid].magnitude = -1.0f;
        local_max[tid].phase = 0.0f;
        local_max[tid].pad = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // ========================================================================
    // PHASE 2: Load magnitude into local memory
    // ========================================================================
    uint base_offset = beam_idx * search_range;
    
    for (uint i = tid; i < search_range; i += local_size) {
        local_mag[i] = magnitude_buffer[base_offset + i];
        local_idx[i] = i;  // Original index in search_range array
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // ========================================================================
    // PHASE 3: Find top-N (first thread only)
    // ========================================================================
    if (tid == 0) {
        for (uint k = 0; k < max_peaks_count; ++k) {
            float max_mag = -1.0f;
            uint max_idx = UINT_MAX;
            
            // Find maximum among remaining elements
            for (uint i = 0; i < search_range; ++i) {
                if (local_mag[i] > max_mag) {
                    max_mag = local_mag[i];
                    max_idx = local_idx[i];
                }
            }
            
            if (max_idx != UINT_MAX && max_mag > 0.0f) {
                // Calculate phase in degrees
                float2 cval = complex_buffer[base_offset + max_idx];
                float phase_rad = atan2(cval.y, cval.x);          // radians
                float phase_deg = phase_rad * 57.29577951f;       // 180/π
                
                // Normalize phase to [-180, 180]
                if (phase_deg > 180.0f) phase_deg -= 360.0f;
                if (phase_deg < -180.0f) phase_deg += 360.0f;
                
                local_max[k].index = max_idx;
                local_max[k].magnitude = max_mag;
                local_max[k].phase = phase_deg;
                
                // Mark as used
                local_mag[max_idx] = -1.0f;
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // ========================================================================
    // PHASE 4: Write results to global memory
    // ========================================================================
    if (tid < max_peaks_count) {
        uint out_idx = beam_idx * max_peaks_count + tid;
        maxima_buffer[out_idx] = local_max[tid];
    }
}
)";
    
    reduction_program_ = engine_->LoadProgram(reduction_kernel_source);
    reduction_kernel_ = engine_->GetKernel(reduction_program_, "findMaximaAndPhase");
}

// ════════════════════════════════════════════════════════════════════════════
// METHOD 3: FindMaximaAllBeamsOnGPU()
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::vector<FFTMaxResult>> AntennaFFTProcMax::FindMaximaAllBeamsOnGPU(
    cl_event wait_event,
    cl_event* out_reduction_event,
    cl_event* out_read_event
) {
    if (!post_callback_userdata_) {
        throw std::runtime_error("post_callback_userdata_ is not initialized");
    }
    
    // Используй out_count_points_fft как search_range (например, 60 для 30+30)
    cl_uint search_range = params_.out_count_points_fft;
    
    // Layout: params | complex_buffer | magnitude_buffer
    size_t post_params_size = sizeof(cl_uint) * 4;
    size_t post_complex_size = params_.beam_count * search_range * sizeof(cl_float2);
    size_t post_magnitude_size = params_.beam_count * search_range * sizeof(float);
    size_t maxima_size = params_.beam_count * params_.max_peaks_count * sizeof(MaxValue);
    
    // STAGE 1: Создай kernel если не существует
    if (!reduction_kernel_) {
        CreateMaxReductionKernel();
    }
    
    // Создай буфер для результатов максимумов
    if (!buffer_maxima_) {
        const size_t maxima_elements = (maxima_size + sizeof(MaxValue) - 1) / sizeof(MaxValue);
        buffer_maxima_ = engine_->CreateBuffer(maxima_elements, gpu::MemoryType::GPU_READ_WRITE);
    }
    
    // STAGE 2: Создай sub-buffers
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
    
    // STAGE 3: Запусти reduction kernel (с зависимостью от wait_event!)
    cl_uint beam_count = static_cast<cl_uint>(params_.beam_count);
    cl_uint search_range_cl = static_cast<cl_uint>(search_range);
    cl_uint max_peaks_count = static_cast<cl_uint>(params_.max_peaks_count);
    
    cl_mem maxima_mem = buffer_maxima_->Get();
    
    clSetKernelArg(reduction_kernel_, 0, sizeof(cl_mem), &complex_sub_buffer);
    clSetKernelArg(reduction_kernel_, 1, sizeof(cl_mem), &magnitude_sub_buffer);
    clSetKernelArg(reduction_kernel_, 2, sizeof(cl_mem), &maxima_mem);
    clSetKernelArg(reduction_kernel_, 3, sizeof(cl_uint), &beam_count);
    clSetKernelArg(reduction_kernel_, 4, sizeof(cl_uint), &search_range_cl);
    clSetKernelArg(reduction_kernel_, 5, sizeof(cl_uint), &max_peaks_count);
    
    // Один work-group на луч
    size_t global_work_size = params_.beam_count * 256;
    size_t local_work_size = 256;
    
    // Адаптируй для меньшего search_range
    if (search_range < 256) {
        local_work_size = 64;
        global_work_size = params_.beam_count * local_work_size;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // EVENT-BASED: Reduction зависит от wait_event (fft_event)
    // ═══════════════════════════════════════════════════════════════════════════
    cl_event reduction_event = nullptr;
    cl_uint num_wait_events = (wait_event != nullptr) ? 1 : 0;
    const cl_event* wait_list = (wait_event != nullptr) ? &wait_event : nullptr;
    
    err = clEnqueueNDRangeKernel(
        queue_,
        reduction_kernel_,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        num_wait_events,      // ← Зависимость от FFT!
        wait_list,            // ← wait_event (fft_event)
        &reduction_event
    );
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(complex_sub_buffer);
        clReleaseMemObject(magnitude_sub_buffer);
        throw std::runtime_error("Failed to enqueue reduction kernel: " + std::to_string(err));
    }
    
    // Вернуть reduction_event если запрошено (ownership переходит к вызывающему)
    if (out_reduction_event) {
        *out_reduction_event = reduction_event;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STAGE 4: NON-BLOCKING Read (зависит от reduction_event)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Статический буфер для результатов (переиспользуется)
    static thread_local std::vector<MaxValue> maxima_result;
    maxima_result.resize(params_.beam_count * params_.max_peaks_count);
    
    cl_event read_event = nullptr;
    err = clEnqueueReadBuffer(
        queue_,
        buffer_maxima_->Get(),
        CL_FALSE,             // ← NON-BLOCKING!
        0,
        maxima_size,
        maxima_result.data(),
        1,                    // ← Зависимость от reduction!
        &reduction_event,
        &read_event
    );
    
    if (err != CL_SUCCESS) {
        clReleaseEvent(reduction_event);
        clReleaseMemObject(complex_sub_buffer);
        clReleaseMemObject(magnitude_sub_buffer);
        throw std::runtime_error("Failed to read maxima from GPU: " + std::to_string(err));
    }
    
    // Вернуть read_event если запрошено (ownership переходит к вызывающему)
    if (out_read_event) {
        *out_read_event = read_event;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ОЖИДАНИЕ ЧТЕНИЯ (только если не возвращаем event наружу)
    // ═══════════════════════════════════════════════════════════════════════════
    if (!out_read_event) {
        // Если вызывающий не хочет event - ждём здесь
        clWaitForEvents(1, &read_event);
        clReleaseEvent(read_event);
    }
    
    if (!out_reduction_event) {
        clReleaseEvent(reduction_event);
    }
    
    clReleaseMemObject(complex_sub_buffer);
    clReleaseMemObject(magnitude_sub_buffer);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STAGE 5: Конвертируй в FFTMaxResult
    // ВАЖНО: Если out_read_event != nullptr, вызывающий ДОЛЖЕН дождаться read_event
    //        перед использованием результатов!
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
                max_result.index_point = mv.index;      // Индекс в диапазоне поиска (0..search_range-1)
                max_result.amplitude = mv.magnitude;    // Амплитуда/Величина
                max_result.phase = mv.phase;            // Фаза в градусах
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

    // ═══════════════════════════════════════════════════════════════════════════
    // Считать FFT комплексный вектор (fftshift) из post_callback_userdata_
    // ВАЖНО: Убедиться что все операции завершены перед чтением!
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<std::complex<float>> fft_data;
    if (post_callback_userdata_) {
        // Синхронизировать queue перед чтением
        clFinish(queue_);
        
        size_t post_params_size = sizeof(cl_uint) * 4;
        size_t post_complex_size = params_.beam_count * params_.out_count_points_fft * sizeof(cl_float2);
        
        // Проверить размер буфера
        size_t buffer_size = 0;
        cl_int err = clGetMemObjectInfo(post_callback_userdata_, CL_MEM_SIZE, sizeof(size_t), &buffer_size, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "⚠️  Warning: Failed to get post_callback_userdata_ size: " << err 
                      << ". Skipping FFT data read.\n";
        } else if (buffer_size < post_params_size + post_complex_size) {
            std::cerr << "⚠️  Warning: post_callback_userdata_ too small (" << buffer_size 
                      << " < " << (post_params_size + post_complex_size) 
                      << "). Skipping FFT data read.\n";
        } else {
            // Читать данные
            fft_data.resize(params_.beam_count * params_.out_count_points_fft);
            err = clEnqueueReadBuffer(
                queue_,
                post_callback_userdata_,
                CL_TRUE,  // Blocking read
                post_params_size,
                post_complex_size,
                fft_data.data(),
                0, nullptr, nullptr
            );
            if (err != CL_SUCCESS) {
                // Не бросаем исключение - просто логируем и продолжаем без FFT данных
                std::cerr << "⚠️  Warning: Failed to read complex FFT data from post_callback_userdata: " 
                          << err << " (CL error code). Continuing without FFT data in JSON.\n";
                fft_data.clear();  // Очистить чтобы не записывать пустые данные
            }
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

