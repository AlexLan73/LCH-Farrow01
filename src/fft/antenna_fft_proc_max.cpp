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

namespace antenna_fft {

// Структура для хранения максимумов (совпадает с kernel структурой)
struct MaxValue {
    cl_uint index;
    float magnitude;
    float phase;
};

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
    
    // Создать reduction kernel для поиска максимумов
    CreateMaxReductionKernel();
    
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
    auto start_total = std::chrono::high_resolution_clock::now();
    
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
        buffer_maxima_ = engine_->CreateBuffer(params_.beam_count * params_.max_peaks_count * sizeof(MaxValue),
                                                gpu::MemoryType::GPU_READ_WRITE);
    }
    
    // Профилирование: загрузка данных
    // Скопировать входные данные из input_signal в pre_callback_userdata буфер
    size_t pre_params_size = sizeof(cl_uint) * 4; // beam_count, count_points, nFFT, padding
    size_t pre_input_size = params_.beam_count * params_.count_points * sizeof(std::complex<float>);

    cl_event upload_event = nullptr;
    cl_int err = clEnqueueCopyBuffer(
        queue_,
        input_signal,                    // Источник: входной сигнал
        pre_callback_userdata_,          // Назначение: userdata буфер
        0,                               // src_offset
        pre_params_size,                 // dst_offset (после параметров)
        pre_input_size,                  // size
        0, nullptr, &upload_event
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to copy input data to pre_callback_userdata: " + std::to_string(err));
    }

    // Профилирование загрузки
    last_profiling_.upload_time_ms = ProfileEvent(upload_event, "Data Upload");
    clReleaseEvent(upload_event);
    
    // Выполнить FFT
    cl_mem fft_input = buffer_fft_input_->Get();
    cl_mem fft_output = buffer_fft_output_->Get();
    cl_event fft_event = nullptr;
    clfftStatus status = clfftEnqueueTransform(
        plan_handle_,
        CLFFT_FORWARD,
        1,
        &queue_,
        0,
        nullptr,
        &fft_event,
        &fft_input,
        &fft_output,
        nullptr
    );
    
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftEnqueueTransform failed with status: " + std::to_string(status));
    }
    
    // Профилирование FFT
    last_profiling_.fft_time_ms = ProfileEvent(fft_event, "FFT Execution");
    
    // Ждать завершения FFT
    clWaitForEvents(1, &fft_event);
    clReleaseEvent(fft_event);

    // Скопировать magnitude из post_callback_userdata_ в buffer_magnitude_
    size_t post_params_size = sizeof(cl_uint) * 4; // beam_count, nFFT, out_count_points_fft, padding
    size_t post_magnitude_size = params_.beam_count * params_.out_count_points_fft * sizeof(float);

    cl_event copy_event = nullptr;
    err = clEnqueueCopyBuffer(
        queue_,
        post_callback_userdata_,        // Источник: userdata с magnitude
        buffer_magnitude_->Get(),       // Назначение: buffer_magnitude_
        post_params_size,               // src_offset (после параметров)
        0,                              // dst_offset
        post_magnitude_size,            // size
        0, nullptr, &copy_event
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to copy magnitude from post_callback_userdata: " + std::to_string(err));
    }
    clWaitForEvents(1, &copy_event);
    clReleaseEvent(copy_event);

    // Поиск максимумов для каждого луча
    AntennaFFTResult result(params_.beam_count, nFFT_, params_.task_id, params_.module_name);
    result.results.reserve(params_.beam_count);
    
    auto reduction_start = std::chrono::high_resolution_clock::now();
    
    for (size_t beam_idx = 0; beam_idx < params_.beam_count; ++beam_idx) {
        FFTResult beam_result(params_.out_count_points_fft, params_.task_id, params_.module_name);
        beam_result.max_values = FindMaximaOnGPU(buffer_fft_output_->Get(), beam_idx);
        result.results.push_back(std::move(beam_result));
    }
    
    auto reduction_end = std::chrono::high_resolution_clock::now();
    last_profiling_.reduction_time_ms = 
        std::chrono::duration<double, std::milli>(reduction_end - reduction_start).count();
    
    // Общее время
    auto end_total = std::chrono::high_resolution_clock::now();
    last_profiling_.total_time_ms = 
        std::chrono::duration<double, std::milli>(end_total - start_total).count();
    
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
    PlanCacheKey key{params_.beam_count, nFFT_, params_.out_count_points_fft, params_.max_peaks_count};
    
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
    struct PostCallbackUserData {
        cl_uint beam_count;
        cl_uint nFFT;
        cl_uint out_count_points_fft;
        cl_uint padding;
    };

    PostCallbackUserData post_cb_params = {
        static_cast<cl_uint>(params_.beam_count),
        static_cast<cl_uint>(nFFT_),
        static_cast<cl_uint>(params_.out_count_points_fft),
        0
    };

    size_t post_params_size = sizeof(PostCallbackUserData);
    size_t post_magnitude_size = params_.beam_count * params_.out_count_points_fft * sizeof(float);
    size_t post_userdata_size = post_params_size + post_magnitude_size;

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

std::string AntennaFFTProcMax::GetPostCallbackSource() const {
    // Post-callback: fftshift для двух диапазонов + вычисление magnitude/phase
    return R"(
        typedef struct {
            uint beam_count;
            uint nFFT;
            uint out_count_points_fft;
            uint padding;
        } PostCallbackUserData;

        void processFFTPost(__global void* output, uint outoffset, __global void* userdata, float2 fftoutput) {
            __global PostCallbackUserData* params = (__global PostCallbackUserData*)userdata;
            __global float2* fft_data = (__global float2*)output;
            __global float* magnitude_buffer = (__global float*)((__global char*)userdata + sizeof(PostCallbackUserData));

            uint beam_count = params->beam_count;
            uint nFFT = params->nFFT;
            uint out_count_points_fft = params->out_count_points_fft;

            // Вычислить индекс луча и позицию в FFT
            uint beam_idx = outoffset / nFFT;
            uint pos_in_fft = outoffset % nFFT;

            if (beam_idx >= beam_count) {
                return;
            }

            // Сохранить результат FFT
            fft_data[outoffset] = fftoutput;

            // Определить два диапазона для fftshift:
            // Диапазон 1: от [(nFFT-1)-out_count_points_fft/2] до (nFFT-1)
            // Диапазон 2: от 0 до out_count_points_fft/2
            uint range1_start = (nFFT - 1) - (out_count_points_fft / 2);
            uint range1_end = nFFT - 1;
            uint range2_start = 0;
            uint range2_end = out_count_points_fft / 2;

            // Проверить попадание в диапазоны
            bool in_range1 = (pos_in_fft >= range1_start && pos_in_fft <= range1_end);
            bool in_range2 = (pos_in_fft >= range2_start && pos_in_fft <= range2_end);

            if (in_range1 || in_range2) {
                // Вычислить magnitude
                float magnitude = length(fftoutput);

                // Вычислить индекс в выходном буфере (после fftshift)
                uint output_idx;
                if (in_range1) {
                    // Диапазон 1: смещение от начала диапазона
                    output_idx = beam_idx * out_count_points_fft + (pos_in_fft - range1_start);
                } else {
                    // Диапазон 2: смещение от начала диапазона + размер диапазона 1
                    output_idx = beam_idx * out_count_points_fft + (out_count_points_fft / 2) + (pos_in_fft - range2_start);
                }

                // Записать magnitude в userdata буфер
                if (output_idx < beam_count * out_count_points_fft) {
                    magnitude_buffer[output_idx] = magnitude;
                }
            }
        }
    )";
}


void AntennaFFTProcMax::CreateMaxReductionKernel() {
    // Reduction kernel для поиска топ-N максимумов на GPU
    std::string reduction_source = R"(
        typedef struct {
            uint index;
            float magnitude;
            float phase;
        } MaxValue;

        // Kernel для поиска топ-N максимумов через parallel reduction
        __kernel void findTopNMaxima(
            __global const float* magnitude_buffer,  // Буфер с magnitude после post-callback
            __global MaxValue* output_maxima,         // Выходной буфер с топ-N максимумами
            uint beam_idx,                            // Индекс луча
            uint out_count_points_fft,                // Количество точек в выходном диапазоне
            uint max_peaks_count                      // Количество максимумов для поиска
        ) {
            uint gid = get_global_id(0);
            uint total_points = out_count_points_fft;

            if (gid >= total_points) return;

            // Вычислить индекс в magnitude_buffer
            uint idx = beam_idx * out_count_points_fft + gid;
            float mag = magnitude_buffer[idx];

            // Используем local memory для reduction
            __local MaxValue local_maxima[256]; // Достаточно для большинства случаев

            uint lid = get_local_id(0);
            uint local_size = get_local_size(0);

            // Инициализировать local максимумы
            if (lid < max_peaks_count) {
                local_maxima[lid].index = UINT_MAX;
                local_maxima[lid].magnitude = -1.0f;
                local_maxima[lid].phase = 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Простой алгоритм: каждый поток проверяет, попадает ли его значение в топ-N
            // Более эффективный вариант - использовать bitonic sort или heap, но для простоты используем этот
            if (mag > 0.0f) {
                // Найти позицию для вставки
                for (uint i = 0; i < max_peaks_count; ++i) {
                    if (lid < local_size && mag > local_maxima[i].magnitude) {
                        // Сдвинуть остальные элементы
                        for (uint j = max_peaks_count - 1; j > i; --j) {
                            local_maxima[j] = local_maxima[j - 1];
                        }
                        // Вставить новое значение
                        local_maxima[i].index = gid;
                        local_maxima[i].magnitude = mag;
                        // Phase нужно будет вычислить отдельно из FFT данных
                        local_maxima[i].phase = 0.0f;
                        break;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Записать результат в глобальную память (только первый поток группы)
            if (lid == 0) {
                for (uint i = 0; i < max_peaks_count && i < local_size; ++i) {
                    uint out_idx = beam_idx * max_peaks_count + i;
                    output_maxima[out_idx] = local_maxima[i];
                }
            }
        }

        // Упрощенный kernel для поиска максимумов (более эффективный вариант)
        __kernel void findMaximaSimple(
            __global const float* magnitude_buffer,
            __global const float2* fft_data,          // FFT данные для вычисления phase
            __global MaxValue* output_maxima,
            uint beam_idx,
            uint nFFT,
            uint out_count_points_fft,
            uint max_peaks_count
        ) {
            uint gid = get_global_id(0);
            uint total_points = out_count_points_fft;

            if (gid >= total_points) return;

            // Вычислить индекс в magnitude_buffer
            uint mag_idx = beam_idx * out_count_points_fft + gid;
            float mag = magnitude_buffer[mag_idx];

            if (mag <= 0.0f) return;

            // Используем атомарные операции для обновления топ-N максимумов
            // Это упрощенная версия, для лучшей производительности нужен более сложный алгоритм
            __global MaxValue* beam_maxima = &output_maxima[beam_idx * max_peaks_count];

            // Простой алгоритм: найти минимальный элемент в топ-N и заменить если нужно
            float min_mag = beam_maxima[0].magnitude;
            uint min_idx = 0;
            for (uint i = 1; i < max_peaks_count; ++i) {
                if (beam_maxima[i].magnitude < min_mag) {
                    min_mag = beam_maxima[i].magnitude;
                    min_idx = i;
                }
            }

            // Если текущее значение больше минимального - заменить
            if (mag > min_mag) {
                // Вычислить phase из FFT данных
                // Нужно найти соответствующий индекс в FFT данных с учетом fftshift
                // gid - это индекс в выходном буфере после fftshift (0..out_count_points_fft-1)
                // Нужно найти исходный индекс в FFT (0..nFFT-1)

                uint fft_pos;
                if (gid < out_count_points_fft / 2) {
                    // Вторая половина диапазона fftshift -> начало FFT
                    fft_pos = gid;
                } else {
                    // Первая половина диапазона fftshift -> конец FFT
                    fft_pos = (nFFT - 1) - (out_count_points_fft / 2) + (gid - out_count_points_fft / 2);
                }

                uint fft_idx = beam_idx * nFFT + fft_pos;
                if (fft_idx < beam_idx * nFFT + nFFT) {
                    float2 fft_val = fft_data[fft_idx];
                    float phase = atan2(fft_val.y, fft_val.x) * 57.2957795f; // Радианы в градусы

                    beam_maxima[min_idx].index = gid;
                    beam_maxima[min_idx].magnitude = mag;
                    beam_maxima[min_idx].phase = phase;
                }
            }
        }
    )";

    reduction_program_ = engine_->LoadProgram(reduction_source);
    reduction_kernel_ = engine_->GetKernel(reduction_program_, "findMaximaSimple");
}

std::vector<FFTMaxResult> AntennaFFTProcMax::FindMaximaOnGPU(cl_mem fft_output, size_t beam_idx) {
    // ВАЖНО: Поиск максимумов выполняется ТОЛЬКО на GPU через reduction kernel
    if (!reduction_kernel_ || !buffer_magnitude_ || !buffer_fft_output_) {
        throw std::runtime_error("Reduction kernel or buffers not initialized");
    }

    // Инициализировать выходной буфер максимумов нулями
    
    std::vector<MaxValue> init_maxima(params_.max_peaks_count);
    for (size_t i = 0; i < params_.max_peaks_count; ++i) {
        init_maxima[i].index = UINT_MAX;
        init_maxima[i].magnitude = -1.0f;
        init_maxima[i].phase = 0.0f;
    }
    
    size_t offset = beam_idx * params_.max_peaks_count * sizeof(MaxValue);
    cl_int err = clEnqueueWriteBuffer(
        queue_,
        buffer_maxima_->Get(),
        CL_TRUE,
        offset,
        params_.max_peaks_count * sizeof(MaxValue),
        init_maxima.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to initialize maxima buffer: " + std::to_string(err));
    }
    
    // Установить аргументы kernel
    cl_mem magnitude_mem = buffer_magnitude_->Get();
    cl_mem fft_output_mem = buffer_fft_output_->Get();
    cl_mem maxima_mem = buffer_maxima_->Get();
    
    err = clSetKernelArg(reduction_kernel_, 0, sizeof(cl_mem), &magnitude_mem);
    if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 0 failed: " + std::to_string(err));
    
    err = clSetKernelArg(reduction_kernel_, 1, sizeof(cl_mem), &fft_output_mem);
    if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 1 failed: " + std::to_string(err));
    
    err = clSetKernelArg(reduction_kernel_, 2, sizeof(cl_mem), &maxima_mem);
    if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 2 failed: " + std::to_string(err));
    
    cl_uint beam_idx_uint = static_cast<cl_uint>(beam_idx);
    err = clSetKernelArg(reduction_kernel_, 3, sizeof(cl_uint), &beam_idx_uint);
    if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 3 failed: " + std::to_string(err));
    
    cl_uint nFFT_uint = static_cast<cl_uint>(nFFT_);
    err = clSetKernelArg(reduction_kernel_, 4, sizeof(cl_uint), &nFFT_uint);
    if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 4 failed: " + std::to_string(err));
    
    cl_uint out_count_uint = static_cast<cl_uint>(params_.out_count_points_fft);
    err = clSetKernelArg(reduction_kernel_, 5, sizeof(cl_uint), &out_count_uint);
    if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 5 failed: " + std::to_string(err));
    
    cl_uint max_peaks_uint = static_cast<cl_uint>(params_.max_peaks_count);
    err = clSetKernelArg(reduction_kernel_, 6, sizeof(cl_uint), &max_peaks_uint);
    if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg 6 failed: " + std::to_string(err));
    
    // Выполнить kernel
    size_t global_work_size = params_.out_count_points_fft;
    size_t local_work_size = 256; // Оптимальный размер для большинства GPU
    
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
        throw std::runtime_error("clEnqueueNDRangeKernel failed: " + std::to_string(err));
    }
    
    // Ждать завершения
    clWaitForEvents(1, &reduction_event);
    clReleaseEvent(reduction_event);
    
    // Прочитать результаты с GPU
    std::vector<MaxValue> maxima_result(params_.max_peaks_count);
    err = clEnqueueReadBuffer(
        queue_,
        buffer_maxima_->Get(),
        CL_TRUE,
        offset,
        params_.max_peaks_count * sizeof(MaxValue),
        maxima_result.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read maxima from GPU: " + std::to_string(err));
    }
    
    // Конвертировать в FFTMaxResult
    std::vector<FFTMaxResult> result;
    result.reserve(params_.max_peaks_count);
    
    for (size_t i = 0; i < params_.max_peaks_count; ++i) {
        if (maxima_result[i].index != UINT_MAX && maxima_result[i].magnitude > 0.0f) {
            FFTMaxResult max_result;
            max_result.index_point = maxima_result[i].index;
            max_result.amplitude = maxima_result[i].magnitude;
            max_result.phase = maxima_result[i].phase;
            result.push_back(max_result);
        }
    }
    
    return result;
}

double AntennaFFTProcMax::ProfileEvent(cl_event event, const std::string& operation_name) {
    if (!event) return 0.0;
    
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

void AntennaFFTProcMax::SaveResultsToFile(const AntennaFFTResult& result, const std::string& filepath) const {
    std::string full_path = filepath;
    if (filepath.find("/") != 0 && filepath.find(":\\") == std::string::npos) {
        // Относительный путь - добавить Reports/
        full_path = "Reports/" + filepath;
    }
    
    // Создать директорию если нужно
    size_t last_slash = full_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        std::string dir = full_path.substr(0, last_slash);
        // TODO: Создать директорию если не существует (можно использовать filesystem или system)
    }
    
    std::ofstream file(full_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + full_path);
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
    file << "# AntennaFFTProcMax Results\n\n";
    file << "**Generated:** " << time_str << "\n\n";
    file << "**Task ID:** " << result.task_id << "\n";
    file << "**Module:** " << result.module_name << "\n";
    file << "**Total Beams:** " << result.total_beams << "\n";
    file << "**nFFT:** " << result.nFFT << "\n\n";
    
    file << "## Results by Beam\n\n";
    file << "| Beam | Index | Amplitude | Phase (deg) |\n";
    file << "|------|-------|-----------|-------------|\n";
    
    for (size_t i = 0; i < result.results.size(); ++i) {
        const auto& beam_result = result.results[i];
        if (beam_result.max_values.empty()) {
            file << "| " << i << " | - | - | - |\n";
        } else {
            for (size_t j = 0; j < beam_result.max_values.size(); ++j) {
                const auto& max_val = beam_result.max_values[j];
                file << "| " << i << " | " << max_val.index_point 
                     << " | " << std::fixed << std::setprecision(6) << max_val.amplitude
                     << " | " << std::setprecision(2) << max_val.phase << " |\n";
            }
        }
    }
    
    // Записать JSON
    file << "\n## JSON Format\n\n";
    file << "```json\n";
    file << "{\n";
    file << "  \"task_id\": \"" << result.task_id << "\",\n";
    file << "  \"module_name\": \"" << result.module_name << "\",\n";
    file << "  \"total_beams\": " << result.total_beams << ",\n";
    file << "  \"nFFT\": " << result.nFFT << ",\n";
    file << "  \"results\": [\n";
    
    for (size_t i = 0; i < result.results.size(); ++i) {
        const auto& beam_result = result.results[i];
        file << "    {\n";
        file << "      \"beam_index\": " << i << ",\n";
        file << "      \"v_fft\": " << beam_result.v_fft << ",\n";
        file << "      \"max_values\": [\n";
        
        for (size_t j = 0; j < beam_result.max_values.size(); ++j) {
            const auto& max_val = beam_result.max_values[j];
            file << "        {\n";
            file << "          \"index_point\": " << max_val.index_point << ",\n";
            file << "          \"amplitude\": " << std::fixed << std::setprecision(6) << max_val.amplitude << ",\n";
            file << "          \"phase\": " << std::setprecision(2) << max_val.phase << "\n";
            file << "        }";
            if (j < beam_result.max_values.size() - 1) file << ",";
            file << "\n";
        }
        
        file << "      ]\n";
        file << "    }";
        if (i < result.results.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    file << "```\n";
    
    file.close();
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

