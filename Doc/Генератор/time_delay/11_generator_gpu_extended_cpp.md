# 💻 11_generator_gpu_extended_cpp.md

## РЕАЛИЗАЦИЯ И KERNEL

### ПОЛНЫЙ КОД KERNEL'А (kernel_lfm_combined):

```opencl
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
    if (ray_id >= num_beams || sample_id >= num_samples) return;
    
    // Получить задержки
    float delay_degrees = combined[ray_id].delay_degrees;
    float delay_time_ns = combined[ray_id].delay_time_ns;
    
    // Конвертировать градусы → время
    float f_center = (f_start + f_stop) / 2.0f;
    float wavelength = speed_of_light / f_center;
    float delay_rad = delay_degrees * 3.14159265f / 180.0f;
    float delay_angle_sec = delay_rad * wavelength / speed_of_light;
    
    // Конвертировать нс → сек
    float delay_time_sec = delay_time_ns * 1e-9f;
    
    // Полная задержка
    float total_delay_sec = delay_angle_sec + delay_time_sec;
    float total_delay_samples = total_delay_sec * sample_rate;
    
    int delayed_sample_int = (int)sample_id - (int)total_delay_samples;
    
    float real, imag;
    if (delayed_sample_int < 0) {
        real = 0.0f;
        imag = 0.0f;
    } else {
        float t = (float)delayed_sample_int / sample_rate;
        float chirp_rate = (f_stop - f_start) / duration;
        float phase = 2.0f * 3.14159265f * (
            f_start * t + 0.5f * chirp_rate * t * t
        );
        real = cos(phase);
        imag = sin(phase);
    }
    
    uint out_idx = ray_id * num_samples + sample_id;
    output[out_idx] = (float2)(real, imag);
}
```

### РЕАЛИЗАЦИЯ signal_combined_delays():

```cpp
cl_mem GeneratorGPU::signal_combined_delays(
    const CombinedDelayParam* combined_delays,
    size_t num_delay_params) {
    
    if (!engine_) {
        throw std::runtime_error("Engine not initialized");
    }
    if (!kernel_lfm_combined_) {
        throw std::runtime_error("kernel_lfm_combined not loaded");
    }
    if (!combined_delays) {
        throw std::invalid_argument("combined_delays is null");
    }
    if (num_delay_params != num_beams_) {
        throw std::invalid_argument("num_delay_params must equal num_beams");
    }
    
    // Создать GPU буфер параметров
    auto combined_gpu_buffer = engine_->CreateBufferWithData(
        std::vector<CombinedDelayParam>(
            combined_delays,
            combined_delays + num_delay_params
        ),
        gpu::MemoryType::GPU_READ_ONLY
    );
    
    // Создать GPU буфер результатов
    auto output = engine_->CreateBuffer(total_size_, gpu::MemoryType::GPU_WRITE_ONLY);
    
    // Выполнить kernel
    ExecuteKernel(kernel_lfm_combined_, output->Get(), combined_gpu_buffer->Get());
    
    // Сохранить в кэш
    buffer_signal_combined_ = std::move(output);
    
    return buffer_signal_combined_->Get();
}
```

### ИНИЦИАЛИЗАЦИЯ (в конструкторе):

```cpp
kernel_lfm_combined_(nullptr),
buffer_signal_combined_(nullptr)
```

### ЗАГРУЗКА KERNEL (в LoadKernels()):

```cpp
kernel_lfm_combined_ = engine_->GetKernel(kernel_program_, "kernel_lfm_combined");
if (!kernel_lfm_combined_) {
    throw std::runtime_error("Failed to create kernel_lfm_combined");
}
```

### ДОБАВИТЬ В GetKernelSource():

Добавить kernel код выше в R"(...)" строку.
