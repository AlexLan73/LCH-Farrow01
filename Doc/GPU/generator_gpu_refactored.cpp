#include "generator_gpu_refactored.h"
#include <stdexcept>
#include <cmath>
#include <sstream>

namespace radar {
namespace gpu {

// ═══════════════════════════════════════════════════════════════════
// CONSTRUCTOR & DESTRUCTOR
// ═══════════════════════════════════════════════════════════════════

GeneratorGPU::GeneratorGPU(const LFMParameters& params)
    : params_(params),
      manager_(OpenCLManager::GetInstance()) {
    
    // Validate before using
    ValidateParameters();
    
    // Calculate number of samples
    num_samples_ = static_cast<uint32_t>(params_.sample_rate * params_.duration);
    
    // Allocate GPU buffers
    AllocateGPUMemory();
    
    // Compile kernels (uses Manager's cache!)
    CompileKernels();
}

GeneratorGPU::~GeneratorGPU() {
    ReleaseGPUMemory();
}

// ═══════════════════════════════════════════════════════════════════
// VALIDATION
// ═══════════════════════════════════════════════════════════════════

void GeneratorGPU::ValidateParameters() {
    if (params_.f_start < 0.0f || params_.f_stop < 0.0f) {
        throw std::invalid_argument("Frequencies must be positive");
    }
    
    if (params_.f_start >= params_.f_stop) {
        throw std::invalid_argument("f_start must be less than f_stop");
    }
    
    if (params_.sample_rate <= 0.0f) {
        throw std::invalid_argument("Sample rate must be positive");
    }
    
    if (params_.duration <= 0.0f) {
        throw std::invalid_argument("Duration must be positive");
    }
    
    if (params_.num_beams == 0) {
        throw std::invalid_argument("Number of beams must be positive");
    }
    
    // Nyquist check
    float nyquist = params_.sample_rate / 2.0f;
    if (params_.f_stop > nyquist) {
        throw std::invalid_argument("f_stop exceeds Nyquist frequency");
    }
}

// ═══════════════════════════════════════════════════════════════════
// GPU MEMORY MANAGEMENT
// ═══════════════════════════════════════════════════════════════════

void GeneratorGPU::AllocateGPUMemory() {
    cl_int err;
    
    // Calculate buffer size
    size_t buffer_size = GetMemorySizeBytes();
    
    // Allocate GPU buffers (read-write)
    gpu_signal_base_ = clCreateBuffer(
        manager_.GetContext(),
        CL_MEM_READ_WRITE,
        buffer_size,
        nullptr,
        &err
    );
    
    if (err != CL_SUCCESS || !gpu_signal_base_) {
        throw std::runtime_error("Failed to allocate GPU memory for base signal");
    }
    
    gpu_signal_delayed_ = clCreateBuffer(
        manager_.GetContext(),
        CL_MEM_READ_WRITE,
        buffer_size,
        nullptr,
        &err
    );
    
    if (err != CL_SUCCESS || !gpu_signal_delayed_) {
        clReleaseMemObject(gpu_signal_base_);
        throw std::runtime_error("Failed to allocate GPU memory for delayed signal");
    }
}

void GeneratorGPU::ReleaseGPUMemory() {
    if (kernel_lfm_basic_) {
        clReleaseKernel(kernel_lfm_basic_);
        kernel_lfm_basic_ = nullptr;
    }
    
    if (kernel_lfm_delayed_) {
        clReleaseKernel(kernel_lfm_delayed_);
        kernel_lfm_delayed_ = nullptr;
    }
    
    if (gpu_signal_base_) {
        clReleaseMemObject(gpu_signal_base_);
        gpu_signal_base_ = nullptr;
    }
    
    if (gpu_signal_delayed_) {
        clReleaseMemObject(gpu_signal_delayed_);
        gpu_signal_delayed_ = nullptr;
    }
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL COMPILATION
// ═══════════════════════════════════════════════════════════════════

void GeneratorGPU::CompileKernels() {
    cl_int err;
    
    // Get kernel source (same for all instances)
    std::string kernel_source = GetLFMKernelSource();
    
    // Get or compile program (uses Manager's cache!)
    cl_program program = manager_.GetOrCompileProgram(kernel_source);
    
    // Create kernels
    kernel_lfm_basic_ = clCreateKernel(program, "lfm_basic", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create lfm_basic kernel");
    }
    
    kernel_lfm_delayed_ = clCreateKernel(program, "lfm_delayed", &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel_lfm_basic_);
        throw std::runtime_error("Failed to create lfm_delayed kernel");
    }
}

// ═══════════════════════════════════════════════════════════════════
// SIGNAL GENERATION
// ═══════════════════════════════════════════════════════════════════

cl_mem GeneratorGPU::signal_base() {
    if (!kernel_lfm_basic_ || !gpu_signal_base_) {
        throw std::runtime_error("Generator not properly initialized");
    }
    
    cl_int err;
    
    // Set kernel arguments
    uint32_t total_samples = params_.num_beams * num_samples_;
    float chirp_rate = (params_.f_stop - params_.f_start) / params_.duration;
    
    err = clSetKernelArg(kernel_lfm_basic_, 0, sizeof(cl_mem), &gpu_signal_base_);
    err |= clSetKernelArg(kernel_lfm_basic_, 1, sizeof(uint32_t), &total_samples);
    err |= clSetKernelArg(kernel_lfm_basic_, 2, sizeof(float), &params_.f_start);
    err |= clSetKernelArg(kernel_lfm_basic_, 3, sizeof(float), &chirp_rate);
    err |= clSetKernelArg(kernel_lfm_basic_, 4, sizeof(float), &params_.sample_rate);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set kernel arguments");
    }
    
    // Execute kernel
    size_t global_size = total_samples;
    err = clEnqueueNDRangeKernel(
        manager_.GetQueue(),
        kernel_lfm_basic_,
        1, nullptr,
        &global_size, nullptr,
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue kernel");
    }
    
    // Wait for completion
    clFinish(manager_.GetQueue());
    
    return gpu_signal_base_;
}

cl_mem GeneratorGPU::signal_delayed(float delay_ms) {
    if (!kernel_lfm_delayed_ || !gpu_signal_delayed_) {
        throw std::runtime_error("Generator not properly initialized");
    }
    
    cl_int err;
    
    // Set kernel arguments
    uint32_t total_samples = params_.num_beams * num_samples_;
    float chirp_rate = (params_.f_stop - params_.f_start) / params_.duration;
    float delay_samples = (delay_ms / 1000.0f) * params_.sample_rate;
    
    err = clSetKernelArg(kernel_lfm_delayed_, 0, sizeof(cl_mem), &gpu_signal_delayed_);
    err |= clSetKernelArg(kernel_lfm_delayed_, 1, sizeof(uint32_t), &total_samples);
    err |= clSetKernelArg(kernel_lfm_delayed_, 2, sizeof(float), &params_.f_start);
    err |= clSetKernelArg(kernel_lfm_delayed_, 3, sizeof(float), &chirp_rate);
    err |= clSetKernelArg(kernel_lfm_delayed_, 4, sizeof(float), &params_.sample_rate);
    err |= clSetKernelArg(kernel_lfm_delayed_, 5, sizeof(float), &delay_samples);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set kernel arguments");
    }
    
    // Execute kernel
    size_t global_size = total_samples;
    err = clEnqueueNDRangeKernel(
        manager_.GetQueue(),
        kernel_lfm_delayed_,
        1, nullptr,
        &global_size, nullptr,
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue kernel");
    }
    
    // Wait for completion
    clFinish(manager_.GetQueue());
    
    return gpu_signal_delayed_;
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL SOURCE CODE
// ═══════════════════════════════════════════════════════════════════

std::string GeneratorGPU::GetLFMKernelSource() {
    return R"(
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        
        /**
         * LFM (Linear Frequency Modulation) signal generation kernel
         * Generates chirp signals on GPU
         */
        
        __kernel void lfm_basic(
            __global float2* output,
            uint total_samples,
            float f_start,
            float chirp_rate,
            float sample_rate
        ) {
            uint idx = get_global_id(0);
            
            if (idx >= total_samples) return;
            
            // Time at this sample
            float t = (float)idx / sample_rate;
            
            // Instantaneous frequency: f(t) = f_start + chirp_rate * t
            float f_inst = f_start + chirp_rate * t;
            
            // Phase: 2*pi*(f_start*t + 0.5*chirp_rate*t^2)
            float phase = 2.0f * M_PI_F * (f_start * t + 0.5f * chirp_rate * t * t);
            
            // Generate complex exponential: exp(j*phase) = cos(phase) + j*sin(phase)
            output[idx].x = cos(phase);
            output[idx].y = sin(phase);
        }
        
        __kernel void lfm_delayed(
            __global float2* output,
            uint total_samples,
            float f_start,
            float chirp_rate,
            float sample_rate,
            float delay_samples
        ) {
            uint idx = get_global_id(0);
            
            if (idx >= total_samples) return;
            
            float result_real = 0.0f;
            float result_imag = 0.0f;
            
            if ((float)idx >= delay_samples) {
                // Time at this sample (accounting for delay)
                float t = ((float)idx - delay_samples) / sample_rate;
                
                // Instantaneous frequency
                float f_inst = f_start + chirp_rate * t;
                
                // Phase
                float phase = 2.0f * M_PI_F * (f_start * t + 0.5f * chirp_rate * t * t);
                
                // Complex exponential
                result_real = cos(phase);
                result_imag = sin(phase);
            }
            
            output[idx].x = result_real;
            output[idx].y = result_imag;
        }
    )";
}

} // namespace gpu
} // namespace radar
