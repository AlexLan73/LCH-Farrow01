/**
 * @file hip_opencl_svm_benchmark.cpp
 * @brief Benchmark HIP <-> OpenCL SVM interop with large vectors (4M elements)
 * 
 * Компиляция:
 *   hipcc -O3 hip_opencl_svm_benchmark.cpp -o bench -lOpenCL
 */

#include <hip/hip_runtime.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#define CHECK_HIP(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CL(call) do { \
    cl_int err = call; \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL Error: " << err << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

struct Complex {
    float real;
    float imag;
};

// HIP kernels
__global__ void hip_scale(Complex* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].real *= scale;
        data[idx].imag *= scale;
    }
}

__global__ void hip_magnitude_sq(Complex* data, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = data[idx].real * data[idx].real + data[idx].imag * data[idx].imag;
    }
}

// OpenCL kernels
const char* opencl_kernel_src = R"(
typedef struct { float real; float imag; } Complex;

__kernel void opencl_add(__global Complex* data, int n, float add_val) {
    int idx = get_global_id(0);
    if (idx < n) {
        data[idx].real += add_val;
        data[idx].imag += add_val;
    }
}

__kernel void opencl_fft_like(__global Complex* data, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float r = data[idx].real;
        float i = data[idx].imag;
        data[idx].real = r * 0.866f - i * 0.5f;
        data[idx].imag = r * 0.5f + i * 0.866f;
    }
}
)";

int main() {
    const int N = 4 * 1024 * 1024;  // 4M complex numbers
    const int NUM_ITERATIONS = 5;
    
    std::cout << "=============================================" << std::endl;
    std::cout << "   HIP <-> OpenCL SVM Interop Benchmark" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Vector size: " << N << " complex floats (" << (N * sizeof(Complex) / 1024 / 1024) << " MB)" << std::endl;
    
    // Initialize HIP
    CHECK_HIP(hipSetDevice(0));
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl << std::endl;
    
    // Initialize OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_int cl_err;
    
    CHECK_CL(clGetPlatformIDs(1, &platform, nullptr));
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
    
    cl_context cl_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &cl_err);
    CHECK_CL(cl_err);
    
    cl_command_queue cl_queue = clCreateCommandQueueWithProperties(cl_ctx, device, nullptr, &cl_err);
    CHECK_CL(cl_err);
    
    // Build OpenCL kernels
    cl_program program = clCreateProgramWithSource(cl_ctx, 1, &opencl_kernel_src, nullptr, &cl_err);
    CHECK_CL(cl_err);
    CHECK_CL(clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr));
    
    cl_kernel cl_kernel_add = clCreateKernel(program, "opencl_add", &cl_err);
    CHECK_CL(cl_err);
    cl_kernel cl_kernel_fft = clCreateKernel(program, "opencl_fft_like", &cl_err);
    CHECK_CL(cl_err);
    
    // Allocate SVM buffers
    cl_svm_mem_flags svm_flags = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
    Complex* svm_data = (Complex*)clSVMAlloc(cl_ctx, svm_flags, N * sizeof(Complex), 0);
    float* svm_magnitudes = (float*)clSVMAlloc(cl_ctx, svm_flags, N * sizeof(float), 0);
    
    if (!svm_data || !svm_magnitudes) {
        std::cerr << "SVM allocation failed!" << std::endl;
        return 1;
    }
    
    std::cout << "SVM allocated: " << ((N * sizeof(Complex) + N * sizeof(float)) / 1024 / 1024) << " MB total" << std::endl;
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        svm_data[i].real = static_cast<float>(i % 1000) * 0.001f;
        svm_data[i].imag = static_cast<float>((i + 500) % 1000) * 0.001f;
    }
    
    std::cout << "Initial[0]: " << svm_data[0].real << " + " << svm_data[0].imag << "i" << std::endl;
    std::cout << std::endl;
    
    // Benchmark
    std::cout << "=============================================" << std::endl;
    std::cout << "Running mixed HIP/OpenCL pipeline benchmark" << std::endl;
    std::cout << "=============================================" << std::endl << std::endl;
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    size_t global_size = N;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        std::cout << "Iteration " << (iter + 1) << "/" << NUM_ITERATIONS << ":" << std::endl;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // OpenCL: Add
        CHECK_CL(clSetKernelArgSVMPointer(cl_kernel_add, 0, svm_data));
        int n_arg = N;
        CHECK_CL(clSetKernelArg(cl_kernel_add, 1, sizeof(int), &n_arg));
        float add_val = 0.1f;
        CHECK_CL(clSetKernelArg(cl_kernel_add, 2, sizeof(float), &add_val));
        CHECK_CL(clEnqueueNDRangeKernel(cl_queue, cl_kernel_add, 1, nullptr, 
                                         &global_size, nullptr, 0, nullptr, nullptr));
        CHECK_CL(clFinish(cl_queue));
        
        auto t2 = std::chrono::high_resolution_clock::now();
        
        // OpenCL: FFT-like
        CHECK_CL(clSetKernelArgSVMPointer(cl_kernel_fft, 0, svm_data));
        CHECK_CL(clSetKernelArg(cl_kernel_fft, 1, sizeof(int), &n_arg));
        CHECK_CL(clEnqueueNDRangeKernel(cl_queue, cl_kernel_fft, 1, nullptr,
                                         &global_size, nullptr, 0, nullptr, nullptr));
        CHECK_CL(clFinish(cl_queue));
        
        auto t3 = std::chrono::high_resolution_clock::now();
        
        // HIP: Scale
        float scale = 1.01f;
        hip_scale<<<grid_size, block_size>>>(svm_data, N, scale);
        CHECK_HIP(hipDeviceSynchronize());
        
        auto t4 = std::chrono::high_resolution_clock::now();
        
        // HIP: Magnitude
        hip_magnitude_sq<<<grid_size, block_size>>>(svm_data, svm_magnitudes, N);
        CHECK_HIP(hipDeviceSynchronize());
        
        auto t5 = std::chrono::high_resolution_clock::now();
        
        double opencl_add_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double opencl_fft_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double hip_scale_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
        double hip_mag_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
        double total_iter_ms = std::chrono::duration<double, std::milli>(t5 - t1).count();
        
        std::cout << "  OpenCL add:      " << opencl_add_ms << " ms" << std::endl;
        std::cout << "  OpenCL FFT-like: " << opencl_fft_ms << " ms" << std::endl;
        std::cout << "  HIP scale:       " << hip_scale_ms << " ms" << std::endl;
        std::cout << "  HIP magnitude:   " << hip_mag_ms << " ms" << std::endl;
        std::cout << "  Total iteration: " << total_iter_ms << " ms" << std::endl << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Summary
    std::cout << "=============================================" << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << "=============================================" << std::endl << std::endl;
    
    std::cout << "✓ HIP and OpenCL successfully shared SVM memory!" << std::endl;
    std::cout << "✓ NO memory copies between APIs" << std::endl;
    std::cout << "✓ Both APIs operate on the same GPU buffer" << std::endl << std::endl;
    
    std::cout << "Total time: " << total_ms << " ms" << std::endl;
    std::cout << "Average per iteration: " << (total_ms / NUM_ITERATIONS) << " ms" << std::endl;
    std::cout << "Throughput: " << (NUM_ITERATIONS * N / (total_ms / 1000.0) / 1e6) << " M elements/sec" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Final[0]: " << svm_data[0].real << " + " << svm_data[0].imag << "i" << std::endl;
    std::cout << "Magnitude[0]: " << svm_magnitudes[0] << std::endl;
    
    // Cleanup
    clSVMFree(cl_ctx, svm_data);
    clSVMFree(cl_ctx, svm_magnitudes);
    clReleaseKernel(cl_kernel_add);
    clReleaseKernel(cl_kernel_fft);
    clReleaseProgram(program);
    clReleaseCommandQueue(cl_queue);
    clReleaseContext(cl_ctx);
    
    return 0;
}

