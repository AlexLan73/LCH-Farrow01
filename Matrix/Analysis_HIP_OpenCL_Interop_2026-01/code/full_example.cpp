/**
 * @file full_example.cpp
 * @brief Полный пример: OpenCL preprocessing + HIP/rocBLAS matrix operations
 * @date 2026-01-26
 * 
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 * ============================================================================
 * 
 * Демонстрирует РЕАЛЬНЫЙ СЦЕНАРИЙ интеграции OpenCL и HIP/rocBLAS:
 * 
 *   ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
 *   │   OpenCL    │ ──► │   rocBLAS    │ ──► │   OpenCL    │
 *   │ Preprocess  │     │    GEMM      │     │ Postprocess │
 *   │  (scale)    │     │   (A × B)    │     │ (magnitude) │
 *   └─────────────┘     └──────────────┘     └─────────────┘
 *          │                   │                    │
 *          └───────────────────┴────────────────────┘
 *                              │
 *                       ОДИН SVM БУФЕР
 *                      (без копирования!)
 * 
 * ============================================================================
 * ЗАЧЕМ ЭТО НУЖНО:
 * ============================================================================
 * 
 * Проблема:
 *   - Много существующего кода на OpenCL (FFT, фильтры, preprocessing)
 *   - Нужно добавить rocBLAS/rocSOLVER для матричных операций
 *   - Переписывать весь OpenCL код на HIP нет времени/возможности
 *   - Копирование GPU↔CPU между операциями - узкое место
 * 
 * Решение:
 *   - Использовать OpenCL SVM (Shared Virtual Memory) буферы
 *   - Тот же указатель работает в OpenCL И в HIP/rocBLAS!
 *   - Данные остаются на GPU всё время - zero-copy
 * 
 * ============================================================================
 * КАК АДАПТИРОВАТЬ ДЛЯ СВОЕГО ПРОЕКТА:
 * ============================================================================
 * 
 *   // 1. Выделить SVM буфер (вместо clCreateBuffer или hipMalloc)
 *   Complex* data = (Complex*)clSVMAlloc(ctx, 
 *       CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, size, 0);
 *   
 *   // 2. Ваш OpenCL код (FFT, фильтры и т.д.)
 *   clSetKernelArgSVMPointer(kernel, 0, data);  // <-- SVM вместо cl_mem
 *   clEnqueueNDRangeKernel(...);
 *   clFinish(queue);  // ВАЖНО: синхронизация перед HIP!
 *   
 *   // 3. HIP/rocBLAS/rocSOLVER (тот же указатель!)
 *   rocblas_cgemm(handle, ..., data, ...);
 *   // или: rocsolver_cpotrf(handle, ..., data, ...);
 *   hipDeviceSynchronize();  // ВАЖНО: синхронизация перед OpenCL!
 *   
 *   // 4. Снова OpenCL (тот же указатель!)
 *   clSetKernelArgSVMPointer(another_kernel, 0, data);
 *   clEnqueueNDRangeKernel(...);
 * 
 * ============================================================================
 * ТРЕБОВАНИЯ:
 * ============================================================================
 * 
 *   - AMD GPU с поддержкой Fine Grain SVM (MI100, MI200, MI300 и др.)
 *   - ROCm 5.0+ (тестировалось на 6.3.2)
 *   - OpenCL 2.0+ с SVM поддержкой
 * 
 * ============================================================================
 * КОМПИЛЯЦИЯ И ЗАПУСК:
 * ============================================================================
 * 
 *   hipcc -O3 full_example.cpp -o full_example -lOpenCL -lrocblas
 *   ./full_example
 * 
 * ============================================================================
 */

#include <hip/hip_runtime.h>
#include <CL/cl.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <chrono>

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

#define CHECK_ROCBLAS(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS Error: " << status << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

using Complex = rocblas_float_complex;

const char* opencl_kernels = R"(
typedef struct { float real; float imag; } Complex;

__kernel void preprocess(__global Complex* data, int n, float scale) {
    int idx = get_global_id(0);
    if (idx < n) {
        data[idx].real = data[idx].real * scale;
        data[idx].imag = data[idx].imag * scale;
    }
}

__kernel void postprocess(__global Complex* data, __global float* magnitudes, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        magnitudes[idx] = data[idx].real * data[idx].real + data[idx].imag * data[idx].imag;
    }
}
)";

int main() {
    const int N = 64;
    
    std::cout << "=============================================" << std::endl;
    std::cout << "   Full Example: OpenCL + rocBLAS Pipeline" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Matrix size: " << N << " x " << N << " complex" << std::endl << std::endl;
    
    // ========================================
    // Initialize HIP
    // ========================================
    CHECK_HIP(hipSetDevice(0));
    
    // ========================================
    // Initialize OpenCL
    // ========================================
    cl_platform_id platform;
    cl_device_id device;
    cl_int cl_err;
    
    CHECK_CL(clGetPlatformIDs(1, &platform, nullptr));
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
    
    cl_context cl_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &cl_err);
    CHECK_CL(cl_err);
    
    cl_command_queue cl_queue = clCreateCommandQueueWithProperties(cl_ctx, device, nullptr, &cl_err);
    CHECK_CL(cl_err);
    
    // ========================================
    // Initialize rocBLAS
    // ========================================
    rocblas_handle rocblas_handle;
    CHECK_ROCBLAS(rocblas_create_handle(&rocblas_handle));
    
    // ========================================
    // Build OpenCL kernels
    // ========================================
    cl_program program = clCreateProgramWithSource(cl_ctx, 1, &opencl_kernels, nullptr, &cl_err);
    CHECK_CL(cl_err);
    CHECK_CL(clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr));
    
    cl_kernel preprocess_kernel = clCreateKernel(program, "preprocess", &cl_err);
    CHECK_CL(cl_err);
    cl_kernel postprocess_kernel = clCreateKernel(program, "postprocess", &cl_err);
    CHECK_CL(cl_err);
    
    // ========================================
    // Allocate SVM buffers
    // ========================================
    cl_svm_mem_flags svm_flags = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
    
    Complex* A = (Complex*)clSVMAlloc(cl_ctx, svm_flags, N * N * sizeof(Complex), 0);
    Complex* B = (Complex*)clSVMAlloc(cl_ctx, svm_flags, N * N * sizeof(Complex), 0);
    Complex* C = (Complex*)clSVMAlloc(cl_ctx, svm_flags, N * N * sizeof(Complex), 0);
    float* magnitudes = (float*)clSVMAlloc(cl_ctx, svm_flags, N * N * sizeof(float), 0);
    
    if (!A || !B || !C || !magnitudes) {
        std::cerr << "SVM allocation failed!" << std::endl;
        return 1;
    }
    
    std::cout << "SVM buffers allocated" << std::endl;
    
    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = {static_cast<float>(i % N) * 0.1f, 0.0f};
        B[i] = {static_cast<float>(i / N) * 0.1f, 0.0f};
        C[i] = {0.0f, 0.0f};
    }
    
    std::cout << "Initial A[0,0]: " << A[0].real() << std::endl;
    std::cout << std::endl;
    
    // ========================================
    // Run pipeline
    // ========================================
    std::cout << "Running pipeline..." << std::endl << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: OpenCL preprocessing
    std::cout << "Step 1: OpenCL preprocessing..." << std::endl;
    {
        size_t global_size = N * N;
        int n_arg = N * N;
        float scale = 2.0f;
        
        CHECK_CL(clSetKernelArgSVMPointer(preprocess_kernel, 0, A));
        CHECK_CL(clSetKernelArg(preprocess_kernel, 1, sizeof(int), &n_arg));
        CHECK_CL(clSetKernelArg(preprocess_kernel, 2, sizeof(float), &scale));
        CHECK_CL(clEnqueueNDRangeKernel(cl_queue, preprocess_kernel, 1, nullptr,
                                         &global_size, nullptr, 0, nullptr, nullptr));
        
        CHECK_CL(clSetKernelArgSVMPointer(preprocess_kernel, 0, B));
        CHECK_CL(clEnqueueNDRangeKernel(cl_queue, preprocess_kernel, 1, nullptr,
                                         &global_size, nullptr, 0, nullptr, nullptr));
        
        CHECK_CL(clFinish(cl_queue));
    }
    std::cout << "  A[0,0] after preprocessing: " << A[0].real() << std::endl;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Step 2: rocBLAS CGEMM
    std::cout << "Step 2: rocBLAS CGEMM (C = A * B)..." << std::endl;
    {
        Complex alpha = {1.0f, 0.0f};
        Complex beta = {0.0f, 0.0f};
        
        CHECK_ROCBLAS(rocblas_cgemm(
            rocblas_handle,
            rocblas_operation_none,
            rocblas_operation_none,
            N, N, N,
            &alpha,
            A, N,
            B, N,
            &beta,
            C, N
        ));
        
        CHECK_HIP(hipDeviceSynchronize());
    }
    std::cout << "  C[0,0] after GEMM: " << C[0].real() << " + " << C[0].imag() << "i" << std::endl;
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // Step 3: OpenCL postprocessing
    std::cout << "Step 3: OpenCL postprocessing..." << std::endl;
    {
        size_t global_size = N * N;
        int n_arg = N * N;
        
        CHECK_CL(clSetKernelArgSVMPointer(postprocess_kernel, 0, C));
        CHECK_CL(clSetKernelArgSVMPointer(postprocess_kernel, 1, magnitudes));
        CHECK_CL(clSetKernelArg(postprocess_kernel, 2, sizeof(int), &n_arg));
        CHECK_CL(clEnqueueNDRangeKernel(cl_queue, postprocess_kernel, 1, nullptr,
                                         &global_size, nullptr, 0, nullptr, nullptr));
        CHECK_CL(clFinish(cl_queue));
    }
    std::cout << "  Magnitude[0,0]: " << magnitudes[0] << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // ========================================
    // Results
    // ========================================
    double preprocess_ms = std::chrono::duration<double, std::milli>(t1 - start).count();
    double gemm_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double postprocess_ms = std::chrono::duration<double, std::milli>(end - t2).count();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "RESULTS" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Timing:" << std::endl;
    std::cout << "  OpenCL preprocess:  " << preprocess_ms << " ms" << std::endl;
    std::cout << "  rocBLAS CGEMM:      " << gemm_ms << " ms" << std::endl;
    std::cout << "  OpenCL postprocess: " << postprocess_ms << " ms" << std::endl;
    std::cout << "  Total:              " << total_ms << " ms" << std::endl;
    std::cout << std::endl;
    
    std::cout << "✅ SUCCESS!" << std::endl;
    std::cout << "  - OpenCL and rocBLAS shared the same SVM memory" << std::endl;
    std::cout << "  - NO memory copies between APIs" << std::endl;
    std::cout << "  - Data stayed on GPU throughout the pipeline" << std::endl;
    
    // ========================================
    // Cleanup
    // ========================================
    clSVMFree(cl_ctx, A);
    clSVMFree(cl_ctx, B);
    clSVMFree(cl_ctx, C);
    clSVMFree(cl_ctx, magnitudes);
    
    clReleaseKernel(preprocess_kernel);
    clReleaseKernel(postprocess_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cl_queue);
    clReleaseContext(cl_ctx);
    
    rocblas_destroy_handle(rocblas_handle);
    
    std::cout << std::endl << "=============================================" << std::endl;
    
    return 0;
}
