/**
 * @file hip_opencl_svm_test.cpp  
 * @brief Базовый тест HIP <-> OpenCL interop через SVM
 * 
 * Демонстрирует что один SVM буфер доступен обоим API
 * 
 * Компиляция:
 *   hipcc -O3 hip_opencl_svm_test.cpp -o test -lOpenCL
 * 
 * Запуск:
 *   ./test
 */

#include <hip/hip_runtime.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>
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

// HIP kernel: add 1 to each element
__global__ void hip_add_one(Complex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].real += 1.0f;
        data[idx].imag += 1.0f;
    }
}

// OpenCL kernel source
const char* opencl_kernel_src = R"(
typedef struct { float real; float imag; } Complex;
__kernel void opencl_add_one(__global Complex* data, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        data[idx].real += 1.0f;
        data[idx].imag += 1.0f;
    }
}
)";

int main() {
    const int N = 11;  // 0 to 10
    const int NUM_ITERATIONS = 10;
    
    std::cout << "=============================================" << std::endl;
    std::cout << "   HIP <-> OpenCL SVM Interop Basic Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
    
    // Initialize HIP
    CHECK_HIP(hipSetDevice(0));
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    
    // Initialize OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_int cl_err;
    
    CHECK_CL(clGetPlatformIDs(1, &platform, nullptr));
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
    
    char device_name[256];
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr));
    std::cout << "OpenCL device: " << device_name << std::endl;
    
    // Check SVM capabilities
    cl_device_svm_capabilities svm_caps;
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr));
    
    std::cout << std::endl << "SVM capabilities:" << std::endl;
    if (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) std::cout << "  - Coarse grain buffer ✓" << std::endl;
    if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) std::cout << "  - Fine grain buffer ✓" << std::endl;
    if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) std::cout << "  - Fine grain system ✓" << std::endl;
    if (svm_caps & CL_DEVICE_SVM_ATOMICS) std::cout << "  - SVM atomics ✓" << std::endl;
    
    if (!(svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
        std::cerr << std::endl << "ERROR: Fine grain SVM not supported!" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    cl_context cl_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &cl_err);
    CHECK_CL(cl_err);
    
    cl_command_queue cl_queue = clCreateCommandQueueWithProperties(cl_ctx, device, nullptr, &cl_err);
    CHECK_CL(cl_err);
    
    // Build OpenCL kernel
    cl_program program = clCreateProgramWithSource(cl_ctx, 1, &opencl_kernel_src, nullptr, &cl_err);
    CHECK_CL(cl_err);
    CHECK_CL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));
    cl_kernel cl_kernel_add = clCreateKernel(program, "opencl_add_one", &cl_err);
    CHECK_CL(cl_err);
    
    // Allocate SVM buffer (Fine Grain)
    cl_svm_mem_flags svm_flags = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
    Complex* svm_ptr = (Complex*)clSVMAlloc(cl_ctx, svm_flags, N * sizeof(Complex), 0);
    if (!svm_ptr) {
        std::cerr << "SVM allocation failed!" << std::endl;
        return 1;
    }
    
    std::cout << "SVM pointer: " << svm_ptr << std::endl;
    
    // Initialize data: {0+0i, 1+1i, 2+2i, ..., 10+10i}
    for (int i = 0; i < N; i++) {
        svm_ptr[i].real = static_cast<float>(i);
        svm_ptr[i].imag = static_cast<float>(i);
    }
    
    std::cout << "Initial: {0+0i, 1+1i, ..., 10+10i}" << std::endl << std::endl;
    
    // Run mixed pipeline
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    size_t global_size = N;
    
    std::cout << "Running " << NUM_ITERATIONS << " iterations..." << std::endl;
    std::cout << "Each iteration: OpenCL +1, then HIP +1" << std::endl << std::endl;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // OpenCL kernel: +1
        CHECK_CL(clSetKernelArgSVMPointer(cl_kernel_add, 0, svm_ptr));
        int n_arg = N;
        CHECK_CL(clSetKernelArg(cl_kernel_add, 1, sizeof(int), &n_arg));
        CHECK_CL(clEnqueueNDRangeKernel(cl_queue, cl_kernel_add, 1, nullptr, 
                                         &global_size, nullptr, 0, nullptr, nullptr));
        CHECK_CL(clFinish(cl_queue));  // Sync before HIP
        
        // HIP kernel: +1
        hip_add_one<<<grid_size, block_size>>>(svm_ptr, N);
        CHECK_HIP(hipDeviceSynchronize());  // Sync before next OpenCL
        
        std::cout << "  Iteration " << (iter + 1) << " complete" << std::endl;
    }
    
    // Verify results
    std::cout << std::endl << "=============================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    bool all_correct = true;
    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>(i) + 2.0f * NUM_ITERATIONS;
        bool correct = (std::abs(svm_ptr[i].real - expected) < 0.001f);
        
        std::cout << "  [" << i << "]: " << svm_ptr[i].real << " + " << svm_ptr[i].imag << "i";
        if (correct) {
            std::cout << " ✓";
        } else {
            std::cout << " ✗ (expected " << expected << ")";
            all_correct = false;
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    if (all_correct) {
        std::cout << "✅ SUCCESS: HIP and OpenCL successfully shared SVM memory!" << std::endl;
    } else {
        std::cout << "❌ FAILURE: Results don't match expected values!" << std::endl;
    }
    
    // Cleanup
    clSVMFree(cl_ctx, svm_ptr);
    clReleaseKernel(cl_kernel_add);
    clReleaseProgram(program);
    clReleaseCommandQueue(cl_queue);
    clReleaseContext(cl_ctx);
    
    std::cout << std::endl << "=============================================" << std::endl;
    
    return all_correct ? 0 : 1;
}

