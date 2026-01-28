# Практическое руководство по внедрению HIP ↔ OpenCL Interop

---

## 1. Подготовка окружения

### 1.1 Проверка требований

```bash
# Проверить версию ROCm
cat /opt/rocm/.info/version

# Проверить поддержку SVM
clinfo | grep -A4 "SVM capabilities"
# Должно быть:
#   Coarse grain buffer: Yes
#   Fine grain buffer:   Yes  <-- ОБЯЗАТЕЛЬНО

# Проверить что HIP видит GPU
hipconfig --full
```

### 1.2 Зависимости

```cmake
# CMakeLists.txt
find_package(hip REQUIRED)
find_package(OpenCL REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

target_link_libraries(your_app
    hip::device
    OpenCL::OpenCL
    roc::rocblas
    roc::rocsolver
)
```

---

## 2. Базовая интеграция

### 2.1 Инициализация

```cpp
#include <hip/hip_runtime.h>
#include <CL/cl.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

class SVMInteropManager {
private:
    cl_context cl_ctx_;
    cl_command_queue cl_queue_;
    cl_device_id cl_device_;
    rocblas_handle rocblas_handle_;
    
public:
    SVMInteropManager() {
        // Инициализация HIP
        CHECK_HIP(hipSetDevice(0));
        
        // Инициализация OpenCL
        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, nullptr);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &cl_device_, nullptr);
        
        // Проверка SVM
        cl_device_svm_capabilities svm_caps;
        clGetDeviceInfo(cl_device_, CL_DEVICE_SVM_CAPABILITIES, 
                        sizeof(svm_caps), &svm_caps, nullptr);
        
        if (!(svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
            throw std::runtime_error("Fine grain SVM not supported!");
        }
        
        cl_int err;
        cl_ctx_ = clCreateContext(nullptr, 1, &cl_device_, nullptr, nullptr, &err);
        cl_queue_ = clCreateCommandQueueWithProperties(cl_ctx_, cl_device_, nullptr, &err);
        
        // Инициализация rocBLAS
        rocblas_create_handle(&rocblas_handle_);
    }
    
    ~SVMInteropManager() {
        rocblas_destroy_handle(rocblas_handle_);
        clReleaseCommandQueue(cl_queue_);
        clReleaseContext(cl_ctx_);
    }
    
    // Выделение SVM буфера
    template<typename T>
    T* allocate(size_t count) {
        cl_svm_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
        T* ptr = (T*)clSVMAlloc(cl_ctx_, flags, count * sizeof(T), 0);
        if (!ptr) {
            throw std::runtime_error("SVM allocation failed");
        }
        return ptr;
    }
    
    // Освобождение
    void free(void* ptr) {
        clSVMFree(cl_ctx_, ptr);
    }
    
    // Синхронизация после OpenCL
    void syncOpenCL() {
        clFinish(cl_queue_);
    }
    
    // Синхронизация после HIP
    void syncHIP() {
        hipDeviceSynchronize();
    }
    
    // Getters
    cl_context context() { return cl_ctx_; }
    cl_command_queue queue() { return cl_queue_; }
    rocblas_handle rocblasHandle() { return rocblas_handle_; }
};
```

### 2.2 Использование с существующим OpenCL кодом

```cpp
// Было (старый OpenCL код):
cl_mem buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, nullptr, &err);
clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);

// Стало:
SVMInteropManager mgr;
float* svm_buffer = mgr.allocate<float>(N);

// OpenCL kernel
clSetKernelArgSVMPointer(kernel, 0, svm_buffer);  // <-- Изменение здесь
clEnqueueNDRangeKernel(mgr.queue(), kernel, ...);
mgr.syncOpenCL();

// HIP kernel (тот же указатель!)
my_hip_kernel<<<grid, block>>>(svm_buffer, N);
mgr.syncHIP();
```

---

## 3. Миграция существующего OpenCL кода

### 3.1 Минимальные изменения для поддержки SVM

```cpp
// === OpenCL Wrapper Class ===

class OpenCLSVMKernel {
private:
    cl_kernel kernel_;
    cl_command_queue queue_;
    
public:
    OpenCLSVMKernel(cl_program program, const char* name, cl_command_queue queue)
        : queue_(queue) {
        cl_int err;
        kernel_ = clCreateKernel(program, name, &err);
    }
    
    // Для SVM буферов
    void setArgSVM(int index, void* svm_ptr) {
        clSetKernelArgSVMPointer(kernel_, index, svm_ptr);
    }
    
    // Для обычных аргументов
    template<typename T>
    void setArg(int index, const T& value) {
        clSetKernelArg(kernel_, index, sizeof(T), &value);
    }
    
    void run(size_t global_size, size_t local_size = 0) {
        size_t* local = local_size > 0 ? &local_size : nullptr;
        clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, 
                               &global_size, local, 0, nullptr, nullptr);
    }
    
    void finish() {
        clFinish(queue_);
    }
};
```

### 3.2 Адаптация clFFT (если используете)

```cpp
// clFFT с SVM буферами
clfftPlanHandle plan;
clfftCreateDefaultPlan(&plan, ctx, CLFFT_1D, &fft_size);
clfftSetPlanPrecision(plan, CLFFT_SINGLE);
clfftSetLayout(plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
clfftBakePlan(plan, 1, &queue, nullptr, nullptr);

// Использование с SVM
cl_mem svm_buffer_wrapper;
// Примечание: clFFT может требовать обёртку SVM в cl_mem
// В этом случае используйте clCreateBuffer с CL_MEM_USE_HOST_PTR

// Альтернатива: rocFFT с тем же SVM указателем
rocfft_plan hip_plan;
rocfft_plan_create(&hip_plan, rocfft_placement_inplace,
                    rocfft_transform_type_complex_forward,
                    rocfft_precision_single, 1, &fft_size, 1, nullptr);
                    
rocfft_execute(hip_plan, (void**)&svm_buffer, nullptr, nullptr);
```

---

## 4. Интеграция с rocBLAS/rocSOLVER

### 4.1 Матричные операции с SVM

```cpp
// Complex matrix в SVM
using Complex = rocblas_float_complex;

SVMInteropManager mgr;
Complex* matrix = mgr.allocate<Complex>(N * N);

// Инициализация через OpenCL или CPU
for (int i = 0; i < N * N; i++) {
    matrix[i] = {(float)(i % N), 0.0f};
}

// rocBLAS операции напрямую с SVM указателем
Complex alpha = {1.0f, 0.0f};
Complex beta = {0.0f, 0.0f};

rocblas_cgemm(mgr.rocblasHandle(),
    rocblas_operation_none, rocblas_operation_none,
    N, N, N,
    &alpha,
    matrix, N,
    matrix, N,
    &beta,
    matrix, N);

mgr.syncHIP();
```

### 4.2 Инверсия матрицы с Cholesky

```cpp
// Выделение
Complex* A = mgr.allocate<Complex>(N * N);
rocblas_int* info = mgr.allocate<rocblas_int>(1);

// Заполнение положительно определённой матрицы...

// Инверсия через Cholesky (rocSOLVER)
rocsolver_cpotrf(mgr.rocblasHandle(), 
                  rocblas_fill_lower, N, A, N, info);
                  
rocsolver_cpotri(mgr.rocblasHandle(),
                  rocblas_fill_lower, N, A, N, info);

mgr.syncHIP();

// Результат в A - можно использовать в OpenCL
```

---

## 5. Паттерны использования

### 5.1 Pipeline: OpenCL preprocessing → HIP matrix ops → OpenCL postprocessing

```cpp
void process_pipeline(SVMInteropManager& mgr, Complex* data, int N) {
    // 1. OpenCL: FFT
    opencl_fft_kernel.setArgSVM(0, data);
    opencl_fft_kernel.setArg(1, N);
    opencl_fft_kernel.run(N);
    mgr.syncOpenCL();  // ВАЖНО!
    
    // 2. HIP: Matrix multiplication
    rocblas_cgemm(mgr.rocblasHandle(), ...);
    mgr.syncHIP();  // ВАЖНО!
    
    // 3. OpenCL: Inverse FFT
    opencl_ifft_kernel.setArgSVM(0, data);
    opencl_ifft_kernel.setArg(1, N);
    opencl_ifft_kernel.run(N);
    mgr.syncOpenCL();
}
```

### 5.2 Batch processing

```cpp
void process_batch(SVMInteropManager& mgr, 
                   std::vector<Complex*>& batch, int N) {
    for (Complex* data : batch) {
        // OpenCL preprocessing
        opencl_preprocess.setArgSVM(0, data);
        opencl_preprocess.run(N);
    }
    mgr.syncOpenCL();
    
    // HIP batch operation
    for (Complex* data : batch) {
        rocblas_operation(mgr.rocblasHandle(), data, N);
    }
    mgr.syncHIP();
}
```

---

## 6. Debugging и профилирование

### 6.1 Проверка корректности

```cpp
void validate_svm_access(void* svm_ptr, size_t size) {
    // Проверка что OpenCL видит данные
    float* test = (float*)svm_ptr;
    test[0] = 42.0f;
    
    // HIP kernel для проверки
    __global__ void check_kernel(float* data) {
        if (threadIdx.x == 0) {
            printf("HIP sees: %f\n", data[0]);
            data[0] = 100.0f;
        }
    }
    
    check_kernel<<<1, 1>>>((float*)svm_ptr);
    hipDeviceSynchronize();
    
    printf("After HIP: %f\n", test[0]);  // Должно быть 100.0
}
```

### 6.2 Профилирование

```bash
# ROCm profiler
rocprof --stats ./your_app

# С timeline
rocprof --hip-trace --hsa-trace ./your_app
```

---

## 7. Troubleshooting

### Проблема: "SVM allocation failed"
```cpp
// Причина: Недостаточно памяти или SVM не поддерживается
// Решение: Проверить доступную память
size_t free, total;
hipMemGetInfo(&free, &total);
printf("Free: %zu MB\n", free / 1024 / 1024);
```

### Проблема: "Memory fault в HIP"
```cpp
// Причина: Забыли синхронизировать после OpenCL
// Решение: Всегда вызывать clFinish() перед HIP операциями
clFinish(queue);  // <-- НЕ ЗАБЫВАТЬ!
hip_kernel<<<grid, block>>>(svm_ptr, n);
```

### Проблема: "Неверные результаты"
```cpp
// Причина: Race condition между API
// Решение: Синхронизация после КАЖДОГО перехода между API
clFinish(queue);           // После OpenCL
hipDeviceSynchronize();    // После HIP
```

---

## 8. Пример полного приложения

См. файл `code/full_example.cpp` в этом каталоге.

