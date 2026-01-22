/**
 * @file matrix_invert.cpp
 * @brief GPU Matrix Inversion for 341×341 Hermitian Matrix
 * @target AMD AI100 (gfx908) - < 4 ms
 * @os Debian Linux
 * 
 * Методы:
 * 1. LU (GETRF + GETRI) - общий метод
 * 2. Hybrid (GETRF + TRSM) - оптимизированный общий
 * 3. Cholesky (POTRF + POTRI) - для положительно определённых эрмитовых матриц (ЛУЧШИЙ)
 * 
 * @author Optimized by Codo
 * @date 2026-01-22
 */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <complex>
#include <algorithm>
#include <numeric>

// ============================================================================
// Configuration
// ============================================================================

constexpr int MATRIX_SIZE = 341;
constexpr int NUM_ITERATIONS = 10;
constexpr int WARMUP_ITERATIONS = 3;
constexpr float TARGET_TIME_MS = 4.0f;  // Цель: < 4 мс

// ============================================================================
// Type Definitions
// ============================================================================

using ComplexFloat = rocblas_float_complex;

// Helper для создания комплексного числа
inline ComplexFloat make_complex(float r, float i) {
    ComplexFloat c;
    c.x = r;
    c.y = i;
    return c;
}

inline float complex_abs(const ComplexFloat& c) {
    return std::sqrt(c.x * c.x + c.y * c.y);
}

inline ComplexFloat complex_conj(const ComplexFloat& c) {
    return make_complex(c.x, -c.y);
}

inline ComplexFloat complex_mul(const ComplexFloat& a, const ComplexFloat& b) {
    return make_complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline ComplexFloat complex_add(const ComplexFloat& a, const ComplexFloat& b) {
    return make_complex(a.x + b.x, a.y + b.y);
}

inline ComplexFloat complex_sub(const ComplexFloat& a, const ComplexFloat& b) {
    return make_complex(a.x - b.x, a.y - b.y);
}

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CHECK_HIP(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_ROCBLAS(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS Error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl; \
        exit(1); \
    } \
} while(0)

// ============================================================================
// GPU Timer using HIP Events (ПРАВИЛЬНЫЙ способ измерения GPU времени)
// ============================================================================

class GPUTimer {
private:
    hipEvent_t start_event, stop_event;
    hipStream_t stream;
    
public:
    GPUTimer(hipStream_t s = nullptr) : stream(s) {
        CHECK_HIP(hipEventCreate(&start_event));
        CHECK_HIP(hipEventCreate(&stop_event));
    }
    
    ~GPUTimer() {
        hipEventDestroy(start_event);
        hipEventDestroy(stop_event);
    }
    
    void start() {
        CHECK_HIP(hipEventRecord(start_event, stream));
    }
    
    float stop() {
        CHECK_HIP(hipEventRecord(stop_event, stream));
        CHECK_HIP(hipEventSynchronize(stop_event));
        float elapsed_ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start_event, stop_event));
        return elapsed_ms;
    }
};

// ============================================================================
// Matrix Initialization
// ============================================================================

/**
 * Генерация положительно определённой эрмитовой матрицы
 * Метод: A = B * B^H + n*I (гарантированно положительно определённая)
 */
void initialize_positive_definite_hermitian(std::vector<ComplexFloat>& matrix, int n) {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Генерируем случайную матрицу B
    std::vector<ComplexFloat> B(n * n);
    for (int i = 0; i < n * n; ++i) {
        B[i] = make_complex(dis(gen), dis(gen));
    }
    
    // A = B * B^H + n*I
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            ComplexFloat sum = make_complex(0.0f, 0.0f);
            for (int k = 0; k < n; ++k) {
                // sum += B[i,k] * conj(B[j,k])
                sum = complex_add(sum, complex_mul(B[i * n + k], complex_conj(B[j * n + k])));
            }
            if (i == j) {
                // Добавляем n на диагональ для численной стабильности
                sum.x += static_cast<float>(n);
            }
            matrix[i * n + j] = sum;
            matrix[j * n + i] = complex_conj(sum);
        }
    }
}

/**
 * Генерация общей эрмитовой матрицы (для LU/Hybrid методов)
 */
void initialize_hermitian_matrix(std::vector<ComplexFloat>& matrix, int n) {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float real = dis(gen) + 0.5f;
            float imag = (i == j) ? 0.0f : dis(gen);  // Диагональ вещественная
            matrix[i * n + j] = make_complex(real, imag);
            if (i != j) {
                matrix[j * n + i] = complex_conj(matrix[i * n + j]);
            }
        }
    }
}

// ============================================================================
// Validation
// ============================================================================

float compute_frobenius_error(const std::vector<ComplexFloat>& A,
                               const std::vector<ComplexFloat>& A_inv,
                               int n) {
    // Вычисляем ||A * A_inv - I||_F
    std::vector<ComplexFloat> product(n * n);
    
    // Матричное умножение A * A_inv
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ComplexFloat sum = make_complex(0.0f, 0.0f);
            for (int k = 0; k < n; ++k) {
                sum = complex_add(sum, complex_mul(A[i * n + k], A_inv[k * n + j]));
            }
            product[i * n + j] = sum;
        }
    }
    
    // Норма Фробениуса ||A*A_inv - I||
    float error = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ComplexFloat expected = (i == j) ? make_complex(1.0f, 0.0f) : make_complex(0.0f, 0.0f);
            ComplexFloat diff = complex_sub(product[i * n + j], expected);
            error += diff.x * diff.x + diff.y * diff.y;
        }
    }
    
    return std::sqrt(error);
}

// ============================================================================
// METHOD 1: rocSOLVER GETRF + GETRI (LU-based)
// ============================================================================

class LUInverter {
private:
    int n;
    rocblas_handle handle;
    ComplexFloat* d_A;
    rocblas_int* d_ipiv;
    rocblas_int* d_info;

public:
    LUInverter(int size) : n(size), d_A(nullptr), d_ipiv(nullptr), d_info(nullptr) {
        CHECK_ROCBLAS(rocblas_create_handle(&handle));
        CHECK_HIP(hipMalloc(&d_A, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_ipiv, n * sizeof(rocblas_int)));
        CHECK_HIP(hipMalloc(&d_info, sizeof(rocblas_int)));
    }
    
    ~LUInverter() {
        if (d_A) hipFree(d_A);
        if (d_ipiv) hipFree(d_ipiv);
        if (d_info) hipFree(d_info);
        rocblas_destroy_handle(handle);
    }
    
    float invert(const std::vector<ComplexFloat>& A_host,
                 std::vector<ComplexFloat>& A_inv_host) {
        GPUTimer timer;
        
        // Copy to device
        CHECK_HIP(hipMemcpy(d_A, A_host.data(), n * n * sizeof(ComplexFloat), hipMemcpyHostToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        
        // START GPU TIMING
        timer.start();
        
        // LU Factorization
        CHECK_ROCBLAS(rocsolver_cgetrf(handle, n, n, d_A, n, d_ipiv, d_info));
        
        // Matrix Inversion from LU factors
        CHECK_ROCBLAS(rocsolver_cgetri(handle, n, d_A, n, d_ipiv, d_info));
        
        // STOP GPU TIMING
        float gpu_time = timer.stop();
        
        // Copy result back
        CHECK_HIP(hipMemcpy(A_inv_host.data(), d_A, n * n * sizeof(ComplexFloat), hipMemcpyDeviceToHost));
        
        return gpu_time;
    }
};

// ============================================================================
// METHOD 2: Hybrid GETRF + TRSM
// ============================================================================

class HybridInverter {
private:
    int n;
    rocblas_handle handle;
    ComplexFloat* d_A;
    ComplexFloat* d_I;
    rocblas_int* d_ipiv;
    rocblas_int* d_info;

public:
    HybridInverter(int size) : n(size), d_A(nullptr), d_I(nullptr), d_ipiv(nullptr), d_info(nullptr) {
        CHECK_ROCBLAS(rocblas_create_handle(&handle));
        CHECK_HIP(hipMalloc(&d_A, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_I, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_ipiv, n * sizeof(rocblas_int)));
        CHECK_HIP(hipMalloc(&d_info, sizeof(rocblas_int)));
        
        // Инициализируем единичную матрицу на GPU
        std::vector<ComplexFloat> h_I(n * n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                h_I[i * n + j] = (i == j) ? make_complex(1.0f, 0.0f) : make_complex(0.0f, 0.0f);
            }
        }
        CHECK_HIP(hipMemcpy(d_I, h_I.data(), n * n * sizeof(ComplexFloat), hipMemcpyHostToDevice));
    }
    
    ~HybridInverter() {
        if (d_A) hipFree(d_A);
        if (d_I) hipFree(d_I);
        if (d_ipiv) hipFree(d_ipiv);
        if (d_info) hipFree(d_info);
        rocblas_destroy_handle(handle);
    }
    
    float invert(const std::vector<ComplexFloat>& A_host,
                 std::vector<ComplexFloat>& A_inv_host) {
        GPUTimer timer;
        
        // Восстанавливаем единичную матрицу
        std::vector<ComplexFloat> h_I(n * n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                h_I[i * n + j] = (i == j) ? make_complex(1.0f, 0.0f) : make_complex(0.0f, 0.0f);
            }
        }
        
        // Copy to device
        CHECK_HIP(hipMemcpy(d_A, A_host.data(), n * n * sizeof(ComplexFloat), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_I, h_I.data(), n * n * sizeof(ComplexFloat), hipMemcpyHostToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        
        // START GPU TIMING
        timer.start();
        
        // LU Factorization
        CHECK_ROCBLAS(rocsolver_cgetrf(handle, n, n, d_A, n, d_ipiv, d_info));
        
        // Solve L*Y = I (forward substitution)
        ComplexFloat alpha = make_complex(1.0f, 0.0f);
        CHECK_ROCBLAS(rocblas_ctrsm(handle, rocblas_side_left, rocblas_fill_lower,
                                     rocblas_operation_none, rocblas_diagonal_unit,
                                     n, n, &alpha, d_A, n, d_I, n));
        
        // Solve U*X = Y (back substitution)
        CHECK_ROCBLAS(rocblas_ctrsm(handle, rocblas_side_left, rocblas_fill_upper,
                                     rocblas_operation_none, rocblas_diagonal_non_unit,
                                     n, n, &alpha, d_A, n, d_I, n));
        
        // STOP GPU TIMING
        float gpu_time = timer.stop();
        
        // Copy result back
        CHECK_HIP(hipMemcpy(A_inv_host.data(), d_I, n * n * sizeof(ComplexFloat), hipMemcpyDeviceToHost));
        
        return gpu_time;
    }
};

// ============================================================================
// METHOD 3: Cholesky (POTRF + POTRI) - ЛУЧШИЙ для положительно определённых
// ============================================================================

class CholeskyInverter {
private:
    int n;
    rocblas_handle handle;
    ComplexFloat* d_A;
    rocblas_int* d_info;

public:
    CholeskyInverter(int size) : n(size), d_A(nullptr), d_info(nullptr) {
        CHECK_ROCBLAS(rocblas_create_handle(&handle));
        CHECK_HIP(hipMalloc(&d_A, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_info, sizeof(rocblas_int)));
    }
    
    ~CholeskyInverter() {
        if (d_A) hipFree(d_A);
        if (d_info) hipFree(d_info);
        rocblas_destroy_handle(handle);
    }
    
    float invert(const std::vector<ComplexFloat>& A_host,
                 std::vector<ComplexFloat>& A_inv_host) {
        GPUTimer timer;
        
        // Copy to device
        CHECK_HIP(hipMemcpy(d_A, A_host.data(), n * n * sizeof(ComplexFloat), hipMemcpyHostToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        
        // START GPU TIMING
        timer.start();
        
        // Cholesky Factorization: A = L * L^H
        CHECK_ROCBLAS(rocsolver_cpotrf(handle, rocblas_fill_upper, n, d_A, n, d_info));
        
        // Matrix Inversion using Cholesky factors
        CHECK_ROCBLAS(rocsolver_cpotri(handle, rocblas_fill_upper, n, d_A, n, d_info));
        
        // STOP GPU TIMING
        float gpu_time = timer.stop();
        
        // Copy result back (верхний треугольник содержит результат)
        CHECK_HIP(hipMemcpy(A_inv_host.data(), d_A, n * n * sizeof(ComplexFloat), hipMemcpyDeviceToHost));
        
        // Заполняем нижний треугольник (симметрия)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                A_inv_host[i * n + j] = complex_conj(A_inv_host[j * n + i]);
            }
        }
        
        return gpu_time;
    }
};

// ============================================================================
// Statistics Helper
// ============================================================================

struct Statistics {
    float min_ms, max_ms, avg_ms, std_ms;
    
    static Statistics compute(const std::vector<float>& times) {
        Statistics s;
        s.min_ms = *std::min_element(times.begin(), times.end());
        s.max_ms = *std::max_element(times.begin(), times.end());
        s.avg_ms = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
        
        float variance = 0.0f;
        for (float t : times) {
            variance += (t - s.avg_ms) * (t - s.avg_ms);
        }
        s.std_ms = std::sqrt(variance / times.size());
        
        return s;
    }
};

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << std::string(75, '=') << std::endl;
    std::cout << "GPU Matrix Inversion Benchmark: " << MATRIX_SIZE << "×" << MATRIX_SIZE 
              << " Hermitian Matrix" << std::endl;
    std::cout << "Target: < " << TARGET_TIME_MS << " ms on AMD MI100" << std::endl;
    std::cout << std::string(75, '=') << std::endl << std::endl;
    
    // Проверка GPU
    int device_count;
    CHECK_HIP(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No GPU found!" << std::endl;
        return 1;
    }
    
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Compute Units: " << props.multiProcessorCount << std::endl;
    std::cout << "Memory: " << (props.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Матрицы
    std::vector<ComplexFloat> A_hermitian(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<ComplexFloat> A_posdef(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<ComplexFloat> A_inv(MATRIX_SIZE * MATRIX_SIZE);
    
    std::cout << "Initializing matrices..." << std::endl;
    initialize_hermitian_matrix(A_hermitian, MATRIX_SIZE);
    initialize_positive_definite_hermitian(A_posdef, MATRIX_SIZE);
    std::cout << "Matrix size: " << (MATRIX_SIZE * MATRIX_SIZE * sizeof(ComplexFloat) / 1024) << " KB" << std::endl;
    std::cout << std::endl;
    
    // Создаём инвертеры
    LUInverter lu_inv(MATRIX_SIZE);
    HybridInverter hybrid_inv(MATRIX_SIZE);
    CholeskyInverter cholesky_inv(MATRIX_SIZE);
    
    // Timing vectors
    std::vector<float> lu_times, hybrid_times, cholesky_times;
    
    // ========================================================================
    // Warmup
    // ========================================================================
    std::cout << "Warmup (" << WARMUP_ITERATIONS << " iterations)..." << std::endl;
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        lu_inv.invert(A_hermitian, A_inv);
        hybrid_inv.invert(A_hermitian, A_inv);
        cholesky_inv.invert(A_posdef, A_inv);
    }
    std::cout << "Warmup complete." << std::endl << std::endl;
    
    // ========================================================================
    // Benchmarking
    // ========================================================================
    std::cout << "Running " << NUM_ITERATIONS << " benchmark iterations..." << std::endl << std::endl;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::cout << "Iteration " << (iter + 1) << "/" << NUM_ITERATIONS << std::endl;
        
        // LU method
        float lu_time = lu_inv.invert(A_hermitian, A_inv);
        lu_times.push_back(lu_time);
        std::cout << "  LU (GETRF+GETRI):       " << std::fixed << std::setprecision(4) << lu_time << " ms" << std::endl;
        
        // Hybrid method
        float hybrid_time = hybrid_inv.invert(A_hermitian, A_inv);
        hybrid_times.push_back(hybrid_time);
        std::cout << "  Hybrid (GETRF+TRSM):    " << std::fixed << std::setprecision(4) << hybrid_time << " ms" << std::endl;
        
        // Cholesky method (используем положительно определённую матрицу)
        float cholesky_time = cholesky_inv.invert(A_posdef, A_inv);
        cholesky_times.push_back(cholesky_time);
        std::cout << "  Cholesky (POTRF+POTRI): " << std::fixed << std::setprecision(4) << cholesky_time << " ms" << std::endl;
        
        // Validation on first iteration
        if (iter == 0) {
            // Validate LU
            lu_inv.invert(A_hermitian, A_inv);
            float lu_error = compute_frobenius_error(A_hermitian, A_inv, MATRIX_SIZE);
            
            // Validate Cholesky
            cholesky_inv.invert(A_posdef, A_inv);
            float cholesky_error = compute_frobenius_error(A_posdef, A_inv, MATRIX_SIZE);
            
            std::cout << "  Validation:" << std::endl;
            std::cout << "    LU error:       " << std::scientific << lu_error << std::endl;
            std::cout << "    Cholesky error: " << std::scientific << cholesky_error << std::endl;
        }
        std::cout << std::endl;
    }
    
    // ========================================================================
    // Statistics
    // ========================================================================
    std::cout << std::string(75, '=') << std::endl;
    std::cout << "BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(75, '=') << std::endl << std::endl;
    
    auto lu_stats = Statistics::compute(lu_times);
    auto hybrid_stats = Statistics::compute(hybrid_times);
    auto cholesky_stats = Statistics::compute(cholesky_times);
    
    std::cout << std::fixed << std::setprecision(4);
    
    std::cout << "LU (GETRF + GETRI):" << std::endl;
    std::cout << "  Min: " << lu_stats.min_ms << " ms | Max: " << lu_stats.max_ms 
              << " ms | Avg: " << lu_stats.avg_ms << " ms | Std: " << lu_stats.std_ms << " ms" << std::endl;
    std::cout << "  Target (<" << TARGET_TIME_MS << " ms): " 
              << (lu_stats.avg_ms < TARGET_TIME_MS ? "✓ ACHIEVED" : "✗ NOT MET") << std::endl << std::endl;
    
    std::cout << "Hybrid (GETRF + TRSM):" << std::endl;
    std::cout << "  Min: " << hybrid_stats.min_ms << " ms | Max: " << hybrid_stats.max_ms 
              << " ms | Avg: " << hybrid_stats.avg_ms << " ms | Std: " << hybrid_stats.std_ms << " ms" << std::endl;
    std::cout << "  Target (<" << TARGET_TIME_MS << " ms): " 
              << (hybrid_stats.avg_ms < TARGET_TIME_MS ? "✓ ACHIEVED" : "✗ NOT MET") << std::endl << std::endl;
    
    std::cout << "Cholesky (POTRF + POTRI) [BEST for Hermitian]:" << std::endl;
    std::cout << "  Min: " << cholesky_stats.min_ms << " ms | Max: " << cholesky_stats.max_ms 
              << " ms | Avg: " << cholesky_stats.avg_ms << " ms | Std: " << cholesky_stats.std_ms << " ms" << std::endl;
    std::cout << "  Target (<" << TARGET_TIME_MS << " ms): " 
              << (cholesky_stats.avg_ms < TARGET_TIME_MS ? "✓ ACHIEVED" : "✗ NOT MET") << std::endl << std::endl;
    
    // Best method
    float best_time = std::min({lu_stats.avg_ms, hybrid_stats.avg_ms, cholesky_stats.avg_ms});
    std::string best_method = "Unknown";
    if (best_time == cholesky_stats.avg_ms) best_method = "Cholesky (POTRF+POTRI)";
    else if (best_time == hybrid_stats.avg_ms) best_method = "Hybrid (GETRF+TRSM)";
    else best_method = "LU (GETRF+GETRI)";
    
    std::cout << std::string(75, '-') << std::endl;
    std::cout << "BEST METHOD: " << best_method << std::endl;
    std::cout << "BEST TIME:   " << best_time << " ms" << std::endl;
    std::cout << "TARGET:      < " << TARGET_TIME_MS << " ms" << std::endl;
    std::cout << "STATUS:      " << (best_time < TARGET_TIME_MS ? "✓ ACHIEVED!" : "✗ NOT MET") << std::endl;
    std::cout << std::string(75, '=') << std::endl << std::endl;
    
    // ========================================================================
    // Save CSV Results
    // ========================================================================
    std::ofstream csv("benchmark_results.csv");
    csv << "Method,Min_ms,Max_ms,Avg_ms,Std_ms,Target_Met\n";
    csv << "LU_GETRF_GETRI," << lu_stats.min_ms << "," << lu_stats.max_ms << "," 
        << lu_stats.avg_ms << "," << lu_stats.std_ms << "," 
        << (lu_stats.avg_ms < TARGET_TIME_MS ? "Yes" : "No") << "\n";
    csv << "Hybrid_GETRF_TRSM," << hybrid_stats.min_ms << "," << hybrid_stats.max_ms << "," 
        << hybrid_stats.avg_ms << "," << hybrid_stats.std_ms << "," 
        << (hybrid_stats.avg_ms < TARGET_TIME_MS ? "Yes" : "No") << "\n";
    csv << "Cholesky_POTRF_POTRI," << cholesky_stats.min_ms << "," << cholesky_stats.max_ms << "," 
        << cholesky_stats.avg_ms << "," << cholesky_stats.std_ms << "," 
        << (cholesky_stats.avg_ms < TARGET_TIME_MS ? "Yes" : "No") << "\n";
    csv.close();
    
    std::cout << "Results saved to: benchmark_results.csv" << std::endl;
    
    return 0;
}

