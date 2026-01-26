#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hipblas/hipblas.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <complex>

// ============================================================================
// Configuration
// ============================================================================

const int MATRIX_SIZE = 341;
const int NUM_ITERATIONS = 10;
const bool ENABLE_TIMING = true;
const bool ENABLE_VALIDATION = true;

// ============================================================================
// Utility Functions
// ============================================================================

using ComplexFloat = std::complex<float>;

// Check HIP errors
#define CHECK_HIP(status) \
    if (status != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(status) << std::endl; \
        exit(1); \
    }

// Check rocSOLVER errors
#define CHECK_ROCSOLVER(status) \
    if (status != rocblas_status_success) { \
        std::cerr << "rocSOLVER Error: " << status << std::endl; \
        exit(1); \
    }

// Timing helper
struct Timer {
    std::chrono::high_resolution_clock::time_point start, end;
    
    void tic() {
        start = std::chrono::high_resolution_clock::now();
    }
    
    float toc() {
        end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0f; // Return milliseconds
    }
};

// ============================================================================
// Matrix Initialization and Validation
// ============================================================================

void initialize_complex_symmetric_matrix(std::vector<ComplexFloat>& matrix, int n) {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float real = dis(gen) + 0.5f;  // Add 0.5 for better conditioning
            float imag = dis(gen);
            matrix[i * n + j] = ComplexFloat(real, imag);
            
            // Make symmetric: A[j,i] = conj(A[i,j])
            if (i != j) {
                matrix[j * n + i] = std::conj(matrix[i * n + j]);
            }
        }
    }
}

float compute_frobenius_error(const std::vector<ComplexFloat>& A, 
                              const std::vector<ComplexFloat>& A_inv,
                              int n) {
    // Compute: ||A * A_inv - I||_F
    std::vector<ComplexFloat> product(n * n, ComplexFloat(0, 0));
    
    // Simple matrix multiplication A * A_inv
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                product[i * n + j] += A[i * n + k] * A_inv[k * n + j];
            }
        }
    }
    
    // Compute Frobenius norm of (A*A_inv - I)
    float error = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ComplexFloat expected = (i == j) ? ComplexFloat(1, 0) : ComplexFloat(0, 0);
            ComplexFloat diff = product[i * n + j] - expected;
            error += std::abs(diff) * std::abs(diff);
        }
    }
    
    return std::sqrt(error);
}

// ============================================================================
// METHOD 1: rocSOLVER GETRI (Native LU-based Inversion)
// ============================================================================

class RocSOLVERInverter {
private:
    int n;
    rocblas_handle handle;
    
    ComplexFloat* d_A = nullptr;
    ComplexFloat* d_A_inv = nullptr;
    int* d_ipiv = nullptr;
    int* d_info = nullptr;
    
    float* d_work = nullptr;
    int work_size = 0;
    
public:
    RocSOLVERInverter(int size) : n(size) {
        CHECK_ROCSOLVER(rocblas_create_handle(&handle));
        
        // Allocate device memory
        CHECK_HIP(hipMalloc(&d_A, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_A_inv, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_ipiv, n * sizeof(int)));
        CHECK_HIP(hipMalloc(&d_info, sizeof(int)));
        
        // Query work size
        rocsolver_cgetri(handle, n, d_A, n, d_ipiv, d_work, -1, d_info);
        // For simplicity, allocate fixed work buffer
        work_size = n * n;
        CHECK_HIP(hipMalloc(&d_work, work_size * sizeof(float)));
    }
    
    ~RocSOLVERInverter() {
        if (d_A) hipFree(d_A);
        if (d_A_inv) hipFree(d_A_inv);
        if (d_ipiv) hipFree(d_ipiv);
        if (d_info) hipFree(d_info);
        if (d_work) hipFree(d_work);
        rocblas_destroy_handle(handle);
    }
    
    void invert(const std::vector<ComplexFloat>& A_host, 
                std::vector<ComplexFloat>& A_inv_host) {
        Timer timer;
        
        // Copy to device
        timer.tic();
        CHECK_HIP(hipMemcpy(d_A, A_host.data(), n * n * sizeof(ComplexFloat), hipMemcpyHostToDevice));
        float transfer_in_time = timer.toc();
        
        // LU Factorization
        timer.tic();
        CHECK_ROCSOLVER(rocsolver_cgetrf(handle, n, n, d_A, n, d_ipiv, d_info));
        float getrf_time = timer.toc();
        
        // Matrix Inversion (using LU factors)
        timer.tic();
        CHECK_ROCSOLVER(rocsolver_cgetri(handle, n, d_A, n, d_ipiv, d_work, work_size, d_info));
        float getri_time = timer.toc();
        
        // Copy result back
        timer.tic();
        CHECK_HIP(hipMemcpy(d_A_inv, d_A, n * n * sizeof(ComplexFloat), hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(A_inv_host.data(), d_A_inv, n * n * sizeof(ComplexFloat), hipMemcpyDeviceToHost));
        float transfer_out_time = timer.toc();
        
        std::cout << "  rocSOLVER Results:" << std::endl;
        std::cout << "    GETRF time:     " << std::fixed << std::setprecision(4) << getrf_time << " ms" << std::endl;
        std::cout << "    GETRI time:     " << std::fixed << std::setprecision(4) << getri_time << " ms" << std::endl;
        std::cout << "    Total GPU time: " << std::fixed << std::setprecision(4) << (getrf_time + getri_time) << " ms" << std::endl;
    }
};

// ============================================================================
// METHOD 2: Hybrid Approach (rocSOLVER GETRF + rocBLAS TRSM)
// ============================================================================

class HybridInverter {
private:
    int n;
    rocblas_handle handle;
    
    ComplexFloat* d_A = nullptr;
    ComplexFloat* d_I = nullptr;
    ComplexFloat* d_L = nullptr;
    ComplexFloat* d_U = nullptr;
    int* d_ipiv = nullptr;
    int* d_info = nullptr;
    
public:
    HybridInverter(int size) : n(size) {
        CHECK_ROCSOLVER(rocblas_create_handle(&handle));
        
        CHECK_HIP(hipMalloc(&d_A, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_I, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_L, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_U, n * n * sizeof(ComplexFloat)));
        CHECK_HIP(hipMalloc(&d_ipiv, n * sizeof(int)));
        CHECK_HIP(hipMalloc(&d_info, sizeof(int)));
    }
    
    ~HybridInverter() {
        if (d_A) hipFree(d_A);
        if (d_I) hipFree(d_I);
        if (d_L) hipFree(d_L);
        if (d_U) hipFree(d_U);
        if (d_ipiv) hipFree(d_ipiv);
        if (d_info) hipFree(d_info);
        rocblas_destroy_handle(handle);
    }
    
    void invert(const std::vector<ComplexFloat>& A_host,
                std::vector<ComplexFloat>& A_inv_host) {
        Timer timer;
        
        // Copy A to device
        timer.tic();
        CHECK_HIP(hipMemcpy(d_A, A_host.data(), n * n * sizeof(ComplexFloat), hipMemcpyHostToDevice));
        float transfer_time = timer.toc();
        
        // LU Factorization
        timer.tic();
        CHECK_ROCSOLVER(rocsolver_cgetrf(handle, n, n, d_A, n, d_ipiv, d_info));
        CHECK_HIP(hipDeviceSynchronize());
        float getrf_time = timer.toc();
        
        // Create identity matrix on device
        timer.tic();
        // Initialize identity matrix (simple kernel call)
        ComplexFloat* h_I = new ComplexFloat[n * n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                h_I[i * n + j] = (i == j) ? ComplexFloat(1, 0) : ComplexFloat(0, 0);
            }
        }
        CHECK_HIP(hipMemcpy(d_I, h_I, n * n * sizeof(ComplexFloat), hipMemcpyHostToDevice));
        float init_time = timer.toc();
        delete[] h_I;
        
        // Triangular solve: L*U*X = I
        // Solve L*Y = I
        timer.tic();
        rocblas_int one = 1;
        ComplexFloat alpha = ComplexFloat(1, 0);
        CHECK_ROCSOLVER(rocblas_ctrsm(handle, rocblas_side_left, rocblas_fill_lower,
                                      rocblas_operation_none, rocblas_diagonal_unit,
                                      n, n, &alpha, d_A, n, d_I, n));
        CHECK_HIP(hipDeviceSynchronize());
        float trsm_l_time = timer.toc();
        
        // Solve U*X = Y
        timer.tic();
        CHECK_ROCSOLVER(rocblas_ctrsm(handle, rocblas_side_left, rocblas_fill_upper,
                                      rocblas_operation_none, rocblas_diagonal_non_unit,
                                      n, n, &alpha, d_A, n, d_I, n));
        CHECK_HIP(hipDeviceSynchronize());
        float trsm_u_time = timer.toc();
        
        // Copy result
        timer.tic();
        CHECK_HIP(hipMemcpy(A_inv_host.data(), d_I, n * n * sizeof(ComplexFloat), hipMemcpyDeviceToHost));
        float transfer_result_time = timer.toc();
        
        std::cout << "  Hybrid Approach Results:" << std::endl;
        std::cout << "    GETRF time:          " << std::fixed << std::setprecision(4) << getrf_time << " ms" << std::endl;
        std::cout << "    TRSM (L) time:       " << std::fixed << std::setprecision(4) << trsm_l_time << " ms" << std::endl;
        std::cout << "    TRSM (U) time:       " << std::fixed << std::setprecision(4) << trsm_u_time << " ms" << std::endl;
        std::cout << "    Total GPU time:      " << std::fixed << std::setprecision(4) 
                  << (getrf_time + trsm_l_time + trsm_u_time) << " ms" << std::endl;
    }
};

// ============================================================================
// Main Benchmarking Function
// ============================================================================

int main() {
    std::cout << "=" << std::string(70, '=') << std::endl;
    std::cout << "GPU Matrix Inversion Profiling: 341x341 Complex Symmetric Matrix" << std::endl;
    std::cout << "=" << std::string(70, '=') << std::endl;
    std::cout << std::endl;
    
    // Initialize host matrices
    std::vector<ComplexFloat> A_host(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<ComplexFloat> A_inv_rocsolver(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<ComplexFloat> A_inv_hybrid(MATRIX_SIZE * MATRIX_SIZE);
    
    std::cout << "Initializing complex symmetric matrix (341x341)..." << std::endl;
    initialize_complex_symmetric_matrix(A_host, MATRIX_SIZE);
    std::cout << "Matrix initialized." << std::endl << std::endl;
    
    // Create inverters
    RocSOLVERInverter rocsolver_inv(MATRIX_SIZE);
    HybridInverter hybrid_inv(MATRIX_SIZE);
    
    // Timing accumulators
    std::vector<float> rocsolver_times;
    std::vector<float> hybrid_times;
    
    std::cout << "Running " << NUM_ITERATIONS << " iterations for profiling..." << std::endl << std::endl;
    
    // Main profiling loop
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::cout << "Iteration " << (iter + 1) << "/" << NUM_ITERATIONS << std::endl;
        
        Timer iter_timer;
        
        // rocSOLVER approach
        iter_timer.tic();
        rocsolver_inv.invert(A_host, A_inv_rocsolver);
        float rocsolver_iter_time = iter_timer.toc();
        rocsolver_times.push_back(rocsolver_iter_time);
        
        // Hybrid approach
        iter_timer.tic();
        hybrid_inv.invert(A_host, A_inv_hybrid);
        float hybrid_iter_time = iter_timer.toc();
        hybrid_times.push_back(hybrid_iter_time);
        
        // Validation
        if (ENABLE_VALIDATION && iter == 0) {
            float rocsolver_error = compute_frobenius_error(A_host, A_inv_rocsolver, MATRIX_SIZE);
            float hybrid_error = compute_frobenius_error(A_host, A_inv_hybrid, MATRIX_SIZE);
            
            std::cout << "  Validation (Iteration 1):" << std::endl;
            std::cout << "    rocSOLVER error:  " << std::scientific << rocsolver_error << std::endl;
            std::cout << "    Hybrid error:     " << std::scientific << hybrid_error << std::endl;
        }
        std::cout << std::endl;
    }
    
    // ========================================================================
    // Statistics and Report Generation
    // ========================================================================
    
    std::cout << "=" << std::string(70, '=') << std::endl;
    std::cout << "PROFILING STATISTICS" << std::endl;
    std::cout << "=" << std::string(70, '=') << std::endl << std::endl;
    
    // Compute statistics
    auto compute_stats = [](const std::vector<float>& times) {
        float min = *std::min_element(times.begin(), times.end());
        float max = *std::max_element(times.begin(), times.end());
        float avg = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
        return std::make_tuple(min, max, avg);
    };
    
    auto [ros_min, ros_max, ros_avg] = compute_stats(rocsolver_times);
    auto [hyb_min, hyb_max, hyb_avg] = compute_stats(hybrid_times);
    
    std::cout << "rocSOLVER Approach:" << std::endl;
    std::cout << "  Min time:  " << std::fixed << std::setprecision(4) << ros_min << " ms" << std::endl;
    std::cout << "  Max time:  " << std::fixed << std::setprecision(4) << ros_max << " ms" << std::endl;
    std::cout << "  Avg time:  " << std::fixed << std::setprecision(4) << ros_avg << " ms" << std::endl << std::endl;
    
    std::cout << "Hybrid Approach (GETRF + TRSM):" << std::endl;
    std::cout << "  Min time:  " << std::fixed << std::setprecision(4) << hyb_min << " ms" << std::endl;
    std::cout << "  Max time:  " << std::fixed << std::setprecision(4) << hyb_max << " ms" << std::endl;
    std::cout << "  Avg time:  " << std::fixed << std::setprecision(4) << hyb_avg << " ms" << std::endl << std::endl;
    
    float speedup = ros_avg / hyb_avg;
    std::cout << "Speedup (Hybrid / rocSOLVER): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "Target (<5 ms): " << (hyb_avg < 5.0 ? "✓ ACHIEVED" : "✗ NOT MET") << std::endl << std::endl;
    
    // Save CSV report
    std::ofstream csv_file("profiling_results.csv");
    csv_file << "Implementation,Min_ms,Max_ms,Avg_ms\n";
    csv_file << "rocSOLVER," << ros_min << "," << ros_max << "," << ros_avg << "\n";
    csv_file << "Hybrid," << hyb_min << "," << hyb_max << "," << hyb_avg << "\n";
    csv_file.close();
    
    std::cout << "Results saved to: profiling_results.csv" << std::endl;
    
    return 0;
}
