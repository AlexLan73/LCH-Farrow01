#ifndef GENERATOR_GPU_REFACTORED_H
#define GENERATOR_GPU_REFACTORED_H

#include "opencl_manager.h"
#include <complex>
#include <vector>
#include <cstdint>

namespace radar {

// LFM signal parameters
struct LFMParameters {
    float f_start{0.0f};        // Start frequency (Hz)
    float f_stop{0.0f};         // Stop frequency (Hz)
    float sample_rate{12.0e6f}; // Sample rate (Hz)
    float duration{0.001f};     // Duration (seconds)
    uint32_t num_beams{256};    // Number of beams
};

namespace gpu {

/**
 * @class GeneratorGPU
 * @brief GPU-based LFM signal generator using Singleton OpenCLManager
 * 
 * REFACTORED to use OpenCLManager singleton instead of managing
 * OpenCL resources directly. This eliminates:
 * - Duplicate platform/device/context initialization
 * - Memory waste from multiple contexts
 * - Redundant kernel compilation
 * 
 * USAGE:
 *   // Initialize manager once in main()
 *   OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
 *   
 *   // Create generators - they all share the same context!
 *   LFMParameters params{...};
 *   GeneratorGPU gen1(params);
 *   GeneratorGPU gen2(params);
 *   
 *   // Use as normal
 *   cl_mem signal = gen1.signal_base();
 */
class GeneratorGPU {
public:
    /**
     * @brief Constructor - initializes GPU kernels
     * @param params LFM signal parameters
     * @throws std::runtime_error if OpenCLManager not initialized
     * @throws std::invalid_argument if parameters invalid
     */
    explicit GeneratorGPU(const LFMParameters& params);

    /**
     * @brief Destructor - releases GPU memory
     */
    ~GeneratorGPU();

    // Prevent copy/move (contains non-copyable OpenCL resources)
    GeneratorGPU(const GeneratorGPU&) = delete;
    GeneratorGPU& operator=(const GeneratorGPU&) = delete;
    GeneratorGPU(GeneratorGPU&&) = delete;
    GeneratorGPU& operator=(GeneratorGPU&&) = delete;

    // ═══════════════════════════════════════════════════════════════
    // SIGNAL GENERATION
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Generate base LFM signal on GPU
     * @return cl_mem GPU memory object (do NOT release)
     */
    cl_mem signal_base();

    /**
     * @brief Generate LFM signal with delay
     * @param delay_ms Delay in milliseconds
     * @return cl_mem GPU memory object (do NOT release)
     */
    cl_mem signal_delayed(float delay_ms);

    // ═══════════════════════════════════════════════════════════════
    // INFORMATION GETTERS
    // ═══════════════════════════════════════════════════════════════

    uint32_t GetNumBeams() const { return params_.num_beams; }
    uint32_t GetNumSamples() const { return num_samples_; }
    uint32_t GetTotalSize() const { return params_.num_beams * num_samples_; }
    size_t GetMemorySizeBytes() const {
        return GetTotalSize() * sizeof(std::complex<float>);
    }

    cl_context GetContext() const {
        return manager_.GetContext();
    }

private:
    // ═══════════════════════════════════════════════════════════════
    // PRIVATE MEMBERS
    // ═══════════════════════════════════════════════════════════════

    LFMParameters params_;
    OpenCLManager& manager_;

    uint32_t num_samples_{0};

    // OpenCL kernels
    cl_kernel kernel_lfm_basic_{nullptr};
    cl_kernel kernel_lfm_delayed_{nullptr};

    // GPU memory buffers
    cl_mem gpu_signal_base_{nullptr};
    cl_mem gpu_signal_delayed_{nullptr};

    // ═══════════════════════════════════════════════════════════════
    // PRIVATE METHODS
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Validate LFM parameters
     */
    void ValidateParameters();

    /**
     * @brief Allocate GPU buffers
     */
    void AllocateGPUMemory();

    /**
     * @brief Compile OpenCL kernels (from Manager cache)
     */
    void CompileKernels();

    /**
     * @brief Release GPU memory
     */
    void ReleaseGPUMemory();

    /**
     * @brief Get kernel source code
     */
    static std::string GetLFMKernelSource();
};

} // namespace gpu
} // namespace radar

#endif // GENERATOR_GPU_REFACTORED_H
