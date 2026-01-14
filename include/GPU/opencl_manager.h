#ifndef OPENCL_MANAGER_H
#define OPENCL_MANAGER_H

#include <CL/cl.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <stdexcept>

// Forward declaration (полное определение в gpu_memory_manager.hpp)
namespace gpu {
    class GPUMemoryBuffer;
    enum class MemoryType;
}

namespace gpu {

/**
 * @class OpenCLManager
 * @brief Singleton for unified OpenCL resource management
 * 
 * Thread-safe singleton that manages:
 * - Platform and device selection
 * - OpenCL context and command queue creation
 * - Program compilation with caching (avoid recompilation)
 * - Error handling and resource cleanup
 * 
 * USAGE:
 *   OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
 *   auto& manager = OpenCLManager::GetInstance();
 *   cl_context ctx = manager.GetContext();
 *   cl_program prog = manager.GetOrCompileProgram(source);
 */
class OpenCLManager {
public:
    /**
     * @brief Get Singleton instance (thread-safe)
     * Uses C++11 static local initialization
     */
    static OpenCLManager& GetInstance();

    /**
     * @brief Initialize OpenCL (must be called once before using GeneratorGPU)
     * @param device_type CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU, etc.
     * @throws std::runtime_error if initialization fails
     */
    static void Initialize(cl_device_type device_type = CL_DEVICE_TYPE_GPU);

    /**
     * @brief Cleanup OpenCL resources (optional, automatic in destructor)
     */
    static void Cleanup();

    /**
     * @brief Check if manager is initialized
     */
    bool IsInitialized() const { return initialized_; }

    // ═══════════════════════════════════════════════════════════════
    // RESOURCE GETTERS
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Get OpenCL context
     */
    cl_context GetContext() const {
        if (!initialized_)
            throw std::runtime_error("OpenCLManager not initialized");
        return context_;
    }

    /**
     * @brief Get command queue
     */
    cl_command_queue GetQueue() const {
        if (!initialized_)
            throw std::runtime_error("OpenCLManager not initialized");
        return queue_;
    }

    /**
     * @brief Get device ID
     */
    cl_device_id GetDevice() const {
        if (!initialized_)
            throw std::runtime_error("OpenCLManager not initialized");
        return device_;
    }

    /**
     * @brief Get platform ID
     */
    cl_platform_id GetPlatform() const {
        if (!initialized_)
            throw std::runtime_error("OpenCLManager not initialized");
        return platform_;
    }

    // ═══════════════════════════════════════════════════════════════
    // PROGRAM COMPILATION WITH CACHE
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Get or compile OpenCL program (with caching)
     * 
     * If program with same source already compiled, returns from cache.
     * Avoids expensive recompilation.
     * 
     * @param source Kernel source code
     * @return cl_program (do NOT release, managed by OpenCLManager)
     * @throws std::runtime_error if compilation fails
     */
    cl_program GetOrCompileProgram(const std::string& source);

    /**
     * @brief Get cache statistics
     */
    std::string GetCacheStatistics() const;

    // ═══════════════════════════════════════════════════════════════
    // DEVICE INFORMATION
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Get device information (name, vendor, memory, etc)
     */
    std::string GetDeviceInfo() const;

    // ═══════════════════════════════════════════════════════════════
    // GPU MEMORY MANAGEMENT
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Create new GPU buffer (OWNING)
     * @param num_elements Number of complex<float> elements
     * @param type Memory type (READ_ONLY, WRITE_ONLY, READ_WRITE)
     * @return unique_ptr to GPUMemoryBuffer
     */
    std::unique_ptr<GPUMemoryBuffer> CreateBuffer(
        size_t num_elements,
        MemoryType type
    );

    /**
     * @brief Wrap external GPU buffer (NON-OWNING)
     * 
     * Automatically validates that external buffer belongs to correct context.
     * Throws if context mismatch detected.
     * 
     * @param external_gpu_buffer Existing cl_mem buffer
     * @param num_elements Number of complex<float> elements
     * @param type Memory type
     * @return unique_ptr to GPUMemoryBuffer wrapper
     * @throws std::runtime_error if context mismatch
     */
    std::unique_ptr<GPUMemoryBuffer> WrapExternalBuffer(
        cl_mem external_gpu_buffer,
        size_t num_elements,
        MemoryType type
    );

    /**
     * @brief Register buffer for reuse (by name)
     * @param name Unique name for buffer
     * @param buffer Shared pointer to buffer (will be stored as weak_ptr)
     */
    void RegisterBuffer(
        const std::string& name,
        std::shared_ptr<GPUMemoryBuffer> buffer
    );

    /**
     * @brief Get registered buffer by name
     * @param name Buffer name
     * @return shared_ptr to buffer, or nullptr if not found/expired
     */
    std::shared_ptr<GPUMemoryBuffer> GetBuffer(const std::string& name);

    /**
     * @brief Get or create buffer (creates if not exists, returns if exists)
     * @param name Buffer name
     * @param num_elements Number of elements (used only if creating new)
     * @param type Memory type (used only if creating new)
     * @return shared_ptr to buffer
     */
    std::shared_ptr<GPUMemoryBuffer> GetOrCreateBuffer(
        const std::string& name,
        size_t num_elements,
        MemoryType type
    );

    /**
     * @brief Get memory management statistics
     */
    void PrintMemoryStatistics() const;

    // ═══════════════════════════════════════════════════════════════
    // DESTRUCTOR
    // ═══════════════════════════════════════════════════════════════

    ~OpenCLManager();

    // Prevent copy/move
    OpenCLManager(const OpenCLManager&) = delete;
    OpenCLManager& operator=(const OpenCLManager&) = delete;
    OpenCLManager(OpenCLManager&&) = delete;
    OpenCLManager& operator=(OpenCLManager&&) = delete;

private:
    // ═══════════════════════════════════════════════════════════════
    // PRIVATE MEMBERS
    // ═══════════════════════════════════════════════════════════════

    bool initialized_{false};

    // OpenCL resources
    cl_platform_id platform_{nullptr};
    cl_device_id device_{nullptr};
    cl_context context_{nullptr};
    cl_command_queue queue_{nullptr};

    // Program cache: source_hash -> cl_program
    std::unordered_map<std::string, cl_program> program_cache_;
    mutable std::mutex cache_mutex_;

    // Cache statistics
    size_t cache_hits_{0};
    size_t cache_misses_{0};

    // ═══════════════════════════════════════════════════════════════
    // GPU MEMORY MANAGEMENT (PRIVATE)
    // ═══════════════════════════════════════════════════════════════

    // Buffer registry for reuse
    std::unordered_map<std::string, std::weak_ptr<GPUMemoryBuffer>> buffer_registry_;
    mutable std::mutex registry_mutex_;

    // Memory statistics
    size_t total_allocated_bytes_{0};
    size_t num_buffers_{0};

    // ═══════════════════════════════════════════════════════════════
    // PRIVATE METHODS
    // ═══════════════════════════════════════════════════════════════

    OpenCLManager() = default;

    /**
     * @brief Initialize OpenCL internals
     */
    void InitializeOpenCL(cl_device_type device_type);

    /**
     * @brief Compile OpenCL program
     */
    cl_program CompileProgram(const std::string& source);

    /**
     * @brief Release all OpenCL resources
     */
    void ReleaseResources();

    /**
     * @brief Validate external buffer context (helper)
     * @throws std::runtime_error if context mismatch
     */
    void ValidateBufferContext(cl_mem external_buffer) const;
};

} // namespace gpu


#endif // OPENCL_MANAGER_H
