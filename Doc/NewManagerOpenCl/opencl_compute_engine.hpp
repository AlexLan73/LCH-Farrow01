#pragma once

#include "opencl_core.hpp"
#include "kernel_program.hpp"
#include <CL/cl.h>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <array>

namespace gpu {

// ════════════════════════════════════════════════════════════════════════════
// Enum для типов памяти
// ════════════════════════════════════════════════════════════════════════════

enum class MemoryType {
    GPU_READ_ONLY,
    GPU_WRITE_ONLY,
    GPU_READ_WRITE
};

// ════════════════════════════════════════════════════════════════════════════
// GPUMemoryBuffer - УЛУЧШЕННАЯ версия с асинхронностью
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class GPUMemoryBuffer
 * @brief RAII-обёртка над GPU памятью с поддержкой асинхронного чтения/записи
 *
 * Особенности:
 * - Владение памятью (OWNING) или использование внешней (NON-OWNING)
 * - Pinned host buffers для быстрого DMA
 * - Асинхронное чтение/запись с cl_event
 * - Синхронное и асинхронное API
 * - Move семантика, Copy запрещена
 *
 * Использование:
 * ```cpp
 * auto buffer = engine.CreateBuffer(1024, MemoryType::GPU_READ_WRITE);
 * buffer->WriteToGPU(data);
 * auto result = buffer->ReadFromGPU();
 * ```
 */
class GPUMemoryBuffer {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструкторы
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief OWNING - создаёт новый буфер на GPU
     */
    GPUMemoryBuffer(
        cl_context context,
        cl_command_queue queue,
        size_t num_elements,
        MemoryType type = MemoryType::GPU_READ_WRITE
    );

    /**
     * @brief NON-OWNING - использует готовый буфер
     */
    GPUMemoryBuffer(
        cl_context context,
        cl_command_queue queue,
        cl_mem external_gpu_buffer,
        size_t num_elements,
        MemoryType type = MemoryType::GPU_READ_WRITE
    );

    /**
     * @brief OWNING с данными - создаёт буфер и копирует данные
     */
    GPUMemoryBuffer(
        cl_context context,
        cl_command_queue queue,
        const void* host_data,
        size_t data_size_bytes,
        size_t num_elements,
        MemoryType type = MemoryType::GPU_READ_ONLY
    );

    // ═══════════════════════════════════════════════════════════════
    // Деструктор и управление ресурсами
    // ═══════════════════════════════════════════════════════════════

    ~GPUMemoryBuffer();

    // Move семантика
    GPUMemoryBuffer(GPUMemoryBuffer&& other) noexcept;
    GPUMemoryBuffer& operator=(GPUMemoryBuffer&& other) noexcept;

    // Copy запрещена
    GPUMemoryBuffer(const GPUMemoryBuffer&) = delete;
    GPUMemoryBuffer& operator=(const GPUMemoryBuffer&) = delete;

    // ═══════════════════════════════════════════════════════════════
    // Синхронные операции
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Синхронно прочитать все данные с GPU
     */
    std::vector<std::complex<float>> ReadFromGPU();

    /**
     * @brief Синхронно прочитать часть данных с GPU
     */
    std::vector<std::complex<float>> ReadPartial(size_t num_elements);

    /**
     * @brief Синхронно записать данные на GPU
     */
    void WriteToGPU(const std::vector<std::complex<float>>& data);

    // ═══════════════════════════════════════════════════════════════
    // Асинхронные операции (с cl_event)
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Асинхронно прочитать все данные с GPU
     * @return Пара (результат_вектор, event)
     */
    std::pair<std::vector<std::complex<float>>, cl_event>
    ReadFromGPUAsync();

    /**
     * @brief Асинхронно записать данные на GPU
     * @return cl_event для синхронизации
     */
    cl_event WriteToGPUAsync(const std::vector<std::complex<float>>& data);

    // ═══════════════════════════════════════════════════════════════
    // Информация о буфере
    // ═══════════════════════════════════════════════════════════════

    size_t GetNumElements() const { return num_elements_; }
    size_t GetSizeBytes() const { return num_elements_ * sizeof(std::complex<float>); }
    bool IsExternalBuffer() const { return is_external_buffer_; }
    bool IsGPUDirty() const { return gpu_dirty_; }
    MemoryType GetMemoryType() const { return type_; }
    cl_mem Get() const { return gpu_buffer_; }

    void PrintStats() const;

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════

    cl_context context_;
    cl_command_queue queue_;
    cl_mem gpu_buffer_;
    std::vector<std::complex<float>> pinned_host_buffer_;
    size_t num_elements_;
    MemoryType type_;
    bool is_external_buffer_;  // Флаг владения
    bool gpu_dirty_;           // Нужно ли читать

    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════

    void AllocateGPUBuffer();
    void AllocatePinnedHostBuffer();
    void ReleasePinnedHostBuffer();
};

// ════════════════════════════════════════════════════════════════════════════
// OpenCLComputeEngine - ГЛАВНЫЙ ФАСАД
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class OpenCLComputeEngine
 * @brief Главный фасад для работы с OpenCL
 *
 * Это единственный класс, который пользователь обычно использует.
 * Он объединяет все компоненты:
 * - OpenCLCore (контекст)
 * - KernelProgram (программы + kernels)
 * - GPUMemoryBuffer (память)
 *
 * Использование:
 * ```cpp
 * // 1. Инициализация
 * gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
 * auto& engine = gpu::OpenCLComputeEngine::GetInstance();
 *
 * // 2. Создать программу и kernel
 * auto program = engine.LoadProgram(kernel_source);
 * auto kernel = engine.GetKernel(program, "my_kernel");
 *
 * // 3. Создать буферы
 * auto input = engine.CreateBuffer(1024, gpu::MemoryType::GPU_READ_WRITE);
 * auto output = engine.CreateBuffer(1024, gpu::MemoryType::GPU_WRITE_ONLY);
 *
 * // 4. Загрузить данные
 * input->WriteToGPU(my_data);
 *
 * // 5. Выполнить kernel
 * engine.ExecuteKernel(kernel, {input.get(), output.get()},
 *                      {{1024, 1, 1}}, {{256, 1, 1}});
 *
 * // 6. Прочитать результаты
 * auto result = output->ReadFromGPU();
 *
 * // 7. Статистика
 * std::cout << engine.GetStatistics();
 * ```
 */
class OpenCLComputeEngine {
public:
    // ═══════════════════════════════════════════════════════════════
    // Singleton
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Инициализировать OpenCL (один раз)
     */
    static void Initialize(DeviceType device_type = DeviceType::GPU);

    /**
     * @brief Получить Singleton
     */
    static OpenCLComputeEngine& GetInstance();

    /**
     * @brief Проверить инициализацию
     */
    static bool IsInitialized();

    /**
     * @brief Очистка (опционально)
     */
    static void Cleanup();

    // ═══════════════════════════════════════════════════════════════
    // Программы и kernels
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Загрузить OpenCL программу (компилируется один раз благодаря кэшу)
     * @param source OpenCL C код
     * @return Shared pointer на KernelProgram (не удалять вручную)
     */
    std::shared_ptr<KernelProgram> LoadProgram(const std::string& source);

    /**
     * @brief Получить kernel из программы
     * @param program Программа (из LoadProgram)
     * @param kernel_name Имя kernel в программе
     * @return cl_kernel (управляется программой)
     */
    cl_kernel GetKernel(
        const std::shared_ptr<KernelProgram>& program,
        const std::string& kernel_name
    );

    // ═══════════════════════════════════════════════════════════════
    // Память
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Создать GPU буфер
     */
    std::unique_ptr<GPUMemoryBuffer> CreateBuffer(
        size_t num_elements,
        MemoryType type = MemoryType::GPU_READ_WRITE
    );

    /**
     * @brief Создать GPU буфер с начальными данными
     */
    std::unique_ptr<GPUMemoryBuffer> CreateBufferWithData(
        const std::vector<std::complex<float>>& data,
        MemoryType type = MemoryType::GPU_READ_ONLY
    );

    // ═══════════════════════════════════════════════════════════════
    // Выполнение kernels
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Выполнить kernel синхронно (ждёт завершения)
     * @param kernel Kernel для выполнения
     * @param buffers Буферы (в порядке аргументов kernel)
     * @param global_work_size Глобальные размеры работы
     * @param local_work_size Локальные размеры работы
     */
    void ExecuteKernel(
        cl_kernel kernel,
        const std::vector<cl_mem>& buffers,
        const std::array<size_t, 3>& global_work_size,
        const std::array<size_t, 3>& local_work_size
    );

    /**
     * @brief Выполнить kernel асинхронно (возвращает event)
     * @return cl_event для синхронизации
     */
    cl_event ExecuteKernelAsync(
        cl_kernel kernel,
        const std::vector<cl_mem>& buffers,
        const std::array<size_t, 3>& global_work_size,
        const std::array<size_t, 3>& local_work_size
    );

    // ═══════════════════════════════════════════════════════════════
    // Синхронизация
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Ждать завершения всех операций в queue
     */
    void Flush();

    /**
     * @brief Ждать полного завершения queue
     */
    void Finish();

    /**
     * @brief Ждать завершения конкретного события
     */
    void WaitForEvent(cl_event event);

    /**
     * @brief Ждать завершения нескольких событий
     */
    void WaitForEvents(const std::vector<cl_event>& events);

    // ═══════════════════════════════════════════════════════════════
    // Информация и статистика
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Получить подробную статистику
     */
    std::string GetStatistics() const;

    /**
     * @brief Получить информацию о девайсе
     */
    std::string GetDeviceInfo() const;

    /**
     * @brief Получить кэш статистику
     */
    std::string GetCacheStatistics() const;

    // ═══════════════════════════════════════════════════════════════
    // Деструктор
    // ═══════════════════════════════════════════════════════════════

    ~OpenCLComputeEngine();

    OpenCLComputeEngine(const OpenCLComputeEngine&) = delete;
    OpenCLComputeEngine& operator=(const OpenCLComputeEngine&) = delete;

private:
    // ═══════════════════════════════════════════════════════════════
    // Singleton реализация
    // ═══════════════════════════════════════════════════════════════

    OpenCLComputeEngine();

    static std::unique_ptr<OpenCLComputeEngine> instance_;
    static bool initialized_;
    static std::mutex initialization_mutex_;

    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════

    size_t total_allocated_bytes_;
    size_t num_buffers_;
    size_t kernel_executions_;
};

}  // namespace gpu
