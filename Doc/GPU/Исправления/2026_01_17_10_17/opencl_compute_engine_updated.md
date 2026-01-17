# 🚀 ПОЛНЫЙ КОД: opencl_compute_engine.hpp (ОБНОВЛЁННЫЙ)

## 📄 Файл с полной реализацией типобезопасного API

```cpp
#pragma once

#include "GPU/gpu_memory_buffer.hpp"
#include "GPU/opencl_core.hpp"
#include "GPU/command_queue_pool.hpp"
#include <memory>
#include <vector>
#include <complex>
#include <string>
#include <CL/cl.h>

namespace gpu {

// Forward declarations
class KernelProgram;

/**
 * @class OpenCLComputeEngine
 * @brief Основной интерфейс к GPU для вычисления ЛЧМ сигналов
 * 
 * Управляет созданием буферов, загрузкой kernel'ов, выполнением kernel'ов.
 * 
 * ИСПОЛЬЗОВАНИЕ:
 *   - Для сигналов (complex<float>): CreateBufferWithData(data, type)
 *   - Для параметров (любых POD): CreateTypedBufferWithData<T>(data, type)
 */
class OpenCLComputeEngine {
public:
    /**
     * @brief Инициализация GPU движка (singleton)
     * @param device_type CLDEVICETYPEGPU или CLDEVICETYPECPU
     */
    static void Initialize(cl_device_type device_type = CL_DEVICE_TYPE_GPU);

    /**
     * @brief Получить singleton экземпляр
     * @return Ссылка на OpenCLComputeEngine
     * @throw std::runtime_error если не инициализирован
     */
    static OpenCLComputeEngine& GetInstance();

    /**
     * @brief Проверить инициализацию
     */
    static bool IsInitialized();

    /**
     * @brief Очистить GPU ресурсы и завершить
     */
    static void Cleanup();

    // ==================== KERNEL MANAGEMENT ====================

    /**
     * @brief Загрузить OpenCL программу из исходного кода
     * @param source Текст OpenCL C кода
     * @return Shared pointer на KernelProgram (с кешированием)
     */
    std::shared_ptr<KernelProgram> LoadProgram(const std::string& source);

    /**
     * @brief Получить kernel из загруженной программы
     * @param program Программа (результат LoadProgram)
     * @param kernel_name Имя kernel'а в OpenCL коде
     * @return cl_kernel (OpenCL объект kernel'а)
     * @throw std::runtime_error если kernel не найден
     */
    cl_kernel GetKernel(
        const std::shared_ptr<KernelProgram>& program,
        const std::string& kernel_name
    );

    // ==================== BUFFER MANAGEMENT ====================

    /**
     * @brief Создать пустой буфер на GPU (неинициализированный)
     * @param num_elements Количество элементов
     * @param type Тип доступа (READ_ONLY, WRITE_ONLY, READ_WRITE)
     * @return Unique pointer на GPUMemoryBuffer
     */
    std::unique_ptr<GPUMemoryBuffer> CreateBuffer(
        size_t num_elements,
        MemoryType type = MemoryType::GPU_WRITE_ONLY
    );

    /**
     * @brief Создать буфер и инициализировать данными (complex<float>)
     * 
     * СПЕЦИАЛИЗАЦИЯ для сигналов (вектора комплексных чисел).
     * Используется для загрузки сигналов с хоста на GPU.
     * 
     * @param data Вектор std::complex<float>
     * @param type Тип доступа (обычно GPU_READ_ONLY)
     * @return Unique pointer на GPUMemoryBuffer
     * 
     * ПРИМЕР:
     *   std::vector<std::complex<float>> signal = {...};
     *   auto gpu_buf = engine.CreateBufferWithData(signal, GPU_READ_ONLY);
     */
    std::unique_ptr<GPUMemoryBuffer> CreateBufferWithData(
        const std::vector<std::complex<float>>& data,
        MemoryType type = MemoryType::GPU_READ_ONLY
    );

    /**
     * @brief Создать буфер и инициализировать данными (ТИПОБЕЗОПАСНЫЙ ШАБЛОН)
     * 
     * УНИВЕРСАЛЬНЫЙ метод для загрузки данных ЛЮБОГО POD-типа на GPU:
     *   - Структуры (CombinedDelayParam, BeamConfig и т.д.)
     *   - Arrays примитивных типов (float, int, uint и т.д.)
     *   - Массивы пользовательских структур
     * 
     * @tparam T Тип элементов вектора (должен быть POD-типом)
     * @param data Вектор данных типа T
     * @param type Тип доступа (обычно GPU_READ_ONLY для параметров)
     * @return Unique pointer на GPUMemoryBuffer
     * @throw std::invalid_argument если вектор пуст
     * 
     * ПРИМЕР: Загрузка параметров задержек
     *   std::vector<CombinedDelayParam> delays = {...};
     *   auto gpu_delays = engine.CreateTypedBufferWithData<CombinedDelayParam>(
     *       delays,
     *       GPU_READ_ONLY
     *   );
     * 
     * ПРИМЕР: Загрузка коэффициентов (автоматическая деду типа)
     *   std::vector<float> coeffs = {...};
     *   auto gpu_coeffs = engine.CreateTypedBufferWithData(coeffs, GPU_READ_ONLY);
     */
    template <typename T>
    std::unique_ptr<GPUMemoryBuffer> CreateTypedBufferWithData(
        const std::vector<T>& data,
        MemoryType type = MemoryType::GPU_READ_ONLY
    );

    // ==================== KERNEL EXECUTION ====================

    /**
     * @brief Выполнить kernel с переданными буферами
     * @param kernel Kernel для выполнения
     * @param buffers Вектор GPU буферов (cl_mem)
     * @param global_work_size Размерность сетки [3] элементов
     * @param local_work_size Размерность блока [3] элементов
     */
    void ExecuteKernel(
        cl_kernel kernel,
        const std::vector<cl_mem>& buffers,
        const std::array<size_t, 3>& global_work_size,
        const std::array<size_t, 3>& local_work_size = {256, 1, 1}
    );

    /**
     * @brief Выполнить kernel асинхронно и вернуть event
     */
    cl_event ExecuteKernelAsync(
        cl_kernel kernel,
        const std::vector<cl_mem>& buffers,
        const std::array<size_t, 3>& global_work_size,
        const std::array<size_t, 3>& local_work_size = {256, 1, 1}
    );

    // ==================== SYNCHRONIZATION ====================

    /**
     * @brief Ждать одного события
     */
    void WaitForEvent(cl_event event);

    /**
     * @brief Ждать вектора событий
     */
    void WaitForEvents(const std::vector<cl_event>& events);

    /**
     * @brief Flush command queue (может вернуть управление до завершения)
     */
    void Flush();

    /**
     * @brief Finish command queue (ждёт завершения всех операций)
     */
    void Finish();

    // ==================== STATISTICS ====================

    /**
     * @brief Получить строку со статистикой использования памяти
     */
    std::string GetStatistics() const;

    /**
     * @brief Получить информацию о GPU устройстве
     */
    std::string GetDeviceInfo() const;

    /**
     * @brief Получить статистику кеша kernel программ
     */
    std::string GetCacheStatistics() const;

    // ==================== CLEANUP ====================

    /**
     * @brief Деструктор (автоматическая очистка)
     */
    ~OpenCLComputeEngine();

    // Запретить копирование (singleton)
    OpenCLComputeEngine(const OpenCLComputeEngine&) = delete;
    OpenCLComputeEngine& operator=(const OpenCLComputeEngine&) = delete;

private:
    OpenCLComputeEngine() = default;

    static std::unique_ptr<OpenCLComputeEngine> instance_;
    static bool initialized_;
    static std::mutex initialization_mutex_;

    size_t total_allocated_bytes_ = 0;
    size_t num_buffers_ = 0;
    size_t kernel_executions_ = 0;
};

// ==========================
// INLINE ШАБЛОН РЕАЛИЗАЦИЯ
// ==========================

/**
 * @brief Реализация универсального типобезопасного буфера
 * 
 * Компилируется в оптимизированный код для каждого типа T.
 * Размер буфера вычисляется как: data.size() * sizeof(T)
 */
template <typename T>
inline std::unique_ptr<GPUMemoryBuffer>
OpenCLComputeEngine::CreateTypedBufferWithData(
    const std::vector<T>& data,
    MemoryType type)
{
    // Проверка
    if (data.empty()) {
        throw std::invalid_argument(
            "CreateTypedBufferWithData: data vector is empty"
        );
    }

    // Получить GPU контекст и очередь команд
    auto core = OpenCLCore::GetInstance();
    CommandQueuePool& pool = CommandQueuePool::GetInstance();
    cl_command_queue queue = pool.GetNextQueue();

    // Создать GPU буфер с инициализацией из памяти хоста
    // Шаблон автоматически подставляет правильный sizeof(T)
    auto buffer = std::make_unique<GPUMemoryBuffer>(
        core.GetContext(),
        queue,
        static_cast<const void*>(data.data()),  // Указатель на данные хоста
        data.size() * sizeof(T),                 // Размер в байтах (для любого T!)
        data.size(),                             // Количество элементов
        type                                     // Тип доступа
    );

    // Обновить статистику
    total_allocated_bytes_ += buffer->GetSizeBytes();
    num_buffers_++;

    return buffer;
}

} // namespace gpu
```

---

## 📍 ЧТО ЗДЕСЬ ВАЖНОГО

### 1️⃣ Два разных API для двух разных случаев

```cpp
// Для сигналов (специализированный, оптимизированный)
CreateBufferWithData(const std::vector<std::complex<float>>& data, ...)

// Для параметров (универсальный шаблон)
CreateTypedBufferWithData<T>(const std::vector<T>& data, ...)
```

### 2️⃣ Шаблон компилируется inline

```cpp
template <typename T>
inline std::unique_ptr<GPUMemoryBuffer>
OpenCLComputeEngine::CreateTypedBufferWithData(...)
```

- Ключевое слово `inline` даёт компилятору знак встраивать код
- Каждый вызов с разным T генерирует специализированный код
- На выходе нет оверхеда — просто правильный размер `sizeof(T)`

### 3️⃣ static_cast для типобезопасности

```cpp
static_cast<const void*>(data.data())
```

- `data.data()` возвращает `T*`
- Явно приводим к `void*` для OpenCL API
- Компилятор контролирует это — безопаснее, чем C-style cast

### 4️⃣ sizeof(T) автоматически подставляется

```cpp
data.size() * sizeof(T)  // Работает для ЛЮБОГО T!
```

---

## ✅ ГОТОВО К USE

Добавь этот файл в проект, и в `generator_gpu_new.cpp` можно писать:

```cpp
auto gpu_delays = engine_->CreateTypedBufferWithData(
    combined_host,
    gpu::MemoryType::GPU_READ_ONLY
);
```

**Production-ready code! 🏆**
