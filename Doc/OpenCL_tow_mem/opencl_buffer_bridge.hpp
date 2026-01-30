#pragma once

#include <CL/cl.h>
#include <cstring>
#include <stdexcept>

namespace ManagerOpenCL {

// ═══════════════════════════════════════════════════════════════════════
// EXTERNAL cl_mem BUFFER SUPPORT - OpenCL Interoperability
// ═══════════════════════════════════════════════════════════════════════
// 
// Purpose: Позволяет работать с cl_mem буферами, созданными в других
//          контекстах или библиотеках, интегрируя их в ManagerOpenCL
//
// Usage:
//   cl_mem external_buffer = /* from external library */;
//   auto info = ExternalBufferInfo::Query(external_buffer);
//   
//   if (CanUseSVM(info)) {
//       auto wrapped = WrapWithSVM(external_buffer, info);
//   } else {
//       CLBufferBridge::CopyFromExternal(external_buffer, info, host_buf, size);
//   }
// ═══════════════════════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────────────────────
// ШАГИ ИНТЕГРАЦИИ ДЛЯ ТЕХ КОГО ИСПОЛЬЗУЕШЬ:
// ───────────────────────────────────────────────────────────────────────
// 
// 1. ADD TO opencl_manager.h - включить новые методы
// 2. ADD TO opencl_manager.cpp - реализовать методы
// 3. ADD THIS FILE - новый заголовочный файл
// 4. ADD opencl_buffer_bridge.cpp - реализация bridge
// 5. Обновить CMakeLists.txt
//
// ───────────────────────────────────────────────────────────────────────

/**
 * @struct ExternalBufferInfo
 * @brief Метаданные о внешнем cl_mem буфере
 * 
 * Используется для получения информации о буфере, который мы не создавали.
 * Важен для понимания, как именно работать с буфером (контекст, размер, etc.)
 */
struct ExternalBufferInfo {
    // ═══════════════════════════════════════════════════════════════
    // MEMBER DATA
    // ═══════════════════════════════════════════════════════════════
    
    size_t num_elements;              ///< Количество элементов (float/int/etc)
    size_t size_bytes;                ///< Размер в байтах
    cl_mem_flags flags;               ///< CL_MEM_READ_ONLY, WRITE_ONLY, READ_WRITE
    cl_mem_object_type type;          ///< CL_MEM_OBJECT_BUFFER, IMAGE2D, etc
    
    cl_context context;               ///< Контекст которому принадлежит буфер
    cl_device_id device;              ///< Девайс основной
    
    void* host_ptr;                   ///< Если буфер с host backing
    bool is_svm_compatible;           ///< Можно ли использовать SVM
    
    // ═══════════════════════════════════════════════════════════════
    // QUERY METHOD - получить информацию о произвольном cl_mem
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Запросить информацию о внешнем буфере
     * 
     * Этот метод - MAIN ENTRY POINT для работы с чужими буферами.
     * Безопасно извлекает все метаданные.
     * 
     * @param buffer cl_mem который нужно проанализировать
     * @return ExternalBufferInfo с полной информацией
     * @throws std::runtime_error если buffer invalid
     */
    static ExternalBufferInfo Query(cl_mem buffer) {
        if (!buffer) {
            throw std::runtime_error("Query: buffer is nullptr");
        }
        
        ExternalBufferInfo info{};
        cl_int err;
        
        // Получить размер
        size_t size = 0;
        err = clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size), &size, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Query: Failed to get buffer size");
        }
        info.size_bytes = size;
        info.num_elements = size / sizeof(float);  // Default assumption
        
        // Получить флаги доступа
        err = clGetMemObjectInfo(buffer, CL_MEM_FLAGS, sizeof(info.flags), 
                                 &info.flags, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Query: Failed to get buffer flags");
        }
        
        // Получить тип объекта
        err = clGetMemObjectInfo(buffer, CL_MEM_TYPE, sizeof(info.type), 
                                 &info.type, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Query: Failed to get buffer type");
        }
        
        // Получить контекст
        err = clGetMemObjectInfo(buffer, CL_MEM_CONTEXT, sizeof(info.context), 
                                 &info.context, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Query: Failed to get buffer context");
        }
        
        // Сохранить контекст
        if (info.context) {
            clRetainContext(info.context);
        }
        
        // Получить host_ptr если есть
        err = clGetMemObjectInfo(buffer, CL_MEM_HOST_PTR, sizeof(info.host_ptr), 
                                 &info.host_ptr, nullptr);
        // Может не быть - это OK
        
        info.is_svm_compatible = false;  // Будет проверено позже
        
        return info;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // HELPER METHODS
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Проверить флаги доступа
     */
    bool IsReadable() const {
        return (flags & CL_MEM_READ_WRITE) != 0 || 
               (flags & CL_MEM_READ_ONLY) != 0;
    }
    
    bool IsWritable() const {
        return (flags & CL_MEM_READ_WRITE) != 0 || 
               (flags & CL_MEM_WRITE_ONLY) != 0;
    }
    
    bool IsReadWrite() const {
        return (flags & CL_MEM_READ_WRITE) != 0;
    }
    
    /**
     * @brief Проверить если это buffer (не image)
     */
    bool IsBuffer() const {
        return type == CL_MEM_OBJECT_BUFFER;
    }
    
    /**
     * @brief Проверить если есть host backing
     */
    bool HasHostPtr() const {
        return host_ptr != nullptr;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// cl_mem BUFFER BRIDGE - Копирование данных между контекстами
// ═══════════════════════════════════════════════════════════════════════

/**
 * @class CLBufferBridge
 * @brief Безопасное копирование данных из/в внешние cl_mem буферы
 * 
 * Когда два контекста OpenCL не совместимы напрямую (разные девайсы,
 * разные платформы), используем host staging buffer для копирования.
 * 
 * Стратегия:
 * - Если SVM доступен → direct memcpy
 * - Если нет → Host staging buffer (malloc -> clEnqueueReadBuffer -> memcpy)
 */
class CLBufferBridge {
public:
    
    // ═══════════════════════════════════════════════════════════════
    // КОПИРОВАНИЕ ИЗ ВНЕШНЕГО BUFFER
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Копировать данные ИЗ внешнего cl_mem буфера
     * 
     * Читает данные из внешнего буфера через host staging buffer.
     * SAFE для работы с буферами из других контекстов/библиотек.
     * 
     * @param external_buffer Буфер который читаем
     * @param external_queue Очередь в его контексте (может быть nullptr)
     * @param offset_bytes Смещение в буфере (по умолчанию 0)
     * @param size_bytes Сколько байт читать
     * @param[out] host_buffer Куда писать результат (должен быть >= size_bytes)
     * 
     * @throws std::runtime_error если операция не удалась
     * 
     * Example:
     *   CLBufferBridge::CopyFromExternal(
     *       external_buffer,
     *       external_queue,
     *       0,                    // offset
     *       size_bytes,
     *       my_host_buffer);
     */
    static void CopyFromExternal(
        cl_mem external_buffer,
        cl_command_queue external_queue,  // From external context
        size_t offset_bytes,
        size_t size_bytes,
        void* host_buffer) {
        
        if (!external_buffer) {
            throw std::runtime_error("CopyFromExternal: buffer is nullptr");
        }
        if (!host_buffer) {
            throw std::runtime_error("CopyFromExternal: host_buffer is nullptr");
        }
        if (size_bytes == 0) {
            return;  // Nothing to do
        }
        
        cl_int err;
        
        // Если queue не предоставлена, нужно создать
        cl_command_queue queue = external_queue;
        bool created_queue = false;
        
        if (!queue) {
            // Получить контекст буфера
            cl_context ctx;
            err = clGetMemObjectInfo(external_buffer, CL_MEM_CONTEXT, 
                                    sizeof(ctx), &ctx, nullptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("CopyFromExternal: Failed to get buffer context");
            }
            
            // Получить device из контекста
            cl_uint num_devices;
            err = clGetContextInfo(ctx, CL_CONTEXT_NUM_DEVICES, 
                                  sizeof(num_devices), &num_devices, nullptr);
            if (err != CL_SUCCESS || num_devices == 0) {
                throw std::runtime_error("CopyFromExternal: Failed to get context devices");
            }
            
            cl_device_id device;
            err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 
                                  sizeof(device), &device, nullptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("CopyFromExternal: Failed to get device from context");
            }
            
            // Создать очередь
            queue = clCreateCommandQueue(ctx, device, 0, &err);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("CopyFromExternal: Failed to create command queue");
            }
            created_queue = true;
        }
        
        try {
            // Читать из буфера в host memory
            err = clEnqueueReadBuffer(
                queue,
                external_buffer,
                CL_TRUE,              // Blocking
                offset_bytes,
                size_bytes,
                host_buffer,
                0, nullptr, nullptr);  // No events
            
            if (err != CL_SUCCESS) {
                throw std::runtime_error("CopyFromExternal: clEnqueueReadBuffer failed");
            }
        } catch (...) {
            if (created_queue) {
                clReleaseCommandQueue(queue);
            }
            throw;
        }
        
        if (created_queue) {
            clReleaseCommandQueue(queue);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    // КОПИРОВАНИЕ В ВНЕШНИЙ BUFFER
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Копировать данные В внешний cl_mem буфер
     * 
     * Пишет данные в внешний буфер через host staging buffer.
     * SAFE для работы с буферами из других контекстов/библиотек.
     * 
     * @param external_buffer Буфер который пишем
     * @param external_queue Очередь в его контексте (может быть nullptr)
     * @param offset_bytes Смещение в буфере
     * @param size_bytes Сколько байт писать
     * @param host_buffer Откуда читать данные (должен быть >= size_bytes)
     * 
     * @throws std::runtime_error если операция не удалась
     * 
     * Example:
     *   CLBufferBridge::CopyToExternal(
     *       external_buffer,
     *       external_queue,
     *       0,
     *       size_bytes,
     *       my_data);
     */
    static void CopyToExternal(
        cl_mem external_buffer,
        cl_command_queue external_queue,  // From external context
        size_t offset_bytes,
        size_t size_bytes,
        const void* host_buffer) {
        
        if (!external_buffer) {
            throw std::runtime_error("CopyToExternal: buffer is nullptr");
        }
        if (!host_buffer) {
            throw std::runtime_error("CopyToExternal: host_buffer is nullptr");
        }
        if (size_bytes == 0) {
            return;  // Nothing to do
        }
        
        cl_int err;
        
        // Если queue не предоставлена, нужно создать
        cl_command_queue queue = external_queue;
        bool created_queue = false;
        
        if (!queue) {
            // Получить контекст буфера
            cl_context ctx;
            err = clGetMemObjectInfo(external_buffer, CL_MEM_CONTEXT, 
                                    sizeof(ctx), &ctx, nullptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("CopyToExternal: Failed to get buffer context");
            }
            
            // Получить device из контекста
            cl_uint num_devices;
            err = clGetContextInfo(ctx, CL_CONTEXT_NUM_DEVICES, 
                                  sizeof(num_devices), &num_devices, nullptr);
            if (err != CL_SUCCESS || num_devices == 0) {
                throw std::runtime_error("CopyToExternal: Failed to get context devices");
            }
            
            cl_device_id device;
            err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 
                                  sizeof(device), &device, nullptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("CopyToExternal: Failed to get device from context");
            }
            
            // Создать очередь
            queue = clCreateCommandQueue(ctx, device, 0, &err);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("CopyToExternal: Failed to create command queue");
            }
            created_queue = true;
        }
        
        try {
            // Писать из host memory в буфер
            err = clEnqueueWriteBuffer(
                queue,
                external_buffer,
                CL_TRUE,              // Blocking
                offset_bytes,
                size_bytes,
                host_buffer,
                0, nullptr, nullptr);  // No events
            
            if (err != CL_SUCCESS) {
                throw std::runtime_error("CopyToExternal: clEnqueueWriteBuffer failed");
            }
        } catch (...) {
            if (created_queue) {
                clReleaseCommandQueue(queue);
            }
            throw;
        }
        
        if (created_queue) {
            clReleaseCommandQueue(queue);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    // АСИНХРОННОЕ КОПИРОВАНИЕ (для больших объёмов)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Асинхронное чтение из внешнего буфера
     * 
     * Полезно для больших объёмов данных чтобы не блокировать программу.
     * 
     * @param external_buffer Источник
     * @param external_queue Очередь
     * @param offset_bytes Смещение
     * @param size_bytes Размер
     * @param[out] host_buffer Результат
     * @param[out] event OpenCL event для синхронизации (может быть nullptr)
     */
    static void CopyFromExternalAsync(
        cl_mem external_buffer,
        cl_command_queue external_queue,
        size_t offset_bytes,
        size_t size_bytes,
        void* host_buffer,
        cl_event* event = nullptr) {
        
        if (!external_buffer || !host_buffer) {
            throw std::runtime_error("CopyFromExternalAsync: invalid parameters");
        }
        
        cl_int err = clEnqueueReadBuffer(
            external_queue,
            external_buffer,
            CL_FALSE,           // Non-blocking
            offset_bytes,
            size_bytes,
            host_buffer,
            0, nullptr, event);
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("CopyFromExternalAsync: clEnqueueReadBuffer failed");
        }
    }
    
    /**
     * @brief Асинхронное писание в внешний буфер
     */
    static void CopyToExternalAsync(
        cl_mem external_buffer,
        cl_command_queue external_queue,
        size_t offset_bytes,
        size_t size_bytes,
        const void* host_buffer,
        cl_event* event = nullptr) {
        
        if (!external_buffer || !host_buffer) {
            throw std::runtime_error("CopyToExternalAsync: invalid parameters");
        }
        
        cl_int err = clEnqueueWriteBuffer(
            external_queue,
            external_buffer,
            CL_FALSE,           // Non-blocking
            offset_bytes,
            size_bytes,
            host_buffer,
            0, nullptr, event);
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("CopyToExternalAsync: clEnqueueWriteBuffer failed");
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// UTILITY HELPERS
// ═══════════════════════════════════════════════════════════════════════

/**
 * @brief Проверить совместимость контекстов
 * 
 * Возвращает true если буферы из разных контекстов могут быть скопированы
 * напрямую (обычно нет - нужен host staging)
 */
inline bool AreContextsCompatible(cl_context ctx1, cl_context ctx2) {
    // В стандартном OpenCL буферы из разных контекстов НЕ совместимы
    // даже если используют один девайс
    return ctx1 == ctx2;
}

/**
 * @brief RAII wrapper для ExternalBufferInfo
 * 
 * Автоматически release контекст когда scope заканчивается
 */
class ExternalBufferHandle {
private:
    ExternalBufferInfo info_;
    
public:
    ExternalBufferHandle(const ExternalBufferInfo& info) : info_(info) {
        if (info_.context) {
            clRetainContext(info_.context);
        }
    }
    
    ~ExternalBufferHandle() {
        if (info_.context) {
            clReleaseContext(info_.context);
        }
    }
    
    ExternalBufferHandle(const ExternalBufferHandle&) = delete;
    ExternalBufferHandle& operator=(const ExternalBufferHandle&) = delete;
    
    const ExternalBufferInfo& GetInfo() const { return info_; }
    ExternalBufferInfo& GetInfo() { return info_; }
};

} // namespace ManagerOpenCL

