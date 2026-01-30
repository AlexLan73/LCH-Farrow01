#pragma once

// ═══════════════════════════════════════════════════════════════════════
// EXTERNAL cl_mem BUFFER USAGE GUIDE
// ═══════════════════════════════════════════════════════════════════════
// 
// Этот файл содержит примеры использования нового функционала для работы
// с cl_mem буферами, созданными в других контекстах или библиотеках.
//
// Сценарий: Библиотека X создаёт cl_mem буфер, ты хочешь использовать
//           его через ManagerOpenCL с твоими kernels.
//
// ═══════════════════════════════════════════════════════════════════════

namespace ManagerOpenCL::Examples {

// ═══════════════════════════════════════════════════════════════════════
// SCENARIO 1: Получить информацию о буфере от внешней библиотеки
// ═══════════════════════════════════════════════════════════════════════

void Example_QueryExternalBuffer() {
    // Предположим, Class A создала свой cl_mem:
    // cl_mem external_buffer = classA.GetBuffer();
    
    cl_mem external_buffer = nullptr;  // от Class A
    
    try {
        // ШАГ 1: Запросить информацию о буфере
        auto info = ExternalBufferInfo::Query(external_buffer);
        
        // ШАГ 2: Проверить параметры
        std::cout << "Buffer size: " << info.size_bytes << " bytes\n";
        std::cout << "Is readable: " << info.IsReadable() << "\n";
        std::cout << "Is writable: " << info.IsWritable() << "\n";
        std::cout << "Has host_ptr: " << info.HasHostPtr() << "\n";
        
        // ШАГ 3: Решить стратегию
        if (info.HasHostPtr()) {
            std::cout << "Можем использовать SVM\n";
        } else {
            std::cout << "Используем host staging buffer\n";
        }
        
        // ВАЖНО: НУЖНО освободить контекст когда не нужен
        if (info.context) {
            clReleaseContext(info.context);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIO 2: Копировать данные ИЗ внешнего буфера
// ═══════════════════════════════════════════════════════════════════════

void Example_CopyFromExternal(cl_mem external_buffer, size_t size_bytes) {
    
    // Создать host буфер для результата
    std::vector<float> host_data(size_bytes / sizeof(float));
    
    try {
        // Способ 1: Если знаем queue из external_context
        cl_command_queue external_queue = nullptr;  // от Class A
        
        CLBufferBridge::CopyFromExternal(
            external_buffer,
            external_queue,  // nullptr → создаст свою
            0,               // offset_bytes
            size_bytes,
            host_data.data());
        
        std::cout << "Успешно скопировали " << size_bytes << " bytes\n";
        
        // Теперь host_data содержит данные из external_buffer
        
    } catch (const std::exception& e) {
        std::cerr << "Copy failed: " << e.what() << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIO 3: Писать данные В внешний буфер
// ═══════════════════════════════════════════════════════════════════════

void Example_CopyToExternal(cl_mem external_buffer, 
                            const std::vector<float>& data) {
    
    try {
        // Писать данные в external_buffer
        CLBufferBridge::CopyToExternal(
            external_buffer,
            nullptr,                              // queue (создаст свою)
            0,                                    // offset
            data.size() * sizeof(float),
            data.data());
        
        std::cout << "Успешно написали данные\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Write failed: " << e.what() << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIO 4: Работать с буфером через unified interface (SVM)
// ═══════════════════════════════════════════════════════════════════════

void Example_WrapWithUnifiedInterface(cl_mem external_buffer, 
                                      size_t num_elements) {
    
    auto& manager = OpenCLManager::GetInstance();
    
    try {
        // Обернуть external_buffer как наш IMemoryBuffer
        auto wrapped = manager.WrapExternalBufferWithSVM(
            external_buffer,
            num_elements,
            MemoryType::GPU_READ_WRITE);
        
        // Теперь можем использовать как обычный буфер
        // wrapped->Write(host_data, 0, size);
        // wrapped->Read(host_data, 0, size);
        
    } catch (const std::exception& e) {
        std::cerr << "Wrap failed: " << e.what() << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIO 5: КОМПЛЕКСНЫЙ - Работа с Class A буфером и own kernel
// ═══════════════════════════════════════════════════════════════════════

void Example_CompleteWorkflow() {
    
    // ДОПУСТИМ:
    // - Class A (другая библиотека) создала cl_mem buffer
    // - Нам нужно запустить на нём свой kernel
    // - Результат отправить обратно Class A
    
    // 1. Получить external buffer от Class A
    // cl_mem external_input = classA.GetInputBuffer();
    // cl_command_queue external_queue = classA.GetQueue();
    
    // 2. Получить информацию
    // auto info = ExternalBufferInfo::Query(external_input);
    // size_t buffer_size = info.size_bytes;
    
    // 3. Скопировать в наш контекст (ManagerOpenCL)
    std::vector<float> host_staging(100);  // Для примера
    
    // CLBufferBridge::CopyFromExternal(
    //     external_input,
    //     external_queue,
    //     0,
    //     host_staging.size() * sizeof(float),
    //     host_staging.data());
    
    // 4. Создать наш собственный буфер для обработки
    // auto our_buffer = OpenCLComputeEngine::GetInstance().CreateBuffer(
    //     host_staging.size(),
    //     MemoryType::GPU_READ_WRITE);
    
    // our_buffer->Write(host_staging, 0, host_staging.size() * sizeof(float));
    
    // 5. Запустить kernel
    // OpenCLComputeEngine::GetInstance().ExecuteKernel(
    //     kernel_program,
    //     our_buffer,
    //     size);
    
    // 6. Скопировать результат обратно
    // our_buffer->Read(host_staging, 0, host_staging.size() * sizeof(float));
    
    // 7. Писать результат в external buffer
    // CLBufferBridge::CopyToExternal(
    //     external_input,  // или другой output buffer
    //     external_queue,
    //     0,
    //     host_staging.size() * sizeof(float),
    //     host_staging.data());
    
    std::cout << "Complete workflow example (pseudocode)\n";
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIO 6: Асинхронное копирование для больших объёмов
// ═══════════════════════════════════════════════════════════════════════

void Example_AsyncCopy(cl_mem external_buffer, 
                       cl_command_queue external_queue,
                       size_t size_bytes) {
    
    std::vector<float> host_data(size_bytes / sizeof(float));
    cl_event event = nullptr;
    
    try {
        // Начать асинхронное копирование
        CLBufferBridge::CopyFromExternalAsync(
            external_buffer,
            external_queue,
            0,
            size_bytes,
            host_data.data(),
            &event);
        
        // Пока копирование идёт, можем что-то делать
        std::cout << "Data is being copied...\n";
        
        // Когда нужны результаты, дождаться event
        if (event) {
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        }
        
        std::cout << "Copy complete\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Async copy failed: " << e.what() << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIO 7: Error handling - что делать с incompatible buffers
// ═══════════════════════════════════════════════════════════════════════

void Example_ErrorHandling(cl_mem external_buffer) {
    
    try {
        // Попытка обернуть buffer
        auto info = ExternalBufferInfo::Query(external_buffer);
        
        // Проверить если буфер read-only
        if (!info.IsWritable()) {
            std::cout << "Buffer is read-only, cannot write\n";
            // Использовать только CopyFromExternal
        }
        
        // Проверить если есть host backing
        if (!info.HasHostPtr()) {
            std::cout << "Buffer has no host backing, using staging copy\n";
            // CLBufferBridge будет использовать host staging
        }
        
        // Проверить если это buffer (не image)
        if (!info.IsBuffer()) {
            std::cerr << "Object is not a buffer (maybe image?)\n";
            return;
        }
        
    } catch (const std::runtime_error& e) {
        std::cerr << "Invalid buffer: " << e.what() << "\n";
        // Graceful degradation
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIO 8: Получить очередь для работы с external buffer
// ═══════════════════════════════════════════════════════════════════════

void Example_GetQueue(cl_mem external_buffer) {
    
    auto& manager = OpenCLManager::GetInstance();
    
    try {
        // Получить очередь которая совместима с буфером
        cl_command_queue queue = manager.CreateQueueForExternalBuffer(
            external_buffer);
        
        // Теперь можем использовать эту очередь для операций
        // CLBufferBridge::CopyFromExternal(
        //     external_buffer,
        //     queue,
        //     ...);
        
        // ВАЖНО: Освободить очередь когда не нужна
        clReleaseCommandQueue(queue);
        
    } catch (const std::exception& e) {
        std::cerr << "Cannot get queue: " << e.what() << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BEST PRACTICES
// ═══════════════════════════════════════════════════════════════════════

/*
 * 1. ВСЕГДА Query() перед использованием:
 *    auto info = ExternalBufferInfo::Query(external_buffer);
 *
 * 2. Проверить compatibilties:
 *    if (!info.IsReadable()) { /* handle error */ }
 *
 * 3. Использовать CLBufferBridge для кросс-контекст операций:
 *    CLBufferBridge::CopyFromExternal(...);
 *    CLBufferBridge::CopyToExternal(...);
 *
 * 4. Для больших объёмов - асинхронные версии:
 *    CLBufferBridge::CopyFromExternalAsync(...);
 *
 * 5. Помнить про release ресурсов:
 *    if (info.context) clReleaseContext(info.context);
 *    if (queue) clReleaseCommandQueue(queue);
 *
 * 6. Использовать ExternalBufferHandle для RAII:
 *    {
 *        ExternalBufferHandle handle(info);
 *        // handle.GetInfo() safe
 *    }  // Автоматически released
 *
 * 7. Правильный размер buffer:
 *    - Всегда проверить info.size_bytes
 *    - Выделять host буфер адекватного размера
 *    - Не полагаться на num_elements - это guess
 *
 * 8. Thread-safe копирование:
 *    - CLBufferBridge методы thread-safe
 *    - Но очереди OpenCL НЕ thread-safe
 *    - Использовать мьютекс при многопоточности
 *
 * 9. Performance:
 *    - Host staging (из CLBufferBridge) медленнее чем прямой доступ
 *    - Если возможно - попросить у Class A использовать SVM
 *    - Асинхронный копирование для конвейерных операций
 *
 * 10. Отладка:
 *     - ExternalBufferInfo::Query выдаст всю информацию
 *     - Проверить флаги доступа, размер, контекст
 *     - Использовать CL_DEVICE_INFO для диагностики
 *
 */

} // namespace ManagerOpenCL::Examples

