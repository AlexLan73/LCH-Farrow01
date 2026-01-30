// ═══════════════════════════════════════════════════════════════════════
// ДОБАВИТЬ В opencl_manager.h - ЗАГОЛОВОК
// ═══════════════════════════════════════════════════════════════════════

// В КОНЕЦ класса OpenCLManager добавить эти методы:

public:
    // ═══════════════════════════════════════════════════════════════
    // EXTERNAL cl_mem BUFFER SUPPORT
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить информацию о произвольном cl_mem буфере
     * 
     * Полезно чтобы узнать параметры буфера, который получил от другой библиотеки.
     * Не требует создания очереди или контекста.
     * 
     * @param buffer cl_mem для анализа
     * @return ExternalBufferInfo с полной информацией
     * @throws std::runtime_error если buffer invalid
     */
    ExternalBufferInfo GetExternalBufferInfo(cl_mem buffer) const;
    
    /**
     * @brief Обернуть внешний cl_mem как наш буфер (SVM стратегия)
     * 
     * Создаёт wrapper вокруг cl_mem буфера, позволяя использовать его
     * через наш unified интерфейс IMemoryBuffer.
     * 
     * ВАЖНО: Буфер ДОЛЖЕН быть создан с флагом CL_MEM_USE_HOST_PTR 
     *        или иметь SVM backing для работы!
     * 
     * @param external_buffer cl_mem буфер из другого контекста
     * @param num_elements Количество элементов (float/int/etc)
     * @param type Тип памяти (GPU_READ_WRITE, GPU_READ_ONLY, etc)
     * @return unique_ptr на IMemoryBuffer wrapper
     * @throws std::runtime_error если буфер не поддерживает требуемую операцию
     */
    std::unique_ptr<IMemoryBuffer> WrapExternalBufferWithSVM(
        cl_mem external_buffer,
        size_t num_elements,
        MemoryType type);
    
    /**
     * @brief Получить очередь совместимую с внешним буфером
     * 
     * Если нам нужна очередь в контексте внешнего буфера
     * (для копирования данных), этот метод создаст её.
     * 
     * @param external_buffer cl_mem буфер
     * @return cl_command_queue в его контексте
     * @throws std::runtime_error если контекст недоступен
     * 
     * ВНИМАНИЕ: Очередь должна быть освобождена вызывающим кодом!
     */
    cl_command_queue CreateQueueForExternalBuffer(cl_mem external_buffer) const;

// ═══════════════════════════════════════════════════════════════════════
// ДОБАВИТЬ В opencl_manager.cpp - РЕАЛИЗАЦИЯ
// ═══════════════════════════════════════════════════════════════════════

// Добавить после метода ReleaseResources()

ExternalBufferInfo OpenCLManager::GetExternalBufferInfo(cl_mem buffer) const {
    return ExternalBufferInfo::Query(buffer);
}

std::unique_ptr<IMemoryBuffer> OpenCLManager::WrapExternalBufferWithSVM(
    cl_mem external_buffer,
    size_t num_elements,
    MemoryType type) {
    
    if (!initialized_) {
        throw std::runtime_error("OpenCLManager not initialized");
    }
    
    if (!external_buffer) {
        throw std::runtime_error("WrapExternalBufferWithSVM: buffer is nullptr");
    }
    
    // Получить информацию о буфере
    auto info = ExternalBufferInfo::Query(external_buffer);
    
    // Проверить если можем использовать как SVM
    if (!info.HasHostPtr()) {
        throw std::runtime_error(
            "WrapExternalBufferWithSVM: buffer must have host_ptr backing");
    }
    
    if (!info.IsReadWrite() && !info.IsReadable() && !info.IsWritable()) {
        throw std::runtime_error(
            "WrapExternalBufferWithSVM: buffer has incompatible access flags");
    }
    
    // Создать SVMBuffer wrapper
    // SVMBuffer требует: host pointer, размер, queue, флаги
    auto svm_buffer = std::make_unique<SVMBuffer>(
        context_,
        queue_,
        num_elements,
        type
    );
    
    // Установить external backing
    // ВАЖНО: это зависит от реализации SVMBuffer
    // Может потребоваться добавить метод SetExternalPointer()
    
    std::unique_lock lock(registry_mutex_);
    total_allocated_bytes_ += info.size_bytes;
    num_buffers_++;
    
    return svm_buffer;
}

cl_command_queue OpenCLManager::CreateQueueForExternalBuffer(
    cl_mem external_buffer) const {
    
    if (!external_buffer) {
        throw std::runtime_error("CreateQueueForExternalBuffer: buffer is nullptr");
    }
    
    cl_int err;
    
    // Получить контекст буфера
    cl_context external_ctx;
    err = clGetMemObjectInfo(
        external_buffer,
        CL_MEM_CONTEXT,
        sizeof(external_ctx),
        &external_ctx,
        nullptr);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "CreateQueueForExternalBuffer: Failed to get buffer context");
    }
    
    // Получить devices из контекста
    cl_uint num_devices;
    err = clGetContextInfo(
        external_ctx,
        CL_CONTEXT_NUM_DEVICES,
        sizeof(num_devices),
        &num_devices,
        nullptr);
    
    if (err != CL_SUCCESS || num_devices == 0) {
        throw std::runtime_error(
            "CreateQueueForExternalBuffer: Failed to get context devices");
    }
    
    // Получить первый device
    cl_device_id device;
    err = clGetContextInfo(
        external_ctx,
        CL_CONTEXT_DEVICES,
        sizeof(device),
        &device,
        nullptr);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "CreateQueueForExternalBuffer: Failed to get device from context");
    }
    
    // Создать очередь
    cl_command_queue queue = clCreateCommandQueue(
        external_ctx,
        device,
        0,  // flags
        &err);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "CreateQueueForExternalBuffer: Failed to create command queue");
    }
    
    return queue;
}

