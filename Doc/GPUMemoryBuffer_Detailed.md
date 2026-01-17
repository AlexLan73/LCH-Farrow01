# GPUMemoryBuffer - Детальная документация

## Обзор

`GPUMemoryBuffer` - обёртка над OpenCL памятью (cl_mem), предоставляющая высокоуровневый интерфейс для операций чтения/записи между CPU и GPU. Поддерживает как owning (владеет буфером), так и non-owning (использует внешний буфер) режимы.

## Архитектура

```
GPUMemoryBuffer
├── cl_context context_              // OpenCL контекст
├── cl_command_queue queue_          // Командная очередь
├── cl_mem gpu_buffer_               // GPU память (может быть внешней)
├── std::vector<std::complex<float>> pinned_host_buffer_  // Pinned CPU память
├── size_t num_elements_             // Количество элементов
├── size_t buffer_size_bytes_        // Размер в байтах
├── MemoryType type_                 // Тип памяти
├── bool is_external_buffer_         // Флаг владения
└── bool gpu_dirty_                  // Флаг изменений на GPU
```

## Конструкторы

### Owning конструктор

```cpp
GPUMemoryBuffer(
    cl_context context,
    cl_command_queue queue,
    size_t num_elements,
    MemoryType type = MemoryType::GPU_WRITE_ONLY
)
```

**Описание:** Создает новый GPU буфер и владеет им.

**Алгоритм:**
1. Сохранение параметров
2. `AllocateGPUBuffer()` - выделение cl_mem
3. `AllocatePinnedHostBuffer()` - выделение pinned памяти
4. Инициализация флагов (owning = true, dirty = false)

### Non-owning конструктор

```cpp
GPUMemoryBuffer(
    cl_context context,
    cl_command_queue queue,
    cl_mem external_gpu_buffer,    // Внешний буфер
    size_t num_elements,
    MemoryType type = MemoryType::GPU_WRITE_ONLY
)
```

**Описание:** Оборачивает существующий GPU буфер без владения.

**Особенности:**
- `is_external_buffer_ = true`
- Не вызывает `AllocateGPUBuffer()`
- Использует предоставленный `external_gpu_buffer`

### Конструктор с данными

```cpp
GPUMemoryBuffer(
    cl_context context,
    cl_command_queue queue,
    const void* host_data,         // Данные для копирования
    size_t data_size_bytes,        // Размер данных
    size_t num_elements,
    MemoryType type
)
```

**Описание:** Создает буфер и копирует данные на GPU.

**Алгоритм:**
1. Преобразование MemoryType в cl_mem_flags
2. `clCreateBuffer()` с `CL_MEM_COPY_HOST_PTR`
3. `AllocatePinnedHostBuffer()`

## Деструктор

```cpp
~GPUMemoryBuffer()
```

**Алгоритм:**
1. Если `!is_external_buffer_` и `gpu_buffer_ != nullptr`:
   - `clReleaseMemObject(gpu_buffer_)`
2. `ReleasePinnedHostBuffer()`

## Операции чтения

### std::vector<std::complex<float>> ReadFromGPU()

**Описание:** Читает все данные с GPU в CPU память.

**Алгоритм:**
1. `ReadPartial(num_elements_)`
2. Возврат полного вектора

### std::vector<std::complex<float>> ReadPartial(size_t num_elements)

**Описание:** Читает часть данных с GPU.

**Алгоритм:**
1. Валидация: `num_elements <= num_elements_`
2. Создание результирующего вектора
3. `clEnqueueReadBuffer()` с параметрами:
   - `queue_` - командная очередь
   - `gpu_buffer_` - источник
   - `CL_TRUE` - blocking read
   - `0` - offset
   - `num_elements * sizeof(std::complex<float>)` - размер
   - `result.data()` - destination
4. `CheckCLError()` для обработки ошибок
5. `gpu_dirty_ = false`

## Операции записи

### void WriteToGPU(const std::vector<std::complex<float>>& data)

**Описание:** Записывает данные из CPU на GPU.

**Алгоритм:**
1. Валидация: `data.size() <= num_elements_`
2. `clEnqueueWriteBuffer()` с параметрами:
   - `queue_` - командная очередь
   - `gpu_buffer_` - destination
   - `CL_TRUE` - blocking write
   - `0` - offset
   - `data.size() * sizeof(std::complex<float>)` - размер
   - `data.data()` - source
3. `CheckCLError()`
4. `gpu_dirty_ = true`

## Асинхронные операции

### std::pair<std::vector<std::complex<float>>, cl_event> ReadFromGPUAsync()

**Описание:** Асинхронное чтение с возвратом события.

**Возвращает:** Пара (данные, событие)

**Алгоритм:**
1. `clEnqueueReadBuffer()` с `CL_FALSE` (non-blocking)
2. Возврат события для синхронизации

### cl_event WriteToGPUAsync(const std::vector<std::complex<float>>& data)

**Описание:** Асинхронная запись.

**Возвращает:** `cl_event` для синхронизации.

## Move семантика

### Move конструктор

```cpp
GPUMemoryBuffer(GPUMemoryBuffer&& other) noexcept
```

**Описание:** Перемещает владение ресурсами.

**Алгоритм:**
1. Копирование всех членов
2. `other.gpu_buffer_ = nullptr` (перенос владения)

### Move оператор присваивания

```cpp
GPUMemoryBuffer& operator=(GPUMemoryBuffer&& other) noexcept
```

**Описание:** Перемещает ресурсы с очисткой старых.

**Алгоритм:**
1. Освобождение текущих ресурсов
2. Перемещение от `other`
3. Обнуление указателей в `other`

## Информация о буфере

### Геттеры

- `size_t GetNumElements() const` - количество элементов
- `size_t GetSizeBytes() const` - размер в байтах
- `bool IsExternalBuffer() const` - флаг внешнего буфера
- `bool IsGPUDirty() const` - флаг изменений на GPU
- `MemoryType GetMemoryType() const` - тип памяти
- `cl_mem Get() const` - получение cl_mem

### void PrintStats() const

**Выводит:**
```
GPUMemoryBuffer Stats:
  Num Elements:   1024
  Memory (MB):    8.00
  External:       NO
  GPU Dirty:      YES
  Type:           WRITE_ONLY
```

## Приватные методы

### void AllocateGPUBuffer()

**Алгоритм:**
1. Преобразование `MemoryType` в `cl_mem_flags`:
   - `GPU_READ_ONLY` → `CL_MEM_READ_ONLY`
   - `GPU_WRITE_ONLY` → `CL_MEM_WRITE_ONLY`
   - `GPU_READ_WRITE` → `CL_MEM_READ_WRITE`
2. `clCreateBuffer(context_, flags, size, nullptr, &error)`
3. `CheckCLError()`

### void AllocatePinnedHostBuffer()

**Алгоритм:**
1. `pinned_host_buffer_.resize(num_elements_)`
2. Используется для быстрого DMA transfer

### void ReleasePinnedHostBuffer()

**Алгоритм:**
1. `pinned_host_buffer_.clear()`
2. `pinned_host_buffer_.shrink_to_fit()`

### static void CheckCLError(cl_int error, const std::string& operation)

**Алгоритм:**
1. Если `error != CL_SUCCESS`:
   - Формирование сообщения: `"OpenCL Error in " + operation + ": " + std::to_string(error)`
   - `throw std::runtime_error(message)`

## Типы памяти

### MemoryType

```cpp
enum class MemoryType {
    GPU_READ_ONLY,    // CL_MEM_READ_ONLY
    GPU_WRITE_ONLY,   // CL_MEM_WRITE_ONLY
    GPU_READ_WRITE    // CL_MEM_READ_WRITE
};
```

**Использование:**
- `READ_ONLY` - данные только читаются GPU (константы)
- `WRITE_ONLY` - данные только записываются GPU (результаты)
- `READ_WRITE` - данные читаются и модифицируются GPU

## Производительность

### Оптимизации

1. **Pinned memory:**
   - `pinned_host_buffer_` для быстрого DMA
   - Избежание копирований между pageable и pinned памятью

2. **Blocking operations:**
   - По умолчанию используются blocking calls
   - Упрощение синхронизации

3. **Move семантика:**
   - Эффективная передача владения
   - Избежание копирований больших буферов

### DMA Transfer

**Blocking read/write:**
```cpp
// CPU ждет завершения
clEnqueueReadBuffer(queue, buffer, CL_TRUE, ...);
// Продолжение выполнения
```

**Non-blocking (async):**
```cpp
// CPU продолжает работу
cl_event event = clEnqueueReadBuffer(queue, buffer, CL_FALSE, ...);
// Синхронизация позже
clWaitForEvents(1, &event);
```

## Безопасность

### RAII

- Автоматическая очистка в деструкторе
- Исключения в конструкторах безопасны
- Move операции предотвращают двойное освобождение

### Thread safety

- Не thread-safe по дизайну
- Каждый буфер используется в одном потоке
- Синхронизация через OpenCL events

### Валидация

- Проверка размеров в операциях чтения/записи
- OpenCL error checking
- Проверка nullptr в конструкторах

## Использование

### Создание буфера

```cpp
// Owning буфер
auto buffer = std::make_unique<GPUMemoryBuffer>(
    context, queue, 1024, MemoryType::GPU_READ_WRITE
);

// Non-owning обертка
GPUMemoryBuffer wrapper(context, queue, existing_cl_mem,
                        1024, MemoryType::GPU_READ_ONLY);
```

### Операции с данными

```cpp
// Запись данных
std::vector<std::complex<float>> data(1024, {1.0f, 0.0f});
buffer->WriteToGPU(data);

// Чтение результатов
auto result = buffer->ReadFromGPU();
std::cout << "First element: " << result[0] << std::endl;
```

### Асинхронные операции

```cpp
// Асинхронная запись
cl_event write_event = buffer->WriteToGPUAsync(data);

// Асинхронное чтение
auto [async_result, read_event] = buffer->ReadFromGPUAsync();

// Ожидание завершения
clWaitForEvents(1, &read_event);
clWaitForEvents(1, &write_event);
```

### Move семантика

```cpp
// Перемещение владения
std::unique_ptr<GPUMemoryBuffer> buf1 = CreateBuffer();
std::unique_ptr<GPUMemoryBuffer> buf2 = std::move(buf1);
// buf1 теперь пустой, buf2 владеет буфером
```

## Интеграция

### С OpenCLComputeEngine

```cpp
// Создание через engine
auto buffer = engine.CreateBuffer(1024, MemoryType::GPU_WRITE_ONLY);

// Engine отслеживает статистику
// Автоматическая очистка
```

### С GeneratorGPU

```cpp
// Generator кэширует буферы
buffer_signal_base_ = std::move(output_buffer);

// Чтение через обертку
GPUMemoryBuffer wrapper(core.GetContext(), queue.Get(),
                       cached_buffer->Get(), total_size,
                       MemoryType::GPU_READ_ONLY);
auto data = wrapper.ReadFromGPU();
```

## Расширения

### Потенциальные улучшения

1. **Memory mapping:**
   - `clEnqueueMapBuffer()` для zero-copy access
   - Поддержка unified memory архитектур

2. **Sub-buffer support:**
   - Создание подбуферов из существующих
   - Разделение больших буферов

3. **Compression:**
   - Автоматическая компрессия данных
   - Lazy decompression

4. **Profiling:**
   - Замер времени transfer'ов
   - Статистика использования памяти