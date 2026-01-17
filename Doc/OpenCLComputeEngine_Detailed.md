# OpenCLComputeEngine - Детальная документация

## Обзор

`OpenCLComputeEngine` - главный фасад системы OpenCL, предоставляющий унифицированный высокоуровневый API для работы с GPU. Класс объединяет все компоненты архитектуры (OpenCLCore, CommandQueuePool, KernelProgram, GPUMemoryBuffer) и предоставляет единую точку входа для OpenCL операций.

## Архитектура

```
OpenCLComputeEngine (Singleton)
├── OpenCLCore* core_              // Контекст OpenCL
├── CommandQueuePool* queues_      // Пул очередей
├── KernelProgramCache* cache_     // Кэш программ
├── std::vector<GPUMemoryBuffer*> buffers_  // Управляемые буферы
├── Статистика:
│   ├── total_allocated_bytes_
│   ├── num_buffers_
│   └── kernel_executions_
```

## Инициализация

### static void Initialize(DeviceType device_type = DeviceType::GPU)

**Описание:** Инициализирует всю OpenCL инфраструктуру.

**Алгоритм:**
1. `OpenCLCore::Initialize(device_type)` - контекст
2. `CommandQueuePool::Initialize()` - очереди
3. Создание singleton экземпляра
4. Инициализация счетчиков статистики

**Исключения:** Передает исключения от компонентов.

### static OpenCLComputeEngine& GetInstance()

**Описание:** Thread-safe получение singleton экземпляра.

**Реализация:** Static local variable initialization (C++11).

## Управление программами и kernels

### std::shared_ptr<KernelProgram> LoadProgram(const std::string& source)

**Описание:** Загружает и компилирует OpenCL программу с кэшированием.

**Алгоритм:**
1. `KernelProgramCache::GetOrCompile(source)` - получение из кэша или компиляция
2. Возврат shared_ptr (автоматическое управление жизненным циклом)

**Преимущества кэширования:**
- Избежание перекомпиляции одинаковых программ
- Хранение скомпилированных бинарных файлов
- Thread-safe доступ

### cl_kernel GetKernel(const std::shared_ptr<KernelProgram>& program, const std::string& kernel_name)

**Описание:** Получает kernel из программы.

**Алгоритм:**
1. `program->GetOrCreateKernel(kernel_name)`
2. Возврат cl_kernel (управляется программой)

## Управление памятью

### std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t num_elements, MemoryType type)

**Описание:** Создает новый GPU буфер.

**Алгоритм:**
1. Получение контекста и очереди от core/queues
2. Создание GPUMemoryBuffer (owning)
3. Обновление статистики
4. Возврат unique_ptr

### std::unique_ptr<GPUMemoryBuffer> CreateBufferWithData(const std::vector<std::complex<float>>& data, MemoryType type)

**Описание:** Создает буфер и копирует данные на GPU.

**Алгоритм:**
1. Создание буфера через конструктор GPUMemoryBuffer с данными
2. Автоматическая загрузка данных (CL_MEM_COPY_HOST_PTR)
3. Обновление статистики

### template<typename T> CreateTypedBufferWithData(const std::vector<T>& data, MemoryType type)

**Описание:** Создает буфер для любого POD типа.

**Особенности:**
- Inline реализация в заголовке
- Поддержка типов: DelayParameter, CombinedDelayParam, etc.
- Автоматический расчет размера

## Выполнение kernels

### void ExecuteKernel(cl_kernel kernel, const std::vector<cl_mem>& buffers, const std::array<size_t, 3>& global_work_size, const std::array<size_t, 3>& local_work_size)

**Описание:** Синхронное выполнение kernel.

**Алгоритм:**
1. Получение очереди из пула
2. Установка аргументов kernel через clSetKernelArg
3. clEnqueueNDRangeKernel с ожиданием завершения
4. Обновление статистики выполнений

**Параметры:**
- `kernel` - Скомпилированный kernel
- `buffers` - Вектора cl_mem в порядке аргументов kernel
- `global_work_size` - Общий размер сетки (x,y,z)
- `local_work_size` - Размер workgroup (x,y,z)

### cl_event ExecuteKernelAsync(...)

**Описание:** Асинхронное выполнение kernel.

**Возвращает:** cl_event для синхронизации.

**Использование:**
```cpp
cl_event event = engine.ExecuteKernelAsync(kernel, buffers, global, local);
// ... другие операции ...
engine.WaitForEvent(event);
```

## Синхронизация

### void Flush()

**Описание:** Flush всех очередей (не ждет завершения).

### void Finish()

**Описание:** Ждет завершения всех операций во всех очередях.

### void WaitForEvent(cl_event event)

**Описание:** Ждет завершения конкретного события.

### void WaitForEvents(const std::vector<cl_event>& events)

**Описание:** Ждет завершения нескольких событий.

## Информация и статистика

### std::string GetStatistics() const

**Возвращает:** Подробную статистику использования.

**Включает:**
- Количество выделенных буферов
- Общий объем памяти
- Количество выполнений kernels
- Статистику кэша программ
- Информацию об устройстве

### std::string GetDeviceInfo() const

**Возвращает:** Информацию об OpenCL устройстве.

**Делегирует:** `core_->GetDeviceInfo()`

### std::string GetCacheStatistics() const

**Возвращает:** Статистику кэширования программ.

**Делегирует:** `KernelProgramCache::GetCacheStatistics()`

## Внутренняя структура

### Singleton реализация

```cpp
static std::unique_ptr<OpenCLComputeEngine> instance_;
static bool initialized_;
static std::mutex initialization_mutex_;
```

**Thread safety:**
- Double-checked locking в Initialize()
- Static local variable в GetInstance()

### Управление ресурсами

**RAII:** Деструктор автоматически очищает все ресурсы.

**Зависимости:**
- Не владеет компонентами (OpenCLCore, CommandQueuePool - singleton'ы)
- Управляет созданными GPUMemoryBuffer через unique_ptr
- Кэши программ управляются KernelProgramCache

## Производительность

### Оптимизации

1. **Кэширование программ:**
   - Хэширование исходников
   - Избежание перекомпиляции
   - Статистика hit/miss rate

2. **Пул очередей:**
   - Асинхронное выполнение
   - Балансировка нагрузки
   - Thread-safe round-robin

3. **Управление памятью:**
   - Owning буферы с автоматической очисткой
   - Статистика использования
   - Предотвращение утечек

### Статистика использования

```
OpenCLComputeEngine Statistics:
├── Memory: 256 MB allocated, 8 buffers active
├── Kernels: 42 executions
├── Cache: 85% hit rate, 5 programs cached
└── Device: NVIDIA RTX 3080, 10 GB VRAM
```

## Использование

### Базовый workflow

```cpp
// 1. Инициализация
gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
auto& engine = gpu::OpenCLComputeEngine::GetInstance();

// 2. Загрузка программы
std::string kernel_source = R"(
    __kernel void my_kernel(__global float* input, __global float* output) {
        // ...
    }
)";
auto program = engine.LoadProgram(kernel_source);
auto kernel = engine.GetKernel(program, "my_kernel");

// 3. Создание буферов
auto input = engine.CreateBuffer(1024, gpu::MemoryType::GPU_READ_ONLY);
auto output = engine.CreateBuffer(1024, gpu::MemoryType::GPU_WRITE_ONLY);

// 4. Загрузка данных
input->WriteToGPU(my_data);

// 5. Выполнение
engine.ExecuteKernel(kernel, {input->Get(), output->Get()},
                    {1024, 1, 1}, {256, 1, 1});

// 6. Синхронизация
engine.Finish();

// 7. Чтение результатов
auto result = output->ReadFromGPU();

// 8. Статистика
std::cout << engine.GetStatistics();
```

### С типизированными данными

```cpp
// Для структур данных
std::vector<DelayParameter> delays = {...};
auto delay_buffer = engine.CreateTypedBufferWithData(delays,
                        gpu::MemoryType::GPU_READ_ONLY);
```

### Асинхронные операции

```cpp
// Асинхронная запись
cl_event write_event = input->WriteToGPUAsync(data);

// Асинхронное выполнение
cl_event kernel_event = engine.ExecuteKernelAsync(kernel, buffers,
                         global_size, local_size);

// Ожидание
engine.WaitForEvents({write_event, kernel_event});
```

## Интеграция с другими компонентами

### С GeneratorGPU

```cpp
// GeneratorGPU использует OpenCLComputeEngine для:
- LoadProgram()     // Загрузка kernel'ов ЛЧМ
- CreateBuffer()    // Выделение памяти под сигналы
- ExecuteKernel()   // Запуск генерации
- Finish()          // Синхронизация перед чтением
```

### С GPUMemoryBuffer

```cpp
// GPUMemoryBuffer создается через:
auto buffer = engine.CreateBuffer(size, type);
// engine отслеживает статистику
// buffer автоматически очищается
```

### С KernelProgram

```cpp
// KernelProgram интегрируется через:
auto program = engine.LoadProgram(source);  // Кэширование
auto kernel = engine.GetKernel(program, name);  // Получение
// engine координирует жизненный цикл
```

## Обработка ошибок

### Валидация

- Проверка инициализации во всех методах
- Валидация размеров буферов
- Проверка типов памяти

### Исключения

- `std::runtime_error` при OpenCL ошибках
- Подробные сообщения с кодами ошибок
- Graceful degradation при ошибках

## Расширения

### Потенциальные улучшения

1. **Многопоточная инициализация**
2. **Поддержка нескольких устройств**
3. **Профилирование производительности**
4. **Отложенная компиляция kernels**
5. **Интеграция с OpenCL 3.0 features**

### Совместимость

- **OpenCL 1.2+** required
- Поддержка CPU/GPU устройств
- Кроссплатформенная (Windows/Linux/macOS)