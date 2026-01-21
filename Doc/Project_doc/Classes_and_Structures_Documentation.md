# Документация классов и структур проекта LCH-Farrow01

## Обзор архитектуры

Проект представляет собой систему для генерации ЛЧМ (линейно-частотно-модулированных) сигналов на GPU с использованием OpenCL. Архитектура построена на принципах разделения ответственности и использования паттерна Singleton для управления ресурсами.

## Схемы взаимодействия

### Основная архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│                    ПОЛЬЗОВАТЕЛЬСКИЙ КОД                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    GeneratorGPU                             │ │
│  │  (Генератор ЛЧМ сигналов)                                   │ │
│  └─────────────────────┬───────────────────────────────────────┘ │
└───────────────────────┼─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                 OpenCLComputeEngine                             │
│              (Главный фасад системы)                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │ │
│  │  │ OpenCLCore  │ │CommandQueue│ │KernelProgram│            │ │
│  │  │ (Контекст)  │ │   Pool     │ │ (Программы) │            │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 GPUMemoryBuffer                             │ │
│  │            (Управление GPU памятью)                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Альтернативная архитектура (OpenCLManager)

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenCLManager                                │
│              (Альтернативный менеджер)                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │ │
│  │  │  Context    │ │   Queue     │ │  Programs   │            │ │
│  │  │             │ │             │ │   Cache     │            │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 GPUMemoryBuffer                             │ │
│  │            (Управление GPU памятью)                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Поток данных при генерации сигнала

```
LFMParameters ──► GeneratorGPU ──► OpenCLComputeEngine ──► KernelProgram
       │                │                        │                │
       │                │                        │                ▼
       │                │                        │          kernel_lfm_basic
       │                │                        │          kernel_lfm_delayed
       │                │                        │          kernel_lfm_combined
       │                │                        │          kernel_sinusoid_combined
       │                │                        ▼                │
       │                │              GPUMemoryBuffer            │
       │                │                   │                     │
       │                │                   ▼                     │
       │                │             GPU Buffer                 │
       │                │                   │                     │
       │                │                   ▼                     │
       │                │             clEnqueueNDRangeKernel      │
       │                │                   │                     │
       │                │                   ▼                     │
       │                │             Результат на GPU           │
       │                │                   │                     │
       │                │                   ▼                     │
       │                └────────►  clEnqueueReadBuffer  ◄────────┘
                                    │
                                    ▼
                              CPU Vector
```

## Структуры данных

### LFMParameters
**Файл:** `include/interface/lfm_parameters.h`

Параметры линейно-частотно-модулированного сигнала.

**Поля:**
- `float f_start` - Начальная частота (Гц)
- `float f_stop` - Конечная частота (Гц)
- `float sample_rate` - Частота дискретизации (Гц)
- `mutable float duration` - Длительность сигнала (сек)
- `size_t num_beams` - Количество лучей
- `float steering_angle` - Базовый угол (градусы)
- `float angle_step_deg` - Шаг по углу (градусы)
- `float angle_start_deg` - Начальный угол (градусы)
- `float angle_stop_deg` - Конечный угол (градусы)
- `mutable size_t count_points` - Количество точек
- `bool apply_heterodyne` - Применять ли гетеродин

**Методы:**
- `bool IsValid() const noexcept` - Проверка валидности параметров
- `float GetChirpRate() const noexcept` - Расчет скорости изменения частоты
- `size_t GetNumSamples() const noexcept` - Получить количество отсчетов
- `float GetWavelength() const noexcept` - Расчет длины волны
- `void SetAngle(float, float)` - Установка углов

### DelayParameter
**Файл:** `include/interface/DelayParameter.h`

Параметры задержки для отдельного луча.

**Поля:**
- `uint32_t beam_index` - Индекс луча (0-255)
- `float delay_degrees` - Задержка в градусах

### CombinedDelayParam
**Файл:** `include/interface/combined_delay_param.h`

Комбинированные параметры задержки (угловая + временная).

**Поля:**
- `size_t beam_index` - Индекс луча
- `float delay_degrees` - Задержка по углу (градусы)
- `float delay_time_ns` - Задержка по времени (наносекунды)

### SinusoidParameter
**Файл:** `include/interface/lfm_parameters.h`

Параметры отдельной синусоиды.

**Поля:**
- `float amplitude` - Амплитуда
- `float period` - Период в точках
- `float phase_deg` - Фаза в градусах

### RaySinusoidParams
**Файл:** `src/generator/generator_gpu_new.cpp`

Параметры синусоид для конкретного луча (используется в OpenCL).

**Поля:**
- `cl_uint ray_index` - Индекс луча
- `cl_uint num_sinusoids` - Количество синусоид
- `SinusoidParam sinusoids[10]` - Массив параметров (макс 10)

## Классы

### OpenCLCore
**Файл:** `include/GPU/opencl_core.hpp`, `src/GPU/opencl_core.cpp`

Singleton для управления единым OpenCL контекстом.

**Ответственность:**
- Инициализация платформы и устройства
- Создание и владение OpenCL контекстом
- Информация об устройстве
- Thread-safe доступ к контексту

**Основные методы:**
- `static void Initialize(DeviceType)` - Инициализация
- `static OpenCLCore& GetInstance()` - Получение экземпляра
- `cl_context GetContext()` - Получение контекста
- `std::string GetDeviceInfo()` - Информация об устройстве

### CommandQueuePool
**Файл:** `include/GPU/command_queue_pool.hpp`, `src/GPU/command_queue_pool.cpp`

Singleton для управления пулом командных очередей OpenCL.

**Ответственность:**
- Создание пула командных очередей
- Асинхронное выполнение (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
- Балансировка нагрузки (round-robin)
- Thread-safe доступ

**Основные методы:**
- `static void Initialize(size_t)` - Инициализация пула
- `static cl_command_queue GetNextQueue()` - Получение следующей очереди
- `static void FinishAll()` - Синхронизация всех очередей

### KernelProgram
**Файл:** `include/GPU/kernel_program.hpp`, `src/GPU/kernel_program.cpp`

Управление OpenCL программами и kernels с кэшированием.

**Ответственность:**
- Компиляция OpenCL программ
- Кэширование программ по хэшу исходника
- Кэширование kernels по имени
- Получение информации о kernel

**Основные методы:**
- `KernelProgram(const std::string&)` - Создание из исходника
- `cl_kernel GetOrCreateKernel(const std::string&)` - Получение kernel
- `cl_program GetProgram()` - Получение программы

### KernelProgramCache
**Файл:** `include/GPU/kernel_program.hpp`

Глобальный кэш откомпилированных программ.

**Ответственность:**
- Кэширование программ по хэшу исходника
- Предотвращение перекомпиляции одинаковых программ

**Основные методы:**
- `static std::shared_ptr<KernelProgram> GetOrCompile(const std::string&)` - Получение или компиляция

### GPUMemoryBuffer
**Файл:** `include/GPU/gpu_memory_buffer.hpp`, `src/GPU/gpu_memory_buffer.cpp`

Обёртка над GPU памятью с поддержкой различных типов владения.

**Ответственность:**
- Управление GPU памятью (cl_mem)
- Чтение/запись данных между CPU и GPU
- Поддержка owning/non-owning буферов
- Асинхронные операции

**Конструкторы:**
- Owning: `GPUMemoryBuffer(context, queue, num_elements, type)`
- Non-owning: `GPUMemoryBuffer(context, queue, external_buffer, num_elements, type)`
- С данными: `GPUMemoryBuffer(context, queue, host_data, size, num_elements, type)`

**Основные методы:**
- `std::vector<std::complex<float>> ReadFromGPU()` - Чтение всех данных
- `void WriteToGPU(const std::vector<std::complex<float>>&)` - Запись данных
- `cl_mem Get()` - Получение cl_mem для OpenCL API

**Детальная документация:** [`Doc/GPUMemoryBuffer_Detailed.md`](Doc/GPUMemoryBuffer_Detailed.md)

### GPUMemoryManager
**Файл:** `include/GPU/gpu_memory_manager.hpp`, `src/GPU/gpu_memory_manager.cpp`

Singleton для централизованного управления GPU буферами.

**Ответственность:**
- Создание и управление буферами
- Статистика использования памяти
- Интеграция с OpenCLManager

**Основные методы:**
- `static void Initialize()` - Инициализация
- `static std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t, MemoryType)` - Создание буфера
- `static void PrintStatistics()` - Статистика

### OpenCLComputeEngine
**Файл:** `include/GPU/opencl_compute_engine.hpp`, `src/GPU/opencl_compute_engine.cpp`

Главный фасад для работы с OpenCL, объединяет все компоненты.

**Ответственность:**
- Единая точка входа для OpenCL операций
- Управление всеми подсистемами
- Предоставление высокоуровневого API

**Основные методы:**
- `static void Initialize(DeviceType)` - Инициализация
- `std::shared_ptr<KernelProgram> LoadProgram(const std::string&)` - Загрузка программы
- `std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t, MemoryType)` - Создание буфера
- `void ExecuteKernel(cl_kernel, const std::vector<cl_mem>&, ...)` - Выполнение kernel

**Детальная документация:** [`Doc/OpenCLComputeEngine_Detailed.md`](Doc/OpenCLComputeEngine_Detailed.md)

### OpenCLManager
**Файл:** `include/GPU/opencl_manager.h`, `src/GPU/opencl_manager.cpp`

Альтернативный менеджер OpenCL с кэшированием программ и kernels.

**Ответственность:**
- Управление OpenCL ресурсами
- Кэширование программ и kernels
- Управление памятью GPU
- Регистрация буферов

**Основные методы:**
- `static void Initialize(cl_device_type)` - Инициализация
- `cl_program GetOrCompileProgram(const std::string&)` - Компиляция программы
- `cl_kernel GetOrCreateKernel(cl_program, const std::string&)` - Создание kernel
- `std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t, MemoryType)` - Создание буфера

### GeneratorGPU
**Файл:** `include/generator/generator_gpu_new.h`, `src/generator/generator_gpu_new.cpp`

Генератор ЛЧМ сигналов на GPU.

**Ответственность:**
- Генерация различных типов сигналов на GPU
- Управление параметрами сигнала
- Интеграция с OpenCLComputeEngine

**Основные методы:**
- `GeneratorGPU(const LFMParameters&)` - Конструктор
- `cl_mem signal_base()` - Генерация базового ЛЧМ сигнала
- `cl_mem signal_valedation(const DelayParameter*, size_t)` - Сигнал с задержками
- `cl_mem signal_combined_delays(const CombinedDelayParam*, size_t)` - Комбинированные задержки
- `cl_mem signal_sinusoids(const SinusoidGenParams&, const RaySinusoidMap&)` - Сумма синусоид
- `std::vector<std::complex<float>> GetSignalAsVector(int)` - Чтение сигнала луча

**Детальная документация:** [`Doc/GeneratorGPU_Detailed.md`](Doc/GeneratorGPU_Detailed.md)

### AntennaFFTProcMax
**Файлы:** `include/fft/antenna_fft_proc_max.h`, `src/fft/antenna_fft_proc_max.cpp`

Высокопроизводительный класс для FFT обработки сигналов с поиском максимальных амплитуд на GPU.

**Ответственность:**
- Выполнение быстрого преобразования Фурье (FFT) с использованием clFFT
- Поиск топ-N максимумов амплитуд в спектральном диапазоне
- Полностью GPU-ориентированная обработка (все операции поиска максимумов выполняются на GPU)
- Использование callback'ов clFFT для оптимизации производительности
- Детальное профилирование всех этапов обработки
- Кэширование планов FFT для переиспользования

**Архитектура:**
- Pre-callback: подготовка данных (перенос + padding до размера nFFT)
- FFT: преобразование Фурье с использованием clFFT
- Post-callback: fftshift + вычисление magnitude/phase
- Reduction kernel: параллельный поиск топ-N максимумов на GPU
- Поддержка batch-обработки для нескольких лучей

**Основные методы:**
- `AntennaFFTProcMax(const AntennaFFTParams&)` - Конструктор с параметрами
- `AntennaFFTResult Process(cl_mem)` - Основная обработка с GPU буфером
- `AntennaFFTResult Process(const std::vector<std::complex<float>>&)` - Обработка с CPU данными
- `void PrintResults(const AntennaFFTResult&)` - Вывод результатов в консоль
- `void SaveResultsToFile(const AntennaFFTResult&, const std::string&)` - Сохранение в файл (Markdown + JSON)
- `std::string GetProfilingStats()` - Получение статистики профилирования
- `void UpdateParams(const AntennaFFTParams&)` - Обновление параметров

**Использование:**
```cpp
// Инициализация OpenCL
gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);

// Создание процессора
AntennaFFTParams params(5, 1000, 512, 3); // 5 лучей, 1000 точек, 512 выходных, 3 максимума
AntennaFFTProcMax processor(params);

// Обработка
cl_mem input_signal = ...; // GPU буфер с комплексными данными
AntennaFFTResult result = processor.Process(input_signal);

// Вывод результатов
processor.PrintResults(result);
processor.SaveResultsToFile(result, "Reports/result.md");
```

## Перечисления

### DeviceType
**Файл:** `include/GPU/opencl_core.hpp`

Тип устройства OpenCL.

**Значения:**
- `GPU` - CL_DEVICE_TYPE_GPU
- `CPU` - CL_DEVICE_TYPE_CPU

### MemoryType
**Файл:** `include/GPU/memory_type.hpp`

Тип памяти GPU буфера.

**Значения:**
- `GPU_READ_ONLY` - Только чтение с GPU
- `GPU_WRITE_ONLY` - Только запись на GPU
- `GPU_READ_WRITE` - Чтение и запись

## Взаимодействия между компонентами

### Инициализация системы

```
Пользователь
    │
    ▼
OpenCLCore::Initialize() ──► Создание контекста
    │
    ▼
CommandQueuePool::Initialize() ──► Создание очередей
    │
    ▼
OpenCLComputeEngine::Initialize() ──► Инициализация фасада
    │
    ▼
GeneratorGPU(params) ──► Создание генератора
```

### Генерация сигнала

```
GeneratorGPU::signal_base()
    │
    ▼
OpenCLComputeEngine::CreateBuffer() ──► Выделение GPU памяти
    │
    ▼
OpenCLComputeEngine::LoadProgram() ──► Загрузка kernel программы
    │
    ▼
OpenCLComputeEngine::GetKernel() ──► Получение kernel
    │
    ▼
OpenCLComputeEngine::ExecuteKernel() ──► Выполнение на GPU
    │
    ▼
CommandQueuePool::GetNextQueue() ──► Получение очереди
    │
    ▼
clEnqueueNDRangeKernel() ──► Запуск kernel
```

### Чтение результатов

```
GeneratorGPU::GetSignalAsVector(beam_index)
    │
    ▼
GeneratorGPU::ClearGPU() ──► Синхронизация
    │
    ▼
GPUMemoryBuffer::ReadFromGPU() ──► Чтение данных
    │
    ▼
clEnqueueReadBuffer() ──► DMA transfer
```

## Управление ресурсами

### RAII и владение

- **OpenCLCore**: Владеет контекстом, платформой, устройством
- **CommandQueuePool**: Владеет командными очередями
- **KernelProgram**: Владеет cl_program и kernels
- **GPUMemoryBuffer**: Может владеть cl_mem (owning) или использовать внешний (non-owning)
- **OpenCLComputeEngine**: Координирует все компоненты

### Автоматическая очистка

- Деструкторы классов автоматически освобождают ресурсы
- Использование unique_ptr/shared_ptr для управления памятью
- Singleton'ы очищаются при завершении программы

### Thread safety

- Singleton'ы используют double-checked locking или static local initialization
- CommandQueuePool thread-safe для получения очередей
- Кэши защищены mutex'ами

## Производительность

### Оптимизации

- **Кэширование программ**: Избегание перекомпиляции одинаковых kernels
- **Кэширование kernels**: Быстрое получение часто используемых kernels
- **Пул очередей**: Асинхронное выполнение и балансировка нагрузки
- **Pinned memory**: Быстрый DMA transfer между CPU и GPU

### Статистика

- OpenCLComputeEngine предоставляет подробную статистику
- GPUMemoryManager отслеживает использование памяти
- KernelProgramCache показывает эффективность кэширования

## Файлы документации

Для детального изучения основных компонентов доступны специализированные файлы:

- **GeneratorGPU**: [`Doc/GeneratorGPU_Detailed.md`](Doc/GeneratorGPU_Detailed.md) - Полная документация генератора сигналов
- **OpenCLComputeEngine**: [`Doc/OpenCLComputeEngine_Detailed.md`](Doc/OpenCLComputeEngine_Detailed.md) - Детали главного фасада
- **GPUMemoryBuffer**: [`Doc/GPUMemoryBuffer_Detailed.md`](Doc/GPUMemoryBuffer_Detailed.md) - Управление GPU памятью

## Файлы документации

Для детального изучения основных компонентов доступны специализированные файлы:

- **GeneratorGPU**: [`Doc/GeneratorGPU_Detailed.md`](Doc/GeneratorGPU_Detailed.md) - Полная документация генератора сигналов
- **OpenCLComputeEngine**: [`Doc/OpenCLComputeEngine_Detailed.md`](Doc/OpenCLComputeEngine_Detailed.md) - Детали главного фасада
- **GPUMemoryBuffer**: [`Doc/GPUMemoryBuffer_Detailed.md`](Doc/GPUMemoryBuffer_Detailed.md) - Управление GPU памятью
- **ООП и Паттерны**: [`Doc/OOP_SOLID_Patterns_Reference.md`](Doc/OOP_SOLID_Patterns_Reference.md) - Справочник по ООП, SOLID, GRASP и GoF паттернам

## Структура файлов проекта

```
Doc/
├── Classes_and_Structures_Documentation.md    # Этот файл (обзор)
├── GeneratorGPU_Detailed.md                    # Детали GeneratorGPU
├── OpenCLComputeEngine_Detailed.md             # Детали OpenCLComputeEngine
├── GPUMemoryBuffer_Detailed.md                 # Детали GPUMemoryBuffer
├── OOP_SOLID_Patterns_Reference.md             # Справочник ООП и паттернов
└── ...

include/
├── generator/generator_gpu_new.h               # GeneratorGPU
├── GPU/
│   ├── opencl_compute_engine.hpp               # Главный фасад
│   ├── gpu_memory_buffer.hpp                   # Управление памятью
│   ├── command_queue_pool.hpp                  # Пул очередей
│   ├── kernel_program.hpp                      # Программы и kernels
│   ├── opencl_core.hpp                         # Базовый контекст
│   └── ...
└── interface/
    ├── lfm_parameters.h                        # Параметры сигнала
    ├── DelayParameter.h                        # Параметры задержки
    └── combined_delay_param.h                  # Комбинированные задержки

src/
├── generator/generator_gpu_new.cpp             # Реализация GeneratorGPU
├── GPU/
│   ├── gpu_memory_manager.cpp                  # Менеджер памяти
│   ├── opencl_manager.cpp                      # Альтернативный менеджер
│   └── ...
└── main.cpp                                    # Точка входа
```