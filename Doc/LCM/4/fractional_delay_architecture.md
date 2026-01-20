# FractionalDelayProcessor - Архитектура и Документация

## 📋 Обзор

**FractionalDelayProcessor** - это высокопроизводительный класс для обработки сигналов с дробной задержкой на GPU. Разработан по паттернам GRASP (General Responsibility Assignment Software Patterns) и GoF (Gang of Four).

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Application                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │Processor │◄─── FractionalDelayProcessor
                    └────┬────┘
                    ╱────┴────╲
                   ╱           ╲
        ┌─────────▼────┐  ┌─────▼───────┐
        │  OpenCL      │  │  Generator  │
        │  ComputeEngine◄──│  GPU        │
        └─────────┬────┘  └─────┬───────┘
                  │              │
        ┌─────────▼────────────────┴──┐
        │   GPU Memory Management     │
        │  (IMemoryBuffer Interface)  │
        └─────────┬────────────────┬──┘
                  │                │
        ┌─────────▼─┐    ┌────────▼───┐
        │ Regular   │    │ SVM Buffer │
        │ Buffer    │    │ (if avail.)│
        └───────────┘    └────────────┘
                  │                │
                  └────────┬────────┘
                           │
                    ┌──────▼──────┐
                    │  GPU VRAM   │
                    │   (Device)  │
                    └─────────────┘
```

## 🎯 Паттерны Проектирования

### 1. **Facade (Фасад)**
- **FractionalDelayProcessor** скрывает сложность архитектуры GPU
- Предоставляет простой интерфейс: `ProcessWithFractionalDelay()`
- Инкапсулирует работу с:
  - OpenCLComputeEngine (управление GPU)
  - GeneratorGPU (генерация сигналов)
  - GPU буферами (memory management)

### 2. **Strategy (Стратегия)**
- Выбор стратегии памяти делегируется **OpenCLComputeEngine**
- Поддерживаются:
  - Traditional Regular buffers (cl_mem)
  - SVM (Shared Virtual Memory) - если GPU поддерживает
  - Автоматический выбор оптимальной стратегии

### 3. **Factory (Фабрика)**
- **OpenCLComputeEngine::CreateBuffer()** создаёт буферы
- Выбирает оптимальную реализацию на основе GPU capabilities
- Кэширует программы через **KernelProgramCache**

### 4. **RAII (Resource Acquisition Is Initialization)**
- Все ресурсы управляются `unique_ptr`
- Автоматическое освобождение в деструкторе
- Безопасная работа с исключениями (exception-safe)

### 5. **Singleton (Синглтон)**
- **OpenCLComputeEngine** - глобальный единственный экземпляр
- Управляет всеми GPU ресурсами
- Thread-safe инициализация

## 📦 Компоненты

### FractionalDelayConfig

```cpp
struct FractionalDelayConfig {
    uint32_t num_beams = 256;           // Лучи (антенны)
    uint32_t num_samples = 8192;        // Отсчёты на луч
    uint32_t local_work_size = 256;     // GPU local work size
    bool verbose = true;                // Диагностика
    gpu::MemoryType result_memory_type; // Тип результатов
};
```

**Предустановки:**
- `FractionalDelayConfig::Standard()` - сбалансированная конфигурация
- `FractionalDelayConfig::Performance()` - максимальная производительность
- `FractionalDelayConfig::Diagnostic()` - с выводом информации

### ProcessingResult

```cpp
struct ProcessingResult {
    bool success;                    // Успех обработки
    std::string error_message;       // Сообщение об ошибке
    double gpu_execution_time_ms;    // Время GPU kernel'а
    double gpu_readback_time_ms;     // Время чтения с GPU
    double total_time_ms;            // Общее время
    uint32_t beams_processed;        // Кол-во обработанных лучей
    ComplexVector output_data;       // Результаты на CPU
};
```

### FractionalDelayProcessor

```cpp
class FractionalDelayProcessor {
public:
    // Инициализация
    FractionalDelayProcessor(const FractionalDelayConfig&, 
                            const LFMParameters&);
    ~FractionalDelayProcessor();
    
    // Основные методы
    ProcessingResult ProcessWithFractionalDelay(const DelayParameter&);
    std::vector<ProcessingResult> ProcessBatch(const std::vector<DelayParameter>&);
    
    // Диагностика
    void PrintInfo() const;
    std::string GetStatistics() const;
    bool IsInitialized() const;
    size_t GetGPUBufferSizeBytes() const;
};
```

## 🔄 Жизненный Цикл Обработки

```
ProcessWithFractionalDelay(delay_param)
    │
    ├─► 1. SyncGPU() - синхронизация GPU
    │
    ├─► 2. GeneratorGPU::signal_base() - генерировать сигнал
    │       └─► Лучи остаются на GPU в buffer_signal_base
    │
    ├─► 3. buffer_input_->Write(gen_data) - загрузить на GPU
    │       └─► Копия данных теперь в buffer_input
    │
    ├─► 4. Установить аргументы kernel:
    │       ├─► arg[0] = buffer_input (входные данные)
    │       ├─► arg[1] = buffer_output (выходные данные)
    │       ├─► arg[2] = delay_radians (параметр задержки)
    │       ├─► arg[3] = num_beams
    │       └─► arg[4] = num_samples
    │
    ├─► 5. engine->ExecuteKernel()
    │       └─► GPU: kernel_fractional_delay_optimized
    │           Каждый thread: output[idx] = input[idx] * exp(j*delay*sample_idx)
    │
    ├─► 6. SyncGPU() - дождаться завершения
    │
    ├─► 7. buffer_output->Read() - чтение результатов на CPU
    │
    └─► 8. Возврат ProcessingResult с данными и статистикой
```

## 💾 Управление Памятью

### GPU Буферы

```
┌─────────────────────────────────────┐
│     GPU Global Memory (VRAM)        │
├─────────────────────────────────────┤
│                                     │
│  buffer_input_                      │  │
│  ├─ num_beams × num_samples         │  │ Создается в
│  └─ sizeof(complex<float>) each     │  │ CreateBuffers()
│                                     │  │
│  buffer_output_                     │  │
│  ├─ num_beams × num_samples         │  │
│  └─ sizeof(complex<float>) each     │  │
│                                     │
└─────────────────────────────────────┘
```

### CPU Буферы

```cpp
// Результат существует только на CPU после ProcessWithFractionalDelay()
ProcessingResult::output_data  // ComplexVector (std::vector<complex<float>>)
    ├─ Выделяется на heap при ReadFromGPU()
    ├─ Автоматическое управление через ComplexVector (std::vector)
    └─ Освобождается при разрушении ProcessingResult
```

### Ключевые моменты

✅ **Данные остаются на GPU** в `buffer_input_` и `buffer_output_`
✅ **Данные выгружены на CPU** в `ProcessingResult::output_data`
✅ **Возможность переиспользования** одних и тех же буферов для нескольких обработок
✅ **Автоматическое управление** через `unique_ptr` и RAII

## 🚀 Пример Использования

### Инициализация

```cpp
// 1. Инициализировать OpenCL
gpu::OpenCLCore::Initialize(gpu::DeviceType::GPU);
gpu::CommandQueuePool::Initialize();
gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);

// 2. Параметры LFM
radar::LFMParameters lfm_params;
lfm_params.f_start = 100.0e6f;
lfm_params.f_stop = 500.0e6f;
lfm_params.num_beams = 256;
lfm_params.count_points = 8192;

// 3. Конфигурация процессора
auto config = radar::FractionalDelayConfig::Standard();
config.num_beams = lfm_params.num_beams;
config.num_samples = lfm_params.count_points;

// 4. Создать процессор
radar::FractionalDelayProcessor processor(config, lfm_params);
```

### Обработка

```cpp
// Параметр задержки: луч 0, задержка 0.5 градуса
radar::DelayParameter delay{0, 0.5f};

// Обработка
auto result = processor.ProcessWithFractionalDelay(delay);

// Проверка результата
if (result.success) {
    std::cout << "GPU time: " << result.gpu_execution_time_ms << " ms\n";
    
    // Получить один луч из результата
    auto beam_0 = result.GetBeam(0, lfm_params.count_points);
    for (size_t i = 0; i < beam_0.size(); ++i) {
        auto val = beam_0[i];
        std::cout << val.real() << " + j" << val.imag() << "\n";
    }
}
```

### Batch Обработка

```cpp
std::vector<radar::DelayParameter> delays{
    {0, 0.0f},
    {64, 0.5f},
    {128, 1.0f},
    {255, 1.5f}
};

auto results = processor.ProcessBatch(delays);

for (const auto& res : results) {
    if (res.success) {
        std::cout << "✅ Processed " << res.output_data.size() << " elements\n";
    }
}
```

## 📊 OpenCL Kernel

```c
__kernel void kernel_fractional_delay_optimized(
    __global float2 *input,      // Входные complex отсчеты
    __global float2 *output,     // Выходные complex отсчеты
    float delay_rad,             // Задержка в радианах
    uint num_beams,              // Кол-во лучей
    uint num_samples             // Отсчеты на луч
) {
    // 2D работа: (beam, sample)
    uint beam_idx = get_global_id(0);
    uint sample_idx = get_global_id(1);
    
    if (beam_idx >= num_beams || sample_idx >= num_samples) return;
    
    // Линейный адрес
    uint idx = beam_idx * num_samples + sample_idx;
    
    // Входные данные
    float2 input_val = input[idx];
    
    // Фазовый сдвиг: phase = delay_rad * sample_idx
    float phase = delay_rad * (float)sample_idx;
    
    // Умножение на комплексную экспоненту: exp(j*phase)
    float cos_phase = cos(phase);
    float sin_phase = sin(phase);
    
    // Результат = input * exp(j*phase)
    float2 output_val;
    output_val.x = input_val.x * cos_phase - input_val.y * sin_phase;
    output_val.y = input_val.x * sin_phase + input_val.y * cos_phase;
    
    output[idx] = output_val;
}
```

## 🔍 Проверка Данных

### На GPU
- Данные остаются в `buffer_input_` и `buffer_output_`
- Могут быть переиспользованы для следующих вызовов
- Требует явного синхронизирования (engine->Finish())

### На CPU
- Данные доступны в `ProcessingResult::output_data`
- Полный контроль над памятью на хосте
- Безопасно использовать в многопоточной программе

## 📈 Производительность

### Профилирование

```cpp
result.gpu_execution_time_ms;    // Kernel execution time
result.gpu_readback_time_ms;     // H2D transfer time
result.total_time_ms;            // CPU overhead + GPU time
```

### Оптимизация

1. **Переиспользование буферов** - каждый ProcessWithFractionalDelay() использует те же GPU буферы
2. **Batch обработка** - обработка нескольких задержек без переаллокации
3. **Асинхронные операции** - kernel выполняется параллельно с CPU работой
4. **SVM при наличии** - zero-copy доступ к памяти

## ❌ Обработка Ошибок

```cpp
// Валидация конфигурации
if (!config.IsValid()) {
    // num_beams, num_samples, local_work_size проверены
}

// Валидация параметров
if (!lfm_params.IsValid()) {
    // f_start, f_stop, sample_rate, num_beams проверены
}

// Проверка инициализации OpenCL
if (!gpu::OpenCLComputeEngine::IsInitialized()) {
    throw std::runtime_error("Initialize OpenCL first!");
}

// Обработка ошибок GPU
if (!result.success) {
    std::cerr << result.error_message << std::endl;
}
```

## 🧪 Тестирование (fractional_delay_example.cpp)

1. ✅ Инициализация OpenCL
2. ✅ Создание процессора
3. ✅ Обработка с одной задержкой
4. ✅ Batch обработка (несколько задержек)
5. ✅ Проверка данных CPU vs GPU
6. ✅ Статистика и профилирование
7. ✅ Переиспользование ресурсов

## 📝 Замечания

- **RAII**: Все ресурсы автоматически управляются
- **Exception-safe**: Безопасно при исключениях
- **Thread-safe**: GPU операции синхронизированы
- **Memory-safe**: Все буферы с проверкой границ
- **Zero-overhead abstraction**: Minimal CPU overhead
- **Scalable**: Работает с любым кол-вом лучей и отсчётов

## 🎓 Паттерны GRASP/GoF

| Паттерн | Компонент | Назначение |
|---------|-----------|-----------|
| **Facade** | FractionalDelayProcessor | Упрощение интерфейса |
| **Strategy** | MemoryStrategy (SVM/Regular) | Выбор способа хранения |
| **Factory** | OpenCLComputeEngine | Создание объектов |
| **RAII** | unique_ptr | Управление ресурсами |
| **Singleton** | OpenCLComputeEngine | Глобальное состояние GPU |

