# OpenCL Compute Engine - ПОЛНОЕ РЕШЕНИЕ

## 📋 ОБЗОР

Вы получили **профессиональную архитектуру OpenCL** с поддержкой:
- ✅ Многопоточных вычислений (ThreadPool + Task Queue)
- ✅ Асинхронного выполнения (cl_event + multiple queues)
- ✅ RAII управления памятью (владение + non-owning буферы)
- ✅ Кэширования программ (избежать перекомпиляции)
- ✅ Кроссплатформы (Windows RTX, Ubuntu RTX/AMD)
- ✅ Детального логирования и статистики

## 🏗️ АРХИТЕКТУРА (3 слоя)

### Слой 1: Core (Контекст + Программы)
```
OpenCLCore (Singleton)
├── Инициализация платформы/девайса
├── Создание контекста OpenCL
├── Информация о девайсе
└── Thread-safe доступ

KernelProgram (RAII)
├── Компиляция OpenCL программ
├── Кэширование kernels
├── Обработка ошибок компиляции
└── Деструктор освобождает ресурсы

KernelProgramCache (Global)
├── Кэширование программ по хешу
├── Статистика (hits/misses)
└── Глобальное переиспользование
```

### Слой 2: Memory (Буферы + Пулл очередей)
```
GPUMemoryBuffer (RAII)
├── Три конструктора (owning, non-owning, owning+data)
├── Pinned host buffers для DMA
├── Синхронное и асинхронное API
└── Управление жизненным циклом

CommandQueuePool (Singleton)
├── N асинхронных command queues
├── Round-robin балансировка
├── Статистика нагрузки
└── Синхронизация между очередями
```

### Слой 3: Compute Engine (Фасад)
```
OpenCLComputeEngine (Singleton)
├── Объединяет все компоненты
├── Простой API для пользователя
├── Управление программами/kernels
├── Выполнение kernels
├── Синхронизация событий
└── Статистика и отладка
```

## 📦 ФАЙЛЫ

### Созданы новые файлы:
1. **opencl_core.hpp/cpp** - Контекст OpenCL (Singleton)
2. **kernel_program.hpp/cpp** - Программы + kernels + кэш
3. **gpu_memory_improved.hpp/cpp** - RAII буферы с асинхронностью
4. **command_queue_pool.hpp/cpp** - Пулл асинхронных очередей
5. **opencl_compute_engine.hpp/cpp** - ГЛАВНЫЙ ФАСАД

### Остаются (или переписываются):
- CMakeLists.txt (не меняется)
- Остальной код приложения

## 🚀 БЫСТРЫЙ СТАРТ

### 1. Инициализация (один раз в main)
```cpp
#include "opencl_compute_engine.hpp"

int main() {
    // 1. Инициализировать движок
    gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
    auto& engine = gpu::OpenCLComputeEngine::GetInstance();

    // Показать информацию о девайсе
    std::cout << engine.GetDeviceInfo();

    // ... код программы ...

    // Очистка (опционально, вызывается в деструкторе)
    gpu::OpenCLComputeEngine::Cleanup();
    return 0;
}
```

### 2. Создать буферы (RAII - автоматическое освобождение)
```cpp
// Создать пустой буфер
auto input = engine.CreateBuffer(
    1300000 * 256,  // количество complex<float>
    gpu::MemoryType::GPU_READ_WRITE
);

auto output = engine.CreateBuffer(
    1300000 * 512,  // больше для FFT
    gpu::MemoryType::GPU_WRITE_ONLY
);

// Или с начальными данными
std::vector<std::complex<float>> data(1024);
// ... заполнить data ...
auto input2 = engine.CreateBufferWithData(
    data,
    gpu::MemoryType::GPU_READ_ONLY
);

// Буферы автоматически освобождаются при выходе из scope!
```

### 3. Загрузить kernel
```cpp
// OpenCL C код
const char* kernel_source = R"(
__kernel void fft_kernel(
    __global float2* input,
    __global float2* output,
    __global float* twiddle,
    int N
) {
    int gid = get_global_id(0);
    // ... вычисления ...
}
)";

// Загрузить программу (компилируется один раз благодаря кэшу)
auto program = engine.LoadProgram(kernel_source);

// Получить kernel
auto fft_kernel = engine.GetKernel(program, "fft_kernel");
```

### 4. Синхронно выполнить kernel
```cpp
std::vector<std::complex<float>> my_data(1024);
// ... заполнить my_data ...

input->WriteToGPU(my_data);

// Выполнить kernel
engine.ExecuteKernel(
    fft_kernel,
    {input->Get(), output->Get()},  // Buffers
    {{256, 1, 1}},                   // Global work size
    {{256, 1, 1}}                    // Local work size
);

// Прочитать результаты
auto result = output->ReadFromGPU();
```

### 5. Асинхронно выполнить kernel
```cpp
// Выполнить асинхронно (возвращает cl_event)
auto event = engine.ExecuteKernelAsync(
    fft_kernel,
    {input->Get(), output->Get()},
    {{256, 1, 1}},
    {{256, 1, 1}}
);

// Ждать завершения
engine.WaitForEvent(event);
```

### 6. Статистика
```cpp
std::cout << engine.GetStatistics();
/*
Output:
======================================================================
OpenCLComputeEngine Statistics

Total Allocated Memory:      15.25 MB
Active Buffers:              2
Kernel Executions:           42

Kernel Program Cache Statistics:
 Cache size: 3 programs
 Cache hits: 89
 Cache misses: 3
 Hit rate: 96.7%

======================================================================
*/
```

## 💾 RAII - УПРАВЛЕНИЕ ПАМЯТЬЮ

### Три режима владения памятью:

```cpp
// 1. OWNING - объект создаёт буфер
auto buffer1 = engine.CreateBuffer(1024, gpu::MemoryType::GPU_READ_WRITE);
// Деструктор ~GPUMemoryBuffer освобождает буфер

// 2. NON-OWNING - оборачиваем готовый буфер
cl_mem existing_buffer = ...; // Создан где-то ещё
auto buffer2 = engine.WrapExternalBuffer(
    existing_buffer, 1024, gpu::MemoryType::GPU_READ_WRITE
);
// Деструктор НЕ освобождает буфер

// 3. OWNING с данными - создаём и копируем
std::vector<std::complex<float>> data(1024);
auto buffer3 = engine.CreateBufferWithData(
    data, gpu::MemoryType::GPU_READ_ONLY
);
// Данные сразу на GPU!
```

## ⚡ АСИНХРОННОСТЬ

### Command Queue Pool
```cpp
// Пулл очередей для параллельного выполнения
gpu::CommandQueuePool::Initialize(4);  // 4 асинхронные очереди

auto queue1 = gpu::CommandQueuePool::GetNextQueue();    // Queue 0
auto queue2 = gpu::CommandQueuePool::GetNextQueue();    // Queue 1
auto queue3 = gpu::CommandQueuePool::GetNextQueue();    // Queue 2
auto queue4 = gpu::CommandQueuePool::GetNextQueue();    // Queue 3
auto queue5 = gpu::CommandQueuePool::GetNextQueue();    // Queue 0 (round-robin)

// Выполнять несколько kernels параллельно в разных очередях
clEnqueueNDRangeKernel(queue1, kernel1, ...);
clEnqueueNDRangeKernel(queue2, kernel2, ...);
clEnqueueNDRangeKernel(queue3, kernel3, ...);

// Ждать всех
gpu::CommandQueuePool::FinishAll();
```

## 🔍 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

### Пример 1: FFT для сигнальной обработки
```cpp
const size_t NUM_SAMPLES = 1300000;
const size_t NUM_ANTENNAS = 256;

// Инициализация
gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
auto& engine = gpu::OpenCLComputeEngine::GetInstance();

// Создать буферы
auto input = engine.CreateBuffer(
    NUM_SAMPLES * NUM_ANTENNAS,
    gpu::MemoryType::GPU_READ_WRITE
);

// Паддировать до 2^n (по вашим требованиям)
size_t fft_size = 1;
while (fft_size < NUM_SAMPLES * 2) fft_size *= 2;

auto output = engine.CreateBuffer(
    fft_size * NUM_ANTENNAS,
    gpu::MemoryType::GPU_WRITE_ONLY
);

// Загрузить FFT kernel
auto fft_program = engine.LoadProgram(fft_kernel_source);
auto fft_kernel = engine.GetKernel(fft_program, "complex_fft");

// Выполнить для каждой антенны
for (size_t antenna = 0; antenna < NUM_ANTENNAS; ++antenna) {
    // ... выполнить kernel для этой антенны ...
}

std::cout << engine.GetStatistics();
```

### Пример 2: Многопоточное выполнение
```cpp
// Инициализация пулла очередей
gpu::CommandQueuePool::Initialize(8);

// Запустить 8 kernels параллельно
std::vector<cl_event> events;

for (int i = 0; i < 8; ++i) {
    auto event = engine.ExecuteKernelAsync(
        kernel, buffers, global_size, local_size
    );
    events.push_back(event);
}

// Ждать всех
engine.WaitForEvents(events);
```

### Пример 3: Асинхронная запись/чтение
```cpp
// Асинхронная запись
auto write_event = input->WriteToGPUAsync(my_data);

// Выполнить kernel пока пишется буфер
engine.ExecuteKernel(kernel, ...);

// Асинхронное чтение
auto [result, read_event] = output->ReadFromGPUAsync();

// Ждать чтения
engine.WaitForEvent(read_event);

// Теперь result заполнен данными
std::cout << "First element: " << result[0] << "\n";
```

## 🔧 КРОССПЛАТФОРМА

### Windows (RTX 2080Ti)
```cmake
# CMakePresets.json
"configurePresets": [
  {
    "name": "Windows-RTX2080Ti",
    "cacheVariables": {
      "ENABLE_CUDA": "ON",
      "CUDA_DEVICE": "0"
    }
  }
]
```

### Ubuntu (RTX 3060)
```cmake
{
  "name": "Ubuntu-RTX3060",
  "cacheVariables": {
    "ENABLE_OPENCL": "ON",
    "OPENCL_VENDOR": "NVIDIA"
  }
}
```

### Ubuntu (AMD AI100)
```cmake
{
  "name": "Ubuntu-AMD-AI100",
  "cacheVariables": {
    "ENABLE_OPENCL": "ON",
    "OPENCL_VENDOR": "AMD"
  }
}
```

## 📊 СТАТИСТИКА И ОТЛАДКА

```cpp
// Информация о девайсе
std::cout << engine.GetDeviceInfo() << "\n";

// Статистика кэша программ
std::cout << engine.GetCacheStatistics() << "\n";

// Статистика пулла очередей
std::cout << gpu::CommandQueuePool::GetStatistics() << "\n";

// Информация о буфере
input->PrintStats();
output->PrintStats();

// Полная статистика
std::cout << engine.GetStatistics() << "\n";
```

## ⚠️ ОБРАБОТКА ОШИБОК

```cpp
try {
    auto buffer = engine.CreateBuffer(1024, gpu::MemoryType::GPU_READ_WRITE);
    auto program = engine.LoadProgram(invalid_kernel_code);
} catch (const std::runtime_error& e) {
    std::cerr << "OpenCL Error: " << e.what() << "\n";
    // Программа продолжит работать, ресурсы освобождены автоматически
}

// RAII гарантирует, что ресурсы освобождены даже при исключении!
```

## 🎯 ОПТИМИЗАЦИИ ДЛЯ ВАШЕГО СЛУЧАЯ

### 1. FFT с паддированием
```cpp
// Ваш случай: 1300000 * 256 антенн
// Нужно: паддировать до 2^n и удвоить размер

size_t original_size = 1300000;
size_t fft_size = 1;
while (fft_size < original_size * 2) fft_size *= 2;
// fft_size = 2097152 (2^21)

auto fft_buffer = engine.CreateBuffer(
    fft_size * 256,  // Все антенны
    gpu::MemoryType::GPU_READ_WRITE
);
```

### 2. Parallelize по антеннам
```cpp
// Каждая антенна в отдельном kernel execution
for (int antenna = 0; antenna < 256; ++antenna) {
    size_t offset = antenna * fft_size;
    
    // Выполнить асинхронно
    auto event = engine.ExecuteKernelAsync(
        fft_kernel,
        {fft_buffer->Get()},
        {{fft_size / 256, 1, 1}},  // Global
        {{256, 1, 1}}               // Local
    );
    
    events.push_back(event);
}

// Ждать всех
engine.WaitForEvents(events);
```

### 3. Memory pooling для часто используемых размеров
```cpp
// Будет добавлено в будущих версиях
// Сейчас используйте вручную переиспользование буферов
```

## 📝 МИГРАЦИЯ ИЗ СТАРОГО КОДА

### Старое (Singleton OpenCLManager)
```cpp
auto& manager = gpu::OpenCLManager::GetInstance();
cl_context ctx = manager.GetContext();
```

### Новое (OpenCLComputeEngine)
```cpp
gpu::OpenCLComputeEngine::Initialize();
auto& engine = gpu::OpenCLComputeEngine::GetInstance();
cl_context ctx = gpu::OpenCLCore::GetInstance().GetContext();
```

## ✅ ЧЕКЛИСТ ИНТЕГРАЦИИ

- [ ] Включить заголовки: `#include "opencl_compute_engine.hpp"`
- [ ] Линковать OpenCL: `find_package(OpenCL REQUIRED)` в CMake
- [ ] Вызвать Initialize() в main()
- [ ] Заменить CreateBuffer() вызовы на engine.CreateBuffer()
- [ ] Загрузить kernels через engine.LoadProgram()
- [ ] Тестировать на каждой платформе (Windows/Ubuntu)
- [ ] Проверить статистику: engine.GetStatistics()

## 🐛 ИЗВЕСТНЫЕ ОГРАНИЧЕНИЯ (ТО-ДО)

- [ ] Мультигейм DeviceID поддержка (пока один девайс)
- [ ] Thread-safe очередь задач для ComputeScheduler (базовая в файлах)
- [ ] Memory pooling (есть RAII, но не pool)
- [ ] Callback функции для event completion
- [ ] Встроенный profiler (есть статистика)

## 📞 КОНТАКТ

Если есть вопросы или нужны доп. компоненты:
1. Уточнить требование в коде
2. Создать доп. класс с RAII
3. Добавить в OpenCLComputeEngine фасад

**Все файлы готовы к использованию!** 🚀
