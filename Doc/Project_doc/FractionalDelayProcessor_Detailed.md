# FractionalDelayProcessor - Детальная документация

## Обзор

`FractionalDelayProcessor` - высокопроизводительный процессор дробной задержки для LFM (линейно-частотно-модулированных) радарных сигналов на GPU. Использует интерполяцию Лагранжа 5-го порядка с предвычисленной матрицей коэффициентов 48×5 для точного вычисления дробной задержки сигналов.

## Архитектура

```
FractionalDelayProcessor
├── FractionalDelayConfig config_         // Конфигурация обработки
├── LagrangeMatrix lagrange_matrix_       // Матрица 48×5 коэффициентов
├── OpenCLComputeEngine* engine_          // Вычислительный движок (не владеет)
├── cl_kernel kernel_                     // OpenCL kernel для обработки
├── GPUMemoryBuffer* buffer_lagrange_     // Матрица Лагранжа на GPU
├── GPUMemoryBuffer* buffer_delays_       // Параметры задержки на GPU
├── GPUMemoryBuffer* buffer_temp_         // Временный буфер (IN-PLACE)
└── FDPProfilingResults last_profiling_   // Последние результаты профилирования
```

## Основные возможности

- ✅ **Дробная задержка** с точностью до 0.02 отсчёта (48 уровней)
- ✅ **IN-PLACE обработка** - экономия памяти GPU
- ✅ **Параллельная обработка** до 256 лучей одновременно
- ✅ **GPU профилирование** через OpenCL Events
- ✅ **Матрица Лагранжа 48×5** - предвычисленные коэффициенты
- ✅ **Два формата задержки**: отсчёты (samples) и градусы (degrees)
- ✅ **Производительность**: ~20 Gsamples/sec на RTX 3060

## Конструкторы и деструктор

### FractionalDelayProcessor(const FractionalDelayConfig& config, const LagrangeMatrix& lagrange_matrix)

**Описание:** Создает процессор с заданной конфигурацией и матрицей Лагранжа.

**Параметры:**
- `config` - Конфигурация обработки (число лучей, отсчётов, размер workgroup)
- `lagrange_matrix` - Матрица коэффициентов Лагранжа (48 строк × 5 столбцов)

**Исключения:**
- `std::invalid_argument` - если конфигурация или матрица невалидны
- `std::runtime_error` - если OpenCLComputeEngine не инициализирован

**Алгоритм инициализации:**
1. Валидация конфигурации через `config.IsValid()`
2. Валидация матрицы Лагранжа
3. Получение ссылки на `OpenCLComputeEngine::GetInstance()`
4. Компиляция OpenCL kernel'а
5. Создание GPU буферов (Lagrange, delays, temp)
6. Загрузка матрицы Лагранжа на GPU

### ~FractionalDelayProcessor()

**Описание:** Деструктор. Освобождает ресурсы GPU (kernel, буферы).

**Особенности:**
- Использует RAII для автоматической очистки `unique_ptr` буферов
- Освобождает OpenCL kernel через `clReleaseKernel()`

## Основные методы

### Process(cl_mem input_output_buffer_gpu, const DelayParams& delay)

**Описание:** Применить дробную задержку к одному лучу (IN-PLACE).

**Параметры:**
- `input_output_buffer_gpu` - GPU буфер с данными (вход и выход)
- `delay` - Параметры задержки для одного луча

**Возвращает:**
- `FDPProfilingResults` - Результаты профилирования (время kernel, throughput)

**Гарантии:**
- Данные остаются на GPU (не копируются на CPU)
- Exception-safe (все ошибки обрабатываются)
- GPU профилирование автоматическое

### Process(cl_mem input_output_buffer_gpu, const std::vector<DelayParams>& delays)

**Описание:** Применить разные задержки к нескольким лучам одновременно (IN-PLACE).

**Параметры:**
- `input_output_buffer_gpu` - GPU буфер с данными всех лучей
- `delays` - Вектор параметров задержки (один элемент на луч)

**Формат данных:**
```
buffer[beam0_sample0, beam0_sample1, ..., beam0_sampleN-1,
       beam1_sample0, beam1_sample1, ..., beam1_sampleN-1, ...]
```

**Размер:** `num_beams × num_samples` комплексных чисел

### Process(cl_mem input_output_buffer_gpu, const std::vector<float>& delays_degrees)

**Описание:** Перегрузка для задержек в градусах (автоматическая конвертация).

**Параметры:**
- `input_output_buffer_gpu` - GPU буфер с данными
- `delays_degrees` - Вектор задержек в градусах для каждого луча

**Конвертация:**
```
delay_samples = (delay_deg × π / 180) × (λ / c) × sample_rate
```

где:
- `λ` = длина волны = `c / f_center`
- `c` = скорость света (3.0e8 м/с)
- `f_center` = средняя частота = `(f_start + f_stop) / 2`

## Структуры данных

### FractionalDelayConfig

**Описание:** Конфигурация процессора дробной задержки.

```cpp
struct FractionalDelayConfig {
    uint32_t num_beams;        // Количество лучей (1..256)
    uint32_t num_samples;      // Количество отсчётов на луч (16+)
    uint32_t local_work_size;  // Размер workgroup (32..1024)
    bool enable_profiling;     // Включить профилирование GPU
    bool verbose;              // Диагностический вывод
};
```

**Фабричные методы:**
- `Standard()` - Стандартная конфигурация (256 лучей, 8K отсчётов)
- `Performance()` - Для максимальной производительности (512 лучей, 131K отсчётов)
- `Diagnostic()` - Для отладки (64 луча, 1K отсчётов, verbose=true)

### DelayParams

**Описание:** Параметры задержки для одного луча.

```cpp
struct DelayParams {
    int32_t  delay_integer;    // Целая часть задержки (отсчёты)
    uint32_t lagrange_row;     // Строка матрицы Лагранжа [0..47]
};
```

**Фабричные методы:**
- `FromSamples(float total_delay)` - Из общей задержки в отсчётах
  ```cpp
  delay_integer = floor(total_delay)
  lagrange_row = round((total_delay - delay_integer) * 47)
  ```
  
- `FromDegrees(float degrees, float sample_rate, float f_center, float c)` - Из градусов

### LagrangeMatrix

**Описание:** Матрица коэффициентов Лагранжа 48×5.

```cpp
struct LagrangeMatrix {
    std::array<std::array<float, 5>, 48> coefficients;  // 48 строк × 5 столбцов
    
    static LagrangeMatrix LoadFromJSON(const std::string& filepath);
    bool IsValid() const;
};
```

**Структура:**
- 48 строк соответствуют дробным частям: 0.00, 0.02, ..., 0.98
- 5 столбцов - коэффициенты для 5-точечной интерполяции
- Формат JSON: `{"data": [[...], [...], ...]}` или `{"lagrange_matrix": [[...], ...]}`

### FDPProfilingResults

**Описание:** Результаты профилирования GPU операции.

```cpp
struct FDPProfilingResults {
    double kernel_time_ms;      // Время выполнения kernel'а (ms)
    double readback_time_ms;    // Время копирования CPU↔GPU (ms) - обычно 0 для IN-PLACE
    double total_time_ms;       // Общее время операции (ms)
    
    double GetThroughput() const;  // samples/sec
};
```

## OpenCL Kernel

### fractional_delay_kernel

**Назначение:** Применить дробную задержку с интерполяцией Лагранжа.

**Параметры:**
1. `__global const Complex* input_buffer` - Входной буфер [num_beams × num_samples]
2. `__global Complex* output_buffer` - Выходной буфер (для IN-PLACE - temp)
3. `__global const float* lagrange_matrix` - Матрица Лагранжа [48 × 5]
4. `__global const DelayParams* delay_params` - Параметры задержки [num_beams]
5. `const uint num_beams` - Количество лучей
6. `const uint num_samples` - Количество отсчётов на луч

**Алгоритм:**
1. Определить луч и позицию: `beam_idx = global_id / num_samples`, `sample_idx = global_id % num_samples`
2. Получить параметры задержки для луча
3. Вычислить центр интерполяции: `center = sample_idx - delay_integer`
4. Загрузить 5 точек: `[center-1, center, center+1, center+2, center+3]` с граничными условиями
5. Применить интерполяцию: `result = Σ(L[k] × sample[k])`
6. Записать результат в выходной буфер

**Оптимизации:**
- Memory coalescing (последовательный доступ к данным)
- #pragma unroll для развёртывания циклов
- Symmetric boundary reflection (без if'ов в основном цикле)

## Примеры использования

### Пример 1: Базовая обработка

```cpp
#include "GPU/fractional_delay_processor.hpp"

// Инициализация OpenCL
gpu::OpenCLCore::Initialize(gpu::DeviceType::GPU);
gpu::CommandQueuePool::Initialize();
gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);

// Загрузка матрицы Лагранжа
auto lagrange = radar::LagrangeMatrix::LoadFromJSON("lagrange_matrix.json");

// Создание конфигурации
auto config = radar::FractionalDelayConfig::Standard();
config.num_beams = 256;
config.num_samples = 8192;

// Создание процессора
radar::FractionalDelayProcessor processor(config, lagrange);

// Подготовка данных на GPU (например, от GeneratorGPU)
cl_mem gpu_buffer = generator.signal_base();

// Параметры задержки для каждого луча
std::vector<radar::DelayParams> delays(config.num_beams);
for (uint32_t i = 0; i < config.num_beams; ++i) {
    delays[i] = radar::DelayParams::FromSamples(0.5f + i * 0.1f);
}

// Обработка (IN-PLACE)
processor.Process(gpu_buffer, delays);

// Получить профилирование
auto prof = processor.GetLastProfiling();
std::cout << "Kernel time: " << prof.kernel_time_ms << " ms\n";
std::cout << "Throughput: " << prof.GetThroughput() / 1e9 << " Gsamples/sec\n";
```

### Пример 2: Задержка в градусах

```cpp
// Задержки в градусах для фазированной антенной решётки
std::vector<float> delays_degrees(256);
float angle_step = 0.5f;  // 0.5° между лучами
for (uint32_t i = 0; i < 256; ++i) {
    delays_degrees[i] = i * angle_step;  // 0°, 0.5°, 1.0°, ...
}

// Обработка (автоматическая конвертация градусы → отсчёты)
processor.Process(gpu_buffer, delays_degrees);
```

### Пример 3: Интеграция с GeneratorGPU

```cpp
// Генерация LFM сигнала
radar::LFMParameters lfm_params;
lfm_params.num_beams = 256;
lfm_params.count_points = 131072;
lfm_params.f_start = 1.0e9f;
lfm_params.f_stop = 2.0e9f;
lfm_params.sample_rate = 5.0e9f;

radar::GeneratorGPU generator(lfm_params);
cl_mem lfm_signal = generator.signal_base();

// Применение дробной задержки
auto lagrange = radar::LagrangeMatrix::LoadFromJSON("lagrange_matrix.json");
auto config = radar::FractionalDelayConfig::Performance();
radar::FractionalDelayProcessor processor(config, lagrange);

std::vector<radar::DelayParams> delays(config.num_beams);
// ... настройка задержек ...

processor.Process(lfm_signal, delays);
```

## Производительность

### Бенчмарки (NVIDIA RTX 3060)

| Конфигурация | Kernel Time | Throughput |
|--------------|-------------|------------|
| 256 × 65536   | 0.81 ms     | 20.68 Gsamples/sec |
| 128 × 32768   | 0.20 ms     | 21.47 Gsamples/sec |
| 64 × 16384    | 0.05 ms     | 21.47 Gsamples/sec |

### Оптимизации

1. **IN-PLACE обработка** - нет копирования CPU↔GPU
2. **Memory coalescing** - последовательный доступ к памяти
3. **Предвычисленная матрица** - коэффициенты Лагранжа на GPU
4. **Векторизация** - использование float2 для комплексных чисел
5. **Развёртывание циклов** - #pragma unroll для интерполяции

## Обработка ошибок

### Исключения

- `std::invalid_argument` - невалидные параметры конфигурации или матрицы
- `std::runtime_error` - ошибки OpenCL (компиляция kernel, создание буферов)
- `std::ios_base::failure` - ошибки чтения JSON файла

### Валидация

```cpp
// Конфигурация
if (!config.IsValid()) {
    // num_beams: 1..256
    // num_samples: >= 16
    // local_work_size: 32..1024
}

// Матрица Лагранжа
if (!lagrange_matrix.IsValid()) {
    // Должна быть 48 строк × 5 столбцов
    // Первая строка: [0, 1, 0, 0, 0] (сумма = 1.0)
}
```

## Тестирование

### Тестовый suite

Все тесты находятся в `src/Test/test_fractional_delay_processor.cpp`:

1. **Zero Delay Test** - Проверка нулевой задержки (сигнал не меняется)
2. **Integer Delay Test** - Проверка целой задержки (сдвиг на N отсчётов)
3. **Fractional Delay Test** - Проверка дробной задержки (интерполяция)
4. **Batch Processing Test** - Обработка нескольких лучей с разными задержками
5. **GeneratorGPU Integration Test** - Интеграция с LFM генератором
6. **Performance Test** - Бенчмарк производительности

**Запуск:**
```bash
cd build/linux-nvidia-debug/src/Test
./test_fractional_delay
```

**Текущий статус:** ✅ 6/6 тестов пройдены

## Зависимости

- `OpenCLComputeEngine` - Вычислительный движок OpenCL
- `GPUMemoryBuffer` - Управление памятью GPU
- `CommandQueuePool` - Пул очередей команд
- `nlohmann/json` - Парсинг JSON (для загрузки матрицы Лагранжа)

## См. также

- [GeneratorGPU_Detailed.md](GeneratorGPU_Detailed.md) - Генератор LFM сигналов
- [OpenCLComputeEngine_Detailed.md](OpenCLComputeEngine_Detailed.md) - OpenCL движок
- [GPUMemoryBuffer_Detailed.md](GPUMemoryBuffer_Detailed.md) - Управление памятью GPU

---

*Последнее обновление: 2026-01-21*

