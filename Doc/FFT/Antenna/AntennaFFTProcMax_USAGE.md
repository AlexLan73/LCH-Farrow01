# AntennaFFTProcMax Usage Guide

## Быстрый старт

```cpp
#include "fft/antenna_fft_proc_max.h"
#include "generator/generator_gpu_new.h"
#include "GPU/opencl_compute_engine.hpp"

// 1. Инициализация OpenCL (один раз)
gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);

// 2. Создание генератора сигналов
LFMParameters lfm_params;
lfm_params.num_beams = 5;
lfm_params.count_points = 1000;
radar::GeneratorGPU gen(lfm_params);

// 3. Генерация сигналов
SinusoidGenParams gen_params(5, 1000);
RaySinusoidMap empty_map;
cl_mem signal_gpu = gen.signal_sinusoids(gen_params, empty_map);

// 4. Создание процессора FFT
antenna_fft::AntennaFFTParams fft_params(
    5,      // beam_count
    1000,   // count_points
    512,    // out_count_points_fft
    3,      // max_peaks_count
    "task_1",
    "module_1"
);
antenna_fft::AntennaFFTProcMax processor(fft_params);

// 5. Обработка
antenna_fft::AntennaFFTResult result = processor.Process(signal_gpu);

// 6. Вывод результатов
processor.PrintResults(result);
processor.SaveResultsToFile(result, "result.md");

// 7. Статистика профилирования
std::cout << processor.GetProfilingStats() << std::endl;
```

## Примеры

### Пример 1: Базовое использование

```cpp
antenna_fft::AntennaFFTParams params(10, 2048, 1024, 3);
antenna_fft::AntennaFFTProcMax processor(params);

// Обработка данных с GPU
cl_mem input_signal = ...; // Ваш входной сигнал
antenna_fft::AntennaFFTResult result = processor.Process(input_signal);
```

### Пример 2: Обработка данных с CPU

```cpp
std::vector<std::complex<float>> input_data(10 * 2048);
// ... заполнить данные ...

antenna_fft::AntennaFFTResult result = processor.Process(input_data);
```

### Пример 3: Обновление параметров

```cpp
antenna_fft::AntennaFFTParams new_params(20, 4096, 2048, 5);
processor.UpdateParams(new_params);
```

## Параметры

### AntennaFFTParams

- `beam_count` - количество лучей/антенн (должно быть > 0)
- `count_points` - количество точек в луче (должно быть > 0)
- `out_count_points_fft` - количество точек в FFT для вывода (должно быть > 0)
- `max_peaks_count` - количество максимальных значений (3-5, по умолчанию 3)
- `task_id` - идентификатор задачи (опционально)
- `module_name` - имя модуля (опционально)

## Результаты

### Структура результата

```cpp
struct AntennaFFTResult {
    std::vector<FFTResult> results;  // Результаты для каждого луча
    std::string task_id;
    std::string module_name;
    size_t total_beams;
    size_t nFFT;
};

struct FFTResult {
    size_t v_fft;
    std::vector<FFTMaxResult> max_values;  // Топ-N максимумов
};

struct FFTMaxResult {
    size_t index_point;  // Индекс точки в спектре
    float amplitude;     // Амплитуда
    float phase;         // Фаза в градусах
};
```

## Вывод результатов

### Консоль

```cpp
processor.PrintResults(result);
```

### Файл (таблица + JSON)

```cpp
processor.SaveResultsToFile(result, "Reports/result.md");
```

Файл содержит:
- Таблицу с результатами для каждого луча
- JSON формат для дальнейшей обработки
- Timestamp генерации

## Профилирование

```cpp
std::string stats = processor.GetProfilingStats();
std::cout << stats << std::endl;
```

Выводит:
- Время загрузки данных
- Время pre-callback
- Время FFT
- Время post-callback
- Время reduction
- Время чтения результатов
- Общее время

## Обработка ошибок

Все методы могут выбрасывать исключения:
- `std::invalid_argument` - невалидные параметры
- `std::runtime_error` - ошибки OpenCL/clFFT

```cpp
try {
    antenna_fft::AntennaFFTProcMax processor(params);
    auto result = processor.Process(input_signal);
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

## Производительность

Для максимальной производительности:
1. Используйте persistent буферы (не пересоздавайте процессор)
2. Переиспользуйте планы FFT (кэшируются автоматически)
3. Минимизируйте копирования данных (работайте с GPU буферами)
4. Используйте асинхронные операции где возможно

## Масштабируемость

Класс поддерживает множественные экземпляры:
- Каждый экземпляр может иметь свои параметры
- Планы FFT кэшируются и переиспользуются
- Thread-safe доступ к ресурсам

