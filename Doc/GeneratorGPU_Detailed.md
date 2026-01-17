# GeneratorGPU - Детальная документация

## Обзор

`GeneratorGPU` - основной класс для генерации ЛЧМ (линейно-частотно-модулированных) сигналов на GPU с использованием OpenCL. Класс предоставляет высокоуровневый API для создания различных типов сигналов с поддержкой задержек и сложных конфигураций.

## Архитектура

```
GeneratorGPU
├── OpenCLComputeEngine* engine_     // Главный фасад (не владеет)
├── LFMParameters params_            // Параметры сигнала
├── size_t num_samples_              // Кэш: отсчётов на луч
├── size_t num_beams_                // Кэш: количество лучей
├── size_t total_size_               // Кэш: beams * samples
├── KernelProgram* kernel_program_   // Программа с kernels
├── cl_kernel kernels_[4]            // Скомпилированные kernels
└── GPUMemoryBuffer* buffers_[4]     // Кэш результатов
```

## Конструкторы и деструктор

### GeneratorGPU(const LFMParameters& params)

**Описание:** Создает генератор с заданными параметрами ЛЧМ сигнала.

**Параметры:**
- `params` - Структура с параметрами сигнала

**Исключения:**
- `std::invalid_argument` - если параметры невалидны
- `std::runtime_error` - если OpenCLComputeEngine не инициализирован

**Алгоритм инициализации:**
1. Валидация параметров через `params.IsValid()`
2. Получение ссылки на `OpenCLComputeEngine::GetInstance()`
3. Расчет размеров (`num_samples_`, `num_beams_`, `total_size_`)
4. Загрузка и компиляция OpenCL kernels
5. Создание кэша буферов

### ~GeneratorGPU()

**Описание:** Деструктор. Освобождает ресурсы GPU.

**Особенности:**
- Ресурсы управляются `OpenCLComputeEngine`, поэтому только обнуляет указатели
- Использует RAII для автоматической очистки `unique_ptr` буферов

## Основные методы генерации сигналов

### cl_mem signal_base()

**Описание:** Генерирует базовый ЛЧМ сигнал без задержек.

**Возвращает:** `cl_mem` - GPU адрес буфера с результатом

**Алгоритм:**
1. Создание выходного буфера через `engine_->CreateBuffer(total_size_, GPU_WRITE_ONLY)`
2. Выполнение kernel `kernel_lfm_basic`
3. Кэширование результата в `buffer_signal_base_`
4. Возврат `cl_mem` из кэшированного буфера

**Структура данных в GPU памяти:**
```
[beam0_sample0, beam0_sample1, ..., beam0_sampleN,
 beam1_sample0, beam1_sample1, ..., beam1_sampleN,
 ...
 beamM_sample0, beamM_sample1, ..., beamM_sampleN]
```

### cl_mem signal_valedation(const DelayParameter* delays, size_t num_delays)

**Описание:** Генерирует ЛЧМ сигнал с дробными задержками по лучам.

**Параметры:**
- `delays` - Массив параметров задержки (размер = num_beams)
- `num_delays` - Количество элементов в массиве

**Возвращает:** `cl_mem` - GPU адрес буфера с результатом

**Алгоритм:**
1. Валидация параметров
2. Создание буфера задержек на GPU
3. Создание выходного буфера
4. Выполнение kernel `kernel_lfm_delayed`
5. Кэширование результата

### cl_mem signal_combined_delays(const CombinedDelayParam* delays, size_t num_delays)

**Описание:** Генерирует сигнал с комбинированными задержками (угловая + временная).

**Особенности:**
- Поддерживает интерполяцию для субдискретных задержек
- Использует `float` вместо `int` для точности

### cl_mem signal_sinusoids(const SinusoidGenParams& params, const RaySinusoidMap& map_ray)

**Описание:** Генерирует сигналы как сумму синусоид для каждого луча.

**Алгоритм:**
1. Преобразование `RaySinusoidMap` в массив `RaySinusoidParams`
2. Создание GPU буфера с параметрами
3. Выполнение kernel `kernel_sinusoid_combined`
4. Кэширование результата

## Методы чтения результатов

### std::vector<std::complex<float>> GetSignalAsVector(int beam_index)

**Описание:** Читает сигнал конкретного луча в CPU память.

**Параметры:**
- `beam_index` - Индекс луча (0 до num_beams-1)

**Возвращает:** Вектор комплексных чисел сигнала луча

**Алгоритм:**
1. Валидация индекса
2. Синхронизация GPU через `ClearGPU()`
3. Поиск активного буфера (sinusoid → combined → delayed → base)
4. Расчет смещения: `offset = beam_index * num_samples_ * sizeof(complex<float>)`
5. `clEnqueueReadBuffer` с правильными параметрами

### std::vector<std::complex<float>> GetSignalAsVectorPartial(int beam_index, size_t num_samples)

**Описание:** Читает только часть сигнала луча.

**Особенности:**
- Использует `GPUMemoryBuffer::ReadPartial()` для эффективности
- Ограничивает `num_samples` размером луча

### std::vector<std::complex<float>> GetSignalAsVectorAll()

**Описание:** Читает все данные со всех лучей.

**Алгоритм:**
1. Синхронизация GPU
2. Обертывание `cl_mem` в `GPUMemoryBuffer` (non-owning)
3. `ReadFromGPU()` для получения всех данных

## Геттеры

### size_t GetNumBeams() const noexcept
Возвращает количество лучей.

### size_t GetNumSamples() const noexcept
Возвращает количество отсчетов на луч.

### size_t GetTotalSize() const noexcept
Возвращает общее количество элементов (`num_beams * num_samples`).

### size_t GetMemorySizeBytes() const noexcept
Возвращает размер данных в байтах.

### const LFMParameters& GetParameters() const noexcept
Возвращает константную ссылку на параметры.

## OpenCL Kernels

### kernel_lfm_basic

**Подпись:**
```cpp
__kernel void kernel_lfm_basic(
    __global float2 *output,
    float f_start, float f_stop, float sample_rate, float duration,
    uint num_samples, uint num_beams
)
```

**Алгоритм:**
- Каждый поток обрабатывает один элемент сигнала
- Расчет индексов: `ray_id = gid / num_samples`, `sample_id = gid % num_samples`
- Генерация ЛЧМ: `phase = 2π * (f_start*t + 0.5*chirp_rate*t²)`
- Результат: `output[idx] = (float2)(cos(phase), sin(phase))`

### kernel_lfm_delayed

**Особенности:**
- Использует `__global const DelayParam *delays`
- Конвертирует градусы в задержку времени через длину волны
- Обрабатывает отрицательные задержки (нули перед сигналом)

### kernel_lfm_combined

**Улучшения:**
- Поддержка временных задержек в наносекундах
- Интерполяция между отсчетами для субдискретных задержек
- Использование `float` для индексов вместо `int`

### kernel_sinusoid_combined

**Особенности:**
- Суммирование нескольких синусоид на луч
- Поддержка до 10 синусоид на луч
- Дефолтные параметры если луч не описан в карте

## Управление памятью

### Кэширование буферов

```cpp
std::unique_ptr<GPUMemoryBuffer> buffer_signal_base_;
std::unique_ptr<GPUMemoryBuffer> buffer_signal_delayed_;
std::unique_ptr<GPUMemoryBuffer> buffer_signal_combined_;
std::unique_ptr<GPUMemoryBuffer> buffer_signal_sinusoid_;
```

**Преимущества:**
- Избежание повторного выделения памяти
- Сохранение результатов между вызовами
- Автоматическая очистка в деструкторе

### Типы владения

- **Owning buffers**: Создаются генератором, уничтожаются в деструкторе
- **Non-owning wrappers**: Используют внешние `cl_mem`, не уничтожают их

## Синхронизация

### void ClearGPU()

**Описание:** Синхронизирует все командные очереди.

**Вызывает:** `engine_->Finish()` - ждет завершения всех операций.

**Использование:** Вызывать перед чтением результатов с GPU.

## Обработка ошибок

### Валидация параметров

- Проверка `params_.IsValid()` в конструкторе
- Проверка размеров массивов в методах генерации
- Проверка индексов в методах чтения

### OpenCL ошибки

- Перехват исключений от `OpenCLComputeEngine`
- Подробные сообщения об ошибках
- Graceful degradation (возврат пустых векторов при ошибках чтения)

## Производительность

### Оптимизации

- **Кэширование kernels**: Однократная компиляция
- **Пул очередей**: Асинхронное выполнение
- **Кэширование буферов**: Избежание reallocations
- **DMA transfers**: Pinned memory для быстрого чтения

### Статистика

- Размер данных: `total_size_ * sizeof(std::complex<float>)`
- Время выполнения: Зависит от GPU и размера данных
- Эффективность кэша: Отслеживается в `OpenCLComputeEngine`

## Использование

### Базовый пример

```cpp
// Инициализация
gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);

// Параметры сигнала
LFMParameters params;
params.f_start = 100.0f;
params.f_stop = 500.0f;
params.sample_rate = 12.0e6f;
params.num_beams = 256;
params.duration = 0.01f;

// Создание генератора
radar::GeneratorGPU gen(params);

// Генерация сигнала
cl_mem gpu_signal = gen.signal_base();

// Синхронизация
gen.ClearGPU();

// Чтение результатов
auto beam_data = gen.GetSignalAsVector(0);

// Очистка (автоматически в деструкторе)
```

### С задержками

```cpp
// Параметры задержек
std::vector<DelayParameter> delays(params.num_beams);
for(size_t i = 0; i < params.num_beams; i++) {
    delays[i].beam_index = i;
    delays[i].delay_degrees = i * 0.5f; // 0.5° на луч
}

// Генерация
cl_mem gpu_signal_delayed = gen.signal_valedation(delays.data(), delays.size());
```

## Потенциальные улучшения

### Память
- Поддержка out-of-core вычислений для больших сигналов
- Компрессия результатов при хранении

### Производительность
- Многопоточная генерация нескольких сигналов
- Асинхронные операции чтения/записи
- Оптимизация kernels под конкретные GPU

### Функциональность
- Поддержка других типов модуляции (NLFM, Barker codes)
- Реальное время генерация с streaming
- Интеграция с другими библиотеками (cuFFT, clFFT)