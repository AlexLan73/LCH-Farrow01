# План: FFT Batch Processing с управлением памятью

## Цель
Обрабатывать большие массивы (256*1300000 комплексных точек) с автоматическим выбором стратегии: полная обработка или batch processing в зависимости от доступной памяти GPU.

## Важно
- **Старый метод `Process()` НЕ ТРОГАТЬ** - он должен остаться без изменений для обратной совместимости
- Создать новый публичный метод `ProcessNew()` с автоматическим выбором стратегии
- Все новые методы - приватные (кроме `ProcessNew()`)

## Архитектура решения

### 1. Проверка памяти GPU
- **В конструкторе** `AntennaFFTProcMax`: предварительная оценка требуемой памяти (опционально, для информации)
- **В `ProcessNew()`**: точная проверка доступной памяти перед обработкой
- Использовать `OpenCLCore::GetGlobalMemorySize()` для получения размера памяти GPU
- Рассчитывать требуемую память: `beam_count * nFFT * sizeof(complex<float>) * 2` (input + output) + overhead

### 2. Стратегия выбора режима
- **Порог использования**: 40% от доступной памяти GPU (настраиваемый параметр)
- **Режим 1 (памяти хватает)**: вызывать старый `Process()` для обработки всех лучей сразу
- **Режим 2 (памяти не хватает)**: использовать новый `ProcessWithBatching()` для batch processing

### 3. Batch Processing
- Размер батча: 20% от общего количества лучей (настраиваемый параметр)
- Использовать `CommandQueuePool` для получения нескольких независимых command queues
- Параллельная обработка батчей через разные command queues (без CPU потоков)
- Синхронизация через OpenCL events

### 4. Профилирование
- GPU время: сумма времени всех батчей
- CPU время: общее время от начала первого батча до завершения последнего
- Детальная статистика по каждому батчу

## Файлы для изменения

### `include/fft/antenna_fft_proc_max.h`
- Добавить публичный метод `ProcessNew(cl_mem input_signal)` - новый метод с автоматическим выбором стратегии
- Добавить приватный метод `EstimateRequiredMemory()` - расчет требуемой памяти
- Добавить приватный метод `CheckAvailableMemory()` - проверка доступной памяти
- Добавить приватный метод `ProcessBatch()` - обработка одного батча лучей
- Добавить приватный метод `ProcessWithBatching()` - основная логика batch processing
- Добавить конфигурационные параметры: `memory_usage_threshold` (40%), `batch_size_percent` (20%)

### `src/fft/antenna_fft_proc_max.cpp`
- Реализовать `ProcessNew()`: проверка памяти и выбор между `Process()` и `ProcessWithBatching()`
- Реализовать `EstimateRequiredMemory()`: расчет на основе `beam_count`, `nFFT`, `count_points`
- Реализовать `CheckAvailableMemory()`: запрос через `OpenCLCore::GetGlobalMemorySize()`
- Реализовать `ProcessBatch()`: обработка подмножества лучей (beam_start, beam_count)
- Реализовать `ProcessWithBatching()`: разбиение на батчи, параллельная обработка через `CommandQueuePool`, сбор результатов

## Детальная реализация

### Шаг 1: Добавить методы проверки памяти
```cpp
// В antenna_fft_proc_max.h (private section)
size_t EstimateRequiredMemory() const;
bool CheckAvailableMemory(size_t required_memory, double threshold = 0.4) const;
size_t CalculateBatchSize(size_t total_beams, double batch_percent = 0.2) const;
```

### Шаг 2: Создать ProcessNew()
```cpp
// В antenna_fft_proc_max.h (public section)
AntennaFFTResult ProcessNew(cl_mem input_signal);

// В antenna_fft_proc_max.cpp
AntennaFFTResult AntennaFFTProcMax::ProcessNew(cl_mem input_signal) {
    // 1. Проверить требуемую память
    size_t required_memory = EstimateRequiredMemory();
    bool memory_ok = CheckAvailableMemory(required_memory, 0.4);
    
    // 2. Выбрать стратегию
    if (memory_ok) {
        return Process(input_signal);  // Используем старый Process() для полной обработки
    } else {
        return ProcessWithBatching(input_signal);  // Новый метод для batch processing
    }
}
```

**Важно**: Старый метод `Process()` остается без изменений!

### Шаг 3: Реализовать ProcessBatch()
- Принимает: `input_signal`, `beam_start`, `beam_count`, `batch_queue` (command queue для этого батча)
- Создает временные буферы только для этого батча
- Использует переданную command queue из `CommandQueuePool`
- Возвращает результаты для этого батча
- Возвращает event для синхронизации

### Шаг 4: Реализовать ProcessWithBatching()
- Рассчитать размер батча (20% от beam_count)
- Разбить лучи на батчи
- Для каждого батча:
  - Получить command queue из `CommandQueuePool::GetNextQueue()`
  - Запустить `ProcessBatch()` асинхронно
  - Сохранить event для синхронизации
- Дождаться завершения всех батчей через `clWaitForEvents()`
- Собрать результаты всех батчей в один `AntennaFFTResult`
- Профилирование: GPU время (сумма), CPU время (общее)

## Профилирование

### Структура данных
```cpp
struct BatchProfiling {
    size_t batch_index;
    size_t beam_start;
    size_t beam_count;
    double gpu_time_ms;
    double cpu_time_ms;
};
```

### Вывод статистики
- Общее время GPU (сумма всех батчей)
- Общее время CPU (от начала до конца)
- Время каждого батча отдельно
- Параллелизм: сколько батчей обрабатывалось одновременно

## Конфигурация

Добавить в класс как статические константы или члены:
- `memory_usage_threshold = 0.4` (40%)
- `batch_size_percent = 0.2` (20%)
- Возможность переопределения через параметры конструктора (опционально)

## Тестирование

1. Малые данные (5 лучей, 1000 точек) - `ProcessNew()` должна использовать старый `Process()` (полная обработка)
2. Большие данные (256 лучей, 1300000 точек) - `ProcessNew()` должна использовать `ProcessWithBatching()` (batch processing)
3. Проверка корректности результатов (сравнение результатов `ProcessNew()` с `Process()` для малых данных)
4. Проверка профилирования (GPU + CPU время)

## TODO List

1. ✅ Добавить приватные методы `EstimateRequiredMemory()` и `CheckAvailableMemory()` в `antenna_fft_proc_max.h` и реализовать в `.cpp`
2. ✅ Создать новый публичный метод `ProcessNew()` с проверкой памяти и выбором стратегии. Старый `Process()` не трогать!
3. ✅ Реализовать приватный метод `ProcessBatch()` для обработки подмножества лучей с отдельной command queue
4. ✅ Реализовать приватный метод `ProcessWithBatching()` с разбиением на батчи и параллельной обработкой через `CommandQueuePool`
5. ✅ Добавить профилирование для batch processing (GPU время каждого батча + общее CPU время)
6. ✅ Добавить конфигурационные параметры (`memory_usage_threshold`, `batch_size_percent`) в класс
7. ✅ Протестировать `ProcessNew()` с малыми данными (5 лучей, 1000 точек) - должна использоваться полная обработка через старый `Process()`
8. ✅ Протестировать `ProcessNew()` с большими данными (256 лучей, 1300000 точек) - должен использоваться batch processing

## Дата создания
2026-01-17

## Автор
Кодо (AI Assistant)

