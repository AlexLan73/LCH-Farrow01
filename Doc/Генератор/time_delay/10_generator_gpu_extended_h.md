# 📖 10_generator_gpu_extended.h

## API И ДОКУМЕНТАЦИЯ

### НОВАЯ СТРУКТУРА:

```cpp
typedef struct {
    uint beam_index;         // Индекс луча [0...num_beams)
    float delay_degrees;     // Задержка от УГЛА (градусы)
    float delay_time_ns;     // Задержка по ВРЕМЕНИ (наносекунды)
} CombinedDelayParam;
```

### НОВЫЙ МЕТОД:

```cpp
/**
 * @brief Сформировать ЛЧМ сигнал с комбинированной задержкой
 * @param combined_delays Массив CombinedDelayParam (размер = num_beams)
 * @param num_delay_params Количество элементов (должно = num_beams)
 * @return cl_mem GPU адрес буфера с задержанными сигналами
 */
cl_mem signal_combined_delays(
    const CombinedDelayParam* combined_delays,
    size_t num_delay_params
);
```

### ПАРАМЕТРЫ:

```cpp
// Угловая задержка: 0...360 градусов
delays[0].delay_degrees = 0.5f;

// Временная задержка: 0...много наносекунд
delays[0].delay_time_ns = 100.0f;

// Результат: τ_total = τ_angle + τ_time
```

### ПРИМЕР ИСПОЛЬЗОВАНИЯ:

```cpp
std::vector<CombinedDelayParam> delays(256);
for (int i = 0; i < 256; i++) {
    delays[i].beam_index = i;
    delays[i].delay_degrees = 0.5f * i;
    delays[i].delay_time_ns = 50.0f * i;
}

cl_mem signal = gen.signal_combined_delays(delays.data(), delays.size());
```

### ЧЛЕНЫ КЛАССА (private):

```cpp
cl_kernel kernel_lfm_combined_;
std::unique_ptr buffer_signal_combined_;
```

### ДРОБНАЯ ЗАДЕРЖКА:

Поддерживает: 50 нс → 0.6 отсчётов (при 12 MHz)
Механизм: интерполяция в kernel'е
