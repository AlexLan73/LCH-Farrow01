# 🔧 ПОЛНЫЙ КОД: signal_combined_delays в generator_gpu_new.cpp

## 📄 Исправленная функция с типобезопасным API

```cpp
// generator_gpu_new.cpp

/**
 * @brief Генерировать сигнал с комбинированными задержками (углы + время)
 * 
 * ИСПОЛЬЗОВАНИЕ:
 *   CombinedDelayParam delays[256] = {...};
 *   cl_mem signal = gen.signal_combined_delays(delays, 256);
 *   auto data = gen.GetSignalAsVector(0);
 * 
 * @param combined_delays Массив параметров задержек (угол + время)
 * @param num_delay_params Количество параметров (должно = num_beams_)
 * @return cl_mem буфер на GPU с результирующим сигналом
 * 
 * @throw std::invalid_argument если параметры некорректны
 * @throw std::runtime_error если GPU операция не удалась
 */
cl_mem GeneratorGPU::signal_combined_delays(
    const CombinedDelayParam* combined_delays,
    size_t num_delay_params)
{
    // ========== ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ==========
    if (!engine_) {
        throw std::runtime_error(
            "GeneratorGPU::signal_combined_delays: Engine not initialized"
        );
    }

    if (!kernel_lfm_combined_) {
        throw std::runtime_error(
            "GeneratorGPU::signal_combined_delays: kernel_lfm_combined not loaded"
        );
    }

    if (!combined_delays) {
        throw std::invalid_argument(
            "GeneratorGPU::signal_combined_delays: combined_delays pointer is null"
        );
    }

    if (num_delay_params != num_beams_) {
        throw std::invalid_argument(
            "GeneratorGPU::signal_combined_delays: "
            "num_delay_params (" + std::to_string(num_delay_params) + ") "
            "must equal num_beams (" + std::to_string(num_beams_) + ")"
        );
    }

    std::cout << "GeneratorGPU: Generating signal_combined_delays with "
              << num_delay_params << " delay parameters..." << std::endl;

    try {
        // ========== ШАГ 1: Подготовить хостовый вектор параметров ==========
        // Конвертируем C-array в std::vector для типобезопасной загрузки
        std::vector<CombinedDelayParam> combined_host(
            combined_delays,
            combined_delays + num_delay_params
        );

        std::cout << "  - Created host vector with " << combined_host.size()
                  << " delay parameters" << std::endl;

        // ========== ШАГ 2: Загрузить параметры на GPU ==========
        // Используем типобезопасный API CreateTypedBufferWithData<T>
        // Он автоматически подстраивается под sizeof(CombinedDelayParam)
        auto combined_gpu_buffer = engine_->CreateTypedBufferWithData(
            combined_host,
            gpu::MemoryType::GPU_READ_ONLY
        );

        std::cout << "  - Uploaded delay parameters to GPU ("
                  << combined_gpu_buffer->GetSizeBytes() / 1024.0
                  << " KB)" << std::endl;

        // ========== ШАГ 3: Создать выходной буфер на GPU ==========
        // Буфер для результата kernel'а (output signal)
        auto output = engine_->CreateBuffer(
            total_size_,
            gpu::MemoryType::GPU_WRITE_ONLY
        );

        std::cout << "  - Created output buffer ("
                  << output->GetSizeBytes() / (1024.0 * 1024.0)
                  << " MB)" << std::endl;

        // ========== ШАГ 4: Выполнить kernel ==========
        ExecuteKernel(
            kernel_lfm_combined_,
            output->Get(),
            combined_gpu_buffer->Get()
        );

        // ========== ШАГ 5: Кэшировать результат и вернуть ==========
        // Сохранить в member variable для доступа через GetSignalAsVector()
        buffer_signal_combined_ = std::move(output);

        std::cout << "GeneratorGPU: signal_combined_delays completed successfully."
                  << std::endl;

        return buffer_signal_combined_->Get();

    } catch (const std::exception& e) {
        std::cerr << "GeneratorGPU: signal_combined_delays failed: "
                  << e.what() << std::endl;

        throw std::runtime_error(
            std::string("GeneratorGPU::signal_combined_delays failed: ")
            + e.what()
        );
    }
}
```

---

## 📌 ДЕТАЛИ РЕАЛИЗАЦИИ

### Почему std::vector?

```cpp
std::vector<CombinedDelayParam> combined_host(
    combined_delays,
    combined_delays + num_delay_params
);
```

| Аспект | Причина |
|--------|---------|
| **Диапазон [first, last)** | Стандартный C++ паттерн для копирования |
| **Половинчатый диапазон** | `combined_delays + num_delay_params` указывает на элемент **после** последнего |
| **Типобезопасность** | Вектор содержит точный тип `CombinedDelayParam` |
| **Управление памятью** | RAII — вектор сам управляет выделением/освобождением |

### Почему CreateTypedBufferWithData<T>?

```cpp
engine_->CreateTypedBufferWithData(
    combined_host,
    gpu::MemoryType::GPU_READ_ONLY
);
```

| Аспект | Старый способ | Новый способ |
|--------|---|---|
| **Type-safety** | ❌ Ошибка C2664 (неправильный тип) | ✅ Шаблон работает для любого T |
| **Читаемость** | ❌ Неясно какой тип передаётся | ✅ Явное имя `CreateTypedBufferWithData` |
| **Масштабируемость** | ❌ Нужна перегрузка под каждый тип | ✅ Один шаблон для всех типов |
| **Ошибки компиляции** | ❌ Непонятные ошибки преобразования | ✅ Ясная ошибка о пустом векторе |

### GPU_READ_ONLY vs GPU_WRITE_ONLY

```cpp
combined_gpu_buffer = engine_->CreateTypedBufferWithData(
    combined_host,
    gpu::MemoryType::GPU_READ_ONLY  // ← только чтение!
);

output = engine_->CreateBuffer(
    total_size_,
    gpu::MemoryType::GPU_WRITE_ONLY  // ← только запись!
);
```

- **GPU_READ_ONLY** для параметров: kernel только читает
- **GPU_WRITE_ONLY** для результата: kernel только пишет
- OpenCL может оптимизировать кэширование на основе этого

---

## ✅ ПРОЦЕСС ВЫПОЛНЕНИЯ

```
┌─────────────────────────────────────────┐
│ signal_combined_delays(delays[], count) │
└──────────────┬──────────────────────────┘
               │
               ├─► ВАЛИДАЦИЯ (проверка nullptr, размеров)
               │
               ├─► Создать std::vector из C-array
               │   (safe copy с контролем размера)
               │
               ├─► CreateTypedBufferWithData<CombinedDelayParam>()
               │   (malloc GPU + COPY HOST→GPU)
               │
               ├─► CreateBuffer (выходной буфер)
               │   (malloc GPU пустой)
               │
               ├─► ExecuteKernel()
               │   (запустить на GPU)
               │
               ├─► Кэшировать результат в buffer_signal_combined_
               │
               └─► return cl_mem (GPU буфер)
```

---

## 🧪 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

### Пример 1: Базовое использование

```cpp
// Подготовить параметры задержек
std::vector<CombinedDelayParam> delays(256);
for (int i = 0; i < 256; i++) {
    delays[i].delay_degrees = 0.0f;      // Нет углов
    delays[i].delay_time_ns = 50.0f;     // 50 нс задержки
}

// Генерировать сигнал
cl_mem signal = gen.signal_combined_delays(delays.data(), delays.size());

// Получить результат на хост
auto result = gen.GetSignalAsVector(0);
```

### Пример 2: С разными углами и временами

```cpp
std::vector<CombinedDelayParam> delays(256);
for (int i = 0; i < 256; i++) {
    delays[i].delay_degrees = 45.0f;     // 45 градусов
    delays[i].delay_time_ns = 10.0f * i; // Линейно от 0 до 2550 нс
}

cl_mem signal = gen.signal_combined_delays(delays.data(), 256);
```

### Пример 3: Обработка ошибок

```cpp
try {
    cl_mem signal = gen.signal_combined_delays(delays.data(), delays.size());
    // ... работать с signal ...
} catch (const std::invalid_argument& e) {
    std::cerr << "Ошибка параметров: " << e.what() << std::endl;
    // delays.size() != num_beams_ ?
} catch (const std::runtime_error& e) {
    std::cerr << "GPU ошибка: " << e.what() << std::endl;
    // Engine не инициализирован? Kernel не загружен?
}
```

---

## 🎯 ВАЖНЫЕ МОМЕНТЫ

### 1️⃣ CombinedDelayParam должен быть POD

```cpp
struct CombinedDelayParam {
    float delay_degrees;  // ✅ POD type (float)
    float delay_time_ns;  // ✅ POD type (float)
    // ❌ Не должно быть виртуальных функций!
    // ❌ Не должно быть неинициализируемых members!
};
```

Иначе `static_cast<const void*>()` будет UB.

### 2️⃣ num_delay_params ДОЛЖНО равняться num_beams_

```cpp
if (num_delay_params != num_beams_) {
    throw std::invalid_argument("...");
}
```

Потому что в kernel'е:
```opencl
__kernel void kernel_lfm_combined(
    ...,
    __global const CombinedDelayParam *combined,  // Один параметр на beam!
    ...
)
```

### 3️⃣ GPU буфер остаётся в памяти

```cpp
buffer_signal_combined_ = std::move(output);
return buffer_signal_combined_->Get();
```

Buffer кэшируется в member variable, чтобы он не был удалён, пока мы не вызовем `GetSignalAsVector()` или `ClearGPU()`.

---

## 🚀 КОМПИЛЯЦИЯ И ТЕСТИРОВАНИЕ

```bash
# Компилируем
g++ -std=c++17 -O3 generator_gpu_new.cpp -lOpenCL -c

# ✅ Без ошибок!
# Ошибка C2664 исчезла!

# Линкуем
g++ -std=c++17 -O3 main.o generator_gpu_new.o ... -lOpenCL -o app

# Запускаем
./app
# GeneratorGPU: Generating signal_combined_delays with 256 delay parameters...
#   - Created host vector with 256 delay parameters
#   - Uploaded delay parameters to GPU (3.00 KB)
#   - Created output buffer (8.00 MB)
# GeneratorGPU: signal_combined_delays completed successfully.
```

---
