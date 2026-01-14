# ✅ НОВАЯ АРХИТЕКТУРА GPU ПАМЯТИ

## Проблема БЫЛА

```cpp
// ❌ МУТОРНО - одинаковые параметры везде!
auto buffer = std::make_unique<gpu::GPUMemoryBuffer>(
    gen_gpu_->GetContext(),      // Одинаково!
    gen_gpu_->GetQueue(),        // Одинаково!
    signal_gpu,
    num_elements,
    gpu::MemoryType::GPU_WRITE_ONLY
);
```

**Проблемы:**
- Transmission context/queue каждый раз
- Трудно масштабировать
- Нарушение принципа DRY
- Много параметров в конструкторе

---

## Решение: Синглтон GPUMemoryManager

```cpp
// ✅ ПРОСТО - контекст берётся автоматически!
auto buffer = gpu::GPUMemoryManager::CreateBuffer(
    num_elements,
    gpu::MemoryType::GPU_WRITE_ONLY
);
```

### Архитектура

```
main()
  ↓
OpenCLManager::Initialize()     ← один раз
  ↓
GPUMemoryManager::Initialize()  ← один раз (использует OpenCLManager)
  ↓
Везде можно использовать:
  - GPUMemoryManager::CreateBuffer()
  - GPUMemoryManager::WrapExternalBuffer()
```

---

## API

### 1️⃣ Инициализация (один раз в main)

```cpp
#include "GPU/gpu_memory_manager.hpp"
#include "GPU/opencl_manager.h"

int main() {
    // Инициализировать OpenCL
    gpu::OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
    
    // Инициализировать менеджер памяти
    gpu::GPUMemoryManager::Initialize();
    
    // Дальше везде доступно через статические методы
    // ...
}
```

### 2️⃣ Создать новый GPU буфер

```cpp
// Создать буфер на GPU
auto buffer = gpu::GPUMemoryManager::CreateBuffer(
    1024,  // количество complex<float> элементов
    gpu::MemoryType::GPU_WRITE_ONLY
);

// Использовать
buffer->PrintStats();
auto data = buffer->ReadFromGPU();
```

### 3️⃣ Обернуть ГОТОВЫЙ буфер (GeneratorGPU и т.п.)

```cpp
// Создать генератор
auto gen = std::make_shared<GeneratorGPU>(params);
cl_mem signal = gen->signal_base();  // Это cl_mem

// ✓ Обернуть (не владеем буфером!)
auto reader = gpu::GPUMemoryManager::WrapExternalBuffer(
    signal,                          // готовый cl_mem
    gen->GetTotalSize(),             // кол-во элементов
    gpu::MemoryType::GPU_WRITE_ONLY
);

// Читать из буфера генератора
auto cpu_data = reader->ReadFromGPU();

// reader уничтожится, но signal остаётся (управляется gen)
```

### 4️⃣ Чтение/запись

```cpp
// Прочитать ВСЕ данные
auto all_data = buffer->ReadFromGPU();

// Прочитать ЧАСТИЧНО (быстрее)
auto partial = buffer->ReadPartial(100);

// Написать на GPU
buffer->WriteToGPU(my_data);

// Проверить статус
bool dirty = buffer->IsGPUDirty();
bool external = buffer->IsExternalBuffer();
```

---

## Пример: Читать от GeneratorGPU

**БЫЛО (❌ муторно):**

```cpp
void Example_Old(const cl_mem& signal_gpu) {
    try {
        auto buffer = std::make_unique<gpu::GPUMemoryBuffer>(
            gen_gpu_->GetContext(),
            gen_gpu_->GetQueue(),
            signal_gpu,
            gen_gpu_->GetNumSamples() * gen_gpu_->GetNumBeams(),
            (cl_mem) nullptr,  // ← ???
            gpu::MemoryType::GPU_WRITE_ONLY
        );
        auto data = buffer->ReadFromGPU();
        PrintFirstSamples(data);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
```

**СТАЛО (✅ чистый API):**

```cpp
void Example_New(const cl_mem& signal_gpu) {
    try {
        auto reader = gpu::GPUMemoryManager::WrapExternalBuffer(
            signal_gpu,
            gen_gpu_->GetTotalSize(),
            gpu::MemoryType::GPU_WRITE_ONLY
        );
        
        auto data = reader->ReadFromGPU();
        PrintFirstSamples(data);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
```

---

## Key Changes

| Что | Было | Стало |
|-----|------|-------|
| **Инициализация context** | Передавать каждый раз | `GPUMemoryManager::Initialize()` один раз |
| **Инициализация queue** | Передавать каждый раз | Берётся из менеджера автоматически |
| **Создание буфера** | `make_unique<GPUMemoryBuffer>(ctx, q, ...)` | `GPUMemoryManager::CreateBuffer(...)` |
| **Обёртка на готовый** | Специальный конструктор с nullptr | `GPUMemoryManager::WrapExternalBuffer()` |
| **Муторные параметры** | Везде одинаковые context/queue | НЕ нужно передавать |
| **Архитектура** | Один класс GPUMemoryBuffer | Синглтон GPUMemoryManager + GPUMemoryBuffer |

---

## Преимущества

✅ **DRY** - context/queue инициализируются один раз  
✅ **Масштабируемо** - легко добавить пулинг, кэширование  
✅ **OOP на уровне seniors** - правильное разделение ответственности  
✅ **Понятно** - API ясен, мало параметров  
✅ **Безопасно** - RAII, управление памятью гарантировано  
✅ **Производительно** - pinned memory, оптимальный DMA  

---

## Файлы

1. **gpu_memory_manager.hpp** - заголовок (синглтон + GPUMemoryBuffer)
2. **gpu_memory_manager.cpp** - реализация
3. **examples_clean.hpp** - примеры использования

---

## Что исправилось

❌ **Было:** Нули при чтении → использовались неправильные конструкторы  
✅ **Стало:** Единый API, который работает правильно

❌ **Было:** Муторное передавание context/queue  
✅ **Стало:** Синглтон управляет всем

❌ **Было:** Confusion между двумя конструкторами  
✅ **Стало:** Ясный API с двумя методами менеджера

---

## Дальнейшие улучшения (если нужны)

- **Buffer pooling** - переиспользовать буферы для производительности
- **Async transfers** - событийно-ориентированные трансферы
- **Memory statistics** - трекинг использования памяти
- **Compression** - сжатие данных перед трансфером
- **Multi-device** - поддержка нескольких GPU

Но базовая архитектура - это фундамент, готовый для расширений! 🚀
