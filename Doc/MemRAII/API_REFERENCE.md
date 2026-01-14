# 📚 API Reference: Unified OpenCLManager

**Версия**: 2.0  
**Дата**: 2026-01-10

---

## 🎯 OpenCLManager - Полный API

### Инициализация

```cpp
// Инициализировать один раз в начале программы
static void Initialize(cl_device_type device_type = CL_DEVICE_TYPE_GPU);

// Получить экземпляр синглтона
static OpenCLManager& GetInstance();

// Проверить инициализацию
bool IsInitialized() const;
```

### Ресурсы OpenCL

```cpp
cl_context GetContext() const;
cl_command_queue GetQueue() const;
cl_device_id GetDevice() const;
cl_platform_id GetPlatform() const;
```

### Управление памятью GPU

#### Создать новый буфер

```cpp
std::unique_ptr<GPUMemoryBuffer> CreateBuffer(
    size_t num_elements,
    MemoryType type = MemoryType::GPU_WRITE_ONLY
);
```

**Пример:**
```cpp
auto buffer = OpenCLManager::GetInstance().CreateBuffer(
    1024,  // количество complex<float> элементов
    MemoryType::GPU_READ_WRITE
);
```

#### Обернуть внешний буфер

```cpp
std::unique_ptr<GPUMemoryBuffer> WrapExternalBuffer(
    cl_mem external_gpu_buffer,
    size_t num_elements,
    MemoryType type = MemoryType::GPU_WRITE_ONLY
);
```

**Пример:**
```cpp
cl_mem signal = generator->signal_base();
auto wrapper = OpenCLManager::GetInstance().WrapExternalBuffer(
    signal,
    1024,
    MemoryType::GPU_WRITE_ONLY
);
```

**Важно:** Автоматически проверяет, что `external_gpu_buffer` принадлежит правильному context.

#### Регистрация буферов для переиспользования

```cpp
// Зарегистрировать буфер по имени
void RegisterBuffer(
    const std::string& name,
    std::shared_ptr<GPUMemoryBuffer> buffer
);

// Получить зарегистрированный буфер
std::shared_ptr<GPUMemoryBuffer> GetBuffer(const std::string& name);

// Создать или получить существующий
std::shared_ptr<GPUMemoryBuffer> GetOrCreateBuffer(
    const std::string& name,
    size_t num_elements,
    MemoryType type = MemoryType::GPU_WRITE_ONLY
);
```

**Пример:**
```cpp
auto& manager = OpenCLManager::GetInstance();

// Создать и зарегистрировать
auto signal = manager.CreateBuffer(1024, MemoryType::GPU_WRITE_ONLY);
manager.RegisterBuffer("signal_base", std::shared_ptr<GPUMemoryBuffer>(signal.release()));

// В другом месте получить
auto cached = manager.GetBuffer("signal_base");
if (cached) {
    auto data = cached->ReadFromGPU();
}

// Или создать/получить одной командой
auto buffer = manager.GetOrCreateBuffer("signal_base", 1024, MemoryType::GPU_WRITE_ONLY);
```

### Компиляция программ

```cpp
cl_program GetOrCompileProgram(const std::string& source);
std::string GetCacheStatistics() const;
```

### Информация об устройстве

```cpp
std::string GetDeviceInfo() const;
void PrintMemoryStatistics() const;
```

---

## 🎯 GPUMemoryBuffer - API

### Операции чтения/записи

```cpp
// Прочитать ВСЕ данные с GPU
std::vector<std::complex<float>> ReadFromGPU();

// Прочитать ЧАСТЬ данных (быстрее)
std::vector<std::complex<float>> ReadPartial(size_t num_elements);

// Записать данные на GPU
void WriteToGPU(const std::vector<std::complex<float>>& data);
```

### Информация

```cpp
size_t GetNumElements() const;
size_t GetSizeBytes() const;
bool IsExternalBuffer() const;
bool IsGPUDirty() const;
MemoryType GetMemoryType() const;
void PrintStats() const;
```

---

## 🔧 Kernel Caching API

### `GetOrCreateKernel(cl_program program, const std::string& kernel_name)`

Получить или создать kernel из программы с автоматическим кэшированием.

**Параметры:**
- `program` - OpenCL program (из `GetOrCompileProgram`)
- `kernel_name` - Имя kernel функции в программе

**Возвращает:** `cl_kernel` (не освобождайте вручную!)

**Пример:**
```cpp
cl_program program = manager.GetOrCompileProgram(source);
cl_kernel kernel = manager.GetOrCreateKernel(program, "my_kernel");
// При повторном вызове вернется закэшированный kernel
```

---

### `GetKernelCacheStatistics() const`

Получить статистику кэша kernels.

**Возвращает:** `std::string` с информацией о размере кэша, hits/misses, hit rate

**Пример:**
```cpp
std::cout << manager.GetKernelCacheStatistics();
// Output:
// Kernel Cache Statistics:
//   Cache size: 10 kernels
//   Cache hits: 45
//   Cache misses: 10
//   Hit rate: 81.8%
```

---

### `ClearKernelCache()`

Очистить весь кэш kernels. Полезно для долгоживущих программ с большим количеством kernels.

**Пример:**
```cpp
// Периодическая очистка
if (manager.GetKernelCacheSize() > 100) {
    manager.ClearKernelCache();
}
```

---

### `ClearKernelsForProgram(cl_program program)`

Очистить все kernels, созданные из конкретного program.

**Параметры:**
- `program` - OpenCL program, чьи kernels нужно очистить

**Пример:**
```cpp
cl_program program = manager.GetOrCompileProgram(source);
// ... использовать kernels из program ...
// Когда program больше не нужен:
manager.ClearKernelsForProgram(program);
```

---

### `GetKernelCacheSize() const`

Получить текущее количество kernels в кэше.

**Возвращает:** `size_t`

**Пример:**
```cpp
size_t num_kernels = manager.GetKernelCacheSize();
std::cout << "Cached kernels: " << num_kernels << "\n";
```

---

## 📝 Примеры использования

### Пример 1: Базовое использование

```cpp
#include "GPU/opencl_manager.h"
#include "GPU/gpu_memory_manager.hpp"

int main() {
    // Инициализация
    gpu::OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
    
    // Создать буфер
    auto buffer = gpu::OpenCLManager::GetInstance().CreateBuffer(
        1024,
        gpu::MemoryType::GPU_READ_WRITE
    );
    
    // Записать данные
    std::vector<std::complex<float>> data(1024);
    // ... заполнить data ...
    buffer->WriteToGPU(data);
    
    // Прочитать обратно
    auto readback = buffer->ReadFromGPU();
    
    return 0;
}
```

### Пример 2: Переиспользование буферов

```cpp
auto& manager = gpu::OpenCLManager::GetInstance();

// Расчет 1: создать и сохранить
auto signal1 = manager.CreateBuffer(1024, gpu::MemoryType::GPU_WRITE_ONLY);
// ... заполнить signal1 ...
manager.RegisterBuffer("calculation_1", 
    std::shared_ptr<gpu::GPUMemoryBuffer>(signal1.release()));

// Расчет 2: использовать тот же буфер
auto signal2 = manager.GetBuffer("calculation_1");
if (signal2) {
    auto data = signal2->ReadFromGPU();
    // Обработать данные
}
```

### Пример 3: Обертка внешнего буфера

```cpp
// GeneratorGPU создает буфер
auto generator = std::make_shared<GeneratorGPU>(params);
cl_mem signal_gpu = generator->signal_base();

// Обернуть для чтения (с автоматической валидацией context)
auto wrapper = gpu::OpenCLManager::GetInstance().WrapExternalBuffer(
    signal_gpu,
    generator->GetTotalSize(),
    gpu::MemoryType::GPU_WRITE_ONLY
);

// Прочитать данные
auto data = wrapper->ReadFromGPU();
```

### Пример 4: Работа с большим количеством kernels

```cpp
auto& manager = gpu::OpenCLManager::GetInstance();

// Создать несколько programs
cl_program program1 = manager.GetOrCompileProgram(source1);
cl_program program2 = manager.GetOrCompileProgram(source2);

// Создать множество kernels
std::vector<cl_kernel> kernels;
for (int i = 0; i < 50; ++i) {
    std::string kernel_name = "kernel_" + std::to_string(i);
    kernels.push_back(manager.GetOrCreateKernel(program1, kernel_name));
}

// Мониторинг размера кэша
if (manager.GetKernelCacheSize() > 100) {
    std::cout << "Cache size: " << manager.GetKernelCacheSize() << "\n";
    manager.ClearKernelCache();
}

// Очистка kernels конкретного program
manager.ClearKernelsForProgram(program1);

// Статистика
std::cout << manager.GetKernelCacheStatistics();
```

---

**Автор**: AI Assistant (Кодо)  
**Дата**: 2026-01-10  
**Обновлено**: Добавлен раздел Kernel Caching для масштабирования

