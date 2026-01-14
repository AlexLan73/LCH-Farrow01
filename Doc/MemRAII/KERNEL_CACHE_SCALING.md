# 🚀 Масштабирование Kernel Cache для Большого Количества Kernels

## 📋 Обзор

При работе с **множеством kernels** (десятки, сотни) важно правильно управлять кэшем, чтобы:
- ✅ Избежать утечек памяти
- ✅ Оптимизировать производительность
- ✅ Контролировать использование ресурсов GPU

---

## 🎯 API для Управления Kernel Cache

### 1. Получение Kernel (основной метод)

```cpp
auto& manager = gpu::OpenCLManager::GetInstance();
cl_program program = manager.GetOrCompileProgram(kernel_source);
cl_kernel kernel = manager.GetOrCreateKernel(program, "my_kernel");
```

**Важно:** Kernel автоматически кэшируется. При повторном вызове с тем же `program` и `kernel_name` вернется закэшированный kernel.

---

### 2. Статистика Kernel Cache

```cpp
std::string stats = manager.GetKernelCacheStatistics();
std::cout << stats << std::endl;
```

**Вывод:**
```
Kernel Cache Statistics:
  Cache size: 45 kernels
  Cache hits: 1234
  Cache misses: 45
  Hit rate: 96.5%
```

---

### 3. Полная Очистка Kernel Cache

```cpp
// Очистить ВСЕ kernels из кэша
manager.ClearKernelCache();
```

**Когда использовать:**
- Периодически в long-running программах
- При смене набора kernels
- При подозрении на утечку памяти

**Важно:** Kernels будут автоматически пересозданы при следующем `GetOrCreateKernel()`.

---

### 4. Очистка Kernels Конкретного Program

```cpp
// Очистить все kernels, созданные из этого program
manager.ClearKernelsForProgram(program);
```

**Когда использовать:**
- Когда program больше не нужен
- При замене program на новую версию
- Для освобождения памяти перед удалением program

---

### 5. Размер Kernel Cache

```cpp
size_t num_kernels = manager.GetKernelCacheSize();
std::cout << "Currently cached: " << num_kernels << " kernels\n";
```

---

## 📊 Стратегии для Большого Количества Kernels

### Стратегия 1: Периодическая Очистка

```cpp
void ProcessManyKernels() {
    auto& manager = gpu::OpenCLManager::GetInstance();
    
    const size_t CLEANUP_THRESHOLD = 100;
    
    for (size_t i = 0; i < 1000; ++i) {
        // Создать/использовать kernel
        cl_program program = manager.GetOrCompileProgram(GetKernelSource(i));
        cl_kernel kernel = manager.GetOrCreateKernel(program, "process_kernel");
        
        // ... использовать kernel ...
        
        // Периодическая очистка
        if (i % CLEANUP_THRESHOLD == 0) {
            size_t cache_size = manager.GetKernelCacheSize();
            if (cache_size > 200) {
                std::cout << "Cleaning up kernel cache (size: " << cache_size << ")\n";
                manager.ClearKernelCache();
            }
        }
    }
}
```

---

### Стратегия 2: Группировка по Program

```cpp
void ProcessKernelGroups() {
    auto& manager = gpu::OpenCLManager::GetInstance();
    
    // Группа 1: Обработка сигналов
    cl_program signal_program = manager.GetOrCompileProgram(signal_kernel_source);
    cl_kernel kernel1 = manager.GetOrCreateKernel(signal_program, "filter");
    cl_kernel kernel2 = manager.GetOrCreateKernel(signal_program, "fft");
    cl_kernel kernel3 = manager.GetOrCreateKernel(signal_program, "correlate");
    
    // ... использовать kernels ...
    
    // Когда группа больше не нужна - очистить
    manager.ClearKernelsForProgram(signal_program);
    
    // Группа 2: Математические операции
    cl_program math_program = manager.GetOrCompileProgram(math_kernel_source);
    cl_kernel kernel4 = manager.GetOrCreateKernel(math_program, "multiply");
    cl_kernel kernel5 = manager.GetOrCreateKernel(math_program, "add");
    
    // ... использовать kernels ...
}
```

---

### Стратегия 3: Мониторинг и Автоматическая Очистка

```cpp
class KernelCacheManager {
private:
    gpu::OpenCLManager& manager_;
    size_t max_cache_size_;
    size_t cleanup_threshold_;
    
public:
    KernelCacheManager(size_t max_size = 500, size_t threshold = 400)
        : manager_(gpu::OpenCLManager::GetInstance())
        , max_cache_size_(max_size)
        , cleanup_threshold_(threshold) {}
    
    cl_kernel GetKernel(cl_program program, const std::string& name) {
        // Проверить размер кэша
        size_t current_size = manager_.GetKernelCacheSize();
        
        if (current_size > cleanup_threshold_) {
            std::cout << "[WARN] Kernel cache size (" << current_size 
                      << ") exceeds threshold. Clearing...\n";
            manager_.ClearKernelCache();
        }
        
        return manager_.GetOrCreateKernel(program, name);
    }
    
    void PrintStatistics() const {
        std::cout << manager_.GetKernelCacheStatistics() << std::endl;
    }
};
```

---

## ⚠️ Важные Замечания

### 1. Не Освобождайте Kernels Вручную

```cpp
// ❌ НЕПРАВИЛЬНО
cl_kernel kernel = manager.GetOrCreateKernel(program, "my_kernel");
// ... использовать ...
clReleaseKernel(kernel);  // ❌ НЕ ДЕЛАЙТЕ ЭТО!

// ✅ ПРАВИЛЬНО
cl_kernel kernel = manager.GetOrCreateKernel(program, "my_kernel");
// ... использовать ...
// Kernel автоматически управляется OpenCLManager
```

---

### 2. Program и Kernel Связаны

```cpp
cl_program program = manager.GetOrCompileProgram(source);

// Все эти kernels связаны с program
cl_kernel k1 = manager.GetOrCreateKernel(program, "kernel1");
cl_kernel k2 = manager.GetOrCreateKernel(program, "kernel2");
cl_kernel k3 = manager.GetOrCreateKernel(program, "kernel3");

// Очистка kernels для program
manager.ClearKernelsForProgram(program);
// Теперь k1, k2, k3 больше недействительны!
```

**Важно:** После `ClearKernelsForProgram()` не используйте старые kernel указатели.

---

### 3. Thread Safety

Все методы kernel cache **thread-safe**:
- ✅ Можно вызывать из разных потоков
- ✅ Мьютексы защищают кэш
- ✅ Нет race conditions

---

## 📈 Производительность

### Преимущества Кэширования

1. **Избежание Повторной Компиляции:**
   - `clCreateKernel()` - быстрая операция (~микросекунды)
   - Но при большом количестве kernels экономия заметна

2. **Снижение Overhead:**
   - Нет повторных вызовов OpenCL API
   - Меньше нагрузка на драйвер GPU

3. **Стабильность:**
   - Один раз создали - используем многократно
   - Меньше точек отказа

### Рекомендации

- **Много kernels (50+):** Используйте периодическую очистку
- **Очень много kernels (200+):** Используйте группировку по program
- **Long-running программы:** Мониторьте размер кэша и очищайте при необходимости

---

## 🔍 Пример: Полный Workflow

```cpp
#include "GPU/opencl_manager.h"
#include <iostream>
#include <vector>

void ProcessMultipleKernelGroups() {
    auto& manager = gpu::OpenCLManager::GetInstance();
    manager.Initialize(CL_DEVICE_TYPE_GPU);
    
    // Группа 1: Генерация сигналов
    std::string signal_source = R"(
        __kernel void generate(__global float* out) { /* ... */ }
        __kernel void modulate(__global float* out) { /* ... */ }
    )";
    
    cl_program signal_program = manager.GetOrCompileProgram(signal_source);
    cl_kernel gen_kernel = manager.GetOrCreateKernel(signal_program, "generate");
    cl_kernel mod_kernel = manager.GetOrCreateKernel(signal_program, "modulate");
    
    // ... использовать ...
    
    // Группа 2: Обработка
    std::string process_source = R"(
        __kernel void filter(__global float* data) { /* ... */ }
        __kernel void transform(__global float* data) { /* ... */ }
    )";
    
    cl_program process_program = manager.GetOrCompileProgram(process_source);
    cl_kernel filter_kernel = manager.GetOrCreateKernel(process_program, "filter");
    cl_kernel transform_kernel = manager.GetOrCreateKernel(process_program, "transform");
    
    // ... использовать ...
    
    // Статистика
    std::cout << manager.GetKernelCacheStatistics() << std::endl;
    // Output:
    // Kernel Cache Statistics:
    //   Cache size: 4 kernels
    //   Cache hits: 0
    //   Cache misses: 4
    
    // Повторное использование
    cl_kernel gen_kernel2 = manager.GetOrCreateKernel(signal_program, "generate");
    // gen_kernel2 == gen_kernel (из кэша!)
    
    std::cout << manager.GetKernelCacheStatistics() << std::endl;
    // Output:
    // Kernel Cache Statistics:
    //   Cache size: 4 kernels
    //   Cache hits: 1  ← увеличилось!
    //   Cache misses: 4
    
    // Очистка группы 1
    manager.ClearKernelsForProgram(signal_program);
    std::cout << "Cache size after cleanup: " 
              << manager.GetKernelCacheSize() << std::endl;
    // Output: Cache size after cleanup: 2
    
    // Полная очистка
    manager.ClearKernelCache();
    std::cout << "Cache size after full cleanup: " 
              << manager.GetKernelCacheSize() << std::endl;
    // Output: Cache size after full cleanup: 0
}

int main() {
    try {
        ProcessMultipleKernelGroups();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

---

## 📚 Связанные Документы

- `API_REFERENCE.md` - Полный API reference
- `PERFORMANCE_OPTIMIZATION.md` - Оптимизация производительности
- `examples_usage.hpp` - Практические примеры

---

**Версия:** 1.0  
**Дата:** 2026-01-10  
**Автор:** AI Assistant

