# LCH-Farrow01: CMake Structure Guide

## 📋 Новая структура CMake-файлов

```
project_root/
├── CMakeLists.txt                 ← ГЛАВНЫЙ (подключает всё)
├── CMakePresets.json              ← ЛОКАЛЬНЫЕ НАСТРОЙКИ (не трогать!)
├── cmake/
│   ├── platform-detection.cmake   ← Определение OS
│   ├── gpu-config.cmake           ← Выбор GPU (CUDA/OpenCL)
│   ├── dependencies.cmake         ← Поиск библиотек (КРИТИЧНО!)
│   ├── compiler-options.cmake     ← Флаги компилятора
│   └── debug-config.cmake         ← Отладка и логирование
└── src/
    ├── CMakeLists.txt             ← Главное приложение
    ├── main.cpp                   ← Точка входа
    ├── ManagerOpenCL/
    │   ├── CMakeLists.txt         ← Библиотека OpenCL (STATIC)
    │   ├── opencl_manager.cpp
    │   ├── gpu_memory_manager.cpp
    │   ├── command_queue_pool.cpp
    │   └── ...
    ├── GPU/
    │   ├── CMakeLists.txt         ← GPU модуль (OBJECT library)
    │   ├── antenna_fft_proc_max.cpp
    │   └── generator_gpu_new.cpp
    └── Test/
        ├── CMakeLists.txt         ← Тесты (OBJECT library)
        ├── test_antenna_fft_proc_max.cpp
        └── test_signal_sinusoids.cpp
```

## 🔧 Как использовать

### Шаг 1: Замена файлов

```bash
# Скопируйте в корень проекта:
CMakeLists.txt                  # Переименуйте старый CMakeLists.txt в CMakeLists_OLD.txt
                                # и замените на CMakeLists_ROOT.txt -> CMakeLists.txt

# Создайте папку cmake/ и скопируйте туда:
cmake/platform-detection.cmake
cmake/gpu-config.cmake
cmake/dependencies.cmake
cmake/compiler-options.cmake
cmake/debug-config.cmake

# Замените src/CMakeLists.txt на src-CMakeLists.txt
# Создайте новые CMakeLists в подпапках:
src/ManagerOpenCL/CMakeLists.txt  ← ManagerOpenCL-CMakeLists.txt
src/GPU/CMakeLists.txt            ← GPU-CMakeLists.txt
src/Test/CMakeLists.txt           ← Test-CMakeLists.txt
```

### Шаг 2: Обновите CMakePresets.json

Замените старый CMakePresets.json на CMakePresets-NEW.json

**ВАЖНО:** На каждом компе может быть свой CMakePresets.json с локальными путями:

```json
{
  "configurePresets": [
    {
      "name": "linux-rocm-opencl",
      "environment": {
        "ROCM_HOME": "/opt/rocm",           ← ВАШ путь к ROCm
        "LD_LIBRARY_PATH": "..."            ← ВАШ путь
      }
    }
  ]
}
```

### Шаг 3: Команды сборки

#### Linux с ROCm/OpenCL

```bash
# Configure
cmake --preset linux-rocm-opencl

# Build
cmake --build build/linux-rocm -j8

# Debug
cmake --preset linux-rocm-debug
cmake --build build/linux-rocm-debug -j4
```

#### Linux с CUDA

```bash
cmake --preset linux-cuda
cmake --build build/linux-cuda -j8
```

#### Windows с CUDA

```bash
cmake --preset windows-cuda
cmake --build build/windows-cuda --config Release -j8
```

#### Windows с OpenCL

```bash
cmake --preset windows-opencl
cmake --build build/windows-opencl --config Release -j8
```

## 🎯 Архитектура библиотек

### Структура зависимостей:

```
main executable
    ↓
    ├─→ lfm_opencl_manager (STATIC library)
    │   └─→ OpenCL::OpenCL
    │   └─→ clFFT (если найдена)
    │
    ├─→ lfm_gpu (OBJECT library)
    │   └─→ lfm_opencl_manager
    │
    └─→ lfm_tests (OBJECT library)
        └─→ lfm_opencl_manager
```

### Типы библиотек:

| Библиотека | Тип | Назначение |
|-----------|-----|-----------|
| `lfm_opencl_manager` | STATIC | Управление OpenCL контекстом и памятью |
| `lfm_gpu` | OBJECT | GPU вычисления (antenna_fft, generator) |
| `lfm_tests` | OBJECT | Тесты и примеры |

## ✨ Преимущества новой структуры

✅ **Модульность** - каждый модуль имеет свой CMakeLists
✅ **Надежность** - зависимости отделены в `dependencies.cmake`
✅ **Переносимость** - пути к библиотекам в `CMakePresets.json` (не теряются)
✅ **Масштабируемость** - легко добавлять новые компоненты
✅ **Отладка** - функции в `debug-config.cmake` для диагностики
✅ **Кроссплатформность** - одна структура для Windows/Linux
✅ **Безопасность** - главный CMakeLists не ломается от изменений

## 🛡️ ВАЖНЫЕ ПРАВИЛА

### ❌ НЕ ТРОГАЙТЕ:

```
cmake/dependencies.cmake          ← Поиск библиотек (базовый)
cmake/compiler-options.cmake      ← Флаги компилятора
cmake/platform-detection.cmake    ← Определение платформы
```

### ✏️ МОЖНО МЕНЯТЬ:

```
CMakePresets.json                 ← Локальные пути (на каждом компе)
src/ManagerOpenCL/CMakeLists.txt  ← Если добавляются файлы
src/GPU/CMakeLists.txt            ← Если добавляются файлы
src/Test/CMakeLists.txt           ← Если добавляются тесты
```

## 🔍 Отладка

Если что-то сломалось, включите verbose logging:

```bash
# В cmake/gpu-config.cmake раскомментируйте:
set(VERBOSE_GPU_CONFIG ON)

# Или передайте в командной строке:
cmake -B build -DVERBOSE_GPU_CONFIG=ON
```

Вызовите функцию логирования в `debug-config.cmake`:

```cmake
# В CMakeLists.txt раскомментируйте:
log_build_configuration()
```

## 📞 Структура файлов создана!

Теперь у вас есть:
1. ✅ Главный CMakeLists.txt (минимальный, чистый)
2. ✅ 5 модулей конфигурации в `cmake/`
3. ✅ 4 CMakeLists для подмодулей (`src/*/CMakeLists.txt`)
4. ✅ Обновленный CMakePresets.json с разными конфигурациями
5. ✅ Это руководство

**Вся информация о библиотеках отделена и НЕ ЛОМАЕТСЯ при изменениях!**
