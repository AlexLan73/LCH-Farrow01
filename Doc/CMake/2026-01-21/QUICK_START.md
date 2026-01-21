# ⚡ БЫСТРЫЙ СТАРТ: LCH-Farrow01 CMake

## 📥 Установка новой структуры (5 минут)

### 1. Скопируйте файлы:

```bash
# Создайте папку cmake/ в корне проекта
mkdir -p cmake

# Скопируйте модули конфигурации:
cp platform-detection.cmake cmake/
cp gpu-config.cmake cmake/
cp dependencies.cmake cmake/
cp compiler-options.cmake cmake/
cp debug-config.cmake cmake/

# Замените главный CMakeLists.txt:
mv CMakeLists.txt CMakeLists_OLD_BACKUP.txt
cp CMakeLists_ROOT.txt CMakeLists.txt

# Замените CMakePresets.json:
mv CMakePresets.json CMakePresets_OLD.json
cp CMakePresets-NEW.json CMakePresets.json
```

### 2. Замените CMakeLists в подпапках:

```bash
# src/CMakeLists.txt
mv src/CMakeLists.txt src/CMakeLists_OLD.txt
cp src-CMakeLists.txt src/CMakeLists.txt

# src/ManagerOpenCL/CMakeLists.txt
cp ManagerOpenCL-CMakeLists.txt src/ManagerOpenCL/CMakeLists.txt

# src/GPU/CMakeLists.txt
cp GPU-CMakeLists.txt src/GPU/CMakeLists.txt

# src/Test/CMakeLists.txt
cp Test-CMakeLists.txt src/Test/CMakeLists.txt
```

### 3. Обновите CMakePresets.json под ваши пути (ВАЖНО!)

Отредактируйте `CMakePresets.json` и установите правильные пути на вашем компе:

#### Для Linux с ROCm:
```json
{
  "configurePresets": [
    {
      "name": "linux-rocm-opencl",
      "environment": {
        "ROCM_HOME": "/opt/rocm",                          ← ВАШ путь!
        "LD_LIBRARY_PATH": "/opt/rocm/lib:$penv{...}",
        "PATH": "/opt/rocm/bin:$penv{PATH}"
      }
    }
  ]
}
```

#### Для Windows с CUDA:
```json
{
  "configurePresets": [
    {
      "name": "windows-cuda",
      "cacheVariables": {
        "CUDA_TOOLKIT_ROOT_DIR": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
                                                          ← ВАШ путь!
      }
    }
  ]
}
```

## 🚀 Сборка проекта

### Linux с ROCm/OpenCL:

```bash
# Configure
cmake --preset linux-rocm-opencl

# Build
cmake --build build/linux-rocm -j8
```

### Linux с CUDA:

```bash
cmake --preset linux-cuda
cmake --build build/linux-cuda -j8
```

### Windows с CUDA (Visual Studio):

```bash
cmake --preset windows-cuda
cmake --build build/windows-cuda --config Release -j8
```

## ✅ Проверка конфигурации

После конфигурации вы увидите примерно такой вывод:

```
╔══════════════════════════════════════════════════════╗
║  LCH-Farrow01: GPU-Accelerated Radar Signal Generator║
║  Loading configuration modules...                    ║
╚══════════════════════════════════════════════════════╝

✅ Platform detected: LINUX (UNIX)
   Compiler: GCC 11.2.0

🔍 GPU Configuration:
  ENABLE_CUDA: OFF
  ENABLE_OPENCL: ON
  TYPE_GPU: AMD-GPU
  CUDA_ARCH: auto

📚 Searching for dependencies...
🔍 Searching for OpenCL...
✅ OpenCL found!
   Version: 3.0
   Include: /opt/rocm/include
   Libraries: /opt/rocm/lib/libamd_comgr.so

🔍 Searching for clFFT...
   [Linux mode] Looking for system clFFT...
✅ clFFT found via pkg-config

📦 Processing: src/
✅ ManagerOpenCL library configured
🎮 Processing: src/GPU/
✅ Created object library: lfm_gpu
🧪 Processing: src/Test/
✅ Test module configured
📋 Creating main executable: LCH-Farrow1
✅ Linked: OpenCL
✅ Linked: clFFT
✅ Main executable configured

╔════════════════════════════════════════╗
║      CONFIGURATION SUMMARY             ║
╠════════════════════════════════════════╣
║ Platform: Linux
║ Compiler: GCC 11.2.0
║ C++ Standard: C17
║ Build Type: Release
╠════════════════════════════════════════╣
║ CUDA Support: False
║ OpenCL Support: True
║ clFFT Support: True
║ nlohmann_json Support: True
╚════════════════════════════════════════╝

✅ Project configured successfully!
```

## 🔍 Если что-то не работает

### Проблема: OpenCL не найден

**Linux:**
```bash
sudo apt install opencl-headers ocl-icd-opencl-dev
```

**Windows:** Проверьте `CMakePresets.json` - правильный ли путь к CUDA?

### Проблема: clFFT не найден

**Linux:**
```bash
sudo apt install libclfft-dev
```

**Windows:** Проверьте, есть ли папка `${sourceDir}/clFFT/` с `include/clFFT.h`

### Проблема: CMake не может найти файлы

Убедитесь, что вы в корне проекта:
```bash
pwd
# должно быть: .../LCH-FARROW01

ls cmake/platform-detection.cmake
# должен вывести файл, а не ошибку
```

## 📋 Файлы которые вы получили:

| Файл | Назначение |
|------|-----------|
| `CMakeLists.txt` | Главный (подключает все модули) |
| `cmake/platform-detection.cmake` | Определение OS |
| `cmake/gpu-config.cmake` | Выбор CUDA/OpenCL |
| `cmake/dependencies.cmake` | Поиск библиотек |
| `cmake/compiler-options.cmake` | Флаги компилятора |
| `cmake/debug-config.cmake` | Отладочная информация |
| `src/CMakeLists.txt` | Главная сборка приложения |
| `src/ManagerOpenCL/CMakeLists.txt` | OpenCL библиотека |
| `src/GPU/CMakeLists.txt` | GPU модуль |
| `src/Test/CMakeLists.txt` | Тесты |
| `CMakePresets.json` | Предустановки для разных конфигураций |
| `CMAKE_STRUCTURE_GUIDE.md` | Подробное руководство |

## 🎯 Ключевые особенности

✅ **Модульность** - каждый компонент в своей папке с CMakeLists
✅ **Надежность** - библиотеки ищутся в отдельном файле `dependencies.cmake`
✅ **Переносимость** - пути к библиотекам в `CMakePresets.json`, не теряются между сборками
✅ **Масштабируемость** - легко добавлять новые модули
✅ **Кроссплатформность** - Windows/Linux с одной структурой

## 💡 Совет

Если вы добавляете новый .cpp или .h файл в какой-то модуль, просто отредактируйте соответствующий CMakeLists.txt:

```cmake
# Например, в src/GPU/CMakeLists.txt:
set(GPU_SOURCES
    antenna_fft_proc_max.cpp
    generator_gpu_new.cpp
    new_file.cpp              ← добавьте здесь
)
```

Никогда не нужно трогать главный CMakeLists.txt!

## 📞 Готово! 🎉

Структура создана и готова к использованию. Все файлы надежны, модульны и не будут ломаться!
