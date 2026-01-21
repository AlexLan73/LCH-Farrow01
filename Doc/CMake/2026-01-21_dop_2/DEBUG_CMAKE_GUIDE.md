# 🔧 ПОЛНЫЙ ГАЙД ПО ОТЛАДКЕ CMake ОШИБОК

## 🎯 ДИАГНОСТИКА ВАШЕЙ ОШИБКИ:

### Ошибка:
```
CMake Error at src/ManagerOpenCL/CMakeLists.txt:47 (add_library):
  Cannot find source file:
    command_queue_pool.cpp
```

### Причина:
В CMakeLists.txt файл указан, но его нет в папке.

### Решение:
Удалить ссылку на несуществующий файл.

---

## ✅ БЫСТРОЕ ИСПРАВЛЕНИЕ (1 КОМАНДА):

```bash
cp ManagerOpenCL-CMakeLists-FIXED.txt src/ManagerOpenCL/CMakeLists.txt
```

Затем пересоберите:
```bash
rm -rf build/
cmake --preset linux-nvidia-debug
cmake --build build/linux-nvidia-debug -j4
```

---

## 📋 ПРОВЕРКА СТРУКТУРЫ ФАЙЛОВ:

### Проверьте что у вас есть:

```bash
# Проверьте src/ManagerOpenCL/
ls -la src/ManagerOpenCL/

# Должны быть:
# opencl_manager.cpp         ✅
# gpu_memory_manager.cpp     ✅
# CMakeLists.txt             ✅ (новый исправленный)
```

### Проверьте что у вас есть в include/:

```bash
# Проверьте заголовки
ls -la include/ManagerOpenCL/

# Должны быть:
# opencl_manager.h           ✅
# gpu_memory_manager.hpp     ✅
```

---

## 🔍 ЕСЛИ ПОСЛЕ ИСПРАВЛЕНИЯ ЕЩЕ ОШИБКИ:

### Ошибка 1: "Cannot find OpenCL"
```
Решение:
sudo apt install opencl-headers ocl-icd-opencl-dev
```

### Ошибка 2: "Cannot find clFFT"
```
Решение:
sudo apt install libclfft-dev
```

### Ошибка 3: "nlohmann_json not found"
```
Решение:
sudo apt install nlohmann-json3-dev
```

---

## 🚨 ЕСЛИ СОВСЕМ НЕ РАБОТАЕТ:

### Попробуйте это (полная очистка):

```bash
# 1. Полная очистка
rm -rf build/
rm -rf CMakeCache.txt
rm -rf CMakeFiles/

# 2. Убедитесь что CMakePresets.json правильный
cat CMakePresets.json | grep -A 5 "linux-nvidia-debug"

# 3. Убедитесь что CMakeLists.txt правильный
head -20 CMakeLists.txt

# 4. Убедитесь что файлы скопированы
ls src/ManagerOpenCL/CMakeLists.txt
ls src/GPU/CMakeLists.txt
ls src/Test/CMakeLists.txt

# 5. Попробуйте заново конфигурировать
cmake --preset linux-nvidia-debug

# 6. Если ошибка - пришлите полный вывод
cmake --preset linux-nvidia-debug 2>&1 | tee cmake-error.log
```

---

## 📝 ЧЕКЛИСТ ПЕРЕД СБОРКОЙ:

- [ ] Файл `ManagerOpenCL-CMakeLists-FIXED.txt` скопирован как `src/ManagerOpenCL/CMakeLists.txt`
- [ ] Файл `GPU-CMakeLists.txt` скопирован как `src/GPU/CMakeLists.txt`
- [ ] Файл `Test-CMakeLists.txt` скопирован как `src/Test/CMakeLists.txt`
- [ ] Файл `CMakeLists_ROOT.txt` скопирован как `CMakeLists.txt` (главный)
- [ ] Файл `CMakePresets-SIMPLIFIED.json` скопирован как `CMakePresets.json`
- [ ] OpenCL установлен: `pkg-config --cflags --libs openCL`
- [ ] clFFT установлен: `pkg-config --cflags --libs clfft`
- [ ] nlohmann_json установлен: `apt list --installed | grep nlohmann`

---

## 🚀 ПОСЛЕ УСПЕШНОЙ КОНФИГУРАЦИИ:

Должны увидеть:
```
✅ Platform detected: LINUX (UNIX)
✅ Compiler: GNU 13.3.0
✅ GPU Configuration: ENABLE_CUDA: OFF, ENABLE_OPENCL: ON
✅ Searching for dependencies...
✅ OpenCL found!
✅ clFFT found via pkg-config
✅ nlohmann_json found!
✅ Dependencies Summary:
   CUDA_ENABLED: FALSE
   OPENCL_ENABLED: TRUE
   CLFFT_FOUND: 1
   NLOHMANN_JSON_FOUND: TRUE
✅ Project configured successfully!
```

Затем собирайте:
```bash
cmake --build build/linux-nvidia-debug -j4
```

---

## 💡 ПОЛЕЗНЫЕ КОМАНДЫ:

```bash
# Посмотреть конфигурацию
cmake --preset linux-nvidia-debug --verbose

# Пересборка (без очистки кэша)
cmake --build build/linux-nvidia-debug -j4

# Полная пересборка
cmake --build build/linux-nvidia-debug --clean-first -j4

# Только конфигурация (без сборки)
cmake --preset linux-nvidia-debug

# Просмотр переменных CMake
cmake --preset linux-nvidia-debug --trace-expand | grep "OpenCL\|clFFT"
```

---

## 📞 ЕСЛИ ВСЕ ЕЩЕ НЕ РАБОТАЕТ:

1. Пришлите вывод: `cmake --preset linux-nvidia-debug 2>&1 | head -50`
2. Пришлите вывод: `ls -la src/ManagerOpenCL/`
3. Пришлите вывод: `cat src/ManagerOpenCL/CMakeLists.txt | head -50`

---

**Используйте: ManagerOpenCL-CMakeLists-FIXED.txt ✅**

**Успехов!** 🚀
