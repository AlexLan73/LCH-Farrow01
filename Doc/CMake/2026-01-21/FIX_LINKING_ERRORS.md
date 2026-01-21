# 🔧 ИСПРАВЛЕНИЕ ОШИБОК ЛИНКОВКИ

## 🔴 ПРОБЛЕМА:

Ошибки типа:
```
undefined reference to `gpu::OpenCLComputeEngine::GetInstance()'
undefined reference to `radar::GeneratorGPU::GeneratorGPU(LFMParameters const&)'
undefined reference to `test_antenna_fft_proc_max::run_all_tests()'
```

### Причина:
- `.cpp` файлы не скомпилированы в исполняемый файл
- Неправильные зависимости между модулями
- Недостаточные include директории

---

## ✅ РЕШЕНИЕ - 3 КОМАНДЫ:

```bash
# 1️⃣ Замените src/CMakeLists.txt
cp src-CMakeLists-FIXED.txt src/CMakeLists.txt

# 2️⃣ Замените src/GPU/CMakeLists.txt
cp GPU-CMakeLists-FIXED.txt src/GPU/CMakeLists.txt

# 3️⃣ Замените src/Test/CMakeLists.txt
cp Test-CMakeLists-FIXED.txt src/Test/CMakeLists.txt
```

Затем пересоберите:
```bash
rm -rf build/
cmake --preset linux-nvidia-debug
cmake --build build/linux-nvidia-debug -j4
```

---

## 📋 ЧТО БЫЛО ИСПРАВЛЕНО:

### src/CMakeLists.txt:
✅ Добавлена правильная линковка всех модулей:
```cmake
target_link_libraries(LCH-Farrow1 PRIVATE
    lfm_opencl_manager      # STATIC библиотека
    lfm_gpu                 # OBJECT модуль
    lfm_tests               # OBJECT модуль
    OpenCL::OpenCL
)
```

✅ Добавлены все include директории для поиска заголовков

✅ Правильно подключены clFFT и nlohmann_json

### src/GPU/CMakeLists.txt:
✅ Правильно создана OBJECT библиотека (скомпилируется в main)

✅ Правильные зависимости: зависит от lfm_opencl_manager

✅ Все include директории добавлены

### src/Test/CMakeLists.txt:
✅ Правильно создана OBJECT библиотека

✅ Правильные зависимости: зависит от lfm_gpu и lfm_opencl_manager

✅ Все нужные include директории добавлены

---

## 🔗 ГРАФ ЗАВИСИМОСТЕЙ (правильный):

```
main.cpp
    ↓
LCH-Farrow1 (исполняемый файл)
    ↓
    ├─ lfm_opencl_manager (STATIC) 
    │   ├─ OpenCL::OpenCL
    │   └─ clFFT
    │
    ├─ lfm_gpu (OBJECT - компилируется в main)
    │   ├─ lfm_opencl_manager
    │   ├─ OpenCL::OpenCL
    │   └─ clFFT
    │
    └─ lfm_tests (OBJECT - компилируется в main)
        ├─ lfm_gpu
        ├─ lfm_opencl_manager
        ├─ OpenCL::OpenCL
        ├─ clFFT
        └─ nlohmann_json
```

---

## ✨ ПОСЛЕ ИСПРАВЛЕНИЯ ВЫ ДОЛЖНЫ УВИДЕТЬ:

```
✅ Processing: src/ (Main build)
✅ Processing: src/ManagerOpenCL/
✅ Created library: lfm_opencl_manager (STATIC)
✅ ManagerOpenCL library configured

✅ Processing: src/GPU/
✅ Created object library: lfm_gpu
✅ GPU module configured

✅ Processing: src/Test/
✅ Created object library: lfm_tests
✅ Test module configured

✅ Creating main executable: LCH-Farrow1
✅ Linked library: lfm_opencl_manager
✅ Linked: OpenCL
✅ Linked: clFFT
✅ Main executable configured: LCH-Farrow1

✅ Project configured successfully!
```

Затем сборка:
```
[100%] Linking CXX executable src/LCH-Farrow1
[100%] Built target LCH-Farrow1
```

---

## 🚀 ЗАТЕМ ЗАПУСТИТЕ:

```bash
# Release сборка
rm -rf build/
cmake --preset linux-nvidia-opencl
cmake --build build/linux-nvidia -j8
./build/linux-nvidia/LCH-Farrow1

# Или Debug
cmake --preset linux-nvidia-debug
cmake --build build/linux-nvidia-debug -j4
./build/linux-nvidia-debug/LCH-Farrow1
```

---

## 📝 ФАЙЛЫ КОТОРЫЕ НУЖНО ЗАМЕНИТЬ:

```
Старые файлы              Новые файлы
─────────────────────────────────────────────
src/CMakeLists.txt     ← src-CMakeLists-FIXED.txt
src/GPU/CMakeLists.txt ← GPU-CMakeLists-FIXED.txt
src/Test/CMakeLists.txt ← Test-CMakeLists-FIXED.txt
```

---

## 🆘 ЕСЛИ ЕЩЕ НЕ РАБОТАЕТ:

```bash
# 1. Проверьте структуру
ls -la src/ManagerOpenCL/*.cpp
ls -la src/GPU/*.cpp
ls -la src/Test/*.cpp

# 2. Проверьте заголовки
ls -la include/GPU/*.hpp
ls -la include/Test/*.hpp

# 3. Проверьте конфигурацию
cmake --preset linux-nvidia-debug --verbose 2>&1 | head -100

# 4. Проверьте линковку
cmake --preset linux-nvidia-debug --verbose 2>&1 | grep -i "link"
```

---

**Используйте исправленные файлы и пересоберите!** 🚀
