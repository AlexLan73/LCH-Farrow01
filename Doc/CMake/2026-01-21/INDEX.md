# 📑 INDEX: Все файлы CMake для LCH-Farrow01

## 🎯 НАЧНИТЕ ОТСЮДА!

**Прочитайте в этом порядке:**

1. 👉 **INSTALLATION_SUMMARY.md** - краткий план установки (2 мин)
2. 👉 **QUICK_START.md** - быстрый старт (5 мин)
3. 👉 **CMAKE_STRUCTURE_GUIDE.md** - подробное руководство
4. 👉 **README_CMAKE.txt** - полный обзор

---

## 📦 ВСЕ ФАЙЛЫ (15 шт)

### 🏗️ ГЛАВНЫЕ (замените в корне):
```
CMakeLists_ROOT.txt             → переименуйте в CMakeLists.txt
CMakePresets-NEW.json           → переименуйте в CMakePresets.json
```

### 📚 МОДУЛИ КОНФИГУРАЦИИ (скопируйте в cmake/):
```
platform-detection.cmake        → cmake/platform-detection.cmake
gpu-config.cmake               → cmake/gpu-config.cmake
dependencies.cmake             → cmake/dependencies.cmake ⭐ КРИТИЧНЫЙ
compiler-options.cmake         → cmake/compiler-options.cmake
debug-config.cmake            → cmake/debug-config.cmake
```

### 🔧 CMAKELISTS ПОДМОДУЛЕЙ (замените/создайте):
```
src-CMakeLists.txt             → src/CMakeLists.txt
ManagerOpenCL-CMakeLists.txt    → src/ManagerOpenCL/CMakeLists.txt
GPU-CMakeLists.txt             → src/GPU/CMakeLists.txt
Test-CMakeLists.txt            → src/Test/CMakeLists.txt
```

### 📖 ДОКУМЕНТАЦИЯ (справка):
```
INSTALLATION_SUMMARY.md        ← План установки (начните отсюда!)
QUICK_START.md                 ← Быстрый старт (5 минут)
CMAKE_STRUCTURE_GUIDE.md       ← Полное руководство
README_CMAKE.txt               ← Полный обзор
INDEX.md                       ← Этот файл
```

---

## ⚡ 3 ШАГА УСТАНОВКИ

### 1️⃣ Скопировать файлы (bash script):
```bash
mkdir -p cmake

cp platform-detection.cmake cmake/
cp gpu-config.cmake cmake/
cp dependencies.cmake cmake/
cp compiler-options.cmake cmake/
cp debug-config.cmake cmake/

mv CMakeLists.txt CMakeLists_OLD.txt
cp CMakeLists_ROOT.txt CMakeLists.txt

mv CMakePresets.json CMakePresets_OLD.json
cp CMakePresets-NEW.json CMakePresets.json

mv src/CMakeLists.txt src/CMakeLists_OLD.txt
cp src-CMakeLists.txt src/CMakeLists.txt

cp ManagerOpenCL-CMakeLists.txt src/ManagerOpenCL/CMakeLists.txt
cp GPU-CMakeLists.txt src/GPU/CMakeLists.txt
cp Test-CMakeLists.txt src/Test/CMakeLists.txt
```

### 2️⃣ Отредактировать CMakePresets.json
Установите правильные пути для вашей системы (ROCM_HOME, CUDA_HOME и т.д.)

### 3️⃣ Собрать проект
```bash
cmake --preset linux-rocm-opencl
cmake --build build/linux-rocm -j8
```

---

## 🎨 АРХИТЕКТУРА

```
CMakeLists.txt (главный)
    ↓
    ├─→ cmake/platform-detection.cmake (определение OS)
    ├─→ cmake/gpu-config.cmake (CUDA/OpenCL)
    ├─→ cmake/dependencies.cmake (поиск библиотек)
    ├─→ cmake/compiler-options.cmake (флаги)
    └─→ cmake/debug-config.cmake (отладка)
    
    ↓
    src/CMakeLists.txt
        ├─→ src/ManagerOpenCL/CMakeLists.txt (библиотека)
        ├─→ src/GPU/CMakeLists.txt (модуль)
        └─→ src/Test/CMakeLists.txt (тесты)
```

---

## 🔑 КЛЮЧЕВЫЕ МОМЕНТЫ

### ✅ Модульность
- Каждый компонент независим
- Легко добавлять новые модули

### ✅ Надежность
- Поиск библиотек в отдельном файле `dependencies.cmake`
- Не ломается при отсутствии некоторых библиотек

### ✅ Переносимость
- Пути хранятся в `CMakePresets.json` (локальный на каждом компе)
- Один файл CMakeLists для всех платформ

### ✅ Масштабируемость
- Главный CMakeLists остается минимальным
- Легко расширять функциональность

---

## 📋 БЫСТРАЯ СПРАВКА

| Файл | Что делает | Менять? |
|------|-----------|---------|
| CMakeLists.txt | Главный, подключает всё | ❌ НЕТ |
| cmake/dependencies.cmake | Поиск библиотек | ❌ НЕТ |
| cmake/platform-detection.cmake | Определение OS | ❌ НЕТ |
| CMakePresets.json | Локальные пути | ✅ ДА |
| src/*/CMakeLists.txt | Компоненты | ✅ ДА (если добавить файлы) |

---

## 🚀 ПРИМЕРЫ КОМАНД

### Linux с ROCm:
```bash
cmake --preset linux-rocm-opencl
cmake --build build/linux-rocm -j8
```

### Linux с CUDA:
```bash
cmake --preset linux-cuda
cmake --build build/linux-cuda -j8
```

### Windows с CUDA:
```bash
cmake --preset windows-cuda
cmake --build build/windows-cuda --config Release
```

---

## 🆘 ЕСЛИ ОШИБКА

1. Проверьте CMakePresets.json - правильные ли пути?
2. Прочитайте CMAKE_STRUCTURE_GUIDE.md (раздел "Отладка")
3. Включите логирование в dependencies.cmake
4. Проверьте установлены ли библиотеки (OpenCL, clFFT)

---

## ✨ РЕЗУЛЬТАТ

После успешной установки:

✅ Модульная структура CMake  
✅ Отдельная библиотека ManagerOpenCL  
✅ GPU модуль как OBJECT library  
✅ Тесты интегрированы  
✅ Поддержка CUDA и OpenCL  
✅ Кроссплатформная сборка  
✅ Полная документация  

---

## 📞 ДОКУМЕНТАЦИЯ

| Документ | Для кого | Время |
|----------|---------|-------|
| **INSTALLATION_SUMMARY.md** | Все | 2 мин |
| **QUICK_START.md** | Торопящиеся | 5 мин |
| **CMAKE_STRUCTURE_GUIDE.md** | Интересующиеся | 15 мин |
| **README_CMAKE.txt** | Полный обзор | 20 мин |
| **INDEX.md** | Навигация | 2 мин |

---

## 🎯 НАЧНИТЕ С:

1. **INSTALLATION_SUMMARY.md** - узнайте план
2. **QUICK_START.md** - установите за 5 минут
3. **Тестируйте!** - `cmake --preset linux-rocm-opencl`

---

## 🎉 ВСЁ ГОТОВО!

Вы получили надежную, модульную и масштабируемую структуру CMake!

**Приступайте к установке!** 🚀
