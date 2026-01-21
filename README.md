# LCH-Farrow01

Проект для работы с вычислениями на GPU с использованием OpenCL. Включает в себя модули для управления памятью, выполнения ядер и обработки данных.

## Описание

Этот проект предоставляет инструменты и библиотеки для эффективной работы с GPU через OpenCL. Он включает в себя:

- **Генераторы LFM сигналов** на GPU (`GeneratorGPU`)
- **Процессор дробной задержки** с интерполяцией Лагранжа (`FractionalDelayProcessor`)
- **Управление памятью** на GPU с автоматическим управлением ресурсами
- **Выполнение вычислительных ядер** через высокоуровневый API
- **Поддержка FFT** через clFFT для обработки антенных данных
- **Оптимизированные алгоритмы** для радарных приложений

## Структура проекта

```
LCH-Farrow01/
├── include/
│   ├── GPU/                    # GPU модули (генераторы, процессоры)
│   │   ├── generator_gpu_new.h
│   │   ├── fractional_delay_processor.hpp
│   │   └── antenna_fft_proc_max.h
│   ├── ManagerOpenCL/          # Менеджер OpenCL
│   │   ├── opencl_manager.h
│   │   ├── opencl_compute_engine.hpp
│   │   └── gpu_memory_buffer.hpp
│   └── interface/              # Интерфейсы параметров
│       └── lfm_parameters.h
├── src/
│   ├── GPU/                    # Реализация GPU модулей
│   ├── ManagerOpenCL/          # Реализация менеджера OpenCL
│   └── Test/                   # Тесты
│       └── test_fractional_delay_processor.cpp
├── Doc/
│   ├── Project_doc/            # Подробная документация
│   ├── Генератор/              # Документация генераторов
│   ├── GPU/                    # Документация GPU
│   └── NewManagerOpenCl/       # Документация OpenCL менеджера
├── clFFT/                      # Библиотека FFT
├── MemoryBank/                 # Заметки и память проекта
├── CMakeLists.txt
├── CMakePresets.json           # CMake пресеты
└── lagrange_matrix.json        # Матрица Лагранжа (48×5)
```

## Установка и запуск

### Требования

- **OpenCL SDK** (NVIDIA CUDA / AMD ROCm / Intel OpenCL)
- **CMake 3.15+**
- **Компилятор с поддержкой C++17** (GCC 7+, Clang 5+, MSVC 2017+)
- **nlohmann/json** (автоматически загружается через CMake)
- **clFFT** (опционально, для FFT операций)

### Сборка

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/yourusername/LCH-Farrow01.git
   cd LCH-Farrow01
   ```

2. Создайте директорию для сборки и выполните сборку:

   ```bash
   # Используя CMake Presets (рекомендуется)
   cmake --preset linux-nvidia-debug
   cmake --build build/linux-nvidia-debug -j8
   
   # Или классический способ
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   cmake --build . -j8
   ```

### Запуск

**Основная программа:**
```bash
cd build/linux-nvidia-debug/src
./LCH-Farrow1
```

**Тесты:**
```bash
cd build/linux-nvidia-debug/src/Test
./test_fractional_delay    # Тесты процессора дробной задержки
```

**Отладка (VS Code):**
- Нажмите `F5` для запуска отладчика
- Убедитесь что проект собран в `Debug` режиме

## Основные компоненты

### FractionalDelayProcessor
Процессор дробной задержки для LFM сигналов с интерполяцией Лагранжа:
- ✅ Матрица Лагранжа 48×5 (точность 0.02 отсчёта)
- ✅ IN-PLACE обработка до 256 лучей одновременно
- ✅ Производительность: ~20 Gsamples/sec (RTX 3060)
- ✅ GPU профилирование через OpenCL Events
- ✅ Поддержка задержек в отсчётах и градусах

**Пример использования:**
```cpp
auto config = radar::FractionalDelayConfig::Standard();
auto lagrange = radar::LagrangeMatrix::LoadFromJSON("lagrange_matrix.json");
radar::FractionalDelayProcessor processor(config, lagrange);

std::vector<radar::DelayParams> delays(num_beams);
// ... настройка задержек ...

processor.Process(gpu_buffer, delays);
```

### GeneratorGPU
Генератор LFM (линейно-частотно-модулированных) сигналов на GPU:
- Генерация комплексных сигналов для радарных приложений
- Поддержка множественных лучей (антенн)
- Интеграция с процессором дробной задержки

## Документация

- **[Карта проекта](Project_map.md)** - Подробная структура и описание проекта
- **[FractionalDelayProcessor](Doc/Project_doc/FractionalDelayProcessor_Detailed.md)** - Подробная документация процессора дробной задержки
- **[GeneratorGPU](Doc/Project_doc/GeneratorGPU_Detailed.md)** - Руководство по генератору LFM сигналов
- **[OpenCLComputeEngine](Doc/Project_doc/OpenCLComputeEngine_Detailed.md)** - Документация OpenCL движка
- **[GPUMemoryBuffer](Doc/Project_doc/GPUMemoryBuffer_Detailed.md)** - Управление памятью GPU
- [Руководство по использованию генераторов GPU](Doc/Генератор/2/FINAL_ANSWER_READ_GPU.md)
- [Руководство по OpenCL](Doc/NewManagerOpenCl/OPENCL_GUIDE.md)

## Производительность

### Бенчмарки (NVIDIA RTX 3060)

| Операция | Конфигурация | Throughput |
|----------|--------------|------------|
| Fractional Delay | 256 × 65536 | 20.68 Gsamples/sec |
| Fractional Delay | 128 × 32768 | 21.47 Gsamples/sec |
| Fractional Delay | 64 × 16384 | 21.47 Gsamples/sec |

## Лицензия

Этот проект лицензирован под лицензией MIT. Подробности смотрите в файле LICENSE.

## Контакты

Для вопросов и предложений, пожалуйста, обращайтесь по адресу: your.email@example.com

---