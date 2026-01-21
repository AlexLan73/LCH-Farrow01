# Карта проекта LCH-Farrow01

## Обзор
Этот проект посвящен разработке и оптимизации вычислений на GPU с использованием OpenCL. Проект включает в себя различные модули для управления памятью, выполнения ядер и обработки данных.

## Структура проекта

### Основные директории

1. **`include/`**
   - Содержит заголовки и интерфейсы для основных компонентов проекта.
   - Поддиректории:
     - `generator/` - Заголовки для генераторов.
     - `GPU/` - Заголовки для управления GPU и вычислениями:
       - `generator_gpu_new.h` - Генератор LFM сигналов
       - `antenna_fft_proc_max.h` - FFT обработка антенн
       - `fractional_delay_processor.hpp` - Процессор дробной задержки
       - `lagrange_matrix_loader.hpp` - Загрузка матрицы Лагранжа
     - `ManagerOpenCL/` - Заголовки менеджера OpenCL:
       - `opencl_manager.h` - Основной менеджер
       - `gpu_memory_manager.hpp` - Управление памятью
       - `gpu_memory_buffer.hpp` - GPU буферы (RAII)
       - `opencl_core.hpp` - Базовые функции OpenCL
       - `opencl_compute_engine.hpp` - Вычислительный движок
       - `command_queue_pool.hpp` - Пул очередей команд
       - `kernel_program.hpp` - Управление программами
       - `i_memory_buffer.hpp` - Интерфейс буферов памяти
     - `interface/` - Интерфейсы для параметров и конфигураций:
       - `lfm_parameters.h` - Параметры LFM сигналов
       - `antenna_fft_params.h` - Параметры FFT обработки
     - `Test/` - Заголовки тестов (если нужны)
     - `radar/` - Заголовки радарных алгоритмов (если нужны)

2. **`src/`**
   - Исходные файлы для реализации компонентов.
   - Поддиректории:
     - `generator/` - Реализация генераторов.
     - `GPU/` - Реализация управления GPU и вычислениями:
       - `generator_gpu_new.cpp` - Генератор LFM сигналов
       - `antenna_fft_proc_max.cpp` - FFT обработка антенн
       - `fractional_delay_processor.cpp` - Процессор дробной задержки
     - `ManagerOpenCL/` - Менеджер OpenCL:
       - `opencl_manager.cpp` - Основной менеджер
       - `gpu_memory_manager.cpp` - Управление памятью
       - `opencl_core.cpp` - Базовые функции OpenCL
       - `opencl_compute_engine.cpp` - Вычислительный движок
       - `command_queue_pool.cpp` - Пул очередей команд
       - `kernel_program.cpp` - Управление программами
     - `Test/` - Тесты:
       - `test_fractional_delay_processor.cpp` - Тесты процессора задержки
       - `test_antenna_fft_proc_max.cpp` - Тесты FFT обработки
       - `test_signal_sinusoids.cpp` - Тесты генераторов

3. **`Doc/`**
   - Документация и примеры использования.
   - Поддиректории:
     - `Генератор/` - Документация по генераторам.
     - `GPU/` - Документация по работе с GPU.
     - `NewManagerOpenCl/` - Документация по новому менеджеру OpenCL.
     - `Project_doc/` - Подробная документация по основным компонентам:
       - `FractionalDelayProcessor_Detailed.md` - Процессор дробной задержки
       - `GeneratorGPU_Detailed.md` - Генератор LFM сигналов
       - `OpenCLComputeEngine_Detailed.md` - OpenCL вычислительный движок
       - `GPUMemoryBuffer_Detailed.md` - Управление памятью GPU
       - `OOP_SOLID_Patterns_Reference.md` - Паттерны проектирования
       - `Classes_and_Structures_Documentation.md` - Описание классов и структур

4. **`clFFT/`**
   - Библиотека для работы с FFT на GPU.

5. **`MemoryBank/`**
   - Документация и заметки по управлению памятью.

### Основные файлы

- **`CMakeLists.txt`** - Конфигурация сборки проекта.
- **`compile_commands.json`** - Файл для поддержки инструментов анализа кода.
- **`.gitignore`** - Список файлов, игнорируемых Git.
- **`run.bat` и `run.sh`** - Скрипты для запуска проекта.

## Описание компонентов

### 1. Генераторы
- **`generator_gpu_new.h`** и **`generator_gpu_new.cpp`**
  Реализуют генерацию LFM (линейно-частотно-модулированных) сигналов на GPU с использованием OpenCL.

### 2. Управление GPU
- **`opencl_manager.h`** и **`opencl_manager.cpp`**
  Основные классы для управления контекстами и очередями команд OpenCL.

- **`opencl_compute_engine.h`** и **`opencl_compute_engine.cpp`**
  Реализуют вычислительные ядра и управление ресурсами GPU.

- **`opencl_core.h`** и **`opencl_core.cpp`**
  Базовые функции инициализации OpenCL устройств и контекстов.

- **`command_queue_pool.h`** и **`command_queue_pool.cpp`**
  Пул очередей команд для параллельной обработки.

- **`kernel_program.h`** и **`kernel_program.cpp`**
  Управление OpenCL программами и kernel'ами.

### 3. Управление памятью
- **`gpu_memory_manager.h`** и **`gpu_memory_manager.cpp`**
  Управление буферами памяти на GPU.

- **`gpu_memory_buffer.hpp`**
  Высокоуровневый интерфейс для работы с GPU буферами (RAII).

### 4. Обработка сигналов
- **`fractional_delay_processor.hpp`** и **`fractional_delay_processor.cpp`**
  Процессор дробной задержки для LFM сигналов с интерполяцией Лагранжа (48×5).
  Поддерживает IN-PLACE обработку до 256 лучей одновременно.
  Производительность: ~20 Gsamples/sec на RTX 3060.

- **`antenna_fft_proc_max.h`** и **`antenna_fft_proc_max.cpp`**
  Обработка FFT для антенных данных.

### 5. Библиотека clFFT
- **`clFFT.h`** и **`clAmdFft.h`**
  Используются для выполнения FFT (Быстрое Преобразование Фурье) на GPU.

## Зависимости
- **OpenCL SDK** - Для GPU вычислений
- **CMake 3.15+** - Система сборки
- **clFFT** - Библиотека FFT для GPU (опционально)
- **nlohmann/json** - Парсинг JSON для загрузки матриц
- **C++17** - Требуемый стандарт языка

## Примеры использования
- **`example_usage.hpp`** - Примеры использования основных компонентов.
- **`gpu_memory_examples.cpp`** - Примеры работы с памятью GPU.

## Документация
- **`Doc/Project_doc/FractionalDelayProcessor_Detailed.md`** - Подробная документация процессора дробной задержки
- **`Doc/Project_doc/GeneratorGPU_Detailed.md`** - Руководство по использованию генераторов GPU
- **`Doc/Project_doc/OpenCLComputeEngine_Detailed.md`** - Документация OpenCL движка
- **`Doc/Project_doc/GPUMemoryBuffer_Detailed.md`** - Управление памятью GPU
- **`Doc/Project_doc/OOP_SOLID_Patterns_Reference.md`** - Паттерны проектирования (GRASP & GoF)
- **`GENERATOR_GPU_USAGE.md`** - Руководство по использованию генераторов GPU
- **`OPENCL_GUIDE.md`** - Руководство по работе с OpenCL
- **`SUMMARY.md`** - Краткое описание проекта и его компонентов

## Сборка и запуск
1. Убедитесь, что установлены все зависимости.
2. Используйте CMake для сборки проекта.
3. Запустите проект с помощью скриптов `run.bat` или `run.sh`.

## Дополнительные ресурсы
- **`MemoryBank/`** - Документация и заметки по управлению памятью.
- **`Doc/`** - Подробная документация по всем компонентам проекта.

---