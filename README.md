# LCH-Farrow01

Проект для работы с вычислениями на GPU с использованием OpenCL. Включает в себя модули для управления памятью, выполнения ядер и обработки данных.

## Описание

Этот проект предоставляет инструменты и библиотеки для эффективной работы с GPU через OpenCL. Он включает в себя:

- Генераторы данных на GPU
- Управление памятью на GPU
- Выполнение вычислительных ядер
- Поддержка FFT через clFFT

## Структура проекта

```
LCH-Farrow01/
├── include/
│   ├── generator/
│   ├── GPU/
│   └── interface/
├── src/
│   ├── generator/
│   └── GPU/
├── Doc/
│   ├── Генератор/
│   ├── GPU/
│   └── NewManagerOpenCl/
├── clFFT/
├── MemoryBank/
├── CMakeLists.txt
├── compile_commands.json
├── .gitignore
├── run.bat
└── run.sh
```

## Установка и запуск

### Требования

- OpenCL SDK
- CMake
- Компилятор с поддержкой C++17

### Сборка

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/yourusername/LCH-Farrow01.git
   cd LCH-Farrow01
   ```

2. Создайте директорию для сборки и выполните сборку:

   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

### Запуск

Используйте скрипты для запуска:

- На Windows:
  ```bash
  run.bat
  ```

- На Unix-подобных системах:
  ```bash
  ./run.sh
  ```

## Документация

- [Карта проекта](Project_map.md) - Подробная структура и описание проекта.
- [Руководство по использованию генераторов GPU](Doc/Генератор/2/FINAL_ANSWER_READ_GPU.md)
- [Руководство по OpenCL](Doc/NewManagerOpenCl/OPENCL_GUIDE.md)

## Лицензия

Этот проект лицензирован под лицензией MIT. Подробности смотрите в файле LICENSE.

## Контакты

Для вопросов и предложений, пожалуйста, обращайтесь по адресу: your.email@example.com

---