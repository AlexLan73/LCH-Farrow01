# Установка библиотеки clFFT для Windows

## Проблема
Проект использует библиотеку clFFT для выполнения FFT на GPU, но библиотека не найдена в проекте.

## Решение

### Вариант 1: Скачать предкомпилированную библиотеку (рекомендуется)

1. Скачайте библиотеку clFFT для Windows:
   - Официальный репозиторий: https://github.com/clMathLibraries/clFFT
   - Или используйте предкомпилированные версии для Windows

2. Создайте структуру директорий:
   ```
   clFFT/
   ├── include/          (уже есть)
   │   ├── clFFT.h
   │   └── ...
   └── lib/
       └── x64/
           ├── clFFT.lib  (нужно добавить)
           └── clFFT.dll  (нужно добавить)
   ```

3. Поместите файлы:
   - `clFFT.lib` → `clFFT/lib/x64/clFFT.lib`
   - `clFFT.dll` → `clFFT/lib/x64/clFFT.dll`

### Вариант 2: Собрать из исходников

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/clMathLibraries/clFFT.git
   cd clFFT
   ```

2. Соберите библиотеку с помощью CMake:
   ```bash
   mkdir build
   cd build
   cmake .. -G "Visual Studio 17 2022" -A x64
   cmake --build . --config Release
   ```

3. Скопируйте файлы:
   - `build/library/Release/clFFT.lib` → `clFFT/lib/x64/clFFT.lib`
   - `build/library/Release/clFFT.dll` → `clFFT/lib/x64/clFFT.dll`

### Вариант 3: Использовать FetchContent (автоматическая загрузка)

CMakeLists.txt уже настроен для автоматической загрузки через FetchContent.
Однако, для Windows это может не работать, так как clFFT требует сборки.

## Проверка установки

После установки библиотеки, пересоберите проект:
```bash
cmake --build build --config Release
```

Если библиотека найдена, вы увидите сообщение:
```
✅ clFFT found!
   Include: E:/C++/LCH-Farrow01/clFFT/include
   Library: E:/C++/LCH-Farrow01/clFFT/lib/x64/clFFT.lib
✅ clFFT library linked
```

## Альтернативное решение

Если библиотека clFFT недоступна, можно временно отключить функциональность `AntennaFFTProcMax` через условную компиляцию, но это потребует изменений в коде.

