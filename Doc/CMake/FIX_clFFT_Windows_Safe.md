🔧 ИСПРАВЛЕНИЕ ОШИБКИ ЛИНКОВКИ: clFFT LNK2019 (Windows/Linux Safe)
⚠️ ПРОБЛЕМА
text
error LNK2019: ссылка на неразрешенный внешний символ __imp_clfftInitSetupData
Компилятор находит заголовочные файлы clFFT, но не находит DLL/LIB файлы при линковке.

🎯 РЕШЕНИЕ: Windows-specific, не трогаем Linux
ШАГ 1: Заменить секцию поиска clFFT
Найди в CMakeLists.txt:

text
if(CLFFT_FOUND)
    ...
endif()
Замени всю секцию на:

text
# ==================== clFFT CONFIGURATION ====================
# ⚠️ WINDOWS-specific: не влияет на Linux/macOS

if(OPENCL_ENABLED)
    set(CLFFT_LOCAL_DIR "${CMAKE_SOURCE_DIR}/clFFT")
    
    if(IS_WINDOWS)
        # ✅ ТОЛЬКО ДЛЯ WINDOWS: ищем локальный clFFT
        message(STATUS "[WINDOWS] Configuring clFFT for Windows...")
        
        # 1️⃣ Установить include директорию
        if(EXISTS "${CLFFT_LOCAL_DIR}/include/clFFT.h")
            set(CLFFT_INCLUDE_DIR "${CLFFT_LOCAL_DIR}/include")
            message(STATUS "✅ clFFT headers found: ${CLFFT_INCLUDE_DIR}")
        else()
            message(WARNING "❌ clFFT headers NOT found in ${CLFFT_LOCAL_DIR}/include")
            set(CLFFT_FOUND FALSE)
        endif()
        
        # 2️⃣ Определить lib путь для Windows (x64)
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(CLFFT_LIB_SEARCH_PATHS
                "${CLFFT_LOCAL_DIR}/lib/x64/Debug"
                "${CLFFT_LOCAL_DIR}/libx64/Debug"
                "${CLFFT_LOCAL_DIR}/lib/x64"
                "${CLFFT_LOCAL_DIR}/libx64"
            )
        else()
            set(CLFFT_LIB_SEARCH_PATHS
                "${CLFFT_LOCAL_DIR}/lib/x64/Release"
                "${CLFFT_LOCAL_DIR}/libx64/Release"
                "${CLFFT_LOCAL_DIR}/lib/x64"
                "${CLFFT_LOCAL_DIR}/libx64"
            )
        endif()
        
        # 3️⃣ Найти clFFT.lib
        find_library(CLFFT_LIB
            NAMES clFFT.lib clFFT
            PATHS ${CLFFT_LIB_SEARCH_PATHS}
            NO_DEFAULT_PATH
        )
        
        if(CLFFT_LIB)
            set(CLFFT_FOUND TRUE)
            message(STATUS "✅ clFFT.lib found: ${CLFFT_LIB}")
            get_filename_component(CLFFT_LIBDIR "${CLFFT_LIB}" DIRECTORY)
            message(STATUS "   Directory: ${CLFFT_LIBDIR}")
            
            # 4️⃣ Найти DLL для рантайма
            find_file(CLFFT_DLL
                NAMES clFFT.dll
                PATHS "${CLFFT_LIBDIR}" "${CLFFT_LOCAL_DIR}/lib/x64" "${CLFFT_LOCAL_DIR}/libx64"
                NO_DEFAULT_PATH
            )
            
            if(CLFFT_DLL)
                message(STATUS "✅ clFFT.dll found: ${CLFFT_DLL}")
            else()
                message(WARNING "⚠️  clFFT.dll not found (may cause runtime errors)")
            endif()
        else()
            set(CLFFT_FOUND FALSE)
            message(WARNING "❌ clFFT.lib NOT found on Windows")
            message(STATUS "   Searched in:")
            foreach(path ${CLFFT_LIB_SEARCH_PATHS})
                message(STATUS "   - ${path}")
            endforeach()
        endif()
        
    elseif(IS_LINUX OR APPLE)
        # ✅ ТОЛЬКО ДЛЯ LINUX/MACOS: используем системные пакеты
        message(STATUS "[LINUX/MACOS] Configuring clFFT for Linux/macOS...")
        
        # Попытка 1: системный find_package
        find_package(clFFT QUIET)
        
        # Попытка 2: pkg-config
        if(NOT CLFFT_FOUND)
            find_package(PkgConfig QUIET)
            if(PKG_CONFIG_FOUND)
                pkg_check_modules(CLFFT QUIET clFFT)
                if(CLFFT_FOUND)
                    set(CLFFT_LIB "${CLFFT_LIBRARIES}")
                    set(CLFFT_INCLUDE_DIR "${CLFFT_INCLUDE_DIRS}")
                    message(STATUS "✅ clFFT found via pkg-config")
                endif()
            endif()
        endif()
        
        # Попытка 3: ручной поиск в стандартных местах
        if(NOT CLFFT_FOUND)
            find_library(CLFFT_LIB
                NAMES clFFT
                PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /opt/AMD/clFFT/lib64
            )
            
            find_path(CLFFT_INCLUDE_DIR
                NAMES clFFT.h
                PATHS /usr/local/include /usr/include /opt/AMD/clFFT/include
            )
            
            if(CLFFT_LIB AND CLFFT_INCLUDE_DIR)
                set(CLFFT_FOUND TRUE)
                message(STATUS "✅ clFFT found (manual search)")
            endif()
        endif()
        
        if(NOT CLFFT_FOUND)
            message(WARNING "❌ clFFT NOT found on Linux/macOS")
            message(STATUS "   Install with:")
            message(STATUS "   Ubuntu/Debian: sudo apt install libclfft-dev")
            message(STATUS "   Fedora: sudo dnf install clFFT-devel")
            message(STATUS "   macOS: brew install amd-clpeak  (or build from source)")
        else()
            message(STATUS "✅ clFFT library: ${CLFFT_LIB}")
            message(STATUS "✅ clFFT include: ${CLFFT_INCLUDE_DIR}")
        endif()
    endif()
endif()

# ==================== END clFFT CONFIGURATION ====================
ШАГ 2: Обновить target_link_libraries
Найди секцию if(OPENCL_ENABLED) с target_link_libraries и замени clFFT часть на:

text
if(OPENCL_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PRIVATE OPENCL_ENABLED=1)
    message(STATUS "✅ OpenCL libraries linked")
    
    # ==================== clFFT LINKING ====================
    if(CLFFT_FOUND AND CLFFT_LIB)
        # Линковать clFFT
        target_link_libraries(${PROJECT_NAME} PRIVATE "${CLFFT_LIB}")
        target_include_directories(${PROJECT_NAME} PRIVATE "${CLFFT_INCLUDE_DIR}")
        target_compile_definitions(${PROJECT_NAME} PRIVATE CLFFT_FOUND=1)
        
        message(STATUS "✅ clFFT linked successfully")
        message(STATUS "   Library: ${CLFFT_LIB}")
        message(STATUS "   Include: ${CLFFT_INCLUDE_DIR}")
        
        # ✅ Копировать DLL для Windows
        if(IS_WINDOWS AND CLFFT_DLL)
            add_custom_command(
                TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CLFFT_DLL}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                COMMENT "Copying clFFT.dll to output directory"
            )
        endif()
        
    else()
        message(WARNING "⚠️  clFFT NOT configured")
        target_compile_definitions(${PROJECT_NAME} PRIVATE CLFFT_FOUND=0)
        
        if(IS_WINDOWS)
            message(WARNING "   Reason: Windows - check if ${CMAKE_SOURCE_DIR}/clFFT exists")
        else()
            message(WARNING "   Reason: Linux/macOS - install libclfft-dev package")
        endif()
    endif()
    # ==================== END clFFT LINKING ====================
    
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE OPENCL_ENABLED=0 CLFFT_FOUND=0)
    message(STATUS "OpenCL support disabled")
endif()
🧪 ТЕСТИРОВАНИЕ
Windows
bash
cmake -B build -G "Visual Studio 17 2022" -DENABLE_OPENCL=ON
Должны быть сообщения:

text
[WINDOWS] Configuring clFFT for Windows...
✅ clFFT headers found: E:\C++\LCH-Farrow01\clFFT\include
✅ clFFT.lib found: E:\C++\LCH-Farrow01\clFFT\libx64\clFFT.lib
✅ clFFT.dll found: E:\C++\LCH-Farrow01\clFFT\libx64\clFFT.dll
✅ clFFT linked successfully
Ubuntu/Linux
bash
cmake -B build -G Ninja -DENABLE_OPENCL=ON
Должны быть сообщения:

text
[LINUX/MACOS] Configuring clFFT for Linux/macOS...
✅ clFFT found via pkg-config
✅ clFFT library: /usr/lib/x86_64-linux-gnu/libclFFT.so
✅ clFFT include: /usr/include
✅ clFFT linked successfully
🎯 KEY DIFFERENCES
Параметр	Windows	Linux/macOS
Условие	IS_WINDOWS	IS_LINUX OR APPLE
Поиск	Локальная папка ./clFFT/	Системные пакеты
Тип файла	.lib + .dll	.so или .a
Метод поиска	find_library в path	find_package + pkg-config
Влияние на Linux	❌ Ноль	✅ Независимо работает
💾 ЕСЛИ ОШИБКИ
Windows:

bash
# Очистить кеш CMake
rm -rf build/

# Проверить структуру
ls E:/C++/LCH-Farrow01/clFFT/

# Пересоздать
cmake -B build -G "Visual Studio 17 2022"
Linux:

bash
# Установить clFFT
sudo apt install libclfft-dev

# Пересобрать
cmake -B build -G Ninja
ninja -C build
Готово! Это не повредит Linux! ✅
