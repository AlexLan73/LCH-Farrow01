# Пример использования функции signal_sinusoids()

## Описание

Функция `signal_sinusoids()` генерирует комплексные сигналы на GPU, где каждый луч формируется как сумма синусоид с заданными параметрами (амплитуда, период, фаза).

## Полный пример кода

```cpp
#include "generator/generator_gpu_new.h"
#include "interface/lfm_parameters.h"
#include "GPU/opencl_compute_engine.hpp"
#include "GPU/command_queue_pool.hpp"
#include "GPU/opencl_core.hpp"
#include <iostream>
#include <vector>

int main() {
    try {
        // ════════════════════════════════════════════════════════════════
        // ШАГ 1: Инициализация OpenCL
        // ════════════════════════════════════════════════════════════════
        
        gpu::OpenCLCore::Initialize(gpu::DeviceType::GPU);
        gpu::CommandQueuePool::Initialize(4);
        gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
        
        std::cout << "✅ OpenCL initialized\n" << std::endl;
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 2: Создать GeneratorGPU (нужен для конструктора)
        // ════════════════════════════════════════════════════════════════
        
        LFMParameters lfm_params;
        lfm_params.f_start = 100.0f;
        lfm_params.f_stop = 500.0f;
        lfm_params.sample_rate = 12.0e6f;
        lfm_params.num_beams = 256;
        lfm_params.count_points = 1024 * 8;
        
        radar::GeneratorGPU gen(lfm_params);
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 3: Параметры генерации синусоид
        // ════════════════════════════════════════════════════════════════
        
        SinusoidGenParams sin_params;
        sin_params.num_rays = 5;        // 5 лучей
        sin_params.count_points = 1300; // 1300 точек на луч
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 4: Создать map с параметрами синусоид
        // ════════════════════════════════════════════════════════════════
        
        RaySinusoidMap map_ray;
        
        // Луч 0: 2 синусоиды
        map_ray[0] = {
            SinusoidParameter(2.0f, 500.0f, 0.0f),   // amp=2.0, period=500, phase=0°
            SinusoidParameter(1.5f, 301.0f, 7.0f)     // amp=1.5, period=301, phase=7°
        };
        
        // Луч 1: 2 синусоиды
        map_ray[1] = {
            SinusoidParameter(2.1f, 495.0f, 10.0f),  // amp=2.1, period=495, phase=10°
            SinusoidParameter(1.7f, 281.0f, -9.0f)   // amp=1.7, period=281, phase=-9°
        };
        
        // Луч 2: 1 синусоида
        map_ray[2] = {
            SinusoidParameter(1.0f, 650.0f, 45.0f)   // amp=1.0, period=650, phase=45°
        };
        
        // Луч 3 и 4 не описаны - будут использованы дефолтные параметры
        // (amplitude=1.0, period=count_points/2, phase=0°)
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 5: Генерация сигналов на GPU
        // ════════════════════════════════════════════════════════════════
        
        std::cout << "Generating sinusoid signals on GPU..." << std::endl;
        cl_mem gpu_signal = gen.signal_sinusoids(sin_params, map_ray);
        
        // ════════════════════════════════════════════════════════════════
        // ШАГ 6: Синхронизация и чтение результатов
        // ════════════════════════════════════════════════════════════════
        
        gen.ClearGPU(); // Синхронизировать GPU перед чтением
        
        // Прочитать луч 0
        auto beam0 = gen.GetSignalAsVector(0);
        std::cout << "\n✅ Beam 0: " << beam0.size() << " samples" << std::endl;
        std::cout << "   First 5 samples:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), beam0.size()); ++i) {
            std::cout << "     [" << i << "] = " 
                      << beam0[i].real() << " + " 
                      << beam0[i].imag() << "j" << std::endl;
        }
        
        // Прочитать луч 1
        auto beam1 = gen.GetSignalAsVector(1);
        std::cout << "\n✅ Beam 1: " << beam1.size() << " samples" << std::endl;
        std::cout << "   First 5 samples:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), beam1.size()); ++i) {
            std::cout << "     [" << i << "] = " 
                      << beam1[i].real() << " + " 
                      << beam1[i].imag() << "j" << std::endl;
        }
        
        // Прочитать луч 2
        auto beam2 = gen.GetSignalAsVector(2);
        std::cout << "\n✅ Beam 2: " << beam2.size() << " samples" << std::endl;
        
        // Прочитать луч 3 (дефолтные параметры)
        auto beam3 = gen.GetSignalAsVector(3);
        std::cout << "\n✅ Beam 3 (default): " << beam3.size() << " samples" << std::endl;
        
        std::cout << "\n✅ All operations completed successfully!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
```

## Пример 2: Пустой map (дефолтные параметры для всех лучей)

```cpp
// Если map пустой, все лучи генерируются с дефолтными параметрами
SinusoidGenParams params;
params.num_rays = 10;
params.count_points = 1024;

RaySinusoidMap empty_map; // Пустой map

cl_mem gpu_signal = gen.signal_sinusoids(params, empty_map);
// Все 10 лучей будут иметь:
// - amplitude = 1.0
// - period = 1024 / 2 = 512
// - phase = 0°
```

## Пример 3: Частично заполненный map

```cpp
SinusoidGenParams params;
params.num_rays = 10;      // Всего 10 лучей
params.count_points = 1300;

RaySinusoidMap map_ray;

// Описываем только лучи 0, 2, 5
map_ray[0] = {
    SinusoidParameter(2.0f, 500.0f, 0.0f)
};

map_ray[2] = {
    SinusoidParameter(1.5f, 300.0f, 10.0f),
    SinusoidParameter(0.8f, 200.0f, -5.0f)
};

map_ray[5] = {
    SinusoidParameter(3.0f, 400.0f, 90.0f)
};

cl_mem gpu_signal = gen.signal_sinusoids(params, map_ray);
// Лучи 0, 2, 5 - с заданными параметрами
// Лучи 1, 3, 4, 6, 7, 8, 9 - с дефолтными параметрами
```

## Параметры синусоиды

### SinusoidParameter

```cpp
struct SinusoidParameter {
    float amplitude;    // Амплитуда сигнала
    float period;       // Период в точках (количество точек на один период)
    float phase_deg;    // Фаза в градусах (0-360)
};
```

### SinusoidGenParams

```cpp
struct SinusoidGenParams {
    size_t num_rays;      // Количество лучей/антенн
    size_t count_points;  // Количество точек на антенну
};
```

## Логика работы

1. **Если map пустой** → генерируются все лучи с дефолтными параметрами:
   - `amplitude = 1.0f`
   - `period = count_points / 2` (целая часть)
   - `phase_deg = 0.0f`

2. **Если map содержит только часть лучей** → генерируются только описанные лучи с заданными параметрами, остальные - с дефолтными

3. **Каждый луч** = сумма всех синусоид из `map_ray[ray_id]`

4. **Результат** записывается на GPU как комплексный вектор размером `num_rays × count_points`

## Ограничения

- Максимум 10 синусоид на луч (если больше - используются только первые 10)
- Индексы лучей должны быть в диапазоне `[0, num_rays-1]`
- Период должен быть > 0

## Примечания

- Функция автоматически обрабатывает случаи с пустым или частично заполненным map
- Все вычисления выполняются параллельно на GPU
- Результат сохраняется в GPU памяти и может быть прочитан через `GetSignalAsVector()`

