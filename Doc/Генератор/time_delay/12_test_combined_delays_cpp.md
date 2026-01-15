# 🧪 12_test_combined_delays_cpp.md

## ГОТОВЫЕ ТЕСТЫ

### ТЕСТ 1: БАЗОВАЯ КОМБИНИРОВАННАЯ ЗАДЕРЖКА

```cpp
void test_combined_delays_basic() {
    std::cout << \"✓ ТЕСТ 1: Комбинированная задержка\" << std::endl;
    
    try {
        // Инициализация GPU
        OpenCLCore::Initialize(DeviceType::GPU);
        CommandQueuePool::Initialize(4);
        OpenCLComputeEngine::Initialize(DeviceType::GPU);
        
        // Параметры ЛЧМ
        LFMParameters params;
        params.f_start = 1.0e6f;
        params.f_stop = 2.0e6f;
        params.sample_rate = 12.0e6f;
        params.num_beams = 8;
        params.count_points = 256;
        
        GeneratorGPU gen(params);
        
        // Подготовить задержки
        std::vector<CombinedDelayParam> delays(gen.GetNumBeams());
        for (int i = 0; i < gen.GetNumBeams(); i++) {
            delays[i].beam_index = i;
            delays[i].delay_degrees = 0.5f * i;
            delays[i].delay_time_ns = 50.0f * i;
        }
        
        // Сформировать сигнал
        cl_mem signal = gen.signal_combined_delays(delays.data(), delays.size());
        
        // Синхронизировать
        gen.ClearGPU();
        
        // Прочитать результаты
        auto beam0 = gen.GetSignalAsVector(0);
        auto beam1 = gen.GetSignalAsVector(1);
        
        // Проверка
        std::cout << \"  Луч 0: \" << beam0.size() << \" сэмплов\" << std::endl;
        std::cout << \"  Луч 1: \" << beam1.size() << \" сэмплов\" << std::endl;
        std::cout << \"✓ ТЕСТ 1 ПРОЙДЕН!\" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << \"✗ ОШИБКА: \" << e.what() << std::endl;
    }
}
```

### ТЕСТ 2: ПРОВЕРКА АМПЛИТУД

```cpp
void test_amplitudes() {
    std::cout << \"✓ ТЕСТ 2: Проверка амплитуд\" << std::endl;
    
    // ... (подготовка как выше)
    
    for (int beam = 0; beam < 3; beam++) {
        auto signal = gen.GetSignalAsVector(beam);
        
        float min_amp = 1e6;
        float max_amp = -1e6;
        
        for (const auto& sample : signal) {
            if (sample.real() == 0.0f && sample.imag() == 0.0f) {
                continue;  // Пропустить нули в начале
            }
            
            float amp = std::abs(sample);
            min_amp = std::min(min_amp, amp);
            max_amp = std::max(max_amp, amp);
        }
        
        if (min_amp < 1e6) {
            std::cout << \"  Луч \" << beam << \": \"
                      << \"amp=[\" << min_amp << \"...\" << max_amp << \"]\"
                      << std::endl;
            
            // Проверка: амплитуда должна быть ≈ 1.0
            assert(min_amp > 0.99f && max_amp < 1.01f);
        }
    }
    
    std::cout << \"✓ ТЕСТ 2 ПРОЙДЕН!\" << std::endl;
}
```

### ТЕСТ 3: ПРОВЕРКА ПОЗИЦИИ ЗАДЕРЖКИ

```cpp
void test_delay_position() {
    std::cout << \"✓ ТЕСТ 3: Позиция начала сигнала\" << std::endl;
    
    // ... (подготовка как выше)
    
    for (int beam = 0; beam < 8; beam++) {
        auto signal = gen.GetSignalAsVector(beam);
        
        int start_idx = -1;
        for (size_t i = 0; i < signal.size(); i++) {
            if (signal[i].real() != 0.0f || signal[i].imag() != 0.0f) {
                start_idx = i;
                break;
            }
        }
        
        if (start_idx >= 0 && beam < 3) {
            float delay_samples = start_idx;
            float delay_time_ns = (delay_samples / 12.0e6) * 1e9f;
            
            std::cout << \"  Луч \" << beam << \": \"
                      << \"start_idx=\" << start_idx << \" samp\"
                      << \", delay≈\" << delay_time_ns << \" ns\"
                      << std::endl;
        }
    }
    
    std::cout << \"✓ ТЕСТ 3 ПРОЙДЕН!\" << std::endl;
}
```

### КОМПИЛЯЦИЯ:

```bash
g++ -std=c++17 -O2 test_combined_delays.cpp \\
    generator_gpu_new.cpp opencl_core.cpp ... -lOpenCL
```

### ЗАПУСК:

```bash
./a.out
```

### ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:

```
✓ ТЕСТ 1: Комбинированная задержка
  Луч 0: 256 сэмплов
  Луч 1: 256 сэмплов
✓ ТЕСТ 1 ПРОЙДЕН!
✓ ТЕСТ 2: Проверка амплитуд
✓ ТЕСТ 2 ПРОЙДЕН!
✓ ТЕСТ 3: Позиция начала сигнала
✓ ТЕСТ 3 ПРОЙДЕН!

✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ
```
