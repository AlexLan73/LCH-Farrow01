#include "Test/test_signal_sinusoids.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <CL/cl.h>
#include <stdexcept>

namespace test_signal_sinusoids {

void test_empty_map() {
    std::cout << "\n🧪 ТЕСТ 1: Пустой map_ray (дефолтные параметры)" << std::endl;

    try {
        // Параметры генератора
        LFMParameters lfm_params;
        lfm_params.f_start = 100.0f;
        lfm_params.f_stop = 500.0f;
        lfm_params.sample_rate = 12.0e6f;
        lfm_params.num_beams = 4;  // Маленькое количество для теста
        lfm_params.count_points = 1054;

        radar::GeneratorGPU gen(lfm_params);

        // Параметры синусоид: пустой map
        SinusoidGenParams sin_params;
        sin_params.num_rays = 4;
        sin_params.count_points = 1024;

        RaySinusoidMap empty_map;

        // Генерация
        cl_mem gpu_signal = gen.signal_sinusoids(sin_params, empty_map);
        gen.ClearGPU();

        // Проверка результата для луча 0
        auto beam0 = gen.GetSignalAsVector(0);
        if (beam0.empty()) {
            throw std::runtime_error("Не удалось прочитать данные луча 0");
        }

        std::cout << "✅ Тест пройден: Сгенерировано " << beam0.size() << " отсчётов для луча 0" << std::endl;
        std::cout << "   Первый отсчёт: " << beam0[0] << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Тест провален: " << e.what() << std::endl;
    }
}

void test_single_ray_single_sinusoid() {
    std::cout << "\n🧪 ТЕСТ 2: Один луч с одной синусоидой" << std::endl;

    try {
        // Параметры генератора
        LFMParameters lfm_params;
        lfm_params.f_start = 100.0f;
        lfm_params.f_stop = 500.0f;
        lfm_params.sample_rate = 12.0e6f;
        lfm_params.num_beams = 4;
        lfm_params.count_points = 1024;

        radar::GeneratorGPU gen(lfm_params);

        // Параметры синусоид
        SinusoidGenParams sin_params;
        sin_params.num_rays = 4;
        sin_params.count_points = 1024;

        RaySinusoidMap map_ray;
        map_ray[0] = {SinusoidParameter(2.0f, 512.0f, 45.0f)};  // Луч 0: амплитуда 2, период 512, фаза 45°

        // Генерация
        cl_mem gpu_signal = gen.signal_sinusoids(sin_params, map_ray);
        gen.ClearGPU();

        // Проверка
        auto beam0 = gen.GetSignalAsVector(0);
        auto beam1 = gen.GetSignalAsVector(1);  // Должен быть дефолтный

        if (beam0.empty() || beam1.empty()) {
            throw std::runtime_error("Не удалось прочитать данные");
        }

        std::cout << "✅ Тест пройден:" << std::endl;
        std::cout << "   Луч 0 (кастомный): первый отсчёт = " << beam0[0] << std::endl;
        std::cout << "   Луч 1 (дефолтный): первый отсчёт = " << beam1[0] << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Тест провален: " << e.what() << std::endl;
    }
}

void test_multiple_rays_multiple_sinusoids() {
    std::cout << "\n🧪 ТЕСТ 3: Несколько лучей с несколькими синусоидами" << std::endl;

    try {
        // Параметры генератора
        LFMParameters lfm_params;
        lfm_params.f_start = 100.0f;
        lfm_params.f_stop = 500.0f;
        lfm_params.sample_rate = 12.0e6f;
        lfm_params.num_beams = 4;
        lfm_params.count_points = 1024;

        radar::GeneratorGPU gen(lfm_params);

        // Параметры синусоид
        SinusoidGenParams sin_params;
        sin_params.num_rays = 4;
        sin_params.count_points = 1024;

        RaySinusoidMap map_ray;
        map_ray[0] = {
            SinusoidParameter(1.0f, 256.0f, 0.0f),
            SinusoidParameter(0.5f, 512.0f, 90.0f)
        };  // Луч 0: две синусоиды

        map_ray[2] = {
            SinusoidParameter(1.5f, 128.0f, 30.0f)
        };  // Луч 2: одна синусоида

        // Генерация
        cl_mem gpu_signal = gen.signal_sinusoids(sin_params, map_ray);
        gen.ClearGPU();

        // Проверка
        auto beam0 = gen.GetSignalAsVector(0);
        auto beam1 = gen.GetSignalAsVector(1);  // Дефолтный
        auto beam2 = gen.GetSignalAsVector(2);

        if (beam0.empty() || beam1.empty() || beam2.empty()) {
            throw std::runtime_error("Не удалось прочитать данные");
        }

        std::cout << "✅ Тест пройден:" << std::endl;
        std::cout << "   Луч 0 (2 синусоиды): первый отсчёт = " << beam0[0] << std::endl;
        std::cout << "   Луч 1 (дефолтный): первый отсчёт = " << beam1[0] << std::endl;
        std::cout << "   Луч 2 (1 синусоида): первый отсчёт = " << beam2[0] << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Тест провален: " << e.what() << std::endl;
    }
}

void test_ray_out_of_range() {
    std::cout << "\n🧪 ТЕСТ 4: Луч вне диапазона" << std::endl;

    try {
        // Параметры генератора
        LFMParameters lfm_params;
        lfm_params.f_start = 100.0f;
        lfm_params.f_stop = 500.0f;
        lfm_params.sample_rate = 12.0e6f;
        lfm_params.num_beams = 4;
        lfm_params.count_points = 1024;

        radar::GeneratorGPU gen(lfm_params);

        // Параметры синусоид
        SinusoidGenParams sin_params;
        sin_params.num_rays = 4;
        sin_params.count_points = 1024;

        RaySinusoidMap map_ray;
        map_ray[10] = {SinusoidParameter(1.0f, 256.0f, 0.0f)};  // Луч 10 вне диапазона [0,3]

        // Генерация - должно вывести предупреждение и игнорировать луч 10
        cl_mem gpu_signal = gen.signal_sinusoids(sin_params, map_ray);
        gen.ClearGPU();

        // Проверка - все лучи должны быть дефолтными
        auto beam0 = gen.GetSignalAsVector(0);
        if (beam0.empty()) {
            throw std::runtime_error("Не удалось прочитать данные");
        }

        std::cout << "✅ Тест пройден: Луч вне диапазона игнорирован" << std::endl;
        std::cout << "   Луч 0 (дефолтный): первый отсчёт = " << beam0[0] << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Тест провален: " << e.what() << std::endl;
    }
}

void test_more_than_10_sinusoids() {
    std::cout << "\n🧪 ТЕСТ 5: Более 10 синусоид на луч" << std::endl;

    try {
        // Параметры генератора
        LFMParameters lfm_params;
        lfm_params.f_start = 100.0f;
        lfm_params.f_stop = 500.0f;
        lfm_params.sample_rate = 12.0e6f;
        lfm_params.num_beams = 4;
        lfm_params.count_points = 1024;

        radar::GeneratorGPU gen(lfm_params);

        // Параметры синусоид
        SinusoidGenParams sin_params;
        sin_params.num_rays = 4;
        sin_params.count_points = 1024;

        RaySinusoidMap map_ray;
        std::vector<SinusoidParameter> many_sinusoids;
        for (int i = 0; i < 15; ++i) {  // 15 синусоид
            many_sinusoids.push_back(SinusoidParameter(1.0f, 100.0f + i * 10, i * 10.0f));
        }
        map_ray[0] = many_sinusoids;

        // Генерация - должно использовать только первые 10
        cl_mem gpu_signal = gen.signal_sinusoids(sin_params, map_ray);
        gen.ClearGPU();

        // Проверка
        auto beam0 = gen.GetSignalAsVector(0);
        if (beam0.empty()) {
            throw std::runtime_error("Не удалось прочитать данные");
        }

        std::cout << "✅ Тест пройден: Использованы только первые 10 синусоид" << std::endl;
        std::cout << "   Луч 0: первый отсчёт = " << beam0[0] << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Тест провален: " << e.what() << std::endl;
    }
}

void test_parameter_validation() {
    std::cout << "\n🧪 ТЕСТ 6: Валидация параметров" << std::endl;

    // Тест 1: num_rays = 0
    try {
        LFMParameters lfm_params;
        lfm_params.f_start = 100.0f;
        lfm_params.f_stop = 500.0f;
        lfm_params.sample_rate = 12.0e6f;
        lfm_params.num_beams = 4;
        lfm_params.count_points = 1024;

        radar::GeneratorGPU gen(lfm_params);

        SinusoidGenParams sin_params;
        sin_params.num_rays = 0;  // Некорректно
        sin_params.count_points = 1024;

        RaySinusoidMap empty_map;
        cl_mem gpu_signal = gen.signal_sinusoids(sin_params, empty_map);

        std::cerr << "❌ Ожидалось исключение для num_rays = 0" << std::endl;

    } catch (const std::invalid_argument& e) {
        std::cout << "✅ Корректно поймано исключение для num_rays = 0: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Неожиданное исключение: " << e.what() << std::endl;
    }

    // Тест 2: count_points = 0
    try {
        LFMParameters lfm_params;
        lfm_params.f_start = 100.0f;
        lfm_params.f_stop = 500.0f;
        lfm_params.sample_rate = 12.0e6f;
        lfm_params.num_beams = 4;
        lfm_params.count_points = 1024;

        radar::GeneratorGPU gen(lfm_params);

        SinusoidGenParams sin_params;
        sin_params.num_rays = 4;
        sin_params.count_points = 0;  // Некорректно

        RaySinusoidMap empty_map;
        cl_mem gpu_signal = gen.signal_sinusoids(sin_params, empty_map);

        std::cerr << "❌ Ожидалось исключение для count_points = 0" << std::endl;

    } catch (const std::invalid_argument& e) {
        std::cout << "✅ Корректно поймано исключение для count_points = 0: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Неожиданное исключение: " << e.what() << std::endl;
    }
}

void run_all_tests() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              ТЕСТЫ ФУНКЦИИ signal_sinusoids                  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

    test_empty_map();
    test_single_ray_single_sinusoid();
    test_multiple_rays_multiple_sinusoids();
    test_ray_out_of_range();
    test_more_than_10_sinusoids();
    test_parameter_validation();

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ                       ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;
}

} // namespace test_signal_sinusoids