#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include "fft/antenna_fft_proc_max.h"
#include "generator/generator_gpu_new.h"
#include "interface/lfm_parameters.h"
#include "GPU/opencl_compute_engine.hpp"

/**
 * @brief Тесты для AntennaFFTProcMax
 * 
 * Тесты проверяют FFT обработку сигналов с поиском максимальных амплитуд.
 */
namespace test_antenna_fft_proc_max {

/**
 * @brief Тест 1: Базовый тест с GeneratorGPU::signal_sinusoids
 * 
 * Использует 5 лучей, 1000 точек, пустой map (дефолтные параметры)
 */
void test_basic_with_generator();

/**
 * @brief Тест 2: Проверка вычисления nFFT
 */
void test_nfft_calculation();

/**
 * @brief Тест 3: Проверка поиска максимумов
 */
void test_maxima_search();

/**
 * @brief Тест 4: Проверка профилирования
 */
void test_profiling();

/**
 * @brief Тест 5: Проверка вывода результатов
 */
void test_output();

/**
 * @brief Запуск всех тестов
 */
void run_all_tests();

} // namespace test_antenna_fft_proc_max

