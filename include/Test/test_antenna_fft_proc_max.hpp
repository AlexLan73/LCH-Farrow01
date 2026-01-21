#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include "GPU/antenna_fft_proc_max.h"
#include "GPU/generator_gpu_new.h"
#include "interface/lfm_parameters.h"
#include "ManagerOpenCL/opencl_compute_engine.hpp"

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
 * @brief Тест 6: ProcessNew() с малыми данными (5 лучей, 1000 точек)
 * Ожидается: SINGLE BATCH (полная обработка без разбиения на батчи)
 */
void test_process_new_small();

/**
 * @brief Тест 7: ProcessNew() с большими данными (256 лучей, 1300000 точек)
 * Ожидается: MULTI-BATCH (batch processing из-за ограничений памяти)
 */
void test_process_new_large();

/**
 * @brief Запуск всех тестов
 */
void run_all_tests();

} // namespace test_antenna_fft_proc_max

