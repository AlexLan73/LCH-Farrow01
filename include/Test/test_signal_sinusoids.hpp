#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <CL/cl.h>
#include "generator/generator_gpu_new.h"
#include "interface/lfm_parameters.h"

/**
 * @brief Набор тестов для функции signal_sinusoids класса GeneratorGPU
 *
 * Тесты проверяют различные сценарии генерации сигналов как суммы синусоид.
 */
namespace test_signal_sinusoids {

/**
 * @brief Тест 1: Пустой map_ray - использование дефолтных параметров для всех лучей
 *
 * Ожидается: Все лучи генерируются с амплитудой 1.0, периодом count_points/2, фазой 0°
 */
void test_empty_map();

/**
 * @brief Тест 2: Один луч с одной синусоидой
 *
 * Ожидается: Только луч 0 генерируется с заданными параметрами, остальные - дефолтные
 */
void test_single_ray_single_sinusoid();

/**
 * @brief Тест 3: Несколько лучей с несколькими синусоидами
 *
 * Ожидается: Каждый луч генерируется как сумма своих синусоид
 */
void test_multiple_rays_multiple_sinusoids();

/**
 * @brief Тест 4: Луч вне диапазона
 *
 * Ожидается: Луч с индексом вне [0, num_rays-1] игнорируется, выводится предупреждение
 */
void test_ray_out_of_range();

/**
 * @brief Тест 5: Более 10 синусоид на луч
 *
 * Ожидается: Используются только первые 10 синусоид, выводится предупреждение
 */
void test_more_than_10_sinusoids();

/**
 * @brief Тест 6: Валидация параметров
 *
 * Ожидается: Исключения при некорректных параметрах
 */
void test_parameter_validation();

/**
 * @brief Запуск всех тестов
 */
void run_all_tests();

} // namespace test_signal_sinusoids