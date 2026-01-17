#include "Test/test_antenna_fft_proc_max.hpp"
#include "fft/antenna_fft_proc_max.h"
#include "generator/generator_gpu_new.h"
#include "interface/lfm_parameters.h"
#include "GPU/opencl_compute_engine.hpp"
#include <iostream>
#include <iomanip>

namespace test_antenna_fft_proc_max {

void test_basic_with_generator() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  Test 1: Basic Test with GeneratorGPU::signal_sinusoids\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
    
    try {
        // Инициализация OpenCL
        if (!gpu::OpenCLComputeEngine::IsInitialized()) {
            gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
        }
        
        // Параметры генератора: 5 лучей, 1000 точек
        SinusoidGenParams gen_params(5, 1000);
        RaySinusoidMap empty_map; // Пустой map - дефолтные параметры
        
        // Создать генератор
        LFMParameters lfm_params;
        lfm_params.num_beams = 5;
        lfm_params.count_points = 1000;
        lfm_params.sample_rate = 1.0e6f;
        
        radar::GeneratorGPU gen(lfm_params);
        
        // Генерировать сигналы
        std::cout << "Generating signals with GeneratorGPU::signal_sinusoids...\n";
        cl_mem signal_gpu = gen.signal_sinusoids(gen_params, empty_map);
        std::cout << "Signals generated successfully.\n\n";
        
        // Параметры для FFT обработки
        antenna_fft::AntennaFFTParams fft_params(
            5,      // beam_count
            1000,   // count_points
            512,    // out_count_points_fft
            3,      // max_peaks_count
            "test_task_1",
            "test_module"
        );
        
        // Создать процессор FFT
        antenna_fft::AntennaFFTProcMax processor(fft_params);
        
        std::cout << "nFFT calculated: " << processor.GetNFFT() << "\n\n";
        
        // Обработать сигналы
        std::cout << "Processing FFT...\n";
        antenna_fft::AntennaFFTResult result = processor.Process(signal_gpu);
        
        // Вывести результаты
        processor.PrintResults(result);
        
        // Вывести статистику профилирования
        std::cout << processor.GetProfilingStats() << "\n";
        
        // Сохранить результаты в файл
        processor.SaveResultsToFile(result, "test_basic_result.md");
        std::cout << "Results saved to Reports/test_basic_result.md\n";
        
        std::cout << "\n✅ Test 1 passed!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test 1 failed: " << e.what() << "\n";
        throw;
    }
}

void test_nfft_calculation() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  Test 2: nFFT Calculation\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
    
    try {
        antenna_fft::AntennaFFTParams params(5, 1000, 512, 3);
        antenna_fft::AntennaFFTProcMax processor(params);
        
        size_t nFFT = processor.GetNFFT();
        std::cout << "count_points: 1000\n";
        std::cout << "nFFT: " << nFFT << "\n";
        std::cout << "Expected: 2048 (1024 * 2, где 1024 - ближайшая степень 2 для 1000)\n";
        
        if (nFFT == 2048) {
            std::cout << "✅ nFFT calculation correct!\n";
        } else {
            std::cout << "❌ nFFT calculation incorrect!\n";
            throw std::runtime_error("nFFT calculation failed");
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test 2 failed: " << e.what() << "\n";
        throw;
    }
}

void test_maxima_search() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  Test 3: Maxima Search\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
    
    // TODO: Реализовать тест поиска максимумов
    std::cout << "⚠️  Test 3 not yet implemented\n";
}

void test_profiling() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  Test 4: Profiling\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
    
    // TODO: Реализовать тест профилирования
    std::cout << "⚠️  Test 4 not yet implemented\n";
}

void test_output() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  Test 5: Output\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
    
    // TODO: Реализовать тест вывода
    std::cout << "⚠️  Test 5 not yet implemented\n";
}

void run_all_tests() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     AntennaFFTProcMax Test Suite                         ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    
    try {
        test_basic_with_generator();
        test_nfft_calculation();
        test_maxima_search();
        test_profiling();
        test_output();
        
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
        std::cout << "║     All Tests Completed                                  ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test suite failed: " << e.what() << "\n";
        throw;
    }
}

} // namespace test_antenna_fft_proc_max

