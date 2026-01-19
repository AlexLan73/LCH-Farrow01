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
        
        // ═══════════════════════════════════════════════════════════════════
        // ПАРАМЕТРЫ ТЕСТА
        // ═══════════════════════════════════════════════════════════════════
        const size_t NUM_BEAMS = 56;
        const size_t COUNT_POINTS = 130000;
        const size_t OUT_COUNT_POINTS_FFT = 100;
        const size_t MAX_PEAKS_COUNT = 3;
        
        std::cout << "  ┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "  │  ПАРАМЕТРЫ ТЕСТА                                            │\n";
        std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";
        printf("  │  num_beams (лучей)           │  %10zu  │\n", NUM_BEAMS);
        printf("  │  count_points (точек/луч)    │  %10zu  │\n", COUNT_POINTS);
        printf("  │  out_count_points_fft        │  %10zu  │\n", OUT_COUNT_POINTS_FFT);
        printf("  │  max_peaks_count             │  %10zu  │\n", MAX_PEAKS_COUNT);
        std::cout << "\n";
        
        // Параметры генератора
        SinusoidGenParams gen_params(NUM_BEAMS, COUNT_POINTS);
        RaySinusoidMap empty_map;
        
        // Создать генератор
        LFMParameters lfm_params;
        lfm_params.num_beams = NUM_BEAMS;
        lfm_params.count_points = COUNT_POINTS;
        lfm_params.sample_rate = 1.0e6f;
        
        radar::GeneratorGPU gen(lfm_params);
        
        // Генерировать сигналы
        std::cout << "Generating signals with GeneratorGPU::signal_sinusoids...\n";
        cl_mem signal_gpu = gen.signal_sinusoids(gen_params, empty_map);
        std::cout << "Signals generated successfully.\n\n";
        
        // Параметры для FFT обработки
        antenna_fft::AntennaFFTParams fft_params(
            NUM_BEAMS,
            COUNT_POINTS,
            OUT_COUNT_POINTS_FFT,
            MAX_PEAKS_COUNT,
            "test_task_1",
            "test_module"
        );
        
        // Создать процессор FFT
        antenna_fft::AntennaFFTProcMax processor(fft_params);
        
        size_t nFFT = processor.GetNFFT();
        size_t total_input_points = NUM_BEAMS * COUNT_POINTS;
        size_t total_fft_points = NUM_BEAMS * nFFT;
        size_t input_size_mb = total_input_points * sizeof(std::complex<float>) / (1024 * 1024);
        size_t fft_size_mb = total_fft_points * sizeof(std::complex<float>) / (1024 * 1024);
        
        std::cout << "  ┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "  │  FFT РАЗМЕРЫ                                                │\n";
        std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";
        printf("  │  nFFT (размер FFT/луч)       │  %10zu  │ (ближайшая степень 2 × 2)\n", nFFT);
        printf("  │  Входных точек (всего)       │  %10zu  │ (%zu лучей × %zu точек)\n", 
               total_input_points, NUM_BEAMS, COUNT_POINTS);
        printf("  │  FFT точек (всего)           │  %10zu  │ (%zu лучей × %zu nFFT)\n", 
               total_fft_points, NUM_BEAMS, nFFT);
        printf("  │  Входные данные              │  %10zu MB │\n", input_size_mb);
        printf("  │  FFT буферы                  │  %10zu MB │\n", fft_size_mb);
        std::cout << "\n";
        
        // Обработать сигналы
        std::cout << "Processing FFT...\n";
        antenna_fft::AntennaFFTResult result = processor.Process(signal_gpu);
        
        // Вывести результаты
        processor.PrintResults(result);
        
        // Вывести статистику профилирования
        std::cout << processor.GetProfilingStats() << "\n";
        
        // Сохранить результаты в файл
        processor.SaveResultsToFile(result, "antenna_result.md");
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

