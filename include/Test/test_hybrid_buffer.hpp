#pragma once

/**
 * @file test_hybrid_buffer.hpp
 * @brief Тест для гибридной системы памяти GPU (SVM/Regular)
 * 
 * Тестирует:
 * - Автоматический выбор стратегии
 * - Fallback на Regular если SVM недоступен
 * - Производительность разных стратегий
 * 
 * @author Codo (AI Assistant)
 * @date 2026-01-19
 */

#include "GPU/gpu_memory.hpp"
#include "GPU/opencl_compute_engine.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>

namespace test {

/**
 * @class HybridBufferTest
 * @brief Тест гибридной системы памяти
 */
class HybridBufferTest {
public:
    /**
     * @brief Запустить все тесты
     */
    static bool RunAll() {
        std::cout << "\n" << std::string(70, '═') << "\n";
        std::cout << "🧪 HYBRID BUFFER TESTS\n";
        std::cout << std::string(70, '═') << "\n\n";
        
        bool all_passed = true;
        
        all_passed &= TestSVMCapabilities();
        all_passed &= TestBufferFactory();
        all_passed &= TestAutoStrategy();
        all_passed &= TestDifferentSizes();
        all_passed &= TestReadWrite();
        
        std::cout << "\n" << std::string(70, '═') << "\n";
        if (all_passed) {
            std::cout << "✅ ALL TESTS PASSED!\n";
        } else {
            std::cout << "❌ SOME TESTS FAILED!\n";
        }
        std::cout << std::string(70, '═') << "\n\n";
        
        return all_passed;
    }
    
    /**
     * @brief Тест определения SVM capabilities
     */
    static bool TestSVMCapabilities() {
        std::cout << "📋 Test: SVM Capabilities Detection\n";
        std::cout << std::string(50, '-') << "\n";
        
        try {
            auto& engine = ManagerOpenCL::OpenCLComputeEngine::GetInstance();
            
            // Вывести SVM info
            std::cout << engine.GetSVMInfo();
            
            auto caps = engine.GetSVMCapabilities();
            std::cout << "\nRecommended strategy: " 
                      << ManagerOpenCL::MemoryStrategyToString(caps.GetBestSVMStrategy()) << "\n";
            
            std::cout << "✅ PASSED\n\n";
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ FAILED: " << e.what() << "\n\n";
            return false;
        }
    }
    
    /**
     * @brief Тест BufferFactory
     */
    static bool TestBufferFactory() {
        std::cout << "📋 Test: BufferFactory Creation\n";
        std::cout << std::string(50, '-') << "\n";
        
        try {
            auto& engine = ManagerOpenCL::OpenCLComputeEngine::GetInstance();
            
            // Создать фабрику
            auto factory = engine.CreateBufferFactory();
            
            factory->PrintInfo();
            
            std::cout << "✅ PASSED\n\n";
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ FAILED: " << e.what() << "\n\n";
            return false;
        }
    }
    
    /**
     * @brief Тест автоматического выбора стратегии
     */
    static bool TestAutoStrategy() {
        std::cout << "📋 Test: Auto Strategy Selection\n";
        std::cout << std::string(50, '-') << "\n";
        
        try {
            auto& engine = ManagerOpenCL::OpenCLComputeEngine::GetInstance();
            auto factory = engine.CreateBufferFactory(ManagerOpenCL::BufferConfig::Default());
            
            // Тест для разных размеров
            std::vector<size_t> sizes = {
                1024,                    // 8 KB (small)
                128 * 1024,              // 1 MB (medium)
                1024 * 1024,             // 8 MB (large)
                16 * 1024 * 1024         // 128 MB (very large)
            };
            
            for (size_t num_elements : sizes) {
                size_t size_bytes = num_elements * sizeof(ManagerOpenCL::ComplexFloat);
                auto strategy = factory->DetermineStrategy(size_bytes);
                
                std::cout << std::setw(12) << num_elements << " elements ("
                          << std::fixed << std::setprecision(2)
                          << (size_bytes / (1024.0 * 1024.0)) << " MB) -> "
                          << ManagerOpenCL::MemoryStrategyToString(strategy) << "\n";
            }
            
            std::cout << "✅ PASSED\n\n";
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ FAILED: " << e.what() << "\n\n";
            return false;
        }
    }
    
    /**
     * @brief Тест создания буферов разных размеров
     */
    static bool TestDifferentSizes() {
        std::cout << "📋 Test: Different Buffer Sizes\n";
        std::cout << std::string(50, '-') << "\n";
        
        try {
            auto& engine = ManagerOpenCL::OpenCLComputeEngine::GetInstance();
            auto factory = engine.CreateBufferFactory();
            
            // Создать буферы разных размеров
            auto small_buffer = factory->Create(1024);  // 8 KB
            auto medium_buffer = factory->Create(128 * 1024);  // 1 MB
            auto large_buffer = factory->Create(1024 * 1024);  // 8 MB
            
            std::cout << "Small:  " << ManagerOpenCL::GetBufferDescription(small_buffer.get()) << "\n";
            std::cout << "Medium: " << ManagerOpenCL::GetBufferDescription(medium_buffer.get()) << "\n";
            std::cout << "Large:  " << ManagerOpenCL::GetBufferDescription(large_buffer.get()) << "\n";
            
            std::cout << factory->GetStatistics();
            
            std::cout << "✅ PASSED\n\n";
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ FAILED: " << e.what() << "\n\n";
            return false;
        }
    }
    
    /**
     * @brief Тест чтения/записи данных
     */
    static bool TestReadWrite() {
        std::cout << "📋 Test: Read/Write Operations\n";
        std::cout << std::string(50, '-') << "\n";
        
        try {
            auto& engine = ManagerOpenCL::OpenCLComputeEngine::GetInstance();
            auto factory = engine.CreateBufferFactory();
            
            const size_t NUM_ELEMENTS = 10000;
            
            // Создать тестовые данные
            ManagerOpenCL::ComplexVector input_data(NUM_ELEMENTS);
            for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
                float angle = static_cast<float>(i) * 0.01f;
                input_data[i] = ManagerOpenCL::ComplexFloat(std::cos(angle), std::sin(angle));
            }
            
            // Создать буфер
            auto buffer = factory->Create(NUM_ELEMENTS);
            
            std::cout << "Buffer: " << ManagerOpenCL::GetBufferDescription(buffer.get()) << "\n";
            
            // Записать данные
            auto start_write = std::chrono::high_resolution_clock::now();
            buffer->Write(input_data);
            auto end_write = std::chrono::high_resolution_clock::now();
            
            // Прочитать данные
            auto start_read = std::chrono::high_resolution_clock::now();
            auto output_data = buffer->Read();
            auto end_read = std::chrono::high_resolution_clock::now();
            
            // Проверить данные
            bool data_correct = true;
            for (size_t i = 0; i < NUM_ELEMENTS && data_correct; ++i) {
                if (std::abs(input_data[i] - output_data[i]) > 1e-6f) {
                    data_correct = false;
                    std::cout << "Mismatch at index " << i << ": "
                              << input_data[i] << " vs " << output_data[i] << "\n";
                }
            }
            
            // Время
            auto write_time = std::chrono::duration<double, std::milli>(end_write - start_write).count();
            auto read_time = std::chrono::duration<double, std::milli>(end_read - start_read).count();
            
            std::cout << "Write time: " << std::fixed << std::setprecision(3) << write_time << " ms\n";
            std::cout << "Read time:  " << std::fixed << std::setprecision(3) << read_time << " ms\n";
            std::cout << "Data correct: " << (data_correct ? "YES ✅" : "NO ❌") << "\n";
            
            if (data_correct) {
                std::cout << "✅ PASSED\n\n";
                return true;
            } else {
                std::cout << "❌ FAILED: Data mismatch\n\n";
                return false;
            }
            
        } catch (const std::exception& e) {
            std::cout << "❌ FAILED: " << e.what() << "\n\n";
            return false;
        }
    }
    
    /**
     * @brief Benchmark разных стратегий
     */
    static void RunBenchmark(size_t num_elements = 1024 * 1024) {
        std::cout << "\n" << std::string(70, '═') << "\n";
        std::cout << "📊 BENCHMARK: " << num_elements << " elements ("
                  << (num_elements * sizeof(ManagerOpenCL::ComplexFloat) / (1024.0 * 1024.0)) 
                  << " MB)\n";
        std::cout << std::string(70, '═') << "\n\n";
        
        auto& engine = ManagerOpenCL::OpenCLComputeEngine::GetInstance();
        
        // Тестовые данные
        ManagerOpenCL::ComplexVector data(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = ManagerOpenCL::ComplexFloat(static_cast<float>(i), 0.0f);
        }
        
        // Список стратегий для тестирования
        std::vector<std::pair<ManagerOpenCL::MemoryStrategy, std::string>> strategies = {
            {ManagerOpenCL::MemoryStrategy::REGULAR_BUFFER, "REGULAR"},
            {ManagerOpenCL::MemoryStrategy::SVM_COARSE_GRAIN, "SVM_COARSE"},
            {ManagerOpenCL::MemoryStrategy::SVM_FINE_GRAIN, "SVM_FINE"}
        };
        
        std::cout << std::left << std::setw(20) << "Strategy" 
                  << std::setw(15) << "Write (ms)"
                  << std::setw(15) << "Read (ms)"
                  << std::setw(15) << "Status" << "\n";
        std::cout << std::string(65, '-') << "\n";
        
        for (const auto& [strategy, name] : strategies) {
            try {
                auto buffer = engine.CreateBufferWithStrategy(
                    num_elements, strategy, ManagerOpenCL::MemoryType::GPU_READ_WRITE
                );
                
                // Benchmark write
                auto start_w = std::chrono::high_resolution_clock::now();
                buffer->Write(data);
                auto end_w = std::chrono::high_resolution_clock::now();
                double write_ms = std::chrono::duration<double, std::milli>(end_w - start_w).count();
                
                // Benchmark read
                auto start_r = std::chrono::high_resolution_clock::now();
                auto result = buffer->Read();
                auto end_r = std::chrono::high_resolution_clock::now();
                double read_ms = std::chrono::duration<double, std::milli>(end_r - start_r).count();
                
                std::cout << std::left << std::setw(20) << name
                          << std::setw(15) << std::fixed << std::setprecision(3) << write_ms
                          << std::setw(15) << std::fixed << std::setprecision(3) << read_ms
                          << std::setw(15) << "✅" << "\n";
                          
            } catch (const std::exception& e) {
                std::cout << std::left << std::setw(20) << name
                          << std::setw(15) << "-"
                          << std::setw(15) << "-"
                          << std::setw(15) << "❌ (N/A)" << "\n";
            }
        }
        
        std::cout << "\n";
    }
};

/**
 * @brief Запустить тесты гибридной памяти
 */
inline bool RunHybridBufferTests() {
    return HybridBufferTest::RunAll();
}

/**
 * @brief Запустить benchmark
 */
inline void RunHybridBufferBenchmark(size_t num_elements = 1024 * 1024) {
    HybridBufferTest::RunBenchmark(num_elements);
}

} // namespace test

