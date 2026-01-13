#include "opencl_manager.h"
#include "generator_gpu_refactored.h"
#include "lfm_signal_generator.h"

#include <iostream>
#include <chrono>
#include <iomanip>

/**
 * @file example_opencl_singleton.cpp
 * @brief Examples of OpenCL Singleton Manager usage
 * 
 * DEMONSTRATES:
 * 1. One-time OpenCLManager initialization
 * 2. Creating multiple GeneratorGPU (reuse context)
 * 3. Program cache in action
 * 4. Performance improvements
 * 5. Error handling
 */

namespace radar {
namespace gpu {

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 1: Basic Initialization
// ═══════════════════════════════════════════════════════════════════

void Example1_BasicInitialization() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "EXAMPLE 1: Basic Initialization" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    try {
        // Initialize OpenCL once
        std::cout << "Initializing OpenCL Manager..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Initialization completed in " << duration.count() << " ms\n" << std::endl;
        
        // Print device information
        auto& manager = OpenCLManager::GetInstance();
        std::cout << manager.GetDeviceInfo() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 2: Multiple GeneratorGPU Objects
// ═══════════════════════════════════════════════════════════════════

void Example2_MultipleGenerators() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "EXAMPLE 2: Multiple GeneratorGPU Objects" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    try {
        // Create LFM parameters
        radar::LFMParameters params;
        params.f_start = 100.0f;
        params.f_stop = 500.0f;
        params.sample_rate = 12.0e6f;
        params.duration = 0.001f;
        params.num_beams = 256;
        
        // Create multiple GeneratorGPU objects
        std::cout << "Creating 3 GeneratorGPU objects..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        GeneratorGPU gen1(params);
        std::cout << "  ✓ GeneratorGPU #1 created (context: " << gen1.GetContext() << ")" << std::endl;
        
        GeneratorGPU gen2(params);
        std::cout << "  ✓ GeneratorGPU #2 created (context: " << gen2.GetContext() << ")" << std::endl;
        
        GeneratorGPU gen3(params);
        std::cout << "  ✓ GeneratorGPU #3 created (context: " << gen3.GetContext() << ")" << std::endl;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\nCreation completed in " << duration.count() << " ms" << std::endl;
        std::cout << "All objects share the SAME context (0x" 
                  << std::hex << (uintptr_t)gen1.GetContext() 
                  << std::dec << ")" << std::endl;
        std::cout << "\nINFO: Same context means no duplication of OpenCL resources!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 3: Program Cache Demonstration
// ═══════════════════════════════════════════════════════════════════

void Example3_ProgramCache() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "EXAMPLE 3: Program Cache Demonstration" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    try {
        radar::LFMParameters params;
        params.f_start = 100.0f;
        params.f_stop = 500.0f;
        params.sample_rate = 12.0e6f;
        params.duration = 0.001f;
        params.num_beams = 256;
        
        std::cout << "Creating GeneratorGPU objects with IDENTICAL kernels..." << std::endl;
        std::cout << "Cache should optimize subsequent compilations\n" << std::endl;
        
        // First object - compiles
        auto start1 = std::chrono::high_resolution_clock::now();
        GeneratorGPU gen1(params);
        auto end1 = std::chrono::high_resolution_clock::now();
        auto dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
        
        std::cout << "GeneratorGPU #1: " << dur1.count() << " ms (compilation + cache)" << std::endl;
        
        // Second object - cache hit!
        auto start2 = std::chrono::high_resolution_clock::now();
        GeneratorGPU gen2(params);
        auto end2 = std::chrono::high_resolution_clock::now();
        auto dur2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
        
        std::cout << "GeneratorGPU #2: " << dur2.count() << " ms (cache hit!) ✓" << std::endl;
        
        // Third object - cache hit!
        auto start3 = std::chrono::high_resolution_clock::now();
        GeneratorGPU gen3(params);
        auto end3 = std::chrono::high_resolution_clock::now();
        auto dur3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
        
        std::cout << "GeneratorGPU #3: " << dur3.count() << " ms (cache hit!) ✓" << std::endl;
        
        // Statistics
        std::cout << "\nCache Statistics:" << std::endl;
        auto& manager = OpenCLManager::GetInstance();
        std::cout << manager.GetCacheStatistics() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 4: Signal Generation
// ═══════════════════════════════════════════════════════════════════

void Example4_SignalGeneration() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "EXAMPLE 4: Signal Generation on GPU" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    try {
        radar::LFMParameters params;
        params.f_start = 100.0f;
        params.f_stop = 500.0f;
        params.sample_rate = 12.0e6f;
        params.duration = 0.001f;
        params.num_beams = 256;
        
        GeneratorGPU gen(params);
        
        std::cout << "Generating base LFM signal on GPU..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        cl_mem signal = gen.signal_base();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\nSignal generated in " << duration.count() << " ms" << std::endl;
        std::cout << "Signal GPU memory address: 0x" << std::hex << (uintptr_t)signal << std::dec << std::endl;
        std::cout << "Number of beams: " << gen.GetNumBeams() << std::endl;
        std::cout << "Number of samples per beam: " << gen.GetNumSamples() << std::endl;
        std::cout << "Total size: " << gen.GetTotalSize() << " complex samples" << std::endl;
        std::cout << "GPU memory used: " << gen.GetMemorySizeBytes() / (1024*1024) << " MB" << std::endl;
        
        // Cleanup
        clReleaseMemObject(signal);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 5: Error Handling
// ═══════════════════════════════════════════════════════════════════

void Example5_ErrorHandling() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "EXAMPLE 5: Error Handling Patterns" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    // Error 1: Using GeneratorGPU before initializing Manager
    std::cout << "Test 1: Create GeneratorGPU before Manager initialization" << std::endl;
    try {
        // Don't initialize Manager
        radar::LFMParameters params;
        params.f_start = 100.0f;
        params.f_stop = 500.0f;
        params.sample_rate = 12.0e6f;
        params.duration = 0.001f;
        params.num_beams = 256;
        
        // This should throw an exception
        GeneratorGPU gen(params);
    }
    catch (const std::runtime_error& e) {
        std::cout << "  ✓ Caught expected error: " << e.what() << "\n" << std::endl;
    }
    
    // Error 2: Invalid LFM parameters
    std::cout << "Test 2: Create GeneratorGPU with invalid parameters" << std::endl;
    try {
        OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
        
        radar::LFMParameters params;
        params.f_start = 500.0f;
        params.f_stop = 100.0f;  // Stop < Start - INVALID!
        params.sample_rate = 12.0e6f;
        params.duration = 0.001f;
        params.num_beams = 256;
        
        GeneratorGPU gen(params);
    }
    catch (const std::invalid_argument& e) {
        std::cout << "  ✓ Caught expected error: " << e.what() << "\n" << std::endl;
    }
    
    std::cout << "Error handling tests completed!" << std::endl;
}

} // namespace gpu
} // namespace radar

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

int main() {
    std::cout << R"(
╔════════════════════════════════════════════════════════════════╗
║    OpenCL Singleton Manager - Complete Examples               ║
║    Demonstration of OpenCLManager with GeneratorGPU           ║
╚════════════════════════════════════════════════════════════════╝
)" << std::endl;
    
    try {
        // Example 1: Basic initialization
        radar::gpu::Example1_BasicInitialization();
        
        // Example 2: Multiple objects
        radar::gpu::Example2_MultipleGenerators();
        
        // Example 3: Program cache
        radar::gpu::Example3_ProgramCache();
        
        // Example 4: Signal generation
        radar::gpu::Example4_SignalGeneration();
        
        // Example 5: Error handling
        radar::gpu::Example5_ErrorHandling();
        
        // Cleanup (optional)
        radar::gpu::OpenCLManager::Cleanup();
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "All examples completed successfully!" << std::endl;
        std::cout << std::string(60, '=') << "\n" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
