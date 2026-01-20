#include <iostream>
#include <string>

#include <CL/cl.h>
#include "GPU/opencl_compute_engine.hpp"
#include "Test/example_usage.hpp"
//#include "Test/example_signal_basic_lfm.hpp"
//#include "Test/example_signal_delayed_lfm.hpp"
//#include "Test/example_signal_combined_delays.hpp"
//#include "Test/test_signal_sinusoids.hpp"
//#include "Test/test_antenna_fft_proc_max.hpp"
#include "Farrow/lagrange_matrix_loader.hpp"


int main(int argc, char* argv[]) {


    LagrangeMatrixLoader loader;

    // Load matrix from JSON file
    if (!loader.loadFromJSON("lagrange_matrix.json")) {
        return 1;
    }

    // Display info
    loader.printMatrixInfo();
    loader.printMatrixSample(8);

    // Access specific element example
    try {
        std::cout << "\nElement [0][1] = " << loader.getElement(0, 1) << std::endl;
        std::cout << "Element [8][0] = " << loader.getElement(8, 0) << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }


  return 0;
}

/**
 * 
     // Запуск тестов Antenna FFT
//   test_antenna_fft_proc_max::run_all_tests();
* 
 * 
 * !!! ВАЖНО !!!
 * Примеры использования GeneratorGPU с signal_sinusoids 
 * 
    gpu::OpenCLComputeEngine::Initialize(gpu::DeviceType::GPU);
    auto& engine = gpu::OpenCLComputeEngine::GetInstance();

    // Показать информацию о девайсе
    std::cout << engine.GetDeviceInfo();
    test_signal_sinusoids::run_all_tests();

    // ... код программы ...

    // Очистка (опционально, вызывается в деструкторе)
    gpu::OpenCLComputeEngine::Cleanup();

    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        GeneratorGPU Examples (NEW ARCHITECTURE)                  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n" << std::endl;
 * 
 * примеры применения GPU OpenCL
    // Запустить примеры
    example_basic_lfm();
    example_delayed_lfm();
    example_multiple_generators();

    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    ALL EXAMPLES COMPLETED                        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n" << std::endl;

    example_02::test001();
    example_02::test002();
    example_02::test003();
    example_02::test004();
 * 
 * 
 * 
 int inicial_opencl_manager(){
  try {
    gpu::OpenCLManager::Initialize(CL_DEVICE_TYPE_GPU);
    auto& opencl_ = gpu::OpenCLManager::GetInstance();
      
    std::cout << "✅ OpenCL Manager инициализирован\n";
    std::cout << opencl_.GetDeviceInfo() << "\n";
      
  } catch (const std::exception& e) {
    std::cerr << "❌ Ошибка инициализации OpenCL: " << e.what() << std::endl;
    return 1;
  }          
   // Запуск тестов для signal_sinusoids
   test_signal_sinusoids::run_all_tests();

  return 0;
}

LFMParameters inicial_params() {
  // ═══════════════════════════════════════════════════════════════
  // 1. ИНИЦИАЛИЗАЦИЯ ПАРАМЕТРОВ ЛЧМ
  // ═══════════════════════════════════════════════════════════════
          
  LFMParameters params;
  params.f_start = 100.0f;           // 100 Гц
  params.f_stop = 500.0f;            // 500 Гц
  params.sample_rate = 12.0e6f;      // 12 МГц
  params.duration = 0.01f;            // 0.1 сек
  params.num_beams = 256;            // 256 лучей
  params.steering_angle = 0.5f;     // 30 градусов
          
  // Вычислить количество отсчётов
  size_t num_samples = static_cast<size_t>(params.duration * params.sample_rate);
          
  std::cout << "📋 ПАРАМЕТРЫ ЛЧМ СИГНАЛА:\n"
    << "  • Частота начальная: " << params.f_start << " Гц\n"
    << "  • Частота конечная: " << params.f_stop << " Гц\n"
    << "  • Частота дискретизации: " << params.sample_rate / 1e6f << " МГц\n"
    << "  • Длительность: " << params.duration << " сек\n"
    << "  • Количество лучей: " << params.num_beams << "\n"
    << "  • Количество отсчётов на луч: " << num_samples << "\n"
    << "  • Всего элементов: " << params.num_beams * num_samples << "\n"
    << "  • Память на GPU: " << (params.num_beams * num_samples * sizeof(std::complex<float>)) / (1024*1024)
    << " MB\n\n";
  return params;    
}
* 
 * 
 * 
 ******************* 
   std::cout << "═══════════════════════════════════════════════════════════\n"
            << "GeneratorGPU - Параллельная генерация ЛЧМ на GPU\n"
            << "═══════════════════════════════════════════════════════════\n\n";
  // Инициализация OpenCL
  if(inicial_opencl_manager())
    return -1;

  LFMParameters params_;
  
  try {      
    params_ = inicial_params();

    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << std::endl;
        return 1;
    }
    auto t_generator_ = std::make_shared<test::generator>(params_);
    //auto gen_gpu_ = t_generator_-> inicial_genegstor(params_);
    cl_mem signal_base_ = t_generator_->gen_base_signal();
    cl_mem signal_delay_ = t_generator_-> gen_signal_delay();    
    t_generator_->gpu_to_cpu(signal_base_);
//    t_generator_->gpu_to_cpu(signal_delay_);

    gpu::GPUMemoryManager::Initialize();
    examples::RunAllExamples();


 
 auto mem_bufer_ = std::make_shared<test::gpu_mem_buffer>(t_generator_->GetGenratorGPU());

    mem_bufer_-> Example1_FullTransfer(t_generator_->mem_gen ); 
    mem_bufer_-> Example2_PartialRead(t_generator_->mem_gen);
    mem_bufer_-> Example3_Bidirectional();
    mem_bufer_-> Example4_BufferPool();
    mem_bufer_-> Example5_ReplacementForGpuToCpu(t_generator_->mem_gen );
* 
 * 
 */
