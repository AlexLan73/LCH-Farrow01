#include <iostream>
#include <exception>
#include <complex>
#include <memory>

#include <CL/cl.h>
#include "GPU/opencl_manager.h"
#include "interface/lfm_parameters.h"
#include "generator/generator_gpu.h"
#include "Test/t_generator.hpp"

//opencl_manager.cpp 
// lfm_parameters.h уже включен в generator_gpu.h

LFMParameters inicial_params(){
  // ═══════════════════════════════════════════════════════════════
  // 1. ИНИЦИАЛИЗАЦИЯ ПАРАМЕТРОВ ЛЧМ
  // ═══════════════════════════════════════════════════════════════
        
  LFMParameters params;
  params.f_start = 100.0f;           // 100 Гц
  params.f_stop = 500.0f;            // 500 Гц
  params.sample_rate = 12.0e6f;      // 12 МГц
  params.duration = 0.01f;            // 0.1 сек
  params.num_beams = 256;            // 256 лучей
  params.steering_angle = 30.0f;     // 30 градусов
        
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
  return 0;
}

std::shared_ptr<radar::GeneratorGPU> inicial_genegstor(const LFMParameters& params){
  // ═══════════════════════════════════════════════════════════════
  // 2. СОЗДАТЬ ГЕНЕРАТОР GPU
  // ═══════════════════════════════════════════════════════════════
        
  std::cout << "⚙️  Инициализация GPU...\n";
  auto time_start = std::chrono::high_resolution_clock::now();
        
  std::shared_ptr generator_gpu_ = std::make_shared<radar::GeneratorGPU>(params);
//  radar::GeneratorGPU gen(params);
        
  auto time_init = std::chrono::high_resolution_clock::now();
  double init_time = std::chrono::duration<double, std::milli>(time_init - time_start).count();
  std::cout << "✓ GPU инициализирована за " << init_time << " мс\n\n";
  return generator_gpu_;
}

cl_mem gen_base_signal(std::shared_ptr<radar::GeneratorGPU>& gen_gpu){
  // ═══════════════════════════════════════════════════════════════
  // 3. ГЕНЕРИРОВАТЬ БАЗОВЫЙ ЛЧМ СИГНАЛ
  // ═══════════════════════════════════════════════════════════════
        
  std::cout << "📡 Генерация БАЗОВОГО ЛЧМ сигнала на GPU...\n";
  auto time_gen_base = std::chrono::high_resolution_clock::now();
        
  cl_mem signal_base = gen_gpu->signal_base();
        
  auto time_gen_base_end = std::chrono::high_resolution_clock::now();
  double gen_base_time = std::chrono::duration<double, std::milli>(time_gen_base_end - time_gen_base).count();
  std::cout << "✓ signal_base() завершена за " << gen_base_time << " мс\n\n";
  return signal_base;
}

cl_mem gen_signal_delay(std::shared_ptr<radar::GeneratorGPU>& gen_gpu){
  // ═══════════════════════════════════════════════════════════════
  // 4. ПОДГОТОВИТЬ ПАРАМЕТРЫ ЗАДЕРЖКИ
  // ═══════════════════════════════════════════════════════════════
        
  std::cout << "📊 Подготовка параметров задержки для " << gen_gpu->GetNumBeams() << " лучей...\n"; //  params.num_beams
        
  std::vector<DelayParameter> m_delay(gen_gpu->GetNumBeams()); // params.num_beams
  gen_gpu->SetParametersAngle();
  float angl_start_ = gen_gpu->GetAngleStart(); 
  for (size_t beam = 0; beam < gen_gpu->GetNumBeams(); ++beam) {    //  params.num_beams
    // Задержка = шаг 0.5° * номер луча
    // Например: луч 0 → 0°, луч 1 → 0.5°, луч 2 → 1.0°, ...
    m_delay[beam].beam_index = beam;
    m_delay[beam].delay_degrees = (beam * 0.5f-angl_start_);  // 0, 0.5, 1.0, 1.5, ...
  }
        
  std::cout << "  • m_delay[0] = {beam_id: " << m_delay[0].beam_index 
      << ", delay: " << m_delay[0].delay_degrees << "°}\n"
      << "  • m_delay[128] = {beam_id: " << m_delay[128].beam_index 
      << ", delay: " << m_delay[128].delay_degrees << "°}\n"
      << "  • m_delay[255] = {beam_id: " << m_delay[255].beam_index 
      << ", delay: " << m_delay[255].delay_degrees << "°}\n\n";
        
  // ═══════════════════════════════════════════════════════════════
  // 5. ГЕНЕРИРОВАТЬ ЛЧМ С ДРОБНОЙ ЗАДЕРЖКОЙ
  // ═══════════════════════════════════════════════════════════════
        
  std::cout << "📡 Генерация ЛЧМ с ДРОБНОЙ ЗАДЕРЖКОЙ на GPU...\n";
  auto time_gen_delayed = std::chrono::high_resolution_clock::now();
        
  cl_mem signal_delayed = gen_gpu->signal_valedation(m_delay.data(), m_delay.size());
        
  auto time_gen_delayed_end = std::chrono::high_resolution_clock::now();
  double gen_delayed_time = std::chrono::duration<double, std::milli>(time_gen_delayed_end - time_gen_delayed).count();
  std::cout << "✓ signal_valedation() завершена за " << gen_delayed_time << " мс\n\n";
  return signal_delayed;
}

void gpu_to_cpu(std::shared_ptr<radar::GeneratorGPU>& gen_gpu, const cl_mem& signal_){
  // ═══════════════════════════════════════════════════════════════
  // 6. ТРАНСФЕР ДАННЫХ GPU → CPU (для проверки)
  // ═══════════════════════════════════════════════════════════════
        
  std::cout << "📤 Трансфер данных GPU → CPU (первый луч, первые 10 отсчётов)...\n";
        
  size_t read_samples = std::min(size_t(10), gen_gpu->GetNumSamples());  // Прочитать первые 10
  std::vector<std::complex<float>> cpu_data(read_samples);
        
  cl_int err = clEnqueueReadBuffer(
    gen_gpu->GetQueue(),
    signal_,
    CL_TRUE,  // Blocking read
    0,        // Offset
    read_samples * sizeof(std::complex<float>),
    cpu_data.data(),
    0, nullptr, nullptr
  );

  if (err == CL_SUCCESS) {
    std::cout << "  ✓ Первый луч, первые " << read_samples << " отсчётов signal_base:\n";
    for (size_t i = 0; i < read_samples; ++i) {
      std::cout << "    [" << i << "] = " << cpu_data[i].real() 
          << " + " << cpu_data[i].imag() << "j\n";
    }
  } else {
    std::cout << "  ⚠️  Ошибка при чтении из GPU (код: " << err << ")\n";
  }
  std::cout << "\n";
}


int main() {
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
    auto t_generator = std::make_shared<test::generator>();
    auto gen_gpu_ = inicial_genegstor(params_);
    cl_mem signal_base_ = gen_base_signal(gen_gpu_);
    cl_mem signal_delay_ =  gen_signal_delay(gen_gpu_);    
    gpu_to_cpu(gen_gpu_, signal_base_);
    gpu_to_cpu(gen_gpu_, signal_delay_);
    return 0;
}

/**
        
        // ═══════════════════════════════════════════════════════════════
        // 7. ИТОГОВАЯ СТАТИСТИКА
        // ═══════════════════════════════════════════════════════════════
        
        auto time_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(time_end - time_start).count();
        
        std::cout << "═══════════════════════════════════════════════════════════\n"
                  << "✅ УСПЕШНО ЗАВЕРШЕНО\n"
                  << "═══════════════════════════════════════════════════════════\n"
                  << "⏱️  ВРЕМЯ ВЫПОЛНЕНИЯ:\n"
                  << "  • Инициализация GPU: " << init_time << " мс\n"
                  << "  • signal_base(): " << gen_base_time << " мс\n"
                  << "  • signal_valedation(): " << gen_delayed_time << " мс\n"
                  << "  • ИТОГО: " << total_time << " мс\n\n"
                  << "📊 ПРОПУСКНАЯ СПОСОБНОСТЬ GPU:\n"
                  << "  • signal_base(): " 
                  << (params.num_beams * num_samples / (gen_base_time / 1000.0) / 1e9) << " Гвыб/сек\n"
                  << "  • signal_valedation(): " 
                  << (params.num_beams * num_samples / (gen_delayed_time / 1000.0) / 1e9) << " Гвыб/сек\n";
        
        // ═══════════════════════════════════════════════════════════════
        // 8. ОЧИСТКА (АВТОМАТИЧЕСКИ в деструкторе ~GeneratorGPU)
        // ═══════════════════════════════════════════════════════════════
        
        std::cout << "🧹 Освобождение GPU ресурсов...\n";
        // signal_base и signal_delayed будут освобождены автоматически
        // когда выйдут из области видимости (RAII)
        
        std::cout << "✓ Программа завершена корректно!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА: " << e.what() << std::endl;
        return 1;
    }
  
  
 */