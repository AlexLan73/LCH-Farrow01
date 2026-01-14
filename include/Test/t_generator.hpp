
#pragma once
#include <iostream>
#include <complex>
#include <memory>
#include <chrono>

#include <CL/cl.h>
#include "interface/lfm_parameters.h"
#include "generator/generator_gpu.h"

namespace test{
  class generator
  {
    private:
      const LFMParameters params_;
      std::shared_ptr<radar::GeneratorGPU> gen_gpu_;       
    public:

      generator(const LFMParameters& params);
      ~generator();

//      LFMParameters inicial_params();      
//      int inicial_opencl_manager();      
      std::shared_ptr<radar::GeneratorGPU> inicial_genegstor(const LFMParameters& params);
      cl_mem gen_base_signal();
      cl_mem gen_signal_delay();
      void gpu_to_cpu(const cl_mem& signal_);
      std::shared_ptr<radar::GeneratorGPU> GetGenratorGPU(){ return gen_gpu_; }

      cl_mem mem_gen;
      cl_mem mem_gen_delay;
      
  };

  /// @brief конструктор получает параметры генератора для инициализации
  /// @param params <- LFMParameters
  inline generator::generator(const LFMParameters& params): params_(params)
  {
    gen_gpu_= inicial_genegstor(params);
  }

  inline generator::~generator()
  {

  }


  inline std::shared_ptr<radar::GeneratorGPU> generator::inicial_genegstor(const LFMParameters& params){
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

  inline cl_mem generator::gen_base_signal(){
    // ═══════════════════════════════════════════════════════════════
    // 3. ГЕНЕРИРОВАТЬ БАЗОВЫЙ ЛЧМ СИГНАЛ
    // ═══════════════════════════════════════════════════════════════
          
    std::cout << "📡 Генерация БАЗОВОГО ЛЧМ сигнала на GPU...\n";
    auto time_gen_base = std::chrono::high_resolution_clock::now();
          
    cl_mem signal_base = gen_gpu_->signal_base();
          
    auto time_gen_base_end = std::chrono::high_resolution_clock::now();
    double gen_base_time = std::chrono::duration<double, std::milli>(time_gen_base_end - time_gen_base).count();
    std::cout << "✓ signal_base() завершена за " << gen_base_time << " мс\n\n";
    mem_gen = signal_base;
    return signal_base;
  }

  inline cl_mem  generator::gen_signal_delay(){
    // ═══════════════════════════════════════════════════════════════
    // 4. ПОДГОТОВИТЬ ПАРАМЕТРЫ ЗАДЕРЖКИ
    // ═══════════════════════════════════════════════════════════════
          
    std::cout << "📊 Подготовка параметров задержки для " << gen_gpu_->GetNumBeams() << " лучей...\n"; //  params.num_beams
          
    std::vector<DelayParameter> m_delay(gen_gpu_->GetNumBeams()); // params.num_beams
    gen_gpu_->SetParametersAngle();
    float angl_start_ = gen_gpu_->GetAngleStart(); 
    for (size_t beam = 0; beam < gen_gpu_->GetNumBeams(); ++beam) {    //  params.num_beams
      // Задержка = шаг 0.5° * номер луча
      // Например: луч 0 → 0°, луч 1 → 0.5°, луч 2 → 1.0°, ...
      m_delay[beam].beam_index = beam;
      m_delay[beam].delay_degrees = (beam * 0.5f+angl_start_);  // 0, 0.5, 1.0, 1.5, ...
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
        
  cl_mem signal_delayed = gen_gpu_->signal_valedation(m_delay.data(), m_delay.size());
        
  auto time_gen_delayed_end = std::chrono::high_resolution_clock::now();
  double gen_delayed_time = std::chrono::duration<double, std::milli>(time_gen_delayed_end - time_gen_delayed).count();
  std::cout << "✓ signal_valedation() завершена за " << gen_delayed_time << " мс\n\n";
  mem_gen_delay = signal_delayed;
  return signal_delayed;
}

  inline void generator::gpu_to_cpu(const cl_mem& signal_){
    // ═══════════════════════════════════════════════════════════════
    // 6. ТРАНСФЕР ДАННЫХ GPU → CPU (для проверки)
    // ═══════════════════════════════════════════════════════════════
          
    std::cout << "📤 Трансфер данных GPU → CPU (первый луч, первые 10 отсчётов)...\n";
          
    size_t read_samples = std::min(size_t(10), gen_gpu_->GetNumSamples());  // Прочитать первые 10
    std::vector<std::complex<float>> cpu_data(read_samples);
          
    cl_int err = clEnqueueReadBuffer(
      gen_gpu_->GetQueue(),
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
