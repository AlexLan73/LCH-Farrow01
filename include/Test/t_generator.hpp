
#pragma once
#include <iostream>
#include <exception>
#include <complex>
#include <memory>

#include <CL/cl.h>
#include "GPU/opencl_manager.h"
#include "interface/lfm_parameters.h"
#include "generator/generator_gpu.h"
#include "Test/t_generator.hpp"

namespace test{
  class generator
  {
    private:
        const LFMParameters params_;
    public:

      generator(const LFMParameters& params);
      ~generator();
      cl_mem mem_gen;
      cl_mem mem_gen_delay;
      
  };

  generator::generator(const LFMParameters& params): params_(params)
  {

  }

  generator::~generator()
  {
  }
    
}