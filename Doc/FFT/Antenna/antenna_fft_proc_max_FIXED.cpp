// ════════════════════════════════════════════════════════════════════════════
// ANTENNA_FFT_PROC_MAX - FIXED METHODS
// ════════════════════════════════════════════════════════════════════════════
//
// KEY CHANGES:
// 1. GetPostCallbackSource() - proper filtering to search_range only
// 2. CreateMaxReductionKernel() - complete phase calculation
// 3. FindMaximaAllBeamsOnGPU() - clear parameter passing
// 4. Struct PostCallbackUserData - renamed field for clarity
//
// USAGE: Copy these methods into your antenna_fft_proc_max.cpp file
// replacing the existing implementations
//

// ════════════════════════════════════════════════════════════════════════════
// METHOD 1: GetPostCallbackSource()
// ════════════════════════════════════════════════════════════════════════════

std::string AntennaFFTProcMax::GetPostCallbackSource() const {
    // User requirement:
    // - search_range parameter (e.g., 60)
    // - Search ONLY in first search_range/2 points AND last search_range/2 points
    // - Example: search_range=60 → search [0..29] and [nFFT-30..nFFT-1]
    
    return R"(
typedef struct {
    uint beam_count;
    uint nFFT;
    uint search_range;              // CHANGED: was out_count_points_fft
    uint max_peaks_count;
} PostCallbackUserData;

void processFFTPost(__global void* output, uint outoffset, __global void* userdata, float2 fftoutput) {
    __global PostCallbackUserData* params = (__global PostCallbackUserData*)userdata;
    
    uint beam_count = params->beam_count;
    uint nFFT = params->nFFT;
    uint search_range = params->search_range;        // e.g., 60
    uint half_search = search_range / 2;            // e.g., 30
    
    // Calculate beam index and position in FFT
    uint beam_idx = outoffset / nFFT;
    uint pos_in_fft = outoffset % nFFT;
    
    if (beam_idx >= beam_count) {
        return;
    }
    
    // ✅ FILTER: Check if position is in interesting range
    // Range 1: [0, half_search) - first half_search points
    // Range 2: [nFFT - half_search, nFFT) - last half_search points
    bool in_range1 = (pos_in_fft < half_search);
    bool in_range2 = (pos_in_fft >= nFFT - half_search);
    
    if (!in_range1 && !in_range2) {
        return;  // ✅ Fast return for 99.9% of threads!
    }
    
    // ✅ Calculate index in output buffer (0..search_range-1)
    uint output_idx;
    if (in_range1) {
        // First half_search points go to beginning
        output_idx = pos_in_fft;
    } else {
        // Last half_search points go after first half
        output_idx = half_search + (pos_in_fft - (nFFT - half_search));
    }
    
    // Layout userdata: params | complex_buffer | magnitude_buffer
    uint params_size = 16;  // sizeof(PostCallbackUserData)
    uint complex_offset = params_size;
    uint magnitude_offset = complex_offset + (beam_count * search_range * 8);  // 8 = sizeof(float2)
    
    __global float2* complex_buffer = (__global float2*)((__global char*)userdata + complex_offset);
    __global float* magnitude_buffer = (__global float*)((__global char*)userdata + magnitude_offset);
    
    // Calculate global index in buffer
    uint base_idx = beam_idx * search_range + output_idx;
    
    // ✅ Write complex spectrum (for future phase calculation)
    complex_buffer[base_idx] = fftoutput;
    
    // ✅ Write magnitude (for finding maxima)
    magnitude_buffer[base_idx] = length(fftoutput);
}
)";
}

// ════════════════════════════════════════════════════════════════════════════
// METHOD 2: CreateMaxReductionKernel()
// ════════════════════════════════════════════════════════════════════════════

void AntennaFFTProcMax::CreateMaxReductionKernel() {
    std::string reduction_kernel_source = R"(
typedef struct {
    uint index;
    float magnitude;
    float phase;
    uint pad;
} MaxValue;

// Find top-N maxima and calculate phase
// Kernel: one work-group per beam, parallel reduction on search_range points
__kernel void findMaximaAndPhase(
    __global const float2* complex_buffer,      // Complex spectrum (search_range points)
    __global const float* magnitude_buffer,     // Magnitude (search_range points)
    __global MaxValue* maxima_buffer,           // Output buffer for top-N
    uint beam_count,
    uint search_range,                          // Total points to search (e.g., 60)
    uint max_peaks_count                        // N = 5
) {
    uint beam_idx = get_group_id(0);
    uint tid = get_local_id(0);
    uint local_size = get_local_size(0);
    
    if (beam_idx >= beam_count) return;
    
    // Local memory for top-N maxima
    __local MaxValue local_max[8];
    __local float local_mag[256];
    __local uint local_idx[256];
    
    // ========================================================================
    // PHASE 1: Initialize top-N
    // ========================================================================
    if (tid < max_peaks_count) {
        local_max[tid].index = UINT_MAX;
        local_max[tid].magnitude = -1.0f;
        local_max[tid].phase = 0.0f;
        local_max[tid].pad = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // ========================================================================
    // PHASE 2: Load magnitude into local memory
    // ========================================================================
    uint base_offset = beam_idx * search_range;
    
    for (uint i = tid; i < search_range; i += local_size) {
        local_mag[i] = magnitude_buffer[base_offset + i];
        local_idx[i] = i;  // Original index in search_range array
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // ========================================================================
    // PHASE 3: Find top-N (first thread only)
    // ========================================================================
    if (tid == 0) {
        for (uint k = 0; k < max_peaks_count; ++k) {
            float max_mag = -1.0f;
            uint max_idx = UINT_MAX;
            
            // Find maximum among remaining elements
            for (uint i = 0; i < search_range; ++i) {
                if (local_mag[i] > max_mag) {
                    max_mag = local_mag[i];
                    max_idx = local_idx[i];
                }
            }
            
            if (max_idx != UINT_MAX && max_mag > 0.0f) {
                // Calculate phase in degrees
                float2 cval = complex_buffer[base_offset + max_idx];
                float phase_rad = atan2(cval.y, cval.x);          // radians
                float phase_deg = phase_rad * 57.29577951f;       // 180/π
                
                // Normalize phase to [-180, 180]
                if (phase_deg > 180.0f) phase_deg -= 360.0f;
                if (phase_deg < -180.0f) phase_deg += 360.0f;
                
                local_max[k].index = max_idx;
                local_max[k].magnitude = max_mag;
                local_max[k].phase = phase_deg;
                
                // Mark as used
                local_mag[max_idx] = -1.0f;
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // ========================================================================
    // PHASE 4: Write results to global memory
    // ========================================================================
    if (tid < max_peaks_count) {
        uint out_idx = beam_idx * max_peaks_count + tid;
        maxima_buffer[out_idx] = local_max[tid];
    }
}
)";
    
    reduction_program_ = engine_->LoadProgram(reduction_kernel_source);
    reduction_kernel_ = engine_->GetKernel(reduction_program_, "findMaximaAndPhase");
}

// ════════════════════════════════════════════════════════════════════════════
// METHOD 3: FindMaximaAllBeamsOnGPU()
// ════════════════════════════════════════════════════════════════════════════

std::vector> AntennaFFTProcMax::FindMaximaAllBeamsOnGPU() {
    if (!post_callback_userdata_) {
        throw std::runtime_error("post_callback_userdata_ is not initialized");
    }
    
    // Use out_count_points_fft as search_range (e.g., 60 for 30+30)
    uint search_range = params_.out_count_points_fft;
    
    // Layout: params | complex_buffer | magnitude_buffer
    size_t post_params_size = sizeof(cl_uint) * 4;
    size_t post_complex_size = params_.beam_count * search_range * sizeof(cl_float2);
    size_t post_magnitude_size = params_.beam_count * search_range * sizeof(float);
    size_t maxima_size = params_.beam_count * params_.max_peaks_count * sizeof(MaxValue);
    
    // STAGE 1: Create kernel if not exists
    if (!reduction_kernel_) {
        CreateMaxReductionKernel();
    }
    
    // Create buffer for maxima results
    if (!buffer_maxima_) {
        const size_t maxima_elements = (maxima_size + sizeof(std::complex) - 1) / sizeof(std::complex);
        buffer_maxima_ = engine_->CreateBuffer(maxima_elements, gpu::MemoryType::GPU_READ_WRITE);
    }
    
    // STAGE 2: Create sub-buffers
    cl_int err;
    
    cl_buffer_region complex_region = {post_params_size, post_complex_size};
    cl_mem complex_sub_buffer = clCreateSubBuffer(
        post_callback_userdata_,
        CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION,
        &complex_region,
        &err
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create complex sub-buffer: " + std::to_string(err));
    }
    
    cl_buffer_region magnitude_region = {post_params_size + post_complex_size, post_magnitude_size};
    cl_mem magnitude_sub_buffer = clCreateSubBuffer(
        post_callback_userdata_,
        CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION,
        &magnitude_region,
        &err
    );
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(complex_sub_buffer);
        throw std::runtime_error("Failed to create magnitude sub-buffer: " + std::to_string(err));
    }
    
    // STAGE 3: Run reduction kernel
    cl_uint beam_count = static_cast(params_.beam_count);
    cl_uint search_range_cl = static_cast(search_range);
    cl_uint max_peaks_count = static_cast(params_.max_peaks_count);
    
    cl_mem maxima_mem = buffer_maxima_->Get();
    
    clSetKernelArg(reduction_kernel_, 0, sizeof(cl_mem), &complex_sub_buffer);
    clSetKernelArg(reduction_kernel_, 1, sizeof(cl_mem), &magnitude_sub_buffer);
    clSetKernelArg(reduction_kernel_, 2, sizeof(cl_mem), &maxima_mem);
    clSetKernelArg(reduction_kernel_, 3, sizeof(cl_uint), &beam_count);
    clSetKernelArg(reduction_kernel_, 4, sizeof(cl_uint), &search_range_cl);  // ← CHANGED
    clSetKernelArg(reduction_kernel_, 5, sizeof(cl_uint), &max_peaks_count);
    
    // One work-group per beam
    size_t global_work_size = params_.beam_count * 256;
    size_t local_work_size = 256;
    
    // Adjust for smaller search_range
    if (search_range < 256) {
        local_work_size = 64;
        global_work_size = params_.beam_count * local_work_size;
    }
    
    cl_event reduction_event = nullptr;
    err = clEnqueueNDRangeKernel(
        queue_,
        reduction_kernel_,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        0, nullptr, &reduction_event
    );
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(complex_sub_buffer);
        clReleaseMemObject(magnitude_sub_buffer);
        throw std::runtime_error("Failed to enqueue reduction kernel: " + std::to_string(err));
    }
    
    last_profiling_.reduction_time_ms = ProfileEvent(reduction_event, "Reduction + Phase");
    clWaitForEvents(1, &reduction_event);
    clReleaseEvent(reduction_event);
    
    clReleaseMemObject(complex_sub_buffer);
    clReleaseMemObject(magnitude_sub_buffer);
    
    // STAGE 4: Read results from GPU
    std::vector maxima_result(params_.beam_count * params_.max_peaks_count);
    err = clEnqueueReadBuffer(
        queue_,
        buffer_maxima_->Get(),
        CL_TRUE,
        0,
        maxima_size,
        maxima_result.data(),
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read maxima from GPU: " + std::to_string(err));
    }
    
    // STAGE 5: Convert to FFTMaxResult
    std::vector> all_results;
    all_results.resize(params_.beam_count);
    
    for (size_t beam_idx = 0; beam_idx < params_.beam_count; ++beam_idx) {
        auto& beam_out = all_results[beam_idx];
        beam_out.reserve(params_.max_peaks_count);
        
        for (size_t i = 0; i < params_.max_peaks_count; ++i) {
            const auto& mv = maxima_result[beam_idx * params_.max_peaks_count + i];
            
            if (mv.index != UINT_MAX && mv.magnitude > 0.0f) {
                FFTMaxResult max_result;
                max_result.index_point = mv.index;      // Index in search_range (0..search_range-1)
                max_result.amplitude = mv.magnitude;    // Amplitude/Magnitude
                max_result.phase = mv.phase;            // Phase in degrees
                beam_out.push_back(max_result);
            }
        }
    }
    
    return all_results;
}

// ════════════════════════════════════════════════════════════════════════════
// UPDATE: CreateOrReuseFFTPlan() - PostCallbackUserData struct
// ════════════════════════════════════════════════════════════════════════════
//
// Find this in CreateOrReuseFFTPlan() method (around line 920):
//
// BEFORE:
// struct PostCallbackUserData {
//     cl_uint beam_count;
//     cl_uint nFFT;
//     cl_uint out_count_points_fft;  // ← CONFUSING!
//     cl_uint max_peaks_count;
// };
//
// AFTER:
// struct PostCallbackUserData {
//     cl_uint beam_count;
//     cl_uint nFFT;
//     cl_uint search_range;           // ← CLARIFIED! (30+30=60)
//     cl_uint max_peaks_count;
// };
//
// PostCallbackUserData post_cb_params = {
//     static_cast(params_.beam_count),
//     static_cast(nFFT_),
//     static_cast(params_.out_count_points_fft),  // Use as search_range
//     static_cast(params_.max_peaks_count)
// };
//
// size_t post_params_size = sizeof(PostCallbackUserData);
// size_t post_complex_size = params_.beam_count * params_.out_count_points_fft * sizeof(cl_float2);
// size_t post_magnitude_size = params_.beam_count * params_.out_count_points_fft * sizeof(float);
// size_t post_userdata_size = post_params_size + post_complex_size + post_magnitude_size;
//
