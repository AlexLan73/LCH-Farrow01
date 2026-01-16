✅ ИСПРАВЛЕННЫЙ KERNEL С ИНТЕРПОЛЯЦИЕЙ
ПОЛНЫЙ ИСПРАВЛЕННЫЙ kernel_lfm_combined()
text
__kernel void kernel_lfm_combined(
    __global float2 *output,
    __global const CombinedDelayParam *combined,
    float f_start, float f_stop, float sample_rate,
    float duration, float speed_of_light,
    uint num_samples, uint num_beams, uint num_delays
) {
    uint gid = get_global_id(0);
    if (gid >= (uint)num_samples * num_beams) return;
    
    uint ray_id = gid / num_samples;
    uint sample_id = gid % num_samples;
    
    float delay_degrees = combined[ray_id].delay_degrees;
    float delay_time_ns = combined[ray_id].delay_time_ns;
    
    float f_center = (f_start + f_stop) / 2.0f;
    float wavelength = speed_of_light / f_center;
    float delay_rad = delay_degrees * 3.14159265f / 180.0f;
    float delay_angle_sec = delay_rad * wavelength / speed_of_light;
    float delay_time_sec = delay_time_ns * 1e-9f;
    float total_delay_sec = delay_angle_sec + delay_time_sec;
    
    // ✅ ГЛАВНОЕ ИСПРАВЛЕНИЕ: ИСПОЛЬЗУЕМ FLOAT ВМЕСТО INT!
    float total_delay_samples = total_delay_sec * sample_rate;
    float delayed_sample_float = (float)sample_id - total_delay_samples;
    
    if (delayed_sample_float < 0.0f) {
        output[ray_id * num_samples + sample_id] = (float2)(0.0f, 0.0f);
        return;
    }
    
    int sample_int = (int)delayed_sample_float;
    float sample_frac = delayed_sample_float - (float)sample_int;
    
    if (sample_int >= (int)num_samples - 1) {
        output[ray_id * num_samples + sample_id] = (float2)(0.0f, 0.0f);
    }
    else if (sample_frac < 1e-6f) {
        float t = (float)sample_int / sample_rate;
        float chirp_rate = (f_stop - f_start) / duration;
        float phase = 2.0f * 3.14159265f * (f_start * t + 0.5f * chirp_rate * t * t);
        output[ray_id * num_samples + sample_id] = (float2)(cos(phase), sin(phase));
    }
    else {
        // ✅ ИНТЕРПОЛЯЦИЯ между двумя соседними отсчётами
        float t1 = (float)sample_int / sample_rate;
        float chirp_rate = (f_stop - f_start) / duration;
        float phase1 = 2.0f * 3.14159265f * (f_start * t1 + 0.5f * chirp_rate * t1 * t1);
        float real1 = cos(phase1), imag1 = sin(phase1);
        
        float t2 = (float)(sample_int + 1) / sample_rate;
        float phase2 = 2.0f * 3.14159265f * (f_start * t2 + 0.5f * chirp_rate * t2 * t2);
        float real2 = cos(phase2), imag2 = sin(phase2);
        
        float real = real1 * (1.0f - sample_frac) + real2 * sample_frac;
        float imag = imag1 * (1.0f - sample_frac) + imag2 * sample_frac;
        output[ray_id * num_samples + sample_id] = (float2)(real, imag);
    }
}
КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ
✅ float total_delay_samples вместо int
✅ float delayed_sample_float вместо int
✅ Линейная интерполяция между отсчётами
✅ Граница на < 0.0f вместо целых чисел