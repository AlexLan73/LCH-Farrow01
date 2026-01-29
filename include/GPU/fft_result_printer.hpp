#pragma once
/**
 * @file fft_result_printer.hpp
 * @brief Класс для форматированного вывода результатов FFT
 * 
 * Принцип Single Responsibility: AntennaFFTProcMax только обрабатывает,
 * FFTResultPrinter только выводит.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <cstdio>
#include "interface/antenna_fft_params.h"

namespace antenna_fft {

/**
 * @brief Класс для форматированного вывода результатов FFT
 * 
 * Использование:
 *   FFTResultPrinter printer;
 *   auto result = processor.Process(data);
 *   printer.PrintAll(result, processor.GetLastProfiling(), params);
 */
class FFTResultPrinter {
public:
    // ═══════════════════════════════════════════════════════════════════════════
    // Настройки вывода
    // ═══════════════════════════════════════════════════════════════════════════
    
    struct PrintOptions {
        bool show_profiling = true;         // Показывать таблицу профилирования
        bool show_results = true;           // Показывать результаты максимумов
        bool show_all_peaks = true;         // Показывать ВСЕ пики (true) или только первый (false)
        bool show_parameters = true;        // Показывать параметры теста
        bool show_pipeline_steps = false;   // Показывать шаги pipeline (для отладки)
        size_t max_beams_to_display = 10;   // Сколько лучей показывать (0 = все)
    };
    
    FFTResultPrinter() = default;
    explicit FFTResultPrinter(const PrintOptions& options) : options_(options) {}
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Основные методы вывода
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Вывести всё: параметры + профилирование + результаты
     */
    void PrintAll(const AntennaFFTResult& result, 
                  const FFTProfilingResults& profiling,
                  const AntennaFFTParams& params) const {
        if (options_.show_parameters) {
            PrintParameters(params);
        }
        if (options_.show_profiling) {
            PrintProfiling(profiling);
        }
        if (options_.show_results) {
            PrintResults(result, params.max_peaks_count);
        }
        PrintComplete();
    }
    
    /**
     * @brief Вывести таблицу параметров
     */
    void PrintParameters(const AntennaFFTParams& params) const {
        std::cout << "\n  ┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "  │  ПАРАМЕТРЫ ОБРАБОТКИ                                        │\n";
        std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";
        
        printf("  ┌─────────────────────────────┬────────────────┐\n");
        printf("  │  Параметр                   │    Значение    │\n");
        printf("  ├─────────────────────────────┼────────────────┤\n");
        printf("  │  beam_count (лучей)         │  %12zu  │\n", params.beam_count);
        printf("  │  count_points (точек/луч)   │  %12zu  │\n", params.count_points);
        printf("  │  out_count_points_fft       │  %12zu  │\n", params.out_count_points_fft);
        printf("  │  max_peaks_count            │  %12zu  │\n", params.max_peaks_count);
        printf("  └─────────────────────────────┴────────────────┘\n");
    }
    
    /**
     * @brief Вывести таблицу профилирования GPU
     */
    void PrintProfiling(const FFTProfilingResults& profiling) const {
        std::cout << "\n  ┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "  │  GPU PROFILING                                              │\n";
        std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";
        
        printf("  ┌─────────────────────────────┬────────────────┐\n");
        printf("  │  Операция                   │    Время (ms)  │\n");
        printf("  ├─────────────────────────────┼────────────────┤\n");
        printf("  │  Upload                     │  %12.4f  │\n", profiling.upload_time_ms);
        printf("  │  FFT + pre-callback         │  %12.4f  │\n", profiling.fft_time_ms);
        printf("  │  Post (mag+max+phase)       │  %12.4f  │\n", profiling.post_callback_time_ms);
        printf("  ├─────────────────────────────┼────────────────┤\n");
        printf("  │  TOTAL GPU                  │  %12.4f  │\n", profiling.total_time_ms);
        printf("  └─────────────────────────────┴────────────────┘\n");
    }
    
    /**
     * @brief Вывести результаты максимумов для каждого луча
     */
    void PrintResults(const AntennaFFTResult& result, size_t max_peaks_count) const {
        std::cout << "\n  ┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "  │  РЕЗУЛЬТАТЫ: Максимумы (Top " << max_peaks_count << " для каждого луча)" << std::string(14 - std::to_string(max_peaks_count).length(), ' ') << "│\n";
        std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";
        
        if (options_.show_all_peaks) {
            PrintResultsAllPeaks(result, max_peaks_count);
        } else {
            PrintResultsFirstPeak(result);
        }
    }
    
    /**
     * @brief Вывести сообщение о завершении
     */
    void PrintComplete() const {
        std::cout << "\n════════════════════════════════════════════════════════════════\n";
        std::cout << "  PROCESSING COMPLETE ✅\n";
        std::cout << "════════════════════════════════════════════════════════════════\n\n";
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Методы для Pipeline отладки
    // ═══════════════════════════════════════════════════════════════════════════
    
    void PrintPipelineStep(int step, const std::string& description) const {
        if (options_.show_pipeline_steps) {
            std::cout << "\n[STEP " << step << "] " << description << "\n";
        }
    }
    
    void PrintPipelineEvent(const std::string& event_name) const {
        if (options_.show_pipeline_steps) {
            std::cout << "  → " << event_name << "\n";
        }
    }
    
    void PrintPipelineComplete() const {
        if (options_.show_pipeline_steps) {
            std::cout << "  ✅ Операция завершена\n";
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Getters/Setters
    // ═══════════════════════════════════════════════════════════════════════════
    
    PrintOptions& GetOptions() { return options_; }
    const PrintOptions& GetOptions() const { return options_; }
    void SetOptions(const PrintOptions& options) { options_ = options; }
    
    // Удобные методы для быстрой настройки
    void EnableAll() {
        options_.show_profiling = true;
        options_.show_results = true;
        options_.show_all_peaks = true;
        options_.show_parameters = true;
        options_.show_pipeline_steps = true;
    }
    
    void EnableMinimal() {
        options_.show_profiling = true;
        options_.show_results = true;
        options_.show_all_peaks = false;
        options_.show_parameters = false;
        options_.show_pipeline_steps = false;
    }
    
    void DisableAll() {
        options_.show_profiling = false;
        options_.show_results = false;
        options_.show_parameters = false;
        options_.show_pipeline_steps = false;
    }

private:
    PrintOptions options_;
    
    /**
     * @brief Форматировать частоту в Гц/кГц/МГц для читаемости
     */
    static std::string FormatFrequency(float freq_hz) {
        char buf[64];
        if (freq_hz >= 1e6f) {
            snprintf(buf, sizeof(buf), "%.4f МГц", freq_hz / 1e6f);
        } else if (freq_hz >= 1e3f) {
            snprintf(buf, sizeof(buf), "%.4f кГц", freq_hz / 1e3f);
        } else {
            snprintf(buf, sizeof(buf), "%.4f Гц", freq_hz);
        }
        return std::string(buf);
    }
    
    /**
     * @brief Вывести ВСЕ пики для каждого луча
     */
    void PrintResultsAllPeaks(const AntennaFFTResult& result, size_t max_peaks_count) const {
        const size_t beams_to_show = (options_.max_beams_to_display == 0) 
                                      ? result.results.size() 
                                      : std::min(options_.max_beams_to_display, result.results.size());
        
        for (size_t beam_idx = 0; beam_idx < beams_to_show; ++beam_idx) {
            const auto& beam = result.results[beam_idx];
            
            printf("  ╔════════════════════════════════════════════════════════════════════════════════════════════╗\n");
            printf("  ║  Луч %3zu                                                                                   ║\n", beam_idx);
            
            // Вывод уточнённой частоты (параболическая интерполяция)
            if (!beam.max_values.empty()) {
                float refined_bin = static_cast<float>(beam.max_values[0].index_point) + beam.freq_offset;
                std::string freq_str = FormatFrequency(beam.refined_frequency);
                printf("  ║  Refined Frequency: %s (bin index: %.4f)                                   ║\n", 
                       freq_str.c_str(), refined_bin);
            }
            
            printf("  ╠════════════════════════════════════════════════════════════════════════════════════════════╣\n");
            printf("  ║  Peak  │  Index  │   Amplitude    │  Phase (°)  │       Re       │       Im       ║\n");
            printf("  ╠────────┼─────────┼────────────────┼─────────────┼────────────────┼────────────────╣\n");
            
            if (beam.max_values.empty()) {
                printf("  ║  (нет данных)                                                                              ║\n");
            } else {
                for (size_t i = 0; i < beam.max_values.size() && i < max_peaks_count; ++i) {
                    const auto& mv = beam.max_values[i];
                    printf("  ║  %4zu  │  %5zu  │  %12.2f  │  %9.2f  │  %12.2f  │  %12.2f  ║\n",
                           i + 1, mv.index_point, mv.amplitude, mv.phase, mv.real, mv.imag);
                }
            }
            printf("  ╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n");
        }
        
        if (result.results.size() > beams_to_show) {
            printf("  ... и ещё %zu лучей (показаны первые %zu)\n\n", 
                   result.results.size() - beams_to_show, beams_to_show);
        }
    }
    
    /**
     * @brief Вывести только первый (максимальный) пик для каждого луча
     */
    void PrintResultsFirstPeak(const AntennaFFTResult& result) const {
        printf("  ┌────────┬─────────┬────────────────┬─────────────┬──────────────────────┐\n");
        printf("  │  Луч   │  Index  │   Amplitude    │   Phase (°) │  Refined Frequency   │\n");
        printf("  ├────────┼─────────┼────────────────┼─────────────┼──────────────────────┤\n");
        
        const size_t beams_to_show = (options_.max_beams_to_display == 0) 
                                      ? result.results.size() 
                                      : std::min(options_.max_beams_to_display, result.results.size());
        
        for (size_t beam_idx = 0; beam_idx < beams_to_show; ++beam_idx) {
            const auto& beam = result.results[beam_idx];
            if (!beam.max_values.empty()) {
                const auto& mv = beam.max_values[0];
                std::string freq_str = FormatFrequency(beam.refined_frequency);
                printf("  │  %4zu  │  %5zu  │  %12.4f  │  %9.2f  │  %18s  │\n", 
                       beam_idx, mv.index_point, mv.amplitude, mv.phase, freq_str.c_str());
            }
        }
        
        if (result.results.size() > beams_to_show) {
            printf("  │  ...   │   ...   │      ...       │     ...     │        ...           │\n");
        }
        printf("  └────────┴─────────┴────────────────┴─────────────┴──────────────────────┘\n");
    }
};

} // namespace antenna_fft

