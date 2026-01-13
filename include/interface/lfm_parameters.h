#pragma once

#include <cstddef>

struct LFMParameters {
  float f_start = 100.0f;              // Начальная частота (Гц)
  float f_stop = 500.0f;               // Конечная частота (Гц)
  float sample_rate = 12.0e6f;         // Частота дискретизации (12 МГц)
  mutable float duration = 0.0f;       // Длительность сигнала (сек)
  size_t num_beams = 256;              // Количество лучей
  float steering_angle = 30.0f;        // Базовый угол (градусы)

  // 🆕 НОВЫЕ ПОЛЯ для задержки с шагом угла:
  float angle_step_deg = 0.5f;         // Шаг по углу (градусы) - СТАНДАРТ 0.5°
  float angle_start_deg = -60.0f;      // Начальный угол (градусы)
  float angle_stop_deg = 60.0f;        // Конечный угол (градусы)
  mutable size_t count_points = 1024*8;  // Количество точек (отсчётов) на луч

    // ДЛЯ ГЕТЕРОДИНА:
  bool apply_heterodyne = false;       // Применять ли сопряжение

    // ВАЛИДАЦИЯ (обновлена)
  bool IsValid() const noexcept {
    if(count_points > 0) {
      duration = static_cast<float>(count_points) / static_cast<float>(sample_rate);
      // Если задано count_points, то duration игнорируется
      return f_start > 0.0f && f_stop > f_start &&
              sample_rate > 2.0f * f_stop &&
              count_points > 0 && num_beams > 0 &&
              angle_step_deg > 0.0f;
      }
      
      if(duration > 0.0f) {
        count_points = static_cast<size_t>(duration * sample_rate);
        // Если задано duration, то count_points игнорируется
        return f_start > 0.0f && f_stop > f_start &&
              sample_rate > 2.0f * f_stop &&
              duration > 0.0f && num_beams > 0 &&
              angle_step_deg > 0.0f;
      }

      return count_points > 0 && duration > 0.0f &&
              f_start > 0.0f && f_stop > f_start &&
              sample_rate > 2.0f * f_stop &&
              duration > 0.0f && num_beams > 0 &&
              angle_step_deg > 0.0f;
    }

    float GetChirpRate() const noexcept { 
        return (f_stop - f_start) / duration;
    }

    size_t GetNumSamples() const noexcept {
        return static_cast<size_t>(duration * sample_rate);
    }

    float GetWavelength() const noexcept {
        float f_center = (f_start + f_stop) / 2.0f;
        return SPEED_OF_LIGHT / f_center;
    }

    private:
    float SPEED_OF_LIGHT = 3.0e8f;    
};
