# Стратегия профилирования матричной инверсии 341×341 на GPU AI100

## 1. Цель профилирования
- **Время выполнения (чистое)**: <5 мс (без transfer CPU↔GPU)
- **Сравнение**: rocSOLVER (нативная) vs кастомный kernel
- **Анализ узких мест**: bottleneck identification
- **Выходные форматы**: CSV, JSON, markdown отчеты

---

## 2. Инструменты профилирования

### 2.1 rocprof (AMD ROCm Profiler)
**Основной инструмент для GPU AI100**

```bash
# Базовая статистика
rocprof --stats --basenames on ./matrix_invert

# С временными метриками
rocprof --timestamp on -i rocprof_counters.txt ./matrix_invert

# CSV вывод
rocprof --csv --stats ./matrix_invert
```

**Выходные файлы:**
- `results.csv` - информация per kernel call
- `results.stats.csv` - агрегированная статистика
- Ключевые метрики: DispatchNs, BeginNs, EndNs, CompleteNs

### 2.2 omniperf (rocprofiler-compute)
**Продвинутый анализ с roofline**

```bash
# Профилирование с автоматической сборкой метрик
omniperf profile -k matrix_invert_kernel --dispatch 0

# Анализ с roofline диаграммой
omniperf analyze -p results/ --roofline
```

**Что собирает:**
- VALUUtilization (вычислительные единицы)
- L1/L2 cache hit rates
- Memory bandwidth utilization
- Wavefront occupancy
- Arithmetic intensity (FLOPs/Byte)

### 2.3 Radeon GPU Profiler (RGP)
**Визуализация и анализ временной шкалы**

```bash
rocprof --sqtt=on ./matrix_invert
# Открыть results.sqtt в RGP GUI
```

---

## 3. Ключевые метрики для анализа

### 3.1 Timing метрики
```
- Kernel Duration: EndNs - BeginNs
- Memory Transfer Time: CPU→GPU, GPU→CPU (ИСКЛЮЧИТЬ ИЗ ПОДСЧЕТА)
- Total Execution Time: чистое время вычислений
```

### 3.2 Memory метрики
```
- Global Memory Bandwidth: GB/s (peak ~900 GB/s для AI100)
- L1 Cache Hit Rate: % (высокий % = хорошо)
- L2 Cache Hit Rate: % (>80% = хорошо)
- LDS (Local Data Store) Utilization: % (полезное для GEMM)
```

### 3.3 Compute метрики
```
- VALU Utilization: % (Vector ALU)
- Wave Occupancy: % (сколько wave активны)
- Register Usage: per thread
- Instruction Throughput: инструкции/цикл
```

### 3.4 Bottleneck indicators
```
- Memory Bound: если bandwidth ~90%+ или L1/L2 misses высоки
- Compute Bound: если VALU <80% utilization
- Latency Bound: если atomic operations blocking
```

---

## 4. Алгоритмический подход

### 4.1 Для матричной инверсии 341×341

#### Вариант 1: LU Decomposition + TRSM (лучший выбор)
```
1. LU factorization: A = L*U (rocSOLVER GETRF)
2. Augment matrix: [L | I]
3. Solve: U*X = I (rocSOLVER TRSM)
4. Solve: L*X = I (rocSOLVER TRSM)
```

**Преимущества:**
- Числовая стабильность с pivoting
- Оптимизирован в rocSOLVER/rocBLAS
- FLOPs: ~4N³/3 (для 341: ~157 млн ops)

#### Вариант 2: QR Decomposition
```
1. QR factorization
2. Solve для инверсии
```

**Минусы:** медленнее LU для общих матриц

#### Вариант 3: Gauss-Jordan (для специальных матриц)
```
Плюсы: можно параллелизировать по строкам
Минусы: менее численно стабилен
```

---

## 5. Гибридный подход (рекомендуется)

### Комбинация технологий:
```
┌─────────────────────────────────────────┐
│ rocSOLVER (LU Factorization)           │
│ - GETRF для LU                         │
│ - Оптимизирован AMD                    │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ rocBLAS (TRSM - Triangular Solve)      │
│ - Оптимизирован для полных матриц      │
│ - Лучше чем кастомный kernel           │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Кастомный kernel (опциональный)        │
│ - LDS-optimized GEMM для K-loop        │
│ - Register tiling                      │
│ - Для доп. оптимизации                 │
└─────────────────────────────────────────┘
```

---

## 6. План профилирования

### Фаза 1: Базовое профилирование (rocprof)
```bash
# Запуск с timing
rocprof --timestamp on ./matrix_invert_basic

# Вывод: results.csv с DispatchNs, BeginNs, EndNs
# Анализируем: время GETRF, TRSM, total
```

### Фаза 2: Детальный анализ (omniperf)
```bash
# Запуск omniperf для каждого kernel
omniperf profile --dispatch-id 0 -k .

# Собираем метрики:
# - L1/L2 hit rates
# - Memory bandwidth
# - VALU utilization
# - Occupancy
```

### Фаза 3: Bottleneck identification
```
Если memory-bound:
  → Optimize data access patterns
  → Increase LDS usage
  → Improve cache locality

Если compute-bound:
  → Increase arithmetic intensity
  → Use register tiling
  → Optimize instruction mixing
```

### Фаза 4: Сравнение реализаций
```bash
# rocSOLVER baseline
./matrix_invert_rocsolver
rocprof --stats ./matrix_invert_rocsolver

# Кастомный kernel
./matrix_invert_custom
rocprof --stats ./matrix_invert_custom

# Гибридный подход
./matrix_invert_hybrid
rocprof --stats ./matrix_invert_hybrid
```

---

## 7. Выходные форматы

### 7.1 CSV для обработки
```csv
Implementation,Kernel,Duration_ms,VALU_Util_%,L1_Hit_%,L2_Hit_%,Bandwidth_GB/s,FLOPs/s
rocSOLVER,GETRF,2.3,87.5,92.1,78.4,650,2.1e11
rocSOLVER,TRSM,1.8,81.2,88.5,72.1,580,1.9e11
Custom,LU_kernel,2.1,89.3,94.2,81.3,720,2.3e11
Hybrid,Fused,1.6,91.5,96.1,85.2,780,2.5e11
```

### 7.2 JSON для парсинга
```json
{
  "profiling_run": {
    "timestamp": "2026-01-22T15:42:00Z",
    "gpu": "AI100",
    "matrix_size": "341x341",
    "implementations": [
      {
        "name": "rocSOLVER",
        "kernels": [
          {
            "name": "GETRF",
            "duration_ms": 2.3,
            "metrics": {
              "valu_util": 87.5,
              "l1_hit_rate": 92.1,
              "bandwidth_gb_s": 650
            }
          }
        ]
      }
    ]
  }
}
```

### 7.3 Markdown отчет
```markdown
# Профилирование матричной инверсии 341×341

## Краткое резюме
- Лучший результат: 1.6 мс (Hybrid approach)
- Улучшение vs rocSOLVER: 44%
- Bottleneck: Memory bandwidth (AI100: 900 GB/s)

## Детальные результаты
[Таблицы, графики, рекомендации]
```

---

## 8. Рекомендуемая последовательность

1. **Установка**: Убедиться rocprof/omniperf установлены
2. **Компиляция**: Все варианты (rocSOLVER, кастомный, гибридный)
3. **Профилирование**: Запустить в последовательности 1→2→3→4
4. **Анализ**: Python скрипт парсит CSV/JSON
5. **Отчет**: Генерирует markdown с рекомендациями

---

## 9. Архитектурные параметры AI100

```
- Compute Units (CU): 120
- Wavefront size: 64 threads
- Peak FP32 Performance: 40 TFLOPS
- Peak Bandwidth: 900 GB/s
- L1 Cache: 16 KB/CU
- L2 Cache: 4 MB (shared)
- LDS: 96 KB/CU
- Max threads/block: 1024 (или 32 waves)
```

**Оптимальные параметры для матричной инверсии:**
- Work group size: 256 threads (4 waves)
- LDS tiling: 32×32 block
- Register tiling: 4×4 per thread
- Memory coalescing: 128 bytes aligned

---

## 10. Возможные оптимизации после профилирования

### Если memory-bound:
```cpp
// 1. Увеличить LDS usage для GEMM
// 2. Padding LDS для избежания bank conflicts
// 3. Optimize memory access patterns (coalesced)
```

### Если compute-bound:
```cpp
// 1. Increase register tiling (C_reg 8x8 per thread)
// 2. Use WMMA/MFMA matrix instructions
// 3. Reduce synchronization barriers
```

### Для достижения <5 мс:
```
Текущий estimate: 1.6-2.5 мс
Запас: ~2 мс для оптимизаций
```

---

## Инструменты анализа (Python)

```python
# analyze_profile.py
import pandas as pd
import json
import matplotlib.pyplot as plt

# 1. Load CSV results
df = pd.read_csv('results.csv')

# 2. Compute metrics
df['efficiency'] = df['VALU_Util_%'] / 100.0
df['memory_bandwidth_pct'] = df['Bandwidth_GB/s'] / 900.0

# 3. Identify bottlenecks
def identify_bottleneck(row):
    if row['Bandwidth_GB/s'] > 800:
        return 'Memory-Bound'
    elif row['VALU_Util_%'] < 70:
        return 'Compute-Bound'
    else:
        return 'Balanced'

# 4. Generate report
report = df.to_markdown()
```

---

**Следующий шаг:** Подготовить исходный код и начать профилирование!
