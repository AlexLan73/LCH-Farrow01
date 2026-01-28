# HIP ↔ OpenCL Zero-Copy Interoperability

## Дата: 2026-01-26
## Статус: ✅ РЕШЕНИЕ НАЙДЕНО

---

## Содержание

| Файл | Описание |
|------|----------|
| [START_HERE.txt](START_HERE.txt) | Быстрый старт |
| [EXECUTIVE_SUMMARY_RU.md](EXECUTIVE_SUMMARY_RU.md) | Краткое резюме для руководства |
| [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md) | Полный технический анализ |
| [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) | Архитектурные диаграммы |
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | Практическое руководство |
| [code/](code/) | Рабочие примеры кода |

---

## Проблема

Большой объём кода написан на OpenCL. Нет возможности переписать всё на HIP.
Нужно интегрировать OpenCL (FFT, preprocessing) с HIP/rocBLAS/rocSOLVER (матричные операции)
**без копирования данных между GPU и CPU**.

---

## Решение

**OpenCL SVM (Shared Virtual Memory)** буферы доступны напрямую из HIP на AMD ROCm.

```cpp
// Один буфер - два API
Complex* data = (Complex*)clSVMAlloc(ctx, flags, size, 0);

// OpenCL kernel
clSetKernelArgSVMPointer(kernel, 0, data);

// HIP kernel - тот же указатель!
hip_kernel<<<grid, block>>>(data, n);
```

---

## Проверено на

- **GPU**: AMD Instinct MI100
- **ROCm**: 6.3.2
- **OpenCL**: 2.1 AMD-APP (3635.0)
- **Вектор**: 4M complex float (32 MB)
- **Результат**: Zero-copy работает ✅

---

## Быстрый запуск тестов

```bash
cd /home/user/lanin/Matrix/Analysis_HIP_OpenCL_Interop_2026-01/code

# Базовый тест
hipcc -O3 hip_opencl_svm_test.cpp -o test -lOpenCL && ./test

# Бенчмарк с 4M элементами
hipcc -O3 hip_opencl_svm_benchmark.cpp -o bench -lOpenCL && ./bench
```

