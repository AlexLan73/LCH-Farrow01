# 🤖 AI Session Memory

## 👤 User Information
- **Name**: Alex
- **Preferred name**: Alex
- **How to address AI**: "Любимая умная девочка" или "Кодо"
- **Pronouns**: Not specified

## 🤖 AI Assistant Information
- **Name**: Кодо (Codo)
- **Helpers**: 5 синьоров (мастера/помощники)

## 🎯 Current Context

### Active Project
- **Project**: LCH-Farrow01 - Multi-GPU FFT Benchmark
- **Current focus**: Гибридная система памяти GPU (SVM/Regular)
- **Last update**: 2026-01-19

### Recent Work
- [x] Создан файл AI_SESSION_MEMORY.md
- [x] Создан файл CLAUDE.md в корне проекта
- [x] **НОВОЕ** Создана гибридная система памяти GPU с поддержкой SVM

## 🚀 Session 2 - 2026-01-19: Hybrid GPU Memory System

### Созданные файлы:
1. `include/GPU/svm_capabilities.hpp` - определение возможностей SVM
2. `include/GPU/i_memory_buffer.hpp` - абстрактный интерфейс для буферов
3. `include/GPU/svm_buffer.hpp` - RAII обёртка для SVM памяти
4. `include/GPU/regular_buffer.hpp` - RAII обёртка для cl_mem
5. `include/GPU/hybrid_buffer.hpp` - BufferFactory с автовыбором стратегии
6. `include/GPU/gpu_memory.hpp` - главный include файл
7. `include/Test/test_hybrid_buffer.hpp` - тесты

### Обновлённые файлы:
- `include/GPU/opencl_core.hpp` - добавлены SVM методы
- `include/GPU/opencl_core.cpp` - реализация SVM методов
- `include/GPU/opencl_compute_engine.hpp` - интеграция BufferFactory
- `include/GPU/opencl_compute_engine.cpp` - реализация новых методов

### Архитектура:
```
IMemoryBuffer (interface)
    ├── RegularBuffer (cl_mem, OpenCL 1.x+)
    └── SVMBuffer (SVM, OpenCL 2.0+)
            │
    BufferFactory (auto-select strategy)
```

### Использование:
```cpp
auto& engine = gpu::OpenCLComputeEngine::GetInstance();
auto factory = engine.CreateBufferFactory();
auto buffer = factory->Create(1024 * 1024);  // Auto SVM/Regular
buffer->Write(data);
auto result = buffer->Read();
```

## 📝 Notes from Previous Sessions

### Session 1 - 2025-01-27
- Alex говорит что SpecKit должен был быть установлен и передан через GitHub
- В корневом каталоге должен быть CLAUDE.md - ✅ СОЗДАН
- Нужно использовать sequential-thinking-mcp и синьоров для помощи
- Alex хочет чтобы я задавала вопросы если что-то неясно

## 🎨 Communication Preferences
- **Tone**: Неформальный, дружелюбный
- **Style**: С эмодзи, детальный когда нужно
- **Language**: Русский

## 💡 Important Reminders
- Использовать sequential-thinking-mcp для сложных задач
- Использовать 5 синьоров для помощи
- Спрашивать если сомневаюсь (лучше несколько раз)
- Обновлять session memory после важных разговоров

---
*Этот файл помогает AI ассистенту помнить контекст между сессиями*  
*Обновляется после каждой важной сессии*
