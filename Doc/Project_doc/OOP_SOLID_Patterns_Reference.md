# Справочник: ООП, SOLID, Паттерны GRASP и GoF

## Обзор

Этот документ содержит справочник по принципам объектно-ориентированного программирования (ООП), принципам SOLID, паттернам GRASP и паттернам Gang of Four (GoF), используемым в проекте LCH-Farrow01.

## Объектно-Ориентированное Программирование (ООП)

### Основные принципы ООП

#### 1. Инкапсуляция (Encapsulation)
**Определение:** Сокрытие внутренней реализации объекта и предоставление публичного интерфейса.

**В проекте:**
```cpp
class GPUMemoryBuffer {
private:
    cl_mem gpu_buffer_;           // Скрытая реализация
    size_t num_elements_;         // Приватные данные
public:
    cl_mem Get() const;           // Публичный интерфейс
    void WriteToGPU(const std::vector<std::complex<float>>& data);
};
```

**Преимущества:**
- Защита внутренних данных
- Изменение реализации без влияния на клиентов
- Упрощение интерфейса

#### 2. Наследование (Inheritance)
**Определение:** Создание новых классов на основе существующих.

**В проекте:**
```cpp
// Базовый класс для всех менеджеров
class OpenCLManager {
protected:
    cl_context context_;
    cl_device_id device_;
public:
    virtual ~OpenCLManager() = default;
};

// Специализированный менеджер
class GPUMemoryManager : public OpenCLManager {
    // Специфичная функциональность
};
```

#### 3. Полиморфизм (Polymorphism)
**Определение:** Возможность объектов разных классов отвечать на одинаковые сообщения по-разному.

**В проекте:**
```cpp
// Полиморфное поведение буферов
std::unique_ptr<GPUMemoryBuffer> buffer = engine.CreateBuffer(1024, MemoryType::GPU_READ_ONLY);
// buffer может быть разных типов (owning/non-owning), но интерфейс одинаковый
```

#### 4. Абстракция (Abstraction)
**Определение:** Выделение существенных характеристик объекта, игнорируя несущественные.

**В проекте:**
```cpp
// Абстракция OpenCL операций
class OpenCLComputeEngine {
public:
    // Высокоуровневый интерфейс
    std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t, MemoryType);
    void ExecuteKernel(cl_kernel, const std::vector<cl_mem>&, ...);

private:
    // Скрытые детали реализации
    OpenCLCore* core_;
    CommandQueuePool* queues_;
};
```

## Принципы SOLID

### 1. Single Responsibility Principle (SRP) - Принцип единственной ответственности

**Определение:** Класс должен иметь только одну причину для изменения.

**В проекте:**

**Хорошо (SRP соблюден):**
```cpp
class GPUMemoryBuffer {
    // Единственная ответственность: управление GPU памятью
    void WriteToGPU(const std::vector<std::complex<float>>& data);
    std::vector<std::complex<float>> ReadFromGPU();
};

class KernelProgram {
    // Единственная ответственность: управление OpenCL программами
    cl_kernel GetOrCreateKernel(const std::string& name);
};
```

**Плохо (SRP нарушен):**
```cpp
class BadManager {
    void CreateBuffer();      // Управление памятью
    void CompileProgram();    // Компиляция программ
    void PrintStatistics();   // Вывод статистики
    void SaveToFile();        // Работа с файлами
};
```

### 2. Open-Closed Principle (OCP) - Принцип открытости-закрытости

**Определение:** Классы должны быть открыты для расширения, но закрыты для модификации.

**В проекте:**
```cpp
// Базовый класс открыт для расширения
class GPUMemoryBuffer {
public:
    virtual ~GPUMemoryBuffer() = default;
    virtual void WriteToGPU(const std::vector<std::complex<float>>& data) = 0;
};

// Расширение без модификации базового класса
class OwningGPUMemoryBuffer : public GPUMemoryBuffer {
    // Специфичная реализация
};

class NonOwningGPUMemoryBuffer : public GPUMemoryBuffer {
    // Другая реализация
};
```

### 3. Liskov Substitution Principle (LSP) - Принцип подстановки Барбары Лисков

**Определение:** Объекты подкласса должны быть заменяемыми на объекты базового класса без нарушения корректности программы.

**В проекте:**
```cpp
// Все буферы могут использоваться одинаково
void ProcessBuffer(GPUMemoryBuffer* buffer) {
    buffer->WriteToGPU(data);
    auto result = buffer->ReadFromGPU();
}

// Работает с любым типом буфера
auto owning = std::make_unique<OwningGPUMemoryBuffer>(...);
auto non_owning = std::make_unique<NonOwningGPUMemoryBuffer>(...);

ProcessBuffer(owning.get());    // OK
ProcessBuffer(non_owning.get()); // OK
```

### 4. Interface Segregation Principle (ISP) - Принцип разделения интерфейсов

**Определение:** Клиенты не должны зависеть от интерфейсов, которые они не используют.

**В проекте:**
```cpp
// Разделение интерфейсов вместо одного большого

// Специфичный интерфейс для чтения
class ReadableBuffer {
public:
    virtual std::vector<std::complex<float>> ReadFromGPU() = 0;
};

// Специфичный интерфейс для записи
class WritableBuffer {
public:
    virtual void WriteToGPU(const std::vector<std::complex<float>>& data) = 0;
};

// Специфичный интерфейс для информации
class BufferInfo {
public:
    virtual size_t GetSizeBytes() const = 0;
    virtual MemoryType GetMemoryType() const = 0;
};

// Класс реализует только нужные интерфейсы
class GPUMemoryBuffer : public ReadableBuffer,
                       public WritableBuffer,
                       public BufferInfo {
    // Реализация всех интерфейсов
};
```

### 5. Dependency Inversion Principle (DIP) - Принцип инверсии зависимостей

**Определение:** Модули верхнего уровня не должны зависеть от модулей нижнего уровня. Оба должны зависеть от абстракций.

**В проекте:**
```cpp
// Высокоуровневый модуль зависит от абстракции
class GeneratorGPU {
public:
    GeneratorGPU(const LFMParameters& params);
private:
    OpenCLComputeEngine* engine_;  // Зависит от абстракции, не от реализации
};

// Низкоуровневый модуль реализует абстракцию
class OpenCLComputeEngine {
public:
    virtual std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t, MemoryType) = 0;
};
```

## Паттерны GRASP

### 1. Information Expert (Информационный эксперт)

**Определение:** Ответственность должна быть назначена тому, кто обладает необходимой информацией.

**В проекте:**
```cpp
class LFMParameters {
private:
    float f_start_, f_stop_, sample_rate_, duration_;
public:
    // LFMParameters - эксперт по расчету параметров сигнала
    float GetChirpRate() const noexcept {
        return (f_stop_ - f_start_) / duration_;
    }

    size_t GetNumSamples() const noexcept {
        return static_cast<size_t>(duration_ * sample_rate_);
    }
};
```

### 2. Creator (Создатель)

**Определение:** Определение, кто должен создавать экземпляры других классов.

**В проекте:**
```cpp
class OpenCLComputeEngine {
public:
    // OpenCLComputeEngine создает GPUMemoryBuffer,
    // потому что:
    // - Содержит GPUMemoryBuffer
    // - Имеет информацию для инициализации
    // - Записывает в GPUMemoryBuffer
    std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t num_elements, MemoryType type);
};
```

### 3. Controller (Контроллер)

**Определение:** Назначение ответственности за обработку системных событий.

**В проекте:**
```cpp
class OpenCLComputeEngine {
public:
    // Контроллер - фасад для всей системы OpenCL
    void ExecuteKernel(cl_kernel kernel, const std::vector<cl_mem>& buffers, ...);
    void Finish();
    std::string GetStatistics() const;
};
```

### 4. Low Coupling (Слабое зацепление)

**Определение:** Минимизация зависимостей между классами.

**В проекте:**
```cpp
// Слабое зацепление через интерфейсы
class GeneratorGPU {
private:
    OpenCLComputeEngine* engine_;  // Зависит только от интерфейса
};

// Легко заменить реализацию
class MockOpenCLComputeEngine : public OpenCLComputeEngine {
    // Для тестирования
};
```

### 5. High Cohesion (Высокая связность)

**Определение:** Класс должен иметь четко определенную ответственность.

**В проекте:**
```cpp
class KernelProgram {
    // Высокая связность: все методы связаны с управлением программами
    cl_program GetProgram() const;
    cl_kernel GetOrCreateKernel(const std::string& name);
    bool HasKernel(const std::string& name) const;
};
```

### 6. Polymorphism (Полиморфизм)

**Определение:** Использование полиморфизма для обработки альтернатив.

**В проекте:**
```cpp
// Полиморфизм для разных типов буферов
enum class MemoryType {
    GPU_READ_ONLY,
    GPU_WRITE_ONLY,
    GPU_READ_WRITE
};

class GPUMemoryBuffer {
public:
    // Полиморфное поведение в зависимости от типа
    GPUMemoryBuffer(..., MemoryType type);
};
```

### 7. Pure Fabrication (Чистая выдумка)

**Определение:** Создание классов для достижения низкого зацепления и высокой связности.

**В проекте:**
```cpp
class KernelProgramCache {
    // "Выдуманный" класс для кэширования
    // Не является частью предметной области
    // Но улучшает производительность
    static std::shared_ptr<KernelProgram> GetOrCompile(const std::string& source);
};
```

### 8. Indirection (Посредник)

**Определение:** Использование промежуточного объекта для обеспечения связи.

**В проекте:**
```cpp
class OpenCLComputeEngine {
    // Посредник между пользователем и низкоуровневыми компонентами
private:
    OpenCLCore* core_;
    CommandQueuePool* queues_;
    KernelProgramCache* cache_;
};
```

### 9. Protected Variations (Защищенные вариации)

**Определение:** Защита от изменений путем инкапсуляции точек вариаций.

**В проекте:**
```cpp
// Защита от изменений типов устройств
enum class DeviceType {
    GPU,
    CPU
};

class OpenCLCore {
public:
    static void Initialize(DeviceType device_type = DeviceType::GPU);
    // Внутренняя реализация скрыта от изменений
};
```

## Паттерны GoF

### 1. Singleton (Одиночка)

**Определение:** Гарантирует, что у класса есть только один экземпляр.

**В проекте:**
```cpp
class OpenCLCore {
public:
    static OpenCLCore& GetInstance() {
        static OpenCLCore instance;  // C++11 thread-safe
        return instance;
    }

    static void Initialize(DeviceType device_type = DeviceType::GPU) {
        // Double-checked locking
        if (!initialized_) {
            std::unique_lock<std::mutex> lock(initialization_mutex_);
            if (!initialized_) {
                instance.InitializeOpenCL(device_type);
                initialized_ = true;
            }
        }
    }

private:
    OpenCLCore() = default;  // Приватный конструктор
    OpenCLCore(const OpenCLCore&) = delete;
    OpenCLCore& operator=(const OpenCLCore&) = delete;
};
```

**Использование в проекте:**
- `OpenCLCore` - единый контекст OpenCL
- `CommandQueuePool` - единый пул очередей
- `OpenCLComputeEngine` - главный фасад
- `GPUMemoryManager` - менеджер памяти

### 2. Factory Method (Фабричный метод)

**Определение:** Определяет интерфейс для создания объектов, но позволяет подклассам решать, какой класс инстанцировать.

**В проекте:**
```cpp
class OpenCLComputeEngine {
public:
    // Фабричный метод для создания буферов
    virtual std::unique_ptr<GPUMemoryBuffer> CreateBuffer(
        size_t num_elements,
        MemoryType type
    ) = 0;

    // Фабричный метод для создания буферов с данными
    virtual std::unique_ptr<GPUMemoryBuffer> CreateBufferWithData(
        const std::vector<std::complex<float>>& data,
        MemoryType type
    ) = 0;
};

// Реализация фабричного метода
std::unique_ptr<GPUMemoryBuffer> OpenCLComputeEngine::CreateBuffer(
    size_t num_elements,
    MemoryType type
) {
    return std::make_unique<GPUMemoryBuffer>(
        core_->GetContext(),
        queues_->GetNextQueue(),
        num_elements,
        type
    );
}
```

### 3. Abstract Factory (Абстрактная фабрика)

**Определение:** Предоставляет интерфейс для создания семейств связанных объектов без указания их конкретных классов.

**В проекте:**
```cpp
// Абстрактная фабрика для создания GPU ресурсов
class GPUResourceFactory {
public:
    virtual std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t, MemoryType) = 0;
    virtual std::shared_ptr<KernelProgram> CreateProgram(const std::string&) = 0;
    virtual cl_kernel CreateKernel(const std::shared_ptr<KernelProgram>&, const std::string&) = 0;
};

// Конкретная фабрика для OpenCL
class OpenCLResourceFactory : public GPUResourceFactory {
public:
    std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t num, MemoryType type) override {
        return engine_.CreateBuffer(num, type);
    }

    std::shared_ptr<KernelProgram> CreateProgram(const std::string& source) override {
        return engine_.LoadProgram(source);
    }

    cl_kernel CreateKernel(const std::shared_ptr<KernelProgram>& prog, const std::string& name) override {
        return engine_.GetKernel(prog, name);
    }

private:
    OpenCLComputeEngine& engine_;
};
```

### 4. Builder (Строитель)

**Определение:** Отделяет конструирование сложного объекта от его представления.

**В проекте:**
```cpp
class LFMParametersBuilder {
public:
    LFMParametersBuilder& SetFrequency(float start, float stop) {
        params_.f_start = start;
        params_.f_stop = stop;
        return *this;
    }

    LFMParametersBuilder& SetSampleRate(float rate) {
        params_.sample_rate = rate;
        return *this;
    }

    LFMParametersBuilder& SetDuration(float duration) {
        params_.duration = duration;
        return *this;
    }

    LFMParametersBuilder& SetBeamCount(size_t beams) {
        params_.num_beams = beams;
        return *this;
    }

    LFMParameters Build() {
        if (!params_.IsValid()) {
            throw std::invalid_argument("Invalid LFM parameters");
        }
        return params_;
    }

private:
    LFMParameters params_;
};

// Использование
LFMParameters params = LFMParametersBuilder()
    .SetFrequency(100.0f, 500.0f)
    .SetSampleRate(12e6f)
    .SetDuration(0.01f)
    .SetBeamCount(256)
    .Build();
```

### 5. Adapter (Адаптер)

**Определение:** Преобразует интерфейс одного класса в интерфейс другого.

**В проекте:**
```cpp
// Адаптер для работы с внешними буферами
class ExternalBufferAdapter : public GPUMemoryBuffer {
public:
    ExternalBufferAdapter(cl_mem external_buffer, size_t num_elements, MemoryType type)
        : GPUMemoryBuffer(context, queue, external_buffer, num_elements, type, false) {
        // Адаптирует внешний буфер к интерфейсу GPUMemoryBuffer
    }

    // GPUMemoryBuffer interface
    void WriteToGPU(const std::vector<std::complex<float>>& data) override {
        // Адаптированная реализация
    }

    std::vector<std::complex<float>> ReadFromGPU() override {
        // Адаптированная реализация
    }
};
```

### 6. Facade (Фасад)

**Определение:** Предоставляет унифицированный интерфейс к набору интерфейсов в подсистеме.

**В проекте:**
```cpp
class OpenCLComputeEngine {
    // ФАСАД для всей OpenCL подсистемы
private:
    OpenCLCore* core_;              // Низкоуровневый контекст
    CommandQueuePool* queues_;      // Управление очередями
    KernelProgramCache* cache_;     // Кэширование программ

public:
    // Простой интерфейс для сложной подсистемы
    std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t, MemoryType);
    void ExecuteKernel(cl_kernel, const std::vector<cl_mem>&, ...);
    std::string GetStatistics() const;
};
```

### 7. Strategy (Стратегия)

**Определение:** Определяет семейство алгоритмов, инкапсулирует каждый и делает их взаимозаменяемыми.

**В проекте:**
```cpp
// Стратегия чтения данных
class DataReadStrategy {
public:
    virtual std::vector<std::complex<float>> Read(GPUMemoryBuffer* buffer) = 0;
};

class BlockingReadStrategy : public DataReadStrategy {
public:
    std::vector<std::complex<float>> Read(GPUMemoryBuffer* buffer) override {
        return buffer->ReadFromGPU();  // Синхронное чтение
    }
};

class AsyncReadStrategy : public DataReadStrategy {
public:
    std::vector<std::complex<float>> Read(GPUMemoryBuffer* buffer) override {
        auto [data, event] = buffer->ReadFromGPUAsync();
        // Ожидание события...
        return data;
    }
};

// Использование стратегии
class DataReader {
public:
    DataReader(DataReadStrategy* strategy) : strategy_(strategy) {}

    std::vector<std::complex<float>> ReadData(GPUMemoryBuffer* buffer) {
        return strategy_->Read(buffer);
    }

private:
    DataReadStrategy* strategy_;
};
```

### 8. Observer (Наблюдатель)

**Определение:** Определяет зависимость один-ко-многим между объектами.

**В проекте:**
```cpp
// Наблюдатель за статистикой GPU
class GPUStatsObserver {
public:
    virtual void OnBufferCreated(size_t size_bytes) = 0;
    virtual void OnKernelExecuted() = 0;
    virtual void OnMemoryAllocated(size_t bytes) = 0;
};

class GPUStatsLogger : public GPUStatsObserver {
public:
    void OnBufferCreated(size_t size_bytes) override {
        std::cout << "Buffer created: " << size_bytes << " bytes\n";
    }

    void OnKernelExecuted() override {
        kernel_count_++;
        std::cout << "Kernel executed. Total: " << kernel_count_ << "\n";
    }

    void OnMemoryAllocated(size_t bytes) override {
        total_memory_ += bytes;
        std::cout << "Memory allocated. Total: " << total_memory_ << " bytes\n";
    }

private:
    size_t kernel_count_ = 0;
    size_t total_memory_ = 0;
};

class OpenCLComputeEngine {
public:
    void AddObserver(GPUStatsObserver* observer) {
        observers_.push_back(observer);
    }

    std::unique_ptr<GPUMemoryBuffer> CreateBuffer(size_t num_elements, MemoryType type) {
        auto buffer = CreateBufferImpl(num_elements, type);

        // Уведомление наблюдателей
        for (auto observer : observers_) {
            observer->OnBufferCreated(buffer->GetSizeBytes());
            observer->OnMemoryAllocated(buffer->GetSizeBytes());
        }

        return buffer;
    }

private:
    std::vector<GPUStatsObserver*> observers_;
};
```

### 9. Command (Команда)

**Определение:** Инкапсулирует запрос как объект.

**В проекте:**
```cpp
// Команда для выполнения kernel
class KernelCommand {
public:
    virtual void Execute() = 0;
    virtual void Undo() = 0;  // Для отмены, если поддерживается
};

class ExecuteLFMKernelCommand : public KernelCommand {
public:
    ExecuteLFMKernelCommand(
        OpenCLComputeEngine* engine,
        cl_kernel kernel,
        const std::vector<cl_mem>& buffers,
        const std::array<size_t, 3>& global_size,
        const std::array<size_t, 3>& local_size
    ) : engine_(engine), kernel_(kernel), buffers_(buffers),
        global_size_(global_size), local_size_(local_size) {}

    void Execute() override {
        engine_->ExecuteKernel(kernel_, buffers_, global_size_, local_size_);
    }

    void Undo() override {
        // Отмена выполнения kernel может быть сложной или невозможной
        // Можно сохранить предыдущее состояние для восстановления
    }

private:
    OpenCLComputeEngine* engine_;
    cl_kernel kernel_;
    std::vector<cl_mem> buffers_;
    std::array<size_t, 3> global_size_;
    std::array<size_t, 3> local_size_;
};

// Менеджер команд
class KernelCommandManager {
public:
    void ExecuteCommand(std::unique_ptr<KernelCommand> command) {
        command->Execute();
        executed_commands_.push_back(std::move(command));
    }

    void UndoLastCommand() {
        if (!executed_commands_.empty()) {
            executed_commands_.back()->Undo();
            executed_commands_.pop_back();
        }
    }

private:
    std::vector<std::unique_ptr<KernelCommand>> executed_commands_;
};
```

### 10. Template Method (Шаблонный метод)

**Определение:** Определяет скелет алгоритма в суперклассе, позволяя подклассам переопределить определенные шаги.

**В проекте:**
```cpp
// Шаблонный метод для инициализации OpenCL компонентов
class OpenCLComponent {
public:
    void Initialize() {
        ValidatePrerequisites();     // Шаг, который могут переопределить
        AllocateResources();         // Шаг, который могут переопределить
        InitializeInternalState();   // Шаг, который могут переопределить
        RegisterComponent();         // Общий финальный шаг
    }

protected:
    virtual void ValidatePrerequisites() {}
    virtual void AllocateResources() = 0;
    virtual void InitializeInternalState() = 0;

private:
    void RegisterComponent() {
        // Общая регистрация в системе
        std::cout << "Component registered\n";
    }
};

class OpenCLCore : public OpenCLComponent {
protected:
    void AllocateResources() override {
        // Выделение контекста, устройства
    }

    void InitializeInternalState() override {
        // Инициализация внутреннего состояния
    }
};

class CommandQueuePool : public OpenCLComponent {
protected:
    void ValidatePrerequisites() override {
        if (!OpenCLCore::IsInitialized()) {
            throw std::runtime_error("OpenCLCore must be initialized first");
        }
    }

    void AllocateResources() override {
        // Создание пула очередей
    }

    void InitializeInternalState() override {
        // Инициализация счетчиков использования
    }
};
```

## Заключение

Проект LCH-Farrow01 демонстрирует применение современных принципов и паттернов ООП:

- **SOLID принципы** обеспечивают поддерживаемость и расширяемость
- **GRASP паттерны** помогают правильно распределить ответственность
- **GoF паттерны** решают типичные проблемы проектирования
- **ООП принципы** формируют основу архитектуры

Эти принципы и паттерны позволяют создавать гибкий, расширяемый и поддерживаемый код для высокопроизводительных GPU вычислений.