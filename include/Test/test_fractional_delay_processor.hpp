/**
 * @file test_fractional_delay_processor.hpp
 * @brief Тесты для FractionalDelayProcessor
 * 
 * ============================================================================
 * ТЕСТОВЫЕ СЦЕНАРИИ:
 * ============================================================================
 * 1. Тест инициализации - создание процессора
 * 2. Тест с нулевой задержкой - проверка что данные не меняются
 * 3. Тест с целой задержкой - сдвиг на целое число отсчётов
 * 4. Тест с дробной задержкой - интерполяция Лагранжа
 * 5. Тест batch обработки - все лучи с разными задержками
 * 6. Тест производительности - замер времени GPU
 * 
 * @author Кодо (AI Assistant)
 * @date 2026-01-21
 */

#ifndef TEST_FRACTIONAL_DELAY_PROCESSOR_HPP
#define TEST_FRACTIONAL_DELAY_PROCESSOR_HPP

#include <string>
#include <vector>
#include <functional>

namespace test {

// ============================================================================
// РЕЗУЛЬТАТ ТЕСТА
// ============================================================================

struct TestResult {
    std::string test_name;
    bool passed;
    std::string message;
    double execution_time_ms;
    
    TestResult() : passed(false), execution_time_ms(0.0) {}
    
    void Print() const;
};

// ============================================================================
// КЛАСС ТЕСТИРОВАНИЯ FractionalDelayProcessor
// ============================================================================

/**
 * @class TestFractionalDelayProcessor
 * @brief Набор тестов для проверки FractionalDelayProcessor
 * 
 * ИСПОЛЬЗОВАНИЕ:
 * @code
 * TestFractionalDelayProcessor tests;
 * tests.RunAllTests();
 * tests.PrintSummary();
 * @endcode
 */
class TestFractionalDelayProcessor {
public:
    TestFractionalDelayProcessor();
    ~TestFractionalDelayProcessor();
    
    // ========================================================================
    // ЗАПУСК ТЕСТОВ
    // ========================================================================
    
    /// Запустить все тесты
    void RunAllTests();
    
    /// Запустить один тест по имени
    TestResult RunTest(const std::string& test_name);
    
    /// Вывести итоговую статистику
    void PrintSummary() const;
    
    // ========================================================================
    // ОТДЕЛЬНЫЕ ТЕСТЫ
    // ========================================================================
    
    /// Тест создания и инициализации процессора
    TestResult TestInitialization();
    
    /// Тест с нулевой задержкой (данные не должны меняться)
    TestResult TestZeroDelay();
    
    /// Тест с целой задержкой (сдвиг данных)
    TestResult TestIntegerDelay();
    
    /// Тест с дробной задержкой (интерполяция)
    TestResult TestFractionalDelay();
    
    /// Тест batch обработки (все лучи с разными задержками)
    TestResult TestBatchProcessing();
    
    /// Тест производительности
    TestResult TestPerformance();
    
    /// Тест с данными от GeneratorGPU
    TestResult TestWithGeneratorGPU();
    
    /// Тест граничных случаев
    TestResult TestEdgeCases();
    
    // ========================================================================
    // ПОЛУЧЕНИЕ РЕЗУЛЬТАТОВ
    // ========================================================================
    
    /// Получить все результаты тестов
    const std::vector<TestResult>& GetResults() const { return results_; }
    
    /// Все тесты прошли?
    bool AllPassed() const;
    
    /// Количество пройденных тестов
    size_t PassedCount() const;
    
    /// Общее количество тестов
    size_t TotalCount() const { return results_.size(); }
    
private:
    std::vector<TestResult> results_;
    
    /// Вспомогательный метод для проверки результатов
    bool CompareBuffers(
        const std::vector<std::complex<float>>& expected,
        const std::vector<std::complex<float>>& actual,
        float tolerance = 1e-4f
    );
};

} // namespace test

#endif // TEST_FRACTIONAL_DELAY_PROCESSOR_HPP

