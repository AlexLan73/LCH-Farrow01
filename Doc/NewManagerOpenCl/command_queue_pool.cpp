#include "command_queue_pool.hpp"
#include <iostream>
#include <thread>
#include <sstream>
#include <iomanip>

namespace gpu {

// ════════════════════════════════════════════════════════════════════════════
// Static инициализация
// ════════════════════════════════════════════════════════════════════════════

std::unique_ptr<CommandQueuePool> CommandQueuePool::instance_ = nullptr;
bool CommandQueuePool::initialized_ = false;
std::mutex CommandQueuePool::initialization_mutex_;

CommandQueuePool::CommandQueuePool() {
}

CommandQueuePool::~CommandQueuePool() {
    ReleaseQueues();
}

void CommandQueuePool::Initialize(size_t num_queues) {
    std::lock_guard<std::mutex> lock(initialization_mutex_);

    if (initialized_) {
        std::cerr << "[WARNING] CommandQueuePool already initialized\n";
        return;
    }

    // Если num_queues = 0, используем количество ядер
    if (num_queues == 0) {
        num_queues = std::thread::hardware_concurrency();
        if (num_queues == 0) num_queues = 4;  // Fallback
    }

    instance_ = std::make_unique<CommandQueuePool>();
    instance_->CreateQueues(num_queues);
    initialized_ = true;

    std::cout << "[OK] CommandQueuePool initialized with " << num_queues
              << " queues\n";
}

bool CommandQueuePool::IsInitialized() {
    return initialized_;
}

void CommandQueuePool::Cleanup() {
    std::lock_guard<std::mutex> lock(initialization_mutex_);

    if (initialized_) {
        instance_->ReleaseQueues();
        instance_.reset();
        initialized_ = false;
        std::cout << "[OK] CommandQueuePool cleaned up\n";
    }
}

void CommandQueuePool::CreateQueues(size_t num_queues) {
    auto& core = OpenCLCore::GetInstance();
    cl_context context = core.GetContext();
    cl_device_id device = core.GetDevice();

    cl_int err;

    for (size_t i = 0; i < num_queues; ++i) {
        // Создать очередь с поддержкой асинхронного выполнения
        cl_command_queue queue = clCreateCommandQueue(
            context, device,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,  // Асинхронное выполнение
            &err);

        CheckCLError(err, "clCreateCommandQueue");

        queues_.push_back(queue);
        queue_usage_count_.emplace_back(0);

        std::cout << "  Queue #" << i << " created\n";
    }
}

void CommandQueuePool::ReleaseQueues() {
    for (auto queue : queues_) {
        if (queue) {
            clReleaseCommandQueue(queue);
        }
    }
    queues_.clear();
    queue_usage_count_.clear();
}

cl_command_queue CommandQueuePool::GetNextQueue() {
    if (!initialized_ || queues_.empty()) {
        throw std::runtime_error("CommandQueuePool not initialized");
    }

    size_t index = current_index_.fetch_add(1) % queues_.size();
    queue_usage_count_[index]++;
    return queues_[index];
}

cl_command_queue CommandQueuePool::GetQueue(size_t index) {
    if (!initialized_ || index >= queues_.size()) {
        throw std::out_of_range("Queue index out of range");
    }
    queue_usage_count_[index]++;
    return queues_[index];
}

cl_command_queue CommandQueuePool::GetRandomQueue() {
    if (!initialized_ || queues_.empty()) {
        throw std::runtime_error("CommandQueuePool not initialized");
    }

    size_t index = std::rand() % queues_.size();
    queue_usage_count_[index]++;
    return queues_[index];
}

cl_command_queue CommandQueuePool::GetCurrentQueue() {
    // Получить очередь для текущего потока (thread-local)
    return GetNextQueue();  // Упрощённая реализация
}

void CommandQueuePool::FinishAll() {
    std::lock_guard<std::mutex> lock(instance_->pool_mutex_);

    for (auto queue : instance_->queues_) {
        if (queue) {
            cl_int err = clFinish(queue);
            CheckCLError(err, "clFinish");
        }
    }
}

void CommandQueuePool::FinishQueue(size_t index) {
    if (index >= instance_->queues_.size()) {
        throw std::out_of_range("Queue index out of range");
    }

    cl_int err = clFinish(instance_->queues_[index]);
    CheckCLError(err, "clFinish");
}

void CommandQueuePool::FlushAll() {
    std::lock_guard<std::mutex> lock(instance_->pool_mutex_);

    for (auto queue : instance_->queues_) {
        if (queue) {
            cl_int err = clFlush(queue);
            CheckCLError(err, "clFlush");
        }
    }
}

size_t CommandQueuePool::GetPoolSize() {
    if (!initialized_) return 0;
    return instance_->queues_.size();
}

size_t CommandQueuePool::GetCurrentQueueIndex() {
    if (!initialized_) return 0;
    return instance_->current_index_ % instance_->queues_.size();
}

std::string CommandQueuePool::GetStatistics() {
    if (!initialized_ || !instance_) {
        return "CommandQueuePool not initialized\n";
    }

    std::ostringstream oss;
    oss << "\nCommandQueuePool Statistics:\n";
    oss << " Total queues: " << instance_->queues_.size() << "\n";
    oss << " Load distribution:\n";

    for (size_t i = 0; i < instance_->queue_usage_count_.size(); ++i) {
        oss << "  Queue #" << i << ": "
            << instance_->queue_usage_count_[i] << " uses\n";
    }

    return oss.str();
}

}  // namespace gpu
