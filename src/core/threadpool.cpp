#include "threadpool.h"
#include <iostream>

namespace JNet {

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    // Ensure we have at least 1 thread, but don't exceed hardware capacity
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1; // Fallback
    }
    
    // Limit to reasonable number of threads to avoid overhead
    size_t max_threads = std::min(num_threads, static_cast<size_t>(std::thread::hardware_concurrency()) * 2);
    
    for (size_t i = 0; i < max_threads; ++i) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }
                    
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    
    condition.notify_all();
    
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

// Global thread pool instance
ThreadPool& getGlobalThreadPool() {
    static ThreadPool instance(std::thread::hardware_concurrency());
    return instance;
}

}
