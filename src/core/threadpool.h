#ifndef JNET_CORE_THREADPOOL_H
#define JNET_CORE_THREADPOOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>

namespace JNet {

class ThreadPool {
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    // Submit a task to the thread pool
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

    // Get number of threads
    size_t size() const { return workers.size(); }

    // Disable copy/move
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};

// Global thread pool instance
extern ThreadPool& getGlobalThreadPool();

// Utility function to parallelize loops
template<typename Func>
void parallel_for(size_t start, size_t end, size_t num_threads, Func&& func) {
    if (end <= start) return;
    
    size_t total_work = end - start;
    if (total_work <= num_threads || num_threads <= 1) {
        // Not worth parallelizing or single thread requested
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    ThreadPool& pool = getGlobalThreadPool();
    std::vector<std::future<void>> futures;
    
    size_t work_per_thread = total_work / num_threads;
    size_t remaining_work = total_work % num_threads;
    
    size_t current_start = start;
    
    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        size_t thread_work = work_per_thread + (thread_id < remaining_work ? 1 : 0);
        size_t current_end = current_start + thread_work;
        
        futures.emplace_back(
            pool.enqueue([func, current_start, current_end]() {
                for (size_t i = current_start; i < current_end; ++i) {
                    func(i);
                }
            })
        );
        
        current_start = current_end;
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
}

// Template implementation
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        if(stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

}

#endif // JNET_CORE_THREADPOOL_H
