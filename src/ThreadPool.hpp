#pragma once
#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <stop_token>
#include <queue>

class ThreadPool {
  public:
    explicit ThreadPool(size_t thread_count) : stop_src_() {
      if (thread_count == 0) {
        thread_count = 1;
      }

      workers_.reserve(thread_count);

      for (size_t i = 0; i < thread_count; ++i) {
        workers_.emplace_back([this](std::stop_token st) {
          worker_loop(st);
        });
      }
    }

    ~ThreadPool() {
      stop();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Submit a task, returns std::future<R>
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) {
      using R = std::invoke_result_t<F, Args...>;

      auto task = std::make_shared<std::packaged_task<R()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
      );

      std::future<R> future = task->get_future();

      {
        std::unique_lock lock(mutex_);
        tasks_.emplace([task]() { (*task)(); });
      }
      cv_.notify_one();
      return future;
    }

    // Stop all workers gracefully
    void stop() {
      if (!stop_src_.stop_possible() || stop_src_.stop_requested()) {
        return;
      }

      stop_src_.request_stop();
      cv_.notify_all();
      workers_.clear();
    }

  private:
    void worker_loop(std::stop_token st) {
      while (!st.stop_requested()) {
        std::function<void()> task;

        {
          std::unique_lock lock(mutex_);
          cv_.wait(lock, st, [this] {
              return stop_src_.stop_requested() || !tasks_.empty();
              });

          if (stop_src_.stop_requested() && tasks_.empty())
            return;

          task = std::move(tasks_.front());
          tasks_.pop();
        }

        task();
      }
    }

  private:
    std::stop_source stop_src_;
    std::vector<std::jthread> workers_;

    std::mutex mutex_;
    std::condition_variable_any cv_;
    std::queue<std::function<void()>> tasks_;
};
