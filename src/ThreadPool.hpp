#pragma once

#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>

#if defined(__cpp_lib_jthread)
  #include <stop_token>
  #include <jthread>
  #define QUTILS_HAS_JTHREAD 1
#else
  #define QUTILS_HAS_JTHREAD 0
#endif

class ThreadPool {
public:
  explicit ThreadPool(size_t thread_count) {
    if (thread_count == 0) {
      thread_count = 1;
    }

    workers_.reserve(thread_count);

#if QUTILS_HAS_JTHREAD
    for (size_t i = 0; i < thread_count; ++i) {
      workers_.emplace_back([this](std::stop_token st) {
        worker_loop(st);
      });
    }
#else
    for (size_t i = 0; i < thread_count; ++i) {
      workers_.emplace_back([this] {
        worker_loop_legacy();
      });
    }
#endif
  }

  ~ThreadPool() {
    stop();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

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

  void stop() {
#if QUTILS_HAS_JTHREAD
    if (stop_src_.stop_requested())
      return;

    stop_src_.request_stop();
    cv_.notify_all();
    workers_.clear();
#else
    bool expected = false;
    if (!stop_flag_.compare_exchange_strong(expected, true))
      return;

    cv_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable())
        t.join();
    }
    workers_.clear();
#endif
  }

private:
#if QUTILS_HAS_JTHREAD

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

  std::stop_source stop_src_;
  std::vector<std::jthread> workers_;

#else

  void worker_loop_legacy() {
    while (true) {
      std::function<void()> task;

      {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] {
          return stop_flag_.load() || !tasks_.empty();
        });

        if (stop_flag_.load() && tasks_.empty())
          return;

        task = std::move(tasks_.front());
        tasks_.pop();
      }

      task();
    }
  }

  std::atomic<bool> stop_flag_{false};
  std::vector<std::thread> workers_;

#endif

  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<std::function<void()>> tasks_;
};

