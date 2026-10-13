// Batch API fk_filter_gathers (basalt/fk.hpp): gathers of one shape are filtered in parallel on a
// small worker pool. Contract: docs/design/FK_PIPELINE.md, and the comments in basalt/fk.hpp.
//
// Pool workers block on a condition variable while they have no work, and a worker that stays idle
// for kIdleTimeout exits. When a worker exits, its thread_local work buffers (about 27 MB per
// worker at 1001 x 1500) are freed with it, and malloc_trim returns the heap memory they used.
// The next batch call starts workers on demand. Between calls the pool therefore uses no CPU and
// holds no workspace memory after kIdleTimeout.

#include "basalt/fk.hpp"
#include "fk_internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

// A sanitizer replaces malloc, so glibc's arena state is not the one malloc_trim expects.
// Under AddressSanitizer the call crashes a worker at thread exit (a SEGV in the TLS destructors,
// or a glibc arena assertion), so it is skipped whenever a sanitizer is active.
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define BASALT_FK_SANITIZED 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define BASALT_FK_SANITIZED 1
#endif

namespace basalt {

namespace {

// How long a pool worker may wait without work before it exits.
constexpr std::chrono::seconds kIdleTimeout{2};

// Joins every thread in the list when it goes out of scope, so unwinding never destroys a joinable
// std::thread (which would call std::terminate).
class ThreadJoiner {
public:
    explicit ThreadJoiner(std::vector<std::thread>& threads) : threads_(threads) {}
    ThreadJoiner(const ThreadJoiner&) = delete;
    ThreadJoiner& operator=(const ThreadJoiner&) = delete;
    ~ThreadJoiner() {
        for (std::thread& t : threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

private:
    std::vector<std::thread>& threads_;
};

// Returns the malloc memory a worker thread used once the thread has ended. An instance is a
// thread_local declared at the start of the worker, so its destructor runs after the destructors of
// the work buffers that the worker's tasks construct later (thread_local objects are destroyed in
// reverse order of construction). It is needed because glibc raises its mmap threshold once large
// blocks have been freed. A 13 MB workspace buffer is then served from the heap, and when the
// thread exits the heap keeps its pages until they are trimmed. Skipped under a sanitizer (above).
struct ArenaRelease {
    ~ArenaRelease() {
#if defined(__GLIBC__) && !defined(BASALT_FK_SANITIZED)
        malloc_trim(0);
#endif
    }
};

// A small pool that runs queued tasks on worker threads. Workers are created on demand, block on a
// condition variable when idle and exit after kIdleTimeout without work. It is process-wide and is
// shared by all batch calls. Tasks must not throw.
class WorkerPool {
public:
    static WorkerPool& instance() {
        static WorkerPool pool;
        return pool;
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    ~WorkerPool() {
        std::map<uint64_t, std::thread> threads;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            threads.swap(threads_);
        }
        cv_.notify_all();
        for (auto& entry : threads) {
            if (entry.second.joinable()) {
                entry.second.join();
            }
        }
    }

    // Queues `task` for a worker. A worker is started when no idle worker can take it. If no
    // worker can be started the task is dropped, so callers must not depend on it running: the
    // batch calls below finish their own work on the calling thread.
    void submit(std::function<void()> task) {
        std::vector<std::thread> finished;
        ThreadJoiner joiner(finished); // declared before the lock, so it joins after the unlock
        std::lock_guard<std::mutex> lock(mutex_);
        collect_finished_locked(finished);
        queue_.push_back(std::move(task));
        if (queue_.size() <= idle_) {
            cv_.notify_one();
        } else if (!start_worker_locked()) {
            queue_.pop_back();
        }
    }

private:
    WorkerPool() = default;

    // Moves the threads of workers that have exited into `out`, which the caller joins. `out` is
    // reserved first, so no move can throw part way.
    void collect_finished_locked(std::vector<std::thread>& out) {
        out.reserve(out.size() + finished_.size());
        for (const uint64_t id : finished_) {
            const auto it = threads_.find(id);
            if (it != threads_.end()) {
                out.push_back(std::move(it->second));
                threads_.erase(it);
            }
        }
        finished_.clear();
    }

    // Starts one worker. Returns false, and leaves no thread behind, when that is not possible.
    bool start_worker_locked() {
        const uint64_t id = next_id_++;
        try {
            // The map node is allocated before any thread starts, so a failure here leaves nothing.
            const auto slot = threads_.emplace(id, std::thread()).first;
            try {
                slot->second = std::thread([this, id] { worker_main(id); });
            } catch (...) {
                threads_.erase(slot);
                throw;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    void worker_main(uint64_t id) {
        thread_local ArenaRelease arena_release;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            if (!queue_.empty()) {
                std::function<void()> task = std::move(queue_.front());
                queue_.pop_front();
                lock.unlock();
                try {
                    task();
                } catch (...) {
                    // Tasks do not throw (see submit). Swallowing keeps the worker alive anyway.
                }
                task = nullptr;
                lock.lock();
                continue;
            }
            if (stop_) {
                break;
            }
            ++idle_;
            const bool woke = cv_.wait_for(lock, kIdleTimeout,
                                           [this] { return !queue_.empty() || stop_; });
            --idle_;
            if (!woke) {
                break; // kIdleTimeout without work: exit and release the thread's memory
            }
        }
        try {
            finished_.push_back(id);
        } catch (...) {
            // The thread stays in threads_ and is joined when the pool is destroyed.
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> queue_;
    std::map<uint64_t, std::thread> threads_;  // live and exited, not yet joined
    std::vector<uint64_t> finished_;           // ids of workers that have exited
    size_t idle_ = 0;                          // workers waiting for work
    uint64_t next_id_ = 0;
    bool stop_ = false;
};

// State of one run_indexed call. Pool tasks hold it by shared_ptr. Every index is claimed and
// accounted for by the time the call returns, so a task that starts later claims nothing and never
// calls the work function, which may refer to the caller's stack.
struct Job {
    Job(std::function<void(size_t)> work, size_t n)
        : fn(std::move(work)), count(n), unfinished(n) {}

    std::function<void(size_t)> fn;
    const size_t count;
    std::atomic<size_t> next{0};       // next index to claim
    std::atomic<size_t> unfinished;    // claimed indices not yet accounted for
    std::atomic<bool> failed{false};   // set after the first exception; later indices are skipped
    std::mutex mutex;
    std::condition_variable done;
    std::exception_ptr error;          // first exception, guarded by mutex
};

// Claims indices of `job` until none are left, running fn on each unless the job has failed.
// Each claimed index is accounted for exactly once, skipped or not, so the owner's wait ends.
void drain(Job& job) {
    for (;;) {
        const size_t i = job.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= job.count) {
            return;
        }
        if (!job.failed.load(std::memory_order_acquire)) {
            try {
                job.fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(job.mutex);
                if (!job.error) {
                    job.error = std::current_exception();
                }
                job.failed.store(true, std::memory_order_release);
            }
        }
        if (job.unfinished.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(job.mutex);
            job.done.notify_all();
        }
    }
}

// The floating-point control register (MXCSR: rounding mode, FTZ, DAZ). It is per thread, and a new
// thread starts with the register of the thread that creates it. Without SSE there is nothing to
// copy.
unsigned current_mxcsr() {
#if defined(__SSE__)
    return _mm_getcsr();
#else
    return 0;
#endif
}

void set_mxcsr(unsigned value) {
#if defined(__SSE__)
    _mm_setcsr(value);
#else
    (void)value;
#endif
}

// Sets the calling thread's floating-point control register to `value` for the life of the object,
// and restores the previous value on exit.
class MxcsrScope {
public:
    explicit MxcsrScope(unsigned value) : saved_(current_mxcsr()) {
        set_mxcsr(value);
    }
    MxcsrScope(const MxcsrScope&) = delete;
    MxcsrScope& operator=(const MxcsrScope&) = delete;
    ~MxcsrScope() {
        set_mxcsr(saved_);
    }

private:
    unsigned saved_;
};

// Runs fn(i) for every i in [0, count), on the calling thread and up to `workers - 1` pool
// workers. Returns when all indices are done. Rethrows the first exception that fn threw; once one
// is thrown the remaining indices are skipped.
//
// A pool worker runs under the floating-point mode of the thread that created it, which need not
// be the caller's. The results must not depend on that, so every task runs under the caller's mode
// (FK_PIPELINE.md, section 4, bit-identity for every params.threads).
void run_indexed(size_t count, size_t workers, std::function<void(size_t)> fn) {
    if (workers <= 1 || count <= 1) {
        for (size_t i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }
    const auto job = std::make_shared<Job>(std::move(fn), count);
    const unsigned caller_mxcsr = current_mxcsr();
    for (size_t w = 1; w < workers; ++w) {
        try {
            WorkerPool::instance().submit([job, caller_mxcsr] {
                MxcsrScope mode(caller_mxcsr);
                drain(*job);
            });
        } catch (...) {
            break; // out of memory while queueing: the calling thread does the rest
        }
    }
    drain(*job);

    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lock(job->mutex);
        job->done.wait(lock, [&job] {
            return job->unfinished.load(std::memory_order_acquire) == 0;
        });
        error = job->error;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

size_t resolve_threads(size_t requested) {
    if (requested != 0) {
        return requested;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<size_t>(hc);
}

// True when the n floats at `a` and the n floats at `b` share at least one byte. Pointers are
// compared as integers, since relational comparison of unrelated pointers is unspecified.
bool ranges_overlap(const float* a, const float* b, size_t n) {
    const auto first_a = reinterpret_cast<uintptr_t>(a);
    const auto first_b = reinterpret_cast<uintptr_t>(b);
    const uintptr_t bytes = static_cast<uintptr_t>(n * sizeof(float));
    return first_a < first_b + bytes && first_b < first_a + bytes;
}

// True when some of the n samples at `src` is NaN or infinite. A float is non-finite exactly when
// its exponent field is all ones, so the test is on the bit pattern. It is branch-free, which lets
// the compiler vectorise the scan. The result is the same as std::isfinite on every sample.
bool any_nonfinite(const float* src, size_t n) {
    uint32_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, src + i, sizeof bits);
        bad |= static_cast<uint32_t>((bits & 0x7F800000u) == 0x7F800000u);
    }
    return bad != 0;
}

} // namespace

void fk_filter_gathers(const float* in, float* out, size_t ngathers,
                       size_t ntraces, size_t nsamples, const FKMuteParams& params) {
    // The checks run in the order of fk_filter_gather (parameters, dimensions, pointers, padded
    // geometry), so the same invalid arguments raise the same error. The batch-only overflow checks
    // follow.
    fk_internal::validate_arguments(params, ntraces, nsamples);
    if (in == nullptr || out == nullptr) {
        throw std::invalid_argument("fk: null input or output pointer");
    }
    const fk_internal::Geometry geo = fk_internal::check_geometry(params, ntraces, nsamples);
    const size_t max = std::numeric_limits<size_t>::max();
    if (nsamples > max / ntraces) {
        throw std::invalid_argument("fk: batch size overflows size_t");
    }
    const size_t gather_len = ntraces * nsamples;
    if (ngathers > max / gather_len) {
        throw std::invalid_argument("fk: batch size overflows size_t");
    }
    if (ngathers == 0) {
        return;
    }
    if (ngathers == 1) {
        // Exactly fk_filter_gather, including its intra-gather threads.
        fk_filter_gather(in, out, ntraces, nsamples, params);
        return;
    }

    // Filtering gather g writes out[g], which can be input that a later gather still reads. A
    // partial overlap therefore corrupts the output, so it is refused before anything is written.
    // (One gather reads all of its input before it writes, which is why ngathers == 1 is allowed.)
    if (in != out && ranges_overlap(in, out, ngathers * gather_len)) {
        throw std::invalid_argument("fk: in and out partially overlap");
    }

    const size_t nthreads = resolve_threads(params.threads);
    const size_t workers = std::min(nthreads, ngathers);

    // Every gather is scanned before any output is written. Filtering gather g writes out[g], and
    // with aliased buffers that overwrites in[g], so a non-finite sample in a later gather would
    // otherwise be found after earlier outputs were already written. The scan reads the input once
    // (a memory-bound pass that is much cheaper than the transforms it precedes), and it runs in
    // parallel over the gathers.
    std::atomic<bool> nonfinite{false};
    run_indexed(ngathers, workers, [&](size_t g) {
        if (any_nonfinite(in + g * gather_len, gather_len)) {
            nonfinite.store(true, std::memory_order_relaxed);
        }
    });
    if (nonfinite.load(std::memory_order_relaxed)) {
        throw std::invalid_argument("fk: input contains NaN or infinite samples");
    }

    // The weight table and column classes depend only on the parameters and the gather shape, so
    // one table serves every gather. With several workers it is built once here, before they start,
    // and only read after. With one worker every gather runs on the calling thread, which reuses
    // its cached table (the same table fk_filter_gather would use), so no table is built per call.
    fk_internal::WeightTable table;
    const bool shared_table = workers > 1;
    if (shared_table) {
        fk_internal::build_weight_table(params, geo, nthreads, table);
    }

    // Each gather runs on one thread. The numerics are those of fk_filter_gather with threads = 1,
    // so the result does not depend on the worker count. The scan above has checked every sample,
    // so each gather skips the test in its own copy-in pass. That test costs about as much as the
    // scan, so the scan adds little to the call (see the cost note in basalt/fk.hpp).
    run_indexed(ngathers, workers, [&](size_t g) {
        fk_internal::filter_gather_impl(in + g * gather_len, out + g * gather_len, ntraces,
                                        nsamples, params, 1, shared_table ? &table : nullptr,
                                        true);
    });
}

} // namespace basalt
