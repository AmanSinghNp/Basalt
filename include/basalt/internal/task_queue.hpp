#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace basalt::internal {

namespace detail {

inline size_t round_up_power_of_two(size_t value) {
    size_t rounded = 1;
    while (rounded < value) {
        rounded <<= 1;
    }
    return rounded;
}

} // namespace detail

/// Bounded multi-producer / multi-consumer queue (Vyukov sequence-number design).
///
/// try_push() never touches its argument unless a slot was claimed: when the queue is
/// full it returns false and the caller's value is left intact, so the caller may retry
/// with the very same object. Constructing a T from the pushed value must not throw
/// (move construction of std::function is noexcept); a throwing copy would leave a
/// claimed slot unpublished.
template <typename T> class TaskQueue {
public:
    explicit TaskQueue(size_t capacity)
        : m_capacity(detail::round_up_power_of_two(capacity < 2 ? 2 : capacity)),
          m_mask(m_capacity - 1), m_buffer(std::make_unique<Cell[]>(m_capacity)) {
        for (size_t i = 0; i < m_capacity; ++i) {
            m_buffer[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    TaskQueue(const TaskQueue&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;

    ~TaskQueue() {
        T value;
        while (try_pop(value)) {
        }
    }

    /// Moves `value` into the queue only if a slot is available. On failure `value` is
    /// unchanged.
    bool try_push(T&& value) { return push_impl(std::move(value)); }

    /// Copies `value` into the queue only if a slot is available.
    bool try_push(const T& value) { return push_impl(value); }

    bool try_pop(T& value) {
        Cell* cell = nullptr;
        size_t position = m_dequeue_pos.load(std::memory_order_relaxed);

        for (;;) {
            cell = &m_buffer[position & m_mask];
            const size_t sequence = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff =
                static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position + 1);
            if (diff == 0) {
                if (m_dequeue_pos.compare_exchange_weak(position, position + 1,
                                                        std::memory_order_relaxed,
                                                        std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                position = m_dequeue_pos.load(std::memory_order_relaxed);
            }
        }

        T* entry = reinterpret_cast<T*>(&cell->storage);
        value = std::move(*entry);
        entry->~T();
        cell->sequence.store(position + m_mask + 1, std::memory_order_release);
        return true;
    }

    /// Approximate number of queued elements (slots claimed by producers and not yet
    /// claimed by consumers). Always within [0, capacity()].
    [[nodiscard]] size_t size() const noexcept {
        // Read the dequeue position first: both counters only grow, so the enqueue
        // position observed afterwards can never be behind it.
        const size_t dequeued = m_dequeue_pos.load(std::memory_order_acquire);
        const size_t enqueued = m_enqueue_pos.load(std::memory_order_acquire);
        const size_t count = enqueued - dequeued;
        if (enqueued < dequeued) {
            return 0;
        }
        return count > m_capacity ? m_capacity : count;
    }

    [[nodiscard]] size_t capacity() const noexcept { return m_capacity; }

private:
    struct Cell {
        std::atomic<size_t> sequence{0};
        alignas(T) unsigned char storage[sizeof(T)];
    };

    template <typename U> bool push_impl(U&& value) {
        Cell* cell = nullptr;
        size_t position = m_enqueue_pos.load(std::memory_order_relaxed);

        for (;;) {
            cell = &m_buffer[position & m_mask];
            const size_t sequence = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position);
            if (diff == 0) {
                if (m_enqueue_pos.compare_exchange_weak(position, position + 1,
                                                        std::memory_order_relaxed,
                                                        std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                position = m_enqueue_pos.load(std::memory_order_relaxed);
            }
        }

        new (&cell->storage) T(std::forward<U>(value));
        cell->sequence.store(position + 1, std::memory_order_release);
        return true;
    }

    const size_t m_capacity;
    const size_t m_mask;
    std::unique_ptr<Cell[]> m_buffer;

    alignas(64) std::atomic<size_t> m_enqueue_pos{0};
    alignas(64) std::atomic<size_t> m_dequeue_pos{0};
};

} // namespace basalt::internal
