#pragma once

#include <atomic>
#include <cstddef>
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

template <typename T>
class TaskQueue {
public:
    explicit TaskQueue(size_t capacity)
        : m_capacity(detail::round_up_power_of_two(capacity < 2 ? 2 : capacity)),
          m_mask(m_capacity - 1),
          m_buffer(std::make_unique<Cell[]>(m_capacity)) {
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

    bool try_push(T value) {
        Cell* cell = nullptr;
        size_t position = m_enqueue_pos.load(std::memory_order_relaxed);

        for (;;) {
            cell = &m_buffer[position & m_mask];
            const size_t sequence = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position);
            if (diff == 0) {
                if (m_enqueue_pos.compare_exchange_weak(
                        position, position + 1,
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

        new (&cell->storage) T(std::move(value));
        cell->sequence.store(position + 1, std::memory_order_release);
        m_size.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool try_pop(T& value) {
        Cell* cell = nullptr;
        size_t position = m_dequeue_pos.load(std::memory_order_relaxed);

        for (;;) {
            cell = &m_buffer[position & m_mask];
            const size_t sequence = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position + 1);
            if (diff == 0) {
                if (m_dequeue_pos.compare_exchange_weak(
                        position, position + 1,
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
        m_size.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    [[nodiscard]] size_t size() const noexcept {
        return m_size.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_t capacity() const noexcept {
        return m_capacity;
    }

private:
    struct Cell {
        std::atomic<size_t> sequence{0};
        typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
    };

    const size_t m_capacity;
    const size_t m_mask;
    std::unique_ptr<Cell[]> m_buffer;

    alignas(64) std::atomic<size_t> m_enqueue_pos{0};
    alignas(64) std::atomic<size_t> m_dequeue_pos{0};
    alignas(64) std::atomic<size_t> m_size{0};
};

} // namespace basalt::internal
