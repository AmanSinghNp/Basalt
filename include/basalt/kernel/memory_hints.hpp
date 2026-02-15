#pragma once

#include <immintrin.h>

namespace basalt::kernel {

/// Prefetch data into L1 cache (temporal, highest priority).
/// Call ~256 bytes ahead of where you're currently processing.
inline void prefetch_l1(const void* addr) {
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
}

/// Prefetch data into L2 cache (temporal, lower priority).
/// Call ~1024 bytes ahead of where you're currently processing.
inline void prefetch_l2(const void* addr) {
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T1);
}

/// Non-temporal store: bypass cache when writing data that won't be reread soon.
/// 'dst' MUST be 32-byte aligned.
inline void stream_store(float* dst, __m256 val) {
    _mm256_stream_ps(dst, val);
}

/// Fence to ensure all non-temporal stores are globally visible.
inline void store_fence() {
    _mm_sfence();
}

} // namespace basalt::kernel
