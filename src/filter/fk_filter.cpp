#include "basalt/filter/fk_filter.hpp"
#include "basalt/utils/fast_math.hpp"

#include <cmath>
#include <algorithm>
#include <immintrin.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace basalt::filter {

void apply_fk_filter(basalt::kernel::ComplexSoA& data, 
                     size_t rows, size_t cols, 
                     const FKParams& params) {
    if (params.taper_width <= 0.0) return;

    // Derived constants
    // Frequency step df = 1 / (Nt * dt) — assuming rows is Nt (after padding)
    // But rows/cols here are just dimensions. 
    // Standard interpretation: 
    // f axis goes from 0 to Nyquist. 
    // k axis goes from -Nyquist to +Nyquist (or 0 to Nyquist then negative).
    // Basalt's FFT layout: Standard positive frequencies 0..N/2.
    // 2D FFT layout (centered vs uncentered):
    // Usually FFT output is uncentered: 0..N/2 (pos), -N/2..-1 (neg).
    // Let's assume standard FFT order:
    // Rows (f): 0, 1, ..., N/2 (real input -> Hermitian, usually simplified)
    // Cols (k): 0, 1, ..., N/2, -N/2+1, ..., -1.

    double df = 1.0 / (rows * params.dt);
    double dk = 1.0 / (cols * params.dx);

    float steepness = 10.0f / static_cast<float>(params.taper_width);
    float v_cut_pos = static_cast<float>(params.mute_vel_max);
    
    const __m256 v_steep = _mm256_set1_ps(steepness);
    const __m256 v_cut = _mm256_set1_ps(v_cut_pos);
    const __m256 ones = _mm256_set1_ps(1.0f);

    for (size_t i = 0; i < rows; ++i) {
        // Frequency f
        // If real-to-complex transform, rows are 0 to N/2 usually.
        // Let's assume full grid for generality or N rows.
        // For f-k, f is usually positive.
        // Frequency f
        double f;
        if (i <= rows / 2) {
            f = static_cast<double>(i) * df;
        } else {
            f = static_cast<double>(static_cast<int>(i) - static_cast<int>(rows)) * df;
        }
        
        // Handle Nyquist folding if i > rows/2
        if (i > rows / 2) {
            f = static_cast<double>(static_cast<int>(i) - static_cast<int>(rows)) * df;
        }

        // Avoid divide-by-zero at DC (v = f/k)
        // At f=0, v=0 regardless of k (except k=0). 
        // We want to pass f=0 usually? Or mute specific velocities?
        // Let's protect f=0.
        
        for (size_t j = 0; j < cols; j += 8) {
            // Wavenumber k for this block
            // Handle wrapping: 0..N/2 is positive, N/2..N is negative
            // We need a vector of k values
            float k_vals[8];
            for (int lane = 0; lane < 8; ++lane) {
                size_t idx = j + lane;
                if (idx < cols) {
                    if (idx <= cols / 2) {
                        k_vals[lane] = static_cast<float>(idx * dk);
                    } else {
                        k_vals[lane] = static_cast<float>((static_cast<int>(idx) - static_cast<int>(cols)) * dk);
                    }
                } else {
                    k_vals[lane] = 1.0f; // Padding
                }
            }
            __m256 vk = _mm256_loadu_ps(k_vals);
            
            // Add epsilon to k to avoid division by zero
            // v = f / (k + eps)
            const __m256 eps = _mm256_set1_ps(1e-6f);
            // Sign of k to preserve velocity direction? 
            // Velocity V = f / k. 
            // If f is approx 0, V is 0. 
            // If k is approx 0, V is infinite.
            
            // Mask for k near 0
            // But simply adding eps is safer for branchless
            __m256 vk_safe = _mm256_add_ps(vk, eps);
            
            __m256 vf = _mm256_set1_ps(static_cast<float>(f));
            __m256 vel = _mm256_div_ps(vf, vk_safe);
            
            // We want to mute if |vel| < v_cut
            // i.e. -v_cut < vel < v_cut
            // Using sigmoid taper logic:
            // Pass weight should be 0 inside mute zone, 1 outside.
            // Let's use two sigmoids? or |vel|?
            // Mute zone is centered at 0.
            // Weight = Sigmoid( (|vel| - v_cut) * steepness )
            // If |vel| >> v_cut, arg is large positive, Sigmoid -> 1 (Pass)
            // If |vel| << v_cut, arg is large negative, Sigmoid -> 0 (Mute)
            
            // abs(vel) = vel & 0x7FFFFFFF
            const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 abs_vel = _mm256_and_ps(vel, sign_mask);
            
            // x = (|vel| - v_cut) * steepness
            // We want to PASS velocities > v_cut (high velocity)
            // or PASS velocities < v_cut (low velocity)?
            // "Velocities inside [-vm, +vm] will be muted." -> Low velocities are muted.
            // So if |vel| < v_cut, weight should be 0.
            // If |vel| > v_cut, weight should be 1.
            // Sigmoid(x) goes 0->1 as x goes -inf -> +inf.
            // Let x = (|vel| - v_cut) * steepness.
            // If |vel| = 0, x = -huge -> Sigmoid = 0 (Mute). Correct.
            // If |vel| = huge, x = +huge -> Sigmoid = 1 (Pass). Correct.
            
            // Swap to v_cut - abs_vel to correct Pass/Mute logic found in testing
            __m256 x = _mm256_sub_ps(v_cut, abs_vel);
            x = _mm256_mul_ps(x, v_steep);
            
            // weight = sigmoid(x)
            __m256 weight = basalt::utils::fast_sigmoid_avx(x);
            
            // Apply weight to real and imag
            // Handle edge (non-8-aligned cols)
            size_t remaining = cols - j;
            if (remaining >= 8) {
                // Load data
                __m256 r = _mm256_loadu_ps(data.real + i * cols + j);
                __m256 im = _mm256_loadu_ps(data.imag + i * cols + j);
                
                // Multiply
                r = _mm256_mul_ps(r, weight);
                im = _mm256_mul_ps(im, weight);
                
                // Store
                _mm256_storeu_ps(data.real + i * cols + j, r);
                _mm256_storeu_ps(data.imag + i * cols + j, im);
            } else {
                // Scalar tail fallback from vector result
                alignas(32) float w[8];
                _mm256_store_ps(w, weight);
                for (size_t k = 0; k < remaining; ++k) {
                    data.real[i * cols + j + k] *= w[k];
                    data.imag[i * cols + j + k] *= w[k];
                }
            }
        }

    }
}
} // namespace basalt::filter
