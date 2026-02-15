#pragma once

#include <immintrin.h>
#include <cmath>
#include <limits>

namespace basalt::utils {

/// Fast vectorizable exponential approximation.
/// Based on Schraudolph's method or a Padé approximant.
/// For this implementation, we use a high-performance 5th-order polynomial approximation 
/// valid for the range [-87, 87] (input to exp).
/// Derived from Agner Fog's VCL or similar high-perf libraries.
inline __m256 fast_exp_avx(__m256 x) {
    // Range clamping is assumed to be handled by caller (sigmoid inputs usually safe)
    
    // Horner's method: (((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0)
    // But standard exp approximation is usually done via 2^x identity.
    // Let's use the valid range reduction method: exp(x) = 2^(x * log2(e))
    
    const __m256 log2e = _mm256_set1_ps(1.44269504088896340736f);

    __m256 t = _mm256_mul_ps(x, log2e);
    // Round to nearest integer
    __m256 e = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    
    // Fraction part
    __m256 f = _mm256_sub_ps(t, e);
    
    // Polynomial approximation for 2^f where f in [-0.5, 0.5]
    // P(f) ~ 2^f
    // Using a degree-5 polynomial
    // Coefficients for 2^x in [-0.5, 0.5]
    const __m256 p0 = _mm256_set1_ps(1.0f);
    const __m256 p1 = _mm256_set1_ps(0.69314718f); // ln(2)
    const __m256 p2 = _mm256_set1_ps(0.24022650f);
    const __m256 p3 = _mm256_set1_ps(0.05550410f);
    const __m256 p4 = _mm256_set1_ps(0.00961812f);
    const __m256 p5 = _mm256_set1_ps(0.00133335f);

    __m256 poly = p5;
    poly = _mm256_fmadd_ps(poly, f, p4);
    poly = _mm256_fmadd_ps(poly, f, p3);
    poly = _mm256_fmadd_ps(poly, f, p2);
    poly = _mm256_fmadd_ps(poly, f, p1);
    poly = _mm256_fmadd_ps(poly, f, p0);

    // Reconstruct 2^x = 2^e * 2^f
    // Compute 2^e using integer shift
    __m256i ei = _mm256_cvtps_epi32(e);
    __m256i bias = _mm256_set1_epi32(127);
    __m256i n = _mm256_add_epi32(ei, bias);
    __m256i s = _mm256_slli_epi32(n, 23); // Shift into exponent position
    __m256 scale = _mm256_castsi256_ps(s);

    return _mm256_mul_ps(poly, scale);
}

/// Fast Sigmoid: 1 / (1 + exp(-x))
inline __m256 fast_sigmoid_avx(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_set1_ps(0.0f);
    
    // Clamp input to avoid overflow in exp
    // exp(-87) is close enough to 0, exp(87) is huge
    const __m256 max_input = _mm256_set1_ps(87.0f);
    const __m256 min_input = _mm256_set1_ps(-87.0f);
    
    x = _mm256_min_ps(x, max_input);
    x = _mm256_max_ps(x, min_input);
    
    // exp(-x)
    __m256 neg_x = _mm256_sub_ps(zero, x);
    __m256 e = fast_exp_avx(neg_x);
    
    // 1 / (1 + e)
    __m256 den = _mm256_add_ps(one, e);
    return _mm256_div_ps(one, den);
}

} // namespace basalt::utils
