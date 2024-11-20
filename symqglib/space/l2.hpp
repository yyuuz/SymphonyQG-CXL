#pragma once

#include <bits/stdc++.h>
#include <immintrin.h>

namespace symqg::space {

inline auto l2_sqr(
    const float* __restrict__ vec0, const float* __restrict__ vec1, size_t dim
) -> float {
    float result = 0;
#if defined(__AVX512F__)
    size_t num16 = dim - (dim & 0b1111);
    auto sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i < num16; i += 16) {
        auto xxx = _mm512_loadu_ps(&vec0[i]);
        auto yyy = _mm512_loadu_ps(&vec1[i]);
        auto ttt = _mm512_sub_ps(xxx, yyy);
        sum = _mm512_fmadd_ps(ttt, ttt, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (; i < dim; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }

#elif defined(__AVX2__)
    size_t num8 = dim - (dim & 0b111);
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i < num8; i += 8) {
        __m256 xx = _mm256_loadu_ps(&vec0[i]);
        __m256 yy = _mm256_loadu_ps(&vec1[i]);
        __m256 t = _mm256_sub_ps(xx, yy);
        sum = _mm256_fmadd_ps(t, t, sum);
    }
    result = reduce_add_m256(sum);
    for (; i < dim; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }

#else
    for (size_t i = 0; i < L; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }
#endif
    return result;
}

}  // namespace symqg::space