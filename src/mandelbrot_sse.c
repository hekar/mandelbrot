#include "mandelbrot_sse.h"

__m128 map_sse(__m128 vec, float factor, float out_min) {
    __m128 mul_vec = _mm_mul_ps(vec, _mm_set_ps1(factor));
    __m128 res     = _mm_add_ps(mul_vec, _mm_set_ps1(out_min));
    return res;
}

__m128i mandelbrot_sse(__m128 cx, __m128 cy, int max_iter) {
    __m128 x = cx;
    __m128 y = cy;

    __m128 iters = _mm_setzero_ps();
    int iter = 0;
    while(iter++ < max_iter) {
        __m128 x2 = _mm_mul_ps(x,x);
        __m128 y2 = _mm_mul_ps(y,y);
        __m128 xy = _mm_mul_ps(x,y);
        y = _mm_add_ps(_mm_add_ps(xy, xy), cy);
        x = _mm_add_ps(_mm_sub_ps(x2, y2), cx);

        x2 = _mm_mul_ps(x,x);
        y2 = _mm_mul_ps(y,y);
        __m128 mag = _mm_add_ps(x2, y2);
        __m128 mask = _mm_cmplt_ps(mag, _mm_set_ps1(4.0f));
        iters = _mm_add_ps(_mm_and_ps(mask, _mm_set_ps1(1.0f)), iters);

        if (_mm_movemask_ps(mask) == 0)
            break;
    }
    return _mm_cvtps_epi32(iters);
}