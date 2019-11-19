#include <emmintrin.h>

__m128 map_sse(__m128 vec, float factor, float out_min);

__m128i mandelbrot_sse(__m128 cx, __m128 cy, int max_iter);