#include <immintrin.h>

__m256d map_avx(__m256d vec, double factor, double out_min);

void mandelbrot_avx(__m256d cx[4], __m256d cy[4], int max_iter, __m256d* iters);