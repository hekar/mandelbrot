#include <stdio.h>
#include <stdlib.h>
#include <SDL2/SDL.h>
#include <omp.h>
#include <emmintrin.h>
#include <time.h>

#define MANDEL_AVX_INNER() \
        x2 = _mm256_mul_pd(x,x); \
        y2 = _mm256_mul_pd(y,y); \
        mag = _mm256_add_pd(x2,y2); \
        xy = _mm256_mul_pd(x,y); \
        y = _mm256_add_pd(_mm256_add_pd(xy,xy),cy); \
        x = _mm256_add_pd(_mm256_sub_pd(x2,y2), cx); \
        mask = _mm256_cmp_pd(mag, _4, _CMP_LT_OQ); \
        iters = _mm256_add_pd(iters, _mm256_and_pd(mask, _1));

#define MANDEL_AVX_DEPENDENT(i) \
        y[i]  = _mm256_add_pd(_mm256_add_pd(xy[i],xy[i]),cy[i]); \
        x[i]  = _mm256_add_pd(_mm256_sub_pd(x2[i],y2[i]),cx[i]); \
        mag[i] = _mm256_add_pd(x2[i],y2[i]); \
        mask[i]  = _mm256_cmp_pd(mag[i], _4, _CMP_LT_OQ); \
        iters[i] = _mm256_add_pd(iters[i], _mm256_and_pd(mask[i], _1));

#define MANDEL_AVX_INDEPENDENT(i) \
        xy[i] = _mm256_mul_pd(x[i],y[i]); \
        x2[i] = _mm256_mul_pd(x[i],x[i]); \
        y2[i] = _mm256_mul_pd(y[i],y[i]);

#define MANDEL_AVX_ITERATION() \
        MANDEL_AVX_INDEPENDENT(0) \
        MANDEL_AVX_DEPENDENT(0) \
        MANDEL_AVX_INDEPENDENT(1) \
        MANDEL_AVX_DEPENDENT(1) \
        MANDEL_AVX_INDEPENDENT(2) \
        MANDEL_AVX_DEPENDENT(2) \
        MANDEL_AVX_INDEPENDENT(3) \
        MANDEL_AVX_DEPENDENT(3) \

#define MANDEL_AVX_MASK() \
        cmp_mask      =   \
            (_mm256_movemask_pd (mask[0]) << 4 ) \
          | (_mm256_movemask_pd (mask[1])      ) \
          | (_mm256_movemask_pd (mask[2]) << 12) \
          | (_mm256_movemask_pd (mask[3]) << 8);

void color(int n, int iter_max, int* colors) {
    int N = 256; // colors per element
    int N3 = N * N * N;
    // map n on the 0..1 interval (real numbers)
    double t = (double)n/(double)iter_max;
    // expand n on the 0 .. 256^3 interval (integers)
    n = (int)(t * (double) N3);

    colors[2] = n/(N * N);
    int nn = n - colors[2] * N * N;
    colors[0] = nn/N;
    colors[1] = nn - colors[0] * N;
}

void mandelbrot_avx(__m256d cx[4], __m256d cy[4], int max_iter, __m256d* iters) {
    __m256d _4 = _mm256_set1_pd(4.0);
    __m256d _1 = _mm256_set1_pd(1.0);
    __m256d _0 = _mm256_set1_pd(0.0);

    __m256d x[4] = {cx[0], cx[1], cx[2], cx[3]};
    __m256d y[4] = {cy[0], cy[1], cy[2], cy[3]};
    // __m256d iters[4] = {_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd()};

    __m256d x2[4], y2[4], mask[4], xy[4];
    __m256d mag[4];

    int iter = 6;
    uint32_t cmp_mask = 0;
    
    for (int iter = 6; iter > 0; --iter) {
        MANDEL_AVX_ITERATION()
        MANDEL_AVX_ITERATION()
        MANDEL_AVX_ITERATION()
        MANDEL_AVX_ITERATION()
        MANDEL_AVX_ITERATION()
        MANDEL_AVX_ITERATION()
        MANDEL_AVX_ITERATION()
        MANDEL_AVX_ITERATION()

        MANDEL_AVX_MASK()

        if (!cmp_mask) {
            return;
        }


    }

    MANDEL_AVX_ITERATION()
    MANDEL_AVX_ITERATION()

    return;
}

__m256d map_avx(__m256d vec, float factor, float out_min) {
    __m256d mul_vec = _mm256_mul_pd(vec, _mm256_set1_pd(factor));
    __m256d res = _mm256_add_pd(mul_vec, _mm256_set1_pd(out_min));
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

// assuming in_min is 0 and defining the factor as a constant for optimization
__m128 map_sse(__m128 vec, float factor, float out_min) {
    __m128 mul_vec = _mm_mul_ps(vec, _mm_set_ps1(factor));
    __m128 res     = _mm_add_ps(mul_vec, _mm_set_ps1(out_min));
    return res;
}

int mandelbrot(float cx, float cy, int max_iter) {
    float x = cx;
    float y = cy;

    int iter = 0;
    while((x * x + y * y < 4) && (iter < max_iter)) {
        float x2 = x * x;
        float y2 = y * y;
        y = 2 * x * y + cy;
        x = x2 - y2 + cx;
        iter++;
    }

    return iter;
}

float map(float val, float in_min, float in_max, float out_min, float out_max) {
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

int main() {

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window*   window;
    SDL_Renderer* renderer;
    SDL_Event     event;

    const int width = 1440;
    const int height = 1080;
    const int max_iter = 255;
    const float min_x = -2.5;
    const float max_x = 1.5;
    const float min_y = -1.5;
    const float max_y = min_y + (max_x - min_x) * height / width; 
    float x_factor = (max_x - min_x) / width;
    float y_factor = (max_y - min_y) / height;

    SDL_CreateWindowAndRenderer(1440, 1080, 0, &window, &renderer);
    SDL_RenderSetLogicalSize(renderer, width, height);

    union _128i {
        __m128i v;
        int a[4];
    };

    int quit = 0;
    while(1) {
        SDL_RenderPresent(renderer);

        if (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = 1;
                break;
            }
            if (event.type == SDL_KEYDOWN && event.key.keysym.sym == 'q') {
                quit = 1;
                break;
            }
        }
        time_t start = clock();

        // REFERENCE
        // #pragma openmp for collapse(2)
        // for (int y = 0; y < height; ++y) {
        //     for (int x = 0; x < width; ++x) {
        //         float cx = map(x, 0, width, min_x, max_y);
        //         float cy = map(y, 0, height, min_y, max_y);
        //         int iter = mandelbrot(cx, cy, max_iter);
        //         int colors[3];
        //         color(iter, max_iter, (void*)&colors);
        //         SDL_SetRenderDrawColor(renderer, colors[0], colors[1], colors[2], 255);
        //         SDL_RenderDrawPoint(renderer, x, y);
        //     }
        // }

        // SSE
        // #pragma openmp for collapse(2)
        // for (int iy = 0; iy < height; ++iy) {
        //     float fy = (float)iy;
        //     __m128 vy = _mm_set_ps1(fy);
        //     __m128 cy = map_sse(vy, y_factor, min_y); 
        //     for (int ix = 0; ix < width; ix += 4){
        //         float fx = (float)ix;
        //         __m128 vx = _mm_add_ps(_mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f), _mm_set_ps1(fx));
        //         __m128 cx = map_sse(vx, x_factor, min_x);
                
        //         union _128i pixels;
        //         pixels.v = mandelbrot_sse(cx, cy, 50);
        //         for (int i = 0; i < 4; i++) {
        //             int colors[3];
        //             color(pixels.a[i], 50, (void*)&colors);
        //             SDL_SetRenderDrawColor(renderer, colors[0], colors[1], colors[2], 255);
        //             SDL_RenderDrawPoint(renderer, ix + i, iy);
        //         }
        //     }
        // }

        // AVX
        #pragma openmp for collapse(2)
        for (int y = 0; y < height; ++y) {
            double dy = (double)y;
            __m256d vy = _mm256_set1_pd(dy);
            __m256d cvy = map_avx(vy, y_factor, min_y);
            __m256d cy[4] = {cvy, cvy, cvy, cvy};
            for (int x = 0; x < width; x += 16) {
                double dx = (double)x;
                __m256d vx0 = _mm256_add_pd(_mm256_set_pd(3.0, 2.0, 1.0, 0.0), _mm256_set1_pd(dx));
                __m256d vx1 = _mm256_add_pd(_mm256_set_pd(4.0, 4.0, 4.0, 4.0), vx0);
                __m256d vx2 = _mm256_add_pd(_mm256_set_pd(4.0, 4.0, 4.0, 4.0), vx1);
                __m256d vx3 = _mm256_add_pd(_mm256_set_pd(4.0, 4.0, 4.0, 4.0), vx2);
                __m256d cx[4] = {map_avx(vx0, x_factor, min_x), map_avx(vx1, x_factor, min_x), map_avx(vx2, x_factor, min_x), map_avx(vx3, x_factor, min_x)};

                union _128i pixels;
                __m256d res[4] = {_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd()};
                mandelbrot_avx(cx, cy, max_iter, res);
                int offset = 0;
                for (int i = 0; i < 4; i++) {
                    pixels.v = _mm256_cvtpd_epi32(res[i]);
                    for (int j = 0; j < 4; j++) {
                        int colors[3];
                        color(pixels.a[j], 50, (void*)&colors);
                        SDL_SetRenderDrawColor(renderer, colors[0], colors[1], colors[2], 255);
                        SDL_RenderDrawPoint(renderer, x + offset++, y);
                    }
                }
            }
        }

        // __m256d min_x_4 = _mm256_set1_pd (min_x);
        // __m256d scale_x_4 = _mm256_set1_pd(x_factor);
        // __m256d lshift_x_4 = _mm256_set_pd(0, 1, 2, 3);
        // __m256d ushift_x_4 = _mm256_set_pd(4, 5, 6, 7);

        // for (int sy = 0; sy < height; sy += 2) {
        //     double dy = (double)sy;
        //     __m256d cy0 = _mm256_set1_pd(y_factor*dy + min_y);
        //     __m256d cy1 = _mm256_set1_pd(y_factor*(dy+1) + min_y);
        //     for (int sx = 0; sx < width; ++sx) {
        //         int x = sx*8;
        //         __m256d x_8 = _mm256_set1_pd(x);
        //         __m256d cx0 = _mm256_add_pd(min_x_4, _mm256_mul_pd(_mm256_add_pd(x_8, lshift_x_4), scale_x_4));
        //         __m256d cx1 = _mm256_add_pd(min_x_4, _mm256_mul_pd(_mm256_add_pd(x_8, ushift_x_4), scale_x_4));
        //         __m256d cx[4] = { cx0, cx1, cx0, cx1 };
        //         __m256d cy[4] = { cy0, cy0, cy1, cy1 };
        //         __m256d res[4] = {_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd()};
        //         mandelbrot_avx(cx, cy, max_iter, res);
        //         union _128i pixels;
        //         for (int i = 0; i < 4; ++i) {
        //             pixels.v = _mm256_cvtpd_epi32(res[i]);
        //             int x_c = sx;
        //             if (i % 2) {
        //                 x_c = sx + 3;
        //             } else {
        //                 x_c = sx + 7;
        //             }
        //             int y_c = sy;
        //             if (i > 1) {
        //                 y_c++;
        //             }
        //             for (int j = 0; j < 4; ++j) {
        //                 int colors[3];
        //                 color(pixels.a[j], 50, (void*)&colors);
        //                 SDL_SetRenderDrawColor(renderer, colors[0], colors[1], colors[2], 255);
        //                 SDL_RenderDrawPoint(renderer, sx + x_c--, y_c);
        //             }
        //         }
        //     }
        // }

        time_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("%f\n", time_spent);
    }
    // Close and destroy the window
    SDL_DestroyWindow(window);

    // Clean up
    SDL_Quit();
}
