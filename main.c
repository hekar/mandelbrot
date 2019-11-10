#include <stdio.h>
#include <stdlib.h>
#include <SDL2/SDL.h>
#include <omp.h>
#include <emmintrin.h>
#include <time.h>

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

        #pragma openmp for collapse(2)
        for (int iy = 0; iy < height; ++iy) {
            __m128 vy = _mm_set_ps1(iy);
            __m128 cy = map_sse(vy, y_factor, min_y); 
            for (int ix = 0; ix < width; ix += 4){
                float fx = (float)ix;
                __m128 vx = _mm_add_ps(_mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f), _mm_set_ps1(fx));
                __m128 cx = map_sse(vx, x_factor, min_x);
                
                union _128i pixels;
                pixels.v = mandelbrot_sse(cx, cy, max_iter);
                for (int i = 0; i < 4; i++) {
                    int colors[3];
                    color(pixels.a[i], max_iter, (void*)&colors);
                    SDL_SetRenderDrawColor(renderer, colors[0], colors[1], colors[2], 255);
                    SDL_RenderDrawPoint(renderer, ix + i, iy);
                }
            }
        }

        time_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("%f\n", time_spent);
    }
    // Close and destroy the window
    SDL_DestroyWindow(window);

    // Clean up
    SDL_Quit();
}
