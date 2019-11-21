#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <SDL2/SDL.h>
#include <omp.h>
#include <time.h>

#include "mandelbrot.h"
#include "mandelbrot_sse.h"
#include "mandelbrot_avx.h"

const int width = 800;
const int height = 800;
const int max_iter = 50;
double min_x;
double max_x;
double min_y;
double max_y;
double x_factor;
double y_factor;

long double factor = 1;

enum instruction_set {
    def,
    avx,
    sse
};

union _128i {
    __m128i v;
    int a[4];
};

union _256d_4 {
    __m256d v[4];
    double  arr[4][4];
};

void color_poly(int n, int iter_max, int* colors) {
	// map n on the 0..1 interval
	double t = (double)n/(double)iter_max;

	// Use smooth polynomials for r, g, b
	colors[0] = (int)(9*(1-t)*t*t*t*255);
	colors[1] = (int)(15*(1-t)*(1-t)*t*t*255);
	colors[2] =  (int)(8.5*(1-t)*(1-t)*(1-t)*t*255);
}

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

#ifdef __x86_64__
#include <cpuid.h>

static inline int
is_avx_supported(void)
{
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return ecx & bit_AVX ? 1 : 0;
}
#endif // __x86_64__

void mandelbrot_driver(int* color_arr, int i_set) {
    __m256d _3210 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    __m256d _4    = _mm256_set1_pd(4.0);

    time_t start = clock();
    #ifdef __x86_64__
    if ((i_set == avx) && is_avx_supported()) {
        #pragma openmp parallel for schedule(guided)
        for (int y = 0; y < height; ++y) {
            __m256d vy = _mm256_set1_pd((double)y);
            __m256d cvy = map_avx(vy, y_factor, min_y);
            __m256d cy[4] = {cvy, cvy, cvy, cvy};
            for (int x = 0; x < width; x += 16) {
                __m256d vx0 = _mm256_add_pd(_3210, _mm256_set1_pd((double)x));
                __m256d vx1 = _mm256_add_pd(_4, vx0);
                __m256d vx2 = _mm256_add_pd(_4, vx1);
                __m256d vx3 = _mm256_add_pd(_4, vx2);
                __m256d cx[4] = {map_avx(vx0, x_factor, min_x), map_avx(vx1, x_factor, min_x),
                                    map_avx(vx2, x_factor, min_x), map_avx(vx3, x_factor, min_x)};

                union _256d_4 res = {_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd()};
                mandelbrot_avx(cx, cy, max_iter, res.v);
                int offset = 0;
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        color_arr[width*y + (x+offset++)] = res.arr[i][j];
                    }
                }
            }
        }
    } else if (i_set == sse) {
        #pragma openmp for collapse(2)
        for (int iy = 0; iy < height; ++iy) {
            float fy = (float)iy;
            __m128 vy = _mm_set_ps1(fy);
            __m128 cy = map_sse(vy, y_factor, min_y);
            for (int ix = 0; ix < width; ix += 4){
                float fx = (float)ix;
                __m128 vx = _mm_add_ps(_mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f), _mm_set_ps1(fx));
                __m128 cx = map_sse(vx, x_factor, min_x);

                union _128i pixels;
                pixels.v = mandelbrot_sse(cx, cy, max_iter);
                for (int i = 0; i < 4; i++) {
                    color_arr[width*iy + (ix+i)] = pixels.a[i];
                }
            }
        }
    }
    #endif // __x86_64__
    else {
        #pragma openmp for collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float cx = map(x, 0, width, min_x, max_x);
                float cy = map(y, 0, height, min_y, max_y);
                int iter = mandelbrot(cx, cy, max_iter);
                color_arr[width*y + x] = iter;
            }
        }
    }

    time_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%f\n", time_spent);
}

void update_display_cfg(double x_min, double x_max, double y_min, double y_max) {
    min_x = x_min;
    max_x = x_max;
    min_y = y_min;
    max_y = y_max;
    // max_y = min_y + (max_x - min_x) * height / width;
    x_factor = (max_x - min_x) / width;
    y_factor = (max_y - min_y) / height;
}

double map_double(double val, double in_max, double out_min, double out_max) {
    // 425 * (2 - 400 / 800) + 400
    return (val) * (out_max - out_min) / (in_max) + out_min;
}

int main(int argc, char** argv) {

    // parse options
    enum instruction_set i_set;
    int opt;
    while((opt = getopt(argc, argv, "i:")) != -1)  
    {  
        switch(opt)  
        {  
            case 'i':
            switch(*optarg) {
                case 'A':
                    i_set = avx;
                    break;
                case 'S':
                    i_set = sse;
                    break;
                default:
                    i_set = def;
                    break;
            }
            break;
        }
    }

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window*   window;
    SDL_Renderer* renderer;
    SDL_Event     event;

    int* color_arr = malloc(sizeof(int)*width*height);

    SDL_CreateWindowAndRenderer(width, height, 0, &window, &renderer);
    SDL_RenderSetLogicalSize(renderer, width, height);

    update_display_cfg(-2.0, 2.0, -2.0, 2.0);
    mandelbrot_driver(color_arr, i_set);

    while(1) {
        SDL_RenderPresent(renderer);

        if (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                break;
            }
            if (event.type == SDL_KEYDOWN && event.key.keysym.sym == 'q') {
                break;
            }
            if(event.type == SDL_MOUSEWHEEL)
            {
                if(event.wheel.y > 0) // scroll up
                {
                    // Put code for handling "scroll up" here!
                    int x, y;
                    SDL_GetMouseState(&x, &y);
                    double dx = map_double(x, width, min_x, max_x);
                    double dy = map_double(y, height, min_y, max_y);
                    // printf("dx is : %f\n", dx);
                    // printf("dy is : %f\n", dy);

                    double x_min = dx + 0.7*(min_x - dx);
                    double x_max = dx + 0.7*(max_x - dx);
                    double y_min = dy + 0.7*(min_y - dy);
                    double y_max = dy + 0.7*(max_y - dy);
                    // // printf("factor is : %f\n", factor);
                    // printf("x_min is : %f\n", x_min);
                    // printf("x_max is : %f\n", x_max);
                    // printf("y_min is : %f\n", y_min);
                    // printf("y_max is : %f\n", y_max);

                    // double x_max = max_x - (0.1* (max_x - dx));
                    // double x_min = min_x + (0.1* (dx - min_x));
                    // double y_max = max_y - (0.1* (max_y - dy));
                    // double y_min = min_y + (0.1* (dy - min_y));

                    // max_x -= 0.1 * factor;
                    // min_x += 0.15 * factor;
                    // max_y -= 0.1 * factor;
                    // min_y += 0.15 * factor;
                    // factor *= 0.9349;
                    update_display_cfg(x_min, x_max, y_min, y_max);
                    mandelbrot_driver(color_arr, i_set);
                }
                else if(event.wheel.y < 0) // scroll down
                {
                    // Put code for handling "scroll down" here!
                    printf("mouse wheel down %d\n", event.wheel.y);
                }
            }
        }

        // DRAWING
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int colors[3];
                color_poly(color_arr[w + (h*width)], 50, (void*)&colors);
                SDL_SetRenderDrawColor(renderer, colors[0], colors[1], colors[2], 255);
                SDL_RenderDrawPoint(renderer, w, h);
            }
        }
    }
    // Close and destroy the window
    SDL_DestroyWindow(window);

    // Clean up
    SDL_Quit();
}
