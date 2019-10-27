#include <stdio.h>
#include <stdlib.h>
#include <SDL2/SDL.h>

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

    const int width = 800;
    const int height = 800;
    const int max_iter = 255;
    const float min_x = -2.0;
    const float max_x = 1.0;
    const float min_y = -1.6;
    const float max_y = min_y + (max_x - min_x) * height / width; 

    SDL_CreateWindowAndRenderer(1920, 1080, 0, &window, &renderer);
    SDL_RenderSetLogicalSize(renderer, width, height);

    int quit = 0;
    while(1) {
        SDL_RenderPresent(renderer);

        for (int y = 0; y < height; ++y) {
            if (quit) {
                break;
            }
            for (int x = 0; x < width; ++x) {
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
                float cx = map(x, 0, width, min_x, max_y);
                float cy = map(y, 0, height, min_y, max_y);
                int iter = mandelbrot(cx, cy, max_iter);
                int colors[3];
                color(iter, max_iter, (void*)&colors);
                SDL_SetRenderDrawColor(renderer, colors[0], colors[1], colors[2], 255);
                SDL_RenderDrawPoint(renderer, x, y);
            }
        }

        if (quit) {
            break;
        }
    }
    // Close and destroy the window
    SDL_DestroyWindow(window);

    // Clean up
    SDL_Quit();
}
