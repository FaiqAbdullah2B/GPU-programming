/*** ![Figure 1: Denoising example (original image by Simpsons, CC BY-SA 3.0, <https://commons.wikimedia.org/w/index.php?curid=8904364>).](denoise.png)

The file [cuda-denoise.c](cuda-denoise.c) contains a serial
implementation of an _image denoising_ algorithm that (to some extent)
can be used to "cleanup" color images. The algorithm replaces the
color of each pixel with the _median_ of the four adjacent pixels plus
itself (_median-of-five_).  The median-of-five algorithm is applied
separately for each color channel (red, green, and blue).

This is particularly useful for removing "hot pixels", i.e., pixels
whose color is way off its intended value, for example due to problems
in the sensor used to acquire the image. However, depending on the
amount of noise, a single pass could be insufficient to remove every
hot pixel; see Figure 1.

The goal of this exercise is to parallelize the denoising algorithm on
the GPU using CUDA. You should launch as many CUDA threads as pixels
in the image, so that each thread is mapped onto a different pixel.

The input image is read from standard input in
[PPM](http://netpbm.sourceforge.net/doc/ppm.html) (Portable Pixmap)
format; the result is written to standard output in the same format.

To compile:

        nvcc cuda-denoise.cu -o cuda-denoise

To execute:

        ./cuda-denoise < input > output

Example:

        ./cuda-denoise < valve-noise.ppm > valve-denoised.ppm

## Files

- [cuda-denoise.cu](cuda-denoise.cu) [hpc.h](hpc.h)
- [valve-noise.ppm](valve-noise.ppm) (sample input)

***/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxcol;  /* Largest color value (Used by the PPM read/write routines) */
    unsigned char *r, *g, *b; /* color channels (arrays of width x height elements each); each value must be less than or equal to maxcol */
} PPM_image;

/**
 * Read a PPM file from file `f`. This function is not very robust; it
 * may fail on perfectly legal PGM images, but works for the provided
 * cat.pgm file.
 */
void read_ppm( FILE *f, PPM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P6") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P6\n")) {
        fprintf(stderr, "FATAL: wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxcol; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxcol));
    if ( img->maxcol > 255 ) {
        fprintf(stderr, "FATAL: maxcol=%d > 255\n", img->maxcol);
        exit(EXIT_FAILURE);
    }
    /* Get the binary data */
    img->r = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->r != NULL);
    img->g = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->g != NULL);
    img->b = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->b != NULL);
    for (int k=0; k<(img->width)*(img->height); k++) {
        nread = fscanf(f, "%c%c%c", img->r + k, img->g + k, img->b + k);
        if (nread != 3) {
            fprintf(stderr, "FATAL: error reading pixel data\n");
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * Write the image `img` to file `f`; is not NULL, use the string
 * `comment` as metadata.
 */
void write_ppm( FILE *f, const PPM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P6\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxcol);
    for (int k=0; k<(img->width)*(img->height); k++) {
        fprintf(f, "%c%c%c", img->r[k], img->g[k], img->b[k]);
    }
}

/**
 * Free all memory used by the structure `img`
 */
void free_ppm( PPM_image* img )
{
    assert(img != NULL);
    free(img->r);
    free(img->g);
    free(img->b);
    img->r = img->g = img->b = NULL; /* not necessary */
    img->width = img->height = img->maxcol = -1;
}

#define BLKDIM 32

/**
 * Swap *a and *b if necessary so that, at the end, *a <= *b
 */
void compare_and_swap( unsigned char *a, unsigned char *b )
{
    if (*a > *b ) {
        unsigned char tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

unsigned char *PTR(unsigned char *bmap, int width, int i, int j)
{
    return (bmap + i*width + j);
}

/**
 * Return the median of v[0..4]
 */
unsigned char median_of_five( unsigned char v[5] )
{
    /* We do a partial sort of v[5] using bubble sort until v[2] is
       correctly placed; this element is the median. (There are better
       ways to compute the median-of-five). */
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v  , v+1 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    return v[2];
}

/**
 * Denoise a single color channel
 */
void denoise( unsigned char *bmap, int width, int height )
{
    unsigned char *out = (unsigned char*)malloc(width*height);
    unsigned char v[5];
    assert(out != NULL);

    memcpy(out, bmap, width*height);
    /* Note that the pixels on the border are left unchanged */
    for (int i=1; i<height - 1; i++) {
        for (int j=1; j<width - 1; j++) {
            v[0] = *PTR(bmap, width, i  , j  );
            v[1] = *PTR(bmap, width, i  , j-1);
            v[2] = *PTR(bmap, width, i  , j+1);
            v[3] = *PTR(bmap, width, i-1, j  );
            v[4] = *PTR(bmap, width, i+1, j  );

            *PTR(out, width, i, j) = median_of_five(v);
        }
    }
    memcpy(bmap, out, width*height);
    free(out);
}

#define TILE_W 32
#define TILE_H 32

__device__ inline void d_swap(unsigned char &a, unsigned char &b) {
    if (a > b) {
        unsigned char tmp = a;
        a = b;
        b = tmp;
    }
}

__device__ inline unsigned char d_median_of_five(unsigned char v0, unsigned char v1, unsigned char v2, unsigned char v3, unsigned char v4) {
    d_swap(v3, v4);
    d_swap(v2, v3);
    d_swap(v1, v2);
    d_swap(v0, v1);
    d_swap(v3, v4);
    d_swap(v2, v3);
    d_swap(v1, v2);
    d_swap(v3, v4);
    d_swap(v2, v3);
    return v2;
}

__global__ void denoise_kernel(const unsigned char *in, unsigned char *out, int width, int height) {
    // Shared memory for a tile + 1 pixel halo on all 4 sides
    __shared__ unsigned char smem[TILE_H + 2][TILE_W + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    if (x < width && y < height) {
        smem[ty + 1][tx + 1] = in[y * width + x];
    } else {
        smem[ty + 1][tx + 1] = 0;
    }

    // halo cells
    if (tx == 0 && x > 0) {
        smem[ty + 1][0] = in[y * width + (x - 1)];
    }
    if (tx == TILE_W - 1 && x < width - 1) {
        smem[ty + 1][TILE_W + 1] = in[y * width + (x + 1)];
    }
    if (ty == 0 && y > 0) {
        smem[0][tx + 1] = in[(y - 1) * width + x];
    }
    if (ty == TILE_H - 1 && y < height - 1) {
        smem[TILE_H + 1][tx + 1] = in[(y + 1) * width + x];
    }

    __syncthreads();

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        unsigned char v0 = smem[ty + 1][tx + 1]; 
        unsigned char v1 = smem[ty + 1][tx];     
        unsigned char v2 = smem[ty + 1][tx + 2]; 
        unsigned char v3 = smem[ty][tx + 1];     
        unsigned char v4 = smem[ty + 2][tx + 1];
        
        out[y * width + x] = d_median_of_five(v0, v1, v2, v3, v4);
    } 
    
    else if (x < width && y < height) {
        out[y * width + x] = smem[ty + 1][tx + 1];
    }
}

int main( void )
{
    PPM_image img;
    read_ppm(stdin, &img);
    
    int img_size = img.width * img.height * sizeof(unsigned char);

    // --- Backup original noisy image ---
    unsigned char *r_copy = (unsigned char*)malloc(img_size);
    unsigned char *g_copy = (unsigned char*)malloc(img_size);
    unsigned char *b_copy = (unsigned char*)malloc(img_size);
    
    memcpy(r_copy, img.r, img_size);
    memcpy(g_copy, img.g, img_size);
    memcpy(b_copy, img.b, img_size);

    // --- CPU Execution ---
    fprintf(stderr, "Starting CPU Denoising...\n");
    const double cpu_start = hpc_gettime();
    
    denoise(img.r, img.width, img.height);
    denoise(img.g, img.width, img.height);
    denoise(img.b, img.width, img.height);
    
    const double cpu_elapsed = hpc_gettime() - cpu_start;
    fprintf(stderr, "CPU Execution time: %.6f seconds\n", cpu_elapsed);

    // Restore the noisy image
    memcpy(img.r, r_copy, img_size);
    memcpy(img.g, g_copy, img_size);
    memcpy(img.b, b_copy, img_size);

    // --- GPU (CUDA) Execution ---
    unsigned char *d_r, *d_g, *d_b;
    unsigned char *d_out_r, *d_out_g, *d_out_b;

    cudaMalloc((void**)&d_r, img_size);
    cudaMalloc((void**)&d_g, img_size);
    cudaMalloc((void**)&d_b, img_size);
    cudaMalloc((void**)&d_out_r, img_size);
    cudaMalloc((void**)&d_out_g, img_size);
    cudaMalloc((void**)&d_out_b, img_size);

    cudaMemcpy(d_r, img.r, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, img.g, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, img.b, img_size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_W, TILE_H);
    dim3 gridSize((img.width + TILE_W - 1) / TILE_W, (img.height + TILE_H - 1) / TILE_H);

    fprintf(stderr, "Starting GPU Denoising...\n");
    const double gpu_start = hpc_gettime();
    
    denoise_kernel<<<gridSize, blockSize>>>(d_r, d_out_r, img.width, img.height);
    denoise_kernel<<<gridSize, blockSize>>>(d_g, d_out_g, img.width, img.height);
    denoise_kernel<<<gridSize, blockSize>>>(d_b, d_out_b, img.width, img.height);
    
    cudaDeviceSynchronize();
    const double gpu_elapsed = hpc_gettime() - gpu_start;
    
    fprintf(stderr, "GPU Execution time: %.6f seconds\n", gpu_elapsed);
    fprintf(stderr, "Speedup: %.2fx\n", cpu_elapsed / gpu_elapsed);

    // Copy final GPU result back for output
    cudaMemcpy(img.r, d_out_r, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img.g, d_out_g, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img.b, d_out_b, img_size, cudaMemcpyDeviceToHost);

    // final cleaned image
    write_ppm(stdout, &img, "Comparison benchmark complete.");

    // Cleaning up
    cudaFree(d_r);
    cudaFree(d_g); 
    cudaFree(d_b);
    cudaFree(d_out_r); 
    cudaFree(d_out_g); 
    cudaFree(d_out_b);

    free(r_copy); 
    free(g_copy); 
    free(b_copy);

    free_ppm(&img);

    return EXIT_SUCCESS;
}