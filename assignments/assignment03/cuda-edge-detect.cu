// ![Result of the Sobel operator](edge-detect.png)

// The [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator) is
// used to detect the edges on an grayscale image. The idea is to compute
// the gradient of color change across each pixel; those pixels for which
// the gradient exceeds a user-defined threshold are considered to be
// part of an edge. Computation of the gradient involves the application
// of a $3 \times 3$ stencil to the input image.

// The program reads an input image fro standard input in
// [PGM](https://en.wikipedia.org/wiki/Netpbm#PGM_example) (_Portable
// Graymap_) format and produces a B/W image to standard output. The user
// can specify an optional threshold on the command line.

// The goal of this exercise is to parallelize the computation of the
// Sobel operator using CUDA; this can be achieved by writing a kernel
// that computes the edge at each pixel, and invoke the kernel from the
// `edge_detect()` function.

// To compile:

//         nvcc cuda-edge-detect.cu -o cuda-edge-detect

// To execute:

//         ./cuda-edge-detect [threshold] < input > output

// Example:

//         ./cuda-edge-detect < BWstop-sign.pgm > BWstop-sign-edges.pgm

// ## Files

// - [cuda-edge-detect.cu](cuda-edge-detect.cu) [hpc.h](hpc.h)
// - [BWstop-sign.pgm](BWstop-sign.pgm)

// ***/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"
#include <string.h>

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    unsigned char *bmap; /* buffer of width*height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

const unsigned char WHITE = 255;
const unsigned char BLACK = 0;

/**
 * Initialize a PGM_image object: allocate space for a bitmap of size
 * `width` x `height`, and set all pixels to color `col`
 */
void init_pgm( PGM_image *img, int width, int height, unsigned char col )
{
    int i, j;

    assert(img != NULL);

    img->width = width;
    img->height = height;
    img->maxgrey = 255;
    img->bmap = (unsigned char*)malloc(width*height);
    assert(img->bmap != NULL);
    for (i=0; i<height; i++) {
        for (j=0; j<width; j++) {
            img->bmap[i*width + j] = col;
        }
    }
}

/**
 * Read a PGM file from file `f`. Warning: this function is not
 * robust: it may fail on legal PGM images, and may crash on invalid
 * files since no proper error checking is done.
 */
void read_pgm( FILE *f, PGM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
#if _XOPEN_SOURCE < 600
    img->bmap = (unsigned char*)malloc((img->width)*(img->height)*sizeof(unsigned char));
#else
    /* The pointer img->bmap must be properly aligned to allow aligned
       SIMD load/stores to work. */
    int ret = posix_memalign((void**)&(img->bmap), __BIGGEST_ALIGNMENT__, (img->width)*(img->height));
    assert( 0 == ret );
#endif
    assert(img->bmap != NULL);
    /* Get the binary data from the file */
    nread = fread(img->bmap, 1, (img->width)*(img->height), f);
    if ( (img->width)*(img->height) != nread ) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n", (img->width)*(img->height), nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write the image `img` to file `f`; if not NULL, use the string
 * `comment` as metadata.
 */
void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width)*(img->height), f);
}

/**
 * Free the bitmap associated with image `img`; note that the
 * structure pointed to by `img` is NOT deallocated; only `img->bmap`
 * is.
 */
void free_pgm( PGM_image *img )
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL; /* not necessary */
    img->width = img->height = img->maxgrey = -1;
}

int IDX(int i, int j, int width)
{
    return (i*width + j);
}


/**
 * Edge detection using the Sobel operator
 */
void edge_detect( const PGM_image* in, PGM_image* edges, int threshold )
{
    const int width = in->width;
    const int height = in->height;
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++)  {
            /* Compute the gradients Gx and Gy along the x and y
               dimensions */
            const int Gx =
                in->bmap[IDX(i-1, j-1, width)] - in->bmap[IDX(i-1, j+1, width)]
                + 2*in->bmap[IDX(i, j-1, width)] - 2*in->bmap[IDX(i, j+1, width)]
                + in->bmap[IDX(i+1, j-1, width)] - in->bmap[IDX(i+1, j+1, width)];
            const int Gy =
                in->bmap[IDX(i-1, j-1, width)] + 2*in->bmap[IDX(i-1, j, width)] + in->bmap[IDX(i-1, j+1, width)]
                - in->bmap[IDX(i+1, j-1, width)] - 2*in->bmap[IDX(i+1, j, width)] - in->bmap[IDX(i+1, j+1, width)];
            const int magnitude = Gx * Gx + Gy * Gy;
            if  (magnitude > threshold*threshold)
                edges->bmap[IDX(i, j, width)] = WHITE;
            else
                edges->bmap[IDX(i, j, width)] = BLACK;
        }
    }
}

#define TILE_W 32
#define TILE_H 32

__global__ void edge_detect_kernel(const unsigned char *in, unsigned char *out, int width, int height, int threshold) {
    // Shared memory for a 32x32 tile + 1 pixel halo on all 4 sides AND corners
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

    if (tx == 0 && x > 0) smem[ty + 1][0] = in[y * width + (x - 1)];
    if (tx == TILE_W - 1 && x < width - 1) smem[ty + 1][TILE_W + 1] = in[y * width + (x + 1)];
    if (ty == 0 && y > 0) smem[0][tx + 1] = in[(y - 1) * width + x];
    if (ty == TILE_H - 1 && y < height - 1) smem[TILE_H + 1][tx + 1] = in[(y + 1) * width + x];

    if (tx == 0 && ty == 0 && x > 0 && y > 0) 
        smem[0][0] = in[(y - 1) * width + (x - 1)];
    if (tx == TILE_W - 1 && ty == 0 && x < width - 1 && y > 0) 
        smem[0][TILE_W + 1] = in[(y - 1) * width + (x + 1)];
    if (tx == 0 && ty == TILE_H - 1 && x > 0 && y < height - 1) 
        smem[TILE_H + 1][0] = in[(y + 1) * width + (x - 1)];
    if (tx == TILE_W - 1 && ty == TILE_H - 1 && x < width - 1 && y < height - 1) 
        smem[TILE_H + 1][TILE_W + 1] = in[(y + 1) * width + (x + 1)];

    __syncthreads();

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        
        int gx = smem[ty][tx] - smem[ty][tx+2] 
               + 2 * smem[ty+1][tx] - 2 * smem[ty+1][tx+2] 
               + smem[ty+2][tx] - smem[ty+2][tx+2];
               
        int gy = smem[ty][tx] + 2 * smem[ty][tx+1] + smem[ty][tx+2] 
               - smem[ty+2][tx] - 2 * smem[ty+2][tx+1] - smem[ty+2][tx+2];

        int magnitude = gx * gx + gy * gy;
        
        if (magnitude > threshold * threshold) {
            out[y * width + x] = 255; // WHITE edge
        } else {
            out[y * width + x] = 0;   // BLACK background
        }
    }
}

int main( int argc, char* argv[] )
{
    PGM_image bmap, out_cpu, out_gpu;
    int threshold = 70;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [threshold] < in.pgm > out.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }
    if ( argc > 1 ) {
        threshold = atoi(argv[1]);
    }
    read_pgm(stdin, &bmap);
    
    init_pgm(&out_cpu, bmap.width, bmap.height, WHITE);
    init_pgm(&out_gpu, bmap.width, bmap.height, WHITE);

    // --- CPU Execution ---
    fprintf(stderr, "Starting CPU Edge Detection...\n");
    const double cpu_start = hpc_gettime();
    
    edge_detect(&bmap, &out_cpu, threshold);
    
    const double cpu_elapsed = hpc_gettime() - cpu_start;
    fprintf(stderr, "CPU Execution time: %.6f seconds\n", cpu_elapsed);

    // --- GPU (CUDA) Execution ---
    int img_size = bmap.width * bmap.height * sizeof(unsigned char);
    unsigned char *d_in, *d_out;

    cudaMalloc((void**)&d_in, img_size);
    cudaMalloc((void**)&d_out, img_size);

    cudaMemcpy(d_in, bmap.bmap, img_size, cudaMemcpyHostToDevice);
    
    cudaMemset(d_out, WHITE, img_size);

    dim3 blockSize(TILE_W, TILE_H);
    dim3 gridSize((bmap.width + TILE_W - 1) / TILE_W, (bmap.height + TILE_H - 1) / TILE_H);

    fprintf(stderr, "Starting GPU Edge Detection...\n");
    const double gpu_start = hpc_gettime();
    
    edge_detect_kernel<<<gridSize, blockSize>>>(d_in, d_out, bmap.width, bmap.height, threshold);
    
    cudaDeviceSynchronize();
    const double gpu_elapsed = hpc_gettime() - gpu_start;
    
    fprintf(stderr, "GPU Execution time: %.6f seconds\n", gpu_elapsed);
    fprintf(stderr, "Speedup: %.2fx\n", cpu_elapsed / gpu_elapsed);

    cudaMemcpy(out_gpu.bmap, d_out, img_size, cudaMemcpyDeviceToHost);

    write_pgm(stdout, &out_gpu, "produced by cuda-edge-detect.cu");
    
    cudaFree(d_in); 
    cudaFree(d_out);
    free_pgm(&bmap);
    free_pgm(&out_cpu);
    free_pgm(&out_gpu);

    return EXIT_SUCCESS;
}
