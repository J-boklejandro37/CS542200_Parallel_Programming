// 2-D blockDim (16, 16), shared memory -> 6.74 ms
// Put MASK_N loop in the most inner loop -> 5.45 ms
// (16, 16) -> 5.4 ms
// (32, 16) -> 4.45 ms
// (32, 32) -> 4.71 ms
// (64, 16) -> 4.73 ms, 4.04 ms
// (128, 8) -> 4.49 ms, 4.10 ms
// (256, 4) -> 4.71 ms, 4.10 ms
// (512, 2) -> 4.65 ms

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define xBound MASK_X / 2
#define yBound MASK_Y / 2
#define SCALE 8
#define BLOCK_WIDTH 256
#define BLOCK_HEIGHT 4

/* Hint 7 */
// this variable is used by device
// Since the matrix is stored in transposed state, it can be accessed with [x][y] and not [y][x]
__constant__ int mask[MASK_N][MASK_X][MASK_Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__shared__ unsigned char sh_mem[BLOCK_HEIGHT + 2*yBound][BLOCK_WIDTH + 2*xBound][3];
__device__ void move_x_to_shmem(unsigned char* s, int gy, int gx, unsigned char (*arr)[BLOCK_WIDTH + 2*xBound][3], int ly, int lx, unsigned height, unsigned width, unsigned channels)
{
    if (gy < 0 || gy >= height || gx < 0 || gx >= width) return;
    #define global_mem_idx(gy, gx, ch) (ch + channels * (gx + width * gy))
    if (lx == 0)
    {
        for (int i = 0; i >= -xBound; --i)
        {
            if (gx + i >= 0)
            {
                sh_mem[ly+yBound][lx+i+xBound][2] = s[global_mem_idx(gy,gx+i,2)];
                sh_mem[ly+yBound][lx+i+xBound][1] = s[global_mem_idx(gy,gx+i,1)];
                sh_mem[ly+yBound][lx+i+xBound][0] = s[global_mem_idx(gy,gx+i,0)];
            }
        }
    }
    else if (lx == blockDim.x - 1)
    {
        for (int i = 0; i <= xBound; ++i)
        {
            if (gx + i < width)
            {
                sh_mem[ly+yBound][lx+i+xBound][2] = s[global_mem_idx(gy,gx+i,2)];
                sh_mem[ly+yBound][lx+i+xBound][1] = s[global_mem_idx(gy,gx+i,1)];
                sh_mem[ly+yBound][lx+i+xBound][0] = s[global_mem_idx(gy,gx+i,0)];
            }
        }
    }
    else
    {
        sh_mem[ly+yBound][lx+xBound][2] = s[global_mem_idx(gy,gx,2)];
        sh_mem[ly+yBound][lx+xBound][1] = s[global_mem_idx(gy,gx,1)];
        sh_mem[ly+yBound][lx+xBound][0] = s[global_mem_idx(gy,gx,0)];
    }
}

/* Hint 5 */
// this function is called by host and executed by device
__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    // Global y, x
    const int gy = blockDim.y * blockIdx.y + threadIdx.y;
    const int gx = blockDim.x * blockIdx.x + threadIdx.x;
    // Local y, x with sh_mem
    const int ly = threadIdx.y;
    const int lx = threadIdx.x;

    // Return if outside image bounds
    if (gx >= width || gy >= height) return;

    if (ly == 0)
    {
        for (int i = 0; i >= -yBound; --i)
        {
            if (gy + i >= 0)
            {
                move_x_to_shmem(s, gy + i, gx, sh_mem, ly + i, lx, height, width, channels);
            }
        }
    }
    else if (ly == blockDim.y - 1)
    {
        for (int i = 0; i <= yBound; ++i)
        {
            if (gy + i < height)
            {
                move_x_to_shmem(s, gy + i, gx, sh_mem, ly + i, lx, height, width, channels);
            }
        }
    }
    else
    {
        move_x_to_shmem(s, gy, gx, sh_mem, ly, lx, height, width, channels);
    }
    
    // Synchronize threads before computation
    __syncthreads();

    // Each thread processes one pixel
    int  i, v, u;
    int  R, G, B;
    float val[MASK_N*3] = {0.0f};

    //Process the pixel using the original sobel logic
    for (v = -yBound; v <= yBound; ++v) {
        for (u = -xBound; u <= xBound; ++u) {
            if ((gx + u) >= 0 && (gx + u) < width && gy + v >= 0 && gy + v < height) {
                R = sh_mem[ly+v+yBound][lx+u+xBound][2];
                G = sh_mem[ly+v+yBound][lx+u+xBound][1];
                B = sh_mem[ly+v+yBound][lx+u+xBound][0];
                for (i = 0; i < MASK_N; ++i) {
                    val[i*3+2] += R * mask[i][u + xBound][v + yBound];
                    val[i*3+1] += G * mask[i][u + xBound][v + yBound];
                    val[i*3+0] += B * mask[i][u + xBound][v + yBound];
                }
            }
        }
    }

    // Calculate final values
    float totalR = sqrt(val[2] * val[2] + val[5] * val[5]) / SCALE;
    float totalG = sqrt(val[1] * val[1] + val[4] * val[4]) / SCALE;
    float totalB = sqrt(val[0] * val[0] + val[3] * val[3]) / SCALE;

    // Write results to output image
    t[channels * (width * gy + gx) + 2] = (totalR > 255.0) ? 255 : totalR;
    t[channels * (width * gy + gx) + 1] = (totalG > 255.0) ? 255 : totalG;
    t[channels * (width * gy + gx) + 0] = (totalB > 255.0) ? 255 : totalB;
}

int main(int argc, char** argv) {

    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char* host_s = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels);
    unsigned char* host_t = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));
    
    /* Hint 1*/
    // cudaMalloc(...) for device src and device dst
    unsigned char *device_s, *device_t;
    cudaMalloc(&device_s, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&device_t, height * width * channels * sizeof(unsigned char));

    /* Hint 2*/
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(device_s, host_s, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    /* Hint 3 */
    // acclerate this function
    // Set up grid and block dimensions
    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    // Launch kernel
    sobel<<<gridDim, blockDim>>>(device_s, device_t, height, width, channels);
    
    /* Hint 4 */
    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(host_t, device_t, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Clean up device memory
    cudaFree(device_s);
    cudaFree(device_t);

    // export image
    write_png(argv[2], host_t, height, width, channels);

    // Clean up host memory
    free(host_s);
    free(host_t);

    return 0;
}
