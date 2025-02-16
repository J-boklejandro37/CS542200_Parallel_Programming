// test sram size

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#define PAD 1

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

__device__ __managed__ int B, N, d;
float *Q, *K, *V, *O;

void input(char *input_filename)
{
    int fd = open(input_filename, O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);
    
    void* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    char* ptr = (char*)data;
    
    memcpy(&B, ptr, sizeof(int)); ptr += sizeof(int);
    memcpy(&N, ptr, sizeof(int)); ptr += sizeof(int);
    memcpy(&d, ptr, sizeof(int)); ptr += sizeof(int);
    
    size_t matrix_size = N * d * sizeof(float);
    size_t tensor_size = B * N * d * sizeof(float);
    Q = (float*)aligned_alloc(64, tensor_size); 
    K = (float*)aligned_alloc(64, tensor_size);
    V = (float*)aligned_alloc(64, tensor_size);
    O = (float*)aligned_alloc(64, tensor_size);
    memset(O, 0, tensor_size);
    
    #pragma unroll
    for (int i = 0; i < B; i++) {
        memcpy(Q + (i * N * d), ptr, matrix_size); ptr += matrix_size;
        memcpy(K + (i * N * d), ptr, matrix_size); ptr += matrix_size;
        memcpy(V + (i * N * d), ptr, matrix_size); ptr += matrix_size;
    }
    
    munmap(data, sb.st_size);
    close(fd);
}

void output(char *output_filename)
{
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);
    fclose(file);

    free(Q); free(K); free(V); free(O);
}

void cuda_flash_attention()
{   
    /*--------------------------- Malloc and Memcpy ---------------------------*/
    // float *d_Q, *d_K, *d_V, *d_O;

    // size_t tensor_size = B * N * d * sizeof(float);

    // cudaMalloc(&d_Q, tensor_size);
    // cudaMalloc(&d_K, tensor_size);
    // cudaMalloc(&d_V, tensor_size);
    // cudaMalloc(&d_O, tensor_size);

    // cudaMemcpy(d_Q, Q, tensor_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_K, K, tensor_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_V, V, tensor_size, cudaMemcpyHostToDevice);

    /*--------------------------- Define block factor ---------------------------*/
    const int Bc = 15;  // Block colum size
    const int Br = 128;  // Block row size
    printf("Br: %d, Bc: %d, PAD: %d\n", Br, Bc, PAD);
    // const int Tc = (N + Bc - 1) / Bc;  // Tile row (number of blocks)
    // const int Tr = (N + Br - 1) / Br;  // Tile col (number of blocks)
    const float softmax_scale = 1.0 / sqrt(d);

    /*--------------------------- Define sram ---------------------------*/
    const int sram_size = (2 * Bc * (d + PAD) * sizeof(float)) +
                          (1 * Br * (d + PAD) * sizeof(float)) +
                          (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d\n", max_sram_size, sram_size);

    /*--------------------------- Kernel launch ---------------------------*/
    // grid.X: Tr, how many blocks to do
    // grid.Y: B, for batch size
    // dim3 grid_dim(Tr, B);
    // block.X: Br, how many rows in a block
    // dim3 block_dim(Br);

    // flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>
    //     (d_Q, d_K, d_V, d_O, Bc, Br, Tc, Tr, softmax_scale);

    /*--------------------------- Memcpy and Free ---------------------------*/
    // cudaMemcpy(O, d_O, tensor_size, cudaMemcpyDeviceToHost);
    // cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);
    printf("N: %d, d: %d\n", N, d);
    cuda_flash_attention();
    output(argv[2]);

    return 0;
}
