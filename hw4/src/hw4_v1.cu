// testcase t20
// 2D baseline

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

__global__
void flash_attention_kernel(
    const float* __restrict__ d_Q, const float* __restrict__ d_K,
    const float* __restrict__ d_V, float* __restrict__ d_O,
    const float softmax_scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    
    if (row >= N) return;
    
    int batch_offset = batch * N * d;
    float mi = -INFINITY;
    float li = 0.0f;
    
    // Calculate max for numerical stability
    for(int j = 0; j < N; j++) {
        float sij = 0.0f;
        for(int k = 0; k < d; k++) {
            sij += d_Q[batch_offset + row * d + k] * d_K[batch_offset + j * d + k];
        }
        mi = fmaxf(mi, sij * softmax_scale);
    }

    // Calculate denominator
    for(int j = 0; j < N; j++) {
        float sij = 0.0f;
        for(int k = 0; k < d; k++) {
            sij += d_Q[batch_offset + row * d + k] * d_K[batch_offset + j * d + k];
        }
        li += __expf(sij * softmax_scale - mi);
    }

    // Calculate output
    for(int k = 0; k < d; k++) {
        float acc = 0.0f;
        for(int j = 0; j < N; j++) {
            float sij = 0.0f;
            for(int l = 0; l < d; l++) {
                sij += d_Q[batch_offset + row * d + l] * d_K[batch_offset + j * d + l];
            }
            acc += (__expf(sij * softmax_scale - mi) / li) * d_V[batch_offset + j * d + k];
        }
        d_O[batch_offset + row * d + k] = acc;
    }
}

void cuda_flash_attention() {
    float *d_Q, *d_K, *d_V, *d_O;
    size_t tensor_size = B * N * d * sizeof(float);

    cudaMalloc(&d_Q, tensor_size);
    cudaMalloc(&d_K, tensor_size);
    cudaMalloc(&d_V, tensor_size);
    cudaMalloc(&d_O, tensor_size);

    cudaMemcpy(d_Q, Q, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, tensor_size, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 1024;
    const float softmax_scale = 1.0f / sqrt(d);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, B);

    flash_attention_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, softmax_scale);

    cudaMemcpy(O, d_O, tensor_size, cudaMemcpyDeviceToHost);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
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
