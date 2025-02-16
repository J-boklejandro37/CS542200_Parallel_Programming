// distributed at calculate_region -> testcase18: 1.42(wrong answer)

#include <cuda.h>
#include <sched.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define ceil(a, b) ((a + b - 1) / b)
#define min(a,b) (a < b ? a : b)

#define FW_BLOCK_SIZE 512
#define CUDA_BLOCK_SIZE 32
#define DEV_NO 0
cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);
int N, M, NUM_THREADS;
int* h_Dist;
int* d_Dist;

void input(char* infile)
{
    FILE* file = fopen(infile, "rb");
    fread(&N, sizeof(int), 1, file);
    fread(&M, sizeof(int), 1, file);
    h_Dist = (int*)malloc(N * N * sizeof(int));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * N; ++i)
    {
        if (i / N == i % N)
            h_Dist[i] = 0;
        else
            h_Dist[i] = INF;
    }

    int pair[3];
    for (int i = 0; i < M; ++i)
    {
        fread(pair, sizeof(int), 3, file);
        h_Dist[pair[0] * N + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outfile)
{
    FILE* file = fopen(outfile, "w");
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            h_Dist[i*N+j] = min(h_Dist[i*N+j], INF);
        }
        fwrite(&h_Dist[i*N], sizeof(int), N, file);
    }
    fclose(file);
}

__global__ void process_block_each_full(int* d_Dist, int N, int k, int block_internal_start_i, int block_internal_start_j)
{
    int i = block_internal_start_i + blockIdx.y * blockDim.y + threadIdx.y;
    int j = block_internal_start_j + blockIdx.x * blockDim.x + threadIdx.x;
    int ij = i * N + j;
    int ik = i * N + k;
    int kj = k * N + j;
    int new_dist = d_Dist[ik] + d_Dist[kj];
    d_Dist[ij] = min(d_Dist[ij], new_dist);
}

__global__ void process_block_each_partial(int* d_Dist, int N, int k, int block_internal_start_i, int block_internal_start_j)
{
    int i = block_internal_start_i + blockIdx.y * blockDim.y + threadIdx.y;
    int j = block_internal_start_j + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N)
    {
        int ij = i * N + j;
        int ik = i * N + k;
        int kj = k * N + j;
        int new_dist = d_Dist[ik] + d_Dist[kj];
        d_Dist[ij] = min(d_Dist[ij], new_dist);
    }
}

void calculate_region(int round, int block_start_i, int block_start_j, int block_width, int block_height)
{
    int k_start = round * FW_BLOCK_SIZE;
    int k_end = min((round + 1) * FW_BLOCK_SIZE, N);

    int global_start_i = block_start_i * FW_BLOCK_SIZE;
    int global_start_j = block_start_j * FW_BLOCK_SIZE;

    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 gridDim(ceil(block_height * FW_BLOCK_SIZE, CUDA_BLOCK_SIZE), ceil(block_width * FW_BLOCK_SIZE, CUDA_BLOCK_SIZE));

    for (int k = k_start; k < k_end; ++k)
    {
        process_block_each_partial<<<gridDim, blockDim>>>(d_Dist, N, k, global_start_i, global_start_j);
    }
}

void blocked_FW()
{
    int round = ceil(N, FW_BLOCK_SIZE);

    for (int r = 0; r < round; ++r)
    {
        /* Phase 1*/
        calculate_region(r, r, r, 1, 1);
        
        /* Phase 2*/
        calculate_region(r, r, 0, r, 1);
        calculate_region(r, r, r + 1, round - r - 1, 1);
        calculate_region(r, 0, r, 1, r);
        calculate_region(r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        calculate_region(r, 0, 0, r, r);
        calculate_region(r, 0, r + 1, round - r - 1, r);
        calculate_region(r, r + 1, 0, r, round - r - 1);
        calculate_region(r, r + 1, r + 1, round - r - 1, round - r - 1); 
    }
}

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NUM_THREADS = CPU_COUNT(&cpu_set);

    cudaSetDevice(DEV_NO);

    input(argv[1]);
    
    // Copy data to GPU
    cudaMalloc((void**)&d_Dist, N * N * sizeof(int));
    cudaMemcpy(d_Dist, h_Dist, N * N * sizeof(int), cudaMemcpyHostToDevice);

    blocked_FW();

    // Copy data back main memory
    cudaMemcpy(h_Dist, d_Dist, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    output(argv[2]);

    // Free device memory
    cudaFree(d_Dist);
    
    return 0;
}