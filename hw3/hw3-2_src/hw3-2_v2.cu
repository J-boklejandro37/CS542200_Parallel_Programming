// 1D h_Dist
// testcase c18.1
// -partial -fw_bz=512 -> 1.27 s
// -partial+full -fw_bz=512 -> 1.17 s
// -partial -fw_bz=512 -shared -> 1.42 s
// -partial+full -fw_bz=512 -shared -> 1.32 s
// -partial+full -fw_bz=1024 -shared -> 1.77 s

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

__global__ void process_block_each_full_shared(int* d_Dist, int N, int k, int block_internal_start_i, int block_internal_start_j) {
    __shared__ int sharedDist_ik[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    __shared__ int sharedDist_kj[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = block_internal_start_i + blockIdx.y * blockDim.y + ty;
    int j = block_internal_start_j + blockIdx.x * blockDim.x + tx;
    
    // Load data into shared memory
    sharedDist_ik[ty][tx] = d_Dist[i * N + k];
    sharedDist_kj[ty][tx] = d_Dist[k * N + j];
    __syncthreads();
    
    // Process only valid indices
    d_Dist[i * N + j] = min(d_Dist[i * N + j], sharedDist_ik[ty][tx] + sharedDist_kj[ty][tx]);
}

__global__ void process_block_each_partial_shared(int* d_Dist, int N, int k, int block_internal_start_i, int block_internal_start_j)
{
    __shared__ int sharedDist_ik[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    __shared__ int sharedDist_kj[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = block_internal_start_i + blockIdx.y * blockDim.y + ty;
    int j = block_internal_start_j + blockIdx.x * blockDim.x + tx;
    
    // Load data into shared memory
    if (i < N && k < N)
        sharedDist_ik[ty][tx] = d_Dist[i * N + k];

    if (k < N && j < N)
        sharedDist_kj[ty][tx] = d_Dist[k * N + j];

    __syncthreads();
    
    // Process only valid indices
    if (i < N && j < N)
        d_Dist[i * N + j] = min(d_Dist[i * N + j], sharedDist_ik[ty][tx] + sharedDist_kj[ty][tx]);
}

void multiple_kernel_launch(int b_i, int b_j, int k_start, int k_end)
{
    int block_internal_start_i = b_i * FW_BLOCK_SIZE;
    int block_internal_start_j = b_j * FW_BLOCK_SIZE;
    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 gridDim(ceil(FW_BLOCK_SIZE, CUDA_BLOCK_SIZE), ceil(FW_BLOCK_SIZE, CUDA_BLOCK_SIZE));

    for (int k = k_start; k < k_end; ++k)
    {
        if ((b_i + 1) * FW_BLOCK_SIZE < N && (b_j + 1) * FW_BLOCK_SIZE < N)
            process_block_each_full_shared<<<gridDim, blockDim>>>(d_Dist, N, k, block_internal_start_i, block_internal_start_j);
        else
            process_block_each_partial_shared<<<gridDim, blockDim>>>(d_Dist, N, k, block_internal_start_i, block_internal_start_j);
    }
}

// __global__ void process_block_all(int* d_Dist, int N, int k_start, int k_end, int block_internal_start_i, int block_internal_start_j)
// {
//     int i = block_internal_start_i + blockIdx.y * blockDim.y + threadIdx.y;
//     int j = block_internal_start_j + blockIdx.x * blockDim.x + threadIdx.x;

//     for (int k = k_start; k < k_end; ++k)
//     {
//         if (i < N && j < N)
//         {
//             int ij = i * N + j;
//             int ik = i * N + k;
//             int kj = k * N + j;
//             int new_dist = d_Dist[ik] + d_Dist[kj];
//             d_Dist[ij] = min(d_Dist[ij], new_dist);
//         }
//         __syncthreads();
//     }
// }

// void one_kernel_launch(int b_i, int b_j, int k_start, int k_end)
// {
//     int block_internal_start_i = b_i * FW_BLOCK_SIZE;
//     int block_internal_start_j = b_j * FW_BLOCK_SIZE;
//     dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
//     dim3 gridDim(ceil(FW_BLOCK_SIZE, CUDA_BLOCK_SIZE), ceil(FW_BLOCK_SIZE, CUDA_BLOCK_SIZE));

//     process_block_all<<<gridDim, blockDim>>>(d_Dist, N, k_start, k_end, block_internal_start_i, block_internal_start_j);
// }

void calculate_region(int round, int block_start_i, int block_start_j, int block_width, int block_height)
{
    int block_end_i = block_start_i + block_height;
    int block_end_j = block_start_j + block_width;

    int k_start = round * FW_BLOCK_SIZE;
    int k_end = min((round + 1) * FW_BLOCK_SIZE, N);

    // #pragma omp parallel for collapse(2) schedule(static)
    for (int b_i = block_start_i; b_i < block_end_i; ++b_i)
    {
        for (int b_j = block_start_j; b_j < block_end_j; ++b_j)
        {
            multiple_kernel_launch(b_i, b_j, k_start, k_end);
            // one_kernel_launch(b_i, b_j, k_start, k_end); // FW_BLOCK_SIZE must <= CUDA_BLOCK_SIZE
        }
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