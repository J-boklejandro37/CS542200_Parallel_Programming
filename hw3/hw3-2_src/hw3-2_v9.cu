// --cudaStream -> not faster

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <omp.h>

#define min(a,b) (a < b ? a : b)
#define ceil(a,b) ((a + b - 1) / b)
#define FW_BZ 78
#define CUDA_BZ 26
#define WORK_PER_THREAD (FW_BZ / CUDA_BZ)

__device__ const int INF = ((1 << 30) - 1);

int n, m, N;
int* h_Dist;

void input(char* infile)
{
    // Open file and get size
    int fd = open(infile, O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);
    void* mapped = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    // Read N and m
    int* data = (int*)mapped;
    N = data[0];
    m = data[1];
    
    // Calculate padded size
    n = N + FW_BZ - (N % FW_BZ);
    
    // Allocate aligned memory
    h_Dist = (int*)aligned_alloc(32, n * n * sizeof(int));
    
    // Initialize distances using SIMD
    #pragma unroll
    #pragma omp parallel for simd
    for (int i = 0; i < n*n; ++i)
        h_Dist[i] = (i/n == i%n) ? 0 : INF;
    
    // Read edge data efficiently
    int* edges = data + 2;  // Skip N and m
    #pragma unroll
    #pragma omp parallel for
    for (int i = 0; i < m; ++i)
    {
        int src = edges[i*3];
        int dst = edges[i*3 + 1];
        int weight = edges[i*3 + 2];
        h_Dist[src*n + dst] = weight;
    }
    
    munmap(mapped, sb.st_size);
    close(fd);
}

void output(char* outFileName)
{
    FILE* outfile = fopen(outFileName, "w");

    #pragma unroll
    for (int i = 0; i < N; ++i)
        fwrite(&h_Dist[i * n], sizeof(int), N, outfile);

    fclose(outfile);
    free(h_Dist);
}

// Phase 1 kernel: Process diagonal block
__global__ void phase1_kernel(int *d_Dist, int n, int round)
{
    // Shared memory for the block
    __shared__ int sh_Dist[FW_BZ][FW_BZ];

    const int global_start = round * FW_BZ;

    int local_i, local_j, global_i, global_j;

    // Load data into shared memory
    #pragma unroll
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        #pragma unroll
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;
            global_i = global_start + local_i;
            global_j = global_start + local_j;

            sh_Dist[local_i][local_j] = d_Dist[global_i*n+global_j];
        }
    }

    __syncthreads();
    
    // Process FW_BZ elements for this block
    #pragma unroll
    for (int k = 0; k < FW_BZ; k++)
    {
        #pragma unroll
        for(int di = 0; di < WORK_PER_THREAD; di++)
        {
            #pragma unroll
            for(int dj = 0; dj < WORK_PER_THREAD; dj++)
            {
                local_i = threadIdx.y + di * CUDA_BZ;
                local_j = threadIdx.x + dj * CUDA_BZ;

                sh_Dist[local_i][local_j] = min(sh_Dist[local_i][local_j],
                                                sh_Dist[local_i][k] + sh_Dist[k][local_j]);
            }
        }
    }

    // Load data back to global memory
    #pragma unroll
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        #pragma unroll
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;
            global_i = global_start + local_i;
            global_j = global_start + local_j;

            d_Dist[global_i*n+global_j] = sh_Dist[local_i][local_j];
        }
    }
}

// Phase 2 kernel: Process row blocks
__global__ void phase2_kernel_row(int *d_Dist, int n, int round)
{
    // Shared memory for the block
    __shared__ int sh_pivot_Dist[FW_BZ][FW_BZ];
    __shared__ int sh_Dist[FW_BZ][FW_BZ];

    const int pivot_start = round * FW_BZ;

    int local_i, local_j, global_i, global_j;
    int p_global_i, p_global_j;

    // Load data into shared memory
    #pragma unroll
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        #pragma unroll
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;

            p_global_i = pivot_start + local_i;
            p_global_j = pivot_start + local_j;
            sh_pivot_Dist[local_i][local_j] = d_Dist[p_global_i*n+p_global_j];

            global_i = p_global_i;
            global_j = blockIdx.x * FW_BZ + local_j;
            sh_Dist[local_i][local_j] = d_Dist[global_i*n+global_j];
        }
    }

    __syncthreads();

    // Process FW_BZ elements for this block
    #pragma unroll
    for (int k = 0; k < FW_BZ; k++)
    {
        #pragma unroll
        for(int di = 0; di < WORK_PER_THREAD; di++)
        {
            #pragma unroll
            for(int dj = 0; dj < WORK_PER_THREAD; dj++)
            {
                local_i = threadIdx.y + di * CUDA_BZ;
                local_j = threadIdx.x + dj * CUDA_BZ;

                sh_Dist[local_i][local_j] = min(sh_Dist[local_i][local_j],
                                                sh_pivot_Dist[local_i][k] + sh_Dist[k][local_j]);
            }
        }
    }
    
    // Load data back to global memory
    #pragma unroll
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        #pragma unroll
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;

            global_i = pivot_start + local_i;
            global_j = blockIdx.x * FW_BZ + local_j;
            d_Dist[global_i*n+global_j] = sh_Dist[local_i][local_j];
        }
    }
}

// Phase 2 kernel: Process column blocks
__global__ void phase2_kernel_col(int *d_Dist, int n, int round)
{
    // Shared memory for the block
    __shared__ int sh_pivot_Dist[FW_BZ][FW_BZ];
    __shared__ int sh_Dist[FW_BZ][FW_BZ];

    const int pivot_start = round * FW_BZ;

    int local_i, local_j, global_i, global_j;
    int p_global_i, p_global_j;

    // Load data into shared memory
    #pragma unroll
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        #pragma unroll
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;

            p_global_i = pivot_start + local_i;
            p_global_j = pivot_start + local_j;
            sh_pivot_Dist[local_i][local_j] = d_Dist[p_global_i*n+p_global_j];

            global_i = blockIdx.x * FW_BZ + local_i;
            global_j = p_global_j;
            sh_Dist[local_i][local_j] = d_Dist[global_i*n+global_j];
        }
    }

    __syncthreads();

    // Process FW_BZ elements for this block
    #pragma unroll
    for (int k = 0; k < FW_BZ; k++)
    {
        #pragma unroll
        for(int di = 0; di < WORK_PER_THREAD; di++)
        {
            #pragma unroll
            for(int dj = 0; dj < WORK_PER_THREAD; dj++)
            {
                local_i = threadIdx.y + di * CUDA_BZ;
                local_j = threadIdx.x + dj * CUDA_BZ;

                sh_Dist[local_i][local_j] = min(sh_Dist[local_i][local_j],
                                                sh_Dist[local_i][k] + sh_pivot_Dist[k][local_j]);
            }
        }
    }
    
    // Load data back to global memory
    #pragma unroll
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        #pragma unroll
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;

            global_i = blockIdx.x * FW_BZ + local_i;
            global_j = pivot_start + local_j;
            d_Dist[global_i*n+global_j] = sh_Dist[local_i][local_j];
        }
    }
}

// Phase 3 kernel: Process all blocks
__global__ void phase3_kernel(int *d_Dist, int n, int round, int row)
{
    // Shared memory for the block
    __shared__ int sh_ik_Dist[FW_BZ][FW_BZ];
    __shared__ int sh_kj_Dist[FW_BZ][FW_BZ];

    register int result[WORK_PER_THREAD][WORK_PER_THREAD];
    const int pivot_start = round * FW_BZ;
    int local_i, local_j, global_i, global_j, p_global_j, p_global_i;

    // Load data into shared memory
    #pragma unroll
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        #pragma unroll
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;
            p_global_i = pivot_start + local_i;
            p_global_j = pivot_start + local_j;
            global_i = row * FW_BZ + local_i; // Row is distributed by stream ID
            global_j = blockIdx.x * FW_BZ + local_j;

            sh_ik_Dist[local_i][local_j] = d_Dist[global_i*n+p_global_j];
            sh_kj_Dist[local_i][local_j] = d_Dist[p_global_i*n+global_j];
            result[di][dj] = d_Dist[global_i*n+global_j];
        }
    }

    __syncthreads();

    // Process FW_BZ elements for this block
    #pragma unroll
    for (int k = 0; k < FW_BZ; k++)
    {
        #pragma unroll
        for(int di = 0; di < WORK_PER_THREAD; di++)
        {
            #pragma unroll
            for(int dj = 0; dj < WORK_PER_THREAD; dj++)
            {
                local_i = threadIdx.y + di * CUDA_BZ;
                local_j = threadIdx.x + dj * CUDA_BZ;

                result[di][dj] = min(result[di][dj],
                                     sh_ik_Dist[local_i][k] + sh_kj_Dist[k][local_j]);
            }
        }
    }

    // Load data back to global memory
    #pragma unroll
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        #pragma unroll
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;
            global_i = row * FW_BZ + local_i;
            global_j = blockIdx.x * FW_BZ + local_j;

            d_Dist[global_i*n+global_j] = result[di][dj];
        }
    }

}

void block_FW()
{
    int* d_Dist;

    cudaMalloc(&d_Dist, n*n*sizeof(int));
    cudaMemcpy(d_Dist, h_Dist, n*n*sizeof(int), cudaMemcpyHostToDevice);

    const int round = n / FW_BZ;
    
    dim3 block(CUDA_BZ, CUDA_BZ);

    // Create streams for phase 3
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for(int stream = 0; stream < num_streams; stream++) {
        cudaStreamCreate(&streams[stream]);
    }
    
    #pragma unroll
    for (int r = 0; r < round; ++r) {
        phase1_kernel<<<1, block>>>(d_Dist, n, r);
        phase2_kernel_row<<<dim3(round, 1), block>>>(d_Dist, n, r);
        phase2_kernel_col<<<dim3(round, 1), block>>>(d_Dist, n, r);
        // Distribute phase3 work across streams
        for(int row = 0; row < round; row++)
            phase3_kernel<<<dim3(round, 1), block, 0, streams[row % num_streams]>>>(d_Dist, n, r, row);
    }

    // Synchronize and cleanup
    for(int stream = 0; stream < num_streams; stream++) {
        cudaStreamSynchronize(streams[stream]);
        cudaStreamDestroy(streams[stream]);
    }

    cudaMemcpy(h_Dist, d_Dist, n*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);
}

// Main function remains mostly the same for now
int main(int argc, char* argv[])
{
    input(argv[1]);
    block_FW();
    output(argv[2]);
    
    return 0;
}