// -multi_work_per_thread -> (til p31k1) 262.33 s
// before adding omp and unroll

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    FILE* file = fopen(infile, "rb");
    fread(&N, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    // Padding to FW_BZ size
    n = N + FW_BZ - (N % FW_BZ);
    // Allocate host memory
    h_Dist = (int*)malloc(n*n*sizeof(int));
    
    // Initialize distances
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
                h_Dist[i*n+j] = 0;
            else
                h_Dist[i*n+j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i)
    {
        fread(pair, sizeof(int), 3, file);
        h_Dist[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName)
{
    FILE* outfile = fopen(outFileName, "w");

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
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
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
    for (int k = 0; k < FW_BZ; k++)
    {
        for(int di = 0; di < WORK_PER_THREAD; di++)
        {
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
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
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
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
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
    for (int k = 0; k < FW_BZ; k++)
    {
        for(int di = 0; di < WORK_PER_THREAD; di++)
        {
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
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
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
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
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
    for (int k = 0; k < FW_BZ; k++)
    {
        for(int di = 0; di < WORK_PER_THREAD; di++)
        {
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
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
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
__global__ void phase3_kernel(int *d_Dist, int n, int round)
{
    // Shared memory for the block
    __shared__ int sh_ik_Dist[FW_BZ][FW_BZ];
    __shared__ int sh_kj_Dist[FW_BZ][FW_BZ];

    int result[WORK_PER_THREAD][WORK_PER_THREAD];
    const int pivot_start = round * FW_BZ;
    int local_i, local_j, global_i, global_j, p_global_j, p_global_i;

    // Load data into shared memory
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;
            p_global_i = pivot_start + local_i;
            p_global_j = pivot_start + local_j;
            global_i = blockIdx.y * FW_BZ + local_i;
            global_j = blockIdx.x * FW_BZ + local_j;

            sh_ik_Dist[local_i][local_j] = d_Dist[global_i*n+p_global_j];
            sh_kj_Dist[local_i][local_j] = d_Dist[p_global_i*n+global_j];
            result[di][dj] = d_Dist[global_i*n+global_j];
        }
    }

    __syncthreads();

    // Process FW_BZ elements for this block
    for (int k = 0; k < FW_BZ; k++)
    {
        for(int di = 0; di < WORK_PER_THREAD; di++)
        {
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
    for (int di = 0; di < WORK_PER_THREAD; di++)
    {
        for (int dj = 0; dj < WORK_PER_THREAD; dj++)
        {
            local_i = threadIdx.y + di * CUDA_BZ;
            local_j = threadIdx.x + dj * CUDA_BZ;
            global_i = blockIdx.y * FW_BZ + local_i;
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
    
    for (int r = 0; r < round; ++r) {
        phase1_kernel<<<1, block>>>(d_Dist, n, r);
        phase2_kernel_row<<<dim3(round, 1), block>>>(d_Dist, n, r);
        phase2_kernel_col<<<dim3(round, 1), block>>>(d_Dist, n, r);
        phase3_kernel<<<dim3(round, round), block>>>(d_Dist, n, r);
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