// -shared_memory
// testcase 18.1
// -block_size=32 -shared_memory -> 0.57 s
// testcase p12k1
// -block_size=32 -shared_memory -> 5.03 s

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define min(a,b) (a < b ? a : b)
#define ceil(a,b) ((a + b - 1) / b)
#define B 32
#define BLOCK_DIM 32  // CUDA thread block dimension

const int INF = ((1 << 30) - 1);
// const int V = 50010;

int n, m;
static int *Dist;        // Changed to 1D array for better CUDA compatibility
static int *d_Dist;      // Device array

// Phase 1 kernel: Process diagonal block
__global__ void phase1_kernel(int *d_Dist, int n, int round)
{
    // Shared memory for the block
    __shared__ int shared[BLOCK_DIM][BLOCK_DIM];

    int local_i = threadIdx.y;
    int local_j = threadIdx.x;
    
    // Calculate starting position for this round's block
    int block_start = round * B;
    
    // Calculate actual x,y coordinates this thread will process
    int global_i = block_start + local_i;
    int global_j = block_start + local_j;

    if (global_i >= n || global_j >= n) return;

    // Load data into shared memory
    shared[local_i][local_j] = d_Dist[global_i * n + global_j];
    __syncthreads();
    
    // Process B elements for this block
    int k_end = min(B, n - block_start);
    for (int k = 0; k < k_end; k++)
    {
        shared[local_i][local_j] = min(shared[local_i][local_j], shared[local_i][k] + shared[k][local_j]);
        __syncthreads();
    }
    
    // Write back to global memory
    d_Dist[global_i * n + global_j] = shared[local_i][local_j];
}

// Phase 2 kernel: Process row blocks
__global__ void phase2_kernel_row(int *d_Dist, int n, int round)
{
    // Shared memory for the block
    __shared__ int shared_pivot[BLOCK_DIM][BLOCK_DIM];
    __shared__ int shared_local[BLOCK_DIM][BLOCK_DIM];

    int block_j = blockIdx.x;
    if (block_j == round) return;  // Skip pivot block
    
    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    // Calculate global indices
    int pivot_start = round * B;
    int pivot_global_i = pivot_start + local_i;
    int pivot_global_j = pivot_start + local_j;
    
    int block_j_start = block_j * B;
    int global_i = pivot_start + local_i;
    int global_j = block_j_start + local_j;
    
    // Load pivot block and current block into shared memory
    if (global_i < n && global_j < n)
        shared_local[local_i][local_j] = d_Dist[global_i * n + global_j];

    if (pivot_global_i < n && pivot_global_j < n)
        shared_pivot[local_i][local_j] = d_Dist[pivot_global_i * n + pivot_global_j];

    __syncthreads();

    if (global_i >= n || global_j >= n) return;

    int k_end = min(B, n - pivot_start);
    for (int k = 0; k < k_end; k++)
    {
        shared_local[local_i][local_j] = min(shared_local[local_i][local_j], shared_pivot[local_i][k] + shared_local[k][local_j]);
        __syncthreads();
    }
    
    // Write back to global memory
    d_Dist[global_i * n + global_j] = shared_local[local_i][local_j];
}

// Phase 2 kernel: Process column blocks
__global__ void phase2_kernel_col(int *d_Dist, int n, int round)
{
    // Shared memory for the block
    __shared__ int shared_pivot[BLOCK_DIM][BLOCK_DIM];
    __shared__ int shared_local[BLOCK_DIM][BLOCK_DIM];

    int block_i = blockIdx.y;
    if (block_i == round) return;  // Skip pivot block
    
    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    // Calculate global indices
    int pivot_start = round * B;
    int pivot_global_i = pivot_start + local_i;
    int pivot_global_j = pivot_start + local_j;
    
    int block_i_start = block_i * B;
    int global_i = block_i_start + local_i;
    int global_j = pivot_start + local_j;
    
    // Load pivot block and current block into shared memory
    if (global_i < n && global_j < n)
        shared_local[local_i][local_j] = d_Dist[global_i * n + global_j];

    if (pivot_global_i < n && pivot_global_j < n)
        shared_pivot[local_i][local_j] = d_Dist[pivot_global_i * n + pivot_global_j];

    __syncthreads();

    if (global_i >= n || global_j >= n) return;

    int k_end = min(B, n - pivot_start);
    for (int k = 0; k < k_end; k++)
    {
        shared_local[local_i][local_j] = min(shared_local[local_i][local_j], shared_local[local_i][k] + shared_pivot[k][local_j]);
        __syncthreads();
    }
    
    // Write back to global memory
    d_Dist[global_i * n + global_j] = shared_local[local_i][local_j];
}

// Phase 3 kernel: Process remaining blocks
__global__ void phase3_kernel(int *d_Dist, int n, int round)
{
    // Shared memory for the block
    __shared__ int shared_ik[BLOCK_DIM][BLOCK_DIM];
    __shared__ int shared_kj[BLOCK_DIM][BLOCK_DIM];

    int block_i = blockIdx.y;
    int block_j = blockIdx.x;
    if (block_i == round && block_j == round) return;  // Skip pivot block
    
    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    int pivot_start = round * B;
    int block_i_start = block_i * B;
    int block_j_start = block_j * B;

    int global_i = block_i_start + local_i;
    int global_j = block_j_start + local_j;
    int pivot_global_i = pivot_start + local_i;
    int pivot_global_j = pivot_start + local_j;

    if (global_i < n && pivot_global_j < n)
    {
        shared_ik[local_i][local_j] = d_Dist[global_i * n + pivot_global_j];
    }
    if (global_j < n && pivot_global_i < n)
    {
        shared_kj[local_i][local_j] = d_Dist[pivot_global_i * n + global_j];
    }
    __syncthreads();

    if (global_i >= n || global_j >= n) return;

    int min_dist = d_Dist[global_i * n + global_j];
    int k_end = min(B, n - pivot_start);
    for (int k = 0; k < k_end; k++)
    {
        min_dist = min(min_dist, shared_ik[local_i][k] + shared_kj[k][local_j]);
    }
    
    d_Dist[global_i * n + global_j] = min_dist;

}

void block_FW()
{
    int round = ceil(n, B);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    
    for (int r = 0; r < round; ++r)
    {
        // Phase 1: Process diagonal block
        phase1_kernel<<<1, block_dim>>>(d_Dist, n, r);
        
        // Phase 2: Process row and column blocks
        phase2_kernel_row<<<dim3(round, 1), block_dim>>>(d_Dist, n, r);
        phase2_kernel_col<<<dim3(1, round), block_dim>>>(d_Dist, n, r);
        
        // Phase 3: Process remaining blocks
        phase3_kernel<<<dim3(round, round), block_dim>>>(d_Dist, n, r);
    }
}

void input(char* infile)
{
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    // Allocate host memory
    Dist = (int*)malloc(n * n * sizeof(int));
    
    // Initialize distances
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
                Dist[i*n+j] = 0;
            else
                Dist[i*n+j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i)
    {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);
    
    // Allocate and copy to device memory
    cudaMalloc(&d_Dist, n * n * sizeof(int));
    cudaMemcpy(d_Dist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
}

void output(char* outFileName)
{
    // Copy back to host
    cudaMemcpy(Dist, d_Dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            Dist[i*n+j] = min(Dist[i*n+j], INF);
        }
        fwrite(&Dist[i*n], sizeof(int), n, outfile);
    }
    fclose(outfile);
    
    // Cleanup
    free(Dist);
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