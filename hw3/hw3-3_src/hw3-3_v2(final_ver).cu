// For submission and profiling
// Only copy one block row between each round
// c06.1 -> 35.07 s
//       -> 34.55 s (unroll round)
// c07.1 -> 48.49 s
//       -> 48.45 s (unroll round)

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
#define MAX_GPU_COUNT 2

__device__ const int INF = ((1 << 30) - 1);

int n, m, N;
int* h_Dist;

void checkCudaError()
{
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess)
   {
       printf("CUDA error: %s\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }
}

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
    
    // Pinned memory
    cudaMallocHost((void**)&h_Dist, n*n*sizeof(int));
    
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
    cudaFreeHost(h_Dist);
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

            global_i = blockIdx.y * FW_BZ + local_i;
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

            global_i = blockIdx.y * FW_BZ + local_i;
            global_j = pivot_start + local_j;
            d_Dist[global_i*n+global_j] = sh_Dist[local_i][local_j];
        }
    }
}

// Phase 3 kernel: Process all blocks
__global__ void phase3_kernel(int *d_Dist, int n, int round, int row_start)
{
    // Shared memory for the block
    __shared__ int sh_ik_Dist[FW_BZ][FW_BZ];
    __shared__ int sh_kj_Dist[FW_BZ][FW_BZ];

    int result[WORK_PER_THREAD][WORK_PER_THREAD];
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
            global_i = (row_start + blockIdx.y) * FW_BZ + local_i;
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
            global_i = (row_start + blockIdx.y) * FW_BZ + local_i;
            global_j = blockIdx.x * FW_BZ + local_j;

            d_Dist[global_i*n+global_j] = result[di][dj];
        }
    }

}

void block_FW()
{
    /*----------------------------initialize----------------------------*/
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    // printf("Number of gpus detected: %d\n", num_gpus);
    // fflush(stdout);

    int* d_Dist[MAX_GPU_COUNT];
    const int round = n / FW_BZ;
    dim3 block(CUDA_BZ, CUDA_BZ);
    const int blocks_per_gpu = round / 2;
    
    /*----------------------------Memory copy----------------------------*/
    for (int device_no = 0; device_no < num_gpus; device_no++)
    {
        cudaSetDevice(device_no);
        cudaMalloc(&d_Dist[device_no], n*n*sizeof(int));
        cudaMemcpy(d_Dist[device_no], h_Dist, n*n*sizeof(int), cudaMemcpyHostToDevice);
    }
    
    #pragma unroll
    for (int r = 0; r < round; ++r)
    {
        /*----------------------------Transfer pivot block row----------------------------*/
        int pivot_offset = r * FW_BZ * n;
        if (r < blocks_per_gpu) {
            cudaMemcpy(d_Dist[1] + pivot_offset, d_Dist[0] + pivot_offset,
					   n*FW_BZ*sizeof(int), cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy(d_Dist[0] + pivot_offset, d_Dist[1] + pivot_offset,
					   n*FW_BZ*sizeof(int), cudaMemcpyDeviceToDevice);
        }

        /*----------------------------Calculate phase----------------------------*/
        for (int device_no = 0; device_no < num_gpus; device_no++)
        {
            cudaSetDevice(device_no);
            /*----------------------------Phase 1 kernel----------------------------*/
            phase1_kernel<<<1, block>>>(d_Dist[device_no], n, r);

            /*----------------------------Phase 2 kernel----------------------------*/
            phase2_kernel_row<<<dim3(round, 1), block>>>
                (d_Dist[device_no], n, r);
            phase2_kernel_col<<<dim3(1, round), block>>>
                (d_Dist[device_no], n, r);

            /*----------------------------Phase 3 kernel----------------------------*/
            if (device_no == 0) {
                phase3_kernel<<<dim3(round, blocks_per_gpu), block>>>
                    (d_Dist[0], n, r, 0);
            } else {
                phase3_kernel<<<dim3(round, round - blocks_per_gpu), block>>>
                    (d_Dist[1], n, r, blocks_per_gpu);
            }
        }

        /*----------------------------Wait for kernel to complete----------------------------*/
        for (int device_no = 0; device_no < num_gpus; device_no++)
        {
            cudaSetDevice(device_no);
            cudaDeviceSynchronize();
        }
    }

    /*----------------------------Memory copy back----------------------------*/
    cudaSetDevice(0);
    cudaMemcpy(h_Dist, d_Dist[0],
               blocks_per_gpu*FW_BZ*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(h_Dist + blocks_per_gpu*FW_BZ*n, d_Dist[1] + blocks_per_gpu*FW_BZ*n,
               (round-blocks_per_gpu)*FW_BZ*n*sizeof(int), cudaMemcpyDeviceToHost);

    /*----------------------------Clean up----------------------------*/
    for (int device_no = 0; device_no < num_gpus; device_no++) {
        cudaFree(d_Dist[device_no]);
    }   
}

// Main function remains mostly the same for now
int main(int argc, char* argv[])
{
    input(argv[1]);
    block_FW();
    output(argv[2]);
    
    return 0;
}