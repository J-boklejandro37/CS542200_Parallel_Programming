// testcase t20
// -shared_memory -unroll -> 2.22 s

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

__global__
void flash_attention_kernel(
    const float* __restrict__ d_Q, const float*__restrict__  d_K,
    const float* __restrict__ d_V, float* __restrict__ d_O,
    const int Bc, const int Br, const int Tc, const int Tr, const float softmax_scale)
{
    int local_y = threadIdx.x;  // local index of row
    int tile_idx_y = blockIdx.x;  // tile block index
    int batch_idx = blockIdx.y;  // batch index

    int global_y = tile_idx_y * Br + local_y;
    if (global_y >= N)
        return;

    /*--------------------------- Initialize ---------------------------*/
    // Offset into Q,K,V,O,l,m - different for each batch
    int batch_offset = batch_idx * N * d;

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int q_tile_size = Br * d;
    int k_tile_size = Bc * d;
    int v_tile_size = Bc * d;
    float* Qi = sram;  // q_tile_size
    float* Ki = sram + q_tile_size;  // k_tile_size
    float* Vi = sram + q_tile_size + k_tile_size; // v_tile_size
    float* Sij = sram + q_tile_size + k_tile_size + v_tile_size; // (Br * Bc) size

    /*--------------------------- Copy q_tile_size into shared memory ---------------------------*/
    const float* gl_iter = d_Q + batch_offset + global_y * d;
    float* sh_iter = Qi + local_y * d;

    #pragma unroll
    for (int x = 0; x < d; x++) {
        *(sh_iter + x) = *(gl_iter + x);
    }

    float li = 0.0f;
    float mi = -INFINITY;  // Record max Sij

    /*--------------------------- Iterate through Tc with Bc size ---------------------------*/
    // Process horizontally for K^T and vertically for V
    // IMPORTANT: This for loop gradually finishes Si, Pi and Oi (one Bc at a time)
    #pragma unroll
    for (int tile_idx_x = 0; tile_idx_x < Tc; tile_idx_x++)
    {
        /*--------------------------- Load Kj, Vj to SRAM ---------------------------*/
        // Since Bc might be larger then Br (num_threads), use a for loop to do it
        #pragma unroll
        for (int y = local_y; y < Bc; y += Br) {
            // K^T goes by Bc in direction of Q's column (X-axis), thus use tile_idx_x * Bc
            // It's still K's row direction (Y-axis), thus use k_global_y 
            int k_global_y = tile_idx_x * Bc + y;
            if (k_global_y < N) {
                const float* gl_iter_k = d_K + batch_offset + k_global_y * d;
                const float* gl_iter_v = d_V + batch_offset + k_global_y * d;

                float* sh_iter_k = Ki + y * d;
                float* sh_iter_v = Vi + y * d;

                #pragma unroll
                for (int x = 0; x < d; x++) {
                    *(sh_iter_k + x) = *(gl_iter_k + x);
                    *(sh_iter_v + x) = *(gl_iter_v + x);
                }
            }
        }
        __syncthreads();  // such that the inner loop can use the correct Ki, Vj

        /*--------------------------- Caltulation ---------------------------*/
        int tile_width = fminf(Bc, N - tile_idx_x * Bc);
        int local_max = -INFINITY;        

        // Calculate S = Q * K^T
        #pragma unroll
        for (int j = 0; j < tile_width; j++) {
            float sij = 0.0f;
            #pragma unroll
            for (int k = 0; k < d; k++) {
                sij += Qi[local_y * d + k] * Ki[j * d + k];
            }
            Sij[local_y * Bc + j] = sij * softmax_scale;
            local_max = fmaxf(local_max, Sij[local_y * Bc + j]);
        }

        // mi_tilede: warp max
        float mi_tilde = __shfl_sync(0xFFFFFFFF, local_max, 0);
        // li_tilde: partial sum
        float li_tilde = 0.0f;

        #pragma unroll
        for (int j = 0; j < tile_width; j++) {
            Sij[local_y * Bc + j] = __expf(Sij[local_y * Bc + j] - mi_tilde);
            li_tilde += Sij[local_y * Bc + j];
        }

        // mi: mi-1
        // mi_new: mi
        float mi_new = fmaxf(mi, mi_tilde);
        // li: li-1
        // li_new: li
        float li_new = __expf(mi - mi_new) * li + __expf(mi_tilde - mi_new) * li_tilde;

        // Incrementally update d_O
        #pragma unroll
        for (int x = 0; x < d; x++) {
            float pv = 0.0f;

            #pragma unroll
            for (int j = 0; j < tile_width; j++) {
                pv += Sij[local_y * Bc + j] * Vi[j * d + x];
            }

            d_O[batch_offset + global_y * d + x] = (1 / li_new) * 
                ((li * __expf(mi - mi_new) * d_O[batch_offset + global_y * d + x]) +
                __expf(mi_tilde - mi_new) * pv);
        }

        li = li_new;
        mi = mi_new;
        __syncthreads();

    }
}

void cuda_flash_attention()
{   
    /*--------------------------- Malloc and Memcpy ---------------------------*/
    float *d_Q, *d_K, *d_V, *d_O;

    size_t tensor_size = B * N * d * sizeof(float);

    cudaMalloc(&d_Q, tensor_size);
    cudaMalloc(&d_K, tensor_size);
    cudaMalloc(&d_V, tensor_size);
    cudaMalloc(&d_O, tensor_size);

    cudaMemcpy(d_Q, Q, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, tensor_size, cudaMemcpyHostToDevice);

    /*--------------------------- Define block factor ---------------------------*/
    const int Bc = 32;  // Block colum size
    const int Br = 32;  // Block row size
    const int Tc = (N + Bc - 1) / Bc;  // Tile row (number of blocks)
    const int Tr = (N + Br - 1) / Br;  // Tile col (number of blocks)
    const float softmax_scale = 1.0 / sqrt(d);

    /*--------------------------- Define sram ---------------------------*/
    const int sram_size = (2 * Bc * d * sizeof(float)) +
                          (1 * Br * d * sizeof(float)) +
                          (Bc * Br * sizeof(float));
    // int max_sram_size;
    // cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    // printf("Max shared memory: %d, requested shared memory: %d\n", max_sram_size, sram_size);

    /*--------------------------- Kernel launch ---------------------------*/
    // grid.X: Tr, how many blocks to do
    // grid.Y: B, for batch size
    dim3 grid_dim(Tr, B);
    // block.X: Br, how many rows in a block
    dim3 block_dim(Br);

    flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>
        (d_Q, d_K, d_V, d_O, Bc, Br, Tc, Tr, softmax_scale);

    /*--------------------------- Memcpy and Free ---------------------------*/
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
