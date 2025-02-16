//

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

#define _max(a,b) (a > b ? a : b)
#define _min(a,b) (a < b ? a : b)

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
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

void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar) {
    for (int i = 0; i < br; i++) {
        for (int j = 0; j < bc; j++) {
            out[i * bc + j] = 0.0F;
            for (int t = 0; t < d; t++) {
                out[i * bc + j] += q[i * d + t] * k[j * d + t];
            }
            out[i * bc + j] *= scalar;
        }
    }
}

void RowMax(float *out, float *in, int br, int bc) {
    for (int i = 0; i < br; i++) {
        out[i] = in[i * bc];
        for (int j = 0; j < bc; j++) {
            out[i] = _max(out[i], in[i * bc + j]);
        }
    }
}

void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc) {
    for (int i = 0; i < br; i++) {
        for (int j = 0; j < bc; j++) {
            out[i * bc + j] = exp(in[i * bc + j] - mx[i]);
        }
    }
}

void RowSum(float *out, float *in, int br, int bc) {
    for (int i = 0; i < br; i++) {
        out[i] = 0.0F;
        for (int j = 0; j < bc; j++) {
            out[i] += in[i * bc + j];
        }
    }
}

void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc) {
    float *mi_new = (float *)malloc(br * sizeof(float));
    float *li_new = (float *)malloc(br * sizeof(float));

    for (int i = 0; i < br; i++) {
        mi_new[i] = _max(mi[i], mij[i]);
        li_new[i] = exp(mi[i] - mi_new[i]) * li[i] + exp(mij[i] - mi_new[i]) * lij[i];
    }

    for (int i = 0; i < br; i++) {
        for (int j = 0; j < d; j++) {
            float pv = 0.0F;
            for (int t = 0; t < bc; t++) {
                pv += pij[i * bc + t] * vj[t * d + j];
            }
            oi[i * d + j] = (li[i] * exp(mi[i] - mi_new[i]) * oi[i * d + j] + exp(mij[i] - mi_new[i]) * pv) / li_new[i];
        }
    }

    memcpy(mi, mi_new, br * sizeof(float));
    memcpy(li, li_new, br * sizeof(float));
    
    free(mi_new);
    free(li_new);
}

void flash_attention(float *q, float *k, float *v, float *o) {
    float *l = (float *)malloc(N * sizeof(float));
    float *m = (float *)malloc(N * sizeof(float));
    memset(l, 0x00, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        m[i] = FLT_MIN;
    }

    int br = 32, bc = 32;
    int tr = N / br, tc = N / bc;
    float *kj = (float *)malloc(bc * d * sizeof(float));
    float *vj = (float *)malloc(bc * d * sizeof(float));
    float *qi = (float *)malloc(br * d * sizeof(float));
    float *oi = (float *)malloc(br * d * sizeof(float));
    float *li = (float *)malloc(br * sizeof(float));
    float *mi = (float *)malloc(br * sizeof(float));

    float *sij = (float *)malloc(br * bc * sizeof(float));
    float *pij = (float *)malloc(br * bc * sizeof(float));
    float *mij = (float *)malloc(br * sizeof(float));
    float *lij = (float *)malloc(br * sizeof(float));

    for (int j = 0; j < tc; j++) {
        memcpy(kj, k + j * bc * d, bc * d * sizeof(float));
        memcpy(vj, v + j * bc * d, bc * d * sizeof(float));
        for (int i = 0; i < tr; i++) {
            memcpy(qi, q + i * br * d, br * d * sizeof(float));
            memcpy(oi, o + i * br * d, br * d * sizeof(float));
            memcpy(li, l + i * br, br * sizeof(float));
            memcpy(mi, m + i * br, br * sizeof(float));

            QKDotAndScalar(sij, qi, kj, br, bc, 1.0 / sqrt(d));
            RowMax(mij, sij, br, bc);
            MinusMaxAndExp(pij, sij, mij, br, bc);
            RowSum(lij, pij, br, bc);

            UpdateMiLiOi(mi, li, oi, mij, lij, pij, vj, br, bc);

            memcpy(o + i * br * d, oi, br * d * sizeof(float));
            memcpy(l + i * br, li, br * sizeof(float));
            memcpy(m + i * br, mi, br * sizeof(float));
        }
    }

    free(sij);
    free(pij);
    free(mij);
    free(lij);

    free(kj);
    free(vj);
    free(qi);
    free(oi);
    free(li);
    free(mi);

    free(l);
    free(m);
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
    cudaMemset(d_O, 0, tensor_size);

    /*--------------------------- Memcpy and Free ---------------------------*/

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
    cuda_flash_attention();
    output(argv[2]);

    return 0;
}
