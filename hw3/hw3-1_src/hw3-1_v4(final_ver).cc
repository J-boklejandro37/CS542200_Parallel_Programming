// For submission
// There's no AVX-512 instruction here, thus use SSE2
// Original -> 55.38 s
// SSE2 -unroll=1 -B=512 -omp=inner -> 23.44 s
// SSE2 -unroll=2 -B=512 -omp=inner -> 21.44 s
// SSE2 -unroll=4 -B=512 -omp=inner -> 20.49 s
// SSE2 -unroll=8 -B=512 -omp=inner -> 21.29 s
// SSE2 -unroll=4 -B=64  -omp=outer -> 18.04 s (best)
// SSE2 -unroll=8 -B=64  -omp=inner -> 18.99 s

#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

#define CHUNK_SIZE 10
#define min(a,b) (a < b ? a : b)
#define ceil(a, b) ((a + b - 1) / b)

const int INF = ((1 << 30) - 1);
const int V = 10000;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
void normal_FW();
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);
void process_block_sse2(int block_internal_start_x, int block_internal_end_x,
                        int block_internal_start_y, int block_internal_end_y, int k);
void process_block_sse2_unroll2(int block_internal_start_x, int block_internal_end_x,
                        int block_internal_start_y, int block_internal_end_y, int k);
void process_block_sse2_unroll4(int block_internal_start_x, int block_internal_end_x,
                        int block_internal_start_y, int block_internal_end_y, int k);
void process_block_sse2_unroll4_collapse2(int block_internal_start_x, int block_internal_end_x,
                        int block_internal_start_y, int block_internal_end_y, int k);
void process_block_sse2_unroll8(int block_internal_start_x, int block_internal_end_x,
                        int block_internal_start_y, int block_internal_end_y, int k) ;
void process_block_sse2_unroll2row(int block_internal_start_x, int block_internal_end_x,
                        int block_internal_start_y, int block_internal_end_y, int k);

int n, m, NUM_THREADS;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    input(argv[1]);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NUM_THREADS = CPU_COUNT(&cpu_set);

    int B = 64;
    block_FW(B);

    output(argv[2]);

    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Dist[i][j] = min(INF, Dist[i][j]);
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void block_FW(int B)
{
    int round = ceil(n, B);

    // #pragma omp parallel num_threads(4)
    // {
        for (int r = 0; r < round; ++r)
        {
            // #pragma omp single
            // {
                // printf("%d %d\n", r, round);
                fflush(stdout);
                /* Phase 1*/
                cal(B, r, r, r, 1, 1);
            // }
            
            /* Phase 2*/
            // #pragma omp sections
            // {
                // #pragma omp section
                cal(B, r, r, 0, r, 1);
                // #pragma omp section
                cal(B, r, r, r + 1, round - r - 1, 1);
                // #pragma omp section
                cal(B, r, 0, r, 1, r);
                // #pragma omp section
                cal(B, r, r + 1, r, 1, round - r - 1);
            // }

            /* Phase 3*/
            // #pragma omp sections
            // {
                // #pragma omp section
                cal(B, r, 0, 0, r, r);
                // #pragma omp section
                cal(B, r, 0, r + 1, round - r - 1, r);
                // #pragma omp section
                cal(B, r, r + 1, 0, r, round - r - 1);
                // #pragma omp section
                cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
            // }     
        }
    // }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    int k_start = Round * B; 
    int k_end = min((Round + 1) * B, n);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            for (int k = k_start; k < k_end; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = min((b_i + 1) * B, n);
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = min((b_j + 1) * B, n);

                process_block_sse2_unroll4(block_internal_start_x, block_internal_end_x,
                        block_internal_start_y, block_internal_end_y, k);
            }
        }
    }
}

void process_block_sse2(int block_internal_start_x, int block_internal_end_x,
                       int block_internal_start_y, int block_internal_end_y, int k) {
    // Number of integers that fit in a 128-bit vector (4 ints of 32 bits each)
    const int simd_width = 4;

    // #pragma omp parallel for schedule(static)
    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
        // Broadcast Dist[i][k] to all elements of the vector
        __m128i dist_ik = _mm_set1_epi32(Dist[i][k]);
        
        // Process 4 elements at a time
        for (int j = block_internal_start_y; j < block_internal_end_y - simd_width + 1; j += simd_width) {
            // Prefetch next cache line
            _mm_prefetch((const char*)&Dist[k][j + simd_width], _MM_HINT_T0);
            _mm_prefetch((const char*)&Dist[i][j + simd_width], _MM_HINT_T0);
            
            // Load 4 consecutive elements
            __m128i dist_kj = _mm_loadu_si128((const __m128i*)&Dist[k][j]);
            __m128i dist_ij = _mm_loadu_si128((const __m128i*)&Dist[i][j]);
            
            // Calculate Dist[i][k] + Dist[k][j]
            __m128i sum = _mm_add_epi32(dist_ik, dist_kj);
            
            // Compare sum < Dist[i][j]
            __m128i cmp = _mm_cmplt_epi32(sum, dist_ij);
            
            // Select values based on comparison
            __m128i result = _mm_or_si128(
                _mm_and_si128(cmp, sum),                // Take sum where cmp is true (all bits 1)
                _mm_andnot_si128(cmp, dist_ij)         // Take dist_ij where cmp is false (all bits 0)
            );
            
            // Store the result back to memory
            _mm_storeu_si128((__m128i*)&Dist[i][j], result);
        }

        // Handle remaining elements that don't fit in a vector
        for (int j = (block_internal_end_y - ((block_internal_end_y - block_internal_start_y) % simd_width));
             j < block_internal_end_y; ++j) {
            int new_dist = Dist[i][k] + Dist[k][j];
            if (new_dist < Dist[i][j]) {
                Dist[i][j] = new_dist;
            }
        }
    }
}

void process_block_sse2_unroll2(int block_internal_start_x, int block_internal_end_x,
                       int block_internal_start_y, int block_internal_end_y, int k) {
    // Process 8 integers at once using two SSE2 vectors
    const int simd_width = 8;

    // #pragma omp parallel for schedule(static)
    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
        // Broadcast Dist[i][k] to all elements of two vectors
        __m128i dist_ik = _mm_set1_epi32(Dist[i][k]);
        
        // Process 8 elements at a time (using two SSE2 vectors)
        for (int j = block_internal_start_y; j < block_internal_end_y - simd_width + 1; j += simd_width) {
            // Prefetch next cache lines
            _mm_prefetch((const char*)&Dist[k][j + simd_width], _MM_HINT_T0);
            _mm_prefetch((const char*)&Dist[i][j + simd_width], _MM_HINT_T0);
            
            // Load first 4 elements
            __m128i dist_kj_1 = _mm_loadu_si128((const __m128i*)&Dist[k][j]);
            __m128i dist_ij_1 = _mm_loadu_si128((const __m128i*)&Dist[i][j]);
            
            // Load next 4 elements
            __m128i dist_kj_2 = _mm_loadu_si128((const __m128i*)&Dist[k][j + 4]);
            __m128i dist_ij_2 = _mm_loadu_si128((const __m128i*)&Dist[i][j + 4]);
            
            // Calculate Dist[i][k] + Dist[k][j] for both vectors
            __m128i sum_1 = _mm_add_epi32(dist_ik, dist_kj_1);
            __m128i sum_2 = _mm_add_epi32(dist_ik, dist_kj_2);
            
            // Compare sum < Dist[i][j] for both vectors
            __m128i cmp_1 = _mm_cmplt_epi32(sum_1, dist_ij_1);
            __m128i cmp_2 = _mm_cmplt_epi32(sum_2, dist_ij_2);
            
            // Select values based on comparison for both vectors
            __m128i result_1 = _mm_or_si128(
                _mm_and_si128(cmp_1, sum_1),
                _mm_andnot_si128(cmp_1, dist_ij_1)
            );
            
            __m128i result_2 = _mm_or_si128(
                _mm_and_si128(cmp_2, sum_2),
                _mm_andnot_si128(cmp_2, dist_ij_2)
            );
            
            // Store both results back to memory
            _mm_storeu_si128((__m128i*)&Dist[i][j], result_1);
            _mm_storeu_si128((__m128i*)&Dist[i][j + 4], result_2);
        }

        // Handle remaining elements that don't fit in the 8-element blocks
        for (int j = (block_internal_end_y - ((block_internal_end_y - block_internal_start_y) % simd_width));
             j < block_internal_end_y; ++j) {
            int new_dist = Dist[i][k] + Dist[k][j];
            if (new_dist < Dist[i][j]) {
                Dist[i][j] = new_dist;
            }
        }
    }
}

void process_block_sse2_unroll4(int block_internal_start_x, int block_internal_end_x,
                       int block_internal_start_y, int block_internal_end_y, int k) {
    // Process 16 integers at once using four SSE2 vectors
    const int SIMD_WIDTH = 4;
    const int UNROLL = 4;

    // #pragma omp parallel for schedule(static)
    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
        // Broadcast Dist[i][k] to all elements of four vectors
        __m128i dist_ik = _mm_set1_epi32(Dist[i][k]);
        
        // Process 16 elements at a time (using four SSE2 vectors)
        int j;
        for (j = block_internal_start_y; j < (block_internal_end_y - SIMD_WIDTH * UNROLL + 1); j += SIMD_WIDTH * UNROLL) {
            
            // Load first group of 4 elements (0-3)
            __m128i dist_kj_1 = _mm_loadu_si128((const __m128i*)&Dist[k][j]);
            __m128i dist_ij_1 = _mm_loadu_si128((const __m128i*)&Dist[i][j]);
            
            // Load second group of 4 elements (4-7)
            __m128i dist_kj_2 = _mm_loadu_si128((const __m128i*)&Dist[k][j + 4]);
            __m128i dist_ij_2 = _mm_loadu_si128((const __m128i*)&Dist[i][j + 4]);
            
            // Load third group of 4 elements (8-11)
            __m128i dist_kj_3 = _mm_loadu_si128((const __m128i*)&Dist[k][j + 8]);
            __m128i dist_ij_3 = _mm_loadu_si128((const __m128i*)&Dist[i][j + 8]);
            
            // Load fourth group of 4 elements (12-15)
            __m128i dist_kj_4 = _mm_loadu_si128((const __m128i*)&Dist[k][j + 12]);
            __m128i dist_ij_4 = _mm_loadu_si128((const __m128i*)&Dist[i][j + 12]);
            
            // Calculate Dist[i][k] + Dist[k][j] for all four vectors
            __m128i sum_1 = _mm_add_epi32(dist_ik, dist_kj_1);
            __m128i sum_2 = _mm_add_epi32(dist_ik, dist_kj_2);
            __m128i sum_3 = _mm_add_epi32(dist_ik, dist_kj_3);
            __m128i sum_4 = _mm_add_epi32(dist_ik, dist_kj_4);
            
            // Compare sum < Dist[i][j] for all four vectors
            sum_1 = _mm_min_epi32(sum_1, dist_ij_1);
            sum_2 = _mm_min_epi32(sum_2, dist_ij_2);
            sum_3 = _mm_min_epi32(sum_3, dist_ij_3);
            sum_4 = _mm_min_epi32(sum_4, dist_ij_4);
            
            // Store all results back to memory
            _mm_storeu_si128((__m128i*)&Dist[i][j], sum_1);
            _mm_storeu_si128((__m128i*)&Dist[i][j + 4], sum_2);
            _mm_storeu_si128((__m128i*)&Dist[i][j + 8], sum_3);
            _mm_storeu_si128((__m128i*)&Dist[i][j + 12], sum_4);
        }

        // Handle remaining elements that don't fit in the 16-element blocks
        for (; j < block_internal_end_y; ++j) {
            int new_dist = Dist[i][k] + Dist[k][j];
            if (new_dist < Dist[i][j]) {
                Dist[i][j] = new_dist;
            }
        }
    }
}

void process_block_sse2_unroll4_collapse2(int block_internal_start_x, int block_internal_end_x,
                       int block_internal_start_y, int block_internal_end_y, int k) {
    // Process 16 integers at once using four SSE2 vectors
    const int simd_width = 16;
    int remainder_start = block_internal_end_y - ((block_internal_end_y - block_internal_start_y) % simd_width);

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static)
        for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
            for (int j = block_internal_start_y; j < remainder_start; j += simd_width) {
                // Broadcast Dist[i][k] to all elements of four vectors
                __m128i dist_ik = _mm_set1_epi32(Dist[i][k]);
                // Prefetch next cache lines
                _mm_prefetch((const char*)&Dist[k][j + simd_width], _MM_HINT_T0);
                _mm_prefetch((const char*)&Dist[i][j + simd_width], _MM_HINT_T0);
                _mm_prefetch((const char*)&Dist[k][j + simd_width + 8], _MM_HINT_T0);
                _mm_prefetch((const char*)&Dist[i][j + simd_width + 8], _MM_HINT_T0);
                
                // Load first group of 4 elements (0-3)
                __m128i dist_kj_1 = _mm_loadu_si128((const __m128i*)&Dist[k][j]);
                __m128i dist_ij_1 = _mm_loadu_si128((const __m128i*)&Dist[i][j]);
                
                // Load second group of 4 elements (4-7)
                __m128i dist_kj_2 = _mm_loadu_si128((const __m128i*)&Dist[k][j + 4]);
                __m128i dist_ij_2 = _mm_loadu_si128((const __m128i*)&Dist[i][j + 4]);
                
                // Load third group of 4 elements (8-11)
                __m128i dist_kj_3 = _mm_loadu_si128((const __m128i*)&Dist[k][j + 8]);
                __m128i dist_ij_3 = _mm_loadu_si128((const __m128i*)&Dist[i][j + 8]);
                
                // Load fourth group of 4 elements (12-15)
                __m128i dist_kj_4 = _mm_loadu_si128((const __m128i*)&Dist[k][j + 12]);
                __m128i dist_ij_4 = _mm_loadu_si128((const __m128i*)&Dist[i][j + 12]);
                
                // Calculate Dist[i][k] + Dist[k][j] for all four vectors
                __m128i sum_1 = _mm_add_epi32(dist_ik, dist_kj_1);
                __m128i sum_2 = _mm_add_epi32(dist_ik, dist_kj_2);
                __m128i sum_3 = _mm_add_epi32(dist_ik, dist_kj_3);
                __m128i sum_4 = _mm_add_epi32(dist_ik, dist_kj_4);
                
                // Compare sum < Dist[i][j] for all four vectors
                __m128i cmp_1 = _mm_cmplt_epi32(sum_1, dist_ij_1);
                __m128i cmp_2 = _mm_cmplt_epi32(sum_2, dist_ij_2);
                __m128i cmp_3 = _mm_cmplt_epi32(sum_3, dist_ij_3);
                __m128i cmp_4 = _mm_cmplt_epi32(sum_4, dist_ij_4);
                
                // Select values based on comparison for all four vectors
                __m128i result_1 = _mm_or_si128(
                    _mm_and_si128(cmp_1, sum_1),
                    _mm_andnot_si128(cmp_1, dist_ij_1)
                );
                
                __m128i result_2 = _mm_or_si128(
                    _mm_and_si128(cmp_2, sum_2),
                    _mm_andnot_si128(cmp_2, dist_ij_2)
                );
                
                __m128i result_3 = _mm_or_si128(
                    _mm_and_si128(cmp_3, sum_3),
                    _mm_andnot_si128(cmp_3, dist_ij_3)
                );
                
                __m128i result_4 = _mm_or_si128(
                    _mm_and_si128(cmp_4, sum_4),
                    _mm_andnot_si128(cmp_4, dist_ij_4)
                );
                
                // Store all results back to memory
                _mm_storeu_si128((__m128i*)&Dist[i][j], result_1);
                _mm_storeu_si128((__m128i*)&Dist[i][j + 4], result_2);
                _mm_storeu_si128((__m128i*)&Dist[i][j + 8], result_3);
                _mm_storeu_si128((__m128i*)&Dist[i][j + 12], result_4);
            }
        }

        // Handle remaining elements that don't fit in the 16-element blocks
        #pragma omp for schedule(static)
        for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
            for (int j = remainder_start; j < block_internal_end_y; ++j) {
                int new_dist = Dist[i][k] + Dist[k][j];
                if (new_dist < Dist[i][j]) {
                    Dist[i][j] = new_dist;
                }
            }
        }
    }
}

void process_block_sse2_unroll8(int block_internal_start_x, int block_internal_end_x,
                       int block_internal_start_y, int block_internal_end_y, int k) {
    // Process 32 integers at once using eight SSE2 vectors
    const int simd_width = 32;
    const int vectors_per_block = simd_width / 4;  // Each SSE2 vector handles 4 integers

    // #pragma omp parallel for schedule(static)
    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
        // Broadcast Dist[i][k] to all elements
        __m128i dist_ik = _mm_set1_epi32(Dist[i][k]);
        
        // Process 32 elements at a time
        for (int j = block_internal_start_y; j < block_internal_end_y - simd_width + 1; j += simd_width) {
            // Prefetch next cache lines
            for (int v = 0; v < vectors_per_block; v += 2) {
                _mm_prefetch((const char*)&Dist[k][j + simd_width + v * 4], _MM_HINT_T0);
                _mm_prefetch((const char*)&Dist[i][j + simd_width + v * 4], _MM_HINT_T0);
            }
            
            __m128i dist_kj[vectors_per_block];
            __m128i dist_ij[vectors_per_block];
            __m128i sum[vectors_per_block];
            __m128i cmp[vectors_per_block];
            __m128i result[vectors_per_block];
            
            // Load and process all vectors
            for (int v = 0; v < vectors_per_block; v++) {
                // Load 4 elements for current vector
                dist_kj[v] = _mm_loadu_si128((const __m128i*)&Dist[k][j + v * 4]);
                dist_ij[v] = _mm_loadu_si128((const __m128i*)&Dist[i][j + v * 4]);
                
                // Calculate sum and comparison
                sum[v] = _mm_add_epi32(dist_ik, dist_kj[v]);
                cmp[v] = _mm_cmplt_epi32(sum[v], dist_ij[v]);
                
                // Select values based on comparison
                result[v] = _mm_or_si128(
                    _mm_and_si128(cmp[v], sum[v]),
                    _mm_andnot_si128(cmp[v], dist_ij[v])
                );
                
                // Store result back to memory
                _mm_storeu_si128((__m128i*)&Dist[i][j + v * 4], result[v]);
            }
        }

        // Handle remaining elements that don't fit in the 32-element blocks
        for (int j = (block_internal_end_y - ((block_internal_end_y - block_internal_start_y) % simd_width));
             j < block_internal_end_y; ++j) {
            int new_dist = Dist[i][k] + Dist[k][j];
            if (new_dist < Dist[i][j]) {
                Dist[i][j] = new_dist;
            }
        }
    }
}

void process_block_sse2_unroll2row(int block_internal_start_x, int block_internal_end_x,
                              int block_internal_start_y, int block_internal_end_y, int k) {
    const int simd_width = 4;  // 4 integers per SSE2 vector

    // Process two rows at once, so we'll increment i by 2
    #pragma omp parallel for schedule(static)
    for (int i = block_internal_start_x; i < block_internal_end_x - 1; i += 2) {
        // Load Dist[i][k] and Dist[i+1][k] values
        __m128i dist_ik_1 = _mm_set1_epi32(Dist[i][k]);
        __m128i dist_ik_2 = _mm_set1_epi32(Dist[i + 1][k]);
        
        // Process 4 elements at a time for two rows simultaneously
        for (int j = block_internal_start_y; j < block_internal_end_y - simd_width + 1; j += simd_width) {
            // Prefetch next cache lines
            _mm_prefetch((const char*)&Dist[k][j + simd_width], _MM_HINT_T0);
            _mm_prefetch((const char*)&Dist[i][j + simd_width], _MM_HINT_T0);
            _mm_prefetch((const char*)&Dist[i + 1][j + simd_width], _MM_HINT_T0);
            
            // Load common Dist[k][j] values for both rows
            __m128i dist_kj = _mm_loadu_si128((const __m128i*)&Dist[k][j]);
            
            // Process first row (i)
            __m128i dist_ij_1 = _mm_loadu_si128((const __m128i*)&Dist[i][j]);
            __m128i sum_1 = _mm_add_epi32(dist_ik_1, dist_kj);
            __m128i cmp_1 = _mm_cmplt_epi32(sum_1, dist_ij_1);
            __m128i result_1 = _mm_or_si128(
                _mm_and_si128(cmp_1, sum_1),
                _mm_andnot_si128(cmp_1, dist_ij_1)
            );
            
            // Process second row (i+1)
            __m128i dist_ij_2 = _mm_loadu_si128((const __m128i*)&Dist[i + 1][j]);
            __m128i sum_2 = _mm_add_epi32(dist_ik_2, dist_kj);
            __m128i cmp_2 = _mm_cmplt_epi32(sum_2, dist_ij_2);
            __m128i result_2 = _mm_or_si128(
                _mm_and_si128(cmp_2, sum_2),
                _mm_andnot_si128(cmp_2, dist_ij_2)
            );
            
            // Store results for both rows
            _mm_storeu_si128((__m128i*)&Dist[i][j], result_1);
            _mm_storeu_si128((__m128i*)&Dist[i + 1][j], result_2);
        }

        // Handle remaining elements for both rows
        for (int j = (block_internal_end_y - ((block_internal_end_y - block_internal_start_y) % simd_width));
             j < block_internal_end_y; ++j) {
            // Process remaining elements for first row
            int new_dist_1 = Dist[i][k] + Dist[k][j];
            if (new_dist_1 < Dist[i][j]) {
                Dist[i][j] = new_dist_1;
            }
            
            // Process remaining elements for second row
            int new_dist_2 = Dist[i + 1][k] + Dist[k][j];
            if (new_dist_2 < Dist[i + 1][j]) {
                Dist[i + 1][j] = new_dist_2;
            }
        }
    }

    // Handle last row if block_internal_end_x is odd
    if (block_internal_end_x % 2 == 1) {
        int i = block_internal_end_x - 1;
        __m128i dist_ik = _mm_set1_epi32(Dist[i][k]);
        
        for (int j = block_internal_start_y; j < block_internal_end_y - simd_width + 1; j += simd_width) {
            __m128i dist_kj = _mm_loadu_si128((const __m128i*)&Dist[k][j]);
            __m128i dist_ij = _mm_loadu_si128((const __m128i*)&Dist[i][j]);
            __m128i sum = _mm_add_epi32(dist_ik, dist_kj);
            __m128i cmp = _mm_cmplt_epi32(sum, dist_ij);
            __m128i result = _mm_or_si128(
                _mm_and_si128(cmp, sum),
                _mm_andnot_si128(cmp, dist_ij)
            );
            _mm_storeu_si128((__m128i*)&Dist[i][j], result);
        }

        // Handle remaining elements for last row
        for (int j = (block_internal_end_y - ((block_internal_end_y - block_internal_start_y) % simd_width));
             j < block_internal_end_y; ++j) {
            int new_dist = Dist[i][k] + Dist[k][j];
            if (new_dist < Dist[i][j]) {
                Dist[i][j] = new_dist;
            }
        }
    }
}
