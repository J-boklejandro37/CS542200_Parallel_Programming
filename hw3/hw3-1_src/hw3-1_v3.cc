// [testcase c18.1]
// normal_FW -> 38.85 s (nothing)
//           ->  6.48 s (omp, parallel for schedule(static) collapse(2)) : the more contiguous the memory access, the faster
//           -> 40.06 s (omp, parallel for schedule(static,1) collapse(2))
//           ->     TLE (omp, parallel for schedule(dynamic,1) collapse(2))
//           ->  7.84 s (omp, parallel for schedule(guided,1) collapse(2))
// block_FW  -> 32.04 s (nothing)
//           -> 19.61 s (omp, block_FW: parallel created in each iteration, block_FW: sections+section)
//           -> 19.62 s (omp, block_FW: parallel created outside for loop, block_FW: sections+section)
//           -> 37.76 s (omp, block_FW: parallel created in each iteration, block_FW: taskgroup+task)
//           -> 37.56 s (omp, block_FW: parallel created outside for loop, block_FW: taskgroup+task)

//           ->  8.49 s (omp, cal: outer parallel for schedule(static) collapse(2)) : contiguous
//           ->  8.49 s (omp, cal: outer parallel for schedule(static,1) collapse(2)) : interleaving
//           ->  8.74 s (omp, cal: outer parallel for schedule(dynamic,1) collapse(2))
//           ->  8.64 s (omp, cal: outer parallel for schedule(guided,1) collapse(2))

//           ->  4.97 s (omp, cal: inner parallel for schedule(static) collapse(2)) ------------------------------------------------> best
//           -> 16.89 s (omp, cal: inner parallel for schedule(static,1) collapse(2))
//           ->     TLE (omp, cal: inner parallel for schedule(dynamic,1) collapse(2))
//           ->  8.73 s (omp, cal: inner parallel for schedule(guided,1) collapse(2))

//           -> 39.81 s (omp nested, cal: outer(static) + inner(static))
//           -> 37.05 s (omp nested, cal: outer(num_threads(2), static) + inner(num_threads(6), static))
//           -> 46.78 s (omp nested, block_FW: sections+section, cal: inner(static))

// [Full testcase]
// normal_FW -> 90.62 s (omp, parallel for schedule(guided,1) collapse(2))
//           -> 73.25 s (omp, parallel for schedule(static) collapse(2)) 
// block_FW  -> 54.93 s (omp, cal: inner(static))

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#define CHUNK_SIZE 10
#define min(a,b) (a < b ? a : b)
#define ceil(a,b) ((a + b - 1) / b)

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
void normal_FW();
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m, NUM_THREADS;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    input(argv[1]);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NUM_THREADS = CPU_COUNT(&cpu_set);
    // omp_set_nested(1);

    int B = 512;
    block_FW(B);
    // normal_FW();

    output(argv[2]);

    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("n: %d, m: %d\n", n, m);

    #pragma omp parallel for schedule(static) collapse(2)
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
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void normal_FW()
{
    for (int k = 0; k < n; k++)
    {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Dist[i][j] = min(Dist[i][j], Dist[i][k] + Dist[k][j]);
            }
        }
    }
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
                printf("%d %d\n", r, round);
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

    // #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            for (int k = k_start; k < k_end; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                #pragma omp parallel for schedule(static) collapse(2)
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        int temp = Dist[i][k] + Dist[k][j];
                        if (temp < Dist[i][j]) {
                            Dist[i][j] = temp;
                        }
                    }
                }
            }
        }
    }
}