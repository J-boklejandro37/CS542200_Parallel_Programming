// no vector + float_sort(omp) + iteration size + check before merge + half merge(less comparison) + Sendrecv + partner method (Time: 97.21s)
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <string>
#include <vector>
#include <utility> // swap
#include <cmath> // ceil
#include <algorithm> // sort, copy, min
#include <boost/sort/spreadsort/float_sort.hpp>

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

void front_merge(float*& left, float* right, float*& buffer, int iend, int jend)
{
    int i = 0, j = 0, k = 0;
    while (k != iend)
    {   // cuz jend is smaller
        if (j == jend || left[i] <= right[j]) buffer[k++] = left[i++];
        else buffer[k++] = right[j++];
    }
    std::swap(left, buffer);
}

void rear_merge(float* left, float*& right, float*& buffer, int iend, int jend)
{
    int i = iend - 1, j = jend - 1, k = j;
    while (k != -1)
    {   // cuz jend is smaller
        if (right[j] >= left[i]) buffer[k--] = right[j--];
        else buffer[k--] = left[i--];
    }
    std::swap(right, buffer);
}

int main(int argc, char* argv[])
{
    /*------------------------------------------- Preparation -------------------------------------------*/
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = std::atoi(argv[1]);

    /*------------------------------------------- Divide tasks -------------------------------------------*/
    int remainder = N % size;
    int self_count = N / size + (rank < remainder); // distribute remainder in one line
    int offset = N / size * rank + min(rank, remainder); // choose partial remainder or full remainder in one line
    int left_count = self_count + (rank == remainder); // only the border one's left side gonna increase one
    int right_count = self_count - (rank + 1 == remainder); // only the border one's right side gonna decrease one

    /*------------------------------------------- Read file -------------------------------------------*/
    float* self_arr = new float[self_count];
    float* partner_arr = new float[max(left_count, right_count)];
    float* buff_arr = new float[self_count];

    MPI_File input_file, output_file;
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_read_at(input_file, offset * sizeof(float), self_arr, self_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    /*------------------------------------------- local sort first -------------------------------------------*/
    #pragma omp parallel
    {
        #pragma omp single
        boost::sort::spreadsort::float_sort(self_arr, self_arr + self_count);
    }

    /*------------------------------------------- odd-even sort (exchange data) -------------------------------------------*/
    MPI_Request s_req, r_req;
    int partner, partner_count;
    for (int iteration = 0; iteration <= size; iteration += 1)
    {   /*------------------------------------------- even-odd-even-odd sort -------------------------------------------*/
        if (iteration & 1)
        {   // odd sort
            if (rank & 1) { partner = rank + 1; partner_count = right_count; } // first half
            else { partner = rank - 1; partner_count = left_count; } // second half
        }
        else
        {   // even sort
            if (rank & 1) { partner = rank - 1; partner_count = left_count; } // second half
            else { partner = rank + 1; partner_count = right_count; } // first half
        }
        
        // make sure partner in range (cuz self is always in range), and is not empty
        if (partner != -1 && partner != size && self_count != 0 && partner_count != 0)
        {
            if (rank < partner) // left part
            {   // loads just one data for comparison
                MPI_Sendrecv(self_arr + self_count - 1, 1, MPI_FLOAT, partner, 0, 
                         partner_arr, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (self_arr[self_count-1] > partner_arr[0]) // meaning unsorted
                {   // loads the whole data
                    MPI_Sendrecv(self_arr, self_count - 1, MPI_FLOAT, partner, 0, 
                            partner_arr + 1, partner_count - 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    front_merge(self_arr, partner_arr, buff_arr, self_count, partner_count);
                }
            }
            else // right part
            {
                MPI_Sendrecv(self_arr, 1, MPI_FLOAT, partner, 0,
                         partner_arr + partner_count - 1, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (partner_arr[partner_count-1] > self_arr[0])
                {   // loads the whole data
                    MPI_Sendrecv(self_arr + 1, self_count - 1, MPI_FLOAT, partner, 0,
                            partner_arr, partner_count - 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    rear_merge(partner_arr, self_arr, buff_arr, partner_count, self_count);
                }
            }
        }
    }

    /*------------------------------------------- Write file -------------------------------------------*/
    MPI_File_write_at(output_file, offset * sizeof(float), self_arr, self_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}