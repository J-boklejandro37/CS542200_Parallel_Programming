// no vector + float_sort(omp) + partial Allreduce + check before merge + half merge(pointer arithmetic) + Sendrecv
// Time: 96.71s(intel mpi) / 90.23(openmpi)
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
#include <cassert>
#include <boost/sort/spreadsort/float_sort.hpp>

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

int front_merge(float*& left, float* right, float*& buffer, int left_count, int right_count)
{
    int swapped = 0;
    float *i = left, *j = right, *k = buffer;
    float *const iend = i + left_count, *const jend = j + right_count, *const kend = k + left_count;
    while (k != kend)
    {   // cuz jend is smaller
        if (j == jend || *i <= *j) *k++ = *i++;
        else 
        {
            *k++ = *j++;
            swapped = 1;
        }
    }
    std::swap(left, buffer);
    return swapped;
}

int rear_merge(float* left, float*& right, float*& buffer, int left_count, int right_count)
{
    int swapped = 0;
    float *i = left + left_count - 1, *j = right + right_count - 1, *k = buffer + right_count - 1;
    float *const iend = left - 1, *const jend = right - 1, *const kend = buffer - 1;
    while (k != kend)
    {   // cuz jend is smaller
        if (*j >= *i) *k-- = *j--;
        else 
        {
            *k-- = *i--;
            swapped = 1;
        }
    }
    std::swap(right, buffer);
    return swapped;
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
    int rank_endpoint = min(size, N); // cuz nprocs could be larger than size
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
    int global_swapped = 1, local_swapped = 0, iteration = 1;
    while (global_swapped)
    {   /*------------------------------------------- even sort -------------------------------------------*/
        if (!(rank & 1) && rank < rank_endpoint - 1) // left part
        {   // loads just one data for comparison
            MPI_Sendrecv(self_arr + self_count - 1, 1, MPI_FLOAT, rank + 1, 0, 
                    partner_arr, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (self_arr[self_count-1] > partner_arr[0]) // meaning unsorted
            {   // loads the whole data
                MPI_Sendrecv(self_arr, self_count - 1, MPI_FLOAT, rank + 1, 0, 
                        partner_arr + 1, right_count - 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                front_merge(self_arr, partner_arr, buff_arr, self_count, right_count);
            }
        }
        else if (rank & 1 && rank < rank_endpoint) // right part
        {
            MPI_Sendrecv(self_arr, 1, MPI_FLOAT, rank - 1, 0,
                    partner_arr + left_count - 1, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (partner_arr[left_count-1] > self_arr[0])
            {   // loads the whole data
                MPI_Sendrecv(self_arr + 1, self_count - 1, MPI_FLOAT, rank - 1, 0,
                        partner_arr, left_count - 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                rear_merge(partner_arr, self_arr, buff_arr, left_count, self_count);
            }
        }
        
        local_swapped = 0;
        /*------------------------------------------- odd sort -------------------------------------------*/
        if ((rank & 1) && rank < rank_endpoint - 1) // left part
        {   // loads just one data for comparison
            MPI_Sendrecv(self_arr + self_count - 1, 1, MPI_FLOAT, rank + 1, 0, 
                    partner_arr, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (self_arr[self_count-1] > partner_arr[0]) // meaning unsorted
            {   // loads the whole data
                MPI_Sendrecv(self_arr, self_count - 1, MPI_FLOAT, rank + 1, 0, 
                        partner_arr + 1, right_count - 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                local_swapped = front_merge(self_arr, partner_arr, buff_arr, self_count, right_count);
            }
        }
        else if (!(rank & 1) && rank != 0 && rank < rank_endpoint) // right part
        {
            MPI_Sendrecv(self_arr, 1, MPI_FLOAT, rank - 1, 0,
                    partner_arr + left_count - 1, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (partner_arr[left_count-1] > self_arr[0])
            {   // loads the whole data
                MPI_Sendrecv(self_arr + 1, self_count - 1, MPI_FLOAT, rank - 1, 0,
                        partner_arr, left_count - 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                local_swapped = rear_merge(partner_arr, self_arr, buff_arr, left_count, self_count);
            }
        }
        // std::cout << "rank " << rank << " here" << std::endl;
        // collect the "swapped" flag
        if (!(iteration & 3)) MPI_Allreduce(&local_swapped, &global_swapped, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        else global_swapped = 1;
        iteration += 1;
    }

    /*------------------------------------------- Write file -------------------------------------------*/
    MPI_File_write_at(output_file, offset * sizeof(float), self_arr, self_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();

    delete[] self_arr;
    delete[] partner_arr;
    delete[] buff_arr;

    return 0;
}