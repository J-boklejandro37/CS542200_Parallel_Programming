// float_sort+ iteration size + check before merge + half merge(less comparison) + Sendrecv
// (Time: 107.44s -> 106.95(omp float_sort) -> 102.43(mpi_open in a row))
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <string>
#include <vector>
#include <utility> // swap
#include <limits> // float_max
#include <cmath> // ceil
#include <algorithm> // sort, copy, min
#include <boost/sort/spreadsort/float_sort.hpp>

void front_merge(std::vector<float>& left, std::vector<float>& right, std::vector<float>& buffer)
{
    int i = 0, j = 0, k = 0, iend = left.size(), jend = right.size();
    while (k != iend)
    {   // cuz jend is smaller
        if (j == jend || left[i] <= right[j]) buffer[k++] = left[i++];
        else buffer[k++] = right[j++];
    }
    std::swap(left, buffer);
}

void rear_merge(std::vector<float>& left, std::vector<float>& right, std::vector<float>& buffer)
{
    int iend = left.size(), jend = right.size();
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
    omp_set_num_threads(omp_get_max_threads());
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = std::stoi(argv[1]);

    MPI_File input_file, output_file;

    /*------------------------------------------- Divide tasks -------------------------------------------*/
    int remainder = N % size;
    int self_count = N / size + (rank < remainder); // distribute remainder in one line
    int offset = N / size * rank + std::min(remainder, rank); // choose partial remainder or full remainder in one line
    int left_count = self_count + (rank == remainder); // only the border one's left side gonna increase one
    int right_count = self_count - (rank + 1 == remainder); // only the border one's right side gonna decrease one

    /*------------------------------------------- Read file -------------------------------------------*/
    std::vector<float> self_arr(self_count), left_arr(left_count), right_arr(right_count), buff_arr(self_count);

    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_read_at(input_file, offset * sizeof(float), self_arr.data(), self_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    /*------------------------------------------- local sort first -------------------------------------------*/
    #pragma omp parallel
    {
        #pragma omp single
        boost::sort::spreadsort::float_sort(self_arr.begin(), self_arr.end());
    }

    /*------------------------------------------- odd-even sort (exchange data) -------------------------------------------*/
    int iteration = size + 1;
    int is_left = rank & 1; // even sort first
    while (iteration--) 
    {   /*------------------------------------------- even-odd-even-odd sort -------------------------------------------*/
        // need to check (count != 0) since nprocs may be larger than N
        if (is_left && rank != size - 1 && self_count != 0 && right_count != 0) // left part
        {   // loads just one data for comparison
            MPI_Sendrecv(&self_arr[self_count-1], 1, MPI_FLOAT, rank + 1, 0, 
                         &right_arr[0], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (self_arr[self_count-1] > right_arr[0]) // meaning unsorted
            {   // loads the whole data
                MPI_Sendrecv(&self_arr[0], self_count - 1, MPI_FLOAT, rank + 1, 0, 
                            &right_arr[1], right_count - 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                front_merge(self_arr, right_arr, buff_arr);
            }
        }
        else if (!is_left && rank != 0 && self_count != 0 && left_count != 0) // right part
        {
            MPI_Sendrecv(&self_arr[0], 1, MPI_FLOAT, rank - 1, 0,
                         &left_arr[left_count-1], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (left_arr[left_count-1] > self_arr[0])
            {   // loads the whole data
                MPI_Sendrecv(&self_arr[1], self_count - 1, MPI_FLOAT, rank - 1, 0,
                            &left_arr[0], left_count - 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                rear_merge(left_arr, self_arr, buff_arr);
            }
        }

        is_left ^= 1; // switch odd-even
    }
    // if (rank == 0) cout << "finished" << endl;

    /*------------------------------------------- Write file -------------------------------------------*/
    MPI_File_write_at(output_file, offset * sizeof(float), self_arr.data(), self_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}