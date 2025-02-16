// full Allreduce + normal merge + ISend/Recv (Time: 177.03s)
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
#include <utility> // swap
#include <functional>
#include <limits> // float_max
#include <cmath> // ceil
#include <algorithm> // sort, copy, min
using namespace std;
typedef unsigned long long ULL;

bool merge_and_keep(vector<float>& arr1, vector<float>& arr2, bool keep_front)
{
    int N = arr1.size(), M = arr2.size();
    vector<float> buff(N + M);
    ULL idx1 = 0, idx2 = 0, buffidx = 0;
    bool swapped = false;
    while (idx1 < N && idx2 < M)
    {
        if (arr1[idx1] <= arr2[idx2]) buff[buffidx++] = arr1[idx1++];
        else 
        {
            buff[buffidx++] = arr2[idx2++];
            swapped = true;
        }
    }
    while (idx1 < N) buff[buffidx++] = arr1[idx1++];
    while (idx2 < M) buff[buffidx++] = arr2[idx2++];

    if (keep_front)
        copy(buff.begin(), buff.begin() + N, arr1.begin()); // those before get til N (store for arr1)
    else
        copy(buff.begin() + N, buff.end(), arr2.begin()); // those after get from N (store for arr2)

    return !swapped;
}

int main(int argc, char* argv[])
{
    /*------------------------------------------- Preparation -------------------------------------------*/
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ULL N = stoll(argv[1]);
    string input_filename = argv[2];
    string output_filename = argv[3];

    MPI_File input_file, output_file;

    /*------------------------------------------- Divide tasks -------------------------------------------*/
    int remainder = N % size;
    int self_count = N / size + (rank < remainder); // distribute remainder in one line
    int offset = N / size * rank + min(remainder, rank); // choose partial remainder or full remainder in one line
    int left_count = self_count + (rank == remainder); // only the border one's left side gonna increase one
    int right_count = self_count - (rank + 1 == remainder); // only the border one's right side gonna decrease one

    /*------------------------------------------- Read file -------------------------------------------*/
    vector<float> self_arr(self_count), left_arr(left_count), right_arr(right_count);

    MPI_File_open(MPI_COMM_WORLD, input_filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, offset * sizeof(float), self_arr.data(), self_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    /*------------------------------------------- local sort first -------------------------------------------*/
    sort(self_arr.begin(), self_arr.end());

    /*------------------------------------------- odd-even sort (exchange data) -------------------------------------------*/
    MPI_Request send_request;
    bool sorted = false;
    while (!sorted)
    {
        /*------------------------------------------- even sort -------------------------------------------*/
        if (rank % 2 == 0 && rank < size - 1) // first half
        {
            MPI_Isend(self_arr.data(), self_count, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_request);
            MPI_Recv(right_arr.data(), right_count, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE); // must check send before leaving even-sort (cuz send_request will be reused)
            merge_and_keep(self_arr, right_arr, true);
        }
        else if (rank % 2 == 1) // second half
        {
            MPI_Isend(self_arr.data(), self_count, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_request);
            MPI_Recv(left_arr.data(), left_count, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            merge_and_keep(left_arr, self_arr, false);
        }

        /*------------------------------------------- odd sort -------------------------------------------*/
        sorted = true;
        if (rank % 2 == 1 && rank < size - 1) // first half
        {
            MPI_Isend(self_arr.data(), self_count, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_request);
            MPI_Recv(right_arr.data(), right_count, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE); // must check send before leaving even-sort (cuz send_request will be reused)
            sorted = merge_and_keep(self_arr, right_arr, true); // only check sorted in odd phase will suffice
        }
        else if (rank % 2 == 0 && rank > 0) // second half
        {
            MPI_Isend(self_arr.data(), self_count, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_request);
            MPI_Recv(left_arr.data(), left_count, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            sorted = merge_and_keep(left_arr, self_arr, false); // only check sorted in odd phase will suffice
        }
        // collect the "sorted" flag
        MPI_Allreduce(MPI_IN_PLACE, &sorted, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }
    // if (rank == 0) cout << "finished" << endl;

    /*------------------------------------------- Write file -------------------------------------------*/
    MPI_File_open(MPI_COMM_WORLD, output_filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, offset * sizeof(float), self_arr.data(), self_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}