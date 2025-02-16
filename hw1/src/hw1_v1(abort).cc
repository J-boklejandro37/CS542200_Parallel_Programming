#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
#include <utility> // swap
#include <functional>
using namespace std;
typedef unsigned long long ULL;

int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ULL N = stoll(argv[1]);
    string input_filename = argv[2];
    string output_filename = argv[3];

    MPI_File input_file, output_file;
    vector<float> arr(N);
    // first task: the one swap the arr[0] element, must be rank0
    // last task: the one that swap the arr[N-1] element
    int last_task = ((N / 2) - 1 + size) % size;

    /*------------------------------------------- Read file -------------------------------------------*/
    MPI_File_open(MPI_COMM_WORLD, input_filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    // only first task reads arr[0]
    if (rank == 0) MPI_File_read_at(input_file, 0, &arr[0], 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    // only last task reads arr[N-1]
    if (rank == last_task) MPI_File_read_at(input_file, sizeof(float) * (N-1), &arr[N-1], 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    // everyone reads odd sort items
    for (int idx = rank * 2 + 1; rank < N - 1; rank += size * 2)
    {
        MPI_File_read_at(input_file, sizeof(float) * idx, &arr[idx], 2, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&input_file);

    cout << "here" << endl;

    /*------------------------------------------- Sorting -------------------------------------------*/
    // function<bool(int)> sort = [&](int is_odd)
    // {
    //     bool sorted = true;
    //     // sorting
    //     for (int idx = rank * 2 + is_odd; rank < N - 1; rank += size * 2)
    //     {
    //         if (arr[idx] > arr[idx+1]) { swap(arr[idx], arr[idx+1]); sorted = false; }
    //     }
    //     // pass to the right and receive from the left
    //     vector<MPI_Request> requests(N/size*2, MPI_REQUEST_NULL);
    //     int num_request = 0;
    //     for (int idx = rank * 2 + is_odd; rank < N; rank += size * 2)
    //     {
    //         int next = (rank + 1) % size;
    //         int prev = (rank - 1 + size) % size;
    //         if (idx < N - 2) // meaning (idx + 1) < N - 1 => no need to send the last one
    //             // MPI_Isend(&buffer, count, type, dest, tag, communicator, &request);
    //             MPI_Isend(&arr[idx+1], 1, MPI_FLOAT, next, idx + 1, MPI_COMM_WORLD, &requests[num_request++]); // use current request for sending
    //         if (idx > 1) // meaning (idx - 1) > 0 => no need to receive arr[0]
    //             // MPI_Irecv(&buffer, count, type, source, tag, communicator, &request);
    //             MPI_Irecv(&arr[idx-1], 1, MPI_FLOAT, prev, idx - 1, MPI_COMM_WORLD, &requests[num_request++]); // use previous request for receiving
    //     }
    //     // barrier for requests
    //     // MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    //     // Wait for all requests to complete
    //     int err = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    //     if (err != MPI_SUCCESS) {
    //         char error_string[MPI_MAX_ERROR_STRING];
    //         int length_of_error_string;
    //         MPI_Error_string(err, error_string, &length_of_error_string);
    //         cerr << "MPI error in rank " << rank << ": " << error_string << endl;
    //         MPI_Abort(MPI_COMM_WORLD, err);
    //     }
    //     return sorted;
    // };

    // bool sorted = false;
    // while (!sorted)
    // {
    //     // odd sort, don't need to check for sorted, checking after even sort will suffice
    //     sort(1);
    //     // even sort
    //     sorted = sort(0);
    //     // collect the "sorted" flag
    //     MPI_Allreduce(MPI_IN_PLACE, &sorted, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    // }

    /*------------------------------------------- Write file -------------------------------------------*/
    MPI_File_open(MPI_COMM_WORLD, output_filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    // only first task writes arr[0]
    if (rank == 0) MPI_File_write_at(output_file, 0, &arr[0], 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    // only last task writes arr[N-1]
    if (rank == last_task) MPI_File_write_at(output_file, sizeof(float) * (N-1), &arr[N-1], 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    // everyone writes odd sort items (won't matter writing which one)
    for (int idx = rank * 2 + 1; rank < N - 1; rank += size * 2)
    {
        MPI_File_write_at(output_file, sizeof(float) * idx, &arr[idx], 2, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}