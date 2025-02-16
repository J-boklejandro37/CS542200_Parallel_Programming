#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <boost/sort/spreadsort/float_sort.hpp>

void merge(std::vector<float>& local_data, std::vector<float>& partner_data, std::vector<float>& merged, int rank, int partner, int local_n, int partner_n) {
    MPI_Request send_request, recv_request;

    if (rank < partner) {
        if (local_n > 0 && partner_n > 0) {
            float local_back=local_data[local_n-1], partner_front;

            MPI_Isend(&local_back, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &send_request);
            MPI_Irecv(&partner_front, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &recv_request);

            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);

            if (local_back <= partner_front) return;
        }
        
        MPI_Isend(local_data.data(), local_n, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &send_request);
        MPI_Irecv(partner_data.data(), partner_n, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &recv_request);
        
        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);

        int i = 0, j = 0, k = 0;
        while (k < local_n) {
            if (j == partner_n || (i < local_n && local_data[i] <= partner_data[j])) {
                merged[k++] = local_data[i++];
            } else {
                merged[k++] = partner_data[j++];
            }
        }
        local_data.swap(merged);
    } else {
        if (local_n > 0 && partner_n > 0) {
            float local_front=local_data[0], partner_back;

            MPI_Isend(&local_front, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &send_request);
            MPI_Irecv(&partner_back, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &recv_request);

            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            
            if (local_front >= partner_back) return;
        }

        MPI_Isend(local_data.data(), local_n, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &send_request);
        MPI_Irecv(partner_data.data(), partner_n, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &recv_request);
        
        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);

        int i = local_n - 1, j = partner_n - 1, k = local_n - 1;
        while (k >= 0) {
            if (j < 0 || (i >= 0 && local_data[i] >= partner_data[j])) {
                merged[k--] = local_data[i--];
            } else {
                merged[k--] = partner_data[j--];
            }
        }
        local_data.swap(merged);
    }
}

void odd_even_sort(std::vector<float>& local_data, int rank, int size, int local_n, int left_n, int right_n) {
    #pragma omp parallel
    {
        #pragma omp single
        boost::sort::spreadsort::float_sort(local_data.begin(), local_data.end());
    }

    int max_n = std::max(left_n, right_n);

    std::vector<float> partner_data(max_n);

    std::vector<float> merged(local_n);

    int phase = 0;
        
    while (phase < size+1) {
        int partner, partner_n;

        if (phase % 2 == 0) {
            if (rank % 2 == 0) {
                partner = rank + 1;
                partner_n = right_n;
            } else {
                partner = rank - 1;
                partner_n = left_n;
            }
        } else {
            if (rank % 2 == 1) {
                partner = rank + 1;
                partner_n = right_n;
            } else {
                partner = rank - 1;
                partner_n = left_n;
            }
        }
        
        if (partner >= 0 && partner < size) {
            merge(local_data, partner_data, merged, rank, partner, local_n, partner_n);
        }
        phase++;
    }
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    omp_set_num_threads(omp_get_max_threads());

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);

    char *input_filename = argv[2];
    char *output_filename = argv[3];
    
    int elements_per_process = n / size;
    int remainder = n % size;

    int local_n = (rank < remainder) ? elements_per_process + 1 : elements_per_process;
    int offset = rank * elements_per_process + (rank < remainder ? rank : remainder);

    int left_n = local_n + (rank == remainder);
    int right_n = local_n - (rank + 1 == remainder);

    std::vector<float> local_data(local_n);

    MPI_File input_file, output_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

    MPI_File_read_at(input_file, offset * sizeof(float), local_data.data(), local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    odd_even_sort(local_data, rank, size, local_n, left_n, right_n);
    
    MPI_File_write_at(output_file, offset * sizeof(float), local_data.data(), local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}