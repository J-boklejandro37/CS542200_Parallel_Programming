#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	/*------------------store argv------------------*/
	assert(argc == 3);
	unsigned long long radius = std::stoull(argv[1]);
	unsigned long long k = std::stoull(argv[2]);
	unsigned long long r_squared = radius * radius;

	/*------------------initialize mpi------------------*/
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/*------------------parallel calculation------------------*/
	unsigned long long local_total_count = 0;
	int num_threads;
	int step;

    #pragma omp parallel reduction(+:local_total_count) // by default it's default(shared), meaning num_threads will be shared among threads
    {
		#pragma omp single
		{
			num_threads = omp_get_num_threads(); // needs to be done inside parallel region, otherwise will always be 1 (the master thread)
			step = size * num_threads;
		}
	
        unsigned long long count = 0;
		// start with (rank * num_threads + thread_id)
		// offset by (size * num_threads), cuz there's multiple processes with multiple threads
        for (unsigned long long x = rank * num_threads + omp_get_thread_num(); x < radius; x += step)
        {
            count += std::ceil(sqrtl(r_squared - x * x));
        }
        local_total_count += count % k;
    }

	/*------------------finalize mpi------------------*/
	unsigned long long all_total_count;
	// MPI_Reduce(&sendbuff, &recvbuff, count, type, operation, destination, communicator);
    MPI_Reduce(&local_total_count, &all_total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Finalize();
	
	/*------------------rank0 print------------------*/
	if (rank == 0)
	{
		std::cout << (all_total_count * 4) % k << std::endl;
	}
}


