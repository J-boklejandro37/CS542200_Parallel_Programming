#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <cmath>
using namespace std;
typedef unsigned long long ULL;

int main(int argc, char* argv[])
{
	ULL r = stoull(argv[1]);
	ULL k = stoull(argv[2]);
	ULL r_squared = r * r;	
	
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// // Calculate the number of iterations each process should handle
    // ULL iterations_per_process = ceil(static_cast<double>(r) / size);

    // // Calculate the start and end values for each process
    // ULL start = iterations_per_process * rank;
   	// ULL end = (rank == size - 1 ? r : start + iterations_per_process);

    // // Perform the work for the assigned range
	// ULL count = 0;
    // for (ULL x = start; x < end; x += 1)
	// {
    //     count += ceil(sqrtl(r_squared - x * x));
    // }
	// count %= k;
	
    // Perform the work with step = size
	ULL count = 0;
    for (ULL x = rank; x < r; x += size)
	{
        count += ceil(sqrtl(r_squared - x * x));
    }
	count %= k;

	// use MPI_Recude to sum all the counts, don't need consecutive MPI_Recv and MPI_Send
	ULL total_count;
    MPI_Reduce(&count, &total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Finalize();
	
	// root process, used for printing out answer
	if (rank == 0)
	{
		cout << (total_count * 4) % k << endl;
	}
	
    return 0;
}
